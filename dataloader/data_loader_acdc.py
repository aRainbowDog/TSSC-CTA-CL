# 相邻3帧的灰度图 - 四阶段课程学习（阶段0用伪GT）
import logging
import os
import torch
import decord
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import re
import random
from PIL import Image
from einops import rearrange
from typing import Dict, List, Tuple
from .video_transforms import *

try:
    import av
except ImportError:
    raise ImportError("请安装 PyAV 库：conda install -c conda-forge av 或 pip install av")

class_labels_map = None
cls_sample_cnt = None
logger = logging.getLogger(__name__)


def is_main_process():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0

def get_filelist(file_path):
    """递归获取文件列表，过滤非视频文件"""
    Filelist = []
    video_ext = ('.mp4', '.avi', '.mov', '.mkv')
    for home, dirs, files in os.walk(file_path):
        dirs.sort()
        files.sort()
        for filename in files:
            if filename.lower().endswith(video_ext):
                Filelist.append(os.path.join(home, filename))
    return sorted(Filelist)


def _group_videos_by_parent(file_path, video_paths):
    grouped_videos = {}
    for video_path in video_paths:
        group_key = os.path.relpath(os.path.dirname(video_path), file_path)
        grouped_videos.setdefault(group_key, []).append(video_path)
    return grouped_videos


def split_train_val_videos(file_path, seed, train_ratio=0.9):
    """Deterministically split videos by parent folder first, falling back to per-file shuffle."""
    all_videos = get_filelist(file_path)
    if not all_videos:
        return [], []

    grouped_videos = _group_videos_by_parent(file_path, all_videos)
    rng = random.Random(int(seed))
    group_keys = sorted(grouped_videos)
    if len(group_keys) > 1:
        rng.shuffle(group_keys)
        num_train_groups = int(len(group_keys) * train_ratio)
        num_train_groups = min(max(1, num_train_groups), len(group_keys) - 1)
        train_group_keys = set(group_keys[:num_train_groups])
        train_videos = []
        val_videos = []
        for group_key in group_keys:
            target_list = train_videos if group_key in train_group_keys else val_videos
            target_list.extend(sorted(grouped_videos[group_key]))
        return train_videos, val_videos

    shuffled_videos = list(all_videos)
    rng.shuffle(shuffled_videos)
    len_train = int(train_ratio * len(shuffled_videos))
    if len(shuffled_videos) > 1:
        len_train = min(max(1, len_train), len(shuffled_videos) - 1)
    return shuffled_videos[:len_train], shuffled_videos[len_train:]


def build_train_val_split_manifest(file_path, seed, train_ratio=0.9):
    """Return a JSON-serializable manifest for the deterministic train/val split."""
    train_videos, val_videos = split_train_val_videos(file_path, seed=seed, train_ratio=train_ratio)
    train_groups = _group_videos_by_parent(file_path, train_videos)
    val_groups = _group_videos_by_parent(file_path, val_videos)

    def _serialize_split(split_name, grouped_videos):
        groups = []
        for group_key in sorted(grouped_videos):
            rel_videos = [os.path.relpath(video_path, file_path) for video_path in sorted(grouped_videos[group_key])]
            groups.append({
                "group": group_key,
                "video_count": len(rel_videos),
                "videos": rel_videos,
            })
        return {
            "name": split_name,
            "group_count": len(groups),
            "video_count": sum(group["video_count"] for group in groups),
            "groups": groups,
        }

    all_videos = get_filelist(file_path)
    grouped_all = _group_videos_by_parent(file_path, all_videos)
    split_strategy = "group_by_parent_dir_deterministic_shuffle" if len(grouped_all) > 1 else "per_file_deterministic_shuffle"
    return {
        "data_path": os.path.abspath(file_path),
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "split_strategy": split_strategy,
        "total_group_count": len(grouped_all),
        "total_video_count": len(all_videos),
        "train": _serialize_split("train", train_groups),
        "val": _serialize_split("val", val_groups),
    }

class DecordInit(object):
    """Using Decord to initialize the video_reader"""
    def __init__(self, num_threads=1, **kwargs):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)
        self.kwargs = kwargs

    def __call__(self, filename):
        reader = decord.VideoReader(filename, ctx=self.ctx, num_threads=self.num_threads)
        return reader

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'num_threads={self.num_threads})')
        return repr_str

class data_loader(torch.utils.data.Dataset):
    """加载心脏视频数据（ACDC），支持四阶段课程学习（阶段0用伪GT）"""
    def __init__(self, configs, stage):
        self.stage = stage
        self.configs = configs
        self.current_step = 0
        
        # ========== 读取课程学习配置 ==========
        if hasattr(configs, 'curriculum_learning') and configs.curriculum_learning is not None:
            cl_config = configs.curriculum_learning
            
            # 阶段0（新增，伪GT）
            self.stage0_steps = cl_config.get('stage0_steps', 500)
            self.stage0_gap = cl_config.get('stage0_frame_gap', 1)
            self.stage0_frames = cl_config.get('stage0_dense_frames', 2)
            self.stage0_use_pseudo_gt = cl_config.get('stage0_use_pseudo_gt', True)
            self.stage0_alpha_range = cl_config.get('stage0_alpha_range', [0.3, 0.7])
            
            # 阶段1（修正gap=2）
            self.stage1_steps = cl_config.get('stage1_steps', 1500)
            self.stage1_gap = cl_config.get('stage1_frame_gap', 2)
            self.stage1_frames = cl_config.get('stage1_dense_frames', 3)
            
            # 阶段2（修正gap=4）
            self.stage2_steps = cl_config.get('stage2_steps', 2500)
            self.stage2_gap = cl_config.get('stage2_frame_gap', 4)
            self.stage2_frames = cl_config.get('stage2_dense_frames', 5)
            
            # 阶段3（修正gap=8）
            self.stage3_gap = cl_config.get('stage3_frame_gap', 8)
            self.stage3_frames = cl_config.get('stage3_dense_frames', 9)
        else:
            # 默认值
            self.stage0_steps = 500
            self.stage0_gap, self.stage0_frames = 1, 2
            self.stage0_use_pseudo_gt = True
            self.stage0_alpha_range = [0.3, 0.7]
            
            self.stage1_steps = 1500
            self.stage1_gap, self.stage1_frames = 2, 3
            
            self.stage2_steps = 2500
            self.stage2_gap, self.stage2_frames = 4, 5
            
            self.stage3_gap, self.stage3_frames = 8, 9

        if is_main_process():
            logger.info(f"[{self.stage}] 四阶段模式 | Gap: {self.stage0_gap}/{self.stage1_gap}/{self.stage2_gap}/{self.stage3_gap}")
        
        self.v_decoder = DecordInit()
        
        # 数据变换
        transform_list = [
            ToTensorVideo(),
            CenterCropResizeVideo(configs.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ]
        if self.stage == 'train':
            transform_list.insert(2, RandomHorizontalFlipVideo())
        self.transform = transforms.Compose(transform_list)

        # 划分数据集
        if self.stage in ['train', 'val']:
            self.data_path = configs.data_path_train
            train_videos, val_videos = split_train_val_videos(
                self.data_path,
                seed=getattr(configs, "global_seed", 3407),
            )
            if self.stage == 'train':
                self.video_lists = train_videos
            else:
                self.video_lists = val_videos
        elif self.stage == 'test':
            self.data_path = configs.data_path_test
            self.video_lists = get_filelist(self.data_path)
        else:
            raise ValueError(f"stage 必须是 train/val/test，当前为 {self.stage}")
        
        if is_main_process():
            logger.info(f"[{self.stage}] 加载视频数量: {len(self.video_lists)}")

    def set_training_step(self, step):
        """从训练循环中设置当前步数"""
        self.current_step = step

    def _read_video_safe(self, video_path):
        """安全读取视频，返回 TCHW 格式"""
        try:
            vframes, aframes, info = torchvision.io.read_video(
                filename=video_path, 
                pts_unit='sec', 
                output_format='TCHW'
            )
            return vframes
        except Exception as e:
            reader = self.v_decoder(video_path)
            frames = reader.get_batch(list(range(len(reader)))).asnumpy()
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
            return frames

    def _get_stage_config(self, step):
        """根据步数返回(stage, gap, dense_frames)"""
        if step < self.stage0_steps:
            return 0, self.stage0_gap, self.stage0_frames
        elif step < self.stage1_steps:
            return 1, self.stage1_gap, self.stage1_frames
        elif step < self.stage2_steps:
            return 2, self.stage2_gap, self.stage2_frames
        else:
            return 3, self.stage3_gap, self.stage3_frames
    
    def __getitem__(self, index):
        # 1. 读取视频
        path = self.video_lists[index]
        file_name = os.path.basename(path)
        patient_dir = os.path.dirname(path)
        patient_name = os.path.basename(patient_dir)

        slice_match = re.search(r'slice_(\d+)', file_name)
        if not slice_match:
            raise ValueError(f"文件名 {file_name} 未匹配到 slice_XX 格式")
        slice_num = int(slice_match.group(1))

        vframes = self._read_video_safe(path)
        vframes = vframes.to(torch.uint8)

        # 2. 根据阶段采样密集帧
        total_frames = len(vframes)
        stage, frame_gap, num_dense_frames = self._get_stage_config(self.current_step)

        # ========== 阶段0的特殊处理：生成伪GT ==========
        if stage == 0 and self.stage == 'train' and self.stage0_use_pseudo_gt:
            # 只需要2帧（首尾）
            if total_frames >= 2:
                if total_frames >= 3:
                    # 随机选择2帧（间隔至少1帧）
                    max_start = total_frames - 2
                    start_idx = random.randint(0, max_start)
                    frame_indices = [start_idx, start_idx + 1]
                else:
                    frame_indices = [0, 1] if total_frames == 2 else [0, 0]
            else:
                frame_indices = [0, 0]
            
            video_pair = vframes[frame_indices]
            video_pair = self.transform(video_pair)  # [2, C, H, W]
            
            # 随机生成α（用于插值）
            alpha = random.uniform(self.stage0_alpha_range[0], self.stage0_alpha_range[1])
            
            # 线性插值生成中间帧
            frame_alpha = (1 - alpha) * video_pair[0] + alpha * video_pair[1]
            
            # 构建三元组 [首帧, 插值帧, 尾帧]
            video = torch.stack([video_pair[0], frame_alpha, video_pair[1]], dim=0)
        
        # ========== 其他阶段：正常采样 ==========
        else:
            if self.stage == 'train':
                # 训练：随机选择局部密集帧
                if total_frames >= num_dense_frames:
                    max_start = total_frames - num_dense_frames
                    start_idx = random.randint(0, max_start)
                    frame_indices = list(range(start_idx, start_idx + num_dense_frames))
                else:
                    frame_indices = list(range(total_frames))
                    frame_indices += [total_frames - 1] * (num_dense_frames - total_frames)
            else:
                # 验证/测试：从头采样
                if total_frames >= num_dense_frames:
                    frame_indices = list(range(num_dense_frames))
                else:
                    frame_indices = list(range(total_frames))
                    frame_indices += [total_frames - 1] * (num_dense_frames - total_frames)

            video = vframes[frame_indices]
            video = self.transform(video)

        return {
            'video': video,  # [2/3/5/9, C, H, W]（阶段0是3帧）
            'video_name': f"{patient_name}_slice_{slice_num:02d}",
            'video_gt': video.clone(),
            'video_path': self.video_lists[index],
            'stage': stage,
            'frame_gap': frame_gap
        }

    def __len__(self):
        return len(self.video_lists)


class full_sequence_data_loader(torch.utils.data.Dataset):
    """加载完整CTA序列，用于滑窗三连帧评测。"""

    def __init__(self, configs, stage):
        if stage not in ['val', 'test']:
            raise ValueError(f"full_sequence_data_loader 只支持 val/test，当前为 {stage}")

        self.stage = stage
        self.configs = configs
        self.v_decoder = DecordInit()

        self.transform = transforms.Compose([
            ToTensorVideo(),
            CenterCropResizeVideo(configs.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

        if self.stage == 'val':
            self.data_path = configs.data_path_train
            _, self.video_lists = split_train_val_videos(
                self.data_path,
                seed=getattr(configs, "global_seed", 3407),
            )
        else:
            self.data_path = configs.data_path_test
            self.video_lists = get_filelist(self.data_path)

        if is_main_process():
            logger.info(f"[{self.stage}] 滑窗评测视频数量: {len(self.video_lists)}")

    def _read_video_safe(self, video_path):
        try:
            vframes, _, _ = torchvision.io.read_video(
                filename=video_path,
                pts_unit='sec',
                output_format='TCHW'
            )
            return vframes
        except Exception:
            reader = self.v_decoder(video_path)
            frames = reader.get_batch(list(range(len(reader)))).asnumpy()
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
            return frames

    def __getitem__(self, index):
        path = self.video_lists[index]
        file_name = os.path.basename(path)
        patient_dir = os.path.dirname(path)
        patient_name = os.path.basename(patient_dir)

        slice_match = re.search(r'slice_(\d+)', file_name)
        if slice_match:
            video_name = f"{patient_name}_slice_{int(slice_match.group(1)):02d}"
        else:
            video_name = os.path.splitext(file_name)[0]

        vframes = self._read_video_safe(path).to(torch.uint8)
        video = self.transform(vframes)

        return {
            'video': video,  # [T, C, H, W]
            'video_name': video_name,
            'video_path': path,
            'num_frames': video.shape[0],
        }

    def __len__(self):
        return len(self.video_lists)


def collate_full_sequence_batch(batch):
    """保留可变长度视频列表，避免默认collate强行stack失败。"""
    return {
        'videos': [item['video'] for item in batch],
        'video_name': [item['video_name'] for item in batch],
        'video_path': [item['video_path'] for item in batch],
        'num_frames': [item['num_frames'] for item in batch],
    }
