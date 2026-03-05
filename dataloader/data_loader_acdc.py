# 相邻3帧的灰度图 - 修改为密集帧返回以支持课程学习
import os
import torch
import decord
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import re
import random
from PIL import Image
from einops import rearrange
from typing import Dict, List, Tuple
from .video_transforms import *

# 解决多进程 DataLoader 中 PyAV 加载问题（提前导入并初始化）
try:
    import av
except ImportError:
    raise ImportError("请安装 PyAV 库：conda install -c conda-forge av 或 pip install av")

class_labels_map = None
cls_sample_cnt = None

def get_filelist(file_path):
    """递归获取文件列表，过滤非视频文件"""
    Filelist = []
    video_ext = ('.mp4', '.avi', '.mov', '.mkv')  # 仅保留视频文件
    for home, dirs, files in os.walk(file_path):
        for filename in files:
            if filename.lower().endswith(video_ext):
                Filelist.append(os.path.join(home, filename))
    return Filelist

class DecordInit(object):
    """Using Decord to initialize the video_reader (备用视频读取方案)"""
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
    """加载心脏视频数据（ACDC），支持课程学习的密集帧返回"""
    def __init__(self, configs, stage):
        self.stage = stage
        self.configs = configs
        self.current_step = 0  # 新增：当前训练步数
        
        # ========== 新增：读取课程学习配置 ==========
        if hasattr(configs, 'curriculum_learning') and configs.curriculum_learning is not None:
            cl_config = configs.curriculum_learning
            self.stage1_steps = cl_config.get('stage1_steps', 1000)
            self.stage2_steps = cl_config.get('stage2_steps', 2000)
            
            self.stage1_gap = cl_config.get('stage1_frame_gap', 1)
            self.stage1_frames = cl_config.get('stage1_dense_frames', 3)
            
            self.stage2_gap = cl_config.get('stage2_frame_gap', 2)
            self.stage2_frames = cl_config.get('stage2_dense_frames', 5)
            
            self.stage3_gap = cl_config.get('stage3_frame_gap', 4)
            self.stage3_frames = cl_config.get('stage3_dense_frames', 9)
        else:
            # 默认值
            self.stage1_steps = 1000
            self.stage2_steps = 2000
            self.stage1_gap, self.stage1_frames = 1, 3
            self.stage2_gap, self.stage2_frames = 2, 5
            self.stage3_gap, self.stage3_frames = 4, 9

        print(f"[{self.stage}] 三元组+密集帧模式 | Gap: {self.stage1_gap}/{self.stage2_gap}/{self.stage3_gap}")
        
        self.v_decoder = DecordInit()
        
        # 数据变换：区分训练/测试（训练可加随机翻转）
        transform_list = [
            ToTensorVideo(),
            CenterCropResizeVideo(configs.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ]
        if self.stage == 'train':
            transform_list.insert(2, RandomHorizontalFlipVideo())  # 训练加水平翻转
        self.transform = transforms.Compose(transform_list)

        # 划分训练/验证/测试集
        if self.stage in ['train', 'val']:
            self.data_path = configs.data_path_train
            all_videos = get_filelist(self.data_path)
            len_train = int(0.9 * len(all_videos))
            if self.stage == 'train':
                self.video_lists = all_videos[:len_train]
            else:
                self.video_lists = all_videos[len_train:]
        elif self.stage == 'test':
            self.data_path = configs.data_path_test
            self.video_lists = get_filelist(self.data_path)
        else:
            raise ValueError(f"stage 必须是 train/val/test，当前为 {self.stage}")
        
        print(f"[{self.stage}] 加载视频数量: {len(self.video_lists)} | 三元组+密集帧模式 | 阶段阈值: {self.stage1_steps}/{self.stage2_steps}")

    def set_training_step(self, step):
        """从训练循环中设置当前步数（用于确定采样间隔）"""
        self.current_step = step

    def _get_interval_for_step(self, step):
        """根据训练步数返回目标采样间隔"""
        if step < self.stage1_steps:  # ✅ 使用实例属性（从配置读取）
            return 1  # 阶段1: 间隔1
        elif step < self.stage2_steps:  # ✅ 使用实例属性（从配置读取）
            return 2  # 阶段2: 间隔2
        else:
            return 4  # 阶段3: 间隔4

    def _read_video_safe(self, video_path):
        """安全读取视频，兼容不同格式，返回 TCHW 格式张量"""
        try:
            # 使用 torchvision 读取（依赖 PyAV）
            vframes, aframes, info = torchvision.io.read_video(
                filename=video_path, 
                pts_unit='sec', 
                output_format='TCHW',
                pts_per_second=15  # 固定帧率，避免不同视频帧率不一致
            )
            return vframes
        except Exception as e:
            # 备用方案：使用 Decord 读取
            reader = self.v_decoder(video_path)
            frames = reader.get_batch(list(range(len(reader)))).asnumpy()  # THWC
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # 转为 TCHW
            return frames

    def _get_stage_config(self, step):
        """根据步数返回(stage, gap, dense_frames)"""
        if step < self.stage1_steps:
            return 1, self.stage1_gap, self.stage1_frames
        elif step < self.stage2_steps:
            return 2, self.stage2_gap, self.stage2_frames
        else:
            return 3, self.stage3_gap, self.stage3_frames
    
    def __getitem__(self, index):
        # 1. 读取当前切片视频
        path = self.video_lists[index]
        file_name = os.path.basename(path)
        patient_dir = os.path.dirname(path)
        patient_name = os.path.basename(patient_dir)

        # 解析切片号（增强正则鲁棒性）
        slice_match = re.search(r'slice_(\d+)', file_name)
        if not slice_match:
            raise ValueError(f"文件名 {file_name} 未匹配到 slice_XX 格式")
        slice_num = int(slice_match.group(1))
        ext = os.path.splitext(file_name)[1]

        # 2. 读取视频（使用安全读取函数）
        vframes = self._read_video_safe(path)
        vframes = vframes.to(torch.uint8)  # 转为uint8，避免归一化异常


        # ========== 根据阶段采样密集帧 ==========
        total_frames = len(vframes)
        stage, frame_gap, num_dense_frames = self._get_stage_config(self.current_step)

        if self.stage == 'train':
            # 训练：随机选择局部密集帧
            if total_frames >= num_dense_frames:
                max_start = total_frames - num_dense_frames
                start_idx = random.randint(0, max_start)
                frame_indices = list(range(start_idx, start_idx + num_dense_frames))
            else:
                # 帧数不足，边界填充
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
            'video': video,  # [3/5/9, C, H, W]
            'video_name': f"{patient_name}_slice_{slice_num:02d}",
            'video_gt': video.clone(),
            'video_path': self.video_lists[index],
            'stage': stage,       # 新增
            'frame_gap': frame_gap  # 新增
        }

    def __len__(self):
        return len(self.video_lists)

if __name__ == '__main__':
    from tqdm import tqdm
    import imageio
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    # 加载配置
    config_path = r'../configs/config_acdc.yaml'
    config_dict = OmegaConf.load(config_path)
    dataset = data_loader(config_dict, 'val')

    # 测试不同阶段的采样
    print("\n测试阶段切换:")
    for step in [0, 20000, 40000]:
        dataset.set_training_step(step)
        sample = dataset[0]
        print(f"Step {step} | 目标间隔: {sample['target_interval']} | 视频形状: {sample['video'].shape}")

    # 禁用多进程（调试阶段），避免 PyAV 多进程问题
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # 调试时设为0，正常训练可设为4
        pin_memory=True,
        drop_last=True
    )

    pbar = tqdm(loader, total=min(len(loader), 3), desc="Processing")
    for i, batch in enumerate(pbar):
        video = batch['video']
        print(f"批次{i} - 视频形状: {video.shape}, 数值范围: [{video.min():.4f}, {video.max():.4f}]")
        
        if i >= 2:  # 仅测试前3个批次
            break
    
    print("\n✅ Dataloader测试完成！")