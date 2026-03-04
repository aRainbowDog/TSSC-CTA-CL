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
        
        # ========== 核心修改：改为密集帧返回 ==========
        self.max_frames = 9  # 密集帧数（支持间隔4采样：0,4,8）
        self.target_video_len = self.max_frames  # 始终返回9帧
        self.current_step = 0  # 当前训练步数（用于确定采样间隔）
        
        # ========== 【修复】读取课程学习阶段阈值 ==========
        if hasattr(configs, 'curriculum_learning') and configs.curriculum_learning is not None:
            cl_config = configs.curriculum_learning
            self.stage1_steps = cl_config.get('stage1_steps', 20000)
            self.stage2_steps = cl_config.get('stage2_steps', 40000)
            max_dense_frames = cl_config.get('max_dense_frames', 9)
            # 验证max_frames与配置一致
            if max_dense_frames != self.max_frames:
                print(f"⚠️  配置中的max_dense_frames ({max_dense_frames}) != 硬编码的max_frames ({self.max_frames}), 使用配置值")
                self.max_frames = max_dense_frames
                self.target_video_len = max_dense_frames
        else:
            # 默认值（向后兼容旧配置文件）
            self.stage1_steps = 20000
            self.stage2_steps = 40000
        
        self.v_decoder = DecordInit()
        self.temporal_sample = TemporalRandomCrop(self.max_frames)  # 修复：使用max_frames而非configs.tar_num_frames
        
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
        
        print(f"[{self.stage}] 加载视频数量: {len(self.video_lists)} | 返回密集帧数: {self.max_frames} | 阶段阈值: {self.stage1_steps}/{self.stage2_steps}")

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

        # ========== 核心修改：始终采样密集的连续帧 ==========
        total_frames = len(vframes)
        target_interval = self._get_interval_for_step(self.current_step)
        
        if self.stage == 'train':
            # 训练：随机选择局部连续的max_frames帧
            required_length = self.max_frames
            if total_frames >= required_length:
                max_start = total_frames - required_length
                start_idx = random.randint(0, max_start)
                frame_indice = np.arange(start_idx, start_idx + required_length)
            else:
                # 帧数不足，用边界填充
                frame_indice = np.arange(0, total_frames)
                frame_indice = np.pad(frame_indice, 
                                      (0, required_length - len(frame_indice)), 
                                      mode='edge')
        else:
            # 验证/测试：从头开始采样密集帧
            frame_indice = np.arange(0, min(total_frames, self.max_frames))
            if len(frame_indice) < self.max_frames:
                frame_indice = np.pad(frame_indice, 
                                      (0, self.max_frames - len(frame_indice)), 
                                      mode='edge')

        video = vframes[frame_indice]  # [F=9, C, H, W]
        
        # 3. 数据变换（归一化等）
        video = self.transform(video)

        return {
            'video': video,  # [F=9, C, H, W]，密集帧
            'video_name': f"{patient_name}_slice_{slice_num:02d}",
            'video_gt': video.clone(),
            'video_path': self.video_lists[index],
            'target_interval': target_interval  # 返回当前阶段的目标间隔
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