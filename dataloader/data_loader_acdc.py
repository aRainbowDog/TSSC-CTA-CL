# # import os
# # import torch
# # import decord
# # import torchvision
# # import torchvision.transforms as transforms
# # import torch.nn.functional as F
# # import numpy as np

# # from PIL import Image
# # from einops import rearrange
# # from typing import Dict, List, Tuple

# # import re  # 新增：正则表达式模块
# # import random  # 新增：随机数模块

# # from .video_transforms import *

# # class_labels_map = None
# # cls_sample_cnt = None


# # def get_filelist(file_path):
# #     Filelist = []
# #     for home, dirs, files in os.walk(file_path):
# #         for filename in files:
# #             Filelist.append(os.path.join(home, filename))
# #             # Filelist.append( filename)
# #     return Filelist

# # class DecordInit(object):
# #     """Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

# #     def __init__(self, num_threads=1, **kwargs):
# #         self.num_threads = num_threads
# #         self.ctx = decord.cpu(0)
# #         self.kwargs = kwargs

# #     def __call__(self, filename):
# #         """Perform the Decord initialization.
# #         Args:
# #             results (dict): The resulting dict to be modified and passed
# #                 to the next transform in pipeline.
# #         """
# #         reader = decord.VideoReader(filename,
# #                                     ctx=self.ctx,
# #                                     num_threads=self.num_threads)
# #         return reader

# #     def __repr__(self):
# #         repr_str = (f'{self.__class__.__name__}('
# #                     f'sr={self.sr},'
# #                     f'num_threads={self.num_threads})')
# #         return repr_str

# # class data_loader(torch.utils.data.Dataset):
# #     """Load the video files

# #     Args:
# #         target_video_len (int): the number of video frames will be load.
# #         align_transform (callable): Align different videos in a specified size.
# #         temporal_sample (callable): Sample the target length of a video.
# #     """

# #     def __init__(self, configs, stage):
# #         self.stage = stage
# #         self.configs = configs
# #         self.target_video_len = self.configs.tar_num_frames
# #         self.v_decoder = DecordInit()

# #         self.temporal_sample = TemporalRandomCrop(configs.tar_num_frames)
# #         self.transform = transforms.Compose([
# #                     ToTensorVideo(),
# #                     CenterCropResizeVideo(configs.image_size),
# #                     # RandomHorizontalFlipVideo(),
# #                     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])

# #         if self.stage == 'train' or self.stage == 'val':
# #             self.data_path = configs.data_path_train
# #             len_train = int(0.9 * len(os.listdir(self.data_path)))
# #             if self.stage == 'train':
# #                 self.video_lists = get_filelist(self.data_path)[:len_train]
# #                 # print(stage, len(self.video_lists), self.video_lists)
# #             elif self.stage == 'val':
# #                 self.video_lists = get_filelist(self.data_path)[len_train:]
# #                 # print(stage, len(self.video_lists), self.video_lists)
# #         if self.stage == 'test':
# #             self.data_path = configs.data_path_test
# #             self.video_lists = get_filelist(self.data_path)
# #             # print(stage, len(self.video_lists), self.video_lists)

# #     # def __getitem__(self, index):
# #     #     path = self.video_lists[index]
# #     #     file_name = os.path.basename(path)
# #     #     vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')
# #     #     total_frames = len(vframes)

# #     #     if self.stage == 'train':
# #     #         # Sampling video frames
# #     #         start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
# #     #         # assert end_frame_ind - start_frame_ind >= self.target_video_len
# #     #         frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.target_video_len, dtype=int)
# #     #     else:
# #     #         if total_frames <= self.target_video_len:
# #     #             # 如果帧数不足，重复采样或补全
# #     #             frame_indice = np.linspace(0, total_frames - 1, self.target_video_len, dtype=int)
# #     #         else:
# #     #             # 均匀采样
# #     #             frame_indice = np.round(np.linspace(0, total_frames - 1, self.target_video_len)).astype(int)
# #     #     # print(total_frames, frame_indice)
# #     #     video = vframes[frame_indice]
# #     #     # videotransformer data proprecess
# #     #     video = self.transform(video)  # T C H W
# #     #     return {'video': video, 'video_name': file_name}
# #     # 核心修改__getitem__函数（替换原ACDC数据加载逻辑）
# #     def __getitem__(self, index):
# #         # 1. 读取当前切片视频（patientXXX/slice_XX.mp4）
# #         path = self.video_lists[index]
# #         file_name = os.path.basename(path)  # slice_08.mp4
# #         patient_dir = os.path.dirname(path)  # .../train/patient001
# #         patient_name = os.path.basename(patient_dir)

# #         # 解析切片号
# #         slice_num = int(re.search(r'slice_(\d+)', file_name).group(1))
# #         ext = os.path.splitext(file_name)[1]

# #         # 2. 读取相邻切片（前/后），保证4D的D维度连续性
# #         prev_slice_path = os.path.join(patient_dir, f"slice_{slice_num-1:02d}{ext}")
# #         next_slice_path = os.path.join(patient_dir, f"slice_{slice_num+1:02d}{ext}")
# #         # 边界处理：无相邻切片则用当前切片替代
# #         if not os.path.exists(prev_slice_path):
# #             prev_slice_path = path
# #         if not os.path.exists(next_slice_path):
# #             next_slice_path = path

# #         # 3. 读取前/当前/后切片的视频（均为T×3×H×W）
# #         vframes_prev, _, _ = torchvision.io.read_video(prev_slice_path, pts_unit='sec', output_format='TCHW')
# #         vframes_curr, _, _ = torchvision.io.read_video(path, pts_unit='sec', output_format='TCHW')
# #         vframes_next, _, _ = torchvision.io.read_video(next_slice_path, pts_unit='sec', output_format='TCHW')

# #         # 4. 转为灰度图（合并RGB通道，保留T×1×H×W）
# #         vframes_prev_gray = 0.299 * vframes_prev[:, 0:1, :, :] + 0.587 * vframes_prev[:, 1:2, :, :] + 0.114 * vframes_prev[:, 2:3, :, :]
# #         vframes_curr_gray = 0.299 * vframes_curr[:, 0:1, :, :] + 0.587 * vframes_curr[:, 1:2, :, :] + 0.114 * vframes_curr[:, 2:3, :, :]
# #         vframes_next_gray = 0.299 * vframes_next[:, 0:1, :, :] + 0.587 * vframes_next[:, 1:2, :, :] + 0.114 * vframes_next[:, 2:3, :, :]

# #         # 5. 对齐帧数（取最小帧数）
# #         min_frames = min(vframes_prev_gray.shape[0], vframes_curr_gray.shape[0], vframes_next_gray.shape[0])
# #         vframes_prev_gray = vframes_prev_gray[:min_frames]
# #         vframes_curr_gray = vframes_curr_gray[:min_frames]
# #         vframes_next_gray = vframes_next_gray[:min_frames]

# #         # 6. 拼接为T×3×H×W（3通道对应前/当前/后切片，保留4D的D维度信息）
# #         vframes = torch.cat([vframes_prev_gray, vframes_curr_gray, vframes_next_gray], dim=1)
# #         vframes = vframes.to(torch.uint8)

# #         # 7. 时间采样（匹配tar_num_frames）
# #         total_frames = len(vframes)
# #         if self.stage == 'train':
# #             if random.random() < 0.4:
# #                 frame_indice = np.linspace(0, total_frames - 1, self.target_video_len, dtype=int)
# #             else:
# #                 start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
# #                 frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.target_video_len, dtype=int)
# #         else:
# #             frame_indice = np.round(np.linspace(0, total_frames - 1, self.target_video_len)).astype(int)
        
# #         video = vframes[frame_indice]
# #         # 8. 数据增强/归一化
# #         if self.stage == 'test':
# #             video = self.transform(video)
# #         else:
# #             video = self.transform(video)
        
# #         return {'video': video, 'video_name': f"{patient_name}_slice_{slice_num}", 'video_gt': video}
# #     def __len__(self):
# #         return len(self.video_lists)


# # if __name__ == '__main__':
# #     from tqdm import tqdm
# #     import imageio
# #     from omegaconf import OmegaConf
# #     from torch.utils.data import DataLoader
# #     config_path = r'../configs/config_acdc.yaml'
# #     config_dict = OmegaConf.load(config_path)
# #     dataset = data_loader(config_dict, 'val')

# #     loader = DataLoader(
# #         dataset,
# #         batch_size=1,
# #         shuffle=True,
# #         num_workers=8,
# #         pin_memory=True,
# #         drop_last=True)

# #     pbar = tqdm((loader), total=len(loader), desc="\033[1;37mProcessing...]")
# #     for i, batch in enumerate(pbar):
# #         video = batch['video']
# #         print(video.shape, video.min(), video.max())
# #         video_ = ((video[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
# #         print(video_.shape)

# 相邻3帧的灰度图
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
    """加载心脏视频数据（ACDC），支持4D（时空+切片）维度"""
    def __init__(self, configs, stage):
        self.stage = stage
        self.configs = configs
        self.target_video_len = self.configs.tar_num_frames
        self.v_decoder = DecordInit()
        self.temporal_sample = TemporalRandomCrop(configs.tar_num_frames)
        
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
        
        print(f"[{self.stage}] 加载视频数量: {len(self.video_lists)}")

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

        # 3. 读取视频（使用安全读取函数）
        # vframes_prev = self._read_video_safe(prev_slice_path)
        vframes = self._read_video_safe(path)   

        # # 6. 拼接为 T×3×H×W（3通道对应前/当前/后切片）
        # vframes = torch.cat([vframes_prev_gray, vframes_curr_gray, vframes_next_gray], dim=1)
        vframes = vframes.to(torch.uint8)  # 转为uint8，避免归一化异常

        # 7. 时间采样（匹配目标帧数）
        total_frames = len(vframes)
        if self.stage == 'train':
            # 训练：随机采样 + 均匀采样混合
            if random.random() < 0.4:
                frame_indice = np.linspace(0, total_frames - 1, self.target_video_len, dtype=int)
            else:
                start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
                frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.target_video_len, dtype=int)
        else:
            # 验证/测试：均匀采样（帧数不足则重复）
            if total_frames <= self.target_video_len:
                frame_indice = np.linspace(0, total_frames - 1, self.target_video_len, dtype=int)
            else:
                frame_indice = np.round(np.linspace(0, total_frames - 1, self.target_video_len)).astype(int)

        video = vframes[frame_indice]
        # 8. 数据变换（归一化等）
        video = self.transform(video)

        return {
            'video': video,
            'video_name': f"{patient_name}_slice_{slice_num:02d}",
            'video_gt': video.clone(),  # GT 与输入一致（根据任务需求可修改）
            'video_path': self.video_lists[index],
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

    # 禁用多进程（调试阶段），避免 PyAV 多进程问题
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,  # 调试时设为0，正常训练可设为8
        pin_memory=True,
        drop_last=True
    )

    pbar = tqdm(loader, total=len(loader), desc="Processing")
    for i, batch in enumerate(pbar):
        video = batch['video']
        print(f"批次{i} - 视频形状: {video.shape}, 数值范围: [{video.min():.4f}, {video.max():.4f}]")
        
        # 可视化验证
        video_vis = ((video[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8)
        video_vis = video_vis.permute(0, 2, 3, 1).cpu().contiguous()
        print(f"可视化形状: {video_vis.shape}")
        
        # 保存示例帧（可选）
        if i == 0:
            imageio.imwrite("sample_frame.png", video_vis[0].numpy())
        if i >= 2:  # 仅测试前2个批次
            break

# # 内存不足，灰度重复三通道
# import os
# import torch
# import decord
# import torchvision
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# import numpy as np
# import re
# import random
# from PIL import Image
# from einops import rearrange
# from typing import Dict, List, Tuple

# # 解决多进程 DataLoader 中 PyAV 加载问题（提前导入并初始化）
# try:
#     import av
# except ImportError:
#     raise ImportError("请安装 PyAV 库：conda install -c conda-forge av 或 pip install av")

# class_labels_map = None
# cls_sample_cnt = None

# def get_filelist(file_path):
#     """递归获取文件列表，过滤非视频文件"""
#     Filelist = []
#     video_ext = ('.mp4', '.avi', '.mov', '.mkv')  # 仅保留视频文件
#     for home, dirs, files in os.walk(file_path):
#         for filename in files:
#             if filename.lower().endswith(video_ext):
#                 Filelist.append(os.path.join(home, filename))
#     return Filelist

# class DecordInit(object):
#     """Using Decord to initialize the video_reader (备用视频读取方案)"""
#     def __init__(self, num_threads=1, **kwargs):
#         self.num_threads = num_threads
#         self.ctx = decord.cpu(0)
#         self.kwargs = kwargs

#     def __call__(self, filename):
#         reader = decord.VideoReader(filename, ctx=self.ctx, num_threads=self.num_threads)
#         return reader

#     def __repr__(self):
#         repr_str = (f'{self.__class__.__name__}('
#                     f'num_threads={self.num_threads})')
#         return repr_str

# # 自定义视频变换类（兼容单通道转3通道）
# class ToTensorVideo(object):
#     """将视频转为张量，保持通道维度"""
#     def __call__(self, video):
#         if not isinstance(video, torch.Tensor):
#             video = torch.from_numpy(np.asarray(video))
#         # 确保维度顺序为 TCHW
#         if video.ndim == 4 and video.shape[-1] in [1, 3]:
#             video = video.permute(0, 3, 1, 2)
#         return video.float()

# class CenterCropResizeVideo(object):
#     """中心裁剪并调整视频大小"""
#     def __init__(self, size):
#         self.size = size

#     def __call__(self, video):
#         # video: TCHW
#         T, C, H, W = video.shape
#         # 先中心裁剪
#         min_side = min(H, W)
#         start_h = (H - min_side) // 2
#         start_w = (W - min_side) // 2
#         video = video[:, :, start_h:start_h+min_side, start_w:start_w+min_side]
#         # 调整大小
#         video = F.interpolate(video, size=(self.size, self.size), mode='bilinear', align_corners=False)
#         return video

# class RandomHorizontalFlipVideo(object):
#     """视频水平翻转（时空一致）"""
#     def __init__(self, p=0.5):
#         self.p = p

#     def __call__(self, video):
#         if random.random() < self.p:
#             video = torch.flip(video, dims=[3])
#         return video

# class TemporalRandomCrop(object):
#     """时间维度随机裁剪"""
#     def __init__(self, target_length):
#         self.target_length = target_length

#     def __call__(self, total_frames):
#         if total_frames <= self.target_length:
#             return 0, total_frames
#         start = random.randint(0, total_frames - self.target_length)
#         return start, start + self.target_length

# class GrayTo3Channel(object):
#     """将单通道灰度图转为3通道（复制单通道值）"""
#     def __call__(self, video):
#         # video: TCHW
#         if video.shape[1] == 1:
#             video = video.repeat(1, 3, 1, 1)  # 复制为3通道
#         return video

# class data_loader(torch.utils.data.Dataset):
#     """加载心脏视频数据（ACDC），支持4D（时空+切片）维度"""
#     def __init__(self, configs, stage):
#         self.stage = stage
#         self.configs = configs
#         self.target_video_len = self.configs.tar_num_frames
#         self.v_decoder = DecordInit()
#         self.temporal_sample = TemporalRandomCrop(configs.tar_num_frames)
        
#         # 关键修复：先转3通道，再用3通道的归一化参数
#         transform_list = [
#             ToTensorVideo(),
#             CenterCropResizeVideo(configs.image_size),
#             GrayTo3Channel(),  # 新增：单通道转3通道
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
#         ]
#         if self.stage == 'train':
#             transform_list.insert(2, RandomHorizontalFlipVideo())  # 训练加水平翻转
#         self.transform = transforms.Compose(transform_list)

#         # 划分训练/验证/测试集
#         if self.stage in ['train', 'val']:
#             self.data_path = configs.data_path_train
#             all_videos = get_filelist(self.data_path)
#             len_train = int(0.9 * len(all_videos))
#             if self.stage == 'train':
#                 self.video_lists = all_videos[:len_train]
#             else:
#                 self.video_lists = all_videos[len_train:]
#         elif self.stage == 'test':
#             self.data_path = configs.data_path_test
#             self.video_lists = get_filelist(self.data_path)
#         else:
#             raise ValueError(f"stage 必须是 train/val/test，当前为 {self.stage}")
        
#         print(f"[{self.stage}] 加载视频数量: {len(self.video_lists)}")

#     def _read_video_safe(self, video_path):
#         """安全读取视频，兼容不同格式，返回 TCHW 格式张量"""
#         try:
#             # 使用 torchvision 读取（依赖 PyAV）
#             vframes, aframes, info = torchvision.io.read_video(
#                 filename=video_path, 
#                 pts_unit='sec', 
#                 output_format='TCHW',
#                 pts_per_second=15  # 固定帧率，避免不同视频帧率不一致
#             )
#             return vframes
#         except Exception as e:
#             # 备用方案：使用 Decord 读取
#             reader = self.v_decoder(video_path)
#             frames = reader.get_batch(list(range(len(reader)))).asnumpy()  # THWC
#             frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # 转为 TCHW
#             return frames

#     def __getitem__(self, index):
#         # 1. 读取当前切片视频
#         path = self.video_lists[index]
#         file_name = os.path.basename(path)
#         patient_dir = os.path.dirname(path)
#         patient_name = os.path.basename(patient_dir)

#         # 解析切片号
#         slice_match = re.search(r'slice_(\d+)', file_name)
#         if not slice_match:
#             raise ValueError(f"文件名 {file_name} 未匹配到 slice_XX 格式")
#         slice_num = int(slice_match.group(1))
#         ext = os.path.splitext(file_name)[1]

#         # 2. 读取视频
#         vframes = self._read_video_safe(path)
        
#         # 转灰度图（保留单通道）
#         if vframes.shape[1] == 3:
#             # RGB转灰度：Gray = 0.299*R + 0.587*G + 0.114*B
#             vframes = 0.299 * vframes[:, 0:1, :, :] + \
#                       0.587 * vframes[:, 1:2, :, :] + \
#                       0.114 * vframes[:, 2:3, :, :]
        
#         # 转为float类型
#         vframes = vframes.float()

#         # 3. 时间采样（匹配目标帧数）
#         total_frames = len(vframes)
#         if self.stage == 'train':
#             if random.random() < 0.4:
#                 frame_indice = np.linspace(0, total_frames - 1, self.target_video_len, dtype=int)
#             else:
#                 start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
#                 frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.target_video_len, dtype=int)
#         else:
#             if total_frames <= self.target_video_len:
#                 frame_indice = np.linspace(0, total_frames - 1, self.target_video_len, dtype=int)
#             else:
#                 frame_indice = np.round(np.linspace(0, total_frames - 1, self.target_video_len)).astype(int)

#         video = vframes[frame_indice]
        
#         # 4. 数据变换（包含转3通道和归一化）
#         video = self.transform(video)

#         return {
#             'video': video,
#             'video_name': f"{patient_name}_slice_{slice_num:02d}",
#             'video_gt': video.clone(),
#             'video_path': self.video_lists[index],
#         }

#     # 补充：优化_read_video_safe，提前限制帧数
#     def _read_video_safe(self, video_path):
#         try:
#             # 限制读取帧数（避免超长视频）
#             vframes, aframes, info = torchvision.io.read_video(
#                 filename=video_path, 
#                 pts_unit='sec', 
#                 output_format='TCHW',
#                 pts_per_second=15,
#                 end_pts=1.0  # 仅读取前1秒（约15帧）
#             )
#             return vframes
#         except Exception as e:
#             reader = self.v_decoder(video_path)
#             # 限制读取帧数
#             max_frames = min(len(reader), 15)
#             frames = reader.get_batch(list(range(max_frames))).asnumpy()
#             frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
#             return frames

#     def __len__(self):
#         return len(self.video_lists)

# if __name__ == '__main__':
#     from tqdm import tqdm
#     import imageio
#     from omegaconf import OmegaConf
#     from torch.utils.data import DataLoader

#     # 加载配置
#     config_path = r'../configs/config_acdc.yaml'
#     config_dict = OmegaConf.load(config_path)
#     dataset = data_loader(config_dict, 'val')

#     # 禁用多进程（调试阶段）
#     loader = DataLoader(
#         dataset,
#         batch_size=1,
#         shuffle=True,
#         num_workers=0,
#         pin_memory=True,
#         drop_last=True
#     )

#     pbar = tqdm(loader, total=len(loader), desc="Processing")
#     for i, batch in enumerate(pbar):
#         video = batch['video']
#         print(f"批次{i} - 视频形状: {video.shape}, 数值范围: [{video.min():.4f}, {video.max():.4f}]")
        
#         # 验证通道数是否为3
#         assert video.shape[2] == 3, f"通道数应为3，实际为{video.shape[2]}"
        
#         # 可视化验证
#         video_vis = ((video[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8)
#         video_vis = video_vis.permute(0, 2, 3, 1).cpu().contiguous()
#         print(f"可视化形状: {video_vis.shape}")
        
#         # 保存示例帧（可选）
#         if i == 0:
#             imageio.imwrite("sample_frame.png", video_vis[0].numpy())
#         if i >= 2:  # 仅测试前2个批次
#             break