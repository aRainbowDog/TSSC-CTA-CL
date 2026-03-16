# autodl
import os
import gc
# 临时开启expandable_segments（加载阶段用，训练时再关闭）
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import math
import argparse
import logging
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.utils import save_image
from glob import glob
from time import time
from copy import deepcopy
from einops import rearrange
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from models.model_dit import MVIF_models
from models.diffusion.gaussian_diffusion import (
    create_diffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)
from dataloader.data_loader_acdc import (
    collate_full_sequence_batch,
    data_loader,
    full_sequence_data_loader,
)
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Sampler
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils.triplet_eval import evaluate_video_sliding_triplets
from utils.utils import (
    clip_grad_norm_, create_logger, update_ema, requires_grad, 
    cleanup, create_experiment_tracker, write_experiment_metric,
    write_experiment_images, close_experiment_tracker, setup_distributed
)
import warnings
# ========== 新增mask生成相关依赖 ==========
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.filters import apply_hysteresis_threshold
warnings.filterwarnings('ignore')


def log_debug(logger, *lines):
    if logger is None or not logger.isEnabledFor(logging.DEBUG):
        return
    for line in lines:
        logger.debug(line)


def create_logger_compat(logging_dir, level="INFO", console=None):
    """Support both old and new create_logger signatures."""
    try:
        return create_logger(logging_dir, level=level, console=console)
    except TypeError:
        logger = create_logger(logging_dir)
        resolved_level = getattr(logging, str(level).upper(), logging.INFO)
        logger.setLevel(resolved_level)
        for handler in logger.handlers:
            handler.setLevel(resolved_level)
        return logger


def distributed_barrier():
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


def create_rich_progress(console=None):
    return Progress(
        SpinnerColumn(style="bold blue"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=28),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[status]}", justify="left"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console or Console(stderr=True),
        transient=False,
        expand=True,
        refresh_per_second=4,
    )


def maybe_cleanup_cuda(step, interval):
    if interval is None or interval <= 0 or step <= 0 or step % interval != 0:
        return
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def get_fixed_visualization_indices(dataset_len, count):
    if dataset_len <= 0 or count <= 0:
        return []
    if count >= dataset_len:
        return list(range(dataset_len))
    return sorted({int(idx) for idx in np.linspace(0, dataset_len - 1, num=count)})


def get_visualization_stage_info(current_step, stage0_steps, stage1_steps, stage2_steps):
    """Return visualization stage id/name for the current training step."""
    if current_step < stage0_steps:
        return 0, "Stage0_Gap1"
    if current_step < stage1_steps:
        return 1, "Stage1_Gap2"
    if current_step < stage2_steps:
        return 2, "Stage2_Gap4"
    return 3, "Stage3_Gap8"


class DistributedEvalSampler(Sampler):
    """Shard validation samples across ranks without padding or duplication."""

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        if remaining <= 0:
            return 0
        return (remaining + self.num_replicas - 1) // self.num_replicas


def get_stage_metadata(train_steps, args):
    if train_steps < args.stage0_steps:
        return {"stage": 0, "gap": 1, "frames": 2, "next": f"s1@{args.stage0_steps}"}
    if train_steps < args.stage1_steps:
        return {"stage": 1, "gap": 2, "frames": 3, "next": f"s2@{args.stage1_steps}"}
    if train_steps < args.stage2_steps:
        return {"stage": 2, "gap": 4, "frames": 5, "next": f"s3@{args.stage2_steps}"}
    return {"stage": 3, "gap": 8, "frames": 9, "next": "done"}


def format_overall_progress_status(epoch, num_train_epochs, train_steps, args):
    meta = get_stage_metadata(train_steps, args)
    return (
        f"epoch {epoch + 1}/{num_train_epochs} | "
        f"opt {train_steps}/{args.max_train_steps} | "
        f"s{meta['stage']} g{meta['gap']} f{meta['frames']} | "
        f"next {meta['next']}"
    )


def format_epoch_progress_status(
        batch_step,
        total_batches,
        train_steps,
        max_train_steps,
        stage=None,
        frame_gap=None,
        num_frames=None,
        loss_value=None,
        recursive_loss_value=None,
        grad_norm=None):
    status = [
        f"batch {batch_step}/{total_batches}",
        f"opt {train_steps}/{max_train_steps}",
    ]
    if stage is not None and frame_gap is not None and num_frames is not None:
        status.append(f"s{stage} g{frame_gap} f{num_frames}")
    if loss_value is not None:
        status.append(f"loss {loss_value:.4f}")
    if recursive_loss_value is not None:
        status.append(f"rec {recursive_loss_value:.4f}")
    if grad_norm is not None:
        status.append(f"gn {grad_norm:.4f}")
    return " | ".join(status)

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    从1D numpy数组中提取值，用于batch处理
    Args:
        arr: 1D numpy array
        timesteps: [B] tensor of timestep indices
        broadcast_shape: 目标形状
    Returns:
        extracted values with correct shape
    """
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr).to(device=timesteps.device, dtype=torch.float32)
    
    res = arr[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    
    return res.expand(broadcast_shape)
# -------------------------- 新增：Mask生成函数 --------------------------
def generate_vessel_mask_adaptive(frames_gray, max_weight=100.0, base_weight=1.0):
    """
    生成自适应血管mask和软权重
    Args:
        frames_gray: list of numpy array, 灰度帧列表 [f, h, w]
        max_weight: float, 血管区域最大权重（最高100倍）
        base_weight: float, 非血管区域基础权重
    Returns:
        mask_final: numpy array, 二值mask [h, w]
        soft_weight: numpy array, 软权重图 [h, w]（值范围[base_weight, max_weight]）
    """
    first_frame = frames_gray[0].astype(np.float32)
    last_frame = frames_gray[-1].astype(np.float32)
    
    # 计算帧差
    diff_map = np.abs(first_frame - last_frame)
    diff_smooth = cv2.GaussianBlur(diff_map, (3, 3), 0)
    
    # 归一化+增强对比度
    diff_norm = (diff_smooth / (diff_smooth.max() + 1e-5) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    diff_enhanced = clahe.apply(diff_norm)
    
    # 自适应阈值
    flat_data = diff_enhanced.flatten()
    dynamic_high = np.percentile(flat_data, 97.5) 
    dynamic_low = np.percentile(flat_data, 96)  
    dynamic_high = max(dynamic_high, 70) 
    dynamic_low = max(dynamic_low, 40)

    # 滞后阈值分割
    mask_binary = apply_hysteresis_threshold(
        diff_enhanced, dynamic_low, dynamic_high
    ).astype(np.uint8) * 255
    
    # 形态学操作优化mask
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_refined = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_connected = cv2.morphologyEx(mask_refined, cv2.MORPH_CLOSE, kernel_close)
    
    # 筛选有效轮廓
    cnts, _ = cv2.findContours(mask_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_final = np.zeros_like(mask_connected)
    h, w = mask_final.shape
    for c in cnts:
        area = cv2.contourArea(c)
        if 20 < area < (h * w * 0.2):  # 过滤过小/过大轮廓
            cv2.drawContours(mask_final, [c], -1, 255, -1)
    
    # 生成软权重（调整到max_weight范围）
    if np.any(mask_final > 0):
        dist = cv2.distanceTransform(mask_final, cv2.DIST_L2, 5)
        # 权重计算：距离越近权重越高，最大值max_weight，最小值base_weight
        soft_weight = np.exp(-dist / 8.0) * (max_weight - base_weight) + base_weight
        # 确保权重不超过最大值
        soft_weight = np.clip(soft_weight, base_weight, max_weight)
    else:
        soft_weight = np.ones_like(diff_map, dtype=np.float32) * base_weight
    
    return mask_final, soft_weight

def prepare_mask_for_latent(soft_weight_np, latent_size, device):
    """
    将256×256的软权重图转换为latent空间的权重tensor
    Args:
        soft_weight_np: numpy array [h, w] (256×256)
        latent_size: int, latent空间尺寸（32）
        device: torch.device
    Returns:
        weight_latent: torch.Tensor [1, 1, latent_size, latent_size]
    """
    # 转换为tensor并调整维度
    weight_tensor = torch.from_numpy(soft_weight_np).float().to(device)  # [256, 256]
    weight_tensor = weight_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 256, 256]
    
    # 下采样到latent尺寸（使用双线性插值，保留权重分布）
    weight_latent = F.interpolate(
        weight_tensor, 
        size=(latent_size, latent_size), 
        mode='bilinear', 
        align_corners=False
    )  # [1, 1, 32, 32]
    
    return weight_latent

# -------------------------- 全局工具函数（不变） --------------------------
def get_raw_model(model):
    """获取DDP包装后的原始模型，若无DDP则返回原模型"""
    return model.module if hasattr(model, 'module') else model


def ddp_sync_context(model, should_sync):
    if isinstance(model, DDP) and not should_sync:
        return model.no_sync()
    return nullcontext()

# ========== 新增：课程学习辅助函数 ==========
def get_visible_frames(total_frames, interval):
    """
    根据间隔返回可见帧的索引
    Args:
        total_frames: 总帧数（例如9）
        interval: 采样间隔（1, 2, 或 4）
    Returns:
        可见帧索引列表（例如interval=2时返回[0,2,4,6,8]）
    """
    visible_idx = list(range(0, total_frames, interval))
    return visible_idx

def compute_recursive_loss_stage2(model, diffusion, latent, t, f, b, h_latent, w_latent, device):
    """
    阶段2的递归链Loss: 用间隔2补全间隔1
    ✅ 方案3：手动计算loss（完全绕过training_losses）
    Args:
        model: 模型
        diffusion: 扩散模型对象
        latent: 潜在编码 [B, F, C, H, W]
        t: 时间步 [B]
        f: 帧数
        其他: 维度信息
    Returns:
        递归链loss (tensor)
    """
    recursive_losses = []
    pairs = [(0, 2, 1), (2, 4, 3), (4, 6, 5), (6, 8, 7)]
    
    for left_idx, right_idx, mid_idx in pairs:
        if mid_idx < f:
            # ========== 步骤1：提取三元组 ==========
            triplet = torch.stack([
                latent[:, left_idx],   # [B, C, H, W]
                latent[:, mid_idx],    # [B, C, H, W] - 目标帧
                latent[:, right_idx]   # [B, C, H, W]
            ], dim=1)  # [B, 3, C, H, W]
            
            x_start = triplet.permute(0, 2, 1, 3, 4)  # [B, C, 3, H, W]
            
            # ========== 步骤2：生成噪声 ==========
            noise = torch.randn_like(x_start)
            
            # ========== 步骤3：提取扩散系数 ==========
            sqrt_alphas_cumprod = _extract_into_tensor(
                diffusion.sqrt_alphas_cumprod, t, x_start.shape
            )
            sqrt_one_minus_alphas_cumprod = _extract_into_tensor(
                diffusion.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            
            # ========== 步骤4：加噪 q(x_t | x_0) ==========
            x_t = sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise
            
            # ========== 步骤5：构建mask（left和right可见，middle需要预测）==========
            # 转换回 [B, 3, C, H, W] 方便mask操作
            x_t_frames = x_t.permute(0, 2, 1, 3, 4)      # [B, 3, C, H, W]
            x_start_frames = x_start.permute(0, 2, 1, 3, 4)  # [B, 3, C, H, W]
            
            # mask: [B, 3, 1, H, W]，1表示需要预测（mask掉），0表示可见
            mask = torch.ones(b, 3, 1, h_latent, w_latent, device=device)
            mask[:, 0] = 0  # left可见
            mask[:, 2] = 0  # right可见
            # middle (idx=1) 保持为1（需要预测）
            
            # ========== 步骤6：融合可见帧和noisy帧 ==========
            model_input = x_start_frames * (1 - mask) + x_t_frames * mask  # [B, 3, C, H, W]
            
            # ========== 步骤7：模型预测噪声 ==========
            model_output = model(model_input, t)  # [B, 3, C, H, W]
            
            # ========== 步骤8：只计算中间帧的loss ==========
            # 提取中间帧的预测和目标
            pred_noise_middle = model_output[:, 1, :, :, :]  # [B, C, H, W]
            target_noise_middle = noise.permute(0, 2, 1, 3, 4)[:, 1, :, :, :]  # [B, C, H, W]
            
            # MSE loss
            loss_middle = F.mse_loss(pred_noise_middle, target_noise_middle, reduction='mean')
            recursive_losses.append(loss_middle)
    
    if recursive_losses:
        return torch.stack(recursive_losses).mean()
    else:
        return torch.tensor(0.0, device=device)

def compute_recursive_loss_stage3(model, diffusion, latent, t, f, b, h_latent, w_latent, device):
    """
    阶段3的多层递归链Loss: 间隔4 → 间隔2 → 间隔1
    ✅ 方案3：手动计算loss
    """
    all_losses = []
    
    # ========== 辅助函数：计算单个三元组的loss ==========
    def compute_single_triplet_loss(left_idx, mid_idx, right_idx):
        """计算单个三元组[left, middle, right]的loss"""
        # 提取三元组
        triplet = torch.stack([
            latent[:, left_idx],
            latent[:, mid_idx],   # 目标帧
            latent[:, right_idx]
        ], dim=1)  # [B, 3, C, H, W]
        
        x_start = triplet.permute(0, 2, 1, 3, 4)  # [B, C, 3, H, W]
        noise = torch.randn_like(x_start)
        
        # 提取扩散系数
        sqrt_alphas = _extract_into_tensor(
            diffusion.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus = _extract_into_tensor(
            diffusion.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        # 加噪
        x_t = sqrt_alphas * x_start + sqrt_one_minus * noise
        
        # 构建mask
        x_t_frames = x_t.permute(0, 2, 1, 3, 4)
        x_start_frames = x_start.permute(0, 2, 1, 3, 4)
        
        mask = torch.ones(b, 3, 1, h_latent, w_latent, device=device)
        mask[:, 0] = 0  # left可见
        mask[:, 2] = 0  # right可见
        
        # 融合
        model_input = x_start_frames * (1 - mask) + x_t_frames * mask
        
        # 模型预测
        model_output = model(model_input, t)
        
        # 计算中间帧loss
        pred_middle = model_output[:, 1, :, :, :]
        target_middle = noise.permute(0, 2, 1, 3, 4)[:, 1, :, :, :]
        
        return F.mse_loss(pred_middle, target_middle, reduction='mean')
    
    # ========== 第一层递归：用帧0,8预测帧4 ==========
    if f > 4:
        layer1_loss = compute_single_triplet_loss(0, 4, 8)
        all_losses.append(layer1_loss)
    
    # ========== 第二层递归：用间隔2补全到间隔1 ==========
    layer2_pairs = [(0, 2, 4), (4, 6, 8)]
    for left_idx, mid_idx, right_idx in layer2_pairs:
        if mid_idx < f:
            loss = compute_single_triplet_loss(left_idx, mid_idx, right_idx)
            all_losses.append(loss)
    
    # ========== 第三层递归：用间隔1补全所有奇数帧 ==========
    layer3_pairs = [(0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, 8)]
    for left_idx, mid_idx, right_idx in layer3_pairs:
        if mid_idx < f:
            loss = compute_single_triplet_loss(left_idx, mid_idx, right_idx)
            all_losses.append(loss)
    
    if all_losses:
        return torch.stack(all_losses).mean()
    else:
        return torch.tensor(0.0, device=device)

def set_optimizer_zeros_grad(optimizer, set_to_none=True):
    """安全的optimizer梯度清零，减少显存碎片"""
    if set_to_none:
        optimizer.zero_grad(set_to_none=True)
    else:
        optimizer.zero_grad()

def safe_load_checkpoint(ckpt_path, device):
    """
    加载训练checkpoint到CPU，后续按需恢复到目标device。
    """
    del device
    return torch.load(ckpt_path, map_location="cpu")


def move_optimizer_state_to_device(optimizer, device):
    """Ensure optimizer state tensors live on the same device as model params after resume."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device, non_blocking=True)


def build_training_checkpoint(
    model,
    ema,
    train_steps,
    epoch,
    best_val_metric,
    opt=None,
    lr_scheduler=None,
    scaler=None,
):
    checkpoint = {
        "model": get_raw_model(model).state_dict(),
        "ema": ema.state_dict(),
        "train_steps": train_steps,
        "epoch": epoch,
        "best_val_metric": best_val_metric,
    }
    if opt is not None:
        checkpoint["opt"] = opt.state_dict()
    if lr_scheduler is not None:
        checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint


def align_weight_batch(weight_batch, batch_size, num_channels, height, width):
    """Accept [B,1,H,W] or [B,C,H,W] weights and expand to [B,C,H,W] for loss weighting."""
    if weight_batch is None:
        return None
    if weight_batch.dim() == 3:
        weight_batch = weight_batch.unsqueeze(1)
    if weight_batch.dim() != 4:
        raise ValueError(f"weight_batch must be 3D or 4D, got shape {tuple(weight_batch.shape)}")
    if weight_batch.shape[0] != batch_size or weight_batch.shape[-2:] != (height, width):
        raise ValueError(
            f"weight_batch shape {tuple(weight_batch.shape)} is incompatible with target "
            f"({batch_size}, {num_channels}, {height}, {width})"
        )
    if weight_batch.shape[1] == 1:
        return weight_batch.expand(-1, num_channels, -1, -1)
    if weight_batch.shape[1] != num_channels:
        raise ValueError(
            f"weight_batch channel dim must be 1 or {num_channels}, got {weight_batch.shape[1]}"
        )
    return weight_batch

# 新增：EMA decay调度函数（关键优化）
def get_ema_decay(train_steps, base_decay=0.9999, min_decay=0.99, warmup_steps=1000):
    """
    动态调整EMA decay值：
    - 热身阶段线性增加decay，避免初始阶段EMA更新过快
    - 训练后期使用高decay，稳定EMA权重
    """
    if train_steps < warmup_steps:
        decay = min_decay + (base_decay - min_decay) * (train_steps / warmup_steps)
    else:
        decay = base_decay
    return decay

def save_val_sample_visualization(epoch, video_val, vae, val_diffusion, device, val_fold,
                                  generate_vessel_mask_adaptive, raw_model,
                                  current_step, stage0_steps, stage1_steps, stage2_steps,
                                  sample_ids=None, ema_model=None, logger=None):
    """
    (epoch, video_val, vae, val_diffusion, device, val_fold, generate_vessel_mask_adaptive, raw_model, ema_model=None)
    三元组验证可视化（4行完整版）
    行1: GT
    行2: Masked输入
    行3: 预测结果
    行4: 血管mask
    """
    b_val, f_val, c_val, h_val, w_val = video_val.shape

    stage, stage_name = get_visualization_stage_info(current_step, stage0_steps, stage1_steps, stage2_steps)

    # 确定当前阶段
    if stage == 0:
        if logger is not None:
            logger.warning(f"Epoch {epoch+1}: Stage0，跳过可视化")
        else:
            print(f"⚠️  Epoch {epoch+1}: Stage0，跳过可视化")
        return {}
    
    # VAE编码全部密集帧
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        video_val_flat = rearrange(video_val, 'b f c h w -> (b f) c h w')
        latent_val = vae.encode(video_val_flat).latent_dist.sample().mul_(0.18215)
        latent_val = rearrange(latent_val, '(b f) c h w -> b f c h w', b=b_val)
    
    b_val, f_val, c_latent, h_latent_val, w_latent_val = latent_val.shape
    sample_ids = list(sample_ids) if sample_ids is not None else list(range(b_val))
    
    # 生成血管mask可视化
    video_val_np = video_val.detach().cpu().numpy()
    video_val_gray = np.mean(video_val_np, axis=2)
    vessel_mask_batches = []
    for batch_idx in range(b_val):
        frames_gray = [video_val_gray[batch_idx, t] for t in range(f_val)]
        mask_final, _ = generate_vessel_mask_adaptive(frames_gray)
        vessel_mask_vis = torch.from_numpy(mask_final).float().to(device)
        vessel_mask_vis = vessel_mask_vis.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        vessel_mask_vis = vessel_mask_vis.repeat(1, f_val, 3, 1, 1)
        vessel_mask_vis = (vessel_mask_vis / 255.0) * 2 - 1
        vessel_mask_batches.append(vessel_mask_vis)
    vessel_mask_vis = torch.cat(vessel_mask_batches, dim=0)
    
    # ========== 1. 基础预测（三元组，一步到位） ==========
    with torch.no_grad():
        if stage == 1:
            # 阶段1：三元组 [s, s+1, s+2]，预测s+1
            triplet_latent = latent_val[:, 0:3, :, :, :]  # [B, 3, C, H, W]
            triplet_video_gt = video_val[:, 0:3, :, :, :]
            triplet_vessel_mask = vessel_mask_vis[:, 0:3, :, :, :]
            frame_indices_original = [0, 1, 2]
            
        elif stage == 2:
            # 阶段2：三元组 [s, s+2, s+4]，预测s+2
            triplet_latent = torch.stack([
                latent_val[:, 0],
                latent_val[:, 2],
                latent_val[:, 4]
            ], dim=1)  # [B, 3, C, H, W]
            triplet_video_gt = torch.stack([
                video_val[:, 0],
                video_val[:, 2],
                video_val[:, 4]
            ], dim=1)
            triplet_vessel_mask = torch.stack([
                vessel_mask_vis[:, 0],
                vessel_mask_vis[:, 2],
                vessel_mask_vis[:, 4]
            ], dim=1)
            frame_indices_original = [0, 2, 4]
            
        else:  # stage == 3
            # 阶段3：三元组 [s, s+4, s+8]，预测s+4
            triplet_latent = torch.stack([
                latent_val[:, 0],
                latent_val[:, 4],
                latent_val[:, 8]
            ], dim=1)
            triplet_video_gt = torch.stack([
                video_val[:, 0],
                video_val[:, 4],
                video_val[:, 8]
            ], dim=1)
            triplet_vessel_mask = torch.stack([
                vessel_mask_vis[:, 0],
                vessel_mask_vis[:, 4],
                vessel_mask_vis[:, 8]
            ], dim=1)
            frame_indices_original = [0, 4, 8]
        
        # 构建三元组mask（首尾可见，中间预测）
        mask_base = torch.ones(b_val, 3, h_latent_val, w_latent_val, device=device)
        mask_base[:, 0, :, :] = 0  # 首帧可见
        mask_base[:, 2, :, :] = 0  # 尾帧可见
        
        # 采样预测
        z = torch.randn_like(triplet_latent.permute(0, 2, 1, 3, 4))
        samples_base = val_diffusion.p_sample_loop(
            raw_model.forward, z.shape, z,
            clip_denoised=False, progress=False, device=device,
            raw_x=triplet_latent.permute(0, 2, 1, 3, 4),
            mask=mask_base
        )
        
        # 解码
        samples_base = samples_base.permute(1, 0, 2, 3, 4) * mask_base + triplet_latent.permute(2, 0, 1, 3, 4) * (1 - mask_base)
        samples_base = samples_base.permute(1, 2, 0, 3, 4)  # [B, 3, C, H, W]
        samples_base_flat = rearrange(samples_base, 'b f c h w -> (b f) c h w') / 0.18215
        decoded_base = vae.decode(samples_base_flat).sample
        decoded_base = rearrange(decoded_base, '(b f) c h w -> b f c h w', b=b_val)
        decoded_base_gray = decoded_base.mean(dim=2, keepdim=True)
        decoded_base = decoded_base_gray.repeat(1, 1, 3, 1, 1)
        
        # 构建mask可视化（灰色遮罩）
        mask_base_vis = torch.ones(b_val, 3, 3, h_val, w_val, device=device) * 0.5
        mask_base_vis[:, 0, :, :, :] = 0  # 第0帧（首帧）不遮罩
        mask_base_vis[:, 2, :, :, :] = 0  # 第2帧（尾帧）不遮罩
        
        base_images = []
        for sample_idx, sample_id in enumerate(sample_ids):
            video_val_single = triplet_video_gt[sample_idx:sample_idx+1]
            decoded_base_single = decoded_base[sample_idx:sample_idx+1]
            mask_base_vis_single = mask_base_vis[sample_idx:sample_idx+1]
            vessel_mask_single = triplet_vessel_mask[sample_idx:sample_idx+1]

            val_pic_base = torch.cat([
                video_val_single,
                video_val_single * (1 - mask_base_vis_single),
                decoded_base_single,
                vessel_mask_single
            ], dim=1)

            val_pic_base_flat = rearrange(val_pic_base, 'b f c h w -> (b f) c h w')
            base_image_path = os.path.join(
                val_fold,
                f"Epoch_{epoch+1}_{stage_name}_base_triplet_idx_{sample_id:04d}.png",
            )
            save_image(
                val_pic_base_flat,
                base_image_path,
                nrow=3,
                normalize=True,
                value_range=(-1, 1)
            )
            base_images.append({
                "path": base_image_path,
                "caption": f"Epoch {epoch+1} {stage_name} base triplet idx {sample_id}",
            })

        del z, samples_base, decoded_base

    generated_images = {
        "Val Examples/BaseTriplet": base_images
    }
    
    # ========== 2. 递归链预测（逐层细化，完整帧序列） ==========
    if stage >= 2:  # 阶段2和3才有递归链
        with torch.no_grad():
            if stage == 2:
                # 阶段2递归链：5帧完整可视化
                result_frames = [None] * 5
                
                # 第1步：用[s, s+4]预测s+2
                triplet_step1 = torch.stack([latent_val[:, 0], latent_val[:, 2], latent_val[:, 4]], dim=1)
                mask_step1 = torch.ones(b_val, 3, h_latent_val, w_latent_val, device=device)
                mask_step1[:, 0, :, :] = 0
                mask_step1[:, 2, :, :] = 0
                
                z1 = torch.randn_like(triplet_step1.permute(0, 2, 1, 3, 4))
                samples1 = val_diffusion.p_sample_loop(
                    raw_model.forward, z1.shape, z1,
                    clip_denoised=False, progress=False, device=device,
                    raw_x=triplet_step1.permute(0, 2, 1, 3, 4),
                    mask=mask_step1
                )
                # samples1 = samples1.permute(1, 0, 2, 3, 4)
                # predicted_s2_latent = samples1[:, 1, :, :, :].clone()
                # 融合mask + permute回来
                samples1 = samples1.permute(1, 0, 2, 3, 4) * mask_step1 + \
                           triplet_step1.permute(2, 0, 1, 3, 4) * (1 - mask_step1)
                samples1 = samples1.permute(1, 2, 0, 3, 4)  # [B, 3, C, H, W] ✅
                
                predicted_s2_latent = samples1[:, 1, :, :, :].clone()  # [B, C, H, W] ✅
                
                # 解码s+2
                decoded_s2 = vae.decode(predicted_s2_latent / 0.18215).sample
                decoded_s2_gray = decoded_s2.mean(dim=1, keepdim=True)
                result_frames[0] = video_val[:, 0]  # s (GT)
                result_frames[2] = decoded_s2_gray.repeat(1, 3, 1, 1)  # s+2 (预测)
                result_frames[4] = video_val[:, 4]  # s+4 (GT)
                
                del z1, samples1
                
                # 第2步：用[s, pred_s+2]预测s+1
                triplet_step2 = torch.stack([latent_val[:, 0], latent_val[:, 1], predicted_s2_latent.detach()], dim=1)
                z2 = torch.randn_like(triplet_step2.permute(0, 2, 1, 3, 4))
                samples2 = val_diffusion.p_sample_loop(
                    raw_model.forward, z2.shape, z2,
                    clip_denoised=False, progress=False, device=device,
                    raw_x=triplet_step2.permute(0, 2, 1, 3, 4),
                    mask=mask_step1
                )
                # samples2 = samples2.permute(1, 0, 2, 3, 4)
                # decoded_s1 = vae.decode(samples2[:, 1, :, :, :] / 0.18215).sample
                samples2 = samples2.permute(1, 0, 2, 3, 4) * mask_step1 + \
                           triplet_step2.permute(2, 0, 1, 3, 4) * (1 - mask_step1)
                samples2 = samples2.permute(1, 2, 0, 3, 4)  # [B, 3, C, H, W] ✅
                
                decoded_s1 = vae.decode(samples2[:, 1, :, :, :] / 0.18215).sample  # ✅
                decoded_s1_gray = decoded_s1.mean(dim=1, keepdim=True)
                result_frames[1] = decoded_s1_gray.repeat(1, 3, 1, 1)
                
                del z2, samples2
                
                # 第3步：用[pred_s+2, s+4]预测s+3
                triplet_step3 = torch.stack([predicted_s2_latent.detach(), latent_val[:, 3], latent_val[:, 4]], dim=1)
                z3 = torch.randn_like(triplet_step3.permute(0, 2, 1, 3, 4))
                samples3 = val_diffusion.p_sample_loop(
                    raw_model.forward, z3.shape, z3,
                    clip_denoised=False, progress=False, device=device,
                    raw_x=triplet_step3.permute(0, 2, 1, 3, 4),
                    mask=mask_step1
                )
                # samples3 = samples3.permute(1, 0, 2, 3, 4)
                # decoded_s3 = vae.decode(samples3[:, 1, :, :, :] / 0.18215).sample
                samples3 = samples3.permute(1, 0, 2, 3, 4) * mask_step1 + \
                           triplet_step3.permute(2, 0, 1, 3, 4) * (1 - mask_step1)
                samples3 = samples3.permute(1, 2, 0, 3, 4)  # [B, 3, C, H, W] ✅
                
                decoded_s3 = vae.decode(samples3[:, 1, :, :, :] / 0.18215).sample  # ✅
                decoded_s3_gray = decoded_s3.mean(dim=1, keepdim=True)
                result_frames[3] = decoded_s3_gray.repeat(1, 3, 1, 1)
                
                del z3, samples3, predicted_s2_latent
                
                # 拼接5帧结果
                result_video = torch.stack(result_frames, dim=1)  # [B, 5, C, H, W]
                num_frames_rec = 5
            else:  # stage == 3
                # ✅ 阶段3递归链：完整版，逐层递归预测9帧
                # 递归结构：
                # 第1层：[0, 8] → 4
                # 第2层：[0, 4] → 2, [4, 8] → 6
                # 第3层：[0, 2] → 1, [2, 4] → 3, [4, 6] → 5, [6, 8] → 7
                
                result_frames = [None] * 9
                
                # 预先填充GT帧
                result_frames[0] = video_val[:, 0]  # frame_0 (GT)
                result_frames[8] = video_val[:, 8]  # frame_8 (GT)
                
                # 通用mask（首尾可见，中间预测）
                mask_triplet = torch.ones(b_val, 3, h_latent_val, w_latent_val, device=device)
                mask_triplet[:, 0, :, :] = 0  # 首帧可见
                mask_triplet[:, 2, :, :] = 0  # 尾帧可见
                
                # ========== 第1层：用[0, 8]预测4 ==========
                triplet_l1 = torch.stack([
                    latent_val[:, 0],   # frame_0
                    latent_val[:, 4],   # frame_4（占位，将被预测）
                    latent_val[:, 8]    # frame_8
                ], dim=1)  # [B, 3, C, H, W]
                
                z_l1 = torch.randn_like(triplet_l1.permute(0, 2, 1, 3, 4))
                samples_l1 = val_diffusion.p_sample_loop(
                    raw_model.forward, z_l1.shape, z_l1,
                    clip_denoised=False, progress=False, device=device,
                    raw_x=triplet_l1.permute(0, 2, 1, 3, 4),
                    mask=mask_triplet
                )
                
                # 融合+提取
                samples_l1 = samples_l1.permute(1, 0, 2, 3, 4) * mask_triplet + \
                             triplet_l1.permute(2, 0, 1, 3, 4) * (1 - mask_triplet)
                samples_l1 = samples_l1.permute(1, 2, 0, 3, 4)  # [B, 3, C, H, W]
                pred_4_latent = samples_l1[:, 1, :, :, :].clone()  # [B, C, H, W]
                
                # 解码frame_4
                decoded_4 = vae.decode(pred_4_latent / 0.18215).sample
                decoded_4_gray = decoded_4.mean(dim=1, keepdim=True)
                result_frames[4] = decoded_4_gray.repeat(1, 3, 1, 1)
                
                del z_l1, samples_l1
                torch.cuda.empty_cache()
                # ========== 第2层：用[0, pred_4]预测2，用[pred_4, 8]预测6 ==========
                # 预测frame_2
                triplet_l2a = torch.stack([
                    latent_val[:, 0],
                    latent_val[:, 2],           # 占位
                    pred_4_latent.detach()
                ], dim=1)
                
                z_l2a = torch.randn_like(triplet_l2a.permute(0, 2, 1, 3, 4))
                samples_l2a = val_diffusion.p_sample_loop(
                    raw_model.forward, z_l2a.shape, z_l2a,
                    clip_denoised=False, progress=False, device=device,
                    raw_x=triplet_l2a.permute(0, 2, 1, 3, 4),
                    mask=mask_triplet
                )
                
                samples_l2a = samples_l2a.permute(1, 0, 2, 3, 4) * mask_triplet + \
                              triplet_l2a.permute(2, 0, 1, 3, 4) * (1 - mask_triplet)
                samples_l2a = samples_l2a.permute(1, 2, 0, 3, 4)
                pred_2_latent = samples_l2a[:, 1, :, :, :].clone()
                
                decoded_2 = vae.decode(pred_2_latent / 0.18215).sample
                decoded_2_gray = decoded_2.mean(dim=1, keepdim=True)
                result_frames[2] = decoded_2_gray.repeat(1, 3, 1, 1)
                
                del z_l2a, samples_l2a
                torch.cuda.empty_cache()
                # 预测frame_6
                triplet_l2b = torch.stack([
                    pred_4_latent.detach(),
                    latent_val[:, 6],           # 占位
                    latent_val[:, 8]
                ], dim=1)
                
                z_l2b = torch.randn_like(triplet_l2b.permute(0, 2, 1, 3, 4))
                samples_l2b = val_diffusion.p_sample_loop(
                    raw_model.forward, z_l2b.shape, z_l2b,
                    clip_denoised=False, progress=False, device=device,
                    raw_x=triplet_l2b.permute(0, 2, 1, 3, 4),
                    mask=mask_triplet
                )
                
                samples_l2b = samples_l2b.permute(1, 0, 2, 3, 4) * mask_triplet + \
                              triplet_l2b.permute(2, 0, 1, 3, 4) * (1 - mask_triplet)
                samples_l2b = samples_l2b.permute(1, 2, 0, 3, 4)
                pred_6_latent = samples_l2b[:, 1, :, :, :].clone()
                
                decoded_6 = vae.decode(pred_6_latent / 0.18215).sample
                decoded_6_gray = decoded_6.mean(dim=1, keepdim=True)
                result_frames[6] = decoded_6_gray.repeat(1, 3, 1, 1)
                
                del z_l2b, samples_l2b
                torch.cuda.empty_cache()
                # ========== 第3层：预测1, 3, 5, 7 ==========
                # 预测frame_1
                triplet_l3a = torch.stack([
                    latent_val[:, 0],
                    latent_val[:, 1],           # 占位
                    pred_2_latent.detach()
                ], dim=1)
                
                z_l3a = torch.randn_like(triplet_l3a.permute(0, 2, 1, 3, 4))
                samples_l3a = val_diffusion.p_sample_loop(
                    raw_model.forward, z_l3a.shape, z_l3a,
                    clip_denoised=False, progress=False, device=device,
                    raw_x=triplet_l3a.permute(0, 2, 1, 3, 4),
                    mask=mask_triplet
                )
                
                samples_l3a = samples_l3a.permute(1, 0, 2, 3, 4) * mask_triplet + \
                              triplet_l3a.permute(2, 0, 1, 3, 4) * (1 - mask_triplet)
                samples_l3a = samples_l3a.permute(1, 2, 0, 3, 4)
                decoded_1 = vae.decode(samples_l3a[:, 1, :, :, :] / 0.18215).sample
                decoded_1_gray = decoded_1.mean(dim=1, keepdim=True)
                result_frames[1] = decoded_1_gray.repeat(1, 3, 1, 1)
                
                del z_l3a, samples_l3a
                torch.cuda.empty_cache()
                # 预测frame_3
                triplet_l3b = torch.stack([
                    pred_2_latent.detach(),
                    latent_val[:, 3],           # 占位
                    pred_4_latent.detach()
                ], dim=1)
                
                z_l3b = torch.randn_like(triplet_l3b.permute(0, 2, 1, 3, 4))
                samples_l3b = val_diffusion.p_sample_loop(
                    raw_model.forward, z_l3b.shape, z_l3b,
                    clip_denoised=False, progress=False, device=device,
                    raw_x=triplet_l3b.permute(0, 2, 1, 3, 4),
                    mask=mask_triplet
                )
                
                samples_l3b = samples_l3b.permute(1, 0, 2, 3, 4) * mask_triplet + \
                              triplet_l3b.permute(2, 0, 1, 3, 4) * (1 - mask_triplet)
                samples_l3b = samples_l3b.permute(1, 2, 0, 3, 4)
                decoded_3 = vae.decode(samples_l3b[:, 1, :, :, :] / 0.18215).sample
                decoded_3_gray = decoded_3.mean(dim=1, keepdim=True)
                result_frames[3] = decoded_3_gray.repeat(1, 3, 1, 1)
                
                del z_l3b, samples_l3b
                torch.cuda.empty_cache()
                # 预测frame_5
                triplet_l3c = torch.stack([
                    pred_4_latent.detach(),
                    latent_val[:, 5],           # 占位
                    pred_6_latent.detach()
                ], dim=1)
                
                z_l3c = torch.randn_like(triplet_l3c.permute(0, 2, 1, 3, 4))
                samples_l3c = val_diffusion.p_sample_loop(
                    raw_model.forward, z_l3c.shape, z_l3c,
                    clip_denoised=False, progress=False, device=device,
                    raw_x=triplet_l3c.permute(0, 2, 1, 3, 4),
                    mask=mask_triplet
                )
                
                samples_l3c = samples_l3c.permute(1, 0, 2, 3, 4) * mask_triplet + \
                              triplet_l3c.permute(2, 0, 1, 3, 4) * (1 - mask_triplet)
                samples_l3c = samples_l3c.permute(1, 2, 0, 3, 4)
                decoded_5 = vae.decode(samples_l3c[:, 1, :, :, :] / 0.18215).sample
                decoded_5_gray = decoded_5.mean(dim=1, keepdim=True)
                result_frames[5] = decoded_5_gray.repeat(1, 3, 1, 1)
                
                del z_l3c, samples_l3c
                torch.cuda.empty_cache()
                # 预测frame_7
                triplet_l3d = torch.stack([
                    pred_6_latent.detach(),
                    latent_val[:, 7],           # 占位
                    latent_val[:, 8]
                ], dim=1)
                
                z_l3d = torch.randn_like(triplet_l3d.permute(0, 2, 1, 3, 4))
                samples_l3d = val_diffusion.p_sample_loop(
                    raw_model.forward, z_l3d.shape, z_l3d,
                    clip_denoised=False, progress=False, device=device,
                    raw_x=triplet_l3d.permute(0, 2, 1, 3, 4),
                    mask=mask_triplet
                )
                
                samples_l3d = samples_l3d.permute(1, 0, 2, 3, 4) * mask_triplet + \
                              triplet_l3d.permute(2, 0, 1, 3, 4) * (1 - mask_triplet)
                samples_l3d = samples_l3d.permute(1, 2, 0, 3, 4)
                decoded_7 = vae.decode(samples_l3d[:, 1, :, :, :] / 0.18215).sample
                decoded_7_gray = decoded_7.mean(dim=1, keepdim=True)
                result_frames[7] = decoded_7_gray.repeat(1, 3, 1, 1)
                
                del z_l3d, samples_l3d, pred_2_latent, pred_4_latent, pred_6_latent
                torch.cuda.empty_cache()
                # 拼接9帧结果
                result_video = torch.stack(result_frames, dim=1)  # [B, 9, 3, H, W]
                num_frames_rec = 9

                # # 构建9帧的GT和mask可视化
                # video_val_rec = video_val[:, :9, :, :, :]
                # vessel_mask_rec = vessel_mask_vis[:, :9, :, :, :]
                
                # mask_rec_vis = torch.ones(b_val, 9, 3, h_val, w_val, device=device) * 0.5  # 灰色
                # mask_rec_vis[:, 0, :, :, :] = 0   # frame_0不遮罩（GT）
                # mask_rec_vis[:, 8, :, :, :] = 0   # frame_8不遮罩（GT）
            # else:  # stage == 3
            #     # 阶段3递归链：9帧完整可视化（简化版，直接预测）
            #     mask_rec_all = torch.ones(b_val, f_val, h_latent_val, w_latent_val, device=device)
            #     mask_rec_all[:, 0, :, :] = 0
            #     mask_rec_all[:, 8, :, :] = 0
                
            #     z_rec = torch.randn_like(latent_val.permute(0, 2, 1, 3, 4))
            #     samples_rec = val_diffusion.p_sample_loop(
            #         raw_model.forward, z_rec.shape, z_rec,
            #         clip_denoised=False, progress=False, device=device,
            #         raw_x=latent_val.permute(0, 2, 1, 3, 4),
            #         mask=mask_rec_all
            #     )
                
            #     samples_rec = samples_rec.permute(1, 0, 2, 3, 4) * mask_rec_all + latent_val.permute(2, 0, 1, 3, 4) * (1 - mask_rec_all)
            #     samples_rec = samples_rec.permute(1, 2, 0, 3, 4)
            #     samples_rec_flat = rearrange(samples_rec, 'b f c h w -> (b f) c h w') / 0.18215
            #     decoded_rec = vae.decode(samples_rec_flat).sample
            #     decoded_rec = rearrange(decoded_rec, '(b f) c h w -> b f c h w', b=b_val)
            #     decoded_rec_gray = decoded_rec.mean(dim=2, keepdim=True)
            #     result_video = decoded_rec_gray.repeat(1, 1, 3, 1, 1)
                
            #     num_frames_rec = 9
            #     del z_rec, samples_rec
            
            # 构建递归链的mask可视化
            video_val_rec = video_val[:, :num_frames_rec, :, :, :]
            vessel_mask_rec = vessel_mask_vis[:, :num_frames_rec, :, :, :]
            
            mask_rec_vis = torch.ones(b_val, num_frames_rec, 3, h_val, w_val, device=device) * 0.5  # 灰色
            mask_rec_vis[:, 0, :, :, :] = 0   # 首帧不遮罩
            mask_rec_vis[:, -1, :, :, :] = 0  # 尾帧不遮罩
            
            recursive_images = []
            for sample_idx, sample_id in enumerate(sample_ids):
                result_single = result_video[sample_idx:sample_idx+1]
                video_val_rec_single = video_val_rec[sample_idx:sample_idx+1]
                mask_rec_vis_single = mask_rec_vis[sample_idx:sample_idx+1]
                vessel_mask_rec_single = vessel_mask_rec[sample_idx:sample_idx+1]
                
                val_pic_rec = torch.cat([
                    video_val_rec_single,
                    video_val_rec_single * (1 - mask_rec_vis_single),
                    result_single,
                    vessel_mask_rec_single
                ], dim=1)
                
                val_pic_rec_flat = rearrange(val_pic_rec, 'b f c h w -> (b f) c h w')
                recursive_image_path = os.path.join(
                    val_fold,
                    f"Epoch_{epoch+1}_{stage_name}_recursive_full_idx_{sample_id:04d}.png",
                )
                save_image(
                    val_pic_rec_flat,
                    recursive_image_path,
                    nrow=num_frames_rec,
                    normalize=True,
                    value_range=(-1, 1)
                )
                recursive_images.append({
                    "path": recursive_image_path,
                    "caption": f"Epoch {epoch+1} {stage_name} recursive {num_frames_rec} frames idx {sample_id}",
                })
            
            generated_images["Val Examples/Recursive"] = recursive_images
    
    torch.cuda.empty_cache()
    return generated_images

# ========== 三元组loss计算函数 ==========
def resolve_triplet_loss_pair(
    diffusion,
    pred_middle_raw,
    x_t_middle,
    x_start_middle,
    noise_middle,
    t,
    loss_target_mode="auto",
):
    """Resolve prediction/target tensors for the requested triplet loss mode."""
    if loss_target_mode == "auto":
        if diffusion.model_mean_type == ModelMeanType.EPSILON:
            return pred_middle_raw, noise_middle
        if diffusion.model_mean_type == ModelMeanType.START_X:
            return pred_middle_raw, x_start_middle
        if diffusion.model_mean_type == ModelMeanType.PREVIOUS_X:
            target_prev, _, _ = diffusion.q_posterior_mean_variance(
                x_start=x_start_middle,
                x_t=x_t_middle,
                t=t,
            )
            return pred_middle_raw, target_prev
        raise NotImplementedError(f"Unsupported model_mean_type: {diffusion.model_mean_type}")

    if loss_target_mode == "epsilon":
        if diffusion.model_mean_type != ModelMeanType.EPSILON:
            raise ValueError(
                "loss_target_mode='epsilon' requires diffusion.model_mean_type == ModelMeanType.EPSILON"
            )
        return pred_middle_raw, noise_middle

    if loss_target_mode == "x0":
        if diffusion.model_mean_type == ModelMeanType.EPSILON:
            pred_middle = diffusion._predict_xstart_from_eps(
                x_t=x_t_middle,
                t=t,
                eps=pred_middle_raw,
            )
        elif diffusion.model_mean_type == ModelMeanType.START_X:
            pred_middle = pred_middle_raw
        else:
            raise NotImplementedError(
                f"loss_target_mode='x0' is not implemented for {diffusion.model_mean_type}"
            )
        return pred_middle, x_start_middle

    raise ValueError(
        f"Unsupported loss_target_mode: {loss_target_mode}. Expected one of: auto, epsilon, x0"
    )


def compute_triplet_loss(
    model,
    diffusion,
    latent_triplet,
    t,
    weight_batch=None,
    device=None,
    loss_target_mode="auto",
):
    """
    计算三元组loss（首尾可见，预测中间）。

    loss_target_mode:
        - auto: follow diffusion.model_mean_type
        - epsilon: compare predicted noise against noise target
        - x0: compare predicted x0 against clean latent target
    """
    b, f, c, h, w = latent_triplet.shape
    assert f == 3, f"必须是三元组，当前{f}帧"
    
    # 构建mask：首尾可见，中间预测
    mask = torch.ones(b, f, h, w, device=device)
    mask[:, 0, :, :] = 0
    mask[:, 2, :, :] = 0
    
    # 使用diffusion的前向过程添加噪声
    latent_permuted = latent_triplet.permute(0, 2, 1, 3, 4)  # [B, C, 3, H, W]
    noise = torch.randn_like(latent_permuted)
    x_t_bcfhw = diffusion.q_sample(latent_permuted, t, noise=noise)  # [B, C, 3, H, W]
    
    # 恢复到 [B, 3, C, H, W] 供模型输入
    x_t = x_t_bcfhw.permute(0, 2, 1, 3, 4)  # [B, 3, C, H, W]
    
    # 构建输入：首尾用GT，中间用噪声
    # mask: [B, 3, H, W]，0表示可见（GT），1表示预测（噪声）
    model_input = latent_triplet * (1 - mask.unsqueeze(2)) + x_t * mask.unsqueeze(2)
    
    # 模型forward
    model_output = model(model_input, t)  # [B, 3, C_out, H, W]
    vb_loss = None
    
    if diffusion.model_var_type in {ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE}:
        if model_output.shape[2] != c * 2:
            raise ValueError(
                f"Expected model output channels {c * 2} for learned sigma, got {model_output.shape[2]}"
            )
        model_output, model_var_values = torch.split(model_output, c, dim=2)
        frozen_out = torch.cat([model_output.detach(), model_var_values], dim=2).permute(0, 2, 1, 3, 4)
        vb_loss = diffusion._vb_terms_bpd(
            model=lambda *args, r=frozen_out: r,
            x_start=latent_permuted,
            x_t=x_t_bcfhw,
            t=t,
            clip_denoised=False,
        )["output"].mean()
        if diffusion.loss_type == LossType.RESCALED_MSE:
            vb_loss = vb_loss * diffusion.num_timesteps / 1000.0
    elif model_output.shape[2] != c:
        raise ValueError(
            f"Expected model output channels {c} for fixed sigma, got {model_output.shape[2]}"
        )
    
    pred_middle_raw = model_output[:, 1, :, :, :]  # [B, C, H, W]
    x_t_middle = x_t[:, 1, :, :, :]  # [B, C, H, W]
    noise_middle = noise.permute(0, 2, 1, 3, 4)[:, 1, :, :, :]  # [B, C, H, W]
    x_start_middle = latent_triplet[:, 1, :, :, :]  # [B, C, H, W]

    pred_middle, target_middle = resolve_triplet_loss_pair(
        diffusion=diffusion,
        pred_middle_raw=pred_middle_raw,
        x_t_middle=x_t_middle,
        x_start_middle=x_start_middle,
        noise_middle=noise_middle,
        t=t,
        loss_target_mode=loss_target_mode,
    )
    
    # 计算MSE loss
    loss_middle = (pred_middle - target_middle) ** 2  # [B, C, H, W]
    
    # 血管mask加权
    if weight_batch is not None:
        weight_batch = align_weight_batch(weight_batch, b, pred_middle.shape[1], h, w)
        weight_mean = weight_batch.mean()
        loss_middle = loss_middle * weight_batch * (1.0 / weight_mean)

    total_loss = loss_middle.mean()
    if vb_loss is not None:
        total_loss = total_loss + vb_loss

    return total_loss

# ========== 快速预测中间帧（无梯度） ==========
@torch.no_grad()
def fast_predict_middle_frame(model, val_diffusion, triplet_latent, mask, device):
    """
    快速预测中间帧（5步DDIM，节省显存）
    Args:
        triplet_latent: [B, 3, C, H, W]
        mask: [B, 3, H, W]
    Returns:
        predicted_middle: [B, C, H, W]
    """
    z = torch.randn_like(triplet_latent.permute(0, 2, 1, 3, 4))
    samples = val_diffusion.p_sample_loop(
        model.forward, z.shape, z,
        clip_denoised=False, progress=False, device=device,
        raw_x=triplet_latent.permute(0, 2, 1, 3, 4),
        mask=mask
    )
    samples = samples.permute(1, 0, 2, 3, 4)
    predicted_middle = samples[:, 1, :, :, :].clone()
    
    # 立即释放
    del samples, z
    torch.cuda.empty_cache()
    
    return predicted_middle

def backward_loss(loss, scaler=None, mixed_precision=True):
    """Backward a scaled or unscaled loss tensor."""
    if mixed_precision:
        scaler.scale(loss).backward()
    else:
        loss.backward()


def build_stage2_recursive_triplets(latent_dense):
    """Build the extra recursive triplets used in stage2, excluding the base triplet."""
    b, f, c, h, w = latent_dense.shape
    assert f == 5, f"阶段2需要5帧，当前{f}帧"
    return [
        torch.stack([latent_dense[:, 0], latent_dense[:, 1], latent_dense[:, 2]], dim=1),
        torch.stack([latent_dense[:, 2], latent_dense[:, 3], latent_dense[:, 4]], dim=1),
    ]


def build_stage3_recursive_triplets(latent_dense):
    """Build the extra recursive triplets used in stage3, excluding the base triplet."""
    b, f, c, h, w = latent_dense.shape
    assert f == 9, f"阶段3需要9帧，当前{f}帧"
    triplets = [
        torch.stack([latent_dense[:, 0], latent_dense[:, 2], latent_dense[:, 4]], dim=1),
        torch.stack([latent_dense[:, 4], latent_dense[:, 6], latent_dense[:, 8]], dim=1),
    ]
    for left, mid, right in [(0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, 8)]:
        triplets.append(torch.stack([latent_dense[:, left], latent_dense[:, mid], latent_dense[:, right]], dim=1))
    return triplets


def backward_recursive_triplets(
    model,
    diffusion,
    triplets,
    t,
    device,
    recursive_weight,
    loss_scale_divisor,
    scaler=None,
    mixed_precision=True,
    loss_target_mode="auto",
    sync_last_backward=False,
):
    """
    Backward each recursive triplet immediately to reduce peak memory.
    Returns the mean recursive loss value for logging.
    """
    if not triplets:
        return 0.0

    loss_sum = 0.0
    scaled_weight = recursive_weight / (len(triplets) * loss_scale_divisor)

    for idx, triplet in enumerate(triplets):
        should_sync = sync_last_backward and idx == len(triplets) - 1
        with ddp_sync_context(model, should_sync=should_sync):
            triplet_loss = compute_triplet_loss(
                model,
                diffusion,
                triplet,
                t,
                None,
                device,
                loss_target_mode=loss_target_mode,
            )
            loss_sum += triplet_loss.detach().item()
            backward_loss(
                scaled_weight * triplet_loss,
                scaler=scaler,
                mixed_precision=mixed_precision,
            )
            del triplet_loss

    torch.cuda.empty_cache()
    return loss_sum / len(triplets)

# -------------------------- 主训练函数（修改loss计算部分） --------------------------
def main(args):
    # 1. 基础校验与分布式初始化
    assert torch.cuda.is_available(), "Training requires at least one GPU."
    ddp_timeout_minutes = int(getattr(args, "ddp_timeout_minutes", 180))
    rank, local_rank, world_size = setup_distributed(
        backend="nccl",
        timeout_minutes=ddp_timeout_minutes,
    )  # 初始化分布式环境
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)  # 绑定当前进程到local_rank对应的GPU
    cuda_empty_cache_interval = int(getattr(args, "cuda_empty_cache_interval", 100))

    # 2. 显存优化配置（加载阶段开启expandable_segments，训练时关闭）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False  # 关闭benchmark减少显存预分配
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

    # 3. 随机种子
    seed = args.global_seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 4. 实验目录（仅rank0创建）
    model_string_name = args.model.replace("/", "-")
    experiment_dir = f"{args.results_dir}/{model_string_name}_{args.cur_date}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    val_fold = f"{experiment_dir}/val_pic"
    log_level = str(getattr(args, "log_level", "INFO")).upper()
    valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if log_level not in valid_log_levels:
        raise ValueError(f"log_level must be one of {sorted(valid_log_levels)}, got {log_level}")
    args.log_level = log_level
    tracking_backend = str(getattr(args, "tracking_backend", "tensorboard")).lower()
    valid_tracking_backends = {"tensorboard", "wandb"}
    if tracking_backend not in valid_tracking_backends:
        raise ValueError(
            f"tracking_backend must be one of {sorted(valid_tracking_backends)}, got {tracking_backend}"
        )
    args.tracking_backend = tracking_backend
    wandb_project = getattr(args, "wandb_project", "TSSC-CTA-CL")
    wandb_entity = getattr(args, "wandb_entity", None)
    wandb_mode = str(getattr(args, "wandb_mode", "online")).lower()
    wandb_run_name = getattr(args, "wandb_run_name", None) or os.path.basename(experiment_dir)
    val_visualization_count = int(getattr(args, "val_visualization_count", 20))
    val_visualization_batch_size = int(getattr(args, "val_visualization_batch_size", 4))
    triplet_eval_batch_size = int(getattr(args, "triplet_eval_batch_size", 4))
    timestep_respacing_val = getattr(args, "timestep_respacing_val", getattr(args, "timestep_respacing_test", "ddim50"))
    shared_console = Console(stderr=True) if rank == 0 else None
    
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(val_fold, exist_ok=True)
        logger = create_logger_compat(experiment_dir, level=log_level, console=shared_console)
        tracker = create_experiment_tracker(
            backend=tracking_backend,
            logging_dir=os.path.join(experiment_dir, "runs") if tracking_backend == "tensorboard" else experiment_dir,
            run_name=wandb_run_name,
            config=OmegaConf.to_container(args, resolve=True),
            project=wandb_project,
            entity=wandb_entity,
            mode=wandb_mode,
        )
        OmegaConf.save(args, os.path.join(experiment_dir, 'config_cta.yaml'))
        logger.info(f"Experiment dir: {experiment_dir}")
        logger.info(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
        logger.info(f"DDP timeout: {ddp_timeout_minutes} minutes")
        logger.info(f"CUDA cache cleanup interval: {cuda_empty_cache_interval} optimizer steps")
        logger.info(f"Tracking backend: {tracking_backend}")
        logger.info(f"Sliding triplet eval batch size: {triplet_eval_batch_size}")
        logger.info(f"Validation sampler respacing: {timestep_respacing_val}")
    else:
        logger = create_logger_compat(None, level=log_level, console=shared_console)
        tracker = None

    # ========== 读取课程学习配置 ==========
    cl_config = None
    if hasattr(args, 'curriculum_learning') and args.curriculum_learning is not None:
        cl_config = args.curriculum_learning
        stage0_steps = cl_config.get('stage0_steps', 5)  # 新增
        stage1_steps = cl_config.get('stage1_steps', 15)
        stage2_steps = cl_config.get('stage2_steps', 25)
        recursive_weight = cl_config.get('recursive_loss_weight', 0.2)
        recursive_sampling_steps = cl_config.get('recursive_sampling_steps', 20)
    else:
        if rank == 0:
            logger.warning("⚠️  未找到课程学习配置，使用默认值")
        stage0_steps = 5
        stage1_steps = 15
        stage2_steps = 25
        recursive_weight = 0.2
        recursive_sampling_steps = 20

    # 验证tar_num_frames
    if args.tar_num_frames != 3:
        if rank == 0:
            logger.warning(f"⚠️  tar_num_frames应为3，当前为{args.tar_num_frames}，自动调整")
        args.tar_num_frames = 3

    # 血管mask配置
    if hasattr(args, 'vessel_mask') and args.vessel_mask is not None:
        vessel_max_weight = args.vessel_mask.get('max_weight', 10.0)
        vessel_mask_enable = args.vessel_mask.get('enable', True)
    else:
        vessel_max_weight = 10.0
        vessel_mask_enable = True

    triplet_loss_mode = getattr(args, 'triplet_loss_mode', 'auto')
    valid_triplet_loss_modes = {'auto', 'epsilon', 'x0'}
    if triplet_loss_mode not in valid_triplet_loss_modes:
        raise ValueError(
            f"triplet_loss_mode must be one of {sorted(valid_triplet_loss_modes)}, got {triplet_loss_mode}"
        )
    args.triplet_loss_mode = triplet_loss_mode
    args.learn_sigma = bool(getattr(args, "learn_sigma", True))

    # 打印配置
    if rank == 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"📚 四阶段课程学习配置:")
        logger.info(f"  阶段0: 0→{stage0_steps} steps, Gap=1, 2帧, 伪GT+随机α")
        logger.info(f"  阶段1: {stage0_steps}→{stage1_steps} steps, Gap=2, 3帧")
        logger.info(f"  阶段2: {stage1_steps}→{stage2_steps} steps, Gap=4, 5帧")
        logger.info(f"  阶段3: {stage2_steps}+ steps, Gap=8, 9帧")
        logger.info(f"  递归链权重: {recursive_weight}")
        logger.info(f"  血管mask: {'启用' if vessel_mask_enable else '禁用'}")
        logger.info(f"  三元组loss模式: {triplet_loss_mode}")
        logger.info("  验证口径: 滑窗三连帧代理评测（首尾帧预测中间帧）")
        logger.info(f"{'='*60}\n")

    # 存储到args
    args.stage0_steps = stage0_steps
    args.stage1_steps = stage1_steps
    args.stage2_steps = stage2_steps
    args.recursive_weight = recursive_weight
    args.recursive_sampling_steps = recursive_sampling_steps

    # 读取验证权重配置
    if hasattr(cl_config, 'val_weights'):
        args.val_weights_config = cl_config['val_weights']
    else:
        # 默认验证权重
        args.val_weights_config = {
            'stage1': {'interval_1': 1.0},
            'stage2': {'interval_1': 0.3, 'interval_2': 0.7},
            'stage3': {'interval_1': 0.1, 'interval_2': 0.3, 'interval_4': 0.6}
        }
    # 5. 模型初始化（先CPU创建，再移到GPU）
    assert args.image_size % 8 == 0, "Image size must be divisible by 8."
    latent_size = args.image_size // 8  # 256//8=32
    additional_kwargs = {'num_frames': args.tar_num_frames, 'mode': 'video'}
    
    # 先在CPU创建模型，减少GPU显存峰值
    model = MVIF_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=args.learn_sigma,
        **additional_kwargs
    )
    # 移到GPU（非阻塞）
    model = model.to(device, non_blocking=True)

    # ========== EMA模型初始化修复 ==========
    # EMA模型直接创建在GPU（避免后续设备迁移错误）
    ema = deepcopy(model).to(device, non_blocking=True)
    requires_grad(ema, False)
    ema.eval()  # 强制设置为eval模式

    # VAE模型（冻结）
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_model_path, 
        subfolder="sd-vae-ft-mse"
    ).to(device, non_blocking=True)
    vae.requires_grad_(False)
    vae.eval()

    # 6. 加载预训练权重（仅rank0打印日志）
    if args.pretrained and os.path.exists(args.pretrained):
        if rank == 0:
            logger.info(f"Loading pretrained model from {args.pretrained}")
        try:
            # 优化：预训练权重也先加载到CPU
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            # 优先加载EMA权重，若无则加载model权重
            model_weights = checkpoint.get("ema", checkpoint.get("model", {}))
            
            # 初始化pretrained_dict，避免未定义
            pretrained_dict = {}
            if model_weights:
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in model_weights.items() if k in model_dict}
                
                # 逐个参数移到GPU，避免一次性加载
                for k in pretrained_dict:
                    pretrained_dict[k] = pretrained_dict[k].to(device, non_blocking=True)
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                
                # 同步更新EMA模型（关键：加载预训练权重后更新EMA）
                update_ema(ema, get_raw_model(model), decay=0.0)  # 完全复制
            
            # 计算加载比例（确保load_ratio始终有值）
            load_ratio = len(pretrained_dict) / len(model.state_dict()) * 100 if pretrained_dict else 0.0
            
            # 清理临时数据
            del checkpoint, model_weights
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # 强制回收未使用显存
            
            if rank == 0:
                logger.info(f"Loaded {load_ratio:.1f}% pretrained weights")
        except Exception as e:
            if rank == 0:
                logger.error(f"Failed to load pretrained weights: {e}")
            load_ratio = 0.0
    else:
        # 未指定预训练权重时，加载比例为0
        load_ratio = 0.0

    # 7. 模型优化配置（延迟编译到续训后）
    if args.gradient_checkpointing:
        if rank == 0:
            logger.info("Enabling gradient checkpointing")
        def enable_ckpt(module):
            if hasattr(module, 'enable_gradient_checkpointing'):
                module.enable_gradient_checkpointing()
            elif hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
        model.apply(enable_ckpt)

    if args.enable_xformers_memory_efficient_attention:
        if rank == 0:
            logger.info("Using Xformers memory-efficient attention")
        model.enable_xformers_memory_efficient_attention()

    # 8. DDP包装（多卡时）
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    with torch.no_grad():
        update_ema(ema, get_raw_model(model), decay=0.0)
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
        logger.info(f"DDP memory opts | gradient_as_bucket_view={'on' if world_size > 1 else 'off'} | adamw_foreach=off")

    # 9. 优化器与学习率调度器
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
    )

    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps
    )

    # 10. 数据加载器
    # 训练集
    dataset_train = data_loader(args, stage='train')
    sampler_train = DistributedSampler(
        dataset_train,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader_train = DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=False,
        sampler=sampler_train,
        # num_workers=args.num_workers,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        # prefetch_factor=2  # 预取优化
    )
    train_batches_per_epoch = len(loader_train)
    train_updates_per_epoch = max(1, math.ceil(train_batches_per_epoch / args.gradient_accumulation_steps))

    # 验证集
    dataset_val_vis = data_loader(args, stage='val')
    dataset_val_metrics = full_sequence_data_loader(args, stage='val')
    val_visualization_indices = get_fixed_visualization_indices(len(dataset_val_vis), val_visualization_count)
    sampler_val_metrics = DistributedEvalSampler(
        dataset_val_metrics,
        num_replicas=world_size,
        rank=rank,
    )
    loader_val_metrics = DataLoader(
        dataset_val_metrics,
        batch_size=args.val_batch_size,
        shuffle=False,
        sampler=sampler_val_metrics,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_full_sequence_batch,
    )
    loader_val_vis = None
    if rank == 0:
        loader_val_vis = DataLoader(
            dataset_val_vis,
            batch_size=args.val_batch_size,
            shuffle=False,
            # num_workers=args.num_workers,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            # prefetch_factor=2
        )

    if rank == 0:
        logger.info(f"Train dataset size: {len(dataset_train)}")
        logger.info(f"Val visualization dataset size: {len(dataset_val_vis)}")
        logger.info(f"Val sliding-triplet dataset size: {len(dataset_val_metrics)}")
        logger.info(f"Train batches per epoch: {train_batches_per_epoch}")
        logger.info(f"Train optimizer steps per epoch: {train_updates_per_epoch}")
        logger.info(f"Val video shards per rank: {len(loader_val_metrics)}")
        logger.info(
            f"Val visualization samples: {len(val_visualization_indices)} "
            f"(batch size {val_visualization_batch_size})"
        )

    # ========== 断点续训核心优化（终极版） ==========
    # 强制清理所有冗余显存
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    train_steps = 0
    first_epoch = 0
    resume_step = 0
    # 初始化最优指标（断点续训时加载历史最优值）
    best_val_metric = float('inf')  # 越小越好

    if args.resume_from_checkpoint:
        latest_ckpt = os.path.join(checkpoint_dir, "latest_epoch_train_model.pth")
        if os.path.exists(latest_ckpt):
            if rank == 0:
                logger.info(f"Resuming from checkpoint: {latest_ckpt}")
                # 打印当前显存状态
                logger.info(f"Memory before load: {torch.cuda.memory_allocated(device)/1024**3:.2f} GiB")
            
            # 1. 先加载checkpoint到CPU，避免GPU瞬间峰值
            checkpoint = safe_load_checkpoint(latest_ckpt, device)
            
            # 2. 恢复权重
            get_raw_model(model).load_state_dict(checkpoint["model"])
            if "ema" in checkpoint and checkpoint["ema"] is not None:
                ema.load_state_dict(checkpoint["ema"])
                ema.eval()  # 重新设置为eval模式

            # 3. 恢复优化器/调度器/AMP状态
            restored_opt_state = False
            if checkpoint.get("opt") is not None:
                opt.load_state_dict(checkpoint["opt"])
                move_optimizer_state_to_device(opt, device)
                restored_opt_state = True
            if checkpoint.get("lr_scheduler") is not None:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            if scaler is not None and checkpoint.get("scaler") is not None:
                scaler.load_state_dict(checkpoint["scaler"])
            
            # 4. 读取元数据（核心修复：先赋值train_steps）
            train_steps = checkpoint.get("train_steps", 0)  # 先拿到checkpoint里的训练步数
            best_val_metric = checkpoint.get("best_val_metric", float('inf'))

            saved_epoch = checkpoint.get("epoch", None)
            if saved_epoch is not None:
                first_epoch = int(saved_epoch)
                resume_step = 0
            else:
                # 兼容旧checkpoint：按每epoch的优化器步数近似反推
                grad_accum = args.gradient_accumulation_steps
                batches_per_epoch = len(loader_train)
                updates_per_epoch = max(1, math.ceil(batches_per_epoch / grad_accum))
                first_epoch = train_steps // updates_per_epoch
                resume_updates = train_steps % updates_per_epoch
                resume_step = min(resume_updates * grad_accum, batches_per_epoch)
            
            # 加载历史最优指标
            if best_val_metric != float('inf') and rank == 0:
                logger.info(f"Resumed best validation metric: {best_val_metric:.4f}")
            
            # 5. 强制清理所有临时数据
            del checkpoint
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            # 打印正确的续训日志
            if rank == 0:
                logger.info(f"Resumed actual epoch: {first_epoch}, batch step: {resume_step}")
                logger.info(f"Resumed optimizer steps: {train_steps}")
                logger.info(f"Memory after load: {torch.cuda.memory_allocated(device)/1024**3:.2f} GiB")
                if restored_opt_state:
                    logger.info("Optimizer/scheduler/scaler state restored")
                else:
                    logger.warning("Checkpoint missing optimizer state; resumed with fresh optimizer/scheduler")
        else:
            if rank == 0:
                logger.warning(f"Checkpoint {latest_ckpt} not found, start from scratch")

    # 续训完成后：关闭expandable_segments + 模型编译
    # 1. 关闭expandable_segments（训练阶段用）
    torch.backends.cuda.expandable_segments = False
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'
    # 2. 延迟编译模型（避免加载阶段占用显存）
    if args.use_compile and torch.cuda.is_available():
        if rank == 0:
            logger.info("Compiling model with torch.compile (delayed)")
        model = torch.compile(model, mode="reduce-overhead")
    
    # ========== 创建快速采样器（用于递归链） ==========
    val_diffusion = create_diffusion(
        timestep_respacing=f"ddim{args.recursive_sampling_steps}",
        diffusion_steps=args.diffusion_steps,
        learn_sigma=args.learn_sigma,
    )
    if rank == 0:
        logger.info(f"递归链采样器: DDIM{args.recursive_sampling_steps}步")
    # 3. 最后一次显存清理
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # 12. 训练循环
    num_update_steps_per_epoch = train_updates_per_epoch
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    model.train()
    ema.eval()  # EMA始终保持eval模式
    running_loss = 0.0
    log_steps = 0
    start_time = time()

    if rank == 0:
        logger.info(f"Start training for {num_train_epochs} epochs (first epoch: {first_epoch})")
        logger.info(f"Vessel mask max weight: {getattr(args, 'vessel_max_weight', 10.0)}")

    train_progress = None
    overall_task = None
    epoch_task = None
    if rank == 0:
        initial_epoch_completed = resume_step if args.resume_from_checkpoint and first_epoch < num_train_epochs else 0
        train_progress = create_rich_progress(shared_console)
        train_progress.start()
        overall_task = train_progress.add_task(
            "Overall Opt Step",
            total=args.max_train_steps,
            completed=train_steps,
            status=format_overall_progress_status(first_epoch, num_train_epochs, train_steps, args),
        )
        epoch_task = train_progress.add_task(
            f"Epoch {first_epoch + 1}/{num_train_epochs} Batch",
            total=len(loader_train),
            completed=initial_epoch_completed,
            status=format_epoch_progress_status(
                batch_step=initial_epoch_completed,
                total_batches=len(loader_train),
                train_steps=train_steps,
                max_train_steps=args.max_train_steps,
            ),
        )

    for epoch in range(first_epoch, num_train_epochs):
        sampler_train.set_epoch(epoch)  # 分布式采样器epoch同步
        
        # ========== 核心修改：每个epoch更新dataset的训练步数 ==========
        dataset_train.set_training_step(train_steps)
        if hasattr(dataset_val_vis, 'set_training_step'):
            dataset_val_vis.set_training_step(train_steps)
        
        diffusion = create_diffusion(
            timestep_respacing=args.timestep_respacing_train,
            diffusion_steps=args.diffusion_steps,
            learn_sigma=args.learn_sigma,
        )
        diffusion.training = True

        if rank == 0 and train_progress is not None:
            epoch_completed = resume_step if args.resume_from_checkpoint and epoch == first_epoch else 0
            train_progress.update(
                overall_task,
                completed=train_steps,
                status=format_overall_progress_status(epoch, num_train_epochs, train_steps, args),
            )
            train_progress.update(
                epoch_task,
                description=f"Epoch {epoch + 1}/{num_train_epochs} Batch",
                total=len(loader_train),
                completed=epoch_completed,
                status=format_epoch_progress_status(
                    batch_step=epoch_completed,
                    total_batches=len(loader_train),
                    train_steps=train_steps,
                    max_train_steps=args.max_train_steps,
                ),
            )

        for step, batch in enumerate(loader_train):
            # 跳过断点续训的步骤
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue

            # ========== 数据加载 ==========
            video = batch['video'].to(device, non_blocking=True)
            stage = batch['stage'][0].item()  # ✅ 从batch获取stage
            frame_gap = batch['frame_gap'][0].item()  # ✅ 从batch获取frame_gap
            b, f, c, h, w = video.shape
            video_flat = None
            latent_sparse = None
            loss = None
            base_loss = None

            # VAE编码
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.mixed_precision):
                video_flat = rearrange(video, 'b f c h w -> (b f) c h w')
                latent = vae.encode(video_flat).latent_dist.sample().mul_(0.18215)
                latent = rearrange(latent, '(b f) c h w -> b f c h w', b=b)

            b, f, c_latent, h_latent, w_latent = latent.shape
            t = torch.randint(0, diffusion.num_timesteps, (b,), device=device)

            # ========== 生成血管mask ==========
            weight_batch = None
            if vessel_mask_enable:
                video_np = video.detach().cpu().numpy()
                video_gray_np = np.mean(video_np, axis=2)
                weight_list = []
                for i in range(b):
                    frames_gray = [video_gray_np[i, t_idx] for t_idx in range(f)]
                    _, soft_weight_np = generate_vessel_mask_adaptive(frames_gray, max_weight=vessel_max_weight)
                    weight_latent = prepare_mask_for_latent(soft_weight_np, h_latent, device)
                    weight_list.append(weight_latent)
                weight_batch = torch.cat(weight_list, dim=0)

            should_step = (
                ((step + 1) % args.gradient_accumulation_steps == 0)
                or ((step + 1) == len(loader_train))
            )
            accum_divisor = args.gradient_accumulation_steps
            if should_step and (step + 1) % args.gradient_accumulation_steps != 0:
                accum_divisor = (step % args.gradient_accumulation_steps) + 1

            # ========== 计算Loss（三元组 + 递归链） ==========
            recursive_loss_value = 0.0
            loss_display_value = None
            grad_norm_value = None
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                if stage == 0:
                    with ddp_sync_context(model, should_sync=should_step):
                        base_loss = compute_triplet_loss(
                            model, diffusion, latent, t, 
                            weight_batch, device,
                            loss_target_mode=args.triplet_loss_mode,
                        )
                        loss_display_value = base_loss.detach().item()
                        loss = base_loss / accum_divisor
                        backward_loss(loss, scaler=scaler, mixed_precision=args.mixed_precision)
                
                elif stage == 1:
                    with ddp_sync_context(model, should_sync=should_step):
                        base_loss = compute_triplet_loss(
                            model, diffusion, latent, t, 
                            weight_batch, device,
                            loss_target_mode=args.triplet_loss_mode,
                        )
                        loss_display_value = base_loss.detach().item()
                        loss = base_loss / accum_divisor
                        backward_loss(loss, scaler=scaler, mixed_precision=args.mixed_precision)
                
                elif stage == 2:
                    latent_sparse = torch.stack([latent[:, 0], latent[:, 2], latent[:, 4]], dim=1)
                    with ddp_sync_context(model, should_sync=False):
                        base_loss = compute_triplet_loss(
                            model, diffusion, latent_sparse, t, 
                            weight_batch, device,
                            loss_target_mode=args.triplet_loss_mode,
                        )
                        base_loss_value = base_loss.detach().item()
                        backward_loss(
                            base_loss / accum_divisor,
                            scaler=scaler,
                            mixed_precision=args.mixed_precision,
                        )
                    recursive_triplets = build_stage2_recursive_triplets(latent)
                    recursive_loss_value = backward_recursive_triplets(
                        model,
                        diffusion,
                        recursive_triplets,
                        t,
                        device,
                        recursive_weight=args.recursive_weight,
                        loss_scale_divisor=accum_divisor,
                        scaler=scaler,
                        mixed_precision=args.mixed_precision,
                        loss_target_mode=args.triplet_loss_mode,
                        sync_last_backward=should_step,
                    )
                    total_loss_value = base_loss_value + args.recursive_weight * recursive_loss_value
                    loss_display_value = total_loss_value
                    loss = torch.tensor(total_loss_value / accum_divisor, device=device)
                    del recursive_triplets
                
                else:  # stage == 3
                    latent_sparse = torch.stack([latent[:, 0], latent[:, 4], latent[:, 8]], dim=1)
                    with ddp_sync_context(model, should_sync=False):
                        base_loss = compute_triplet_loss(
                            model, diffusion, latent_sparse, t, 
                            weight_batch, device,
                            loss_target_mode=args.triplet_loss_mode,
                        )
                        base_loss_value = base_loss.detach().item()
                        backward_loss(
                            base_loss / accum_divisor,
                            scaler=scaler,
                            mixed_precision=args.mixed_precision,
                        )
                    recursive_triplets = build_stage3_recursive_triplets(latent)
                    recursive_loss_value = backward_recursive_triplets(
                        model,
                        diffusion,
                        recursive_triplets,
                        t,
                        device,
                        recursive_weight=args.recursive_weight,
                        loss_scale_divisor=accum_divisor,
                        scaler=scaler,
                        mixed_precision=args.mixed_precision,
                        loss_target_mode=args.triplet_loss_mode,
                        sync_last_backward=should_step,
                    )
                    total_loss_value = base_loss_value + args.recursive_weight * recursive_loss_value
                    loss_display_value = total_loss_value
                    loss = torch.tensor(total_loss_value / accum_divisor, device=device)
                    del recursive_triplets

            # 梯度裁剪（累计到指定步数）
            if should_step:
                if args.mixed_precision:
                    scaler.unscale_(opt)
                grad_norm = clip_grad_norm_(
                    get_raw_model(model).parameters(),
                    args.clip_max_norm,
                    clip_grad=(train_steps >= args.start_clip_iter)
                )
                grad_norm_value = float(grad_norm)

                # 优化器步进
                if args.mixed_precision:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                lr_scheduler.step()
                set_optimizer_zeros_grad(opt)
                
                # ✅ 调试：记录train_steps的变化
                prev_train_steps = train_steps
                train_steps += 1
                
                # ✅ 调试日志1：每次都打印（只在前100步）
                if train_steps <= 100 and rank == 0:
                    logger.debug(
                        f"Step {train_steps}: prev={prev_train_steps}, "
                        f"stage0_steps={args.stage0_steps}, "
                        f"stage1_steps={args.stage1_steps}"
                    )
                
                # 检查是否跨越了阶段边界
                stage_changed = False
                
                # ✅ 调试日志2：检查每个条件
                if prev_train_steps < args.stage0_steps <= train_steps:
                    stage_changed = True
                    if rank == 0:
                        logger.debug(f"触发条件1: {prev_train_steps} < {args.stage0_steps} <= {train_steps}")
                        logger.info(f"🎯 切换到 Stage 1 (Gap=2) at step {train_steps}")
                
                elif prev_train_steps < args.stage1_steps <= train_steps:
                    stage_changed = True
                    if rank == 0:
                        logger.debug(f"触发条件2: {prev_train_steps} < {args.stage1_steps} <= {train_steps}")
                        logger.info(f"🎯 切换到 Stage 2 (Gap=4) at step {train_steps}")
                
                elif prev_train_steps < args.stage2_steps <= train_steps:
                    stage_changed = True
                    if rank == 0:
                        logger.debug(f"触发条件3: {prev_train_steps} < {args.stage2_steps} <= {train_steps}")
                        logger.info(f"🎯 切换到 Stage 3 (Gap=8) at step {train_steps}")
                
                # 阶段切换时更新dataset
                if stage_changed:
                    if rank == 0:
                        logger.debug(f"正在更新dataset，train_steps={train_steps}")
                    dataset_train.set_training_step(train_steps)
                    if hasattr(dataset_val_vis, 'set_training_step'):
                        dataset_val_vis.set_training_step(train_steps)
                    if rank == 0:
                        logger.debug("Dataset已更新")

                # EMA更新
                ema_decay = get_ema_decay(train_steps, base_decay=0.9999, min_decay=0.99, warmup_steps=1000)
                with torch.no_grad():
                    update_ema(ema, get_raw_model(model), decay=ema_decay)

                # 日志记录
                running_loss += loss.item() * accum_divisor
                log_steps += 1

                # 定期日志
                if train_steps % args.log_every == 0 and rank == 0:
                    avg_loss = running_loss / log_steps
                    steps_per_sec = log_steps / (time() - start_time)
                    logger.info(
                        f"Step {train_steps:07d} | Stage {stage} | Gap {frame_gap} | F={f} | "  # ✅
                        f"Loss: {avg_loss:.4f} | Rec Loss: {recursive_loss_value:.4f} | "
                        f"Grad Norm: {grad_norm:.4f} | Steps/Sec: {steps_per_sec:.2f}"
                    )
                    write_experiment_metric(tracker, args.tracking_backend, 'Train/Loss', avg_loss, train_steps)
                    write_experiment_metric(tracker, args.tracking_backend, 'Train/RecursiveLoss', recursive_loss_value, train_steps)
                    write_experiment_metric(tracker, args.tracking_backend, 'Train/Stage', stage, train_steps)
                    write_experiment_metric(tracker, args.tracking_backend, 'Train/GradNorm', grad_norm, train_steps)
                    write_experiment_metric(tracker, args.tracking_backend, 'Train/LR', opt.param_groups[0]['lr'], train_steps)
                    write_experiment_metric(tracker, args.tracking_backend, 'Train/EMADecay', ema_decay, train_steps)
                    running_loss = 0.0
                    log_steps = 0
                    start_time = time()

                maybe_cleanup_cuda(train_steps, cuda_empty_cache_interval)

            if rank == 0 and train_progress is not None:
                train_progress.update(
                    epoch_task,
                    completed=step + 1,
                    status=format_epoch_progress_status(
                        batch_step=step + 1,
                        total_batches=len(loader_train),
                        train_steps=train_steps,
                        max_train_steps=args.max_train_steps,
                        stage=stage,
                        frame_gap=frame_gap,
                        num_frames=f,
                        loss_value=loss_display_value,
                        recursive_loss_value=recursive_loss_value,
                        grad_norm=grad_norm_value,
                    ),
                )
                train_progress.update(
                    overall_task,
                    completed=train_steps,
                    status=format_overall_progress_status(epoch, num_train_epochs, train_steps, args),
                )

            del batch, video, latent, t
            if video_flat is not None:
                del video_flat
            if latent_sparse is not None:
                del latent_sparse
            if weight_batch is not None:
                del weight_batch
            if loss is not None:
                del loss
            if base_loss is not None:
                del base_loss

            # 终止条件
            if train_steps >= args.max_train_steps:
                break

        distributed_barrier()

        # ========== 每10个epoch保存模型 ==========
        if rank == 0 and (epoch + 1) % 10 == 0:
            checkpoint = build_training_checkpoint(
                model=model,
                ema=ema,
                train_steps=train_steps,
                epoch=epoch + 1,
                best_val_metric=best_val_metric,
                opt=opt,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
            )
            ckpt_path = f"{checkpoint_dir}/epoch_{epoch+1:03d}.pth"
            torch.save(checkpoint, ckpt_path)
            # 保存后清理
            del checkpoint
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info(f"Saved epoch checkpoint: {ckpt_path}")

        # 保存最新epoch检查点（仅保存model/ema）
        if rank == 0:
            checkpoint = build_training_checkpoint(
                model=model,
                ema=ema,
                train_steps=train_steps,
                epoch=epoch + 1,
                best_val_metric=best_val_metric,
                opt=opt,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
            )
            torch.save(checkpoint, f"{checkpoint_dir}/latest_epoch_train_model.pth")
            # 保存后立即清理
            del checkpoint
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # -------------------------- 验证阶段（同步修改loss加权） --------------------------
        # ========== 1. 每个epoch都执行：单样本可视化保存（轻量） ==========
        # 每个epoch的轻量可视化（调用修改后的函数）
        if rank == 0:
            model.eval()
            vae.eval()
            val_diffusion = create_diffusion(
                timestep_respacing=timestep_respacing_val,
                diffusion_steps=args.diffusion_steps,
                learn_sigma=args.learn_sigma,
            )
            
            with torch.no_grad():
                dataset_val_vis.set_training_step(train_steps)
                val_example_images = {}
                vis_chunk_count = 0
                for vis_start in range(0, len(val_visualization_indices), val_visualization_batch_size):
                    chunk_indices = val_visualization_indices[vis_start: vis_start + val_visualization_batch_size]
                    if not chunk_indices:
                        continue
                    vis_chunk_count += 1
                    video_val = torch.stack(
                        [dataset_val_vis[idx]['video'] for idx in chunk_indices],
                        dim=0,
                    ).to(device)
                    chunk_images = save_val_sample_visualization(
                        epoch=epoch,
                        video_val=video_val,
                        vae=vae,
                        val_diffusion=val_diffusion,
                        device=device,
                        val_fold=val_fold,
                        generate_vessel_mask_adaptive=generate_vessel_mask_adaptive,
                        raw_model=get_raw_model(model),
                        current_step=train_steps,
                        stage0_steps=args.stage0_steps,
                        stage1_steps=args.stage1_steps,
                        stage2_steps=args.stage2_steps,
                        sample_ids=chunk_indices,
                        logger=logger,
                    )
                    for image_name, image_spec in chunk_images.items():
                        if image_spec is None:
                            continue
                        existing_images = val_example_images.setdefault(image_name, [])
                        if isinstance(image_spec, list):
                            existing_images.extend(image_spec)
                        else:
                            existing_images.append(image_spec)
                    del video_val, chunk_images
                    torch.cuda.empty_cache()

                vis_stage, vis_stage_name = get_visualization_stage_info(
                    train_steps,
                    args.stage0_steps,
                    args.stage1_steps,
                    args.stage2_steps,
                )
                if vis_stage > 0 and logger is not None and len(val_visualization_indices) > 0:
                    vis_modes = "基础预测+递归链" if vis_stage >= 2 else "基础预测"
                    logger.info(
                        f"Epoch {epoch+1} {vis_stage_name} 可视化保存完成"
                        f"（{vis_modes}，样本数={len(val_visualization_indices)}，批次数={vis_chunk_count}）"
                    )
                write_experiment_images(
                    tracker,
                    args.tracking_backend,
                    val_example_images,
                    step=train_steps,
                )
            
            model.train()  # 回到训练模式
            torch.cuda.empty_cache()

        distributed_barrier()

        # ========== 2. 每val_interval个epoch执行：完整验证（指标+最优模型） ==========
        # ========== 完整验证（每val_interval个epoch） ==========
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            vae.eval()
            val_diffusion = create_diffusion(
                timestep_respacing=timestep_respacing_val,
                diffusion_steps=args.diffusion_steps,
                learn_sigma=args.learn_sigma,
            )

            stage_meta = get_stage_metadata(train_steps, args)
            stage_name = f"Stage{stage_meta['stage']}"
            local_video_total = len(sampler_val_metrics)

            if rank == 0:
                logger.info(f"\n{'='*60}")
                logger.info(
                    f"开始 {stage_name} 滑窗三连帧验证 | "
                    f"代理任务: 首尾帧预测中间帧 | "
                    f"triplet batch size={triplet_eval_batch_size}"
                )
                logger.info(f"{'='*60}\n")

            val_mae_sum, val_mse_sum, val_psnr_sum = 0.0, 0.0, 0.0
            val_triplet_count = 0
            val_video_count = 0
            processed_videos = 0
            val_task = None

            if train_progress is not None:
                val_task = train_progress.add_task(
                    "Val Sliding Triplets",
                    total=max(1, local_video_total),
                    completed=0,
                    status=(
                        f"epoch {epoch + 1}/{num_train_epochs} | "
                        f"{stage_name} | videos 0/{local_video_total} | world {world_size}"
                    ),
                )

            with torch.no_grad():
                for val_batch in loader_val_metrics:
                    for video_val in val_batch['videos']:
                        metrics = evaluate_video_sliding_triplets(
                            video_val,
                            model_forward=get_raw_model(model).forward,
                            vae=vae,
                            diffusion=val_diffusion,
                            device=device,
                            triplet_batch_size=triplet_eval_batch_size,
                            return_pred_video=False,
                            force_grayscale=True,
                        )
                        val_mae_sum += metrics['mae_sum']
                        val_mse_sum += metrics['mse_sum']
                        val_psnr_sum += metrics['psnr_sum']
                        val_triplet_count += metrics['count']
                        val_video_count += 1
                        processed_videos += 1

                        if val_task is not None:
                            train_progress.update(
                                val_task,
                                completed=min(processed_videos, max(1, local_video_total)),
                                status=(
                                    f"epoch {epoch + 1}/{num_train_epochs} | "
                                    f"{stage_name} | videos {processed_videos}/{local_video_total} | "
                                    f"triplets {val_triplet_count} | world {world_size}"
                                ),
                            )

            if val_task is not None:
                train_progress.remove_task(val_task)

            metrics_tensor = torch.tensor(
                [
                    val_mae_sum,
                    val_mse_sum,
                    val_psnr_sum,
                    float(val_triplet_count),
                    float(val_video_count),
                ],
                device=device,
                dtype=torch.float64,
            )
            if world_size > 1:
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

            total_triplet_count = max(int(metrics_tensor[3].item()), 1)
            total_video_count = max(int(metrics_tensor[4].item()), 1)
            val_mae = metrics_tensor[0].item() / total_triplet_count
            val_mse = metrics_tensor[1].item() / total_triplet_count
            val_psnr = metrics_tensor[2].item() / total_triplet_count
            val_metric = (val_mae + val_mse) / 2

            if rank == 0:
                logger.info(f"\n{'='*60}")
                logger.info(
                    f"Epoch {epoch+1} {stage_name} 滑窗三连帧验证 | "
                    f"Videos: {total_video_count} | Triplets: {total_triplet_count} | "
                    f"MAE: {val_mae:.4f} | MSE: {val_mse:.4f} | "
                    f"PSNR: {val_psnr:.4f} | Metric: {val_metric:.4f} | "
                    f"Best: {best_val_metric:.4f}"
                )
                logger.info(f"{'='*60}\n")

                write_experiment_metric(tracker, args.tracking_backend, 'Val_Triplet/MAE', val_mae, train_steps)
                write_experiment_metric(tracker, args.tracking_backend, 'Val_Triplet/MSE', val_mse, train_steps)
                write_experiment_metric(tracker, args.tracking_backend, 'Val_Triplet/PSNR', val_psnr, train_steps)
                write_experiment_metric(tracker, args.tracking_backend, 'Val_Triplet/Metric', val_metric, train_steps)

                if val_metric < best_val_metric:
                    prev_best = best_val_metric
                    best_val_metric = val_metric
                    checkpoint = {
                        "model": get_raw_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "val_metric": val_metric,
                        "epoch": epoch + 1,
                        "train_steps": train_steps,
                        "prev_best_metric": prev_best,
                        "stage": stage_name,
                        "triplet_metrics": {
                            "mae": val_mae,
                            "mse": val_mse,
                            "psnr": val_psnr,
                            "triplet_count": total_triplet_count,
                            "video_count": total_video_count,
                        },
                    }
                    best_ckpt_path = f"{checkpoint_dir}/best_epoch_train_model.pth"
                    torch.save(checkpoint, best_ckpt_path)
                    del checkpoint
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    logger.info(
                        f"✅ 保存最优模型到 {best_ckpt_path} | "
                        f"新最优指标: {val_metric:.4f} (之前: {prev_best:.4f}) | "
                        f"提升: {prev_best - val_metric:.4f}"
                    )

                latest_checkpoint = build_training_checkpoint(
                    model=model,
                    ema=ema,
                    train_steps=train_steps,
                    epoch=epoch + 1,
                    best_val_metric=best_val_metric,
                    opt=opt,
                    lr_scheduler=lr_scheduler,
                    scaler=scaler,
                )
                torch.save(latest_checkpoint, f"{checkpoint_dir}/latest_epoch_train_model.pth")
                del latest_checkpoint

            model.train()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        distributed_barrier()

        # 终止训练
        if train_steps >= args.max_train_steps:
            break

    # 训练结束
    distributed_barrier()
    if rank == 0:
        if train_progress is not None:
            train_progress.stop()
        logger.info(f"Training finished! Total steps: {train_steps}")
        close_experiment_tracker(tracker, args.tracking_backend)
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_cta.yaml")
    # ========== 新增参数：血管mask最大权重 ==========
    parser.add_argument("--vessel_max_weight", type=float, default=None, help="Max weight for vessel region")
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=None,
        choices=["tensorboard", "wandb"],
        help="Experiment tracking backend",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity/team",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="Weights & Biases logging mode",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--ddp-timeout-minutes",
        type=int,
        default=None,
        help="Process group timeout in minutes for long rank-0 validation/checkpoint sections",
    )
    cli_args = parser.parse_args()

    config = OmegaConf.load(cli_args.config)
    if cli_args.vessel_max_weight is not None:
        if not hasattr(config, "vessel_mask") or config.vessel_mask is None:
            config.vessel_mask = OmegaConf.create({})
        config.vessel_mask.max_weight = cli_args.vessel_max_weight
    if cli_args.log_level is not None:
        config.log_level = cli_args.log_level
    if cli_args.tracker is not None:
        config.tracking_backend = cli_args.tracker
    if cli_args.wandb_project is not None:
        config.wandb_project = cli_args.wandb_project
    if cli_args.wandb_entity is not None:
        config.wandb_entity = cli_args.wandb_entity
    if cli_args.wandb_mode is not None:
        config.wandb_mode = cli_args.wandb_mode
    if cli_args.wandb_run_name is not None:
        config.wandb_run_name = cli_args.wandb_run_name
    if cli_args.ddp_timeout_minutes is not None:
        config.ddp_timeout_minutes = cli_args.ddp_timeout_minutes

    main(config)
