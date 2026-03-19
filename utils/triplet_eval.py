import numpy as np
import torch
import cv2
from einops import rearrange
from skimage.metrics import peak_signal_noise_ratio


def _channel_dim(video):
    if video.ndim == 4:
        return 1
    if video.ndim == 5:
        return 2
    raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(video.shape)}")


def _repeat_to_three_channels(video):
    channel_dim = _channel_dim(video)
    repeat_dims = [1] * video.ndim
    repeat_dims[channel_dim] = 3
    return video.repeat(*repeat_dims)


def prepare_gt_video_for_eval(video):
    """GT口径与训练内validation保持一致：单通道扩成3通道，已有3通道则原样保留。"""
    if video.shape[_channel_dim(video)] == 1:
        return _repeat_to_three_channels(video)
    return video


def prepare_pred_video_for_eval(video, force_grayscale=True):
    """预测结果默认转为3通道灰度，与训练内validation保持一致。"""
    channel_dim = _channel_dim(video)
    if force_grayscale:
        if video.shape[channel_dim] == 1:
            return _repeat_to_three_channels(video)
        return _repeat_to_three_channels(video.mean(dim=channel_dim, keepdim=True))
    if video.shape[channel_dim] == 1:
        return _repeat_to_three_channels(video)
    return video


def build_sliding_triplets(video):
    """从 [T, C, H, W] 构建所有连续三连帧，返回 [T-2, 3, C, H, W]。"""
    if video.ndim != 4:
        raise ValueError(f"Expected video with shape [T, C, H, W], got {tuple(video.shape)}")
    if video.shape[0] < 3:
        empty_shape = (0, 3, video.shape[1], video.shape[2], video.shape[3])
        return video.new_empty(empty_shape)
    return torch.stack([video[idx: idx + 3] for idx in range(video.shape[0] - 2)], dim=0)


def _frame_to_gray_01(frame):
    if frame.ndim != 3:
        raise ValueError(f"Expected frame with shape [C, H, W], got {tuple(frame.shape)}")
    if frame.shape[0] == 1:
        gray = frame[0]
    else:
        gray = frame.mean(dim=0)
    gray = torch.clamp(gray * 0.5 + 0.5, 0.0, 1.0)
    return gray.detach().cpu().numpy().astype(np.float32)


def predict_intermediate_frame_dis_flow(start_frame, end_frame, alpha, method):
    start_gray = _frame_to_gray_01(start_frame)
    end_gray = _frame_to_gray_01(end_frame)
    start_u8 = np.clip(start_gray * 255.0, 0, 255).astype(np.uint8)
    end_u8 = np.clip(end_gray * 255.0, 0, 255).astype(np.uint8)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    grid_y, grid_x = np.mgrid[0:start_gray.shape[0], 0:start_gray.shape[1]].astype(np.float32)

    if method == "si_dis_flow":
        flow_forward = dis.calc(start_u8, end_u8, None)
        map_x = (grid_x + flow_forward[..., 0] * alpha).astype(np.float32)
        map_y = (grid_y + flow_forward[..., 1] * alpha).astype(np.float32)
        pred_gray = cv2.remap(
            np.ascontiguousarray(start_gray, dtype=np.float32),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    elif method == "bi_dis_flow":
        flow_forward = dis.calc(start_u8, end_u8, None)
        flow_backward = dis.calc(end_u8, start_u8, None)
        warp_f = cv2.remap(
            np.ascontiguousarray(start_gray, dtype=np.float32),
            (grid_x + flow_forward[..., 0] * alpha).astype(np.float32),
            (grid_y + flow_forward[..., 1] * alpha).astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        warp_b = cv2.remap(
            np.ascontiguousarray(end_gray, dtype=np.float32),
            (grid_x + flow_backward[..., 0] * (1.0 - alpha)).astype(np.float32),
            (grid_y + flow_backward[..., 1] * (1.0 - alpha)).astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        pred_gray = (1.0 - alpha) * warp_f + alpha * warp_b
    else:
        raise ValueError(f"Unsupported DIS flow baseline method: {method}")

    pred_gray = np.clip(pred_gray, 0.0, 1.0)
    return torch.from_numpy(pred_gray * 2.0 - 1.0).float().unsqueeze(0)


def _predict_middle_frame_dis_flow(start_frame, end_frame, method):
    return predict_intermediate_frame_dis_flow(start_frame, end_frame, alpha=0.5, method=method)


def predict_middle_frames_from_triplets_baseline(
        triplets,
        method,
        force_grayscale=True):
    """
    对一批三连帧做 baseline 中间帧预测。
    triplets: [B, 3, C, H, W]
    """
    gt_middle = prepare_gt_video_for_eval(triplets)[:, 1, :, :, :].detach().cpu()

    if method == "quadratic":
        raise ValueError(
            "quadratic baseline is not compatible with sliding-triplet evaluation: "
            "its original implementation uses the center frame as an anchor."
        )

    if method == "linear":
        pred_middle = 0.5 * (triplets[:, 0, :, :, :] + triplets[:, 2, :, :, :])
    elif method in {"si_dis_flow", "bi_dis_flow"}:
        pred_middle = torch.stack([
            _predict_middle_frame_dis_flow(sample[0], sample[2], method)
            for sample in triplets
        ], dim=0)
    else:
        raise ValueError(f"Unsupported baseline method: {method}")

    pred_middle = prepare_pred_video_for_eval(pred_middle, force_grayscale=force_grayscale)
    return pred_middle.detach().cpu(), gt_middle


def predict_middle_frames_from_triplets(
        triplets,
        model_forward,
        vae,
        diffusion,
        device,
        force_grayscale=True,
        latent_triplets=None):
    """
    对一批三连帧做“首尾预测中间帧”推理。
    triplets: [B, 3, C, H, W]
    返回:
      pred_middle: [B, C_out, H, W]
      gt_middle: [B, C_out, H, W]
    """
    batch_size = triplets.shape[0]
    gt_middle = prepare_gt_video_for_eval(triplets)[:, 1, :, :, :].detach().cpu()

    if latent_triplets is None:
        triplets_device = triplets.to(device, non_blocking=True)
        triplet_flat = rearrange(triplets_device, 'b f c h w -> (b f) c h w')
        latent_triplet = vae.encode(triplet_flat).latent_dist.sample().mul_(0.18215)
        latent_triplet = rearrange(latent_triplet, '(b f) c h w -> b f c h w', b=batch_size)
    else:
        latent_triplet = latent_triplets.to(device, non_blocking=True)

    _, _, _, h_latent, w_latent = latent_triplet.shape
    mask = torch.zeros(batch_size, 3, h_latent, w_latent, device=device)
    mask[:, 1, :, :] = 1.0

    latent_triplet_bcfhw = latent_triplet.permute(0, 2, 1, 3, 4)
    noise = torch.randn_like(latent_triplet_bcfhw)
    samples = diffusion.p_sample_loop(
        model_forward,
        noise.shape,
        noise,
        clip_denoised=True,
        progress=False,
        device=device,
        raw_x=latent_triplet_bcfhw,
        mask=mask,
    )
    samples = torch.clamp(samples, -5.0, 5.0)
    samples = samples * mask.unsqueeze(1) + latent_triplet_bcfhw * (1 - mask.unsqueeze(1))
    pred_middle_latent = samples[:, :, 1, :, :]
    pred_middle = vae.decode(pred_middle_latent / 0.18215).sample
    pred_middle = torch.clamp(pred_middle, -1.0, 1.0)
    pred_middle = prepare_pred_video_for_eval(pred_middle, force_grayscale=force_grayscale)

    return pred_middle.detach().cpu(), gt_middle


def evaluate_video_sliding_triplets(
        video,
        model_forward,
        vae,
        diffusion,
        device,
        triplet_batch_size=4,
        return_pred_video=False,
        force_grayscale=True):
    """
    对单个完整序列做滑窗三连帧评测。
    video: [T, C, H, W]
    """
    triplets = build_sliding_triplets(video)
    gt_video_for_eval = prepare_gt_video_for_eval(video).cpu()

    if triplets.shape[0] == 0:
        return {
            'mae_sum': 0.0,
            'mse_sum': 0.0,
            'psnr_sum': 0.0,
            'count': 0,
            'gt_video': gt_video_for_eval if return_pred_video else None,
            'pred_video': gt_video_for_eval.clone() if return_pred_video else None,
        }

    mae_sum = 0.0
    mse_sum = 0.0
    psnr_sum = 0.0
    predicted_middles = []
    video_device = video.to(device, non_blocking=True)
    latent_video = vae.encode(video_device).latent_dist.sample().mul_(0.18215)
    latent_triplets = build_sliding_triplets(latent_video)

    for start in range(0, triplets.shape[0], max(1, int(triplet_batch_size))):
        triplet_chunk = triplets[start: start + max(1, int(triplet_batch_size))]
        latent_triplet_chunk = latent_triplets[start: start + max(1, int(triplet_batch_size))]
        pred_middle, gt_middle = predict_middle_frames_from_triplets(
            triplet_chunk,
            model_forward=model_forward,
            vae=vae,
            diffusion=diffusion,
            device=device,
            force_grayscale=force_grayscale,
            latent_triplets=latent_triplet_chunk,
        )

        pred_np = pred_middle.detach().cpu().numpy()
        gt_np = gt_middle.detach().cpu().numpy()
        pred_flat = pred_np.reshape(pred_np.shape[0], -1)
        gt_flat = gt_np.reshape(gt_np.shape[0], -1)

        mae_sum += float(np.sum(np.abs(pred_flat - gt_flat).mean(axis=1)))
        mse_sum += float(np.sum(((pred_flat - gt_flat) ** 2).mean(axis=1)))
        psnr_sum += float(np.sum([
            peak_signal_noise_ratio(gt_np[idx], pred_np[idx], data_range=2.0)
            for idx in range(gt_np.shape[0])
        ]))

        if return_pred_video:
            predicted_middles.append(pred_middle.detach().cpu())

    pred_video = None
    if return_pred_video:
        pred_video = gt_video_for_eval.clone()
        pred_video[1:-1, :, :, :] = torch.cat(predicted_middles, dim=0)
    del video_device, latent_video, latent_triplets

    return {
        'mae_sum': mae_sum,
        'mse_sum': mse_sum,
        'psnr_sum': psnr_sum,
        'count': int(triplets.shape[0]),
        'gt_video': gt_video_for_eval if return_pred_video else None,
        'pred_video': pred_video,
    }


def evaluate_video_sliding_triplets_baseline(
        video,
        method,
        triplet_batch_size=4,
        return_pred_video=False,
        force_grayscale=True):
    """
    使用 baseline 方法对单个完整序列做滑窗三连帧评测。
    video: [T, C, H, W]
    """
    triplets = build_sliding_triplets(video)
    gt_video_for_eval = prepare_gt_video_for_eval(video).cpu()

    if triplets.shape[0] == 0:
        return {
            'mae_sum': 0.0,
            'mse_sum': 0.0,
            'psnr_sum': 0.0,
            'count': 0,
            'gt_video': gt_video_for_eval if return_pred_video else None,
            'pred_video': gt_video_for_eval.clone() if return_pred_video else None,
        }

    mae_sum = 0.0
    mse_sum = 0.0
    psnr_sum = 0.0
    predicted_middles = []

    for start in range(0, triplets.shape[0], max(1, int(triplet_batch_size))):
        triplet_chunk = triplets[start: start + max(1, int(triplet_batch_size))]
        pred_middle, gt_middle = predict_middle_frames_from_triplets_baseline(
            triplet_chunk,
            method=method,
            force_grayscale=force_grayscale,
        )

        pred_np = pred_middle.detach().cpu().numpy()
        gt_np = gt_middle.detach().cpu().numpy()
        pred_flat = pred_np.reshape(pred_np.shape[0], -1)
        gt_flat = gt_np.reshape(gt_np.shape[0], -1)

        mae_sum += float(np.sum(np.abs(pred_flat - gt_flat).mean(axis=1)))
        mse_sum += float(np.sum(((pred_flat - gt_flat) ** 2).mean(axis=1)))
        psnr_sum += float(np.sum([
            peak_signal_noise_ratio(gt_np[idx], pred_np[idx], data_range=2.0)
            for idx in range(gt_np.shape[0])
        ]))

        if return_pred_video:
            predicted_middles.append(pred_middle.detach().cpu())

    pred_video = None
    if return_pred_video:
        pred_video = gt_video_for_eval.clone()
        pred_video[1:-1, :, :, :] = torch.cat(predicted_middles, dim=0)

    return {
        'mae_sum': mae_sum,
        'mse_sum': mse_sum,
        'psnr_sum': psnr_sum,
        'count': int(triplets.shape[0]),
        'gt_video': gt_video_for_eval if return_pred_video else None,
        'pred_video': pred_video,
    }


def tensor_video_to_uint8_numpy(video):
    """将 [-1, 1] 范围的 [T, C, H, W] 转为 uint8 的 [T, H, W, C]。"""
    video_cpu = video.detach().cpu()
    return (
        ((video_cpu * 0.5 + 0.5) * 255)
        .add(0.5)
        .clamp(0, 255)
        .to(dtype=torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
        .numpy()
    )
