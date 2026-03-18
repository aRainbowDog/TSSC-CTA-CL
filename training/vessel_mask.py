import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.filters import apply_hysteresis_threshold


def generate_vessel_mask_adaptive(frames_gray, max_weight=100.0, base_weight=1.0):
    """
    Generate a vessel mask from first/last frame differences plus a soft weighting map.
    """
    first_frame = frames_gray[0].astype(np.float32)
    last_frame = frames_gray[-1].astype(np.float32)

    diff_map = np.abs(first_frame - last_frame)
    diff_smooth = cv2.GaussianBlur(diff_map, (3, 3), 0)

    diff_norm = (diff_smooth / (diff_smooth.max() + 1e-5) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    diff_enhanced = clahe.apply(diff_norm)

    flat_data = diff_enhanced.flatten()
    dynamic_high = max(np.percentile(flat_data, 97.5), 70)
    dynamic_low = max(np.percentile(flat_data, 96), 40)

    mask_binary = apply_hysteresis_threshold(
        diff_enhanced, dynamic_low, dynamic_high
    ).astype(np.uint8) * 255

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_refined = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_connected = cv2.morphologyEx(mask_refined, cv2.MORPH_CLOSE, kernel_close)

    cnts, _ = cv2.findContours(mask_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_final = np.zeros_like(mask_connected)
    h, w = mask_final.shape
    for contour in cnts:
        area = cv2.contourArea(contour)
        if 20 < area < (h * w * 0.2):
            cv2.drawContours(mask_final, [contour], -1, 255, -1)

    if np.any(mask_final > 0):
        dist = cv2.distanceTransform(mask_final, cv2.DIST_L2, 5)
        soft_weight = np.exp(-dist / 8.0) * (max_weight - base_weight) + base_weight
        soft_weight = np.clip(soft_weight, base_weight, max_weight)
    else:
        soft_weight = np.ones_like(diff_map, dtype=np.float32) * base_weight

    return mask_final, soft_weight


def prepare_mask_for_latent(soft_weight_np, latent_size, device):
    """
    Downsample a pixel-space soft mask to latent resolution.
    """
    weight_tensor = torch.from_numpy(soft_weight_np).float().to(device)
    weight_tensor = weight_tensor.unsqueeze(0).unsqueeze(0)
    return F.interpolate(
        weight_tensor,
        size=(latent_size, latent_size),
        mode="bilinear",
        align_corners=False,
    )
