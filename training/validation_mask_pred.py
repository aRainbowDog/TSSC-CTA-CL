import os

import numpy as np
import torch
import torch.distributed as dist
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision.utils import save_image

from dataloader.data_loader_mask import collate_mask_prediction_batch
from training.latent_utils import decode_latent_batch_to_image, encode_image_batch_to_latent
from utils.triplet_eval import prepare_gt_video_for_eval, prepare_pred_video_for_eval


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


def get_fixed_visualization_indices(dataset_len, count):
    if dataset_len <= 0 or count <= 0:
        return []
    if count >= dataset_len:
        return list(range(dataset_len))
    return sorted({int(idx) for idx in np.linspace(0, dataset_len - 1, num=count)})


def get_validation_modes(mask_cfg):
    modes = mask_cfg.get("validation_modes", None)
    if modes is None:
        return ["single", "span", "anchor"]
    return [str(mode) for mode in list(modes)]


def canonicalize_validation_mode(mode):
    mode_name = str(mode).strip().lower()
    if mode_name == "single":
        return "single_mid"
    return mode_name


def build_validation_frame_mask_variants(valid_mask, mode, min_visible=2, span_length=2, device=None):
    device = device or valid_mask.device
    batch_size, sequence_length = valid_mask.shape
    canonical_mode = canonicalize_validation_mode(mode)

    if canonical_mode == "single_all":
        variants = []
        for target_idx in range(sequence_length):
            frame_mask = torch.zeros(batch_size, sequence_length, dtype=torch.float32, device=device)
            has_any = False
            for batch_idx in range(batch_size):
                valid_indices = torch.nonzero(valid_mask[batch_idx] > 0.5, as_tuple=False).flatten().tolist()
                if len(valid_indices) <= min_visible:
                    continue
                internal_candidates = valid_indices[1:-1] if len(valid_indices) > 2 else []
                if target_idx in internal_candidates:
                    frame_mask[batch_idx, target_idx] = 1.0
                    has_any = True
            if has_any:
                variants.append((f"frame_{target_idx:02d}", frame_mask))
        return variants

    frame_mask = torch.zeros(batch_size, sequence_length, dtype=torch.float32, device=device)
    variant_name = canonical_mode
    for batch_idx in range(batch_size):
        valid_indices = torch.nonzero(valid_mask[batch_idx] > 0.5, as_tuple=False).flatten().tolist()
        if len(valid_indices) <= min_visible:
            continue

        if canonical_mode == "single_mid":
            single_candidates = valid_indices[1:-1] if len(valid_indices) > 2 else valid_indices
            masked_indices = [single_candidates[len(single_candidates) // 2]]
            variant_name = "mid"
        elif canonical_mode == "span":
            max_span = max(1, len(valid_indices) - min_visible)
            current_span = min(max(1, int(span_length)), max_span)
            start = max(0, (len(valid_indices) - current_span) // 2)
            masked_indices = valid_indices[start:start + current_span]
        elif canonical_mode == "anchor":
            anchor_count = max(2, int(min_visible))
            anchor_positions = np.linspace(0, len(valid_indices) - 1, num=min(anchor_count, len(valid_indices)))
            visible_indices = {valid_indices[int(round(pos))] for pos in anchor_positions}
            masked_indices = [idx for idx in valid_indices if idx not in visible_indices]
            if not masked_indices:
                masked_indices = [valid_indices[len(valid_indices) // 2]]
        else:
            raise ValueError(f"Unsupported validation mask mode: {mode}")

        frame_mask[batch_idx, masked_indices] = 1.0

    if float(frame_mask.sum().item()) <= 0:
        return []
    return [(variant_name, frame_mask)]


def build_validation_frame_mask(valid_mask, mode, min_visible=2, span_length=2, device=None):
    variants = build_validation_frame_mask_variants(
        valid_mask=valid_mask,
        mode=mode,
        min_visible=min_visible,
        span_length=span_length,
        device=device,
    )
    if variants:
        return variants[0][1]
    device = device or valid_mask.device
    return torch.zeros(valid_mask.shape[0], valid_mask.shape[1], dtype=torch.float32, device=device)


@torch.no_grad()
def reconstruct_masked_sequence(model_forward, diffusion, latent_sequence, frame_times, frame_mask, device, use_amp=True):
    if float(frame_mask.sum().item()) <= 0:
        return latent_sequence.clone()

    raw_x = latent_sequence.permute(0, 2, 1, 3, 4).contiguous()
    latent_height, latent_width = latent_sequence.shape[-2:]
    sample_mask = frame_mask[:, :, None, None].expand(-1, -1, latent_height, latent_width).contiguous()
    noise = torch.randn_like(raw_x)

    with torch.cuda.amp.autocast(enabled=use_amp):
        samples = diffusion.p_sample_loop(
            model_forward,
            noise.shape,
            noise,
            clip_denoised=True,
            progress=False,
            device=device,
            model_kwargs={"frame_times": frame_times},
            raw_x=raw_x,
            mask=sample_mask,
        )

    samples = torch.clamp(samples, -5.0, 5.0)
    reconstructed = samples * sample_mask.unsqueeze(1) + raw_x * (1 - sample_mask.unsqueeze(1))
    return reconstructed.permute(0, 2, 1, 3, 4).contiguous()


@torch.no_grad()
def decode_latent_sequence(vae, latent_sequence, use_amp=True, decode_batch_size=0):
    batch_size, sequence_length, channels, height, width = latent_sequence.shape
    latent_flat = latent_sequence.reshape(-1, channels, height, width)
    decoded = decode_latent_batch_to_image(
        vae,
        latent_flat,
        use_amp=use_amp,
        chunk_size=decode_batch_size,
    )
    return decoded.reshape(batch_size, sequence_length, *decoded.shape[1:])


def compute_masked_frame_metrics(pred_video, gt_video, frame_mask, valid_mask):
    pred_np = pred_video.detach().cpu().numpy()
    gt_np = gt_video.detach().cpu().numpy()
    active_mask = ((frame_mask > 0.5) & (valid_mask > 0.5)).detach().cpu().numpy()

    metrics = {
        "mae_sum": 0.0,
        "mse_sum": 0.0,
        "psnr_sum": 0.0,
        "ssim_sum": 0.0,
        "count": 0.0,
        "video_count": 0.0,
    }
    for batch_idx in range(active_mask.shape[0]):
        masked_indices = np.where(active_mask[batch_idx])[0]
        if masked_indices.size == 0:
            continue
        metrics["video_count"] += 1.0
        for frame_idx in masked_indices:
            pred_frame = pred_np[batch_idx, frame_idx]
            gt_frame = gt_np[batch_idx, frame_idx]
            diff = pred_frame - gt_frame
            metrics["mae_sum"] += float(np.abs(diff).mean())
            metrics["mse_sum"] += float(np.square(diff).mean())
            metrics["psnr_sum"] += float(peak_signal_noise_ratio(gt_frame, pred_frame, data_range=2.0))
            pred_gray = pred_frame.mean(axis=0)
            gt_gray = gt_frame.mean(axis=0)
            metrics["ssim_sum"] += float(structural_similarity(gt_gray, pred_gray, data_range=2.0))
            metrics["count"] += 1.0
    return metrics


def reduce_metrics_across_ranks(metric_totals, device):
    mode_names = list(metric_totals.keys())
    flat_values = []
    for mode_name in mode_names:
        mode_metrics = metric_totals[mode_name]
        flat_values.extend(
            [
                mode_metrics["mae_sum"],
                mode_metrics["mse_sum"],
                mode_metrics["psnr_sum"],
                mode_metrics["ssim_sum"],
                mode_metrics["count"],
                mode_metrics["video_count"],
            ]
        )

    metrics_tensor = torch.tensor(flat_values, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

    reduced = {}
    offset = 0
    for mode_name in mode_names:
        reduced[mode_name] = {
            "mae_sum": float(metrics_tensor[offset].item()),
            "mse_sum": float(metrics_tensor[offset + 1].item()),
            "psnr_sum": float(metrics_tensor[offset + 2].item()),
            "ssim_sum": float(metrics_tensor[offset + 3].item()),
            "count": float(metrics_tensor[offset + 4].item()),
            "video_count": float(metrics_tensor[offset + 5].item()),
        }
        offset += 6
    return reduced


def summarize_validation_metrics(metric_totals):
    mode_summaries = {}
    macro_mae = []
    macro_mse = []
    macro_psnr = []
    macro_ssim = []
    macro_metric = []
    weighted_mae_sum = 0.0
    weighted_mse_sum = 0.0
    weighted_psnr_sum = 0.0
    weighted_ssim_sum = 0.0
    weighted_count = 0.0

    for mode_name, totals in metric_totals.items():
        count = max(0.0, float(totals["count"]))
        if count > 0:
            mae = totals["mae_sum"] / count
            mse = totals["mse_sum"] / count
            psnr = totals["psnr_sum"] / count
            ssim = totals["ssim_sum"] / count
            metric = (mae + mse) / 2.0
            macro_mae.append(mae)
            macro_mse.append(mse)
            macro_psnr.append(psnr)
            macro_ssim.append(ssim)
            macro_metric.append(metric)
            weighted_mae_sum += totals["mae_sum"]
            weighted_mse_sum += totals["mse_sum"]
            weighted_psnr_sum += totals["psnr_sum"]
            weighted_ssim_sum += totals["ssim_sum"]
            weighted_count += count
        else:
            mae = float("nan")
            mse = float("nan")
            psnr = float("nan")
            ssim = float("nan")
            metric = float("nan")
        mode_summaries[mode_name] = {
            "mae": mae,
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim,
            "metric": metric,
            "count": count,
            "video_count": float(totals["video_count"]),
        }

    overall = {
        "macro_mae": float(np.mean(macro_mae)) if macro_mae else float("nan"),
        "macro_mse": float(np.mean(macro_mse)) if macro_mse else float("nan"),
        "macro_psnr": float(np.mean(macro_psnr)) if macro_psnr else float("nan"),
        "macro_ssim": float(np.mean(macro_ssim)) if macro_ssim else float("nan"),
        "macro_metric": float(np.mean(macro_metric)) if macro_metric else float("inf"),
        "weighted_mae": weighted_mae_sum / weighted_count if weighted_count > 0 else float("nan"),
        "weighted_mse": weighted_mse_sum / weighted_count if weighted_count > 0 else float("nan"),
        "weighted_psnr": weighted_psnr_sum / weighted_count if weighted_count > 0 else float("nan"),
        "weighted_ssim": weighted_ssim_sum / weighted_count if weighted_count > 0 else float("nan"),
        "masked_frame_count": weighted_count,
        "mode_count": len(macro_metric),
    }
    return {
        "modes": mode_summaries,
        "overall": overall,
    }


@torch.no_grad()
def evaluate_mask_prediction(
    data_loader,
    model_forward,
    vae,
    diffusion,
    device,
    mask_cfg,
    use_amp=True,
    max_batches=0,
    vae_encode_batch_size=0,
    vae_decode_batch_size=0,
):
    mode_names = get_validation_modes(mask_cfg)
    min_visible = int(mask_cfg.get("min_visible_frames", 2))
    span_length = int(mask_cfg.get("validation_span_length", mask_cfg.get("max_masked_frames", 2)))
    metric_totals = {
        mode_name: {
            "mae_sum": 0.0,
            "mse_sum": 0.0,
            "psnr_sum": 0.0,
            "ssim_sum": 0.0,
            "count": 0.0,
            "video_count": 0.0,
        }
        for mode_name in mode_names
    }

    for batch_idx, batch in enumerate(data_loader):
        if max_batches and batch_idx >= max_batches:
            break

        video = batch["video"].to(device, non_blocking=True)
        frame_times = batch["frame_times"].to(device, non_blocking=True)
        valid_mask = batch["frame_valid_mask"].to(device, non_blocking=True)
        batch_size = video.shape[0]

        video_flat = video.reshape(-1, *video.shape[2:])
        latent = encode_image_batch_to_latent(
            vae,
            video_flat,
            use_amp=use_amp,
            chunk_size=vae_encode_batch_size,
        )
        latent = latent.reshape(batch_size, video.shape[1], *latent.shape[1:])

        gt_video_eval = prepare_gt_video_for_eval(video.detach().cpu())
        valid_mask_cpu = valid_mask.detach().cpu()

        for mode_name in mode_names:
            frame_mask_variants = build_validation_frame_mask_variants(
                valid_mask=valid_mask,
                mode=mode_name,
                min_visible=min_visible,
                span_length=span_length,
                device=device,
            )
            for _, frame_mask in frame_mask_variants:
                reconstructed_latent = reconstruct_masked_sequence(
                    model_forward=model_forward,
                    diffusion=diffusion,
                    latent_sequence=latent,
                    frame_times=frame_times,
                    frame_mask=frame_mask,
                    device=device,
                    use_amp=use_amp,
                )
                decoded_video = decode_latent_sequence(
                    vae,
                    reconstructed_latent,
                    use_amp=use_amp,
                    decode_batch_size=vae_decode_batch_size,
                )
                pred_video_eval = prepare_pred_video_for_eval(decoded_video.detach().cpu(), force_grayscale=True)
                batch_metrics = compute_masked_frame_metrics(
                    pred_video=pred_video_eval,
                    gt_video=gt_video_eval,
                    frame_mask=frame_mask.detach().cpu(),
                    valid_mask=valid_mask_cpu,
                )
                for key, value in batch_metrics.items():
                    metric_totals[mode_name][key] += value

        del batch, video, frame_times, valid_mask, video_flat, latent

    reduced_totals = reduce_metrics_across_ranks(metric_totals, device=device)
    return summarize_validation_metrics(reduced_totals)


@torch.no_grad()
def save_mask_prediction_visualizations(
    dataset,
    output_dir,
    sample_indices,
    batch_size,
    model_forward,
    vae,
    diffusion,
    device,
    mask_cfg,
    epoch,
    train_steps,
    use_amp=True,
    vae_encode_batch_size=0,
    vae_decode_batch_size=0,
):
    if not sample_indices:
        return {}

    os.makedirs(output_dir, exist_ok=True)
    subset = Subset(dataset, sample_indices)
    loader = DataLoader(
        subset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_mask_prediction_batch,
    )

    mode_names = get_validation_modes(mask_cfg)
    min_visible = int(mask_cfg.get("min_visible_frames", 2))
    span_length = int(mask_cfg.get("validation_span_length", mask_cfg.get("max_masked_frames", 2)))
    generated_images = {f"Val_MaskPred/{mode_name}": [] for mode_name in mode_names}

    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        frame_times = batch["frame_times"].to(device, non_blocking=True)
        valid_mask = batch["frame_valid_mask"].to(device, non_blocking=True)
        dataset_indices = batch["dataset_index"].tolist()
        video_names = batch["video_name"]
        batch_size_current = video.shape[0]

        video_flat = video.reshape(-1, *video.shape[2:])
        latent = encode_image_batch_to_latent(
            vae,
            video_flat,
            use_amp=use_amp,
            chunk_size=vae_encode_batch_size,
        )
        latent = latent.reshape(batch_size_current, video.shape[1], *latent.shape[1:])

        gt_video_eval = prepare_gt_video_for_eval(video.detach().cpu())
        sequence_length = gt_video_eval.shape[1]

        for mode_name in mode_names:
            frame_mask_variants = build_validation_frame_mask_variants(
                valid_mask=valid_mask,
                mode=mode_name,
                min_visible=min_visible,
                span_length=span_length,
                device=device,
            )
            for variant_name, frame_mask in frame_mask_variants:
                reconstructed_latent = reconstruct_masked_sequence(
                    model_forward=model_forward,
                    diffusion=diffusion,
                    latent_sequence=latent,
                    frame_times=frame_times,
                    frame_mask=frame_mask,
                    device=device,
                    use_amp=use_amp,
                )
                decoded_video = decode_latent_sequence(
                    vae,
                    reconstructed_latent,
                    use_amp=use_amp,
                    decode_batch_size=vae_decode_batch_size,
                )
                pred_video_eval = prepare_pred_video_for_eval(decoded_video.detach().cpu(), force_grayscale=True)

                frame_mask_vis = frame_mask.detach().cpu()[:, :, None, None, None]
                masked_input = gt_video_eval * (1 - frame_mask_vis) + (-1.0) * frame_mask_vis
                mask_row = frame_mask_vis.expand_as(gt_video_eval) * 2 - 1

                for sample_offset, dataset_index in enumerate(dataset_indices):
                    image_tensor = torch.cat(
                        [
                            gt_video_eval[sample_offset:sample_offset + 1],
                            masked_input[sample_offset:sample_offset + 1],
                            pred_video_eval[sample_offset:sample_offset + 1],
                            mask_row[sample_offset:sample_offset + 1],
                        ],
                        dim=1,
                    )
                    image_tensor = image_tensor.reshape(-1, *image_tensor.shape[2:])
                    image_path = os.path.join(
                        output_dir,
                        f"epoch_{epoch + 1:04d}_step_{train_steps:07d}_{mode_name}_{variant_name}_idx_{dataset_index:04d}.png",
                    )
                    save_image(
                        image_tensor,
                        image_path,
                        nrow=sequence_length,
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    generated_images[f"Val_MaskPred/{mode_name}"].append(
                        {
                            "path": image_path,
                            "caption": (
                                f"Epoch {epoch + 1} step {train_steps} "
                                f"{mode_name}/{variant_name} {video_names[sample_offset]}"
                            ),
                        }
                    )

        del batch, video, frame_times, valid_mask, video_flat, latent

    return generated_images
