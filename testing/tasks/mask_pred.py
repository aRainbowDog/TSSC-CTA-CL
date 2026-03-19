import argparse
import os
import random
import re
import warnings
from functools import lru_cache

import numpy as np
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from scipy import linalg
import torch.nn.functional as F
from torchvision.utils import save_image

from dataloader.data_loader_mask import collate_mask_prediction_batch, mask_prediction_data_loader
from models.diffusion.gaussian_diffusion import create_diffusion
from models.model_dit import MVIF_models
from testing.common import (
    add_root_test_args,
    build_sample_key,
    default_eval_dir,
    list_prediction_files,
    load_manifest,
    save_prediction_record,
    sanitize_name,
    write_json,
    write_manifest,
)
from training.checkpointing import resolve_experiment_dir, safe_load_checkpoint
from training.common import create_logger_compat, create_rich_progress, distributed_barrier, load_config_from_parsed_args
from training.latent_utils import decode_latent_batch_to_image, encode_image_batch_to_latent
from training.runtime import get_raw_model
from training.tasks.mask_pred import apply_mask_pred_overrides, build_parser as build_train_parser
from training.validation_mask_pred import (
    DistributedEvalSampler,
    build_validation_frame_mask_variants,
    canonicalize_validation_mode,
    compute_masked_frame_metrics,
    decode_latent_sequence,
    get_fixed_visualization_indices,
    get_validation_modes,
    reconstruct_masked_sequence,
    save_mask_prediction_visualizations,
    summarize_validation_metrics,
)
from utils.triplet_eval import prepare_gt_video_for_eval, prepare_pred_video_for_eval
from utils.triplet_eval import predict_intermediate_frame_dis_flow
from utils.utils import cleanup, requires_grad, setup_distributed

warnings.filterwarnings("ignore")


def build_infer_parser():
    parser = build_train_parser(default_config="configs/config_mask_pred.yaml")
    add_root_test_args(parser, action="infer", task_aliases=["mask_pred", "mask_prediction", "mask-pred"])
    parser.add_argument("--stage", type=str, default="val", choices=["val", "test"], help="Dataset split used for inference/export.")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint filename under experiment checkpoints/ or an absolute path.")
    parser.add_argument(
        "--mode",
        type=str,
        default="reconstruct",
        choices=["reconstruct", "densify"],
        help="Save masked-frame reconstructions or export densified sequences.",
    )
    parser.add_argument(
        "--mask-mode",
        type=str,
        default="all",
        choices=["all", "single", "single_mid", "single_all", "span", "anchor"],
        help="Mask pattern used when `--mode reconstruct`.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to store inference outputs.")
    parser.add_argument("--max-batches", type=int, default=0, help="Optional cap for reconstruction batches.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for densification exports per rank.")
    parser.add_argument("--num-visualizations", type=int, default=8, help="How many qualitative reconstruction visualizations to export on rank 0.")
    parser.add_argument("--use-ema", type=str, default="auto", choices=["auto", "true", "false"], help="Whether to prefer EMA weights.")
    parser.add_argument("--densify-fps", type=float, default=1.0, help="Target FPS for densification export.")
    parser.add_argument(
        "--baseline-method",
        type=str,
        default=None,
        choices=["nearest", "linear", "si_dis_flow", "bi_dis_flow"],
        help="Run an image-space baseline under the same mask-pred reconstruct protocol.",
    )
    return parser


def build_eval_parser():
    parser = argparse.ArgumentParser(
        description="Read saved mask-pred reconstruction outputs and compute MAE/MSE/PSNR/SSIM offline."
    )
    add_root_test_args(parser, action="eval", task_aliases=["mask_pred", "mask_prediction", "mask-pred"])
    parser.add_argument("--input-dir", type=str, required=True, help="Directory produced by `test.py infer --task mask_pred --mode reconstruct`.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional directory for eval summary. Defaults to the input dir.")
    parser.add_argument(
        "--roi-mask-root",
        type=str,
        default=None,
        help="Optional root dir for per-slice ROI masks. Expected layout: <root>/<stage>/<patient>/slice_xxx/mask_union.png",
    )
    parser.add_argument(
        "--perceptual-batch-size",
        type=int,
        default=32,
        help="Batch size used for LPIPS/FID feature extraction.",
    )
    return parser


def build_visualize_parser():
    parser = argparse.ArgumentParser(
        description="Read saved mask-pred reconstruct outputs and export per-slice GT/pred overview figures."
    )
    add_root_test_args(parser, action="visualize", task_aliases=["mask_pred", "mask_prediction", "mask-pred"])
    parser.add_argument("--input-dir", type=str, required=True, help="Directory produced by `test.py infer --task mask_pred --mode reconstruct`.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional directory for visualization images. Defaults to <input-dir>/figures.")
    return parser


def resolve_eval_checkpoint_path(args, checkpoint_dir):
    requested = args.ckpt or getattr(args, "test_ckpt", None)
    if requested:
        return requested if os.path.isabs(requested) else os.path.join(checkpoint_dir, requested)
    best_ckpt = os.path.join(checkpoint_dir, "best_mask_pred.pth")
    if os.path.exists(best_ckpt):
        return best_ckpt
    return os.path.join(checkpoint_dir, "latest_mask_pred.pth")


def should_use_ema(args, checkpoint):
    if args.use_ema == "true":
        return True
    if args.use_ema == "false":
        return False
    val_weight_source = str(checkpoint.get("val_weight_source", "")).lower()
    if val_weight_source == "ema":
        return True
    if val_weight_source == "raw":
        return False
    return checkpoint.get("ema") is not None


def clone_mask_cfg(mask_cfg, mask_mode):
    cloned = dict(OmegaConf.to_container(mask_cfg, resolve=True))
    if mask_mode != "all":
        cloned["validation_modes"] = [mask_mode]
    return cloned


def apply_infer_overrides(config, cli_args):
    override_fields = [
        "ckpt",
        "stage",
        "mode",
        "mask_mode",
        "output_dir",
        "max_batches",
        "max_samples",
        "num_visualizations",
        "use_ema",
        "densify_fps",
        "baseline_method",
    ]
    for field_name in override_fields:
        setattr(config, field_name, getattr(cli_args, field_name, None))
    return config


def build_densified_timeline(relative_times, fps):
    observed = np.asarray(relative_times, dtype=np.float32)
    if observed.size == 0:
        raise ValueError("Observed timeline is empty")
    step = 1.0 / max(float(fps), 1e-6)
    grid = np.arange(observed[0], observed[-1] + step * 0.5, step, dtype=np.float32)
    merged = np.concatenate([observed, grid], axis=0)
    merged = np.unique(np.round(merged.astype(np.float64), 5)).astype(np.float32)
    merged.sort()
    observed_mask = np.array(
        [np.any(np.isclose(observed, time_point, atol=1e-4)) for time_point in merged],
        dtype=np.float32,
    )
    return merged, observed_mask


def convert_relative_times(relative_times, time_field):
    if time_field == "relative_time_seconds":
        return relative_times
    start_time = float(relative_times[0])
    end_time = float(relative_times[-1])
    denom = max(end_time - start_time, 1e-6)
    return ((relative_times - start_time) / denom).astype(np.float32)


def build_default_output_dir(args):
    if args.mode == "reconstruct":
        run_name = f"reconstruct_{args.stage}_{args.mask_mode}"
        if getattr(args, "baseline_method", None):
            run_name = f"{run_name}_{sanitize_name(args.baseline_method)}"
        return default_eval_dir(args.results_dir, "mask_pred", "reconstruct", args.stage, run_name)
    else:
        dir_name = f"densify_{args.stage}_{sanitize_name(f'{float(args.densify_fps):g}fps')}"
        return default_eval_dir(args.results_dir, "mask_pred", "densify", args.stage, dir_name)


@lru_cache(maxsize=16384)
def _load_union_roi_mask(mask_path):
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D ROI mask at {mask_path}, got shape {mask.shape}")
    return mask > 127.5


def _resolve_record_roi_mask_path(record, roi_mask_root):
    stage = str(record.get("stage", "")).strip()
    video_path = str(record.get("video_path", "")).strip()
    video_name = str(record.get("video_name", "")).strip()

    patient_key = os.path.basename(os.path.dirname(video_path)) if video_path else ""
    if not patient_key and "_slice_" in video_name:
        patient_key = video_name.rsplit("_slice_", 1)[0]
    if not patient_key:
        raise ValueError(f"Unable to resolve patient key from record: video_path={video_path}, video_name={video_name}")

    slice_match = re.search(r"slice_(\d+)", os.path.basename(video_path)) if video_path else None
    if slice_match is None:
        slice_match = re.search(r"slice_(\d+)", video_name)
    if slice_match is None:
        raise ValueError(f"Unable to resolve slice index from record: video_path={video_path}, video_name={video_name}")

    if not stage:
        raise ValueError(f"Record is missing dataset stage: {record}")

    slice_dir = f"slice_{int(slice_match.group(1)):03d}"
    return os.path.join(roi_mask_root, stage, patient_key, slice_dir, "mask_union.png")


def _mean_dict_values(values):
    valid = [float(v) for v in values if np.isfinite(v)]
    if not valid:
        return float("nan")
    return float(np.mean(valid))


class RunningStats:
    def __init__(self, dim):
        self.dim = int(dim)
        self.count = 0
        self.sum = np.zeros(self.dim, dtype=np.float64)
        self.sum_outer = np.zeros((self.dim, self.dim), dtype=np.float64)

    def update(self, features):
        feats = np.asarray(features, dtype=np.float64)
        if feats.ndim != 2 or feats.shape[1] != self.dim:
            raise ValueError(f"Expected features with shape [N, {self.dim}], got {feats.shape}")
        if feats.shape[0] == 0:
            return
        self.count += int(feats.shape[0])
        self.sum += feats.sum(axis=0)
        self.sum_outer += feats.T @ feats

    def mean_and_cov(self):
        if self.count <= 0:
            return None, None
        mean = self.sum / float(self.count)
        if self.count == 1:
            cov = np.zeros((self.dim, self.dim), dtype=np.float64)
        else:
            centered_outer = self.sum_outer - self.count * np.outer(mean, mean)
            cov = centered_outer / float(self.count - 1)
        return mean, cov


def _frechet_distance_from_stats(real_stats, pred_stats):
    if real_stats is None or pred_stats is None or real_stats.count == 0 or pred_stats.count == 0:
        return float("nan")

    mu_real, cov_real = real_stats.mean_and_cov()
    mu_pred, cov_pred = pred_stats.mean_and_cov()
    if mu_real is None or mu_pred is None:
        return float("nan")

    diff = mu_real - mu_pred
    cov_prod = cov_real @ cov_pred
    covmean, _ = linalg.sqrtm(cov_prod, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(cov_real + cov_pred - 2.0 * covmean)
    return float(max(fid, 0.0))


class LPIPSEvaluator:
    def __init__(self, device):
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError(
                "LPIPS requires the `lpips` package. Install it in the eval environment first."
            ) from exc

        self.device = device
        self.model = lpips.LPIPS(net="vgg").to(device)
        self.model.eval()

    def __call__(self, pred_batch, gt_batch):
        pred_batch = pred_batch.to(self.device, non_blocking=True)
        gt_batch = gt_batch.to(self.device, non_blocking=True)
        if min(pred_batch.shape[-2:]) < 64:
            pred_batch = F.interpolate(pred_batch, size=(224, 224), mode="bilinear", align_corners=False)
            gt_batch = F.interpolate(gt_batch, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            values = self.model(pred_batch, gt_batch)
        return values.flatten().detach().cpu().numpy().astype(np.float64)


class FIDFeatureExtractor:
    def __init__(self, device):
        try:
            from torchvision.models import Inception_V3_Weights, inception_v3
        except ImportError as exc:
            raise RuntimeError("FID requires torchvision with InceptionV3 available.") from exc

        self.device = device
        try:
            self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        except Exception as exc:
            raise RuntimeError(
                "FID requires pretrained InceptionV3 weights. Cache/download them in the eval environment first."
            ) from exc
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.model.to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.feature_dim = 2048

    def __call__(self, image_batch):
        image_batch = image_batch.to(self.device, non_blocking=True)
        image_batch = torch.clamp(image_batch * 0.5 + 0.5, 0.0, 1.0)
        image_batch = F.interpolate(image_batch, size=(299, 299), mode="bilinear", align_corners=False)
        image_batch = (image_batch - self.mean) / self.std
        with torch.no_grad():
            feats = self.model(image_batch)
        if isinstance(feats, tuple):
            feats = feats[0]
        return feats.detach().cpu().numpy().astype(np.float64)


def _init_perceptual_metric_totals():
    return {
        "lpips_sum": 0.0,
        "lpips_count": 0.0,
    }


def _update_lpips_metric_totals(metric_totals, mode_names, lpips_values):
    if lpips_values is None:
        return
    for mode_name, lpips_value in zip(mode_names, lpips_values):
        if lpips_value is None or not np.isfinite(lpips_value):
            continue
        totals = metric_totals.setdefault(mode_name, _init_perceptual_metric_totals())
        totals["lpips_sum"] += float(lpips_value)
        totals["lpips_count"] += 1.0


def _update_fid_stats(fid_stats_by_mode, mode_names, gt_features, pred_features):
    for mode_name, gt_feat, pred_feat in zip(mode_names, gt_features, pred_features):
        stats_entry = fid_stats_by_mode.setdefault(
            mode_name,
            {
                "real": RunningStats(gt_features.shape[1]),
                "pred": RunningStats(pred_features.shape[1]),
            },
        )
        stats_entry["real"].update(gt_feat[None, :])
        stats_entry["pred"].update(pred_feat[None, :])


def _merge_perceptual_metrics_into_summary(summary, perceptual_metric_totals, fid_stats_by_mode):
    macro_lpips = []
    weighted_lpips_sum = 0.0
    weighted_lpips_count = 0.0

    for mode_name in summary["modes"].keys():
        totals = perceptual_metric_totals.get(mode_name, _init_perceptual_metric_totals())
        lpips_count = max(0.0, float(totals["lpips_count"]))
        lpips_value = totals["lpips_sum"] / lpips_count if lpips_count > 0 else float("nan")
        if lpips_count > 0:
            macro_lpips.append(lpips_value)
            weighted_lpips_sum += totals["lpips_sum"]
            weighted_lpips_count += lpips_count

        fid_stats = fid_stats_by_mode.get(mode_name)
        fid_value = _frechet_distance_from_stats(fid_stats["real"], fid_stats["pred"]) if fid_stats else float("nan")

        summary["modes"][mode_name]["lpips"] = lpips_value
        summary["modes"][mode_name]["fid"] = fid_value

    overall = summary["overall"]
    overall["macro_lpips"] = _mean_dict_values(macro_lpips)
    overall["weighted_lpips"] = weighted_lpips_sum / weighted_lpips_count if weighted_lpips_count > 0 else float("nan")
    overall["fid"] = _frechet_distance_from_stats(
        fid_stats_by_mode.get("__overall__", {}).get("real"),
        fid_stats_by_mode.get("__overall__", {}).get("pred"),
    )


def _collapse_single_mode_summary(summary):
    mode_names = sorted(summary["modes"].keys())
    if len(mode_names) != 1:
        raise ValueError(
            f"Offline mask-pred eval expects exactly one mode per input dir, found {mode_names}. "
            "Run eval on a single-mode export directory."
        )

    mode_name = mode_names[0]
    mode_summary = dict(summary["modes"][mode_name])
    plain_overall = {
        "mode_name": mode_name,
        "mae": mode_summary.get("mae", float("nan")),
        "mse": mode_summary.get("mse", float("nan")),
        "psnr": mode_summary.get("psnr", float("nan")),
        "ssim": mode_summary.get("ssim", float("nan")),
        "metric": mode_summary.get("metric", float("nan")),
        "count": mode_summary.get("count", float("nan")),
        "video_count": mode_summary.get("video_count", float("nan")),
        "masked_frame_count": summary["overall"].get("masked_frame_count", float("nan")),
        "mode_count": 1,
    }

    if "lpips" in mode_summary:
        plain_overall["lpips"] = mode_summary.get("lpips", float("nan"))
    if "fid" in mode_summary:
        plain_overall["fid"] = mode_summary.get("fid", float("nan"))

    summary["overall"] = plain_overall
    summary["summary_layout"] = "single_mode_flat"
    return summary


def _find_neighbor_indices(visible_indices, target_idx):
    left_idx = None
    right_idx = None
    for idx in visible_indices:
        if idx < target_idx:
            left_idx = idx
        elif idx > target_idx:
            right_idx = idx
            break
    return left_idx, right_idx


def _predict_masked_frame_baseline(gt_video_eval, relative_times, visible_mask, target_idx, method):
    visible_indices = torch.nonzero(visible_mask > 0.5, as_tuple=False).flatten().tolist()
    if not visible_indices:
        return gt_video_eval[target_idx].clone()

    target_time = float(relative_times[target_idx].item())
    if method == "nearest":
        best_idx = min(visible_indices, key=lambda idx: abs(float(relative_times[idx].item()) - target_time))
        return gt_video_eval[best_idx].clone()

    left_idx, right_idx = _find_neighbor_indices(visible_indices, target_idx)
    if left_idx is None or right_idx is None:
        best_idx = min(visible_indices, key=lambda idx: abs(float(relative_times[idx].item()) - target_time))
        return gt_video_eval[best_idx].clone()

    left_time = float(relative_times[left_idx].item())
    right_time = float(relative_times[right_idx].item())
    denom = max(right_time - left_time, 1e-6)
    alpha = float(np.clip((target_time - left_time) / denom, 0.0, 1.0))

    if method == "linear":
        pred_frame = (1.0 - alpha) * gt_video_eval[left_idx] + alpha * gt_video_eval[right_idx]
        return prepare_pred_video_for_eval(pred_frame.unsqueeze(0), force_grayscale=True)[0]

    pred_frame = predict_intermediate_frame_dis_flow(
        gt_video_eval[left_idx],
        gt_video_eval[right_idx],
        alpha=alpha,
        method=method,
    )
    return prepare_pred_video_for_eval(pred_frame.unsqueeze(0), force_grayscale=True)[0]


def _predict_masked_sequence_baseline(gt_video_eval_sample, relative_times_sample, frame_mask_sample, valid_mask_sample, method):
    pred_video_eval = gt_video_eval_sample.clone()
    visible_mask = (valid_mask_sample > 0.5) & (frame_mask_sample <= 0.5)
    masked_indices = torch.nonzero(frame_mask_sample > 0.5, as_tuple=False).flatten().tolist()
    for target_idx in masked_indices:
        pred_video_eval[target_idx] = _predict_masked_frame_baseline(
            gt_video_eval=gt_video_eval_sample,
            relative_times=relative_times_sample,
            visible_mask=visible_mask,
            target_idx=target_idx,
            method=method,
        )
    return pred_video_eval


def _tensor_frame_to_pil(frame_tensor):
    frame = frame_tensor.detach().cpu().float()
    if frame.ndim != 3:
        raise ValueError(f"Expected frame tensor [C, H, W], got {tuple(frame.shape)}")
    if frame.shape[0] == 1:
        frame = frame.repeat(3, 1, 1)
    elif frame.shape[0] > 3:
        frame = frame[:3]
    frame = torch.clamp(frame, -1.0, 1.0)
    image = ((frame * 0.5 + 0.5) * 255.0).round().to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(image)


def _build_reconstruct_overview_figure(columns, title_text):
    if not columns:
        raise ValueError("No columns available for reconstruct overview figure")

    first_gt = columns[0]["gt_frame"]
    frame_height = int(first_gt.shape[1])
    frame_width = int(first_gt.shape[2])
    header_h = 26
    title_h = 28
    row_gap = 8
    col_gap = 8
    margin = 10
    label_w = 56
    rows = 2
    total_width = label_w + margin * 2 + len(columns) * frame_width + max(0, len(columns) - 1) * col_gap
    total_height = (
        margin * 2
        + title_h
        + header_h
        + rows * frame_height
        + row_gap
    )
    canvas = Image.new("RGB", (total_width, total_height), color=(10, 10, 10))
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, margin), title_text, fill=(255, 255, 255))
    gt_y = margin + title_h + header_h
    pred_y = gt_y + frame_height
    draw.text((margin, gt_y + frame_height // 2 - 6), "GT", fill=(220, 220, 220))
    draw.text((margin, pred_y + frame_height // 2 - 6), "Pred", fill=(220, 220, 220))

    for col_idx, column in enumerate(columns):
        x = margin + label_w + col_idx * (frame_width + col_gap)
        header_text = column["label"]
        draw.text((x, margin + title_h), header_text, fill=(255, 255, 255))
        canvas.paste(_tensor_frame_to_pil(column["gt_frame"]), (x, gt_y))
        canvas.paste(_tensor_frame_to_pil(column["pred_frame"]), (x, pred_y))

    return canvas


def export_reconstruct_visualizations(input_dir, output_dir, progress=None, load_task=None, export_task=None):
    manifest = load_manifest(input_dir)
    if manifest is not None:
        if manifest.get("task") != "mask_pred":
            raise ValueError(f"Input dir {input_dir} is not a mask-pred prediction set.")
        if manifest.get("infer_mode") != "reconstruct":
            raise ValueError("Visualization is only supported for `infer --task mask_pred --mode reconstruct` outputs.")

    prediction_files = list_prediction_files(input_dir)
    if not prediction_files:
        raise FileNotFoundError(f"No prediction files found under {input_dir}")

    grouped_records = {}
    loaded_records = 0
    for prediction_path in prediction_files:
        record = torch.load(prediction_path, map_location="cpu")
        if record.get("task") != "mask_pred" or record.get("infer_mode") != "reconstruct":
            continue
        group_key = (
            int(record.get("dataset_index", -1)),
            str(record.get("video_name", "")),
            str(record.get("mode_name", "")),
            str(record.get("baseline_method", "")),
        )
        grouped_records.setdefault(group_key, []).append(record)
        loaded_records += 1
        if progress is not None and load_task is not None:
            progress.update(load_task, advance=1, status=f"loaded {loaded_records}")

    os.makedirs(output_dir, exist_ok=True)
    overview_dir = os.path.join(output_dir, "overview")
    os.makedirs(overview_dir, exist_ok=True)

    manifest_payload = {
        "task": "mask_pred",
        "phase": "visualize",
        "input_dir": input_dir,
        "figure_count": 0,
        "groups": [],
    }

    for (dataset_index, video_name, mode_name, baseline_method), records in sorted(grouped_records.items()):
        columns = []
        for record in records:
            target_indices = [int(idx) for idx in record["target_frame_indices"]]
            gt_targets = record["gt_target_frames"].float()
            pred_targets = record["pred_target_frames"].float()
            relative_times = record["target_frame_times_relative"].float()
            variant_name = str(record.get("mask_variant", "variant"))
            for column_offset, target_idx in enumerate(target_indices):
                time_value = float(relative_times[column_offset].item())
                columns.append(
                    {
                        "frame_idx": int(target_idx),
                        "label": f"f{int(target_idx):02d} t={time_value:.2f}",
                        "variant": variant_name,
                        "gt_frame": gt_targets[column_offset],
                        "pred_frame": pred_targets[column_offset],
                    }
                )

        if not columns:
            continue

        columns.sort(key=lambda item: (item["frame_idx"], item["variant"]))
        title_parts = [f"{video_name}", mode_name]
        if baseline_method:
            title_parts.append(f"baseline={baseline_method}")
        title_text = " | ".join(title_parts)
        figure = _build_reconstruct_overview_figure(columns, title_text=title_text)
        base_name = build_sample_key(f"idx_{dataset_index:06d}", video_name, mode_name, baseline_method or "model")
        image_path = os.path.join(overview_dir, f"{base_name}.png")
        json_path = os.path.join(overview_dir, f"{base_name}.json")
        figure.save(image_path)
        write_json(
            json_path,
            {
                "dataset_index": dataset_index,
                "video_name": video_name,
                "mode_name": mode_name,
                "baseline_method": baseline_method or None,
                "columns": [
                    {
                        "frame_index": int(column["frame_idx"]),
                        "variant": column["variant"],
                        "label": column["label"],
                    }
                    for column in columns
                ],
            },
        )
        manifest_payload["groups"].append(
            {
                "dataset_index": dataset_index,
                "video_name": video_name,
                "mode_name": mode_name,
                "baseline_method": baseline_method or None,
                "image": image_path,
                "metadata": json_path,
                "column_count": len(columns),
            }
        )
        manifest_payload["figure_count"] += 1
        if progress is not None and export_task is not None:
            progress.update(export_task, advance=1, status=f"figures {manifest_payload['figure_count']}")

    write_json(os.path.join(output_dir, "visualization_manifest.json"), manifest_payload)
    return manifest_payload


@torch.no_grad()
def save_reconstruct_predictions(
    args,
    data_loader,
    vae,
    diffusion,
    eval_model,
    device,
    output_dir,
    max_batches=0,
    progress=None,
    progress_task=None,
):
    mask_cfg = clone_mask_cfg(args.mask_prediction, args.mask_mode)
    mode_names = get_validation_modes(mask_cfg)
    min_visible = int(mask_cfg.get("min_visible_frames", 2))
    span_length = int(mask_cfg.get("validation_span_length", mask_cfg.get("max_masked_frames", 2)))
    saved_records = 0
    baseline_method = getattr(args, "baseline_method", None)

    for batch_idx, batch in enumerate(data_loader):
        if max_batches and batch_idx >= max_batches:
            break

        video = batch["video"].to(device, non_blocking=True)
        valid_mask = batch["frame_valid_mask"].to(device, non_blocking=True)
        dataset_indices = batch["dataset_index"].tolist()
        video_names = batch["video_name"]
        video_paths = batch["video_path"]
        frame_times_relative = batch["frame_times_relative"].detach().cpu()
        frame_times_normalized = batch["frame_times_normalized"].detach().cpu()
        batch_size = video.shape[0]
        gt_video_eval = prepare_gt_video_for_eval(video.detach().cpu())
        valid_mask_cpu = valid_mask.detach().cpu()

        latent = None
        frame_times = None
        if baseline_method is None:
            frame_times = batch["frame_times"].to(device, non_blocking=True)
            video_flat = video.reshape(-1, *video.shape[2:])
            latent = encode_image_batch_to_latent(
                vae,
                video_flat,
                use_amp=bool(args.mixed_precision),
                chunk_size=int(getattr(args, "vae_encode_batch_size", 0)),
            )
            latent = latent.reshape(batch_size, video.shape[1], *latent.shape[1:])

        for mode_name in mode_names:
            frame_mask_variants = build_validation_frame_mask_variants(
                valid_mask=valid_mask,
                mode=mode_name,
                min_visible=min_visible,
                span_length=span_length,
                device=device,
            )
            for variant_name, frame_mask in frame_mask_variants:
                frame_mask_cpu = frame_mask.detach().cpu()
                if baseline_method is None:
                    reconstructed_latent = reconstruct_masked_sequence(
                        model_forward=eval_model.forward,
                        diffusion=diffusion,
                        latent_sequence=latent,
                        frame_times=frame_times,
                        frame_mask=frame_mask,
                        device=device,
                        use_amp=bool(args.mixed_precision),
                    )
                    decoded_video = decode_latent_sequence(
                        vae,
                        reconstructed_latent,
                        use_amp=bool(args.mixed_precision),
                        decode_batch_size=int(getattr(args, "vae_decode_batch_size", 0)),
                    )
                    pred_video_eval = prepare_pred_video_for_eval(decoded_video.detach().cpu(), force_grayscale=True)
                else:
                    pred_video_eval = torch.stack(
                        [
                            _predict_masked_sequence_baseline(
                                gt_video_eval_sample=gt_video_eval[sample_offset],
                                relative_times_sample=frame_times_relative[sample_offset],
                                frame_mask_sample=frame_mask_cpu[sample_offset],
                                valid_mask_sample=valid_mask_cpu[sample_offset],
                                method=baseline_method,
                            )
                            for sample_offset in range(batch_size)
                        ],
                        dim=0,
                    )

                for sample_offset, dataset_index in enumerate(dataset_indices):
                    masked_indices = torch.nonzero(frame_mask_cpu[sample_offset] > 0.5, as_tuple=False).flatten().tolist()
                    if not masked_indices:
                        continue

                    sample_key = build_sample_key(
                        f"idx_{int(dataset_index):06d}",
                        video_names[sample_offset],
                        canonicalize_validation_mode(mode_name),
                        variant_name,
                    )
                    save_prediction_record(
                        output_dir,
                        sample_key,
                        {
                            "task": "mask_pred",
                            "infer_mode": "reconstruct",
                            "stage": args.stage,
                            "mode_name": canonicalize_validation_mode(mode_name),
                            "mask_variant": variant_name,
                            "video_name": video_names[sample_offset],
                            "video_path": video_paths[sample_offset],
                            "dataset_index": int(dataset_index),
                            "baseline_method": baseline_method,
                            "target_frame_indices": masked_indices,
                            "target_frame_times_relative": frame_times_relative[sample_offset, masked_indices].to(torch.float32),
                            "target_frame_times_normalized": frame_times_normalized[sample_offset, masked_indices].to(torch.float32),
                            "gt_target_frames": gt_video_eval[sample_offset, masked_indices].to(torch.float16),
                            "pred_target_frames": pred_video_eval[sample_offset, masked_indices].to(torch.float16),
                        },
                    )
                    saved_records += 1

        del batch, video, valid_mask, latent
        if progress is not None and progress_task is not None:
            progress.update(progress_task, advance=1, status=f"saved {saved_records}")

    return saved_records


@torch.no_grad()
def export_densified_sequences(
    args,
    data_loader,
    eval_model,
    vae,
    device,
    output_dir,
    max_samples=0,
    progress=None,
    progress_task=None,
):
    eval_model.eval()
    time_field = str(args.mask_prediction.get("time_field", "normalized_time"))
    saved_samples = 0

    for batch in data_loader:
        video = batch["video"].to(device, non_blocking=True)
        valid_mask = batch["frame_valid_mask"].to(device, non_blocking=True)
        relative_times = batch["frame_times_relative"].to(device, non_blocking=True)
        video_names = batch["video_name"]
        video_paths = batch["video_path"]
        dataset_indices = batch["dataset_index"].tolist()

        video_flat = video.reshape(-1, *video.shape[2:])
        latent = encode_image_batch_to_latent(
            vae,
            video_flat,
            use_amp=bool(args.mixed_precision),
            chunk_size=int(getattr(args, "vae_encode_batch_size", 0)),
        )
        latent = latent.reshape(video.shape[0], video.shape[1], *latent.shape[1:])

        for batch_idx in range(video.shape[0]):
            if max_samples > 0 and saved_samples >= max_samples:
                return saved_samples

            valid_indices = torch.nonzero(valid_mask[batch_idx] > 0.5, as_tuple=False).flatten()
            observed_video = video[batch_idx : batch_idx + 1, valid_indices]
            observed_latent = latent[batch_idx : batch_idx + 1, valid_indices]
            observed_rel = relative_times[batch_idx, valid_indices].detach().cpu().numpy().astype(np.float32)

            merged_rel, observed_slot_mask = build_densified_timeline(observed_rel, fps=args.densify_fps)
            merged_times = convert_relative_times(merged_rel, time_field=time_field)

            total_frames = len(merged_rel)
            combined_latent = torch.zeros(
                1,
                total_frames,
                observed_latent.shape[2],
                observed_latent.shape[3],
                observed_latent.shape[4],
                device=device,
                dtype=observed_latent.dtype,
            )
            observed_positions = np.where(observed_slot_mask > 0.5)[0]
            combined_latent[:, observed_positions] = observed_latent

            frame_times = torch.from_numpy(merged_times).to(device=device, dtype=torch.float32).unsqueeze(0)
            query_mask = torch.from_numpy(1.0 - observed_slot_mask).to(device=device, dtype=torch.float32).unsqueeze(0)

            diffusion = create_diffusion(
                timestep_respacing=getattr(args, "timestep_respacing_test", "ddim50"),
                diffusion_steps=args.diffusion_steps,
                learn_sigma=args.learn_sigma,
                num_frames=total_frames,
            )
            diffusion.training = False

            reconstructed = reconstruct_masked_sequence(
                model_forward=eval_model.forward,
                diffusion=diffusion,
                latent_sequence=combined_latent,
                frame_times=frame_times,
                frame_mask=query_mask,
                device=device,
                use_amp=bool(args.mixed_precision),
            )

            decoded = decode_latent_batch_to_image(
                vae,
                reconstructed.reshape(-1, *reconstructed.shape[2:]),
                use_amp=bool(args.mixed_precision),
                chunk_size=int(getattr(args, "vae_decode_batch_size", 0)),
            )
            decoded = decoded.reshape(1, total_frames, *decoded.shape[1:])
            pred_video_eval = prepare_pred_video_for_eval(decoded.detach().cpu(), force_grayscale=True)
            observed_gt = prepare_gt_video_for_eval(observed_video.detach().cpu())
            sample_key = build_sample_key(
                f"idx_{int(dataset_indices[batch_idx]):06d}",
                video_names[batch_idx],
                "densify",
            )
            save_prediction_record(
                output_dir,
                sample_key,
                {
                    "task": "mask_pred",
                    "infer_mode": "densify",
                    "stage": args.stage,
                    "video_name": video_names[batch_idx],
                    "video_path": video_paths[batch_idx],
                    "dataset_index": int(dataset_indices[batch_idx]),
                    "time_field": time_field,
                    "densify_fps": float(args.densify_fps),
                    "observed_relative_times": observed_rel.tolist(),
                    "merged_relative_times": merged_rel.tolist(),
                    "observed_positions": observed_positions.tolist(),
                    "query_positions": np.where(observed_slot_mask < 0.5)[0].tolist(),
                    "observed_video": observed_gt[0].to(torch.float16),
                    "pred_video": pred_video_eval[0].to(torch.float16),
                },
            )

            observed_row = torch.full_like(pred_video_eval, -1.0)
            observed_row[:, observed_positions] = observed_gt
            mask_row = (
                torch.from_numpy((1.0 - observed_slot_mask)[None, :, None, None, None]).float().expand_as(pred_video_eval) * 2
                - 1
            )
            export_image = torch.cat([observed_row, pred_video_eval, mask_row], dim=1).reshape(-1, *pred_video_eval.shape[2:])
            image_path = os.path.join(output_dir, f"{sample_key}.png")
            save_image(export_image, image_path, nrow=total_frames, normalize=True, value_range=(-1, 1))
            write_json(
                os.path.join(output_dir, f"{sample_key}.json"),
                {
                    "video_name": video_names[batch_idx],
                    "dataset_index": int(dataset_indices[batch_idx]),
                    "time_field": time_field,
                    "densify_fps": float(args.densify_fps),
                    "observed_relative_times": observed_rel.tolist(),
                    "merged_relative_times": merged_rel.tolist(),
                    "observed_positions": observed_positions.tolist(),
                    "query_positions": np.where(observed_slot_mask < 0.5)[0].tolist(),
                },
            )
            saved_samples += 1
            if progress is not None and progress_task is not None:
                progress.update(progress_task, advance=1, status=f"saved {saved_samples}")

    return saved_samples


def run_infer(args):
    baseline_method = getattr(args, "baseline_method", None)
    requested_gpu = getattr(args, "test_gpu_id", None) or getattr(args, "gpu_id", None)
    is_distributed = "RANK" in os.environ or "WORLD_SIZE" in os.environ or "LOCAL_RANK" in os.environ
    if baseline_method:
        if is_distributed and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("mask-pred baselines use a CPU-only single-process path. Run them without torchrun/DDP.")
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device("cpu")
    else:
        if requested_gpu and not is_distributed:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(requested_gpu)

        rank, local_rank, world_size = setup_distributed(timeout_minutes=int(getattr(args, "ddp_timeout_minutes", 180)))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)

    seed = int(getattr(args, "global_seed", 3407)) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    experiment_dir = resolve_experiment_dir(args)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    output_dir = args.output_dir or build_default_output_dir(args)
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    distributed_barrier()

    logger = create_logger_compat(output_dir if rank == 0 else None, level=getattr(args, "log_level", "INFO"))
    sequence_length = int(args.mask_prediction.get("sequence_length", getattr(args, "tar_num_frames", 15)))
    args.tar_num_frames = sequence_length

    if baseline_method and args.mode != "reconstruct":
        raise ValueError("mask-pred baselines are only supported for `--mode reconstruct`.")
    if rank == 0 and baseline_method:
        logger.info(f"Running mask-pred baseline `{baseline_method}` on CPU-only path.")

    ckpt_path = None
    use_ema = False
    eval_model = None
    vae = None
    diffusion = None
    checkpoint = {}
    if baseline_method is None:
        ckpt_path = resolve_eval_checkpoint_path(args, checkpoint_dir)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        latent_size = args.image_size // 8
        model = MVIF_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
            learn_sigma=bool(getattr(args, "learn_sigma", True)),
            mode="video",
            num_frames=sequence_length,
        ).to(device, non_blocking=True)
        ema = MVIF_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
            learn_sigma=bool(getattr(args, "learn_sigma", True)),
            mode="video",
            num_frames=sequence_length,
        ).to(device, non_blocking=True)
        requires_grad(model, False)
        requires_grad(ema, False)
        model.eval()
        ema.eval()

        vae = AutoencoderKL.from_pretrained(
            args.pretrained_vae_model_path,
            subfolder="sd-vae-ft-mse",
        ).to(device, non_blocking=True)
        vae.requires_grad_(False)
        vae.eval()

        checkpoint = safe_load_checkpoint(ckpt_path, device)
        get_raw_model(model).load_state_dict(checkpoint["model"])
        if checkpoint.get("ema") is not None:
            ema.load_state_dict(checkpoint["ema"])
        use_ema = should_use_ema(args, checkpoint) and checkpoint.get("ema") is not None
        eval_model = ema if use_ema else model

    dataset = mask_prediction_data_loader(args, stage=args.stage)
    sampler = DistributedEvalSampler(dataset, num_replicas=world_size, rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_mask_prediction_batch,
    )
    progress = None
    progress_task = None
    if rank == 0:
        progress = create_rich_progress()
        progress.start()
        if args.mode == "reconstruct":
            total_units = len(data_loader)
            if int(args.max_batches) > 0:
                total_units = min(total_units, int(args.max_batches))
            progress_task = progress.add_task(
                f"{args.stage} reconstruct infer",
                total=max(1, total_units),
                completed=0,
                status="saved 0",
            )
        else:
            total_samples = len(dataset)
            if hasattr(sampler, "__len__"):
                total_samples = len(sampler)
            if int(args.max_samples) > 0:
                total_samples = min(total_samples, int(args.max_samples))
            progress_task = progress.add_task(
                f"{args.stage} densify infer",
                total=max(1, total_samples),
                completed=0,
                status="saved 0",
            )

    try:
        if rank == 0:
            write_manifest(
                output_dir,
                {
                    "task": "mask_pred",
                    "phase": "infer",
                    "infer_mode": args.mode,
                    "stage": args.stage,
                    "mask_mode": args.mask_mode,
                    "baseline_method": baseline_method,
                    "checkpoint": ckpt_path,
                    "use_ema": bool(use_ema),
                    "world_size": world_size,
                    "densify_fps": float(args.densify_fps),
                },
            )

        if args.mode == "reconstruct":
            mask_cfg = clone_mask_cfg(args.mask_prediction, args.mask_mode)
            if baseline_method is None:
                diffusion = create_diffusion(
                    timestep_respacing=getattr(args, "timestep_respacing_test", "ddim50"),
                    diffusion_steps=args.diffusion_steps,
                    learn_sigma=args.learn_sigma,
                    num_frames=sequence_length,
                )
                diffusion.training = False
            local_saved = save_reconstruct_predictions(
                args=args,
                data_loader=data_loader,
                vae=vae,
                diffusion=diffusion,
                eval_model=eval_model,
                device=device,
                output_dir=output_dir,
                max_batches=int(args.max_batches),
                progress=progress,
                progress_task=progress_task,
            )
            if rank == 0 and int(args.num_visualizations) > 0 and baseline_method is None:
                sample_indices = get_fixed_visualization_indices(len(dataset), int(args.num_visualizations))
                if sample_indices:
                    save_mask_prediction_visualizations(
                        dataset=dataset,
                        output_dir=os.path.join(output_dir, "visualizations"),
                        sample_indices=sample_indices,
                        batch_size=max(1, min(args.val_visualization_batch_size, args.num_visualizations)),
                        model_forward=eval_model.forward,
                        vae=vae,
                        diffusion=diffusion,
                        device=device,
                        mask_cfg=mask_cfg,
                        epoch=0,
                        train_steps=int(checkpoint.get("train_steps", 0)),
                        use_amp=bool(args.mixed_precision),
                        vae_encode_batch_size=int(getattr(args, "vae_encode_batch_size", 0)),
                        vae_decode_batch_size=int(getattr(args, "vae_decode_batch_size", 0)),
                    )
        else:
            local_saved = export_densified_sequences(
                args=args,
                data_loader=data_loader,
                eval_model=eval_model,
                vae=vae,
                device=device,
                output_dir=output_dir,
                max_samples=int(args.max_samples),
                progress=progress,
                progress_task=progress_task,
            )
    finally:
        if progress is not None:
            progress.stop()

    saved_tensor = torch.tensor([local_saved], dtype=torch.long, device=device)
    if world_size > 1:
        dist.all_reduce(saved_tensor, op=dist.ReduceOp.SUM)
    distributed_barrier()
    if rank == 0:
        logger.info(f"Mask-pred {args.mode} outputs: {int(saved_tensor.item())} records -> {output_dir}")

    cleanup()


def run_eval(args):
    manifest = load_manifest(args.input_dir)
    if manifest is not None:
        if manifest.get("task") != "mask_pred":
            raise ValueError(f"Input dir {args.input_dir} is not a mask-pred prediction set.")
        if manifest.get("infer_mode") == "densify":
            raise ValueError("Offline eval is only supported for `infer --task mask_pred --mode reconstruct` outputs.")

    prediction_files = list_prediction_files(args.input_dir)
    if not prediction_files:
        raise FileNotFoundError(f"No prediction files found under {args.input_dir}")

    metric_totals = {}
    record_count = 0
    roi_mask_enabled = bool(args.roi_mask_root)
    perceptual_metric_totals = {}
    fid_stats_by_mode = {}
    perceptual_pred_buffer = []
    perceptual_gt_buffer = []
    perceptual_mode_buffer = []
    perceptual_buffer_count = 0
    lpips_evaluator = None
    fid_extractor = None
    metric_warnings = []
    perceptual_batch_size = max(1, int(getattr(args, "perceptual_batch_size", 32)))
    perceptual_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        lpips_evaluator = LPIPSEvaluator(perceptual_device)
    except RuntimeError as exc:
        metric_warnings.append(str(exc))
    try:
        fid_extractor = FIDFeatureExtractor(perceptual_device)
    except RuntimeError as exc:
        metric_warnings.append(str(exc))
    if fid_extractor is not None:
        fid_stats_by_mode["__overall__"] = {
            "real": RunningStats(fid_extractor.feature_dim),
            "pred": RunningStats(fid_extractor.feature_dim),
        }

    def flush_perceptual_metric_buffer():
        nonlocal perceptual_pred_buffer, perceptual_gt_buffer, perceptual_mode_buffer, perceptual_buffer_count
        if not perceptual_pred_buffer:
            return

        pred_batch = torch.cat(perceptual_pred_buffer, dim=0)
        gt_batch = torch.cat(perceptual_gt_buffer, dim=0)
        lpips_values = lpips_evaluator(pred_batch, gt_batch).tolist() if lpips_evaluator is not None else None
        _update_lpips_metric_totals(
            perceptual_metric_totals,
            perceptual_mode_buffer,
            lpips_values,
        )

        if fid_extractor is not None:
            pred_features = fid_extractor(pred_batch)
            gt_features = fid_extractor(gt_batch)
            _update_fid_stats(fid_stats_by_mode, perceptual_mode_buffer, gt_features, pred_features)
            fid_stats_by_mode["__overall__"]["real"].update(gt_features)
            fid_stats_by_mode["__overall__"]["pred"].update(pred_features)

        perceptual_pred_buffer = []
        perceptual_gt_buffer = []
        perceptual_mode_buffer = []
        perceptual_buffer_count = 0

    progress = create_rich_progress()
    progress.start()
    eval_task = progress.add_task("mask_pred eval", total=len(prediction_files), completed=0, status="records 0")
    try:
        for prediction_path in prediction_files:
            record = torch.load(prediction_path, map_location="cpu")
            if record.get("task") != "mask_pred":
                raise ValueError(f"Unexpected record type in {prediction_path}")
            if record.get("infer_mode") != "reconstruct":
                raise ValueError("Offline eval is only supported for reconstruct outputs.")

            mode_name = str(record["mode_name"])
            metric_totals.setdefault(
                mode_name,
                {
                    "mae_sum": 0.0,
                    "mse_sum": 0.0,
                    "psnr_sum": 0.0,
                    "ssim_sum": 0.0,
                    "count": 0.0,
                    "video_count": 0.0,
                },
            )
            pred_targets = record["pred_target_frames"].float()
            gt_targets = record["gt_target_frames"].float()
            target_count = int(gt_targets.shape[0])
            if target_count <= 0:
                progress.update(eval_task, advance=1, status=f"records {record_count}")
                continue

            spatial_mask = None
            roi_mask_path = None
            if roi_mask_enabled:
                roi_mask_path = _resolve_record_roi_mask_path(record, args.roi_mask_root)
                if not os.path.exists(roi_mask_path):
                    raise FileNotFoundError(f"ROI mask not found for record {prediction_path}: {roi_mask_path}")
                roi_mask_np = _load_union_roi_mask(roi_mask_path)
                spatial_mask = torch.from_numpy(roi_mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)

            ones_mask = torch.ones(1, target_count, dtype=torch.float32)
            batch_metrics = compute_masked_frame_metrics(
                pred_video=pred_targets.unsqueeze(0),
                gt_video=gt_targets.unsqueeze(0),
                frame_mask=ones_mask,
                valid_mask=ones_mask,
                spatial_mask=spatial_mask,
            )
            for key, value in batch_metrics.items():
                metric_totals[mode_name][key] += value

            perceptual_pred_buffer.append(pred_targets)
            perceptual_gt_buffer.append(gt_targets)
            perceptual_mode_buffer.extend([mode_name] * pred_targets.shape[0])
            perceptual_buffer_count += int(pred_targets.shape[0])
            if perceptual_buffer_count >= perceptual_batch_size:
                flush_perceptual_metric_buffer()

            record_count += 1
            progress.update(eval_task, advance=1, status=f"records {record_count}")
    finally:
        flush_perceptual_metric_buffer()
        progress.stop()

    summary = summarize_validation_metrics(metric_totals)
    _merge_perceptual_metrics_into_summary(summary, perceptual_metric_totals, fid_stats_by_mode)
    summary = _collapse_single_mode_summary(summary)
    summary["task"] = "mask_pred"
    summary["input_dir"] = args.input_dir
    summary["records"] = record_count
    summary["metric_scope"] = {
        "roi_mask_enabled": roi_mask_enabled,
        "roi_mask_root": args.roi_mask_root,
        "pixel_metrics": "union-mask ROI only" if roi_mask_enabled else "full frame",
        "ssim": "full frame",
        "lpips": "full frame",
        "fid": "full frame",
        "psnr_definition": "dataset-level PSNR from summary MSE: 10 * log10((data_range^2) / mse)",
    }
    if metric_warnings:
        summary["metric_warnings"] = metric_warnings

    output_dir = args.output_dir or args.input_dir
    summary_path = os.path.join(output_dir, "eval_summary.json")
    write_json(summary_path, summary)

    overall = summary["overall"]
    print("Mask-pred offline eval summary")
    print(f"input_dir: {args.input_dir}")
    print(f"records: {record_count}")
    print(f"pixel_metrics: {summary['metric_scope']['pixel_metrics']}")
    print(f"ssim: {summary['metric_scope']['ssim']}")
    print(f"mode_name: {overall['mode_name']}")
    print(f"metric: {overall['metric']:.6f}")
    print(f"mae: {overall['mae']:.6f}")
    print(f"mse: {overall['mse']:.6f}")
    print(f"psnr: {overall['psnr']:.4f}")
    print(f"ssim: {overall['ssim']:.4f}")
    print(f"lpips: {overall.get('lpips', float('nan')):.4f}")
    print(f"fid: {overall.get('fid', float('nan')):.4f}")
    for warning in metric_warnings:
        print(f"warning: {warning}")
    print(f"summary: {summary_path}")


def run_visualize(args):
    output_dir = args.output_dir or os.path.join(args.input_dir, "figures")
    manifest = load_manifest(args.input_dir)
    if manifest is not None:
        if manifest.get("task") != "mask_pred":
            raise ValueError(f"Input dir {args.input_dir} is not a mask-pred prediction set.")
        if manifest.get("infer_mode") != "reconstruct":
            raise ValueError("Visualization is only supported for `infer --task mask_pred --mode reconstruct` outputs.")

    prediction_files = list_prediction_files(args.input_dir)
    if not prediction_files:
        raise FileNotFoundError(f"No prediction files found under {args.input_dir}")

    progress = create_rich_progress()
    progress.start()
    load_task = progress.add_task("mask_pred visualize load", total=len(prediction_files), completed=0, status="loaded 0")
    export_task = progress.add_task("mask_pred visualize export", total=None, completed=0, status="figures 0")
    try:
        summary = export_reconstruct_visualizations(
            args.input_dir,
            output_dir,
            progress=progress,
            load_task=load_task,
            export_task=export_task,
        )
        progress.update(export_task, total=max(1, summary["figure_count"]), completed=summary["figure_count"], status=f"figures {summary['figure_count']}")
    finally:
        progress.stop()
    print("Mask-pred visualization export summary")
    print(f"input_dir: {args.input_dir}")
    print(f"figures: {summary['figure_count']}")
    print(f"output_dir: {output_dir}")
    print(f"manifest: {os.path.join(output_dir, 'visualization_manifest.json')}")


def run_infer_cli(argv=None):
    parser = build_infer_parser()
    cli_args = parser.parse_args(argv)
    config = load_config_from_parsed_args(cli_args)
    config = apply_mask_pred_overrides(config, cli_args)
    config = apply_infer_overrides(config, cli_args)
    run_infer(config)


def run_eval_cli(argv=None):
    parser = build_eval_parser()
    cli_args = parser.parse_args(argv)
    run_eval(cli_args)


def run_visualize_cli(argv=None):
    parser = build_visualize_parser()
    cli_args = parser.parse_args(argv)
    run_visualize(cli_args)
