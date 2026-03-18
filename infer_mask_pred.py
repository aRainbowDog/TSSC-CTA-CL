import json
import math
import os
import random
import warnings

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf
from torchvision.utils import save_image

from dataloader.data_loader_mask import collate_mask_prediction_batch, mask_prediction_data_loader
from models.diffusion.gaussian_diffusion import create_diffusion
from models.model_dit import MVIF_models
from train_mask_pred import apply_mask_pred_overrides, build_parser
from training.checkpointing import resolve_experiment_dir, safe_load_checkpoint
from training.common import create_logger_compat, distributed_barrier, parse_train_config
from training.latent_utils import decode_latent_batch_to_image, encode_image_batch_to_latent
from training.runtime import get_raw_model
from training.validation_mask_pred import (
    DistributedEvalSampler,
    evaluate_mask_prediction,
    get_fixed_visualization_indices,
    reconstruct_masked_sequence,
    save_mask_prediction_visualizations,
)
from utils.triplet_eval import prepare_gt_video_for_eval, prepare_pred_video_for_eval
from utils.utils import cleanup, requires_grad, setup_distributed

warnings.filterwarnings("ignore")


def build_infer_parser():
    parser = build_parser()
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path. Defaults to best_mask_pred or latest_mask_pred.")
    parser.add_argument("--stage", type=str, default="val", choices=["val", "test"], help="Dataset split used for evaluation/export.")
    parser.add_argument(
        "--mode",
        type=str,
        default="reconstruct",
        choices=["reconstruct", "densify"],
        help="Reconstruct masked observed frames or export densified sequences.",
    )
    parser.add_argument(
        "--mask-mode",
        type=str,
        default="all",
        choices=["all", "single", "span", "anchor"],
        help="Validation mask mode for reconstruction evaluation.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for evaluation outputs.")
    parser.add_argument("--max-batches", type=int, default=0, help="Optional cap for reconstruction evaluation batches.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for densification exports per rank.")
    parser.add_argument("--num-visualizations", type=int, default=8, help="How many qualitative samples to export on rank 0.")
    parser.add_argument("--use-ema", type=str, default="auto", choices=["auto", "true", "false"], help="Whether to prefer EMA weights.")
    parser.add_argument("--densify-fps", type=float, default=1.0, help="Target FPS for densification export.")
    return parser


def resolve_eval_checkpoint_path(args, checkpoint_dir):
    if args.ckpt:
        return args.ckpt
    if getattr(args, "test_ckpt", None):
        return args.test_ckpt
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


def build_densified_timeline(relative_times, fps):
    observed = np.asarray(relative_times, dtype=np.float32)
    if observed.size == 0:
        raise ValueError("Observed timeline is empty")
    step = 1.0 / max(float(fps), 1e-6)
    grid = np.arange(observed[0], observed[-1] + step * 0.5, step, dtype=np.float32)
    merged = np.concatenate([observed, grid], axis=0)
    merged = np.unique(np.round(merged.astype(np.float64), 5)).astype(np.float32)
    merged.sort()
    observed_mask = np.array([np.any(np.isclose(observed, time_point, atol=1e-4)) for time_point in merged], dtype=np.float32)
    return merged, observed_mask


def convert_relative_times(relative_times, time_field):
    if time_field == "relative_time_seconds":
        return relative_times
    start_time = float(relative_times[0])
    end_time = float(relative_times[-1])
    denom = max(end_time - start_time, 1e-6)
    return ((relative_times - start_time) / denom).astype(np.float32)


@torch.no_grad()
def export_densified_sequences(
    args,
    data_loader,
    eval_model,
    vae,
    device,
    output_dir,
    max_samples=0,
):
    eval_model.eval()
    time_field = str(args.mask_prediction.get("time_field", "normalized_time"))
    saved_samples = 0

    for batch in data_loader:
        video = batch["video"].to(device, non_blocking=True)
        valid_mask = batch["frame_valid_mask"].to(device, non_blocking=True)
        relative_times = batch["frame_times_relative"].to(device, non_blocking=True)
        video_names = batch["video_name"]
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
                return

            valid_indices = torch.nonzero(valid_mask[batch_idx] > 0.5, as_tuple=False).flatten()
            observed_video = video[batch_idx:batch_idx + 1, valid_indices]
            observed_latent = latent[batch_idx:batch_idx + 1, valid_indices]
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

            observed_row = torch.full_like(pred_video_eval, -1.0)
            observed_gt = prepare_gt_video_for_eval(observed_video.detach().cpu())
            observed_row[:, observed_positions] = observed_gt
            mask_row = torch.from_numpy((1.0 - observed_slot_mask)[None, :, None, None, None]).float().expand_as(pred_video_eval) * 2 - 1

            export_image = torch.cat([observed_row, pred_video_eval, mask_row], dim=1).reshape(-1, *pred_video_eval.shape[2:])
            image_path = os.path.join(output_dir, f"{video_names[batch_idx]}_idx_{dataset_indices[batch_idx]:04d}_densify.png")
            save_image(export_image, image_path, nrow=total_frames, normalize=True, value_range=(-1, 1))

            metadata = {
                "video_name": video_names[batch_idx],
                "dataset_index": int(dataset_indices[batch_idx]),
                "time_field": time_field,
                "densify_fps": float(args.densify_fps),
                "observed_relative_times": observed_rel.tolist(),
                "merged_relative_times": merged_rel.tolist(),
                "observed_positions": observed_positions.tolist(),
                "query_positions": np.where(observed_slot_mask < 0.5)[0].tolist(),
            }
            meta_path = os.path.join(output_dir, f"{video_names[batch_idx]}_idx_{dataset_indices[batch_idx]:04d}_densify.json")
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2)

            saved_samples += 1


def main(args):
    requested_gpu = getattr(args, "test_gpu_id", None) or getattr(args, "gpu_id", None)
    is_distributed = "RANK" in os.environ or "WORLD_SIZE" in os.environ or "LOCAL_RANK" in os.environ
    if requested_gpu and not is_distributed:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(requested_gpu)

    rank, local_rank, world_size = setup_distributed(timeout_minutes=int(getattr(args, "ddp_timeout_minutes", 180)))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    seed = int(getattr(args, "global_seed", 3407)) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    experiment_dir = resolve_experiment_dir(args)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    output_dir = args.output_dir or os.path.join(experiment_dir, f"{args.mode}_{args.stage}")
    os.makedirs(output_dir, exist_ok=True)
    distributed_barrier()

    logger = create_logger_compat(output_dir if rank == 0 else None, level=getattr(args, "log_level", "INFO"))
    sequence_length = int(args.mask_prediction.get("sequence_length", getattr(args, "tar_num_frames", 15)))
    args.tar_num_frames = sequence_length

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
    eval_model = ema if should_use_ema(args, checkpoint) and checkpoint.get("ema") is not None else model

    dataset = mask_prediction_data_loader(args, stage=args.stage)
    sampler = DistributedEvalSampler(dataset, num_replicas=world_size, rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_mask_prediction_batch,
    )

    if args.mode == "reconstruct":
        mask_cfg = clone_mask_cfg(args.mask_prediction, args.mask_mode)
        diffusion = create_diffusion(
            timestep_respacing=getattr(args, "timestep_respacing_test", "ddim50"),
            diffusion_steps=args.diffusion_steps,
            learn_sigma=args.learn_sigma,
            num_frames=sequence_length,
        )
        diffusion.training = False
        summary = evaluate_mask_prediction(
            data_loader=data_loader,
            model_forward=eval_model.forward,
            vae=vae,
            diffusion=diffusion,
            device=device,
            mask_cfg=mask_cfg,
            use_amp=bool(args.mixed_precision),
            max_batches=int(args.max_batches),
            vae_encode_batch_size=int(getattr(args, "vae_encode_batch_size", 0)),
            vae_decode_batch_size=int(getattr(args, "vae_decode_batch_size", 0)),
        )

        if rank == 0:
            summary_path = os.path.join(output_dir, f"reconstruct_{args.stage}_{args.mask_mode}_summary.json")
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
            logger.info(f"Saved reconstruction summary to {summary_path}")
            logger.info(
                f"Reconstruction summary | macro_metric={summary['overall']['macro_metric']:.4f} | "
                f"weighted_mae={summary['overall']['weighted_mae']:.4f} | "
                f"weighted_mse={summary['overall']['weighted_mse']:.4f} | "
                f"weighted_psnr={summary['overall']['weighted_psnr']:.4f} | "
                f"weighted_ssim={summary['overall']['weighted_ssim']:.4f}"
            )
            for mode_name, mode_metrics in summary["modes"].items():
                logger.info(
                    f"[{mode_name}] metric={mode_metrics['metric']:.4f} | "
                    f"mae={mode_metrics['mae']:.4f} | mse={mode_metrics['mse']:.4f} | "
                    f"psnr={mode_metrics['psnr']:.4f} | ssim={mode_metrics['ssim']:.4f}"
                )

            if int(args.num_visualizations) > 0:
                sample_indices = get_fixed_visualization_indices(len(dataset), int(args.num_visualizations))
                save_mask_prediction_visualizations(
                    dataset=dataset,
                    output_dir=output_dir,
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
        distributed_barrier()
    else:
        export_densified_sequences(
            args=args,
            data_loader=data_loader,
            eval_model=eval_model,
            vae=vae,
            device=device,
            output_dir=output_dir,
            max_samples=int(args.max_samples),
        )
        distributed_barrier()
        if rank == 0:
            logger.info(f"Densified sequences exported to {output_dir}")

    cleanup()


if __name__ == "__main__":
    parser = build_infer_parser()
    config, cli_args = parse_train_config(parser=parser, default_config="configs/config_mask_pred.yaml")
    config = apply_mask_pred_overrides(config, cli_args)
    main(config)
