import argparse
import os
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf
from torchvision.utils import save_image

from dataloader.data_loader_mask import collate_mask_prediction_batch, mask_prediction_data_loader
from models.diffusion.gaussian_diffusion import create_diffusion
from models.model_dit import MVIF_models
from testing.common import (
    add_root_test_args,
    build_sample_key,
    list_prediction_files,
    load_manifest,
    save_prediction_record,
    sanitize_name,
    write_json,
    write_manifest,
)
from training.checkpointing import resolve_experiment_dir, safe_load_checkpoint
from training.common import create_logger_compat, distributed_barrier, load_config_from_parsed_args
from training.latent_utils import decode_latent_batch_to_image, encode_image_batch_to_latent
from training.runtime import get_raw_model
from training.tasks.mask_pred import apply_mask_pred_overrides, build_parser as build_train_parser
from training.validation_mask_pred import (
    DistributedEvalSampler,
    build_validation_frame_mask,
    compute_masked_frame_metrics,
    decode_latent_sequence,
    get_fixed_visualization_indices,
    get_validation_modes,
    reconstruct_masked_sequence,
    save_mask_prediction_visualizations,
    summarize_validation_metrics,
)
from utils.triplet_eval import prepare_gt_video_for_eval, prepare_pred_video_for_eval
from utils.utils import cleanup, requires_grad, setup_distributed

warnings.filterwarnings("ignore")


def build_infer_parser():
    parser = build_train_parser(default_config="configs/config_mask_pred.yaml")
    add_root_test_args(parser, action="infer", task_aliases=["mask_pred", "mask_prediction", "mask-pred"])
    parser.add_argument("--stage", type=str, default="val", choices=["val", "test"], help="Dataset split used for inference/export.")
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
        choices=["all", "single", "span", "anchor"],
        help="Mask pattern used when `--mode reconstruct`.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to store inference outputs.")
    parser.add_argument("--max-batches", type=int, default=0, help="Optional cap for reconstruction batches.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for densification exports per rank.")
    parser.add_argument("--num-visualizations", type=int, default=8, help="How many qualitative reconstruction visualizations to export on rank 0.")
    parser.add_argument("--use-ema", type=str, default="auto", choices=["auto", "true", "false"], help="Whether to prefer EMA weights.")
    parser.add_argument("--densify-fps", type=float, default=1.0, help="Target FPS for densification export.")
    return parser


def build_eval_parser():
    parser = argparse.ArgumentParser(
        description="Read saved mask-pred reconstruction outputs and compute MAE/MSE/PSNR/SSIM offline."
    )
    add_root_test_args(parser, action="eval", task_aliases=["mask_pred", "mask_prediction", "mask-pred"])
    parser.add_argument("--input-dir", type=str, required=True, help="Directory produced by `test.py infer --task mask_pred --mode reconstruct`.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional directory for eval summary. Defaults to the input dir.")
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
    ]
    for field_name in override_fields:
        setattr(config, field_name, getattr(cli_args, field_name))
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


def build_default_output_dir(args, experiment_dir):
    if args.mode == "reconstruct":
        dir_name = f"reconstruct_{args.stage}_{args.mask_mode}"
    else:
        dir_name = f"densify_{args.stage}_{sanitize_name(f'{float(args.densify_fps):g}fps')}"
    return os.path.join(experiment_dir, "test", "mask_pred", dir_name)


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
):
    mask_cfg = clone_mask_cfg(args.mask_prediction, args.mask_mode)
    mode_names = get_validation_modes(mask_cfg)
    min_visible = int(mask_cfg.get("min_visible_frames", 2))
    span_length = int(mask_cfg.get("validation_span_length", mask_cfg.get("max_masked_frames", 2)))
    saved_records = 0

    for batch_idx, batch in enumerate(data_loader):
        if max_batches and batch_idx >= max_batches:
            break

        video = batch["video"].to(device, non_blocking=True)
        frame_times = batch["frame_times"].to(device, non_blocking=True)
        valid_mask = batch["frame_valid_mask"].to(device, non_blocking=True)
        dataset_indices = batch["dataset_index"].tolist()
        video_names = batch["video_name"]
        video_paths = batch["video_path"]
        frame_times_relative = batch["frame_times_relative"].detach().cpu()
        frame_times_normalized = batch["frame_times_normalized"].detach().cpu()
        batch_size = video.shape[0]

        video_flat = video.reshape(-1, *video.shape[2:])
        latent = encode_image_batch_to_latent(
            vae,
            video_flat,
            use_amp=bool(args.mixed_precision),
            chunk_size=int(getattr(args, "vae_encode_batch_size", 0)),
        )
        latent = latent.reshape(batch_size, video.shape[1], *latent.shape[1:])

        gt_video_eval = prepare_gt_video_for_eval(video.detach().cpu())
        valid_mask_cpu = valid_mask.detach().cpu()

        for mode_name in mode_names:
            frame_mask = build_validation_frame_mask(
                valid_mask=valid_mask,
                mode=mode_name,
                min_visible=min_visible,
                span_length=span_length,
                device=device,
            )
            if float(frame_mask.sum().item()) <= 0:
                continue

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
            frame_mask_cpu = frame_mask.detach().cpu()

            for sample_offset, dataset_index in enumerate(dataset_indices):
                if float(frame_mask_cpu[sample_offset].sum().item()) <= 0:
                    continue

                sample_key = build_sample_key(
                    f"idx_{int(dataset_index):06d}",
                    video_names[sample_offset],
                    mode_name,
                )
                save_prediction_record(
                    output_dir,
                    sample_key,
                    {
                        "task": "mask_pred",
                        "infer_mode": "reconstruct",
                        "stage": args.stage,
                        "mode_name": mode_name,
                        "video_name": video_names[sample_offset],
                        "video_path": video_paths[sample_offset],
                        "dataset_index": int(dataset_index),
                        "frame_mask": frame_mask_cpu[sample_offset].to(torch.uint8),
                        "valid_mask": valid_mask_cpu[sample_offset].to(torch.uint8),
                        "frame_times_relative": frame_times_relative[sample_offset].to(torch.float32),
                        "frame_times_normalized": frame_times_normalized[sample_offset].to(torch.float32),
                        "gt_video": gt_video_eval[sample_offset].to(torch.float16),
                        "pred_video": pred_video_eval[sample_offset].to(torch.float16),
                    },
                )
                saved_records += 1

        del batch, video, frame_times, valid_mask, video_flat, latent

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

    return saved_samples


def run_infer(args):
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
    output_dir = args.output_dir or build_default_output_dir(args, experiment_dir)
    if rank == 0:
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
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_mask_prediction_batch,
    )

    if rank == 0:
        write_manifest(
            output_dir,
            {
                "task": "mask_pred",
                "phase": "infer",
                "infer_mode": args.mode,
                "stage": args.stage,
                "mask_mode": args.mask_mode,
                "checkpoint": ckpt_path,
                "use_ema": bool(use_ema),
                "world_size": world_size,
                "densify_fps": float(args.densify_fps),
            },
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
        local_saved = save_reconstruct_predictions(
            args=args,
            data_loader=data_loader,
            vae=vae,
            diffusion=diffusion,
            eval_model=eval_model,
            device=device,
            output_dir=output_dir,
            max_batches=int(args.max_batches),
        )
        if rank == 0 and int(args.num_visualizations) > 0:
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
        )

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
        batch_metrics = compute_masked_frame_metrics(
            pred_video=record["pred_video"].float().unsqueeze(0),
            gt_video=record["gt_video"].float().unsqueeze(0),
            frame_mask=record["frame_mask"].float().unsqueeze(0),
            valid_mask=record["valid_mask"].float().unsqueeze(0),
        )
        for key, value in batch_metrics.items():
            metric_totals[mode_name][key] += value
        record_count += 1

    summary = summarize_validation_metrics(metric_totals)
    summary["task"] = "mask_pred"
    summary["input_dir"] = args.input_dir
    summary["records"] = record_count

    output_dir = args.output_dir or args.input_dir
    summary_path = os.path.join(output_dir, "eval_summary.json")
    write_json(summary_path, summary)

    overall = summary["overall"]
    print("Mask-pred offline eval summary")
    print(f"input_dir: {args.input_dir}")
    print(f"records: {record_count}")
    print(f"macro_metric: {overall['macro_metric']:.6f}")
    print(f"weighted_mae: {overall['weighted_mae']:.6f}")
    print(f"weighted_mse: {overall['weighted_mse']:.6f}")
    print(f"weighted_psnr: {overall['weighted_psnr']:.4f}")
    print(f"weighted_ssim: {overall['weighted_ssim']:.4f}")
    print(f"summary: {summary_path}")


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
