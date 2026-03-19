import argparse
import os
import warnings

import numpy as np
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from dataloader.data_loader_acdc import collate_full_sequence_batch, full_sequence_data_loader
from models.diffusion.gaussian_diffusion import create_diffusion
from models.model_dit import MVIF_models
from testing.common import (
    add_root_test_args,
    build_sample_key,
    list_prediction_files,
    load_manifest,
    sanitize_name,
    save_prediction_record,
    sha1_short,
    write_json,
    write_manifest,
)
from training.common import distributed_barrier
from utils.triplet_eval import (
    evaluate_video_sliding_triplets,
    evaluate_video_sliding_triplets_baseline,
)
from utils.utils import cleanup, setup_distributed

warnings.filterwarnings("ignore")


class DistributedEvalSampler(Sampler):
    """Shard evaluation samples across ranks without padding or duplication."""

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


def load_eval_state_dict(checkpoint, ckpt_mode):
    mode = str(ckpt_mode or "auto").lower()
    if mode == "auto":
        mode = str(checkpoint.get("val_weight_source", "raw")).lower()
    if mode == "ema" and "ema" in checkpoint:
        return checkpoint["ema"], "ema"
    return checkpoint["model"], "raw"


def resolve_experiment_dir(args):
    model_string_name = args.model.replace("/", "-")
    return f"{args.results_dir}/{model_string_name}_{args.cur_date}"


def resolve_checkpoint_path(args):
    checkpoint_name = getattr(args, "eval_ckpt", None) or getattr(args, "test_ckpt", None)
    if not checkpoint_name:
        checkpoint_name = "best_epoch_train_model.pth"

    if os.path.isabs(checkpoint_name):
        return checkpoint_name

    experiment_dir = resolve_experiment_dir(args)
    return os.path.join(experiment_dir, "checkpoints", checkpoint_name)


def resolve_respacing(args):
    split = str(getattr(args, "eval_split", "val")).lower()
    if split == "test":
        return getattr(args, "timestep_respacing_eval", None) or getattr(
            args, "timestep_respacing_test", "ddim50"
        )
    return getattr(args, "timestep_respacing_eval", None) or getattr(
        args,
        "timestep_respacing_val",
        getattr(args, "timestep_respacing_test", "ddim50"),
    )


def resolve_loader_batch_size(args):
    split = str(getattr(args, "eval_split", "val")).lower()
    if getattr(args, "eval_batch_size", None) is not None:
        return int(args.eval_batch_size)
    if split == "test":
        return int(getattr(args, "test_batch_size", 1))
    return int(getattr(args, "val_batch_size", 1))


def apply_overrides(config, cli_args):
    if cli_args.split is not None:
        config.eval_split = cli_args.split
    if cli_args.ckpt is not None:
        config.eval_ckpt = cli_args.ckpt
    if cli_args.ckpt_mode is not None:
        config.test_ckpt_mode = cli_args.ckpt_mode
    if cli_args.respacing is not None:
        config.timestep_respacing_eval = cli_args.respacing
    if cli_args.triplet_batch_size is not None:
        config.triplet_eval_batch_size = cli_args.triplet_batch_size
    if cli_args.eval_batch_size is not None:
        config.eval_batch_size = cli_args.eval_batch_size
    if cli_args.gpu_id is not None:
        config.test_gpu_id = cli_args.gpu_id
    if cli_args.output_dir is not None:
        config.eval_output_dir = cli_args.output_dir
    if cli_args.baseline_method is not None:
        config.eval_baseline_method = cli_args.baseline_method
    return config


def build_default_output_dir(args, checkpoint_path, baseline_method, resolved_mode, respacing):
    experiment_dir = resolve_experiment_dir(args)
    split = str(getattr(args, "eval_split", "val")).lower()
    run_name = baseline_method or os.path.splitext(os.path.basename(checkpoint_path))[0]
    dir_name = "_".join(
        [
            sanitize_name(run_name),
            sanitize_name(baseline_method or resolved_mode),
            sanitize_name(respacing),
        ]
    )
    return os.path.join(experiment_dir, "test", "sliding_triplets", split, dir_name)


def build_infer_parser():
    parser = argparse.ArgumentParser(
        description="Run sliding-triplet inference and save per-video predictions for offline evaluation."
    )
    add_root_test_args(parser, action="infer", task_aliases=["sliding_triplets", "triplet"])
    parser.add_argument("--config", type=str, default="configs/config_cta.yaml")
    parser.add_argument("--split", type=str, choices=["val", "test"], help="Evaluation split. Defaults to val.")
    parser.add_argument("--ckpt", type=str, help="Checkpoint filename under experiment checkpoints/ or an absolute path.")
    parser.add_argument("--ckpt-mode", type=str, choices=["auto", "raw", "ema"], help="Weight source inside checkpoint.")
    parser.add_argument("--respacing", type=str, help="Override DDIM timestep respacing for evaluation.")
    parser.add_argument("--triplet-batch-size", type=int, help="Triplet batch size inside sliding-triplet inference.")
    parser.add_argument("--eval-batch-size", type=int, help="Sequence batch size for the outer DataLoader.")
    parser.add_argument("--gpu-id", type=str, help="CUDA_VISIBLE_DEVICES value for this run.")
    parser.add_argument("--output-dir", type=str, help="Directory to store inference outputs.")
    parser.add_argument(
        "--baseline-method",
        type=str,
        choices=["linear", "quadratic", "si_dis_flow", "bi_dis_flow"],
        help="Run a baseline with the same sliding-triplet protocol instead of loading a model checkpoint.",
    )
    return parser


def build_eval_parser():
    parser = argparse.ArgumentParser(
        description="Read saved sliding-triplet predictions and compute MAE/MSE/PSNR offline."
    )
    add_root_test_args(parser, action="eval", task_aliases=["sliding_triplets", "triplet"])
    parser.add_argument("--input-dir", type=str, required=True, help="Directory produced by `test.py infer --task sliding_triplets`.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional directory for eval summary. Defaults to the input dir.")
    return parser


def run_infer(args):
    requested_gpu = getattr(args, "test_gpu_id", None)
    is_distributed = "RANK" in os.environ or "WORLD_SIZE" in os.environ or "LOCAL_RANK" in os.environ
    if requested_gpu and not is_distributed:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(requested_gpu)

    ddp_timeout_minutes = int(getattr(args, "ddp_timeout_minutes", 180))
    rank, local_rank, world_size = setup_distributed(timeout_minutes=ddp_timeout_minutes)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    seed = int(getattr(args, "global_seed", 3407)) + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    eval_split = str(getattr(args, "eval_split", "val")).lower()
    if eval_split not in {"val", "test"}:
        raise ValueError(f"eval_split must be 'val' or 'test', got {eval_split}")

    baseline_method = getattr(args, "eval_baseline_method", None)
    baseline_method = str(baseline_method).lower() if baseline_method else None
    if baseline_method == "quadratic":
        raise ValueError(
            "quadratic baseline cannot be evaluated with the validation-style sliding-triplet protocol. "
            "Its original implementation uses the center frame as an interpolation anchor."
        )

    respacing = resolve_respacing(args)
    triplet_eval_batch_size = int(getattr(args, "triplet_eval_batch_size", 4))
    loader_batch_size = resolve_loader_batch_size(args)
    ckpt_mode = getattr(args, "test_ckpt_mode", "auto")
    checkpoint_path = None
    resolved_mode = baseline_method or "-"

    diffusion = None
    vae = None
    model = None

    if not baseline_method:
        checkpoint_path = resolve_checkpoint_path(args)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        learn_sigma = bool(getattr(args, "learn_sigma", True))
        diffusion = create_diffusion(
            timestep_respacing=respacing,
            diffusion_steps=args.diffusion_steps,
            learn_sigma=learn_sigma,
        )
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        latent_size = args.image_size // 8
        additional_kwargs = {"num_frames": args.tar_num_frames, "mode": "video"}
        model = MVIF_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
            learn_sigma=learn_sigma,
            **additional_kwargs,
        ).to(device)
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_vae_model_path,
            subfolder="sd-vae-ft-mse",
        ).to(device)
        vae.requires_grad_(False)
        state_dict, resolved_mode = load_eval_state_dict(checkpoint, ckpt_mode)
        model_dict = model.state_dict()
        pretrained_dict = {key: value for key, value in state_dict.items() if key in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.eval()
        vae.eval()
        diffusion.training = False

    output_dir = getattr(args, "eval_output_dir", None) or build_default_output_dir(
        args=args,
        checkpoint_path=checkpoint_path or "baseline",
        baseline_method=baseline_method,
        resolved_mode=resolved_mode,
        respacing=respacing,
    )
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        write_manifest(
            output_dir,
            {
                "task": "sliding_triplets",
                "phase": "infer",
                "split": eval_split,
                "baseline_method": baseline_method,
                "checkpoint": checkpoint_path,
                "loaded_weight_mode": resolved_mode,
                "respacing": respacing,
                "triplet_eval_batch_size": triplet_eval_batch_size,
                "loader_batch_size": loader_batch_size,
                "world_size": world_size,
            },
        )
    distributed_barrier()

    dataset = full_sequence_data_loader(args, stage=eval_split)
    sampler = DistributedEvalSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(
        dataset,
        batch_size=loader_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_full_sequence_batch,
    )

    local_saved = 0
    progress = tqdm(loader, total=len(loader), desc=f"{eval_split} sliding infer", disable=rank != 0)
    with torch.no_grad():
        for batch in progress:
            for video, video_name, video_path in zip(
                batch["videos"],
                batch["video_name"],
                batch["video_path"],
            ):
                metrics = (
                    evaluate_video_sliding_triplets(
                        video,
                        model_forward=model.forward,
                        vae=vae,
                        diffusion=diffusion,
                        device=device,
                        triplet_batch_size=triplet_eval_batch_size,
                        return_pred_video=True,
                        force_grayscale=True,
                    )
                    if not baseline_method
                    else evaluate_video_sliding_triplets_baseline(
                        video,
                        method=baseline_method,
                        triplet_batch_size=triplet_eval_batch_size,
                        return_pred_video=True,
                        force_grayscale=True,
                    )
                )

                triplet_count = int(metrics["count"])
                gt_video = metrics["gt_video"]
                pred_video = metrics["pred_video"]
                sample_key = build_sample_key(video_name, sha1_short(video_path))
                save_prediction_record(
                    output_dir,
                    sample_key,
                    {
                        "task": "sliding_triplets",
                        "split": eval_split,
                        "video_name": video_name,
                        "video_path": video_path,
                        "triplet_count": triplet_count,
                        "gt_middle": gt_video[1:-1].to(dtype=torch.float16),
                        "pred_middle": pred_video[1:-1].to(dtype=torch.float16),
                    },
                )
                local_saved += 1
                if rank == 0:
                    progress.set_postfix({"saved": local_saved})

    saved_tensor = torch.tensor([local_saved], dtype=torch.long, device=device)
    if world_size > 1:
        dist.all_reduce(saved_tensor, op=dist.ReduceOp.SUM)
    distributed_barrier()
    if rank == 0:
        print(f"Sliding-triplet inference outputs: {int(saved_tensor.item())} videos -> {output_dir}")
    cleanup()


def run_eval(args):
    manifest = load_manifest(args.input_dir)
    if manifest is not None and manifest.get("task") != "sliding_triplets":
        raise ValueError(f"Input dir {args.input_dir} is not a sliding-triplets prediction set.")

    prediction_files = list_prediction_files(args.input_dir)
    if not prediction_files:
        raise FileNotFoundError(f"No prediction files found under {args.input_dir}")

    total_mse = 0.0
    total_mae = 0.0
    total_psnr = 0.0
    total_triplet_count = 0
    total_video_count = 0

    for prediction_path in prediction_files:
        record = torch.load(prediction_path, map_location="cpu")
        if record.get("task") != "sliding_triplets":
            raise ValueError(f"Unexpected record type in {prediction_path}")

        gt_middle = record["gt_middle"].float()
        pred_middle = record["pred_middle"].float()
        triplet_count = int(record.get("triplet_count", gt_middle.shape[0]))
        if triplet_count <= 0:
            continue

        pred_np = pred_middle.numpy()
        gt_np = gt_middle.numpy()
        pred_flat = pred_np.reshape(pred_np.shape[0], -1)
        gt_flat = gt_np.reshape(gt_np.shape[0], -1)

        total_mae += float(np.sum(np.abs(pred_flat - gt_flat).mean(axis=1)))
        total_mse += float(np.sum(((pred_flat - gt_flat) ** 2).mean(axis=1)))
        total_psnr += float(
            np.sum(
                [
                    peak_signal_noise_ratio(gt_np[idx], pred_np[idx], data_range=2.0)
                    for idx in range(gt_np.shape[0])
                ]
            )
        )
        total_triplet_count += triplet_count
        total_video_count += 1

    global_triplet_count = max(total_triplet_count, 1)
    avg_total_mse = total_mse / global_triplet_count
    avg_total_mae = total_mae / global_triplet_count
    avg_total_psnr = total_psnr / global_triplet_count
    proxy_metric = (avg_total_mae + avg_total_mse) / 2.0

    summary = {
        "task": "sliding_triplets",
        "input_dir": args.input_dir,
        "videos": total_video_count,
        "triplets": total_triplet_count,
        "mse": avg_total_mse,
        "mae": avg_total_mae,
        "psnr": avg_total_psnr,
        "metric": proxy_metric,
    }

    output_dir = args.output_dir or args.input_dir
    summary_path = os.path.join(output_dir, "eval_summary.json")
    write_json(summary_path, summary)

    print("Sliding-triplet offline eval summary")
    print(f"input_dir: {args.input_dir}")
    print(f"videos: {total_video_count}")
    print(f"triplets: {total_triplet_count}")
    print(f"mse: {avg_total_mse:.6f}")
    print(f"mae: {avg_total_mae:.6f}")
    print(f"psnr: {avg_total_psnr:.4f}")
    print(f"metric: {proxy_metric:.6f}")
    print(f"summary: {summary_path}")


def run_infer_cli(argv=None):
    parser = build_infer_parser()
    cli_args = parser.parse_args(argv)
    config = OmegaConf.load(cli_args.config)
    config = apply_overrides(config, cli_args)
    run_infer(config)


def run_eval_cli(argv=None):
    parser = build_eval_parser()
    cli_args = parser.parse_args(argv)
    run_eval(cli_args)
