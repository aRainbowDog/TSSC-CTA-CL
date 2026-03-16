import argparse
import datetime
import os
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from dataloader.data_loader_acdc import collate_full_sequence_batch, full_sequence_data_loader
from models.diffusion.gaussian_diffusion import create_diffusion
from models.model_dit import MVIF_models
from utils.triplet_eval import (
    evaluate_video_sliding_triplets,
    evaluate_video_sliding_triplets_baseline,
)
from utils.utils import cleanup, setup_distributed


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
    mode = str(ckpt_mode or "raw").lower()
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
        args, "timestep_respacing_val",
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


def maybe_log(metrics_log, line=""):
    if metrics_log is not None:
        metrics_log.write(line)


def main(args):
    requested_gpu = getattr(args, "test_gpu_id", None)
    is_distributed = "RANK" in os.environ or "WORLD_SIZE" in os.environ or "LOCAL_RANK" in os.environ
    if requested_gpu and not is_distributed:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(requested_gpu)

    ddp_timeout_minutes = int(getattr(args, "ddp_timeout_minutes", 180))
    rank, local_rank, world_size = setup_distributed(timeout_minutes=ddp_timeout_minutes)
    device = torch.device("cuda", local_rank)

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

    experiment_dir = resolve_experiment_dir(args)
    respacing = resolve_respacing(args)
    triplet_eval_batch_size = int(getattr(args, "triplet_eval_batch_size", 4))
    loader_batch_size = resolve_loader_batch_size(args)
    ckpt_mode = getattr(args, "test_ckpt_mode", "raw")
    checkpoint_path = None

    if not baseline_method:
        checkpoint_path = resolve_checkpoint_path(args)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    default_output_dir = os.path.join(experiment_dir, "validation_style_eval", eval_split)
    output_dir = getattr(args, "eval_output_dir", default_output_dir)
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    run_name = baseline_method or os.path.splitext(os.path.basename(checkpoint_path))[0]
    metrics_path = os.path.join(
        output_dir,
        f"{run_name}_{(baseline_method or ckpt_mode)}_{respacing}.log",
    )

    metrics_log = open(metrics_path, "w", encoding="utf-8") if rank == 0 else None
    try:
        maybe_log(metrics_log, f"validation-style eval started: {datetime.datetime.now()}\n")
        maybe_log(metrics_log, f"split: {eval_split}\n")
        maybe_log(metrics_log, f"baseline_method: {baseline_method or '-'}\n")
        maybe_log(metrics_log, f"checkpoint: {checkpoint_path or '-'}\n")
        maybe_log(metrics_log, f"checkpoint_mode: {ckpt_mode if checkpoint_path else '-'}\n")
        maybe_log(metrics_log, f"respacing: {respacing}\n")
        maybe_log(metrics_log, f"triplet_eval_batch_size: {triplet_eval_batch_size}\n")
        maybe_log(metrics_log, f"loader_batch_size: {loader_batch_size}\n\n")

        diffusion = None
        vae = None
        model = None
        resolved_mode = baseline_method or "-"
        pretrained_dict = {}

        if not baseline_method:
            learn_sigma = bool(getattr(args, "learn_sigma", True))
            diffusion = create_diffusion(
                timestep_respacing=respacing,
                diffusion_steps=args.diffusion_steps,
                learn_sigma=learn_sigma,
            )
            vae = AutoencoderKL.from_pretrained(
                args.pretrained_vae_model_path,
                subfolder="sd-vae-ft-mse",
            ).to(device)
            vae.requires_grad_(False)

            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            latent_size = args.image_size // 8
            additional_kwargs = {"num_frames": args.tar_num_frames, "mode": "video"}
            model = MVIF_models[args.model](
                input_size=latent_size,
                num_classes=args.num_classes,
                learn_sigma=learn_sigma,
                **additional_kwargs,
            ).to(device)

            state_dict, resolved_mode = load_eval_state_dict(checkpoint, ckpt_mode)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

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

        if rank == 0:
            print(f"Validation-style eval split: {eval_split}")
            print(f"Dataset size: {len(dataset)}")
            print(f"World size: {world_size}")
            if baseline_method:
                print(f"Baseline method: {baseline_method}")
            else:
                print(f"Checkpoint: {checkpoint_path}")
                print(f"Loaded weight mode: {resolved_mode}")
            print(f"Respacing: {respacing}")
            print(f"Triplet eval batch size: {triplet_eval_batch_size}")
            print(f"Loader batch size: {loader_batch_size}")

        maybe_log(metrics_log, f"dataset_size: {len(dataset)}\n")
        maybe_log(metrics_log, f"world_size: {world_size}\n")
        maybe_log(metrics_log, f"loaded_weight_mode: {resolved_mode}\n")
        if not baseline_method:
            maybe_log(metrics_log, f"loaded_keys: {len(pretrained_dict)} / {len(model_dict)}\n\n")
            model.eval()
            vae.eval()
            diffusion.training = False
        else:
            maybe_log(metrics_log, "loaded_keys: -\n\n")

        total_mse = 0.0
        total_mae = 0.0
        total_psnr = 0.0
        total_triplet_count = 0
        total_video_count = 0

        progress = tqdm(loader, total=len(loader), desc=f"{eval_split} sliding triplets", disable=rank != 0)
        with torch.no_grad():
            for batch in progress:
                for video, video_name, video_path in zip(
                        batch["videos"],
                        batch["video_name"],
                        batch["video_path"]):
                    metrics = evaluate_video_sliding_triplets(
                        video,
                        model_forward=model.forward,
                        vae=vae,
                        diffusion=diffusion,
                        device=device,
                        triplet_batch_size=triplet_eval_batch_size,
                        return_pred_video=False,
                        force_grayscale=True,
                    ) if not baseline_method else evaluate_video_sliding_triplets_baseline(
                        video,
                        method=baseline_method,
                        triplet_batch_size=triplet_eval_batch_size,
                        return_pred_video=False,
                        force_grayscale=True,
                    )

                    video_triplet_count = metrics["count"]
                    if video_triplet_count <= 0:
                        continue

                    video_mse = metrics["mse_sum"] / video_triplet_count
                    video_mae = metrics["mae_sum"] / video_triplet_count
                    video_psnr = metrics["psnr_sum"] / video_triplet_count

                    total_mse += metrics["mse_sum"]
                    total_mae += metrics["mae_sum"]
                    total_psnr += metrics["psnr_sum"]
                    total_triplet_count += video_triplet_count
                    total_video_count += 1

                    if rank == 0:
                        progress.set_postfix({
                            "local_videos": total_video_count,
                            "local_triplets": total_triplet_count,
                            "psnr": f"{video_psnr:.4f}",
                        })

        metrics_tensor = torch.tensor(
            [
                total_mse,
                total_mae,
                total_psnr,
                float(total_triplet_count),
                float(total_video_count),
            ],
            device=device,
            dtype=torch.float64,
        )
        if world_size > 1:
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

        global_triplet_count = max(int(metrics_tensor[3].item()), 1)
        global_video_count = max(int(metrics_tensor[4].item()), 1)
        avg_total_mse = metrics_tensor[0].item() / global_triplet_count
        avg_total_mae = metrics_tensor[1].item() / global_triplet_count
        avg_total_psnr = metrics_tensor[2].item() / global_triplet_count
        proxy_metric = (avg_total_mae + avg_total_mse) / 2

        if rank == 0:
            summary_lines = [
                "",
                "==================== validation-style eval summary ====================",
                f"split: {eval_split}",
                f"videos: {global_video_count}",
                f"triplets: {global_triplet_count}",
                f"mse: {avg_total_mse:.6f}",
                f"mae: {avg_total_mae:.6f}",
                f"psnr: {avg_total_psnr:.4f}",
                f"metric: {proxy_metric:.6f}",
                "====================================================================",
            ]
            print("\n".join(summary_lines))
            maybe_log(metrics_log, "\n".join(summary_lines) + "\n")
    finally:
        if metrics_log is not None:
            metrics_log.close()
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the same sliding-triplet evaluation used by training validation.")
    parser.add_argument("--config", type=str, default="configs/config_cta.yaml")
    parser.add_argument("--split", type=str, choices=["val", "test"], help="Evaluation split. Defaults to val.")
    parser.add_argument("--ckpt", type=str, help="Checkpoint filename under experiment checkpoints/ or an absolute path.")
    parser.add_argument("--ckpt-mode", type=str, choices=["raw", "ema"], help="Weight source inside checkpoint.")
    parser.add_argument("--respacing", type=str, help="Override DDIM timestep respacing for evaluation.")
    parser.add_argument("--triplet-batch-size", type=int, help="Triplet batch size inside sliding-triplet evaluation.")
    parser.add_argument("--eval-batch-size", type=int, help="Sequence batch size for the outer DataLoader.")
    parser.add_argument("--gpu-id", type=str, help="CUDA_VISIBLE_DEVICES value for this run.")
    parser.add_argument("--output-dir", type=str, help="Directory to store evaluation logs.")
    parser.add_argument(
        "--baseline-method",
        type=str,
        choices=["linear", "quadratic", "si_dis_flow", "bi_dis_flow"],
        help="Run a baseline with the same sliding-triplet protocol instead of loading a model checkpoint.",
    )
    cli_args = parser.parse_args()

    config = OmegaConf.load(cli_args.config)
    config = apply_overrides(config, cli_args)
    main(config)
