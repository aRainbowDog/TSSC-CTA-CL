import json
import math
import os
import random
import warnings
from copy import deepcopy
from time import time

import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from rich.console import Console
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataloader.data_loader_acdc import build_train_val_split_manifest
from dataloader.data_loader_mask import collate_mask_prediction_batch, mask_prediction_data_loader
from models.diffusion.gaussian_diffusion import create_diffusion
from models.model_dit import MVIF_models
from training.checkpointing import (
    build_training_checkpoint,
    get_ema_decay,
    move_optimizer_state_to_device,
    resolve_experiment_dir,
    safe_load_checkpoint,
)
from training.latent_utils import encode_image_batch_to_latent
from training.common import (
    build_common_train_parser,
    create_logger_compat,
    create_rich_progress,
    distributed_barrier,
    maybe_cleanup_cuda,
)
from training.losses_mask_pred import compute_masked_sequence_loss, sample_frame_mask_batch
from training.runtime import ddp_sync_context, get_raw_model, set_optimizer_zeros_grad
from training.validation_mask_pred import (
    DistributedEvalSampler,
    evaluate_mask_prediction,
    get_fixed_visualization_indices,
    save_mask_prediction_visualizations,
)
from training.vessel_mask import generate_vessel_mask_adaptive, prepare_mask_for_latent
from utils.utils import (
    cleanup,
    clip_grad_norm_,
    close_experiment_tracker,
    create_experiment_tracker,
    requires_grad,
    setup_distributed,
    update_ema,
    write_experiment_artifact,
    write_experiment_images,
    write_experiment_metric,
)

warnings.filterwarnings("ignore")


def sync_any_rank_true(local_flag, device):
    flag = torch.tensor(1 if local_flag else 0, dtype=torch.int32, device=device)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def build_parser(default_config="configs/config_mask_pred.yaml"):
    parser = build_common_train_parser(default_config=default_config)
    parser.add_argument(
        "--timing-csv",
        type=str,
        default=None,
        help="Override timing CSV path for mask prediction training",
    )
    parser.add_argument(
        "--time-field",
        type=str,
        default=None,
        choices=["normalized_time", "relative_time_seconds"],
        help="Timestamp field fed into the model",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Number of sparse frames per training sample",
    )
    return parser


def apply_mask_pred_overrides(config, cli_args):
    if not hasattr(config, "mask_prediction") or config.mask_prediction is None:
        config.mask_prediction = OmegaConf.create({})
    if not hasattr(config, "time_conditioning") or config.time_conditioning is None:
        config.time_conditioning = OmegaConf.create({})
    if cli_args.timing_csv is not None:
        config.mask_prediction.timing_csv_path = cli_args.timing_csv
    if cli_args.time_field is not None:
        config.mask_prediction.time_field = cli_args.time_field
        config.time_conditioning.field = cli_args.time_field
    if cli_args.sequence_length is not None:
        config.mask_prediction.sequence_length = cli_args.sequence_length
        config.tar_num_frames = cli_args.sequence_length
    return config


def resolve_resume_checkpoint_path(args, checkpoint_dir):
    resume_value = getattr(args, "resume_from_checkpoint", False)
    if resume_value in (False, None, "", 0):
        return None
    if isinstance(resume_value, str):
        lowered = resume_value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return os.path.join(checkpoint_dir, "latest_mask_pred.pth")
        if lowered in {"false", "0", "no"}:
            return None
        return resume_value
    if bool(resume_value):
        return os.path.join(checkpoint_dir, "latest_mask_pred.pth")
    return None


def load_pretrained_weights(model, checkpoint_path, logger, rank):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    pretrained_weights = checkpoint.get("ema", checkpoint.get("model", checkpoint))
    model_dict = get_raw_model(model).state_dict()
    pretrained_dict = {key: value for key, value in pretrained_weights.items() if key in model_dict}
    model_dict.update(pretrained_dict)
    get_raw_model(model).load_state_dict(model_dict)
    if rank == 0:
        load_ratio = len(pretrained_dict) / max(1, len(get_raw_model(model).state_dict())) * 100.0
        logger.info(f"Loaded pretrained weights from {checkpoint_path} ({load_ratio:.1f}% matched)")


def save_checkpoint(
    checkpoint_path,
    model,
    ema,
    train_steps,
    epoch,
    epoch_index,
    batch_in_epoch,
    best_val_metric,
    opt,
    lr_scheduler,
    scaler,
    extra_metadata=None,
):
    metadata = {
        "best_val_metric": best_val_metric,
        "epoch_index": epoch_index,
        "batch_in_epoch": batch_in_epoch,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    checkpoint = build_training_checkpoint(
        model=model,
        ema=ema,
        train_steps=train_steps,
        epoch=epoch,
        opt=opt,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        **metadata,
    )
    torch.save(checkpoint, checkpoint_path)


def build_vessel_weight_batch(video, valid_mask, latent_size, device, vessel_cfg):
    if vessel_cfg is None or not bool(vessel_cfg.get("enable", False)):
        return None

    max_weight = float(vessel_cfg.get("max_weight", 10.0))
    base_weight = float(vessel_cfg.get("base_weight", 1.0))
    video_gray_np = video.detach().cpu().mean(dim=2).numpy()
    weight_list = []

    for batch_idx in range(video.shape[0]):
        valid_count = max(int(valid_mask[batch_idx].sum().item()), 1)
        frames_gray = [video_gray_np[batch_idx, frame_idx] for frame_idx in range(valid_count)]
        _, soft_weight_np = generate_vessel_mask_adaptive(
            frames_gray,
            max_weight=max_weight,
            base_weight=base_weight,
        )
        weight_latent = prepare_mask_for_latent(soft_weight_np, latent_size, device)
        weight_list.append(weight_latent)

    return torch.cat(weight_list, dim=0)


def log_validation_summary(logger, val_summary, best_val_metric, epoch, train_steps):
    overall = val_summary["overall"]
    logger.info(
        f"\n{'=' * 72}\n"
        f"Epoch {epoch + 1} Step {train_steps:07d} sequence-mask validation | "
        f"MacroMetric: {overall['macro_metric']:.4f} | "
        f"Weighted MAE: {overall['weighted_mae']:.4f} | "
        f"Weighted MSE: {overall['weighted_mse']:.4f} | "
        f"Weighted PSNR: {overall['weighted_psnr']:.4f} | "
        f"Weighted SSIM: {overall['weighted_ssim']:.4f} | "
        f"Masked Frames: {int(overall['masked_frame_count'])} | "
        f"Best: {best_val_metric:.4f}\n"
        f"{'=' * 72}"
    )
    for mode_name, mode_metrics in val_summary["modes"].items():
        logger.info(
            f"[{mode_name}] metric={mode_metrics['metric']:.4f} | "
            f"mae={mode_metrics['mae']:.4f} | mse={mode_metrics['mse']:.4f} | "
            f"psnr={mode_metrics['psnr']:.4f} | ssim={mode_metrics['ssim']:.4f} | "
            f"frames={int(mode_metrics['count'])} | "
            f"videos={int(mode_metrics['video_count'])}"
        )


def run_validation(
    args,
    rank,
    logger,
    tracker,
    dataset_val,
    loader_val,
    model,
    ema,
    vae,
    diffusion,
    device,
    experiment_dir,
    checkpoint_dir,
    train_steps,
    epoch,
    best_val_metric,
):
    if dataset_val is None or loader_val is None:
        return best_val_metric, None

    eval_use_ema = bool(args.mask_prediction.get("eval_use_ema", True))
    val_weight_source = "ema" if eval_use_ema else "raw"
    eval_model = ema if eval_use_ema else get_raw_model(model)
    eval_model.eval()
    diffusion.training = False

    val_summary = evaluate_mask_prediction(
        data_loader=loader_val,
        model_forward=eval_model.forward,
        vae=vae,
        diffusion=diffusion,
        device=device,
        mask_cfg=args.mask_prediction,
        use_amp=bool(args.mixed_precision),
        max_batches=int(getattr(args, "val_max_batches", 0)),
        vae_encode_batch_size=int(getattr(args, "vae_encode_batch_size", 0)),
        vae_decode_batch_size=int(getattr(args, "vae_decode_batch_size", 0)),
    )

    if rank == 0:
        log_validation_summary(logger, val_summary, best_val_metric, epoch, train_steps)
        overall = val_summary["overall"]
        write_experiment_metric(tracker, args.tracking_backend, "Val_MaskPred/MacroMetric", overall["macro_metric"], train_steps)
        write_experiment_metric(tracker, args.tracking_backend, "Val_MaskPred/WeightedMAE", overall["weighted_mae"], train_steps)
        write_experiment_metric(tracker, args.tracking_backend, "Val_MaskPred/WeightedMSE", overall["weighted_mse"], train_steps)
        write_experiment_metric(tracker, args.tracking_backend, "Val_MaskPred/WeightedPSNR", overall["weighted_psnr"], train_steps)
        write_experiment_metric(tracker, args.tracking_backend, "Val_MaskPred/WeightedSSIM", overall["weighted_ssim"], train_steps)
        write_experiment_metric(
            tracker,
            args.tracking_backend,
            "Val_MaskPred/MaskedFrameCount",
            overall["masked_frame_count"],
            train_steps,
        )
        for mode_name, mode_metrics in val_summary["modes"].items():
            prefix = f"Val_MaskPred/{mode_name}"
            write_experiment_metric(tracker, args.tracking_backend, f"{prefix}/Metric", mode_metrics["metric"], train_steps)
            write_experiment_metric(tracker, args.tracking_backend, f"{prefix}/MAE", mode_metrics["mae"], train_steps)
            write_experiment_metric(tracker, args.tracking_backend, f"{prefix}/MSE", mode_metrics["mse"], train_steps)
            write_experiment_metric(tracker, args.tracking_backend, f"{prefix}/PSNR", mode_metrics["psnr"], train_steps)
            write_experiment_metric(tracker, args.tracking_backend, f"{prefix}/SSIM", mode_metrics["ssim"], train_steps)

        val_visualization_count = int(getattr(args, "val_visualization_count", 0))
        if val_visualization_count > 0:
            sample_indices = get_fixed_visualization_indices(len(dataset_val), val_visualization_count)
            val_image_dir = os.path.join(experiment_dir, "val_mask_pred")
            generated_images = save_mask_prediction_visualizations(
                dataset=dataset_val,
                output_dir=val_image_dir,
                sample_indices=sample_indices,
                batch_size=int(getattr(args, "val_visualization_batch_size", 4)),
                model_forward=eval_model.forward,
                vae=vae,
                diffusion=diffusion,
                device=device,
                mask_cfg=args.mask_prediction,
                epoch=epoch,
                train_steps=train_steps,
                use_amp=bool(args.mixed_precision),
                vae_encode_batch_size=int(getattr(args, "vae_encode_batch_size", 0)),
                vae_decode_batch_size=int(getattr(args, "vae_decode_batch_size", 0)),
            )
            write_experiment_images(tracker, args.tracking_backend, generated_images, train_steps)

        current_metric = float(overall["macro_metric"])
        if current_metric < best_val_metric:
            previous_best = best_val_metric
            best_val_metric = current_metric
            best_ckpt_path = os.path.join(checkpoint_dir, "best_mask_pred.pth")
            save_checkpoint(
                checkpoint_path=best_ckpt_path,
                model=model,
                ema=ema,
                train_steps=train_steps,
                epoch=epoch + 1,
                epoch_index=epoch + 1,
                batch_in_epoch=0,
                best_val_metric=best_val_metric,
                opt=None,
                lr_scheduler=None,
                scaler=None,
                extra_metadata={
                    "val_summary": val_summary,
                    "val_metric": current_metric,
                    "prev_best_metric": previous_best,
                    "checkpoint_type": "best",
                    "val_weight_source": val_weight_source,
                },
            )
            write_experiment_artifact(
                tracker,
                args.tracking_backend,
                artifact_name=f"{os.path.basename(experiment_dir)}-best-mask-pred",
                artifact_path=best_ckpt_path,
                artifact_type="checkpoint",
                metadata={
                    "train_steps": train_steps,
                    "epoch": epoch + 1,
                    "macro_metric": current_metric,
                },
            )
            logger.info(
                f"Saved best checkpoint: {best_ckpt_path} | "
                f"metric={current_metric:.4f} | previous={previous_best:.4f}"
            )

    diffusion.training = True
    return best_val_metric, val_summary


def main(args):
    requested_gpu = getattr(args, "gpu_id", None)
    is_distributed = "RANK" in os.environ or "WORLD_SIZE" in os.environ or "LOCAL_RANK" in os.environ
    if requested_gpu and not is_distributed:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(requested_gpu)

    ddp_timeout_minutes = int(getattr(args, "ddp_timeout_minutes", 180))
    rank, local_rank, world_size = setup_distributed(timeout_minutes=ddp_timeout_minutes)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    seed = int(getattr(args, "global_seed", 3407)) + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    sequence_length = int(args.mask_prediction.get("sequence_length", getattr(args, "tar_num_frames", 15)))
    args.tar_num_frames = sequence_length
    args.mask_prediction.sequence_length = sequence_length
    val_weight_source = "ema" if bool(args.mask_prediction.get("eval_use_ema", True)) else "raw"

    experiment_dir = resolve_experiment_dir(args)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True) if rank == 0 else None

    tracking_backend = getattr(args, "tracking_backend", None)
    wandb_project = getattr(args, "wandb_project", "TSSC-CTA-CL")
    wandb_entity = getattr(args, "wandb_entity", None)
    wandb_mode = str(getattr(args, "wandb_mode", "online")).lower()
    wandb_run_name = getattr(args, "wandb_run_name", None) or os.path.basename(experiment_dir)
    shared_console = Console(stderr=True) if rank == 0 else None
    logger = create_logger_compat(experiment_dir if rank == 0 else None, level=getattr(args, "log_level", "INFO"), console=shared_console)

    tracker = None
    if rank == 0:
        tracker = create_experiment_tracker(
            backend=tracking_backend,
            logging_dir=os.path.join(experiment_dir, "runs") if tracking_backend == "tensorboard" else experiment_dir,
            run_name=wandb_run_name,
            config=OmegaConf.to_container(args, resolve=True),
            project=wandb_project,
            entity=wandb_entity,
            mode=wandb_mode,
        ) if tracking_backend else None
        OmegaConf.save(args, os.path.join(experiment_dir, "config_mask_pred.yaml"))
        logger.info(f"Mask prediction training | time_field={args.mask_prediction.time_field} | sequence_length={sequence_length}")
        logger.info(f"Experiment dir: {experiment_dir}")
        logger.info(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
        logger.info(f"Tracking backend: {tracking_backend}")
        logger.info(f"Vessel mask weighting: {'on' if bool(getattr(args, 'vessel_mask', {}).get('enable', False)) else 'off'}")

        split_manifest = build_train_val_split_manifest(
            args.data_path_train,
            seed=getattr(args, "global_seed", 3407),
        )
        split_manifest_path = os.path.join(experiment_dir, "train_val_split_manifest.json")
        with open(split_manifest_path, "w", encoding="utf-8") as handle:
            json.dump(split_manifest, handle, ensure_ascii=False, indent=2)
        write_experiment_artifact(
            tracker,
            tracking_backend,
            artifact_name=f"{os.path.basename(experiment_dir)}-train-val-split",
            artifact_path=split_manifest_path,
            artifact_type="data_split",
            metadata={
                "seed": split_manifest["seed"],
                "train_ratio": split_manifest["train_ratio"],
                "split_strategy": split_manifest["split_strategy"],
                "train_group_count": split_manifest["train"]["group_count"],
                "train_video_count": split_manifest["train"]["video_count"],
                "val_group_count": split_manifest["val"]["group_count"],
                "val_video_count": split_manifest["val"]["video_count"],
            },
        )

    assert args.image_size % 8 == 0, "Image size must be divisible by 8."
    latent_size = args.image_size // 8
    model = MVIF_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=bool(getattr(args, "learn_sigma", True)),
        mode="video",
        num_frames=sequence_length,
    ).to(device, non_blocking=True)
    ema = deepcopy(model).to(device, non_blocking=True)
    requires_grad(ema, False)
    ema.eval()

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_model_path,
        subfolder="sd-vae-ft-mse",
    ).to(device, non_blocking=True)
    vae.requires_grad_(False)
    vae.eval()

    load_pretrained_weights(model, args.pretrained, logger, rank)

    if args.gradient_checkpointing:
        def enable_ckpt(module):
            if hasattr(module, "enable_gradient_checkpointing"):
                module.enable_gradient_checkpointing()
            elif hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()
        model.apply(enable_ckpt)

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

    if args.use_compile and torch.cuda.is_available():
        if rank == 0:
            logger.info("Compiling mask prediction model with torch.compile")
        model = torch.compile(model, mode="reduce-overhead")

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
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.mixed_precision))
    set_optimizer_zeros_grad(opt)

    dataset_train = mask_prediction_data_loader(args, stage="train")
    sampler_train = DistributedSampler(
        dataset_train,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    loader_train = DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=False,
        sampler=sampler_train,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_mask_prediction_batch,
    )

    dataset_val = mask_prediction_data_loader(args, stage="val")
    loader_val = None
    if len(dataset_val) > 0:
        sampler_val = DistributedEvalSampler(dataset_val, num_replicas=world_size, rank=rank)
        loader_val = DataLoader(
            dataset_val,
            batch_size=args.val_batch_size,
            shuffle=False,
            sampler=sampler_val,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_mask_prediction_batch,
        )

    train_batches_per_epoch = len(loader_train)
    train_updates_per_epoch = max(1, math.ceil(train_batches_per_epoch / args.gradient_accumulation_steps))
    num_train_epochs = math.ceil(args.max_train_steps / train_updates_per_epoch)

    if rank == 0:
        logger.info(f"Train dataset size: {len(dataset_train)}")
        logger.info(f"Val dataset size: {len(dataset_val)}")
        logger.info(f"Train batches per epoch: {train_batches_per_epoch}")
        logger.info(f"Optimizer steps per epoch: {train_updates_per_epoch}")

    train_diffusion = create_diffusion(
        timestep_respacing=args.timestep_respacing_train,
        diffusion_steps=args.diffusion_steps,
        learn_sigma=args.learn_sigma,
        num_frames=sequence_length,
    )
    train_diffusion.training = True

    val_diffusion = None
    if loader_val is not None:
        val_diffusion = create_diffusion(
            timestep_respacing=getattr(args, "timestep_respacing_val", getattr(args, "timestep_respacing_test", "ddim20")),
            diffusion_steps=args.diffusion_steps,
            learn_sigma=args.learn_sigma,
            num_frames=sequence_length,
        )
        val_diffusion.training = False

    train_steps = 0
    first_epoch = 0
    resume_step = 0
    best_val_metric = float("inf")

    resume_path = resolve_resume_checkpoint_path(args, checkpoint_dir)
    if resume_path is not None and os.path.exists(resume_path):
        checkpoint = safe_load_checkpoint(resume_path, device)
        get_raw_model(model).load_state_dict(checkpoint["model"])
        if checkpoint.get("ema") is not None:
            ema.load_state_dict(checkpoint["ema"])
            ema.eval()
        if checkpoint.get("opt") is not None:
            opt.load_state_dict(checkpoint["opt"])
            move_optimizer_state_to_device(opt, device)
        if checkpoint.get("lr_scheduler") is not None:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])

        train_steps = int(checkpoint.get("train_steps", 0))
        best_val_metric = float(checkpoint.get("best_val_metric", float("inf")))

        epoch_index = checkpoint.get("epoch_index", None)
        batch_in_epoch = checkpoint.get("batch_in_epoch", None)
        if epoch_index is not None and batch_in_epoch is not None:
            first_epoch = int(epoch_index)
            resume_step = int(batch_in_epoch)
            if resume_step >= train_batches_per_epoch:
                first_epoch += 1
                resume_step = 0
        else:
            saved_epoch = checkpoint.get("epoch", None)
            if saved_epoch is not None:
                first_epoch = int(saved_epoch)
            else:
                first_epoch = train_steps // train_updates_per_epoch
                resume_updates = train_steps % train_updates_per_epoch
                resume_step = min(resume_updates * args.gradient_accumulation_steps, train_batches_per_epoch)

        if rank == 0:
            logger.info(f"Resuming from checkpoint: {resume_path}")
            logger.info(f"Resumed train steps: {train_steps}")
            logger.info(f"Resumed epoch index: {first_epoch}, batch offset: {resume_step}")
            if best_val_metric != float("inf"):
                logger.info(f"Resumed best validation metric: {best_val_metric:.4f}")
        del checkpoint
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if train_steps >= args.max_train_steps:
        if rank == 0:
            logger.info(f"train_steps={train_steps} already reached max_train_steps={args.max_train_steps}, exiting.")
            close_experiment_tracker(tracker, tracking_backend)
        cleanup()
        return

    train_progress = None
    overall_task = None
    epoch_task = None
    if rank == 0:
        train_progress = create_rich_progress(console=shared_console)
        train_progress.start()
        overall_task = train_progress.add_task(
            "Overall Opt Step",
            total=args.max_train_steps,
            completed=train_steps,
            status=f"opt {train_steps}/{args.max_train_steps}",
        )
        epoch_task = train_progress.add_task(
            f"Epoch {first_epoch + 1}/{num_train_epochs} Batch",
            total=len(loader_train),
            completed=resume_step,
            status=f"opt {train_steps}/{args.max_train_steps}",
        )

    running_loss = 0.0
    running_masked_frames = 0.0
    log_steps = 0
    start_time = time()
    consecutive_nonfinite_steps = 0
    max_consecutive_nonfinite_steps = max(1, int(getattr(args, "max_consecutive_nonfinite_steps", 4)))
    last_saved_epoch_label = first_epoch
    last_saved_epoch_index = first_epoch
    last_saved_batch_in_epoch = resume_step
    model.train()

    for epoch in range(first_epoch, num_train_epochs):
        sampler_train.set_epoch(epoch)
        last_batch_in_epoch = resume_step if epoch == first_epoch else 0
        completed_full_epoch = True
        if rank == 0 and train_progress is not None:
            train_progress.update(
                epoch_task,
                description=f"Epoch {epoch + 1}/{num_train_epochs} Batch",
                total=len(loader_train),
                completed=resume_step if epoch == first_epoch else 0,
                status=f"opt {train_steps}/{args.max_train_steps}",
            )

        for step, batch in enumerate(loader_train):
            if epoch == first_epoch and step < resume_step:
                if rank == 0 and train_progress is not None:
                    train_progress.update(epoch_task, completed=step + 1, status=f"opt {train_steps}/{args.max_train_steps}")
                continue

            video = batch["video"].to(device, non_blocking=True)
            frame_times = batch["frame_times"].to(device, non_blocking=True)
            valid_mask = batch["frame_valid_mask"].to(device, non_blocking=True)
            batch_size = video.shape[0]

            with torch.no_grad():
                video_flat = video.reshape(-1, *video.shape[2:])
                latent = encode_image_batch_to_latent(
                    vae,
                    video_flat,
                    use_amp=bool(args.mixed_precision),
                    chunk_size=int(getattr(args, "vae_encode_batch_size", 0)),
                )
                latent = latent.reshape(batch_size, sequence_length, *latent.shape[1:])

            has_nonfinite_inputs = (
                not bool(torch.isfinite(frame_times).all().item())
                or not bool(torch.isfinite(latent).all().item())
            )
            if sync_any_rank_true(has_nonfinite_inputs, device):
                raise FloatingPointError(
                    f"Encountered non-finite frame_times or VAE latents at epoch {epoch + 1}, batch {step + 1}."
                )

            frame_mask = sample_frame_mask_batch(valid_mask, args.mask_prediction, device=device)
            if sync_any_rank_true(float(frame_mask.sum().item()) <= 0.0, device):
                raise ValueError(
                    f"Encountered an empty frame mask at epoch {epoch + 1}, batch {step + 1}. "
                    "Check sequence lengths and mask sampling constraints."
                )
            weight_batch = build_vessel_weight_batch(
                video=video,
                valid_mask=valid_mask,
                latent_size=latent.shape[-1],
                device=device,
                vessel_cfg=getattr(args, "vessel_mask", None),
            )
            masked_frames_per_sample = frame_mask.sum(dim=1).mean().item()
            t = torch.randint(0, train_diffusion.num_timesteps, (batch_size,), device=device)

            should_step = ((step + 1) % args.gradient_accumulation_steps == 0) or ((step + 1) == len(loader_train))
            accum_divisor = args.gradient_accumulation_steps
            if should_step and (step + 1) % args.gradient_accumulation_steps != 0:
                accum_divisor = (step % args.gradient_accumulation_steps) + 1

            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                with ddp_sync_context(model, should_sync=should_step):
                    loss = compute_masked_sequence_loss(
                        model=model,
                        diffusion=train_diffusion,
                        latent_sequence=latent,
                        frame_times=frame_times,
                        frame_mask=frame_mask,
                        t=t,
                        loss_target_mode=str(args.mask_prediction.get("loss_target_mode", "auto")),
                        weight_batch=weight_batch,
                    )
                has_nonfinite_loss = not bool(torch.isfinite(loss.detach()).all().item())
                if not has_nonfinite_loss:
                    loss_to_backward = loss / accum_divisor
                    scaler.scale(loss_to_backward).backward()
                else:
                    loss_to_backward = None

            if sync_any_rank_true(has_nonfinite_loss, device):
                consecutive_nonfinite_steps += 1
                set_optimizer_zeros_grad(opt)
                if rank == 0:
                    logger.warning(
                        f"Skipping non-finite loss at epoch {epoch + 1}, batch {step + 1}, "
                        f"opt step {train_steps} | consecutive skips {consecutive_nonfinite_steps}/"
                        f"{max_consecutive_nonfinite_steps}"
                    )
                if consecutive_nonfinite_steps >= max_consecutive_nonfinite_steps:
                    raise FloatingPointError(
                        f"Aborting after {consecutive_nonfinite_steps} consecutive non-finite losses."
                    )
                continue

            if should_step:
                if args.mixed_precision:
                    scaler.unscale_(opt)
                grad_norm = clip_grad_norm_(
                    get_raw_model(model).parameters(),
                    args.clip_max_norm,
                    clip_grad=(train_steps >= args.start_clip_iter),
                )
                has_nonfinite_grad = not bool(torch.isfinite(grad_norm.detach()).all().item())
                if sync_any_rank_true(has_nonfinite_grad, device):
                    consecutive_nonfinite_steps += 1
                    if args.mixed_precision:
                        scaler.step(opt)
                        scaler.update()
                    set_optimizer_zeros_grad(opt)
                    if rank == 0:
                        logger.warning(
                            f"Skipping non-finite gradients at epoch {epoch + 1}, batch {step + 1}, "
                            f"opt step {train_steps} | consecutive skips {consecutive_nonfinite_steps}/"
                            f"{max_consecutive_nonfinite_steps}"
                        )
                    if consecutive_nonfinite_steps >= max_consecutive_nonfinite_steps:
                        raise FloatingPointError(
                            f"Aborting after {consecutive_nonfinite_steps} consecutive non-finite gradient steps."
                        )
                    continue
                scaler.step(opt)
                scaler.update()
                lr_scheduler.step()
                set_optimizer_zeros_grad(opt)
                train_steps += 1
                consecutive_nonfinite_steps = 0

                ema_decay = get_ema_decay(
                    train_steps,
                    warmup_steps=int(getattr(args, "ema_warmup_steps", 1000)),
                )
                with torch.no_grad():
                    update_ema(ema, get_raw_model(model), decay=ema_decay)

                running_loss += loss.detach().item()
                running_masked_frames += masked_frames_per_sample
                log_steps += 1

                if train_steps % args.log_every == 0 and rank == 0:
                    avg_loss = running_loss / max(1, log_steps)
                    avg_masked = running_masked_frames / max(1, log_steps)
                    steps_per_sec = log_steps / max(1e-6, (time() - start_time))
                    logger.info(
                        f"Step {train_steps:07d} | Loss: {avg_loss:.4f} | "
                        f"Masked Frames/Sample: {avg_masked:.2f} | "
                        f"Grad Norm: {float(grad_norm):.4f} | LR: {opt.param_groups[0]['lr']:.6e} | "
                        f"EMA: {ema_decay:.6f} | Steps/Sec: {steps_per_sec:.2f}"
                    )
                    write_experiment_metric(tracker, tracking_backend, "Train/Loss", avg_loss, train_steps)
                    write_experiment_metric(tracker, tracking_backend, "Train/MaskedFramesPerSample", avg_masked, train_steps)
                    write_experiment_metric(tracker, tracking_backend, "Train/GradNorm", float(grad_norm), train_steps)
                    write_experiment_metric(tracker, tracking_backend, "Train/LR", opt.param_groups[0]["lr"], train_steps)
                    write_experiment_metric(tracker, tracking_backend, "Train/EMADecay", ema_decay, train_steps)
                    running_loss = 0.0
                    running_masked_frames = 0.0
                    log_steps = 0
                    start_time = time()

                if train_steps % args.ckpt_every_iter == 0 and rank == 0:
                    ckpt_path = os.path.join(checkpoint_dir, f"step_{train_steps:07d}.pth")
                    save_checkpoint(
                        checkpoint_path=ckpt_path,
                        model=model,
                        ema=ema,
                        train_steps=train_steps,
                        epoch=epoch + 1,
                        epoch_index=epoch,
                        batch_in_epoch=step + 1,
                        best_val_metric=best_val_metric,
                        opt=opt,
                        lr_scheduler=lr_scheduler,
                        scaler=scaler,
                        extra_metadata={
                            "checkpoint_type": "step",
                            "val_weight_source": val_weight_source,
                        },
                    )
                    logger.info(f"Saved checkpoint: {ckpt_path}")

                maybe_cleanup_cuda(train_steps, getattr(args, "cuda_empty_cache_interval", 0))

            if rank == 0 and train_progress is not None:
                train_progress.update(
                    epoch_task,
                    completed=step + 1,
                    status=f"opt {train_steps}/{args.max_train_steps} | mask {masked_frames_per_sample:.2f}",
                )
                train_progress.update(
                    overall_task,
                    completed=train_steps,
                    status=f"opt {train_steps}/{args.max_train_steps}",
                )

            last_batch_in_epoch = step + 1
            del batch, video, frame_times, valid_mask, video_flat, latent, frame_mask, weight_batch, t, loss, loss_to_backward
            if train_steps >= args.max_train_steps:
                completed_full_epoch = False
                break

        distributed_barrier()

        current_epoch_label = epoch + 1
        current_epoch_index = epoch + 1 if completed_full_epoch else epoch
        current_batch_in_epoch = 0 if completed_full_epoch else last_batch_in_epoch
        last_saved_epoch_label = current_epoch_label
        last_saved_epoch_index = current_epoch_index
        last_saved_batch_in_epoch = current_batch_in_epoch

        if rank == 0:
            latest_epoch_path = os.path.join(checkpoint_dir, "latest_mask_pred.pth")
            save_checkpoint(
                checkpoint_path=latest_epoch_path,
                model=model,
                ema=ema,
                train_steps=train_steps,
                epoch=current_epoch_label,
                epoch_index=current_epoch_index,
                batch_in_epoch=current_batch_in_epoch,
                best_val_metric=best_val_metric,
                opt=opt,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                extra_metadata={
                    "checkpoint_type": "latest_epoch",
                    "val_weight_source": val_weight_source,
                },
            )

        should_validate = (
            loader_val is not None
            and int(getattr(args, "val_interval", 0)) > 0
            and (((epoch + 1) % int(args.val_interval) == 0) or train_steps >= args.max_train_steps)
        )
        if should_validate:
            best_val_metric, _ = run_validation(
                args=args,
                rank=rank,
                logger=logger,
                tracker=tracker,
                dataset_val=dataset_val,
                loader_val=loader_val,
                model=model,
                ema=ema,
                vae=vae,
                diffusion=val_diffusion,
                device=device,
                experiment_dir=experiment_dir,
                checkpoint_dir=checkpoint_dir,
                train_steps=train_steps,
                epoch=epoch,
                best_val_metric=best_val_metric,
            )
            distributed_barrier()
            model.train()
            ema.eval()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            if rank == 0:
                latest_epoch_path = os.path.join(checkpoint_dir, "latest_mask_pred.pth")
                save_checkpoint(
                    checkpoint_path=latest_epoch_path,
                    model=model,
                    ema=ema,
                    train_steps=train_steps,
                    epoch=last_saved_epoch_label,
                    epoch_index=last_saved_epoch_index,
                    batch_in_epoch=last_saved_batch_in_epoch,
                    best_val_metric=best_val_metric,
                    opt=opt,
                    lr_scheduler=lr_scheduler,
                    scaler=scaler,
                    extra_metadata={
                        "checkpoint_type": "latest_epoch_post_val",
                        "val_weight_source": val_weight_source,
                    },
                )

        if train_steps >= args.max_train_steps:
            break
        resume_step = 0

    distributed_barrier()
    if rank == 0:
        latest_path = os.path.join(checkpoint_dir, "latest_mask_pred.pth")
        save_checkpoint(
            checkpoint_path=latest_path,
            model=model,
            ema=ema,
            train_steps=train_steps,
            epoch=last_saved_epoch_label,
            epoch_index=last_saved_epoch_index,
            batch_in_epoch=last_saved_batch_in_epoch,
            best_val_metric=best_val_metric,
            opt=opt,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            extra_metadata={
                "checkpoint_type": "final",
                "val_weight_source": val_weight_source,
            },
        )
        logger.info(f"Training finished. Latest checkpoint: {latest_path}")
        if train_progress is not None:
            train_progress.stop()
        close_experiment_tracker(tracker, tracking_backend)
    cleanup()
