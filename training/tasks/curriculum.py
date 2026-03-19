# autodl
import warnings
import os
import json
import math
import logging
import numpy as np
import torch
import torch.distributed as dist
from time import time
from copy import deepcopy
from einops import rearrange
from rich.console import Console
from models.model_dit import MVIF_models
from models.diffusion.gaussian_diffusion import create_diffusion
from dataloader.data_loader_acdc import (
    build_train_val_split_manifest,
    collate_full_sequence_batch,
    data_loader,
    full_sequence_data_loader,
)
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from training.checkpointing import (
    build_training_checkpoint,
    get_ema_decay,
    move_optimizer_state_to_device,
    resolve_experiment_dir,
    safe_load_checkpoint,
)
from training.common import (
    create_logger_compat,
    create_rich_progress,
    distributed_barrier,
    maybe_cleanup_cuda,
)
from training.losses_triplet import (
    backward_recursive_triplets,
    build_stage2_recursive_triplets,
    build_stage3_recursive_triplets,
    compute_triplet_loss,
)
from training.runtime import ddp_sync_context, get_raw_model, set_optimizer_zeros_grad
from training.validation_triplet import (
    DistributedEvalSampler,
    format_epoch_progress_status,
    format_overall_progress_status,
    get_fixed_visualization_indices,
    save_val_sample_visualization,
)
from training.vessel_mask import generate_vessel_mask_adaptive, prepare_mask_for_latent
from utils.triplet_eval import evaluate_video_sliding_triplets
from utils.utils import (
    clip_grad_norm_, update_ema, requires_grad, 
    cleanup, create_experiment_tracker, write_experiment_metric,
    write_experiment_artifact, write_experiment_images, close_experiment_tracker, setup_distributed
)
warnings.filterwarnings('ignore')


def log_debug(logger, *lines):
    if logger is None or not logger.isEnabledFor(logging.DEBUG):
        return
    for line in lines:
        logger.debug(line)

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

    # 2. 显存优化配置
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False  # 关闭benchmark减少显存预分配
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

    # 3. 随机种子
    seed = args.global_seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 4. 实验目录（仅rank0创建）
    experiment_dir = resolve_experiment_dir(args)
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
        split_manifest = build_train_val_split_manifest(
            args.data_path_train,
            seed=getattr(args, "global_seed", 3407),
        )
        split_manifest_path = os.path.join(experiment_dir, "train_val_split_manifest.json")
        with open(split_manifest_path, "w", encoding="utf-8") as split_manifest_file:
            json.dump(split_manifest, split_manifest_file, ensure_ascii=False, indent=2)
        logger.info(
            "Data split saved: %s | train groups=%d videos=%d | val groups=%d videos=%d",
            split_manifest_path,
            split_manifest["train"]["group_count"],
            split_manifest["train"]["video_count"],
            split_manifest["val"]["group_count"],
            split_manifest["val"]["video_count"],
        )
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
        logger.info(f"{'='*60}")
        logger.info(f"📚 四阶段课程学习配置:")
        logger.info(f"  阶段0: 0→{stage0_steps} steps, Gap=1, 2帧, 伪GT+随机α")
        logger.info(f"  阶段1: {stage0_steps}→{stage1_steps} steps, Gap=2, 3帧")
        logger.info(f"  阶段2: {stage1_steps}→{stage2_steps} steps, Gap=4, 5帧")
        logger.info(f"  阶段3: {stage2_steps}+ steps, Gap=8, 9帧")
        logger.info(f"  递归链权重: {recursive_weight}")
        logger.info(f"  血管mask: {'启用' if vessel_mask_enable else '禁用'}")
        logger.info(f"  三元组loss模式: {triplet_loss_mode}")
        logger.info("  验证口径: 滑窗三连帧代理评测（首尾帧预测中间帧）")
        logger.info(f"{'='*60}")

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
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
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
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
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
        num_workers=args.num_workers,
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
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
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

    # 续训完成后：延迟编译模型（避免加载阶段占用显存）
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
    val_weight_source = "ema"
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
                val_weight_source=val_weight_source,
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
                val_weight_source=val_weight_source,
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
            ema.eval()
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
                        raw_model=ema,
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
            ema.eval()
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
                    f"代理任务: 首尾帧预测中间帧 | 权重: {val_weight_source} | "
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
                            model_forward=ema.forward,
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
                        "val_weight_source": val_weight_source,
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
                    val_weight_source=val_weight_source,
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
