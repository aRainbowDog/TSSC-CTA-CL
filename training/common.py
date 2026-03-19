import argparse
import gc
import logging

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from utils.utils import create_logger


def build_common_train_parser(default_config="configs/config_cta.yaml"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--vessel_max_weight", type=float, default=None, help="Max weight for vessel region")
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=None,
        choices=["tensorboard", "wandb"],
        help="Experiment tracking backend",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity/team",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="Weights & Biases logging mode",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--ddp-timeout-minutes",
        type=int,
        default=None,
        help="Process group timeout in minutes for long validation/checkpoint sections",
    )
    return parser


def apply_common_train_overrides(config, cli_args):
    if cli_args.vessel_max_weight is not None:
        if not hasattr(config, "vessel_mask") or config.vessel_mask is None:
            config.vessel_mask = OmegaConf.create({})
        config.vessel_mask.max_weight = cli_args.vessel_max_weight
    if cli_args.log_level is not None:
        config.log_level = cli_args.log_level
    if cli_args.tracker is not None:
        config.tracking_backend = cli_args.tracker
    if cli_args.wandb_project is not None:
        config.wandb_project = cli_args.wandb_project
    if cli_args.wandb_entity is not None:
        config.wandb_entity = cli_args.wandb_entity
    if cli_args.wandb_mode is not None:
        config.wandb_mode = cli_args.wandb_mode
    if cli_args.wandb_run_name is not None:
        config.wandb_run_name = cli_args.wandb_run_name
    if cli_args.ddp_timeout_minutes is not None:
        config.ddp_timeout_minutes = cli_args.ddp_timeout_minutes
    return config


def load_config_from_parsed_args(cli_args):
    config = OmegaConf.load(cli_args.config)
    return apply_common_train_overrides(config, cli_args)


def parse_train_config(parser=None, default_config="configs/config_cta.yaml"):
    parser = parser or build_common_train_parser(default_config=default_config)
    cli_args = parser.parse_args()
    config = load_config_from_parsed_args(cli_args)
    return config, cli_args


def run_train_entrypoint(main_fn, parser=None, default_config="configs/config_cta.yaml"):
    config, _ = parse_train_config(parser=parser, default_config=default_config)
    main_fn(config)


def create_logger_compat(logging_dir, level="INFO", console=None):
    try:
        return create_logger(logging_dir, level=level, console=console)
    except TypeError:
        logger = create_logger(logging_dir)
        resolved_level = getattr(logging, str(level).upper(), logging.INFO)
        logger.setLevel(resolved_level)
        for handler in logger.handlers:
            handler.setLevel(resolved_level)
        return logger


def distributed_barrier():
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


def create_rich_progress(console=None):
    return Progress(
        SpinnerColumn(style="bold blue"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=28),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[status]}", justify="left"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console or Console(stderr=True),
        transient=False,
        expand=True,
        refresh_per_second=4,
    )


def maybe_cleanup_cuda(step, interval):
    if interval is None or interval <= 0 or step <= 0 or step % interval != 0:
        return
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
