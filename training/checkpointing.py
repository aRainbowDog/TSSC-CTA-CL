import os

import torch

from training.runtime import get_raw_model


def resolve_experiment_dir(args):
    model_string_name = args.model.replace("/", "-")
    return f"{args.results_dir}/{model_string_name}_{args.cur_date}"


def safe_load_checkpoint(ckpt_path, device):
    """
    Load a checkpoint to CPU first and move tensors later as needed.
    """
    del device
    return torch.load(ckpt_path, map_location="cpu")


def move_optimizer_state_to_device(optimizer, device):
    """Ensure optimizer state tensors live on the same device as model params after resume."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device, non_blocking=True)


def build_training_checkpoint(
    model,
    ema,
    train_steps,
    epoch,
    opt=None,
    lr_scheduler=None,
    scaler=None,
    **metadata,
):
    checkpoint = {
        "model": get_raw_model(model).state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "train_steps": train_steps,
        "epoch": epoch,
    }
    checkpoint.update(metadata)
    if opt is not None:
        checkpoint["opt"] = opt.state_dict()
    if lr_scheduler is not None:
        checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint


def get_ema_decay(train_steps, base_decay=0.9999, min_decay=0.99, warmup_steps=1000):
    """
    Linearly warm up EMA decay to stabilize early training.
    """
    if train_steps <= 0:
        return min_decay
    if train_steps < warmup_steps:
        return min_decay + (base_decay - min_decay) * (train_steps / warmup_steps)
    return base_decay
