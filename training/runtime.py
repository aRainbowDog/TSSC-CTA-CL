from contextlib import nullcontext

from torch.nn.parallel import DistributedDataParallel as DDP


def unwrap_model(model):
    """Strip torch.compile / DDP wrappers until the underlying module is reached."""
    current = model
    while True:
        if hasattr(current, "_orig_mod"):
            current = current._orig_mod
            continue
        if hasattr(current, "module"):
            current = current.module
            continue
        return current


def get_raw_model(model):
    """Return the wrapped module when using DDP, otherwise the model itself."""
    return unwrap_model(model)


def get_ddp_wrapper(model):
    """Return the inner DDP wrapper when present, even if wrapped by torch.compile."""
    current = model
    while hasattr(current, "_orig_mod"):
        current = current._orig_mod
    return current if isinstance(current, DDP) else None


def ddp_sync_context(model, should_sync):
    ddp_model = get_ddp_wrapper(model)
    if ddp_model is not None and not should_sync:
        return ddp_model.no_sync()
    return nullcontext()


def set_optimizer_zeros_grad(optimizer, set_to_none=True):
    """Zero gradients with set_to_none by default to reduce allocator churn."""
    if set_to_none:
        optimizer.zero_grad(set_to_none=True)
    else:
        optimizer.zero_grad()


def backward_loss(loss, scaler=None, mixed_precision=True):
    """Backward a scaled or unscaled loss tensor."""
    if mixed_precision:
        scaler.scale(loss).backward()
    else:
        loss.backward()
