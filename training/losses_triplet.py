import torch

from models.diffusion.gaussian_diffusion import LossType, ModelMeanType, ModelVarType
from training.runtime import backward_loss, ddp_sync_context


def align_weight_batch(weight_batch, batch_size, num_channels, height, width):
    """Accept [B,1,H,W] or [B,C,H,W] weights and expand to [B,C,H,W] for loss weighting."""
    if weight_batch is None:
        return None
    if weight_batch.dim() == 3:
        weight_batch = weight_batch.unsqueeze(1)
    if weight_batch.dim() != 4:
        raise ValueError(f"weight_batch must be 3D or 4D, got shape {tuple(weight_batch.shape)}")
    if weight_batch.shape[0] != batch_size or weight_batch.shape[-2:] != (height, width):
        raise ValueError(
            f"weight_batch shape {tuple(weight_batch.shape)} is incompatible with target "
            f"({batch_size}, {num_channels}, {height}, {width})"
        )
    if weight_batch.shape[1] == 1:
        return weight_batch.expand(-1, num_channels, -1, -1)
    if weight_batch.shape[1] != num_channels:
        raise ValueError(
            f"weight_batch channel dim must be 1 or {num_channels}, got {weight_batch.shape[1]}"
        )
    return weight_batch


def resolve_triplet_loss_pair(
    diffusion,
    pred_middle_raw,
    x_t_middle,
    x_start_middle,
    noise_middle,
    t,
    loss_target_mode="auto",
):
    """Resolve prediction/target tensors for the requested triplet loss mode."""
    if loss_target_mode == "auto":
        if diffusion.model_mean_type == ModelMeanType.EPSILON:
            return pred_middle_raw, noise_middle
        if diffusion.model_mean_type == ModelMeanType.START_X:
            return pred_middle_raw, x_start_middle
        if diffusion.model_mean_type == ModelMeanType.PREVIOUS_X:
            target_prev, _, _ = diffusion.q_posterior_mean_variance(
                x_start=x_start_middle,
                x_t=x_t_middle,
                t=t,
            )
            return pred_middle_raw, target_prev
        raise NotImplementedError(f"Unsupported model_mean_type: {diffusion.model_mean_type}")

    if loss_target_mode == "epsilon":
        if diffusion.model_mean_type != ModelMeanType.EPSILON:
            raise ValueError(
                "loss_target_mode='epsilon' requires diffusion.model_mean_type == ModelMeanType.EPSILON"
            )
        return pred_middle_raw, noise_middle

    if loss_target_mode == "x0":
        if diffusion.model_mean_type == ModelMeanType.EPSILON:
            pred_middle = diffusion._predict_xstart_from_eps(
                x_t=x_t_middle,
                t=t,
                eps=pred_middle_raw,
            )
        elif diffusion.model_mean_type == ModelMeanType.START_X:
            pred_middle = pred_middle_raw
        else:
            raise NotImplementedError(
                f"loss_target_mode='x0' is not implemented for {diffusion.model_mean_type}"
            )
        return pred_middle, x_start_middle

    raise ValueError(
        f"Unsupported loss_target_mode: {loss_target_mode}. Expected one of: auto, epsilon, x0"
    )


def compute_triplet_loss(
    model,
    diffusion,
    latent_triplet,
    t,
    weight_batch=None,
    device=None,
    loss_target_mode="auto",
):
    """
    Compute conditional middle-frame loss from a visible-start/end triplet.
    """
    bsz, num_frames, channels, height, width = latent_triplet.shape
    assert num_frames == 3, f"必须是三元组，当前{num_frames}帧"

    mask = torch.ones(bsz, num_frames, height, width, device=device)
    mask[:, 0, :, :] = 0
    mask[:, 2, :, :] = 0

    latent_permuted = latent_triplet.permute(0, 2, 1, 3, 4)
    noise = torch.randn_like(latent_permuted)
    x_t_bcfhw = diffusion.q_sample(latent_permuted, t, noise=noise)

    x_t = x_t_bcfhw.permute(0, 2, 1, 3, 4)
    model_input = latent_triplet * (1 - mask.unsqueeze(2)) + x_t * mask.unsqueeze(2)
    model_output = model(model_input, t)
    vb_loss = None

    if diffusion.model_var_type in {ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE}:
        if model_output.shape[2] != channels * 2:
            raise ValueError(
                f"Expected model output channels {channels * 2} for learned sigma, got {model_output.shape[2]}"
            )
        model_output, model_var_values = torch.split(model_output, channels, dim=2)
        frozen_out = torch.cat([model_output.detach(), model_var_values], dim=2).permute(0, 2, 1, 3, 4)
        vb_loss = diffusion._vb_terms_bpd(
            model=lambda *args, r=frozen_out: r,
            x_start=latent_permuted,
            x_t=x_t_bcfhw,
            t=t,
            clip_denoised=False,
        )["output"].mean()
        if diffusion.loss_type == LossType.RESCALED_MSE:
            vb_loss = vb_loss * diffusion.num_timesteps / 1000.0
    elif model_output.shape[2] != channels:
        raise ValueError(
            f"Expected model output channels {channels} for fixed sigma, got {model_output.shape[2]}"
        )

    pred_middle_raw = model_output[:, 1, :, :, :]
    x_t_middle = x_t[:, 1, :, :, :]
    noise_middle = noise.permute(0, 2, 1, 3, 4)[:, 1, :, :, :]
    x_start_middle = latent_triplet[:, 1, :, :, :]

    pred_middle, target_middle = resolve_triplet_loss_pair(
        diffusion=diffusion,
        pred_middle_raw=pred_middle_raw,
        x_t_middle=x_t_middle,
        x_start_middle=x_start_middle,
        noise_middle=noise_middle,
        t=t,
        loss_target_mode=loss_target_mode,
    )

    loss_middle = (pred_middle - target_middle) ** 2

    if weight_batch is not None:
        weight_batch = align_weight_batch(weight_batch, bsz, pred_middle.shape[1], height, width)
        weight_mean = weight_batch.mean()
        loss_middle = loss_middle * weight_batch * (1.0 / weight_mean)

    total_loss = loss_middle.mean()
    if vb_loss is not None:
        total_loss = total_loss + vb_loss
    return total_loss


@torch.no_grad()
def fast_predict_middle_frame(model, val_diffusion, triplet_latent, mask, device):
    """
    Fast DDIM-style middle-frame prediction helper.
    """
    z = torch.randn_like(triplet_latent.permute(0, 2, 1, 3, 4))
    samples = val_diffusion.p_sample_loop(
        model.forward,
        z.shape,
        z,
        clip_denoised=False,
        progress=False,
        device=device,
        raw_x=triplet_latent.permute(0, 2, 1, 3, 4),
        mask=mask,
    )
    samples = samples.permute(1, 0, 2, 3, 4)
    predicted_middle = samples[:, 1, :, :, :].clone()
    del samples, z
    torch.cuda.empty_cache()
    return predicted_middle


def build_stage2_recursive_triplets(latent_dense):
    """Build the extra recursive triplets used in stage2, excluding the base triplet."""
    _, num_frames, _, _, _ = latent_dense.shape
    assert num_frames == 5, f"阶段2需要5帧，当前{num_frames}帧"
    return [
        torch.stack([latent_dense[:, 0], latent_dense[:, 1], latent_dense[:, 2]], dim=1),
        torch.stack([latent_dense[:, 2], latent_dense[:, 3], latent_dense[:, 4]], dim=1),
    ]


def build_stage3_recursive_triplets(latent_dense):
    """Build the extra recursive triplets used in stage3, excluding the base triplet."""
    _, num_frames, _, _, _ = latent_dense.shape
    assert num_frames == 9, f"阶段3需要9帧，当前{num_frames}帧"
    triplets = [
        torch.stack([latent_dense[:, 0], latent_dense[:, 2], latent_dense[:, 4]], dim=1),
        torch.stack([latent_dense[:, 4], latent_dense[:, 6], latent_dense[:, 8]], dim=1),
    ]
    for left, mid, right in [(0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, 8)]:
        triplets.append(torch.stack([latent_dense[:, left], latent_dense[:, mid], latent_dense[:, right]], dim=1))
    return triplets


def backward_recursive_triplets(
    model,
    diffusion,
    triplets,
    t,
    device,
    recursive_weight,
    loss_scale_divisor,
    scaler=None,
    mixed_precision=True,
    loss_target_mode="auto",
    sync_last_backward=False,
):
    """
    Backward each recursive triplet immediately to reduce peak memory.
    Returns the mean recursive loss value for logging.
    """
    del device
    if not triplets:
        return 0.0

    loss_sum = 0.0
    scaled_weight = recursive_weight / (len(triplets) * loss_scale_divisor)

    for idx, triplet in enumerate(triplets):
        should_sync = sync_last_backward and idx == len(triplets) - 1
        with ddp_sync_context(model, should_sync=should_sync):
            triplet_loss = compute_triplet_loss(
                model,
                diffusion,
                triplet,
                t,
                None,
                triplet.device,
                loss_target_mode=loss_target_mode,
            )
            loss_sum += triplet_loss.detach().item()
            backward_loss(
                scaled_weight * triplet_loss,
                scaler=scaler,
                mixed_precision=mixed_precision,
            )
            del triplet_loss

    torch.cuda.empty_cache()
    return loss_sum / len(triplets)
