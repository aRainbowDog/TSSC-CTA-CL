import random

import torch

from models.diffusion.gaussian_diffusion import LossType, ModelMeanType, ModelVarType
from training.losses_triplet import align_weight_batch


def sample_frame_mask_batch(valid_mask, mask_cfg, device):
    batch_size, sequence_length = valid_mask.shape
    mask = torch.zeros(batch_size, sequence_length, dtype=torch.float32, device=device)

    min_visible = int(mask_cfg.get("min_visible_frames", 2))
    min_masked = int(mask_cfg.get("min_masked_frames", 1))
    max_masked = int(mask_cfg.get("max_masked_frames", 4))
    single_prob = float(mask_cfg.get("single_frame_mask_prob", 0.4))
    span_prob = float(mask_cfg.get("contiguous_span_mask_prob", 0.4))
    anchor_prob = float(mask_cfg.get("anchor_drop_mask_prob", 0.2))
    single_mask_internal_only = bool(mask_cfg.get("single_mask_internal_only", False))
    prob_total = single_prob + span_prob + anchor_prob
    if prob_total <= 0:
        raise ValueError("Mask sampling probabilities must sum to a positive value")

    for batch_idx in range(batch_size):
        valid_indices = torch.nonzero(valid_mask[batch_idx] > 0.5, as_tuple=False).flatten().tolist()
        if len(valid_indices) <= min_visible:
            continue

        mode = random.random() * prob_total
        masked_indices = []

        if mode < single_prob:
            single_candidates = valid_indices
            if single_mask_internal_only and len(valid_indices) > 2:
                interior_candidates = valid_indices[1:-1]
                if interior_candidates:
                    single_candidates = interior_candidates
            masked_indices = random.sample(single_candidates, k=1)
        elif mode < single_prob + span_prob:
            max_span = min(max_masked, len(valid_indices) - min_visible)
            span_len = max(min_masked, max_span)
            if span_len <= 0:
                continue
            if max_span > min_masked:
                span_len = random.randint(min_masked, max_span)
            start = random.randint(0, len(valid_indices) - span_len)
            masked_indices = valid_indices[start:start + span_len]
        else:
            visible_indices = {valid_indices[0], valid_indices[-1]}
            candidates = valid_indices[1:-1]
            while len(visible_indices) < min_visible and candidates:
                visible_indices.add(random.choice(candidates))
            masked_indices = [idx for idx in valid_indices if idx not in visible_indices]
            if not masked_indices:
                fallback_candidates = [idx for idx in valid_indices if idx not in visible_indices]
                if fallback_candidates:
                    masked_indices = [random.choice(fallback_candidates)]

        if len(masked_indices) < min_masked:
            remaining = [idx for idx in valid_indices if idx not in masked_indices]
            extra_needed = min_masked - len(masked_indices)
            if len(remaining) > min_visible:
                random.shuffle(remaining)
                masked_indices.extend(remaining[:extra_needed])

        mask[batch_idx, masked_indices] = 1.0

    return mask


def resolve_sequence_loss_pair(
    diffusion,
    model_output_raw,
    x_t_bcthw,
    x_start_bcthw,
    noise_bcthw,
    t,
    loss_target_mode="auto",
):
    if loss_target_mode == "auto":
        if diffusion.model_mean_type == ModelMeanType.EPSILON:
            return model_output_raw, noise_bcthw.permute(0, 2, 1, 3, 4)
        if diffusion.model_mean_type == ModelMeanType.START_X:
            return model_output_raw, x_start_bcthw.permute(0, 2, 1, 3, 4)
        if diffusion.model_mean_type == ModelMeanType.PREVIOUS_X:
            target_prev, _, _ = diffusion.q_posterior_mean_variance(
                x_start=x_start_bcthw,
                x_t=x_t_bcthw,
                t=t,
            )
            return model_output_raw, target_prev.permute(0, 2, 1, 3, 4)
        raise NotImplementedError(f"Unsupported model_mean_type: {diffusion.model_mean_type}")

    if loss_target_mode == "epsilon":
        if diffusion.model_mean_type != ModelMeanType.EPSILON:
            raise ValueError("loss_target_mode='epsilon' requires epsilon prediction diffusion")
        return model_output_raw, noise_bcthw.permute(0, 2, 1, 3, 4)

    if loss_target_mode == "x0":
        if diffusion.model_mean_type == ModelMeanType.EPSILON:
            pred_x0 = diffusion._predict_xstart_from_eps(
                x_t=x_t_bcthw,
                t=t,
                eps=model_output_raw.permute(0, 2, 1, 3, 4),
            ).permute(0, 2, 1, 3, 4)
        elif diffusion.model_mean_type == ModelMeanType.START_X:
            pred_x0 = model_output_raw
        else:
            raise NotImplementedError(f"loss_target_mode='x0' is not implemented for {diffusion.model_mean_type}")
        return pred_x0, x_start_bcthw.permute(0, 2, 1, 3, 4)

    raise ValueError(f"Unsupported loss_target_mode: {loss_target_mode}")


def compute_masked_sequence_loss(
    model,
    diffusion,
    latent_sequence,
    frame_times,
    frame_mask,
    t,
    loss_target_mode="auto",
    weight_batch=None,
):
    batch_size, _, channels, _, _ = latent_sequence.shape
    x_start_bcthw = latent_sequence.permute(0, 2, 1, 3, 4)
    noise_bcthw = torch.randn_like(x_start_bcthw)
    x_t_bcthw = diffusion.q_sample(x_start_bcthw, t, noise=noise_bcthw)

    x_t = x_t_bcthw.permute(0, 2, 1, 3, 4)
    mask_5d = frame_mask[:, :, None, None, None]
    model_input = latent_sequence * (1 - mask_5d) + x_t * mask_5d
    model_output = model(model_input, t, frame_times=frame_times)
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
            x_start=x_start_bcthw,
            x_t=x_t_bcthw,
            t=t,
            clip_denoised=False,
        )["output"].mean()
        if diffusion.loss_type == LossType.RESCALED_MSE:
            vb_loss = vb_loss * diffusion.num_timesteps / 1000.0
    elif model_output.shape[2] != channels:
        raise ValueError(f"Expected model output channels {channels}, got {model_output.shape[2]}")

    pred, target = resolve_sequence_loss_pair(
        diffusion=diffusion,
        model_output_raw=model_output,
        x_t_bcthw=x_t_bcthw,
        x_start_bcthw=x_start_bcthw,
        noise_bcthw=noise_bcthw,
        t=t,
        loss_target_mode=loss_target_mode,
    )
    masked_loss = ((pred - target) ** 2) * mask_5d
    if weight_batch is not None:
        weight_batch = align_weight_batch(weight_batch, batch_size, pred.shape[2], pred.shape[3], pred.shape[4])
        weight_mean = weight_batch.mean()
        weight_5d = weight_batch[:, None, :, :, :]
        masked_loss = masked_loss * weight_5d * (1.0 / weight_mean)
    denom = mask_5d.sum() * pred.shape[2] * pred.shape[3] * pred.shape[4]
    if denom.item() <= 0:
        raise ValueError("Frame mask produced zero masked elements")
    total_loss = masked_loss.sum() / denom
    if vb_loss is not None:
        total_loss = total_loss + vb_loss
    return total_loss
