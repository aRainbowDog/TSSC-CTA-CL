import os

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from torch.utils.data import Sampler
from torchvision.utils import save_image


def get_fixed_visualization_indices(dataset_len, count):
    if dataset_len <= 0 or count <= 0:
        return []
    if count >= dataset_len:
        return list(range(dataset_len))
    return sorted({int(idx) for idx in np.linspace(0, dataset_len - 1, num=count)})


def get_visualization_stage_info(current_step, stage0_steps, stage1_steps, stage2_steps):
    """Return visualization stage id/name for the current training step."""
    if current_step < stage0_steps:
        return 0, "Stage0_Gap1"
    if current_step < stage1_steps:
        return 1, "Stage1_Gap2"
    if current_step < stage2_steps:
        return 2, "Stage2_Gap4"
    return 3, "Stage3_Gap8"


class DistributedEvalSampler(Sampler):
    """Shard validation samples across ranks without padding or duplication."""

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


def get_stage_metadata(train_steps, args):
    if train_steps < args.stage0_steps:
        return {"stage": 0, "gap": 1, "frames": 2, "next": f"s1@{args.stage0_steps}"}
    if train_steps < args.stage1_steps:
        return {"stage": 1, "gap": 2, "frames": 3, "next": f"s2@{args.stage1_steps}"}
    if train_steps < args.stage2_steps:
        return {"stage": 2, "gap": 4, "frames": 5, "next": f"s3@{args.stage2_steps}"}
    return {"stage": 3, "gap": 8, "frames": 9, "next": "done"}


def format_overall_progress_status(epoch, num_train_epochs, train_steps, args):
    meta = get_stage_metadata(train_steps, args)
    return (
        f"epoch {epoch + 1}/{num_train_epochs} | "
        f"opt {train_steps}/{args.max_train_steps} | "
        f"s{meta['stage']} g{meta['gap']} f{meta['frames']} | "
        f"next {meta['next']}"
    )


def format_epoch_progress_status(
    batch_step,
    total_batches,
    train_steps,
    max_train_steps,
    stage=None,
    frame_gap=None,
    num_frames=None,
    loss_value=None,
    recursive_loss_value=None,
    grad_norm=None,
):
    status = [
        f"batch {batch_step}/{total_batches}",
        f"opt {train_steps}/{max_train_steps}",
    ]
    if stage is not None and frame_gap is not None and num_frames is not None:
        status.append(f"s{stage} g{frame_gap} f{num_frames}")
    if loss_value is not None:
        status.append(f"loss {loss_value:.4f}")
    if recursive_loss_value is not None:
        status.append(f"rec {recursive_loss_value:.4f}")
    if grad_norm is not None:
        status.append(f"gn {grad_norm:.4f}")
    return " | ".join(status)


def save_val_sample_visualization(
    epoch,
    video_val,
    vae,
    val_diffusion,
    device,
    val_fold,
    generate_vessel_mask_adaptive,
    raw_model,
    current_step,
    stage0_steps,
    stage1_steps,
    stage2_steps,
    sample_ids=None,
    ema_model=None,
    logger=None,
):
    """
    Full validation visualization for the curriculum triplet/recursive pipeline.
    """
    del ema_model
    b_val, f_val, _, h_val, w_val = video_val.shape

    stage, stage_name = get_visualization_stage_info(current_step, stage0_steps, stage1_steps, stage2_steps)
    if stage == 0:
        if logger is not None:
            logger.warning(f"Epoch {epoch + 1}: Stage0，跳过可视化")
        else:
            print(f"⚠️  Epoch {epoch + 1}: Stage0，跳过可视化")
        return {}

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        video_val_flat = rearrange(video_val, "b f c h w -> (b f) c h w")
        latent_val = vae.encode(video_val_flat).latent_dist.sample().mul_(0.18215)
        latent_val = rearrange(latent_val, "(b f) c h w -> b f c h w", b=b_val)

    b_val, f_val, _, h_latent_val, w_latent_val = latent_val.shape
    sample_ids = list(sample_ids) if sample_ids is not None else list(range(b_val))

    video_val_np = video_val.detach().cpu().numpy()
    video_val_gray = np.mean(video_val_np, axis=2)
    vessel_mask_batches = []
    for batch_idx in range(b_val):
        frames_gray = [video_val_gray[batch_idx, t] for t in range(f_val)]
        mask_final, _ = generate_vessel_mask_adaptive(frames_gray)
        vessel_mask_vis = torch.from_numpy(mask_final).float().to(device)
        vessel_mask_vis = vessel_mask_vis.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        vessel_mask_vis = vessel_mask_vis.repeat(1, f_val, 3, 1, 1)
        vessel_mask_vis = (vessel_mask_vis / 255.0) * 2 - 1
        vessel_mask_batches.append(vessel_mask_vis)
    vessel_mask_vis = torch.cat(vessel_mask_batches, dim=0)

    with torch.no_grad():
        if stage == 1:
            triplet_latent = latent_val[:, 0:3, :, :, :]
            triplet_video_gt = video_val[:, 0:3, :, :, :]
            triplet_vessel_mask = vessel_mask_vis[:, 0:3, :, :, :]
        elif stage == 2:
            triplet_latent = torch.stack([latent_val[:, 0], latent_val[:, 2], latent_val[:, 4]], dim=1)
            triplet_video_gt = torch.stack([video_val[:, 0], video_val[:, 2], video_val[:, 4]], dim=1)
            triplet_vessel_mask = torch.stack([vessel_mask_vis[:, 0], vessel_mask_vis[:, 2], vessel_mask_vis[:, 4]], dim=1)
        else:
            triplet_latent = torch.stack([latent_val[:, 0], latent_val[:, 4], latent_val[:, 8]], dim=1)
            triplet_video_gt = torch.stack([video_val[:, 0], video_val[:, 4], video_val[:, 8]], dim=1)
            triplet_vessel_mask = torch.stack([vessel_mask_vis[:, 0], vessel_mask_vis[:, 4], vessel_mask_vis[:, 8]], dim=1)

        mask_base = torch.ones(b_val, 3, h_latent_val, w_latent_val, device=device)
        mask_base[:, 0, :, :] = 0
        mask_base[:, 2, :, :] = 0

        z = torch.randn_like(triplet_latent.permute(0, 2, 1, 3, 4))
        samples_base = val_diffusion.p_sample_loop(
            raw_model.forward,
            z.shape,
            z,
            clip_denoised=False,
            progress=False,
            device=device,
            raw_x=triplet_latent.permute(0, 2, 1, 3, 4),
            mask=mask_base,
        )

        samples_base = samples_base.permute(1, 0, 2, 3, 4) * mask_base + triplet_latent.permute(2, 0, 1, 3, 4) * (1 - mask_base)
        samples_base = samples_base.permute(1, 2, 0, 3, 4)
        samples_base_flat = rearrange(samples_base, "b f c h w -> (b f) c h w") / 0.18215
        decoded_base = vae.decode(samples_base_flat).sample
        decoded_base = rearrange(decoded_base, "(b f) c h w -> b f c h w", b=b_val)
        decoded_base_gray = decoded_base.mean(dim=2, keepdim=True)
        decoded_base = decoded_base_gray.repeat(1, 1, 3, 1, 1)

        mask_base_vis = torch.ones(b_val, 3, 3, h_val, w_val, device=device) * 0.5
        mask_base_vis[:, 0, :, :, :] = 0
        mask_base_vis[:, 2, :, :, :] = 0

        base_images = []
        for sample_idx, sample_id in enumerate(sample_ids):
            val_pic_base = torch.cat(
                [
                    triplet_video_gt[sample_idx:sample_idx + 1],
                    triplet_video_gt[sample_idx:sample_idx + 1] * (1 - mask_base_vis[sample_idx:sample_idx + 1]),
                    decoded_base[sample_idx:sample_idx + 1],
                    triplet_vessel_mask[sample_idx:sample_idx + 1],
                ],
                dim=1,
            )
            val_pic_base_flat = rearrange(val_pic_base, "b f c h w -> (b f) c h w")
            base_image_path = os.path.join(
                val_fold,
                f"Epoch_{epoch + 1}_{stage_name}_base_triplet_idx_{sample_id:04d}.png",
            )
            save_image(
                val_pic_base_flat,
                base_image_path,
                nrow=3,
                normalize=True,
                value_range=(-1, 1),
            )
            base_images.append(
                {
                    "path": base_image_path,
                    "caption": f"Epoch {epoch + 1} {stage_name} base triplet idx {sample_id}",
                }
            )

        del z, samples_base, decoded_base

    generated_images = {
        "Val Examples/BaseTriplet": base_images
    }

    if stage >= 2:
        with torch.no_grad():
            if stage == 2:
                result_frames = [None] * 5

                triplet_step1 = torch.stack([latent_val[:, 0], latent_val[:, 2], latent_val[:, 4]], dim=1)
                mask_step1 = torch.ones(b_val, 3, h_latent_val, w_latent_val, device=device)
                mask_step1[:, 0, :, :] = 0
                mask_step1[:, 2, :, :] = 0

                z1 = torch.randn_like(triplet_step1.permute(0, 2, 1, 3, 4))
                samples1 = val_diffusion.p_sample_loop(
                    raw_model.forward,
                    z1.shape,
                    z1,
                    clip_denoised=False,
                    progress=False,
                    device=device,
                    raw_x=triplet_step1.permute(0, 2, 1, 3, 4),
                    mask=mask_step1,
                )
                samples1 = samples1.permute(1, 0, 2, 3, 4) * mask_step1 + triplet_step1.permute(2, 0, 1, 3, 4) * (1 - mask_step1)
                samples1 = samples1.permute(1, 2, 0, 3, 4)

                predicted_s2_latent = samples1[:, 1, :, :, :].clone()
                decoded_s2 = vae.decode(predicted_s2_latent / 0.18215).sample
                decoded_s2_gray = decoded_s2.mean(dim=1, keepdim=True)
                result_frames[0] = video_val[:, 0]
                result_frames[2] = decoded_s2_gray.repeat(1, 3, 1, 1)
                result_frames[4] = video_val[:, 4]
                del z1, samples1

                triplet_step2 = torch.stack([latent_val[:, 0], latent_val[:, 1], predicted_s2_latent.detach()], dim=1)
                z2 = torch.randn_like(triplet_step2.permute(0, 2, 1, 3, 4))
                samples2 = val_diffusion.p_sample_loop(
                    raw_model.forward,
                    z2.shape,
                    z2,
                    clip_denoised=False,
                    progress=False,
                    device=device,
                    raw_x=triplet_step2.permute(0, 2, 1, 3, 4),
                    mask=mask_step1,
                )
                samples2 = samples2.permute(1, 0, 2, 3, 4) * mask_step1 + triplet_step2.permute(2, 0, 1, 3, 4) * (1 - mask_step1)
                samples2 = samples2.permute(1, 2, 0, 3, 4)

                decoded_s1 = vae.decode(samples2[:, 1, :, :, :] / 0.18215).sample
                decoded_s1_gray = decoded_s1.mean(dim=1, keepdim=True)
                result_frames[1] = decoded_s1_gray.repeat(1, 3, 1, 1)
                del z2, samples2

                triplet_step3 = torch.stack([predicted_s2_latent.detach(), latent_val[:, 3], latent_val[:, 4]], dim=1)
                z3 = torch.randn_like(triplet_step3.permute(0, 2, 1, 3, 4))
                samples3 = val_diffusion.p_sample_loop(
                    raw_model.forward,
                    z3.shape,
                    z3,
                    clip_denoised=False,
                    progress=False,
                    device=device,
                    raw_x=triplet_step3.permute(0, 2, 1, 3, 4),
                    mask=mask_step1,
                )
                samples3 = samples3.permute(1, 0, 2, 3, 4) * mask_step1 + triplet_step3.permute(2, 0, 1, 3, 4) * (1 - mask_step1)
                samples3 = samples3.permute(1, 2, 0, 3, 4)

                decoded_s3 = vae.decode(samples3[:, 1, :, :, :] / 0.18215).sample
                decoded_s3_gray = decoded_s3.mean(dim=1, keepdim=True)
                result_frames[3] = decoded_s3_gray.repeat(1, 3, 1, 1)
                del z3, samples3, predicted_s2_latent

                result_video = torch.stack(result_frames, dim=1)
                num_frames_rec = 5
            else:
                result_frames = [None] * 9
                result_frames[0] = video_val[:, 0]
                result_frames[8] = video_val[:, 8]

                mask_triplet = torch.ones(b_val, 3, h_latent_val, w_latent_val, device=device)
                mask_triplet[:, 0, :, :] = 0
                mask_triplet[:, 2, :, :] = 0

                triplet_l1 = torch.stack([latent_val[:, 0], latent_val[:, 4], latent_val[:, 8]], dim=1)
                z_l1 = torch.randn_like(triplet_l1.permute(0, 2, 1, 3, 4))
                samples_l1 = val_diffusion.p_sample_loop(
                    raw_model.forward,
                    z_l1.shape,
                    z_l1,
                    clip_denoised=False,
                    progress=False,
                    device=device,
                    raw_x=triplet_l1.permute(0, 2, 1, 3, 4),
                    mask=mask_triplet,
                )
                samples_l1 = samples_l1.permute(1, 0, 2, 3, 4) * mask_triplet + triplet_l1.permute(2, 0, 1, 3, 4) * (1 - mask_triplet)
                samples_l1 = samples_l1.permute(1, 2, 0, 3, 4)
                pred_4_latent = samples_l1[:, 1, :, :, :].clone()
                decoded_4 = vae.decode(pred_4_latent / 0.18215).sample
                decoded_4_gray = decoded_4.mean(dim=1, keepdim=True)
                result_frames[4] = decoded_4_gray.repeat(1, 3, 1, 1)
                del z_l1, samples_l1
                torch.cuda.empty_cache()

                triplet_l2a = torch.stack([latent_val[:, 0], latent_val[:, 2], pred_4_latent.detach()], dim=1)
                z_l2a = torch.randn_like(triplet_l2a.permute(0, 2, 1, 3, 4))
                samples_l2a = val_diffusion.p_sample_loop(
                    raw_model.forward,
                    z_l2a.shape,
                    z_l2a,
                    clip_denoised=False,
                    progress=False,
                    device=device,
                    raw_x=triplet_l2a.permute(0, 2, 1, 3, 4),
                    mask=mask_triplet,
                )
                samples_l2a = samples_l2a.permute(1, 0, 2, 3, 4) * mask_triplet + triplet_l2a.permute(2, 0, 1, 3, 4) * (1 - mask_triplet)
                samples_l2a = samples_l2a.permute(1, 2, 0, 3, 4)
                pred_2_latent = samples_l2a[:, 1, :, :, :].clone()
                decoded_2 = vae.decode(pred_2_latent / 0.18215).sample
                decoded_2_gray = decoded_2.mean(dim=1, keepdim=True)
                result_frames[2] = decoded_2_gray.repeat(1, 3, 1, 1)
                del z_l2a, samples_l2a
                torch.cuda.empty_cache()

                triplet_l2b = torch.stack([pred_4_latent.detach(), latent_val[:, 6], latent_val[:, 8]], dim=1)
                z_l2b = torch.randn_like(triplet_l2b.permute(0, 2, 1, 3, 4))
                samples_l2b = val_diffusion.p_sample_loop(
                    raw_model.forward,
                    z_l2b.shape,
                    z_l2b,
                    clip_denoised=False,
                    progress=False,
                    device=device,
                    raw_x=triplet_l2b.permute(0, 2, 1, 3, 4),
                    mask=mask_triplet,
                )
                samples_l2b = samples_l2b.permute(1, 0, 2, 3, 4) * mask_triplet + triplet_l2b.permute(2, 0, 1, 3, 4) * (1 - mask_triplet)
                samples_l2b = samples_l2b.permute(1, 2, 0, 3, 4)
                pred_6_latent = samples_l2b[:, 1, :, :, :].clone()
                decoded_6 = vae.decode(pred_6_latent / 0.18215).sample
                decoded_6_gray = decoded_6.mean(dim=1, keepdim=True)
                result_frames[6] = decoded_6_gray.repeat(1, 3, 1, 1)
                del z_l2b, samples_l2b
                torch.cuda.empty_cache()

                triplet_l3a = torch.stack([latent_val[:, 0], latent_val[:, 1], pred_2_latent.detach()], dim=1)
                z_l3a = torch.randn_like(triplet_l3a.permute(0, 2, 1, 3, 4))
                samples_l3a = val_diffusion.p_sample_loop(
                    raw_model.forward,
                    z_l3a.shape,
                    z_l3a,
                    clip_denoised=False,
                    progress=False,
                    device=device,
                    raw_x=triplet_l3a.permute(0, 2, 1, 3, 4),
                    mask=mask_triplet,
                )
                samples_l3a = samples_l3a.permute(1, 0, 2, 3, 4) * mask_triplet + triplet_l3a.permute(2, 0, 1, 3, 4) * (1 - mask_triplet)
                samples_l3a = samples_l3a.permute(1, 2, 0, 3, 4)
                decoded_1 = vae.decode(samples_l3a[:, 1, :, :, :] / 0.18215).sample
                decoded_1_gray = decoded_1.mean(dim=1, keepdim=True)
                result_frames[1] = decoded_1_gray.repeat(1, 3, 1, 1)
                del z_l3a, samples_l3a
                torch.cuda.empty_cache()

                triplet_l3b = torch.stack([pred_2_latent.detach(), latent_val[:, 3], pred_4_latent.detach()], dim=1)
                z_l3b = torch.randn_like(triplet_l3b.permute(0, 2, 1, 3, 4))
                samples_l3b = val_diffusion.p_sample_loop(
                    raw_model.forward,
                    z_l3b.shape,
                    z_l3b,
                    clip_denoised=False,
                    progress=False,
                    device=device,
                    raw_x=triplet_l3b.permute(0, 2, 1, 3, 4),
                    mask=mask_triplet,
                )
                samples_l3b = samples_l3b.permute(1, 0, 2, 3, 4) * mask_triplet + triplet_l3b.permute(2, 0, 1, 3, 4) * (1 - mask_triplet)
                samples_l3b = samples_l3b.permute(1, 2, 0, 3, 4)
                decoded_3 = vae.decode(samples_l3b[:, 1, :, :, :] / 0.18215).sample
                decoded_3_gray = decoded_3.mean(dim=1, keepdim=True)
                result_frames[3] = decoded_3_gray.repeat(1, 3, 1, 1)
                del z_l3b, samples_l3b
                torch.cuda.empty_cache()

                triplet_l3c = torch.stack([pred_4_latent.detach(), latent_val[:, 5], pred_6_latent.detach()], dim=1)
                z_l3c = torch.randn_like(triplet_l3c.permute(0, 2, 1, 3, 4))
                samples_l3c = val_diffusion.p_sample_loop(
                    raw_model.forward,
                    z_l3c.shape,
                    z_l3c,
                    clip_denoised=False,
                    progress=False,
                    device=device,
                    raw_x=triplet_l3c.permute(0, 2, 1, 3, 4),
                    mask=mask_triplet,
                )
                samples_l3c = samples_l3c.permute(1, 0, 2, 3, 4) * mask_triplet + triplet_l3c.permute(2, 0, 1, 3, 4) * (1 - mask_triplet)
                samples_l3c = samples_l3c.permute(1, 2, 0, 3, 4)
                decoded_5 = vae.decode(samples_l3c[:, 1, :, :, :] / 0.18215).sample
                decoded_5_gray = decoded_5.mean(dim=1, keepdim=True)
                result_frames[5] = decoded_5_gray.repeat(1, 3, 1, 1)
                del z_l3c, samples_l3c
                torch.cuda.empty_cache()

                triplet_l3d = torch.stack([pred_6_latent.detach(), latent_val[:, 7], latent_val[:, 8]], dim=1)
                z_l3d = torch.randn_like(triplet_l3d.permute(0, 2, 1, 3, 4))
                samples_l3d = val_diffusion.p_sample_loop(
                    raw_model.forward,
                    z_l3d.shape,
                    z_l3d,
                    clip_denoised=False,
                    progress=False,
                    device=device,
                    raw_x=triplet_l3d.permute(0, 2, 1, 3, 4),
                    mask=mask_triplet,
                )
                samples_l3d = samples_l3d.permute(1, 0, 2, 3, 4) * mask_triplet + triplet_l3d.permute(2, 0, 1, 3, 4) * (1 - mask_triplet)
                samples_l3d = samples_l3d.permute(1, 2, 0, 3, 4)
                decoded_7 = vae.decode(samples_l3d[:, 1, :, :, :] / 0.18215).sample
                decoded_7_gray = decoded_7.mean(dim=1, keepdim=True)
                result_frames[7] = decoded_7_gray.repeat(1, 3, 1, 1)
                del z_l3d, samples_l3d, pred_2_latent, pred_4_latent, pred_6_latent
                torch.cuda.empty_cache()

                result_video = torch.stack(result_frames, dim=1)
                num_frames_rec = 9

            video_val_rec = video_val[:, :num_frames_rec, :, :, :]
            vessel_mask_rec = vessel_mask_vis[:, :num_frames_rec, :, :, :]
            mask_rec_vis = torch.ones(b_val, num_frames_rec, 3, h_val, w_val, device=device) * 0.5
            mask_rec_vis[:, 0, :, :, :] = 0
            mask_rec_vis[:, -1, :, :, :] = 0

            recursive_images = []
            for sample_idx, sample_id in enumerate(sample_ids):
                val_pic_rec = torch.cat(
                    [
                        video_val_rec[sample_idx:sample_idx + 1],
                        video_val_rec[sample_idx:sample_idx + 1] * (1 - mask_rec_vis[sample_idx:sample_idx + 1]),
                        result_video[sample_idx:sample_idx + 1],
                        vessel_mask_rec[sample_idx:sample_idx + 1],
                    ],
                    dim=1,
                )
                val_pic_rec_flat = rearrange(val_pic_rec, "b f c h w -> (b f) c h w")
                recursive_image_path = os.path.join(
                    val_fold,
                    f"Epoch_{epoch + 1}_{stage_name}_recursive_full_idx_{sample_id:04d}.png",
                )
                save_image(
                    val_pic_rec_flat,
                    recursive_image_path,
                    nrow=num_frames_rec,
                    normalize=True,
                    value_range=(-1, 1),
                )
                recursive_images.append(
                    {
                        "path": recursive_image_path,
                        "caption": f"Epoch {epoch + 1} {stage_name} recursive {num_frames_rec} frames idx {sample_id}",
                    }
                )

            generated_images["Val Examples/Recursive"] = recursive_images

    torch.cuda.empty_cache()
    return generated_images
