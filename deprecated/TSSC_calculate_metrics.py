import os
import argparse
import datetime
import warnings

import imageio
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL

from dataloader.data_loader_acdc import collate_full_sequence_batch, full_sequence_data_loader
from models.diffusion.gaussian_diffusion import create_diffusion
from models.model_dit import MVIF_models
from utils.triplet_eval import evaluate_video_sliding_triplets, tensor_video_to_uint8_numpy

warnings.filterwarnings('ignore')


def load_eval_state_dict(checkpoint, ckpt_mode):
    mode = str(ckpt_mode or "raw").lower()
    if mode == "ema" and "ema" in checkpoint:
        return checkpoint["ema"], "使用EMA权重！"
    return checkpoint["model"], "使用raw权重！"


def main(args):
    seed = args.global_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    model_string_name = args.model.replace("/", "-")
    experiment_dir = f"{args.results_dir}/{model_string_name}_{args.cur_date}"
    save_dir = f"{experiment_dir}/test_sliding_triplet_eval"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "GT"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "Pred"), exist_ok=True)

    metrics_log = os.path.join(save_dir, "sliding_triplet_eval.log")
    f_metrics = open(metrics_log, 'w', encoding='utf-8')
    f_metrics.write(f"滑窗三连帧测试日志 - {datetime.datetime.now()}\n")
    f_metrics.write(f"测试模型权重: {args.test_ckpt}\n")
    f_metrics.write(f"测试数据集路径: {args.data_path_test}\n\n")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.test_gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    triplet_eval_batch_size = int(getattr(args, "triplet_eval_batch_size", 4))

    diffusion = create_diffusion(
        timestep_respacing=args.timestep_respacing_test,
        diffusion_steps=args.diffusion_steps,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_model_path,
        subfolder="sd-vae-ft-mse",
    ).to(device)
    vae.requires_grad_(False)
    print(f"VAE加载完成: {args.pretrained_vae_model_path}")

    if not args.test_ckpt:
        raise ValueError("test_ckpt 不能为空，无法进行离线评测。")

    checkpoint_path = os.path.join(experiment_dir, "checkpoints", args.test_ckpt)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    latent_size = args.image_size // 8
    additional_kwargs = {'num_frames': args.tar_num_frames, 'mode': 'video'}
    model = MVIF_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        **additional_kwargs,
    ).to(device)

    state_dict, state_msg = load_eval_state_dict(checkpoint, getattr(args, "test_ckpt_mode", "raw"))
    print(state_msg)
    f_metrics.write(state_msg + "\n")

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"模型加载完成: {checkpoint_path}")
    print(f"Loaded keys: {len(pretrained_dict)} / {len(model_dict)}")
    if len(pretrained_dict) == 0:
        print("警告：模型权重完全没加载上！检查 state_dict 的 key。")

    dataset_test = full_sequence_data_loader(args, stage='test')
    loader_test = DataLoader(
        dataset_test,
        batch_size=int(args.test_batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_full_sequence_batch,
    )
    print(f"测试视频数量: {len(dataset_test)}")
    f_metrics.write(f"测试视频数量: {len(dataset_test)}\n")
    f_metrics.write(f"Triplet评测batch size: {triplet_eval_batch_size}\n\n")

    vae.eval()
    model.eval()
    diffusion.training = False

    total_mse = 0.0
    total_mae = 0.0
    total_psnr = 0.0
    total_triplet_count = 0
    total_video_count = 0

    pbar = tqdm(loader_test, total=len(loader_test), desc="滑窗三连帧测试")
    for batch in pbar:
        with torch.no_grad():
            for video, video_name, video_path in zip(
                    batch['videos'],
                    batch['video_name'],
                    batch['video_path']):
                print(f"正在测试: {video_path}")
                metrics = evaluate_video_sliding_triplets(
                    video,
                    model_forward=model.forward,
                    vae=vae,
                    diffusion=diffusion,
                    device=device,
                    triplet_batch_size=triplet_eval_batch_size,
                    return_pred_video=True,
                    force_grayscale=True,
                )

                video_triplet_count = metrics['count']
                if video_triplet_count <= 0:
                    continue

                video_mse = metrics['mse_sum'] / video_triplet_count
                video_mae = metrics['mae_sum'] / video_triplet_count
                video_psnr = metrics['psnr_sum'] / video_triplet_count

                total_mse += metrics['mse_sum']
                total_mae += metrics['mae_sum']
                total_psnr += metrics['psnr_sum']
                total_triplet_count += video_triplet_count
                total_video_count += 1

                pbar.set_postfix({
                    "PSNR": f"{video_psnr:.4f}",
                    "MAE": f"{video_mae:.4f}",
                    "Triplets": video_triplet_count,
                })

                f_metrics.write(f"视频 {total_video_count} - {video_name}\n")
                f_metrics.write(f"  Triplets: {video_triplet_count}\n")
                f_metrics.write(f"  MSE: {video_mse:.6f}\n")
                f_metrics.write(f"  MAE: {video_mae:.6f}\n")
                f_metrics.write(f"  PSNR: {video_psnr:.4f}\n\n")

                gt_video = tensor_video_to_uint8_numpy(metrics['gt_video'])
                pred_video = tensor_video_to_uint8_numpy(metrics['pred_video'])
                video_filename = f"{video_name}.mp4"
                imageio.mimwrite(os.path.join(save_dir, 'GT', video_filename), gt_video, fps=2, codec='libx264')
                imageio.mimwrite(os.path.join(save_dir, 'Pred', video_filename), pred_video, fps=2, codec='libx264')

    avg_total_mse = total_mse / max(total_triplet_count, 1)
    avg_total_mae = total_mae / max(total_triplet_count, 1)
    avg_total_psnr = total_psnr / max(total_triplet_count, 1)
    proxy_metric = (avg_total_mae + avg_total_mse) / 2

    print("\n==================== 滑窗三连帧测试集平均指标 ====================")
    print(f"视频数量: {total_video_count}")
    print(f"Triplet数量: {total_triplet_count}")
    print(f"平均MSE: {avg_total_mse:.6f}")
    print(f"平均MAE: {avg_total_mae:.6f}")
    print(f"平均PSNR: {avg_total_psnr:.4f}")
    print(f"平均Metric: {proxy_metric:.6f}")
    print("==============================================================")

    f_metrics.write("==================== 滑窗三连帧测试集平均指标 ====================\n")
    f_metrics.write(f"视频数量: {total_video_count}\n")
    f_metrics.write(f"Triplet数量: {total_triplet_count}\n")
    f_metrics.write(f"平均MSE: {avg_total_mse:.6f}\n")
    f_metrics.write(f"平均MAE: {avg_total_mae:.6f}\n")
    f_metrics.write(f"平均PSNR: {avg_total_psnr:.4f}\n")
    f_metrics.write(f"平均Metric: {proxy_metric:.6f}\n")
    f_metrics.write("==============================================================\n")
    f_metrics.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_cta.yaml")
    cli_args = parser.parse_args()
    main(OmegaConf.load(cli_args.config))
