# Training

Run from the repo root:

```bash
cd /Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL
```

Result layout:

- training runs: `results/runs/{task}/...`
- evaluation outputs: `results/eval/{task}/...`

Use [train.py](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/train.py) as the unified training entrypoint.
Select the setup with `--task curriculum` or `--task mask_pred`.

Before starting curriculum training, update these paths in [config_cta.yaml](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/configs/config_cta.yaml):

- `data_path_train`
- `data_path_test`
- `pretrained_vae_model_path`

`pretrained_vae_model_path` must contain the `sd-vae-ft-mse` subfolder because training loads:

```python
AutoencoderKL.from_pretrained(args.pretrained_vae_model_path, subfolder="sd-vae-ft-mse")
```

Recommended single-GPU curriculum training command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --task curriculum --config configs/config_cta.yaml --log-level INFO
```

Detailed debug logging:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --task curriculum --config configs/config_cta.yaml --log-level DEBUG
```

Recommended multi-GPU curriculum training command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py --task curriculum --config configs/config_cta.yaml
```

Notes:

- CUDA is required. The script asserts `torch.cuda.is_available()`.
- `gpu_id` and `test_gpu_id` in the YAML are not used by the training launcher. GPU selection is controlled by `CUDA_VISIBLE_DEVICES` / `torchrun`.
- If you want to change the triplet loss target, set `triplet_loss_mode` in [config_cta.yaml](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/configs/config_cta.yaml) to `auto`, `epsilon`, or `x0`.

# Mask Prediction Training

Before starting mask-pred training, update these paths in [config_mask_pred.yaml](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/configs/config_mask_pred.yaml):

- `data_path_train`
- `data_path_test`
- `pretrained_vae_model_path`
- `mask_prediction.timing_csv_path`

`pretrained_vae_model_path` must contain the `sd-vae-ft-mse` subfolder because training loads:

```python
AutoencoderKL.from_pretrained(args.pretrained_vae_model_path, subfolder="sd-vae-ft-mse")
```

Recommended single-GPU mask-pred training command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --task mask_pred --config configs/config_mask_pred.yaml --log-level INFO
```

Recommended 8-GPU mask-pred training command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py --task mask_pred --config configs/config_mask_pred.yaml
```

Smaller multi-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py --task mask_pred --config configs/config_mask_pred.yaml
```

Resume mask-pred training from a chosen checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --task mask_pred --config configs/config_mask_pred_resume_step_0064000.yaml
```

Recommended resume policy:

- use `step_0064000.pth` for a real continuation point before the old run became unstable
- do not use `best_mask_pred.pth` as an exact resume checkpoint because it does not store optimizer, LR scheduler, or AMP scaler state
- use an absolute path in `resume_from_checkpoint`
- keep `results_dir: "results/runs/mask_pred"` and `cur_date: MASK_PRED_V1` if you want the resumed run to continue writing into `results/runs/mask_pred/MVIF-L-2_MASK_PRED_V1`

Draft resume config:

- [config_mask_pred_resume_step_0064000.yaml](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/configs/config_mask_pred_resume_step_0064000.yaml)

Server-side example after moving the old run into the latest path layout:

```bash
cd /home/yjwang/projects/TSSC-CTA-CL

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
train.py --task mask_pred --config configs/config_mask_pred_resume_step_0064000.yaml
```

If you want the resumed training to write into a new folder instead of reusing `MVIF-L-2_MASK_PRED_V1`, copy the resume config and change only `cur_date`, for example:

```yaml
cur_date: MASK_PRED_V1_RESUME_STEP64000
```

Warm-start from the best checkpoint instead of exact resume:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
train.py --task mask_pred --config configs/config_mask_pred.yaml
```

Then set:

```yaml
resume_from_checkpoint: false
pretrained: "/home/yjwang/projects/TSSC-CTA-CL/results/runs/mask_pred/MVIF-L-2_MASK_PRED_V1/checkpoints/best_mask_pred.pth"
cur_date: MASK_PRED_V1_WARMSTART_BEST
```

Notes:

- The current default mask-pred setup is `single`-only training.
- The current default validation protocol is also `single`-only: mask one internal observed frame and use all other observed frames as context.
- Vessel-weighted loss is available through `vessel_mask.enable`, but the default is `false`.
- The latest trainer now aborts after repeated non-finite steps instead of silently continuing through NaNs.

# Testing

Use [test.py](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/test.py) as the unified testing entrypoint.

- `infer` runs a checkpoint or baseline on a dataset split and saves generated outputs under `predictions/`.
- `eval` reads those saved outputs and writes `eval_summary.json`.
- `visualize` reads saved reconstruction outputs and exports per-slice GT/pred overview figures.

## Testing Summary

Current runnable baselines:

- `sliding_triplets`: 3 runnable baselines
  - `linear`
  - `si_dis_flow`
  - `bi_dis_flow`
- `mask_pred reconstruct`: 4 runnable baselines
  - `nearest`
  - `linear`
  - `si_dis_flow`
  - `bi_dis_flow`

Important:

- `quadratic` exists historically on the triplet side but is **not** supported in the current sliding-triplet protocol.
- `sliding_triplets` baselines and `mask_pred` baselines are **not** the same test protocol.
- If you want to compare against the current mask-pred model, use the `mask_pred` reconstruct protocol, not `sliding_triplets`.

## Sliding-Triplet Testing

Protocol:

- build all consecutive 3-frame windows from each real sequence
- use the first and third frames as input
- predict the middle frame
- compute `MAE`, `MSE`, and `PSNR` only on the predicted middle frames

### Model checkpoint

Validation split:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py infer --task sliding_triplets --config configs/config_cta.yaml --split val --output-dir results/eval/sliding_triplets/val/model_default
python test.py eval --task sliding_triplets --input-dir results/eval/sliding_triplets/val/model_default
```

Test split with a specific checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py infer --task sliding_triplets --config configs/config_cta.yaml --split test --ckpt best_epoch_train_model.pth --ckpt-mode raw --output-dir results/eval/sliding_triplets/test/best_epoch_train_model_raw
python test.py eval --task sliding_triplets --input-dir results/eval/sliding_triplets/test/best_epoch_train_model_raw
```

Use EMA weights instead of raw weights:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py infer --task sliding_triplets --config configs/config_cta.yaml --split test --ckpt best_epoch_train_model.pth --ckpt-mode ema --output-dir results/eval/sliding_triplets/test/best_epoch_train_model_ema
```

Override sampler respacing:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py infer --task sliding_triplets --config configs/config_cta.yaml --split test --ckpt best_epoch_train_model.pth --ckpt-mode raw --respacing ddim20 --output-dir results/eval/sliding_triplets/test/best_epoch_train_model_raw_ddim20
```

Multi-GPU inference:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 test.py infer --task sliding_triplets --config configs/config_cta.yaml --split test --output-dir results/eval/sliding_triplets/test/model_multi_gpu
```

### Baselines

Linear:

```bash
python test.py infer --task sliding_triplets --config configs/config_cta.yaml --split val --baseline-method linear --output-dir results/eval/sliding_triplets/val/baseline_linear
python test.py eval --task sliding_triplets --input-dir results/eval/sliding_triplets/val/baseline_linear
```

Single-direction DIS flow:

```bash
python test.py infer --task sliding_triplets --config configs/config_cta.yaml --split val --baseline-method si_dis_flow --output-dir results/eval/sliding_triplets/val/baseline_si_dis_flow
python test.py eval --task sliding_triplets --input-dir results/eval/sliding_triplets/val/baseline_si_dis_flow
```

Bidirectional DIS flow:

```bash
python test.py infer --task sliding_triplets --config configs/config_cta.yaml --split val --baseline-method bi_dis_flow --output-dir results/eval/sliding_triplets/val/baseline_bi_dis_flow
python test.py eval --task sliding_triplets --input-dir results/eval/sliding_triplets/val/baseline_bi_dis_flow
```

Notes:

- `quadratic` is not supported because its original implementation uses the center frame as an interpolation anchor.
- `--split val` uses the deterministic train/val split from `data_path_train`.
- `--split test` reads `data_path_test`.
- These baselines are only comparable to the `sliding_triplets` task.
- Sliding-triplet baselines now run on a CPU-only single-process path; do not launch them with `torchrun`.
- If `--output-dir` is omitted, outputs are written under `results/eval/sliding_triplets/<split>/...`.

## Mask-Pred Testing

The `mask_pred` task supports two inference modes:

- `reconstruct`
  - save held-out sparse-frame reconstructions for offline `eval` and `visualize`
- `densify`
  - export denser generated sequences from sparse observed anchors

Supported reconstruction protocols:

- `single_mid`
  - mask one deterministic internal frame near the middle of the sequence
- `single_all`
  - iterate over all internal observed frames, save one record per target frame, then aggregate offline

`single` is kept as an alias of `single_mid` for backward compatibility.

### Model checkpoint

Default behavior:

- if `--ckpt` is omitted, `mask_pred` testing uses `best_mask_pred.pth` if it exists, otherwise `latest_mask_pred.pth`
- relative checkpoint names such as `best_mask_pred.pth` are resolved under the experiment `checkpoints/` directory
- absolute checkpoint paths are also supported
- in the examples below, eval outputs are nested under the run name `MVIF-L-2_MASK_PRED_V1`; replace that path component if you are evaluating a different run

Validation split with the current default-style protocol:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py infer --task mask_pred --config configs/config_mask_pred.yaml --mode reconstruct --stage val --mask-mode single_mid --output-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/val/model_single_mid
python test.py eval --task mask_pred --input-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/val/model_single_mid --roi-mask-root data/TSSC_stage1_masks_output_2D
python test.py visualize --task mask_pred --input-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/val/model_single_mid --output-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/val/model_single_mid_figures
```

Test split with a specific saved checkpoint:

```bash
torchrun --standalone --nproc_per_node=8 test.py infer --task mask_pred --config configs/config_mask_pred.yaml --mode reconstruct --stage test --mask-mode single_all --ckpt best_mask_pred.pth --use-ema true --output-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/best_mask_pred_ema_single_all
python test.py eval --task mask_pred --input-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/best_mask_pred_ema_single_all --roi-mask-root data/TSSC_stage1_masks_output_2D
python test.py visualize --task mask_pred --input-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/best_mask_pred_ema_single_all --output-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/best_mask_pred_ema_single_all_figures
```

Force raw or EMA weights:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py infer --task mask_pred --config configs/config_mask_pred.yaml --mode reconstruct --stage test --mask-mode single_all --ckpt best_mask_pred.pth --use-ema false --output-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/best_mask_pred_raw_single_all
```

```bash
CUDA_VISIBLE_DEVICES=0 python test.py infer --task mask_pred --config configs/config_mask_pred.yaml --mode reconstruct --stage test --mask-mode single_all --ckpt best_mask_pred.pth --use-ema true --output-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/best_mask_pred_ema_single_all
```

Multi-GPU reconstruction inference:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 test.py infer --task mask_pred --config configs/config_mask_pred.yaml --mode reconstruct --stage test --mask-mode single_all --output-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/model_single_all_multi_gpu
```

### Baselines Under the Same Reconstruct Protocol

Nearest-frame baseline:

```bash
python test.py infer --task mask_pred --config configs/config_mask_pred.yaml --mode reconstruct --stage test --mask-mode single_all --baseline-method nearest --output-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/baseline_nearest_single_all
python test.py eval --task mask_pred --input-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/baseline_nearest_single_all --roi-mask-root data/TSSC_stage1_masks_output_2D
python test.py visualize --task mask_pred --input-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/baseline_nearest_single_all --output-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/baseline_nearest_single_all_figures
```

Linear baseline:

```bash
python test.py infer --task mask_pred --config configs/config_mask_pred.yaml --mode reconstruct --stage test --mask-mode single_all --baseline-method linear --output-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/baseline_linear_single_all
python test.py eval --task mask_pred --input-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/baseline_linear_single_all --roi-mask-root data/TSSC_stage1_masks_output_2D
python test.py visualize --task mask_pred --input-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/baseline_linear_single_all --output-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/baseline_linear_single_all_figures
```

Single-direction DIS flow baseline:

```bash
python test.py infer --task mask_pred --config configs/config_mask_pred.yaml --mode reconstruct --stage test --mask-mode single_all --baseline-method si_dis_flow --output-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/baseline_si_dis_flow_single_all
python test.py eval --task mask_pred --input-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/baseline_si_dis_flow_single_all --roi-mask-root data/TSSC_stage1_masks_output_2D
python test.py visualize --task mask_pred --input-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/baseline_si_dis_flow_single_all --output-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/baseline_si_dis_flow_single_all_figures
```

Bidirectional DIS flow baseline:

```bash
python test.py infer --task mask_pred --config configs/config_mask_pred.yaml --mode reconstruct --stage test --mask-mode single_all --baseline-method bi_dis_flow --output-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/baseline_bi_dis_flow_single_all
python test.py eval --task mask_pred --input-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/baseline_bi_dis_flow_single_all --roi-mask-root data/TSSC_stage1_masks_output_2D
python test.py visualize --task mask_pred --input-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/baseline_bi_dis_flow_single_all --output-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/reconstruct/test/baseline_bi_dis_flow_single_all_figures
```

### Densify Export

Export densified sequences at `1 FPS`:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py infer --task mask_pred --config configs/config_mask_pred.yaml --mode densify --stage test --densify-fps 1.0 --output-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/densify/test/densify_1fps
```

Limit the number of exported samples:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py infer --task mask_pred --config configs/config_mask_pred.yaml --mode densify --stage test --densify-fps 1.0 --max-samples 16 --output-dir results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/densify/test/densify_1fps_small
```

Notes:

- `reconstruct` is the main quantitative testing path for mask-pred.
- `eval` and `visualize` both consume the saved `.pt` outputs from `infer`; neither reruns inference.
- `eval` commands above use `--roi-mask-root data/TSSC_stage1_masks_output_2D`, so `MAE/MSE/PSNR` are computed inside the union vessel ROI, while `SSIM/LPIPS/FID` are computed on the full frame.
- `visualize` writes one overview figure per slice and aggregates all saved targets for that slice.
- `--stage val` uses the same deterministic train/val split as mask-pred training.
- `--stage test` reads `data_path_test`.
- `--baseline-method` is only supported with `--mode reconstruct`.
- The current mask-pred baselines are `nearest`, `linear`, `si_dis_flow`, and `bi_dis_flow`.
- Mask-pred baselines now run on a CPU-only single-process path; do not launch them with `torchrun`.
- `densify` has no offline numeric `eval` because no dense GT sequence is available.
- If `--output-dir` is omitted, outputs are written under `results/eval/mask_pred/...`, but the recommended practice is to include the run name explicitly, for example `results/eval/mask_pred/MVIF-L-2_MASK_PRED_V1/...`.
