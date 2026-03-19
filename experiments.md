# Training

Run from the repo root:

```bash
cd /Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL
```

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

Resume mask-pred training from the latest checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --task mask_pred --config configs/config_mask_pred.yaml
```

Then set `resume_from_checkpoint: true` in [config_mask_pred.yaml](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/configs/config_mask_pred.yaml).

Notes:

- The current default mask-pred setup is `single`-only training.
- The current default validation protocol is also `single`-only: mask one internal observed frame and use all other observed frames as context.
- Vessel-weighted loss is available through `vessel_mask.enable`, but the default is `false`.

# Testing

Use [test.py](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/test.py) as the unified testing entrypoint.

- `infer` runs a checkpoint or baseline on a dataset split and saves generated outputs under `predictions/`.
- `eval` reads those saved outputs and writes `eval_summary.json`.

## Sliding-Triplet Testing

The `sliding_triplets` task uses the same protocol as training validation:

- for each real CTA sequence, build all consecutive 3-frame windows
- use the first and third frames as input
- predict the middle frame
- compute `MAE`, `MSE`, and `PSNR` only on the predicted middle frames

Run validation-split inference:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py infer --task sliding_triplets --config configs/config_cta.yaml --split val --output-dir tmp/sliding_triplets_val
```

Then evaluate the saved outputs:

```bash
python test.py eval --task sliding_triplets --input-dir tmp/sliding_triplets_val
```

Run test-split inference with a specific checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py infer --task sliding_triplets --config configs/config_cta.yaml --split test --ckpt best_epoch_train_model.pth --ckpt-mode raw --output-dir tmp/sliding_triplets_test_raw
```

Evaluate those saved test outputs:

```bash
python test.py eval --task sliding_triplets --input-dir tmp/sliding_triplets_test_raw
```

Use EMA weights instead of raw weights:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py infer --task sliding_triplets --config configs/config_cta.yaml --split test --ckpt best_epoch_train_model.pth --ckpt-mode ema --output-dir tmp/sliding_triplets_test_ema
```

Override the sampler respacing:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py infer --task sliding_triplets --config configs/config_cta.yaml --split test --ckpt best_epoch_train_model.pth --ckpt-mode raw --respacing ddim20 --output-dir tmp/sliding_triplets_test_ddim20
```

Run multi-GPU inference:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 test.py infer --task sliding_triplets --config configs/config_cta.yaml --split test --output-dir tmp/sliding_triplets_test_multi
```

Run baseline methods with the same protocol:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py infer --task sliding_triplets --config configs/config_cta.yaml --split val --baseline-method linear --output-dir tmp/sliding_triplets_linear
```

```bash
python test.py eval --task sliding_triplets --input-dir tmp/sliding_triplets_linear
```

```bash
CUDA_VISIBLE_DEVICES=0 python test.py infer --task sliding_triplets --config configs/config_cta.yaml --split val --baseline-method si_dis_flow --output-dir tmp/sliding_triplets_si_dis_flow
```

```bash
CUDA_VISIBLE_DEVICES=0 python test.py infer --task sliding_triplets --config configs/config_cta.yaml --split val --baseline-method bi_dis_flow --output-dir tmp/sliding_triplets_bi_dis_flow
```

Notes:

- `quadratic` is not supported in this protocol because its original implementation uses the center frame as an interpolation anchor.
- `--split val` reads the last 10% of `data_path_train`, matching training validation.
- `--split test` reads `data_path_test`.
- If `--output-dir` is omitted, infer outputs are written under the experiment directory in `test/sliding_triplets/<split>/...`.

## Mask-Pred Testing

The `mask_pred` task supports two inference modes:

- `reconstruct`
  - save held-out sparse-frame reconstructions for later offline eval
- `densify`
  - export denser generated sequences from sparse observed anchors

Run reconstruction inference on the validation split:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py infer --task mask_pred --config configs/config_mask_pred.yaml --mode reconstruct --stage val --output-dir tmp/mask_pred_reconstruct_val
```

Then evaluate the saved reconstruction outputs:

```bash
python test.py eval --task mask_pred --input-dir tmp/mask_pred_reconstruct_val
```

Run reconstruction inference on the test split with a specific checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py infer --task mask_pred --config configs/config_mask_pred.yaml --mode reconstruct --stage test --ckpt best_mask_pred.pth --output-dir tmp/mask_pred_reconstruct_test
```

Evaluate those saved test outputs:

```bash
python test.py eval --task mask_pred --input-dir tmp/mask_pred_reconstruct_test
```

Force raw or EMA weights during reconstruction inference:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py infer --task mask_pred --config configs/config_mask_pred.yaml --mode reconstruct --stage test --ckpt best_mask_pred.pth --use-ema false --output-dir tmp/mask_pred_reconstruct_raw
```

```bash
CUDA_VISIBLE_DEVICES=0 python test.py infer --task mask_pred --config configs/config_mask_pred.yaml --mode reconstruct --stage test --ckpt best_mask_pred.pth --use-ema true --output-dir tmp/mask_pred_reconstruct_ema
```

Run multi-GPU reconstruction inference:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 test.py infer --task mask_pred --config configs/config_mask_pred.yaml --mode reconstruct --stage test --output-dir tmp/mask_pred_reconstruct_test_multi
```

Run densification export at `1 FPS` on the test split:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py infer --task mask_pred --config configs/config_mask_pred.yaml --mode densify --stage test --densify-fps 1.0 --output-dir tmp/mask_pred_densify_test
```

Export only a limited number of densified samples:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py infer --task mask_pred --config configs/config_mask_pred.yaml --mode densify --stage test --densify-fps 1.0 --max-samples 16 --output-dir tmp/mask_pred_densify_test_small
```

Notes:

- Reconstruction `eval` reads the saved `.pt` outputs from `infer` and writes `eval_summary.json`.
- Reconstruction `infer` uses the current mask-pred validation protocol from [config_mask_pred.yaml](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/configs/config_mask_pred.yaml). With the current default config, this is `single`-frame masking only.
- `--stage val` uses the same deterministic train/val split as mask-pred training.
- `--stage test` reads `data_path_test`.
- `densify` mode has no offline `eval` step because no dense GT sequence is available.
- If `--output-dir` is omitted, infer outputs are written under the corresponding mask-pred experiment directory in `test/mask_pred/...`.
