# Training

Run from the repo root:

```bash
cd /Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL
```

Before starting, update these paths in [config_cta.yaml](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/configs/config_cta.yaml):

- `data_path_train`
- `data_path_test`
- `pretrained_vae_model_path`

`pretrained_vae_model_path` must contain the `sd-vae-ft-mse` subfolder because training loads:

```python
AutoencoderKL.from_pretrained(args.pretrained_vae_model_path, subfolder="sd-vae-ft-mse")
```

Recommended single-GPU training command:

```bash
CUDA_VISIBLE_DEVICES=0 python train_CurriculumLearning.py --config configs/config_cta.yaml  --log-level INFO
```

If you want to suppress debug diagnostics, keep the default:

```bash
CUDA_VISIBLE_DEVICES=0 python train_CurriculumLearning.py --config configs/config_cta.yaml --log-level INFO
```

If you want the detailed debug traces, use:

```bash
CUDA_VISIBLE_DEVICES=0 python train_CurriculumLearning.py --config configs/config_cta.yaml --log-level DEBUG
```

Recommended multi-GPU training command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train_CurriculumLearning.py --config configs/config_cta.yaml
```

Notes:

- CUDA is required. The script asserts `torch.cuda.is_available()`.
- `gpu_id` and `test_gpu_id` in the YAML are not used by the training launcher. GPU selection is controlled by `CUDA_VISIBLE_DEVICES` / `torchrun`.
- If you want to change the triplet loss target, set `triplet_loss_mode` in [config_cta.yaml](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/configs/config_cta.yaml) to `auto`, `epsilon`, or `x0`.

# Mask Prediction Training

Before starting, update these paths in [config_mask_pred.yaml](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/configs/config_mask_pred.yaml):

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
CUDA_VISIBLE_DEVICES=0 python train_mask_pred.py --config configs/config_mask_pred.yaml --log-level INFO
```

Recommended 8-GPU mask-pred training command for the full experiment:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train_mask_pred.py --config configs/config_mask_pred.yaml
```

If you want a smaller multi-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train_mask_pred.py --config configs/config_mask_pred.yaml
```

Resume mask-pred training from the latest checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python train_mask_pred.py --config configs/config_mask_pred.yaml
```

Then set `resume_from_checkpoint: true` in [config_mask_pred.yaml](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/configs/config_mask_pred.yaml).

Notes:

- The current default mask-pred setup is `single`-only training.
- The current default validation protocol is also `single`-only: mask one internal observed frame and use all other observed frames as context.
- Vessel-weighted loss is available through `vessel_mask.enable`, but the default is `false`.

# Evaluation

The standalone evaluation script [eval_sliding_triplets.py](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/eval_sliding_triplets.py) uses the same sliding-triplet protocol as training validation:

- for each real CTA sequence, build all consecutive 3-frame windows
- use the first and third frames as input
- predict the middle frame
- compute `MAE`, `MSE`, and `PSNR` only on the middle frame

Run validation-style evaluation on the validation split:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_sliding_triplets.py --config configs/config_cta.yaml --split val
```

Run the same protocol on the test split:

```bash
CUDA_VISIBLE_DEVICES=7 python eval_sliding_triplets.py --config configs/config_cta.yaml --split test
```

Run the same protocol with multi-GPU parallel evaluation:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 eval_sliding_triplets.py --config configs/config_cta.yaml --split test
```

Evaluate a specific checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_sliding_triplets.py --config configs/config_cta.yaml --split test --ckpt best_epoch_train_model.pth --ckpt-mode raw
```

Evaluate EMA weights instead of raw weights:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_sliding_triplets.py --config configs/config_cta.yaml --split test --ckpt best_epoch_train_model.pth --ckpt-mode ema
```

Force the same sampler respacing as training validation:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_sliding_triplets.py --config configs/config_cta.yaml --split test --ckpt best_epoch_train_model.pth --ckpt-mode raw --respacing ddim20
```

Run baseline methods with the same sliding-triplet protocol:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_sliding_triplets.py --config configs/config_cta.yaml --split val --baseline-method linear
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 eval_sliding_triplets.py --config configs/config_cta.yaml --split val --baseline-method linear
```

```bash
CUDA_VISIBLE_DEVICES=0 python eval_sliding_triplets.py --config configs/config_cta.yaml --split val --baseline-method si_dis_flow
```

```bash
CUDA_VISIBLE_DEVICES=0 python eval_sliding_triplets.py --config configs/config_cta.yaml --split val --baseline-method bi_dis_flow
```

Notes:

- `quadratic` is not supported in this validation-style protocol because its original implementation uses the center frame as an interpolation anchor.
- `--split val` reads the last 10% of `data_path_train`, matching training validation.
- `--split test` reads `data_path_test`.
- By default, validation-style evaluation logs are written under the experiment directory in `validation_style_eval/<split>/`.

# Mask Prediction Evaluation

The standalone mask-pred evaluation script [infer_mask_pred.py](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/infer_mask_pred.py) supports two modes:

- `reconstruct`
  - quantitative held-out sparse-frame reconstruction
- `densify`
  - export denser generated sequences from sparse observed anchors

Run quantitative reconstruction evaluation on the validation split:

```bash
CUDA_VISIBLE_DEVICES=0 python infer_mask_pred.py --config configs/config_mask_pred.yaml --mode reconstruct --stage val
```

Run quantitative reconstruction evaluation on the test split:

```bash
CUDA_VISIBLE_DEVICES=0 python infer_mask_pred.py --config configs/config_mask_pred.yaml --mode reconstruct --stage test
```

Evaluate a specific mask-pred checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python infer_mask_pred.py --config configs/config_mask_pred.yaml --mode reconstruct --stage test --ckpt best_mask_pred.pth
```

Force raw or EMA weights during reconstruction evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python infer_mask_pred.py --config configs/config_mask_pred.yaml --mode reconstruct --stage test --ckpt best_mask_pred.pth --use-ema false
```

```bash
CUDA_VISIBLE_DEVICES=0 python infer_mask_pred.py --config configs/config_mask_pred.yaml --mode reconstruct --stage test --ckpt best_mask_pred.pth --use-ema true
```

Run multi-GPU reconstruction evaluation:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 infer_mask_pred.py --config configs/config_mask_pred.yaml --mode reconstruct --stage test
```

Run densification export at `1 FPS` on the test split:

```bash
CUDA_VISIBLE_DEVICES=0 python infer_mask_pred.py --config configs/config_mask_pred.yaml --mode densify --stage test --densify-fps 1.0
```

Export only a limited number of densified samples:

```bash
CUDA_VISIBLE_DEVICES=0 python infer_mask_pred.py --config configs/config_mask_pred.yaml --mode densify --stage test --densify-fps 1.0 --max-samples 16
```

Notes:

- Reconstruction evaluation uses the current mask-pred validation protocol from [config_mask_pred.yaml](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/configs/config_mask_pred.yaml). With the current default config, this is `single`-frame masking only.
- `--stage val` uses the same deterministic train/val split as mask-pred training.
- `--stage test` reads `data_path_test`.
- `densify` mode does not compute quantitative reconstruction metrics because no dense GT sequence is available.
- By default, reconstruction outputs and densified exports are written under the corresponding mask-pred experiment directory.
