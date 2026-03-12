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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --standalone --nproc_per_node=7 train_CurriculumLearning.py --config configs/config_cta.yaml
```

Notes:

- CUDA is required. The script asserts `torch.cuda.is_available()`.
- `gpu_id` and `test_gpu_id` in the YAML are not used by the training launcher. GPU selection is controlled by `CUDA_VISIBLE_DEVICES` / `torchrun`.
- If you want to change the triplet loss target, set `triplet_loss_mode` in [config_cta.yaml](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/configs/config_cta.yaml) to `auto`, `epsilon`, or `x0`.
