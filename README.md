# Learning Temporal Super-Resolution for Sparse 4D CTA

This repository contains training and testing code for sparse CTA slice-sequence reconstruction.

It currently supports two experiment lines:

- `curriculum`
  - triplet-style curriculum learning for middle-frame reconstruction
- `mask_pred`
  - time-aware masked sequence reconstruction for sparse observed CTA sequences

Default artifact layout:

- training runs: `results/runs/{task}/...`
- evaluation outputs: `results/eval/{task}/...`

## Main Entry Points

Training uses the unified entrypoint:

- [train.py](train.py)

Testing uses the unified entrypoint:

- [test.py](test.py)

The testing workflow is split into two stages:

1. `infer`
   - run a checkpoint or baseline and save generated outputs
2. `eval`
   - read saved outputs and compute metrics
3. `visualize`
   - read saved reconstruction outputs and export per-slice overview figures

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Curriculum training:

```bash
python train.py --task curriculum --config configs/config_cta.yaml
```

Mask-pred training:

```bash
python train.py --task mask_pred --config configs/config_mask_pred.yaml
```

Sliding-triplet testing:

```bash
python test.py infer --task sliding_triplets --config configs/config_cta.yaml --split val --output-dir tmp/sliding_triplets_val
python test.py eval --task sliding_triplets --input-dir tmp/sliding_triplets_val
```

Mask-pred reconstruction testing:

```bash
python test.py infer --task mask_pred --config configs/config_mask_pred.yaml --mode reconstruct --stage val --mask-mode single_mid --output-dir tmp/mask_pred_reconstruct_val
python test.py eval --task mask_pred --input-dir tmp/mask_pred_reconstruct_val
python test.py visualize --task mask_pred --input-dir tmp/mask_pred_reconstruct_val --output-dir tmp/mask_pred_reconstruct_val_figures
```

Mask-pred checkpoint testing on the test split:

```bash
python test.py infer --task mask_pred --config configs/config_mask_pred.yaml --mode reconstruct --stage test --mask-mode single_all --ckpt best_mask_pred.pth --output-dir tmp/mask_pred_reconstruct_test
python test.py eval --task mask_pred --input-dir tmp/mask_pred_reconstruct_test
python test.py visualize --task mask_pred --input-dir tmp/mask_pred_reconstruct_test --output-dir tmp/mask_pred_reconstruct_test_figures
```

## Documentation

Start with these files:

- [experiments.md](experiments.md)
  - training and testing commands
- [documentation/README.md](documentation/README.md)
  - technical notes for model, testing, and output format
- [mask_pred_experiment_protocol.md](mask_pred_experiment_protocol.md)
  - detailed mask-pred design and evaluation protocol
- [data/ctp_slice_timing/README.md](data/ctp_slice_timing/README.md)
  - slice timing metadata notes

Useful configs:

- [configs/config_cta.yaml](configs/config_cta.yaml)
- [configs/config_mask_pred.yaml](configs/config_mask_pred.yaml)
- [configs/config_mask_pred_smoke.yaml](configs/config_mask_pred_smoke.yaml)

## Repository Layout

- [`training/`](training)
  - training utilities and task-specific training implementations
- [`testing/`](testing)
  - unified testing utilities and task-specific infer/eval implementations
- [`dataloader/`](dataloader)
  - dataset loaders and timing alignment
- [`models/`](models)
  - DiT / diffusion model code
- [`scripts/preprocess/`](scripts/preprocess)
  - standalone data preprocessing and format conversion scripts

## Preprocessing Scripts

The standalone data conversion scripts are under:

- [scripts/preprocess/mp4_to_nii.py](scripts/preprocess/mp4_to_nii.py)
- [scripts/preprocess/nii_to_mp4.py](scripts/preprocess/nii_to_mp4.py)
- [scripts/preprocess/resample.py](scripts/preprocess/resample.py)

These scripts are not part of the main train/test entrypoints; they are utility scripts for dataset preparation and conversion.
