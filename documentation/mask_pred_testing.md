# Mask-Pred Testing and Output Format

## 1. Unified test entrypoints

Mask-pred testing uses the unified [test.py](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/test.py) entrypoint with three decoupled actions:

- `infer`
  - run model or baseline inference
  - save `.pt` prediction records
- `eval`
  - read saved reconstruction records
  - compute offline `MAE`, `MSE`, `PSNR`, and `SSIM`
- `visualize`
  - read saved reconstruction records
  - export per-slice overview figures for qualitative comparison

This keeps inference, metric aggregation, and visualization decoupled.

## 2. Reconstruct protocols

The `mask_pred` reconstruct path supports two single-frame protocols:

- `single_mid`
  - mask one deterministic internal frame near the middle of the sequence
- `single_all`
  - iterate over all internal observed frames
  - save one record per target frame

The legacy `single` name is kept as an alias of `single_mid`.

The mask variants are generated in [training/validation_mask_pred.py](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/training/validation_mask_pred.py#L59).

## 3. Baselines under the same protocol

The `mask_pred` reconstruct path also supports baselines that use the same saved-record format and the same offline eval path.

Current baselines:

- `nearest`
- `linear`
- `si_dis_flow`
- `bi_dis_flow`

For each masked target frame, the baseline sees the same visible frames as the model.
The current baseline implementations predict each target frame from the nearest visible temporal neighbors.
These baselines now run on a CPU-only single-process path rather than loading CUDA/DDP.

The baseline logic is implemented in [testing/tasks/mask_pred.py](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/testing/tasks/mask_pred.py#L191).

## 4. Saved output layout

After `infer --task mask_pred --mode reconstruct`, the output directory contains:

```text
output_dir/
  manifest.json
  predictions/
    *.pt
```

If `--output-dir` is omitted, the default location is under:

```text
results/eval/mask_pred/reconstruct/<stage>/...
```

For `single_mid`, each slice usually produces one `.pt` record.

For `single_all`, each slice produces multiple `.pt` records, one per target frame variant.

## 5. Reconstruction record format

Each `.pt` reconstruction record stores:

- `task`
- `infer_mode`
- `stage`
- `mode_name`
- `mask_variant`
- `video_name`
- `video_path`
- `dataset_index`
- `baseline_method`
- `target_frame_indices`
- `target_frame_times_relative`
- `target_frame_times_normalized`
- `gt_target_frames`
- `pred_target_frames`

Important points:

- reconstruct outputs now use a `targets_only` layout by default
- `target_frame_indices` records which frame(s) this saved record corresponds to
- `gt_target_frames` and `pred_target_frames` store only the reconstructed target frames
- this makes `single_all` much lighter than storing the full sequence for every variant

This format allows offline metric computation and offline visualization without rerunning inference.

## 6. Offline eval

`test.py eval --task mask_pred` reads the saved `.pt` reconstruction records and computes metrics directly on:

- `gt_target_frames`
- `pred_target_frames`

The offline eval path now expects exactly one `mode_name` per input directory.
It writes a flattened single-mode `overall` summary with:

- `metric`
- `mae`
- `mse`
- `psnr`
- `ssim`
- `lpips`
- `fid`

When `--roi-mask-root` is provided, it only changes the pixel-wise metric scope:

- `mae`
- `mse`
- `psnr`

These three are computed inside the union vessel ROI, while:

- `ssim`
- `lpips`
- `fid`

remain full-frame metrics.

This logic is implemented in [testing/tasks/mask_pred.py](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/testing/tasks/mask_pred.py#L685) and uses the shared metric helpers from [training/validation_mask_pred.py](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/training/validation_mask_pred.py#L126).

## 7. Offline visualization

`test.py visualize --task mask_pred` reads the same saved reconstruction records and groups them by:

- slice / sample
- reconstruct mode
- baseline or model source

Then it exports one overview figure per slice.

Each overview figure contains:

- top row: GT target frames
- bottom row: predicted target frames

For `single_all`, all target frames for that slice are shown in the same figure.

The visualization exporter is implemented in [testing/tasks/mask_pred.py](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/testing/tasks/mask_pred.py#L262).

The visualization output directory contains:

```text
figures/
  overview/
    *.png
    *.json
  visualization_manifest.json
```

Where:

- each `.png` is one per-slice overview figure
- each `.json` stores the column metadata for that figure
- `visualization_manifest.json` summarizes all exported figures

## 8. Densify outputs

For `infer --task mask_pred --mode densify`, the output directory stores:

- `manifest.json`
- `predictions/*.pt`
- per-sample `.png`
- per-sample `.json`

If `--output-dir` is omitted, the default location is under:

```text
results/eval/mask_pred/densify/<stage>/...
```

There is no offline numeric `eval` for densify because no dense GT sequence is available.

## 9. Recommended workflow

For quantitative reconstruction testing:

1. run `infer`
2. run `eval`

For qualitative review:

1. run `infer`
2. run `visualize`

This allows the same inference outputs to be reused for both offline metrics and per-slice visual inspection.
