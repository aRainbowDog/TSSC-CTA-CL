# Mask Prediction Experiment Protocol

## 1. Task Definition

This experiment line treats sparse CTP/4D CTA slice sequences as a **time-aware masked sequence reconstruction** problem.

Observed data per sample:

- A slice video with about `12-15` sparse observed frames.
- Real acquisition timestamps for each observed frame, aligned by:
  - `slice_XX -> slice_index_in_volume`
  - frame order `0..14 -> pair_index 0..14`

Training goal:

- Randomly hide part of the observed sparse sequence.
- Condition on the remaining visible observed frames plus their real timestamps.
- Reconstruct the hidden observed frames.

Final inference goal:

- Use the whole observed sparse sequence as anchors.
- Insert denser query timestamps such as `1 FPS`.
- Generate plausible intermediate frames jointly.

Important limitation:

- We do **not** have dense GT at `1 FPS`.
- Therefore, only held-out sparse-frame reconstruction is quantitatively evaluable.
- Densification is a generation task, not a supervised interpolation benchmark.

## 2. Data And Time Conditioning

Dataset implementation:

- `dataloader/data_loader_mask.py`
- `dataloader/ctp_timing.py`

Each batch sample returns:

- `video`
- `frame_times`
- `frame_times_relative`
- `frame_times_normalized`
- `frame_valid_mask`
- `dataset_index`
- sample metadata such as `video_name`, `patient_label`, `slice_index`

Default time field:

- `normalized_time`

Why `normalized_time` is the default:

- It aligns enhancement phase patterns better across patients.
- It reduces patient-specific timing scale variation.
- It is more suitable for learning phase-conditioned appearance dynamics.

Still retained:

- `relative_time_seconds`

Recommended interpretation:

- `normalized_time` is the default modeling time coordinate.
- `relative_time_seconds` is kept for ablations and future hybrid conditioning.

## 3. Model Conditioning

Model:

- `models/model_dit.py`

The mask-pred path uses:

- the normal diffusion timestep embedding for noise step `t`
- plus a separate **continuous frame-time embedding** for physical acquisition time

Behavior:

- If `frame_times` is provided, the model uses real continuous-time embedding.
- If `frame_times` is absent, it falls back to old slot/index-based temporal embedding.

This keeps backward compatibility with the curriculum training path.

## 4. Training Protocol

Training entrypoint:

- `train.py --task mask_pred`

Core objective:

- sequence-level masked latent reconstruction

Pipeline:

1. Load the full sparse observed sequence.
2. Encode all observed frames into VAE latent space.
3. Sample a frame-level mask over valid observed frames.
4. Replace masked latent slots with diffusion noise.
5. Feed visible latents, masked latents, and `frame_times` into the model.
6. Compute loss only on masked frames.

Loss implementation:

- `training/losses_mask_pred.py`

Current default training mask mode:

- `single_frame_mask_prob = 1.0`
  - mask exactly one observed frame
- `contiguous_span_mask_prob = 0.0`
- `anchor_drop_mask_prob = 0.0`

Current default single-frame sampling rule:

- only sample from internal observed frames
- do not mask the first or last observed frame unless no internal frame exists

Other mask modes still exist in code for ablation:

- contiguous span masking
- anchor-drop masking

Current default objective target:

- `loss_target_mode: auto`

This follows the configured diffusion parameterization.

Optional training-only vessel weighting:

- available through `vessel_mask.enable`
- default is `false`
- when enabled, it applies a soft spatial weight map to the masked-frame loss in latent space
- it does not change validation/test metrics

## 5. EMA Protocol

EMA is enabled during mask-pred training.

Relevant code:

- `utils/utils.py`
- `training/runtime.py`
- `training/tasks/mask_pred.py`
- `testing/tasks/mask_pred.py`

Current behavior:

- `ema` is initialized as a deep copy of the train model.
- After model initialization and optional pretrained loading, EMA is hard-synced once with decay `0.0`.
- After each optimizer step, EMA is updated with a warmup decay schedule from `get_ema_decay(...)`.
- Validation defaults to EMA weights.
- Test/inference can choose EMA or raw model weights.

Important fixes already applied:

1. `torch.compile` / DDP wrapper compatibility

- The runtime helper now unwraps compiled and DDP-wrapped models before checkpointing or EMA syncing.
- This prevents EMA/checkpoint code from accidentally reading wrapper state instead of the actual model state.

2. EMA update now uses full `state_dict()`

- Previous EMA logic only updated `named_parameters()`.
- That is fragile because it ignores buffers.
- Current EMA update iterates over the full `state_dict()`:
  - floating tensors use EMA update
  - non-floating tensors are copied exactly

Practical conclusion:

- The current EMA path is now structurally correct for this codebase.
- I did not find another confirmed EMA bug beyond the wrapper-unwrapping issue and the old parameter-only update.

## 6. Validation Protocol

Validation is **not** triplet-based anymore.

Validation code:

- `training/validation_mask_pred.py`
- invoked from `training/tasks/mask_pred.py`

Validation split:

- `stage="val"`

Validation cadence:

- controlled by `val_interval`

Validation objective:

- held-out sparse-frame reconstruction on the full sparse sequence

Procedure:

1. Encode the entire sparse sequence into latent space.
2. Apply a deterministic validation mask pattern.
3. Run conditional diffusion reconstruction over masked slots.
4. Decode reconstructed latents back to image space.
5. Compute metrics only on masked valid frames.

Validation mask modes:

- `single`
  - hide one center-like observed frame
- `span`
  - hide a contiguous short span of observed frames
- `anchor`
  - keep sparse anchors and hide the rest

Configured in:

- `configs/config_mask_pred.yaml`

Default validation mode:

- `["single"]`

Default validation single-frame rule:

- keep all other observed frames as context
- mask one internal observed frame
- compute metrics only on that masked frame

## 7. Validation Metrics

Metrics are computed only on frames that satisfy:

- `frame_mask == 1`
- `frame_valid_mask == 1`

Per-mode metrics:

- `MAE`
- `MSE`
- `PSNR`
- `SSIM`

Overall summary contains:

- `macro_mae`
- `macro_mse`
- `macro_psnr`
- `macro_ssim`
- `macro_metric`
- `weighted_mae`
- `weighted_mse`
- `weighted_psnr`
- `weighted_ssim`
- `masked_frame_count`

Definition notes:

- `macro_*`
  - average across validation mask modes
- `weighted_*`
  - average over all masked frames pooled together

Current best-checkpoint selection criterion:

- `macro_metric = mean_over_modes((MAE + MSE) / 2)`

Important:

- `SSIM` is logged and saved, but it is **not** currently used for best-checkpoint selection.
- This was kept intentionally to avoid silently changing the historical ranking rule.

## 8. Validation Visualization

Validation visualization is exported on fixed sample indices.

Visualization rows:

- GT sequence
- masked input sequence
- reconstructed sequence
- binary mask row

Purpose:

- verify whether the model reconstructs hidden observed frames coherently
- inspect failure modes under different mask patterns

## 9. Test Protocol

Test entrypoint:

- `test.py infer --task mask_pred`

Offline metric entrypoint:

- `test.py eval --task mask_pred`

There are two test modes.

### 9.1 Reconstruction Mode

Command-level intent:

- quantitatively evaluate held-out sparse-frame reconstruction

Mode:

- `--mode reconstruct`

Supported split:

- `--stage val`
- `--stage test`

Procedure:

1. Load a checkpoint.
2. Load the full sparse sequence with timestamps.
3. Apply one or more deterministic validation/test mask modes.
4. Reconstruct the hidden observed frames and save per-sample outputs.
5. Run offline eval on the saved outputs.
6. Compute `MAE/MSE/PSNR/SSIM` only on the hidden observed frames.
7. Export `eval_summary.json` and optional visualizations.

This is the main quantitative evaluation protocol for the mask-pred model.

### 9.2 Densification Mode

Command-level intent:

- generate denser sequences from sparse observed anchors

Mode:

- `--mode densify`

Procedure:

1. Load the sparse observed sequence and observed timestamps.
2. Build a denser query timeline, for example `1 FPS`.
3. Merge observed times and query times.
4. Mark observed slots as visible anchors.
5. Mark query slots as masked positions to generate.
6. Run joint conditional reconstruction over the merged timeline.
7. Export generated sequences and metadata JSON.

Important limitation:

- Densification mode does **not** have dense GT.
- Therefore, densification mode is currently for:
  - qualitative review
  - downstream analysis
  - future uncertainty or TIC-based evaluation

It is **not** a direct accuracy benchmark against dense truth.

## 10. Checkpoints

Training checkpoint directory:

- `<experiment_dir>/checkpoints`

Checkpoint types:

- `latest_mask_pred.pth`
  - latest resumable checkpoint
- `step_XXXXXXX.pth`
  - periodic training checkpoints
- `best_mask_pred.pth`
  - best validation checkpoint

Stored training state:

- model weights
- EMA weights
- optimizer state
- LR scheduler state
- AMP scaler state
- `train_steps`
- epoch metadata
- best validation metric

Resume behavior:

- `resume_from_checkpoint=true`
  - resume from `latest_mask_pred.pth`
- or provide an explicit checkpoint path

## 11. Tracker Logging

Supported backends:

- `wandb`
- `tensorboard`

Logged training signals:

- train loss
- masked frames per sample
- gradient norm
- learning rate
- EMA decay

Logged validation signals:

- overall weighted metrics
- per-mode metrics
- validation images
- best-checkpoint artifact
- train/val split manifest

## 12. Vessel Mask Status

Current status for the mask-pred pipeline:

- training can optionally use vessel-weighted loss through a config switch
- default setting is still **off**
- validation does **not** use vessel-weighted metrics
- test/inference does **not** use vessel-weighted metrics

The vessel-mask logic currently exists only in the older curriculum-learning path:

- `training/vessel_mask.py`
- `training/tasks/curriculum.py`

Implication:

- the default mask-pred baseline still optimizes all pixels uniformly
- vessel emphasis is now an ablation, not a default behavior

## 13. Recommended Reporting Table

For reconstruction experiments, report at least:

- checkpoint name
- time field used
- validation mask modes
- macro metric
- weighted MAE
- weighted MSE
- weighted PSNR
- weighted SSIM
- whether EMA was used

For densification experiments, report at least:

- checkpoint name
- time field used
- target FPS
- number of observed anchors
- number of generated query frames
- qualitative cases
- downstream proxy metrics if available

## 14. Current Default Scripts

Training:

```bash
python train.py --task mask_pred --config configs/config_mask_pred.yaml
```

Reconstruction infer:

```bash
python test.py infer --task mask_pred --config configs/config_mask_pred.yaml --mode reconstruct --stage val --output-dir tmp/mask_pred_reconstruct_val
```

Offline reconstruction eval:

```bash
python test.py eval --task mask_pred --input-dir tmp/mask_pred_reconstruct_val
```

Densification export:

```bash
python test.py infer --task mask_pred --config configs/config_mask_pred.yaml --mode densify --stage test --densify-fps 1.0 --output-dir tmp/mask_pred_densify_test
```

## 15. Current Open Gaps

The current mask-pred experiment stack is complete enough to train, validate, resume, and test, but a few research-facing gaps remain:

- best-checkpoint ranking still ignores SSIM
- densification mode has no quantitative dense-GT benchmark
- no vessel-focused weighting in the mask-pred path
- no uncertainty estimation or multi-sample consistency reporting yet
- no downstream TIC/clinical proxy evaluation yet

These are research extensions, not blockers for running the current experiment line.
