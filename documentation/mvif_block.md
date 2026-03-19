# MVIF Backbone and MVIFBlock

## 1. What `MVIF` is in this repository

In this repository, `MVIF` is the main diffusion transformer backbone defined in [models/model_dit.py](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/models/model_dit.py).

From the file header:

- the implementation is originally inspired by DiT
- it was modified in this repository for video diffusion transformer use

So the most accurate description is:

- `MVIF` is a repository-specific video diffusion transformer backbone
- `MVIFBlock` is a video-oriented transformer block built on top of DiT-style conditioning

This is not presented in the codebase as a standard public module name with a clearly cited standalone paper.

## 2. High-level model flow

The full model pipeline is:

```text
input video frames
-> VAE encoder
-> latent video sequence
-> patch embedding
-> spatial position embedding
-> frame-time embedding
-> stacked MVIF blocks
-> final projection
-> unpatchify
-> predicted latent video sequence
-> optional VAE decoder
-> reconstructed image frames
```

The main `MVIF.forward()` path is implemented in [models/model_dit.py](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/models/model_dit.py#L393).

## 3. Symbols and shapes

To describe the tensor flow, use:

- `B`: batch size
- `T`: number of frames in the sequence
- `C_latent`: latent channels from the VAE
- `H, W`: latent spatial resolution
- `P`: patch size
- `N`: number of spatial patches per frame
- `D`: transformer hidden size

Where:

```text
N = (H / P) * (W / P)
```

For the common current setup:

- input latent shape: `[B, 15, 4, 32, 32]`
- patch size: `2`
- hidden size for `MVIF-L/2`: `1152`

Then:

- each `32 x 32` latent frame becomes `16 x 16 = 256` patches
- so `N = 256`

## 4. Input to the backbone

At the `MVIF.forward()` entry, the model receives:

- `x`: latent video sequence, shape `[B, T, C_latent, H, W]`
- `t`: diffusion timestep, shape `[B]`
- `frame_times`: frame timestamps, shape `[B, T]` when provided

Important distinction:

- `frame_times` is the physical acquisition time for each frame
- `t` is the diffusion denoising timestep

These are different kinds of time.

## 5. How frames become tokens

The model does not treat one frame as one token.

Instead:

1. all frames are flattened from `[B, T, C_latent, H, W]` to `[B*T, C_latent, H, W]`
2. each frame passes through `PatchEmbed`
3. each frame is split into `N` spatial patch tokens
4. the output becomes `[B*T, N, D]`

This happens in [models/model_dit.py](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/models/model_dit.py#L401) and [models/model_dit.py](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/models/model_dit.py#L404).

So:

- one frame corresponds to many tokens
- each token is one spatial patch from that frame

## 6. How time enters the model

There are two separate conditioning paths.

### 6.1 Frame acquisition time

If `frame_times` is provided:

1. the timestamps are encoded by `FrameTimeEmbedder`
2. this produces one embedding vector per frame
3. the frame-time embedding is added to all patch tokens belonging to that frame

This happens in [models/model_dit.py](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/models/model_dit.py#L408).

This means frame time is used as:

- a per-frame temporal condition
- injected directly into the token representation before the transformer blocks

### 6.2 Diffusion timestep

The diffusion timestep `t` is embedded separately by `TimestepEmbedder` and combined with the class embedding:

```text
t -> t_embedder -> [B, D]
y -> y_embedder -> [B, D]
c = t + y -> [B, D]
```

This happens in [models/model_dit.py](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/models/model_dit.py#L420).

The resulting `c` is the condition used by `adaLN-Zero` inside each block.

## 7. What enters a single `MVIFBlock`

After patch embedding and temporal embedding, the input to each block is:

- `x`: `[B*T, N, D]`
- `c`: `[B, D]`

Here:

- `x` is the patch-token representation of the whole video sequence
- `c` is the per-sample diffusion condition

The block itself is defined in [models/model_dit.py](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/models/model_dit.py#L178).

## 8. `MVIFBlock` structure

One `MVIFBlock` has three main computation stages:

1. temporal attention
2. spatial attention with adaLN modulation
3. MLP with adaLN modulation

The input and output shapes of the block are the same:

```text
input:  [B*T, N, D]
output: [B*T, N, D]
```

So this is a shape-preserving transformer block.

## 9. Step-by-step shape flow inside `MVIFBlock`

### 9.1 Generate modulation parameters from `c`

At the beginning of `forward()`:

```python
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    = self.adaLN_modulation(c).chunk(6, dim=1)
```

Input:

- `c: [B, D]`

Outputs:

- `shift_msa: [B, D]`
- `scale_msa: [B, D]`
- `gate_msa: [B, D]`
- `shift_mlp: [B, D]`
- `scale_mlp: [B, D]`
- `gate_mlp: [B, D]`

These are used to condition the spatial attention branch and MLP branch.

### 9.2 Temporal attention

Initial input:

```text
x: [B*T, N, D]
```

The first reshape is:

```text
[B*T, N, D] -> [(B*N), T, D]
```

This means:

- fix one spatial patch position
- collect its representation across all `T` frames
- run attention over the time dimension

So temporal attention operates on:

- sequence length = `T`
- batch-like dimension = `B*N`

After temporal attention:

```text
[(B*N), T, D] -> [(B*N), T, D]
```

Then it is reshaped back:

```text
[(B*N), T, D] -> [B*T, N, D]
```

Then a linear layer is applied and added back to the residual:

```text
x = x + res_temporal
```

Output after temporal branch:

```text
[B*T, N, D]
```

Interpretation:

- each spatial location can communicate with the same spatial location at other frames
- this is the main source of temporal modeling in the block

### 9.3 Spatial attention with adaLN

After temporal mixing, the block performs spatial self-attention.

Input:

```text
x: [B*T, N, D]
```

First:

```text
x -> norm1(x) -> [B*T, N, D]
```

Then `modulate(...)` applies sample-wise conditioning using `shift_msa` and `scale_msa`.

The modulation keeps the same tensor shape:

```text
[B*T, N, D] -> [B*T, N, D]
```

Then spatial attention is applied:

```text
[B*T, N, D] -> [B*T, N, D]
```

Here the attention sequence length is `N`, so attention is computed across:

- all spatial patches within one frame

After that, the result is reshaped to `[B, T*N, D]`, multiplied by `gate_msa`, reshaped back to `[B*T, N, D]`, and added to the residual.

Output after spatial branch:

```text
[B*T, N, D]
```

Interpretation:

- temporal branch models change across time
- spatial branch models structure within each frame

### 9.4 MLP with adaLN

After spatial attention, the block applies the MLP branch.

Input:

```text
[B*T, N, D]
```

The flow is:

```text
norm2
-> modulate with shift_mlp / scale_mlp
-> MLP
-> gate with gate_mlp
-> residual add
```

All operations preserve shape:

```text
[B*T, N, D] -> [B*T, N, D]
```

This is the standard transformer feed-forward stage, but conditioned with the same DiT-style modulation mechanism.

## 10. `SDPA_Attention` internal shape

Both temporal attention and spatial attention use the same attention module, `SDPA_Attention`, defined in [models/model_dit.py](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/models/model_dit.py#L39).

If the input is:

```text
[X, L, D]
```

then internally:

1. project to Q, K, V:

```text
[X, L, D] -> [X, L, 3, num_heads, head_dim]
```

2. rearrange to:

```text
[3, X, num_heads, L, head_dim]
```

3. split into:

```text
q, k, v: [X, num_heads, L, head_dim]
```

4. apply scaled dot-product attention

5. project back to:

```text
[X, L, D]
```

The difference between temporal and spatial attention is only the meaning of `L`:

- temporal attention: `L = T`
- spatial attention: `L = N`

## 11. Full block summary

The full `MVIFBlock` shape flow can be summarized as:

```text
input:
x: [B*T, N, D]
c: [B, D]

1. c -> adaLN parameters
[B, D] -> 6 x [B, D]

2. temporal branch
[B*T, N, D]
-> [(B*N), T, D]
-> temporal attention
-> [B*T, N, D]
-> residual add

3. spatial branch
[B*T, N, D]
-> norm + modulation
-> spatial attention over N patches
-> gate
-> residual add

4. mlp branch
[B*T, N, D]
-> norm + modulation
-> mlp
-> gate
-> residual add

output:
[B*T, N, D]
```

## 12. What happens after stacked blocks

After all `MVIFBlock`s are applied:

1. `final_layer` maps token features to patch outputs
2. `unpatchify` reconstructs 2D latent maps
3. the tensor is reshaped back to a video latent sequence

This path is implemented in [models/model_dit.py](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/models/model_dit.py#L430) and [models/model_dit.py](/Users/liskie/Projects/PycharmProjects/TSSC-CTA-CL/models/model_dit.py#L371).

So the outer flow is:

```text
[B*T, N, D]
-> final layer
-> [B*T, N, P*P*out_channels]
-> unpatchify
-> [B*T, out_channels, H, W]
-> reshape
-> [B, T, out_channels, H, W]
```

## 13. One-sentence interpretation

`MVIFBlock` is a video diffusion transformer block that separates temporal mixing and spatial mixing:

- temporal attention models how the same spatial location evolves across frames
- spatial attention models how patches interact within one frame
- adaLN-Zero injects diffusion conditioning into the block
```
