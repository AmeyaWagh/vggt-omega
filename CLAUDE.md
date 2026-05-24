# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
pip install -e .

# For Gradio demo
pip install -r requirements_demo.txt
```

## Running the demo

```bash
python demo_gradio.py \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --image-resolution 512
```

For the text-aligned checkpoint, add `--enable-alignment` and use `--image-resolution 256`.

## Linting

```bash
.venv/bin/ruff check .
.venv/bin/ruff format .
```

There is no test suite.

## Architecture

**Entry point:** `vggt_omega/models/vggt_omega.py` — `VGGTOmega(nn.Module)`

The model takes a batch of images `(B, N, 3, H, W)` and runs them through three components:

### 1. Aggregator (`vggt_omega/models/aggregator.py`)

Alternating-attention encoder with 24 transformer blocks. Each block runs:
- **Frame-level** self-attention (`frame_blocks`): processes each frame independently with RoPE position encoding.
- **Inter-frame** attention (`inter_frame_blocks`): either "global" (all tokens across all frames) or "register" (only camera+register tokens, skipping patch tokens). The mode is determined by `register_attention_block_indices = [2, 6, 9, 14, 20]`.

**Token layout** per frame: `[camera_token (1), register_tokens (16), patch_tokens (H/16 × W/16)]`. `patch_token_start = 17`.

The aggregator caches layer outputs at indices `[4, 11, 17, 23]`. Each cached tensor concatenates frame-local and inter-frame token states along the last dimension, yielding `embed_dim * 2 = 2048` features. Uncached layers are `None` in `aggregated_tokens_list`.

The patch embedding backbone is a `DinoVisionTransformer` (`vggt_omega/models/layers/vision_transformer.py`) used as a single-pass feature extractor.

### 2. Prediction heads

- **`CameraHead`** (`models/heads/camera_head.py`): takes `aggregated_tokens_list[-1]`, applies 4 transformer blocks over camera+register tokens across all frames, then predicts a **9D pose encoding** per frame: `[tx, ty, tz, qw, qx, qy, qz, fov_h, fov_w]`. Extrinsics are camera-from-world in OpenCV convention.
- **`DenseHead`** (`models/heads/dense_head.py`): DPT-style multi-scale decoder. Reads from the 4 cached layers, projects to feature maps at 4 scales, fuses them with `FeatureFusionBlock`s (`refinenet1–4`), then predicts **depth** (`exp(logits)`) and **depth confidence** (`1 + exp(logits)`) via pixel-shuffle upsampling. Processes in chunks of 8 frames to manage memory.
- **`TextAlignmentHead`** (`models/heads/text_alignment_head.py`): optional, enabled with `VGGTOmega(enable_alignment=True)`.

### 3. Model outputs (dict)

| Key | Shape | Description |
|---|---|---|
| `pose_enc` | `(B, N, 9)` | Raw pose encoding |
| `depth` | `(B, N, H, W, 1)` | Depth in camera space |
| `depth_conf` | `(B, N, H, W)` | Confidence (≥ 1.0) |
| `camera_and_register_tokens` | `(B, N, 17, 2048)` | Final encoder tokens |
| `images` | `(B, N, 3, H, W)` | Input images (eval mode only) |

### Utilities

- **`vggt_omega/utils/load_fn.py`** — `load_and_preprocess_images()`: resizes images to fit `image_resolution` with either `"balanced"` (preserve total token count) or `"max_size"` (longest-side resize) modes. Pads mixed-size inputs.
- **`vggt_omega/utils/pose_enc.py`** — `encoding_to_camera()` decodes `pose_enc` to `(extrinsics [3×4], intrinsics [3×3])`; `extri_intri_to_pose_encoding()` is the inverse.
- **`vggt_omega/utils/geometry.py`** — `closed_form_inverse_se3()` for batch SE(3) inversion.
- **`visual_util.py`** — `predictions_to_glb()` converts model output to a trimesh GLB scene with point cloud and camera frustums. Requires `world_points_from_depth` and `extrinsic` keys added by the demo pipeline.

### Checkpoints

| Checkpoint | Resolution | Text alignment |
|---|---|---|
| `vggt_omega_1b_512.pt` | 512 | No |
| `vggt_omega_1b_256_text.pt` | 256 | Yes (`enable_alignment=True`) |

The RoPE position encoding must use `normalize_coords="max"` to match the released checkpoints — the model warns if this is misconfigured.

### Autocast behavior

The aggregator runs under bfloat16/float16 autocast. The prediction heads disable autocast and operate in float32. The `_warn_if_rope_not_max` check in `VGGTOmega.__init__` guards against mismatched RoPE config.
