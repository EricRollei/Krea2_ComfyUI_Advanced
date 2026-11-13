# Checkpoints, quantization & DeGrid

[<- Back to the README](../README.md) | [All docs](../README.md#documentation)

## Native quantized compute

**Component Loader → `compute`** (`bf16` default | `native`) and **`native_format`**
(`keep` | `int8` | `mxfp8` | `nvfp4`), both appended.

`bf16` dequantizes every checkpoint to bf16 (best image quality). `native` re-quantizes the
28 joint blocks' 224 Linear layers after loading and runs them on ComfyUI's `comfy_kitchen`
tensor-core kernels - the same path ComfyUI core uses; everything else stays bf16.

- `keep` uses the checkpoint's own format (INT8 → int8-ConvRot when the file is rotated,
  MXFP8, NVFP4, fp8); a bf16 / GGUF source falls back to mxfp8. Explicit `int8` / `mxfp8` /
  `nvfp4` quantize any source on load. NVFP4 uses a dynamic per-call activation scale.
- Requires `precision` bf16 and a CUDA device.
- **LoRAs work:** PEFT LoRAs run their delta in bf16 on top of the quantized layer (effect
  identical to bf16 in tests); direct-merge LoRAs (LoKr / LoHa / diff) become exact bf16
  side terms instead of being written into the quantized weights.
- **Image quality:** activations are quantized too, so results differ slightly from bf16 -
  at the same seed, typically 26-30 dB PSNR (looks the same side by side); fp8 can shift
  composition a little. Your call per workflow.

Measured, RTX PRO 5000, Ultra S1-only 3 MP, 8 steps:

| checkpoint (`keep` →) | bf16 s/step / VRAM | native s/step / VRAM |
|---|---|---|
| INT8-ConvRot (int8_convrot) | 2.9 / 34.8 GB | **2.0 / 22.9 GB** |
| same + a style LoRA | 3.1 / 35.3 GB | **2.2 / 22.9 GB** |
| MXFP8 (mxfp8) | 2.9 / 35.0 GB | **2.1 / 23.3 GB** |
| fp8-scaled (fp8) | 2.9 / 35.1 GB | **2.0 / 22.9 GB** |

Headless bench (same card): nvfp4 1.75 s/step (1.67x). The gain shrinks at 8 MP, where
attention (untouched by quantization) dominates the step.

## Quantized checkpoints the Component Loader reads

Single-file transformers are dequantized on load (on the GPU): **fp8** (bare / scaled),
**INT8** and **INT8-ConvRot** (group-wise Hadamard un-rotation), and - through
`comfy_kitchen`'s own layouts, mirroring `comfy/ops.py` - **MXFP8**, **NVFP4**,
**ConvRot-W4A4**, **W4A8** and **W6A8**. GGUF loads as before (dequantize on load).
`docs/tools/krea2_checkpoint_inspector.py` reports any file's format and whether it loads,
without loading it. Note: a file's *name* is not proof of its format - one "INT8-ConvRot"
download in testing was actually bf16 (the loader logs what it found).

## DeGrid (VAE 2-pixel lattice)

The Qwen-Image VAE leaves a faint fixed-phase 2-pixel grid in decoded images (strongest in
flat / dark areas), and the spacepxl 2x upscale VAE builds its output with a 2x2
`pixel_shuffle`. When an inter-stage VAE hop decodes, upsamples and re-encodes, that lattice
is fed into the next stage, which can develop it into a weave-like texture. DeGrid is an
exact 2-pixel notch filter (filter core from lunaaispace-eng/ComfyUI-DeGrid), amplitude-
limited so real edges pass, and an image with no detected lattice is left untouched.

- **Eric Krea2 DeGrid** - standalone node for any image: `mode` auto / manual (`limit`),
  `skip_when_clean`, and a `removed_grid` preview (x `grid_gain`) + report. Put it straight
  after a decode, before sharpening / resizing.
- **`degrid`** toggle on **Upscale Decode (2x)** and **VAE Decode** (appended).
- **Ultra `degrid`** (appended): `hops` cleans the inter-stage VAE upscale before re-encoding
  (no effect on plain latent hops - they have no pixel step), `final` cleans the decoded
  output, `hops+final` both. The console prints the measured grid amplitude per pass.
