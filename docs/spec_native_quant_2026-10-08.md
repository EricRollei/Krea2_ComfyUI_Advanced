# Spec - Native quantized compute toggle (2026-10-08)

Status: DRAFT for Eric's review. Nothing built.

## Why
The component loader always dequantizes quantized checkpoints to bf16 (best image quality,
but the speed of the quantized format is lost). Benchmarked earlier (per sampling step,
docs/tools/bench_quant_matmul.py):

| mode | 3.3 MP (S1) | 8.5 MP (S3) |
|---|---|---|
| bf16 | x1.00 | x1.00 |
| fp8 per-tensor | x1.10 | x1.06 |
| mxfp8 | **x1.32** | x1.16 |
| nvfp4 | **x1.67** | x1.32 |
| int8 | not measured yet | - |

The gain is largest for S1-only runs at ~3 MP, which Eric runs a lot. At 8 MP, attention is
~60% of a step, and quantizing the linears doesn't touch attention.

## What ComfyUI core already gives us (read at source)
- `comfy/quant_ops.py`: `QuantizedTensor` and registered layouts, with comfy_kitchen kernels behind them:
  - `TensorCoreMXFP8Layout`, `TensorCoreNVFP4Layout`, `TensorCoreFP8E4M3Layout`
  - `TensorWiseINT8Layout`, `TensorCoreConvRotW4A4Layout`, `AsymW4A8Int8Layout`
- `QUANT_ALGOS` tells you whether the input gets quantized:
  - mxfp8 / nvfp4 / fp8 quantize the activation.
  - int8_tensorwise / convrot_w4a4 / w4a8 do not (the layout handles it inside the kernel).
- `comfy/ops.py` mixed-precision Linear does exactly two things:
  1. `x = QuantizedTensor.from_float(x2d, layout, scale=input_scale)` (when the algo quantizes input)
  2. `F.linear(x, qweight, bias)`, which `__torch_dispatch__` routes to the kitchen kernel.

So no new kernels are needed. A thin `nn.Linear` subclass doing those two steps, with the
weight held as a `QuantizedTensor`, gives Krea2's diffusers model the same compute path
ComfyUI uses.

## Proposed design
1. **Loader toggle (append at END):** `compute: bf16 | native` on the Component Loader. Default bf16.
2. **native** keeps the checkpoint's own quant format for the 28 joint blocks' linears
   (to_q/k/v, to_gate, to_out, the SwiGLU linears, 8 per block = 224 modules). It builds the
   `QuantizedTensor` with the same layout Params the loader already constructs for
   dequantization (MXFP8 / NVFP4 / INT8 / INT8-ConvRot / W4A4 / W4A8 / fp8-scaled).
   Everything else stays bf16: img_in, txt_in, the text-fusion blocks, time embeddings, final layer.
3. **bf16 checkpoints + native:** optional quantize-on-load to `mxfp8` (`ck.quantize_mxfp8`).
   A second widget, `native_format: keep | mxfp8 | nvfp4` (append), applies to bf16 sources only.
   NVFP4-on-load would need calibration of the activation global scale, so mxfp8 first.
4. **GGUF:** unchanged (dequantized). A separate later item could move the GGUF dequant to the GPU.
5. **Fallbacks:** if a layout or kernel isn't available on the device, or the input isn't
   2D-able, that module falls back to dequantize-on-the-fly for that call. Logged once.

## LoRA interaction (the hard part)
- **PEFT LoRA** wraps the base `nn.Linear` and calls `base_layer(x)` plus the low-rank delta.
  With our subclass as the base, the base runs quantized and the LoRA delta runs in bf16.
  That works, and matches how ComfyUI applies LoRAs to quantized models (the weight stays
  quantized; the patch is applied separately). Must verify: PEFT reading `base_layer.weight`
  dtype/shape on a `QuantizedTensor` (it needs `in_features` / `out_features`, plus a dtype for lora_A/B).
- **Direct-merge fallback** (`_apply_krea_direct_deltas`, used when PEFT can't wrap)
  writes into `weight.data`, which would corrupt a quantized weight. In native mode, route
  those deltas to an unmerged forward hook (`out += x @ A^T @ B^T * s`), the same
  mechanism `_control.py` already uses.
- **Control LoRA and paint/style runtimes:** hooks and processors only, unaffected.

## Image-quality caveat
Native mode also quantizes **activations** (mxfp8/nvfp4) or uses int8 activation paths.
That is extra loss on top of the weight quantization the checkpoint already has. Measure
PSNR/LPIPS vs bf16-dequant on a fixed seed set before recommending any mode as default.

## Test plan
1. **INT8 bench first** (the most common Krea2 fine-tune format in the library):
   - add `int8_tensorwise` (+ConvRot) to `bench_quant_matmul.py` via the ComfyUI
     `QuantizedTensor` path, not raw kernels, so the number matches what we'd ship.
   - Run on the 5000, which isn't throttled, for trustworthy absolute times.
2. **Parity** per format: same seed, native vs dequant, PSNR + visual at 1 MP and 3 MP S1.
3. **LoRA:** a style LoRA via PEFT on a native MXFP8 model vs dequant (visual + delta norm),
   plus one LoRA that takes the direct-merge path.
4. **Speed** in Ultra: S1-only 3 MP and a full 3-stage run, native vs bf16.

## Decisions for Eric
1. Toggle on the Component Loader only (not the diffusers-folder Loader)? (recommended)
2. Quantize-on-load for bf16 sources: mxfp8 only in v1? (recommended)
3. OK to bench INT8 + measure IQ before building the toggle? (recommended: yes; if INT8 or
   MXFP8 IQ loss is visible, the toggle still ships but defaults to bf16)
