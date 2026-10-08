# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 native quantized compute (Component Loader `compute = native`)
=========================================================================
Runs the 28 joint blocks' Linears (to_q/k/v, to_gate, to_out, SwiGLU - 8 per block)
on comfy_kitchen's quantized kernels instead of bf16, reusing the exact path ComfyUI
core uses (comfy/ops.py mixed-precision Linear):

    x  = QuantizedTensor.from_float(x2d, layout)    # only for formats that quantize input
    y  = F.linear(x, weight_qt, bias)                # __torch_dispatch__ -> kitchen kernel

Weights are (re)quantized AFTER the normal bf16 load, on the GPU:
  * "keep"  -> the checkpoint's own format (INT8 / INT8-ConvRot / MXFP8 / NVFP4 / fp8).
    The dequantized values are exactly representable in that format, so the codes come
    back identical (INT8, MXFP8, fp8) or within float rounding (ConvRot rotation, NVFP4).
  * explicit int8 | mxfp8 | nvfp4 -> quantize-on-load (any source, incl. bf16 / GGUF);
    int8 uses ConvRot (Hadamard-rotated) weights - same speed, far closer to bf16.
Measured 2026-10-08 (RTX PRO 5000, 3 MP, s/step vs bf16 2.93): int8-convrot 2.06 (1.42x),
int8 2.02, mxfp8 2.11, fp8 2.14, nvfp4 1.75 (1.67x); peak VRAM 48.2 -> 36 GB (nvfp4 30.8).
NVFP4 activations use a dynamic per-call tensor scale (amax), so no calibration pass.

Everything else (img_in, txt_in, text-fusion blocks, time embeddings, final layer,
attention) stays bf16. Unsupported layout/kernel -> that call dequantizes (logged once).

LoRA:
  * PEFT LoRA wraps the Linear and calls base_layer(x) - runs quantized, delta in bf16.
  * Direct-merge paths (diff / LoKr / LoHa / low-rank fallback) must NOT add_ into a
    QuantizedTensor (comfy_kitchen falls back to a dequantized temporary -> the delta is
    silently lost). `merge_param_delta` stores the dense delta as a bf16 side term that
    QuantLinear adds to its output (exact, no re-quantization); unload clears it.

Author: Eric Hiss (GitHub: EricRollei)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG = "[EricKrea2-NQ]"

# fmt -> (layout name, quantize the activation?, weight quantize kwargs)
_FMT = {
    "int8":         ("TensorWiseINT8Layout", False, {"is_weight": True, "per_channel": True}),
    "int8_convrot": ("TensorWiseINT8Layout", False, {"is_weight": True, "per_channel": True,
                                                     "convrot": True}),
    "mxfp8":        ("TensorCoreMXFP8Layout", True, {}),
    "nvfp4":        ("TensorCoreNVFP4Layout", True, {}),
    "fp8":          ("TensorCoreFP8Layout", True, {}),
}

# checkpoint quant format (comfy_quant conf "format") -> our fmt for "keep"
_SRC_TO_FMT = {
    "int8_tensorwise": "int8", "int8": "int8",
    "mxfp8": "mxfp8", "nvfp4": "nvfp4",
    "float8_e4m3fn": "fp8", "fp8": "fp8", "fp8_scaled": "fp8",
}

FORMAT_CHOICES = ["keep", "int8", "mxfp8", "nvfp4"]


def _ck():
    import comfy_kitchen.tensor as ckt   # registers the layouts
    return ckt


def _qt_cls():
    return _ck().QuantizedTensor


def is_quantized(t) -> bool:
    try:
        return isinstance(t, _qt_cls())
    except Exception:
        return False


def resolve_format(choice, src_info, log=print):
    """choice: keep|int8|mxfp8|nvfp4 ; src_info: {"format": str|None, "convrot": bool,
    "convrot_groupsize": int} recorded by the loader. Returns (fmt, convrot_gs)."""
    src = (src_info or {}).get("format")
    gs = int((src_info or {}).get("convrot_groupsize", 256) or 256)
    if choice == "int8":
        # quantize-on-load: ConvRot rotation tames activation outliers - same speed as plain
        # INT8, much closer to bf16 (measured 2026-10-08: 33.5 vs 26.6 dB at 1 MP)
        return "int8_convrot", gs
    if choice == "keep":
        fmt = _SRC_TO_FMT.get(str(src or "").lower())
        if fmt == "int8" and (src_info or {}).get("convrot"):
            fmt = "int8_convrot"
        if fmt is None:
            fmt = "mxfp8"
            log(f"{_LOG} native 'keep': source format is {src or 'bf16/GGUF (unquantized)'} - "
                "no native kernel to keep, quantizing to mxfp8 instead (choose int8/mxfp8/nvfp4 "
                "explicitly to pick)")
        return fmt, gs
    return choice, gs


class QuantLinear(nn.Linear):
    """nn.Linear whose weight is a comfy_kitchen QuantizedTensor (see module docstring)."""

    _eric_native = True

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        try:
            xin = _qt_cls().from_float(x2, self._nq_layout) if self._nq_qin else x2
            out = F.linear(xin, self.weight, self.bias)
        except Exception as e:
            if not getattr(QuantLinear, "_warned", False):
                QuantLinear._warned = True
                print(f"{_LOG} native kernel failed ({type(e).__name__}: {str(e)[:160]}); "
                      "this layer runs dequantized (logged once)")
            out = F.linear(x2, self.weight.dequantize().to(x2.dtype),
                           None if self.bias is None else self.bias.to(x2.dtype))
        d = self.__dict__.get("_eric_wdelta")
        if d is not None:
            out = out + F.linear(x2, d.to(dtype=x2.dtype, device=x2.device))
        return out.reshape(*shp[:-1], out.shape[-1])

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, native={self._nq_fmt}")


def quantize_weight(w, fmt, convrot_gs=256):
    layout, _qin, kw = _FMT[fmt]
    kw = dict(kw)
    if fmt == "int8_convrot":
        kw["convrot_groupsize"] = int(convrot_gs)
    return _qt_cls().from_float(w.contiguous(), layout, **kw)


def _linear_targets(transformer):
    for bi, blk in enumerate(transformer.transformer_blocks):
        for name, mod in blk.named_modules():
            if type(mod) is nn.Linear:
                yield f"transformer_blocks.{bi}.{name}", mod


@torch.no_grad()
def convert_transformer(transformer, fmt, convrot_gs=256, log=print):
    """Quantize the joint blocks' Linears in place (weights must be on CUDA). Returns the
    number of converted modules; on any per-module failure that module stays bf16."""
    if fmt not in _FMT:
        raise ValueError(f"unknown native format {fmt}")
    layout, qin, _ = _FMT[fmt]
    try:
        _ck().get_layout_class(layout)
    except Exception as e:
        log(f"{_LOG} layout {layout} unavailable in this comfy_kitchen ({e}); staying bf16")
        return 0
    n, failed, bytes_before, bytes_after = 0, 0, 0, 0
    for path, mod in list(_linear_targets(transformer)):
        w = mod.weight
        if w.device.type != "cuda":
            log(f"{_LOG} {path} is on {w.device} - native quant needs the weights on CUDA")
            return n
        if fmt == "int8_convrot" and w.shape[1] % int(convrot_gs):
            failed += 1
            continue
        try:
            qt = quantize_weight(w.detach().to(torch.bfloat16), fmt, convrot_gs)
        except Exception as e:
            failed += 1
            if failed == 1:
                log(f"{_LOG} quantize failed for {path} ({type(e).__name__}: {str(e)[:120]}); "
                    "leaving such layers bf16")
            continue
        bytes_before += w.numel() * w.element_size()
        bytes_after += qt._qdata.numel() * qt._qdata.element_size()
        mod.__class__ = QuantLinear
        mod._nq_layout, mod._nq_qin, mod._nq_fmt = layout, qin, fmt
        mod.weight = nn.Parameter(qt, requires_grad=False)
        n += 1
    torch.cuda.empty_cache()
    transformer._eric_native_fmt = fmt
    log(f"{_LOG} native compute: {n} block Linears -> {fmt} ({layout}, activation quant "
        f"{'on' if qin else 'in-kernel'}); weights {bytes_before / 1e9:.1f} -> "
        f"{bytes_after / 1e9:.1f} GB" + (f"; {failed} left bf16" if failed else ""))
    return n


# -- LoRA direct-merge support ---------------------------------------------------------

def merge_param_delta(transformer, target_key, param, delta):
    """For quantized weights: add `delta` as a bf16 side term on the owning QuantLinear
    instead of add_ into the QuantizedTensor. Returns True when handled (caller must then
    skip its own backup/add_), False for ordinary parameters."""
    if not is_quantized(param) or not target_key.endswith(".weight"):
        return False
    mod_path = target_key[: -len(".weight")]   # PEFT-wrapped: "...X.base_layer" (ours)
    try:
        mod = transformer.get_submodule(mod_path)
    except Exception:
        return False
    if not getattr(mod, "_eric_native", False):
        return False
    d = delta.to(device=param.device, dtype=torch.bfloat16)
    cur = mod.__dict__.get("_eric_wdelta")
    mod.__dict__["_eric_wdelta"] = d if cur is None else cur + d
    reg = transformer.__dict__.setdefault("_eric_wdelta_modules", set())
    reg.add(mod_path)
    return True


def clear_param_deltas(transformer) -> int:
    reg = transformer.__dict__.get("_eric_wdelta_modules") or set()
    n = 0
    for mod_path in list(reg):
        try:
            mod = transformer.get_submodule(mod_path)
            if mod.__dict__.pop("_eric_wdelta", None) is not None:
                n += 1
        except Exception:
            pass
    transformer.__dict__["_eric_wdelta_modules"] = set()
    return n
