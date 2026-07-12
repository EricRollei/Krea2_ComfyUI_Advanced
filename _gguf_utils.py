# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 GGUF / comfy-checkpoint utilities
============================================
Path-A GGUF loading + comfy/original -> diffusers key remapping for the Krea2
transformer.

  * read_gguf_tensors: read a GGUF (pip `gguf`), dequantize any quant type to a
    compute dtype (full precision in VRAM).
  * remap_krea2_state_dict: convert a comfy/original SingleStreamDiT state dict
    (as used by ComfyUI UNETLoader checkpoints AND krea2 GGUFs) to the diffusers
    Krea2Transformer2DModel key layout. Complete map (Linears + norms + biases +
    per-block modulation table), derived from comfy/ldm/krea2/model.py and
    diffusers/models/transformers/transformer_krea2.py.

Modulation note: comfy stores per-block modulation as a flat nn.Parameter
(DoubleSharedModulation.lin, shape [6*dim]); diffusers stores it as
scale_shift_table [6, hidden]. Same math - we reshape. The final layer's
SimpleModulation.lin [2, dim] maps directly to final_layer.scale_shift_table.

Author: Eric Hiss (GitHub: EricRollei)
"""

import numpy as np
import torch


# ── GGUF reading / dequant ───────────────────────────────────────────────────

def read_gguf_tensors(path, compute_dtype=torch.bfloat16, log=print):
    """Read a GGUF -> ({name: dequantized torch.Tensor}, {'arch': str|None}).
    Quantized tensors are dequantized to fp32 then cast to compute_dtype; F16/F32
    pass through. Reshaped to reversed(gguf_shape) so a Linear weight is [out, in]."""
    import gguf
    from gguf.quants import dequantize
    import os

    reader = gguf.GGUFReader(path)
    arch = None
    try:
        fld = reader.get_field("general.architecture")
        if fld is not None:
            arch = str(bytes(fld.parts[fld.data[-1]]), encoding="utf-8")
    except Exception:
        pass

    f16, f32 = gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32
    sd = {}
    n_quant = 0
    for t in reader.tensors:
        shape = tuple(int(d) for d in reversed(t.shape))
        if t.tensor_type in (f16, f32):
            arr = np.array(t.data)
        else:
            arr = dequantize(t.data, t.tensor_type)
            n_quant += 1
        sd[t.name] = torch.from_numpy(np.ascontiguousarray(arr).reshape(shape)).to(compute_dtype)
    log(f"[EricKrea2-GGUF] read {len(sd)} tensors from {os.path.basename(path)} "
        f"({n_quant} dequantized, arch={arch})")
    return sd, {"arch": arch}


# ── comfy/original -> diffusers key map ──────────────────────────────────────

# per-(text-fusion or main) block: comfy leaf -> diffusers leaf (Linears + norms)
_BLOCK_MAP = {
    "attn.wq.weight": "attn.to_q.weight",
    "attn.wk.weight": "attn.to_k.weight",
    "attn.wv.weight": "attn.to_v.weight",
    "attn.wo.weight": "attn.to_out.0.weight",
    "attn.gate.weight": "attn.to_gate.weight",
    "attn.qknorm.qnorm.scale": "attn.norm_q.weight",
    "attn.qknorm.knorm.scale": "attn.norm_k.weight",
    "mlp.gate.weight": "ff.gate.weight",
    "mlp.up.weight": "ff.up.weight",
    "mlp.down.weight": "ff.down.weight",
    "prenorm.scale": "norm1.weight",
    "postnorm.scale": "norm2.weight",
}

# top-level: comfy -> diffusers
_TOP_MAP = {
    "first.weight": "img_in.weight", "first.bias": "img_in.bias",
    "tmlp.0.weight": "time_embed.linear_1.weight", "tmlp.0.bias": "time_embed.linear_1.bias",
    "tmlp.2.weight": "time_embed.linear_2.weight", "tmlp.2.bias": "time_embed.linear_2.bias",
    "tproj.1.weight": "time_mod_proj.weight", "tproj.1.bias": "time_mod_proj.bias",
    "txtmlp.0.scale": "txt_in.norm.weight",
    "txtmlp.1.weight": "txt_in.linear_1.weight", "txtmlp.1.bias": "txt_in.linear_1.bias",
    "txtmlp.3.weight": "txt_in.linear_2.weight", "txtmlp.3.bias": "txt_in.linear_2.bias",
    "txtfusion.projector.weight": "text_fusion.projector.weight",
    "last.linear.weight": "final_layer.linear.weight", "last.linear.bias": "final_layer.linear.bias",
    "last.norm.scale": "final_layer.norm.weight",
    "last.modulation.lin": "final_layer.scale_shift_table",
}


def _build_krea2_map(n_layers, n_layerwise, n_refiner):
    m = dict(_TOP_MAP)
    for i in range(n_layers):
        for c, d in _BLOCK_MAP.items():
            m[f"blocks.{i}.{c}"] = f"transformer_blocks.{i}.{d}"
        m[f"blocks.{i}.mod.lin"] = f"transformer_blocks.{i}.scale_shift_table"  # reshape [6*dim]->[6,dim]
    for kind in ("layerwise_blocks", "refiner_blocks"):
        cnt = n_layerwise if kind == "layerwise_blocks" else n_refiner
        for i in range(cnt):
            for c, d in _BLOCK_MAP.items():
                m[f"txtfusion.{kind}.{i}.{c}"] = f"text_fusion.{kind}.{i}.{d}"
    return m


def _strip_prefix(name):
    for pre in ("model.diffusion_model.", "diffusion_model.", "model."):
        if name.startswith(pre):
            return name[len(pre):]
    return name


def _count_blocks(clean_keys, prefix):
    idxs = set()
    for k in clean_keys:
        if k.startswith(prefix):
            idxs.add(k[len(prefix):].split(".")[0])
    return len(idxs)


def remap_krea2_state_dict(state_dict, n_layers, log=print):
    """comfy/original Krea2 state dict -> diffusers Krea2Transformer2DModel keys.
    Safe on already-diffusers-named dicts (unknown keys pass through unchanged)."""
    clean = {_strip_prefix(n): t for n, t in state_dict.items()}
    keys = clean.keys()
    if not n_layers:
        n_layers = _count_blocks(keys, "blocks.")
    n_lw = _count_blocks(keys, "txtfusion.layerwise_blocks.") or 2
    n_rf = _count_blocks(keys, "txtfusion.refiner_blocks.") or 2
    kmap = _build_krea2_map(n_layers, n_lw, n_rf)
    log(f"[EricKrea2-GGUF] remap map built: {n_layers} blocks, "
        f"{n_lw} layerwise + {n_rf} refiner text-fusion blocks")

    out = {}
    mapped = 0
    passthrough = []
    for c, t in clean.items():
        d = kmap.get(c)
        if d is None:
            out[c] = t
            passthrough.append(c)
            continue
        if c.endswith(".mod.lin") and c.startswith("blocks."):
            t = t.reshape(6, -1)                     # [6*dim] -> [6, hidden]
        out[d] = t
        mapped += 1

    log(f"[EricKrea2-GGUF] remapped {mapped} keys; {len(passthrough)} passthrough")
    if passthrough:
        log("[EricKrea2-GGUF]   passthrough sample: " + ", ".join(passthrough[:8]))
    return out


def load_krea2_transformer_from_gguf(base_pipeline_path, gguf_path, dtype, log=print):
    """Build a Krea2Transformer2DModel from the base config, then load dequantized
    GGUF weights (Path A: full precision in VRAM)."""
    from diffusers import Krea2Transformer2DModel

    cfg = Krea2Transformer2DModel.load_config(base_pipeline_path, subfolder="transformer")
    # Meta-device init: skip the random weight-initialization pass (previously
    # ~50s of CPU time for weights load_state_dict immediately overwrites).
    # include_buffers=False keeps computed buffers (rotary tables etc.) real.
    try:
        from accelerate import init_empty_weights
        with init_empty_weights(include_buffers=False):
            model = Krea2Transformer2DModel.from_config(cfg)
        _on_meta = True
    except Exception as _e:
        log(f"[EricKrea2-GGUF] accelerate unavailable ({_e}); slow full init")
        model = Krea2Transformer2DModel.from_config(cfg).to(dtype)
        _on_meta = False
    n_layers = getattr(model.config, "num_layers", None)
    if not n_layers:
        n_layers = sum(1 for n, _ in model.named_modules()
                       if n.startswith("transformer_blocks.") and n.count(".") == 1)
    log(f"[EricKrea2-GGUF] base transformer config: {n_layers} layers")

    gguf_sd, meta = read_gguf_tensors(gguf_path, compute_dtype=dtype, log=log)
    if meta.get("arch") and meta["arch"] not in ("krea2", None):
        log(f"[EricKrea2-GGUF] WARNING: GGUF arch '{meta['arch']}' != 'krea2'. Proceeding anyway.")
    diff_sd = {k: v.to(dtype) for k, v in remap_krea2_state_dict(gguf_sd, n_layers, log=log).items()}
    del gguf_sd

    # assign=True is required for meta params (copy_ into meta is impossible) and
    # also skips a full tensor copy. A key missing under meta init is left as a
    # meta tensor (crash/garbage at inference), so verify and fail loudly.
    missing, unexpected = model.load_state_dict(diff_sd, strict=False, assign=_on_meta)
    if _on_meta:
        _metas = [pn for pn, p in model.named_parameters() if p.is_meta]
        if _metas:
            raise RuntimeError(
                f"GGUF transformer: {len(_metas)} parameter(s) missing from the checkpoint "
                f"were left uninitialized (meta). Sample: " + ", ".join(_metas[:8]))
        model = model.to(dtype)
    log(f"[EricKrea2-GGUF] loaded {len(diff_sd) - len(unexpected)} tensors | "
        f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        log("[EricKrea2-GGUF]   MISSING: " + ", ".join(list(missing)[:10]) + (" ..." if len(missing) > 10 else ""))
    if unexpected:
        log("[EricKrea2-GGUF]   UNEXPECTED: " + ", ".join(list(unexpected)[:10]) + (" ..." if len(unexpected) > 10 else ""))
    return model
