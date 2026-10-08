# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.

"""
Attention-backend selection for the Krea 2 transformer, mirroring the helper in
Eric_Qwen_Edit. Detects capability by introspection and never raises: a bad or
unavailable choice falls back to the diffusers default (SDPA) instead of
breaking generation.

2026-10-08 - mask-safe attention (flash_varlen prefix-mask bug):
diffusers' ``_flash_varlen_attention`` keeps ``key[b, :valid_len]`` - it assumes the
valid keys are a PREFIX of the sequence. Krea 2 always pads the text to 512 tokens
and the joint sequence is ``[text (padded) | image]``, so the padding sits in the
MIDDLE: that backend attended to the ~(512 - prompt) garbage padded text keys and
DROPPED the same number of keys off the END = the bottom image rows (the bottom
"boundary band" crop_bottom was hiding; also weaker prompt adherence). Verified by a
same-seed 3 MP A/B against SDPA.
Fix: every Krea2 attention module gets ``Krea2MaskSafeProcessor`` - the stock
processor's math, but masked-out keys are removed explicitly (index_select) and the
kernel is called WITHOUT a mask, which is exact for any backend and keeps flash speed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Friendly node choice -> ordered diffusers backend names to try. Varlen first
# because Krea 2 passes a padding mask; dense flash/sage can reject it at runtime.
# (Varlen is safe since 2026-10-08 only because of Krea2MaskSafeProcessor below.)
_ATTENTION_BACKEND_CANDIDATES = {
    "auto":   ["flash_varlen", "flash", "native"],
    "flash":  ["flash_varlen", "flash"],
    "sage":   ["sage_varlen", "sage"],
    # 2026-10-08: dense SageAttention 2 int8-QK / fp8-PV CUDA kernel (1.6-1.8x flash-attn 2
    # on sm120, ~4% per-call error). Dense is fine: padded keys are compacted out first.
    "sage_fp8": ["_sage_qk_int8_pv_fp8_cuda", "sage", "flash_varlen", "flash"],
    "sdpa":   ["native"],
    "native": ["native"],
}


# -- mask-safe key compaction ------------------------------------------------------------

def compact_kv(k, v, mask):
    """k/v: [B, Lk, H, D]. mask: None or bool broadcastable to (B, 1, 1, Lk) / (B, Lk).
    Returns (k, v, mask_or_None, needs_dense_mask). When every batch row shares the same
    key mask the masked keys are dropped and mask=None is returned (exact, any backend).
    If rows differ, inputs come back unchanged with needs_dense_mask=True."""
    if mask is None:
        return k, v, None, False
    m = mask.bool()
    m2 = m.reshape(m.shape[0], m.shape[-1]) if m.dim() >= 2 else m[None]
    if m2.shape[-1] != k.shape[1]:
        return k, v, mask, True
    if bool(m2.all()):
        return k, v, None, False
    if m2.shape[0] > 1 and not bool((m2 == m2[:1]).all()):
        return k, v, mask, True
    idx = m2[0].nonzero(as_tuple=False).squeeze(1)
    return k.index_select(1, idx), v.index_select(1, idx), None, False


def _sdpa_dense(q, k, v, mask, gqa):
    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
    if gqa and kt.shape[1] != qt.shape[1]:
        rep = qt.shape[1] // kt.shape[1]
        kt, vt = kt.repeat_interleave(rep, 1), vt.repeat_interleave(rep, 1)
    m = mask.bool() if mask is not None and mask.dtype != torch.bool else mask
    return F.scaled_dot_product_attention(qt, kt, vt, attn_mask=m).transpose(1, 2)


class Krea2MaskSafeProcessor:
    """Drop-in for diffusers' Krea2AttnProcessor (same projections / norms / RoPE / gate)
    with padded keys compacted out before the attention kernel."""

    _attention_backend = None
    _parallel_config = None
    _eric_mask_safe = True

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.embeddings import apply_rotary_emb
        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        gate = attn.to_gate(hidden_states)
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        gqa = attn.num_heads != attn.num_kv_heads
        key, value, mask, dense = compact_kv(key, value, attention_mask)
        if dense:
            hidden_states = _sdpa_dense(query, key, value, mask, gqa)
        else:
            hidden_states = dispatch_attention_fn(
                query, key, value, attn_mask=None, enable_gqa=gqa,
                backend=self._attention_backend, parallel_config=self._parallel_config)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states * torch.sigmoid(gate)
        return attn.to_out[0](hidden_states)


def install_mask_safe_attention(transformer, log=print) -> int:
    """Replace every stock Krea2AttnProcessor in the transformer (joint blocks and the
    text-fusion blocks) with Krea2MaskSafeProcessor, carrying over the backend settings.
    Idempotent; leaves foreign processors (ours or other packs') untouched. Returns the
    number of processors swapped."""
    n = 0
    for mod in transformer.modules():
        proc = getattr(mod, "processor", None)
        if proc is None or getattr(proc, "_eric_mask_safe", False):
            continue
        if type(proc).__name__ != "Krea2AttnProcessor":
            continue
        new = Krea2MaskSafeProcessor()
        new._attention_backend = getattr(proc, "_attention_backend", None)
        new._parallel_config = getattr(proc, "_parallel_config", None)
        mod.processor = new
        n += 1
    if n and log is not None:
        log(f"[attn] mask-safe Krea2 attention installed on {n} modules "
            "(padded keys compacted - fixes the flash_varlen bottom-band bug)")
    return n


def _try_set_backend(transformer, cand, log=print) -> bool:
    setter = getattr(transformer, "set_attention_backend", None)
    if setter is None:
        return False
    try:
        setter(cand)
        return True
    except Exception as e:
        if log is not None:
            log(f"[attn] backend '{cand}' unavailable ({type(e).__name__}); trying next")
        return False


def mask_safe_disabled() -> bool:
    """DIAGNOSTIC (2026-10-08): ERIC_KREA2_MASK_SAFE=0 in the environment before ComfyUI starts
    turns the mask-safe fix OFF (old flash_varlen behaviour) for A/B comparisons."""
    import os
    return os.environ.get("ERIC_KREA2_MASK_SAFE", "").strip().lower() in ("0", "off", "false", "no")


def uninstall_mask_safe_attention(transformer, log=print) -> int:
    """Put the stock diffusers Krea2AttnProcessor back on every module that carries ours."""
    from diffusers.models.transformers.transformer_krea2 import Krea2AttnProcessor
    n = 0
    for mod in transformer.modules():
        proc = getattr(mod, "processor", None)
        if proc is None or not getattr(proc, "_eric_mask_safe", False):
            continue
        new = Krea2AttnProcessor()
        new._attention_backend = getattr(proc, "_attention_backend", None)
        new._parallel_config = getattr(proc, "_parallel_config", None)
        mod.processor = new
        n += 1
    return n


def _finish(transformer, log):
    """Install the mask-safe processor after the backend is chosen. It copies the backend
    from the stock processor it replaces; later set_attention_backend() calls reach it too
    (diffusers sets _attention_backend on every processor that has the attribute)."""
    if mask_safe_disabled():
        try:
            n = uninstall_mask_safe_attention(transformer, log=log)
        except Exception as e:
            n = f"? ({type(e).__name__}: {e})"
        if log is not None:
            log("[attn] *** DIAGNOSTIC: ERIC_KREA2_MASK_SAFE=0 - mask-safe attention is OFF "
                f"(stock processor, restored on {n} module(s)). flash_varlen attends to text "
                "padding and drops the bottom image rows - use crop_bottom. A/B only. ***")
        return
    try:
        install_mask_safe_attention(transformer, log=log)
    except Exception as e:  # never break loading over the fix
        if log is not None:
            log(f"[attn] mask-safe attention not installed ({type(e).__name__}: {e})")


def apply_attention_backend(transformer, backend="auto", *, log=print):
    """Route the Krea2 transformer onto flash/sage when available; else SDPA.
    Always installs the mask-safe Krea2 processor afterwards (see module docstring).

    Returns the backend name applied, or None if left at the diffusers default.
    """
    name = (backend or "auto").strip().lower()
    if name in ("sdpa", "native", "default", "", "off"):
        _try_set_backend(transformer, "native", log=None)
        _finish(transformer, log)
        return None
    if getattr(transformer, "set_attention_backend", None) is None:
        log("[attn] this diffusers build has no set_attention_backend(); leaving default SDPA")
        _finish(transformer, log)
        return None
    for cand in _ATTENTION_BACKEND_CANDIDATES.get(name, [name]):
        if _try_set_backend(transformer, cand, log=log):
            log(f"[attn] attention backend set to '{cand}'")
            _finish(transformer, log)
            return cand
    log(f"[attn] no '{name}' backend available; using diffusers default SDPA")
    _try_set_backend(transformer, "native", log=None)
    _finish(transformer, log)
    return None
