# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.

"""
Attention-backend selection for the Krea 2 transformer, mirroring the helper in
Eric_Qwen_Edit. Detects capability by introspection and never raises: a bad or
unavailable choice falls back to the diffusers default (SDPA) instead of
breaking generation.
"""

from __future__ import annotations

# Friendly node choice -> ordered diffusers backend names to try. Varlen first
# because Krea 2 passes a padding mask; dense flash/sage can reject it at runtime.
_ATTENTION_BACKEND_CANDIDATES = {
    "auto":   ["flash_varlen", "flash", "native"],
    "flash":  ["flash_varlen", "flash"],
    "sage":   ["sage_varlen", "sage"],
    "sdpa":   ["native"],
    "native": ["native"],
}


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


def apply_attention_backend(transformer, backend="auto", *, log=print):
    """Route the Krea2 transformer onto flash/sage when available; else SDPA.

    Returns the backend name applied, or None if left at the diffusers default.
    """
    name = (backend or "auto").strip().lower()
    if name in ("sdpa", "native", "default", "", "off"):
        _try_set_backend(transformer, "native", log=None)
        return None
    if getattr(transformer, "set_attention_backend", None) is None:
        log("[attn] this diffusers build has no set_attention_backend(); leaving default SDPA")
        return None
    for cand in _ATTENTION_BACKEND_CANDIDATES.get(name, [name]):
        if _try_set_backend(transformer, cand, log=log):
            log(f"[attn] attention backend set to '{cand}'")
            return cand
    log(f"[attn] no '{name}' backend available; using diffusers default SDPA")
    _try_set_backend(transformer, "native", log=None)
    return None
