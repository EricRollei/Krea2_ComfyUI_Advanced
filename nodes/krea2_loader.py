# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.

"""
Eric Krea2 Loader
=================
Loads a Krea 2 diffusers checkpoint (Raw/midtrain or Turbo/distilled) via
Krea2Pipeline, applies the chosen attention backend, optionally keeps it
resident in VRAM, and auto-detects whether the checkpoint is distilled (so
downstream nodes can pick sane step/CFG defaults).

Requires a diffusers build that includes Krea2Pipeline (install from source:
    pip install git+https://github.com/huggingface/diffusers.git
). The loader checks for it and gives a clear message if it's missing.
"""

from __future__ import annotations

import json
import os

import torch
import builtins

from .._compat import apply_attention_backend

# Module-level cache so keep_in_vram avoids reloading across runs. Anchored on
# `builtins` so it survives module hot-reloads / re-imports (see the matching note
# in krea2_component_loader.py): otherwise a re-exec of this module resets the cache
# and forces a full reload every queue even with keep_in_vram on.
_PIPELINE_CACHE = getattr(builtins, "_ERIC_KREA2_PIPELINE_CACHE", None)
if not isinstance(_PIPELINE_CACHE, dict):
    _PIPELINE_CACHE = {"pipeline": None, "cache_key": None, "is_distilled": False}
    builtins._ERIC_KREA2_PIPELINE_CACHE = _PIPELINE_CACHE

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _norm_path(p):
    """Canonicalize a path for cache-key comparison so cosmetic differences (trailing
    slash, `/` vs `\\`, drive-letter casing) between two tabs don't force a spurious
    reload / second copy in VRAM. Blank stays ""."""
    p = (p or "").strip()
    if not p:
        return ""
    try:
        return os.path.normcase(os.path.normpath(p))
    except Exception:
        return p


def _read_is_distilled(model_path: str) -> bool:
    """Read is_distilled from model_index.json (Turbo=True, Raw=False)."""
    try:
        with open(os.path.join(model_path, "model_index.json"), "r", encoding="utf-8") as f:
            return bool(json.load(f).get("is_distilled", False))
    except Exception:
        return False


class EricKrea2Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": r"H:\Training\Krea-2-Raw",
                    "tooltip": "Path to a Krea 2 diffusers folder (Raw/midtrain or Turbo/distilled)."
                }),
                "precision": (["bf16", "fp16", "fp32"], {
                    "default": "bf16",
                    "tooltip": "Model precision (bf16 recommended for Blackwell)."
                }),
                "attention_backend": (["auto", "flash", "sage", "sdpa"], {
                    "default": "auto",
                    "tooltip": (
                        "Attention kernel for the transformer.\n"
                        "auto = flash if available else SDPA (lossless, ~3x faster on Blackwell);\n"
                        "flash = FlashAttention varlen (lossless); sage = SageAttention (fastest,\n"
                        "slight quality trade-off); sdpa = PyTorch default. Falls back to SDPA if\n"
                        "a kernel is unavailable."
                    )
                }),
                "device": (["cuda", "cuda:0", "cuda:1", "cpu"], {"default": "cuda"}),
                "keep_in_vram": ("BOOLEAN", {"default": True, "tooltip": "Cache pipeline between runs."}),
            }
        }

    RETURN_TYPES = ("KREA2_PIPELINE",)
    RETURN_NAMES = ("krea2_pipeline",)
    FUNCTION = "load"
    CATEGORY = "Eric/Krea2"

    @classmethod
    def IS_CHANGED(cls, model_path, precision="bf16", attention_backend="auto",
                   device="cuda", keep_in_vram=True):
        # See EricKrea2ComponentLoader.IS_CHANGED - same one-shot-rebuild-after-unload logic.
        if _PIPELINE_CACHE.get("pipeline") is None:
            import time
            return time.time()
        return _PIPELINE_CACHE.get("cache_key")

    def load(self, model_path, precision="bf16", attention_backend="auto",
             device="cuda", keep_in_vram=True):
        try:
            from diffusers import Krea2Pipeline
        except Exception:
            raise RuntimeError(
                "Krea2Pipeline is not available in this diffusers build. Install a "
                "diffusers version that includes Krea 2 (from source):\n"
                "    python_embeded\\python.exe -m pip install --upgrade "
                "git+https://github.com/huggingface/diffusers.git"
            )

        is_distilled = _read_is_distilled(model_path)
        cache_key = f"{_norm_path(model_path)}|{precision}|{device}"

        cache = _PIPELINE_CACHE
        if keep_in_vram and cache["pipeline"] is not None and cache["cache_key"] == cache_key:
            print("[EricKrea2] Using cached pipeline")
            apply_attention_backend(cache["pipeline"].transformer, attention_backend,
                                    log=lambda m: print("[EricKrea2] " + m))
            return ({"pipeline": cache["pipeline"], "is_distilled": cache["is_distilled"],
                     "model_path": model_path},)

        dtype = _DTYPES.get(precision, torch.bfloat16)
        print(f"[EricKrea2] Loading Krea2Pipeline from {model_path} "
              f"({'distilled/Turbo' if is_distilled else 'base/Raw'}, {precision})")
        pipe = Krea2Pipeline.from_pretrained(model_path, torch_dtype=dtype)
        if device != "cpu":
            pipe.to(device)

        apply_attention_backend(pipe.transformer, attention_backend,
                                log=lambda m: print("[EricKrea2] " + m))

        if keep_in_vram:
            cache["pipeline"] = pipe
            cache["cache_key"] = cache_key
            cache["is_distilled"] = is_distilled

        return ({"pipeline": pipe, "is_distilled": is_distilled, "model_path": model_path},)


NODE_CLASS_MAPPINGS = {"EricKrea2Loader": EricKrea2Loader}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2Loader": "Eric Krea2 Loader"}
