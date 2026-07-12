# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 LoRA nodes
=====================
DECLARATIVE + EPHEMERAL LoRA model for the Krea2 pipeline.

Design (differs from the eager Qwen Apply-LoRA on purpose):
  * Apply-LoRA nodes DO NOT touch the pipe. They only append a request
    {lora, per-stage weights, ephemeral} onto pipeline["lora_stack"] and pass a
    copied pipeline dict downstream. Chain several to stack LoRAs.
  * The Multi-Stage Ultra node REALIZES the stack at generation time: unload-all
    (clean slate) -> load the declared stack fresh with the robust key-fix ->
    set per-stage weights before S1/S2/S3 -> unload-all in a finally.

Why: the persistent, cached pipe object is what caused stale adapters (changing
the LoRA left the old one active) and orphaned VRAM (cancel / workflow switch).
Realize-at-generation + ephemeral teardown makes adapters exist ONLY during the
generation call, so there is nothing to go stale or leak.

The heavy lifting (standard LoRA / LoKR / LoHa, key normalization, kohya->diffusers
renaming, PEFT injection with a direct-merge fallback, diagnostics) lives in the
vendored _lora_utils (ported from Eric's Qwen Apply-LoRA node).

Author: Eric Hiss (GitHub: EricRollei)
"""

import os
import re
from typing import Tuple

from .. import _settings

from .._lora_utils import (
    get_lora_list, get_lora_full_path,
    _diagnose_lora_compatibility, unload_all_loras,
)


def _sanitize_adapter_name(lora_path: str) -> str:
    # PEFT/PyTorch ModuleDict keys cannot contain '.', so sanitize the stem.
    stem = os.path.splitext(os.path.basename(lora_path))[0]
    return re.sub(r"[.\s]", "_", stem)


def _stack_settings(stack) -> str:
    """Serialize a lora_stack into the KREA2_SETTINGS 'lora' section (recipe fields only)."""
    entries = [{
        "lora_name": e.get("lora_name") or e.get("filename", ""),
        "strength": e.get("strength", 1.0),
        "weight_s1": e.get("weight_s1"),
        "weight_s2": e.get("weight_s2"),
        "weight_s3": e.get("weight_s3"),
        "ephemeral": e.get("ephemeral", True),
    } for e in (stack or [])]
    return _settings.wrap("lora", entries)


# =======================================================================
#  Apply LoRA (declarative, stackable)
# =======================================================================

class EricKrea2ApplyLoRA:
    """Declare a LoRA onto the Krea2 pipeline's stack. Chain several of these to
    stack multiple LoRAs. Nothing is loaded here - the Multi-Stage Ultra node
    loads the whole stack fresh at generation and clears it afterward."""

    CATEGORY = "Eric/Krea2"
    FUNCTION = "apply_lora"
    RETURN_TYPES = ("KREA2_PIPELINE", "STRING")
    RETURN_NAMES = ("pipeline", "settings")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("KREA2_PIPELINE",),
                "lora_name": (get_lora_list("krea2"), {
                    "tooltip": "LoRA from ComfyUI/models/loras/Krea2/. 'none' = pass through unchanged. "
                               "Use lora_path_override for a LoRA outside that folder."}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 50000, "step": 0.05,
                    "tooltip": "LoRA weight applied to ALL stages (unless per_stage_weights is on). "
                               "This is the LoRA scale, unrelated to the Krea2 cfg/g convention."}),
            },
            "optional": {
                "per_stage_weights": ("BOOLEAN", {"default": False,
                    "tooltip": "OFF: use 'strength' for every stage. ON: use the weight_s1/s2/s3 "
                               "fields instead (e.g. detail LoRA 0/0.7/1.0, style LoRA 1.0/0.6/0.3)."}),
                "weight_s1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 50000, "step": 0.05,
                    "tooltip": "Stage-1 (composition) weight. Only used when per_stage_weights is ON."}),
                "weight_s2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 50000, "step": 0.05,
                    "tooltip": "Stage-2 (refine) weight. Only used when per_stage_weights is ON."}),
                "weight_s3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 50000, "step": 0.05,
                    "tooltip": "Stage-3 (final detail) weight. Only used when per_stage_weights is ON."}),
                "ephemeral": ("BOOLEAN", {"default": True,
                    "tooltip": "ON (recommended): load this LoRA fresh each run and unload it after "
                               "generation (no stale/orphaned weights; small reload cost). OFF: leave "
                               "it resident after the run (faster reuse, but you must Unload manually "
                               "to free VRAM / before switching workflows). If ANY LoRA in the stack "
                               "is ephemeral, the whole stack is cleared after the run."}),
                "lora_path_override": ("STRING", {"default": "",
                    "tooltip": "Optional absolute path to a LoRA file (overrides the dropdown)."}),
            },
        }

    def apply_lora(self, pipeline, lora_name, strength=1.0,
                   per_stage_weights=False, weight_s1=1.0, weight_s2=1.0, weight_s3=1.0,
                   ephemeral=True, lora_path_override=""):
        # Resolve path
        if lora_path_override and lora_path_override.strip():
            lora_path = lora_path_override.strip()
            if not os.path.exists(lora_path):
                raise ValueError(f"LoRA file not found: {lora_path}")
        else:
            if lora_name == "none" or not lora_name:
                return (pipeline, _stack_settings(pipeline.get("lora_stack", [])))   # pass through
            lora_path = get_lora_full_path(lora_name)
            if lora_path is None:
                raise ValueError(f"LoRA file not found: {lora_name}")

        if not per_stage_weights:
            weight_s1 = weight_s2 = weight_s3 = strength

        # Copy the pipeline dict + stack list so chaining accumulates immutably and
        # ComfyUI sees a distinct output (the underlying pipe object stays shared).
        new_pipeline = dict(pipeline)
        stack = list(pipeline.get("lora_stack", []))

        adapter_name = _sanitize_adapter_name(lora_path)
        existing = {e["adapter_name"] for e in stack}
        if adapter_name in existing:   # de-dupe within a chain
            n = 2
            while f"{adapter_name}_{n}" in existing:
                n += 1
            adapter_name = f"{adapter_name}_{n}"

        stack.append({
            "path": lora_path,
            "filename": os.path.basename(lora_path),
            "lora_name": lora_name or os.path.basename(lora_path),
            "adapter_name": adapter_name,
            "strength": float(strength),
            "weight_s1": float(weight_s1),
            "weight_s2": float(weight_s2),
            "weight_s3": float(weight_s3),
            "ephemeral": bool(ephemeral),
        })
        new_pipeline["lora_stack"] = stack

        wtxt = (f"all={strength}" if not per_stage_weights
                else f"S1={weight_s1}, S2={weight_s2}, S3={weight_s3}")
        print(f"[EricKrea2-LoRA] queued '{os.path.basename(lora_path)}' as '{adapter_name}' "
              f"({wtxt}, ephemeral={ephemeral}); stack depth {len(stack)}. "
              "Realized at generation by the Multi-Stage Ultra node.")
        return (new_pipeline, _stack_settings(stack))


# =======================================================================
#  Unload LoRA (manual clear / panic button)
# =======================================================================

class EricKrea2UnloadLoRA:
    """Unload all LoRA adapters from the Krea2 pipe NOW and clear the declared
    stack. Use before switching workflows, or to force-free VRAM. (Ephemeral
    LoRAs already self-clear after each run; this is the explicit override.)"""

    CATEGORY = "Eric/Krea2"
    FUNCTION = "unload"
    RETURN_TYPES = ("KREA2_PIPELINE",)
    RETURN_NAMES = ("pipeline",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipeline": ("KREA2_PIPELINE",)}}

    def unload(self, pipeline):
        pipe = pipeline["pipeline"]
        unload_all_loras(pipe)
        new_pipeline = dict(pipeline)
        new_pipeline["lora_stack"] = []
        return (new_pipeline,)


# =======================================================================
#  Diagnose LoRA (inspect compatibility without applying)
# =======================================================================

class EricKrea2DiagnoseLoRA:
    """Inspect a LoRA file against the Krea2 transformer WITHOUT applying it.
    Reports adapter type (LoRA/LoKR/LoHa), rank/alpha, how many modules match
    the architecture, and a COMPATIBLE / PARTIAL / INCOMPATIBLE verdict - use it
    when a LoRA produces noise or fails to load."""

    CATEGORY = "Eric/Krea2"
    FUNCTION = "diagnose"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("KREA2_PIPELINE",),
                "lora_name": (get_lora_list("krea2"), {"tooltip": "LoRA to inspect (NOT applied). Krea2 subfolder only; use lora_path_override otherwise."}),
            },
            "optional": {
                "lora_path_override": ("STRING", {"default": "",
                    "tooltip": "Optional absolute path (overrides the dropdown)."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def diagnose(self, pipeline, lora_name, lora_path_override="") -> Tuple:
        pipe = pipeline["pipeline"]
        if lora_path_override and lora_path_override.strip():
            lora_path = lora_path_override.strip()
        else:
            if lora_name == "none" or not lora_name:
                return ("No LoRA selected.",)
            lora_path = get_lora_full_path(lora_name)
            if lora_path is None:
                return (f"ERROR: LoRA file not found: {lora_name}",)
        if not os.path.exists(lora_path):
            return (f"ERROR: File not found: {lora_path}",)
        report = _diagnose_lora_compatibility(lora_path, pipe.transformer)
        print(report)
        return (report,)


NODE_CLASS_MAPPINGS = {
    "EricKrea2ApplyLoRA": EricKrea2ApplyLoRA,
    "EricKrea2UnloadLoRA": EricKrea2UnloadLoRA,
    "EricKrea2DiagnoseLoRA": EricKrea2DiagnoseLoRA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EricKrea2ApplyLoRA": "Eric Krea2 Apply LoRA",
    "EricKrea2UnloadLoRA": "Eric Krea2 Unload LoRA",
    "EricKrea2DiagnoseLoRA": "Eric Krea2 Diagnose LoRA",
}
