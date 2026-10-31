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
from .._pipe_handles import derive_handle
from typing import Tuple

from .. import _settings

from .._lora_utils import (
    get_lora_list, get_lora_full_path,
    _diagnose_lora_compatibility, unload_all_loras,
)
from .._trigger_words import get_trigger_words, merge_triggers_into_prompt


# 'catalog' trigger mode (2026-10-13): one trigger per LoRA from its LoRA Catalog card, prepended.
# 2026-10-16: four plain names appended (the list is append-only: saved workflows keep their values, and the
# JS shows only the four new ones, turning off / catalog / recipe into none / trigger / saved pick on load):
#   none        no words                                  (= off)
#   trigger     the LoRA's main trigger from its card      (= catalog; may be several words, e.g. 'retro artstyle')
#   full        its whole recipe: official words + trigger + style phrase (no picks)
#   saved pick  your pick for this checkpoint, else a trigger test's pick, else full's trigger + phrase (= recipe)
# prepend / append (every looked-up Civitai trigger word) still work for old workflows.
TRIGGER_MODES = ["off", "prepend", "append", "catalog", "recipe",     # recipe appended 2026-10-15
                 "none", "trigger", "full", "saved pick"]           # 2026-10-16
TRIGGER_LABELS = ["none", "trigger", "full", "saved pick"]
_CANON = {"none": "off", "trigger": "catalog", "saved pick": "recipe", "saved_pick": "recipe"}
TRIGGER_TOOLTIP = ("Words added to the prompt for each LoRA, from the LoRA Catalog:\n"
                   "none = nothing.\n"
                   "trigger = the LoRA's main trigger from its card (your priority trigger first; can be a few "
                   "words, e.g. 'retro artstyle illustration'). Enough on the base model, often too weak on "
                   "fine-tunes.\n"
                   "full = the whole recipe: official trained words + trigger + style phrase (no card recipe: "
                   "all its card triggers). Strongest; uses more of the prompt.\n"
                   "saved pick = the words chosen for the loaded checkpoint - yours (card editor / Sweeps "
                   "page), else a trigger test's - else trigger + style phrase. A pick of 'no words' adds "
                   "nothing on purpose.\n"
                   "With two or more LoRAs, full and saved pick are merged into one tidy list (repeats "
                   "dropped, length shared by weight, clashes reported). Words already in the prompt are not "
                   "added twice; all are put in front of the prompt.")


def canon_trigger_mode(mode) -> str:
    """The internal name of a trigger mode: off | catalog | recipe | full | prepend | append."""
    m = str(mode or "off").strip()
    return _CANON.get(m, m)


def checkpoint_of(pipeline):
    """The loaded checkpoint's name, as the sweep names it (transformer file stem) - for per-checkpoint picks."""
    pd = pipeline if isinstance(pipeline, dict) else {}
    for key in ("transformer_path", "model_name", "model_path"):
        v = pd.get(key)
        if v:
            return os.path.splitext(os.path.basename(str(v)))[0]
    return ""


def resolve_triggers(lora_path, mode, fetch, force, checkpoint=""):
    """(triggers, merge mode, source) for one LoRA. catalog: [the card's first active trigger] (or the first looked-up
    trigger when the LoRA has no card trigger), merged by prepending; recipe (2026-10-15): [the LoRA Catalog's prompt
    words for this checkpoint] - your pick, a trigger test's pick, else the recipe default - prepended, and an empty
    pick adds nothing on purpose; no catalog words at all: as catalog. Other modes: the usual lookup, unchanged."""
    mode = canon_trigger_mode(mode)
    if mode == "full":
        try:
            from .._catalog_lookup import card_full
            words, src = card_full(lora_path)
        except Exception as e:
            print(f"[EricKrea2-LoRA] catalog full words lookup failed (non-fatal): {e}")
            words, src = "", "none"
        if words:
            return [words], "prepend", f"full ({src})"
        looked = get_trigger_words(lora_path, force=force) if fetch else []
        return ([", ".join(looked)] if looked else []), "prepend", "full (civitai words)" if looked else "none"
    if mode == "recipe":
        try:
            from .._catalog_lookup import card_prompt
            words, src = card_prompt(lora_path, checkpoint)
        except Exception as e:
            print(f"[EricKrea2-LoRA] catalog prompt words lookup failed (non-fatal): {e}")
            words, src = "", "none"
        if src != "none":
            return ([words] if words else []), "prepend", f"recipe ({src})"
        mode = "catalog"
    if mode == "catalog":
        try:
            from .._catalog_lookup import card_trigger
            found = card_trigger(lora_path)
        except Exception as e:                       # no catalog: fall back to the lookup
            print(f"[EricKrea2-LoRA] catalog trigger lookup failed (non-fatal): {e}")
            found = ""
        if found:
            return [found], "prepend", "card"
        looked = get_trigger_words(lora_path, force=force) if fetch else []
        return looked[:1], "prepend", "lookup" if looked else "none"
    return (get_trigger_words(lora_path, force=force) if fetch else []), mode, "lookup"


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
    # trigger_words / prompt outputs appended at the END so existing links
    # (by index) keep working in saved workflows.
    RETURN_TYPES = ("KREA2_PIPELINE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("pipeline", "settings", "trigger_words", "prompt")

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
                # New widgets are appended at the END so old workflows' positional
                # widgets_values arrays still line up (missing tail -> defaults).
                "enabled": ("BOOLEAN", {"default": True, "label_on": "on", "label_off": "off",
                    "tooltip": "Quick on/off for this node without clearing its settings. OFF = pass "
                               "the pipeline through untouched (the LoRA is not queued this run)."}),
                "apply_lora_preset": (_settings.list_preset_names("apply_lora"), {"default": "custom",
                    "tooltip": "Load a saved Apply-LoRA recipe (lora / strength / per-stage weights / "
                               "toggles) into the panel for this run. 'custom' = use the panel as-is. "
                               "Use the \u2605 Save Preset button to add one; the list refreshes on "
                               "graph reload."}),
                # Trigger-word widgets appended at the END (positional widgets_values
                # safety, same convention as 'enabled' above). 'prompt' is a socket
                # (forceInput), so it does not occupy a widgets_values slot at all.
                "fetch_triggers": ("BOOLEAN", {"default": True,
                    "tooltip": "Look up this LoRA's trigger words (safetensors metadata / Civitai, "
                               "cached locally in Eric_Krea2/data/lora_triggers.json - only the first "
                               "lookup per file touches the network). Fills the trigger_words output."}),
                "add_triggers": (TRIGGER_MODES, {"default": "none",
                    "tooltip": TRIGGER_TOOLTIP + "\nThe words go into the 'prompt' output (connect your "
                               "prompt to the prompt input)."}),
                "force_refetch": ("BOOLEAN", {"default": False,
                    "tooltip": "Bypass the local trigger cache and re-query Civitai for this LoRA "
                               "(use if triggers were updated on Civitai or a lookup failed). "
                               "Leave OFF normally."}),
                "prompt": ("STRING", {"default": "", "forceInput": True,
                    "tooltip": "Optional prompt passthrough. Connect your prompt here and take the "
                               "'prompt' output onward; with add_triggers on, trigger words are "
                               "merged in. Without a connection the prompt output is just the "
                               "triggers (or empty)."}),
                # Appended LAST 2026-10-29 (the socket above takes no widget value).
                "trigger_join": (["comma", "period"], {"default": "comma", "tooltip": "How the trigger words join the prompt (when add_triggers is on). comma: 'trigA, trigB, your prompt' (the behaviour before 2026-10-29). period: each LoRA's words become their own sentence - 'trigA. trigB. Your prompt' - the way many Civitai prompts start; it may stop a trigger being read as text to draw (being tested with the Sweep Plan trigger_joins axis)."}),
            },
        }

    def apply_lora(self, pipeline, lora_name, strength=1.0,
                   per_stage_weights=False, weight_s1=1.0, weight_s2=1.0, weight_s3=1.0,
                   ephemeral=True, lora_path_override="", enabled=True,
                   apply_lora_preset="custom",
                   fetch_triggers=True, add_triggers="off", force_refetch=False,
                   prompt="", trigger_join="comma"):
        # Headless / API-run fallback: apply a named preset over the panel values
        # (the JS normally writes these into the visible widgets at edit time).
        if apply_lora_preset and apply_lora_preset != "custom":
            _vals = {
                "lora_name": lora_name, "strength": strength,
                "per_stage_weights": per_stage_weights,
                "weight_s1": weight_s1, "weight_s2": weight_s2, "weight_s3": weight_s3,
                "ephemeral": ephemeral, "lora_path_override": lora_path_override,
                "enabled": enabled,
                "fetch_triggers": fetch_triggers, "add_triggers": add_triggers,
                "force_refetch": force_refetch,
            }
            n = _settings.apply_named_preset("apply_lora", apply_lora_preset, _vals,
                                             valid_keys=set(_vals.keys()))
            (lora_name, strength, per_stage_weights, weight_s1, weight_s2, weight_s3,
             ephemeral, lora_path_override, enabled,
             fetch_triggers, add_triggers, force_refetch) = (
                _vals["lora_name"], _vals["strength"], _vals["per_stage_weights"],
                _vals["weight_s1"], _vals["weight_s2"], _vals["weight_s3"],
                _vals["ephemeral"], _vals["lora_path_override"], _vals["enabled"],
                _vals["fetch_triggers"], _vals["add_triggers"], _vals["force_refetch"])
            print(f"[EricKrea2-LoRA] apply_lora_preset '{apply_lora_preset}': applied {n} field(s).")

        add_triggers = canon_trigger_mode(add_triggers)
        if not enabled:
            print("[EricKrea2-LoRA] Apply LoRA is toggled OFF - passing pipeline through.")
            return (pipeline, _stack_settings(pipeline.get("lora_stack", [])),
                    "", prompt or "")

        # Resolve path
        if lora_path_override and lora_path_override.strip():
            lora_path = lora_path_override.strip()
            if not os.path.exists(lora_path):
                raise ValueError(f"LoRA file not found: {lora_path}")
        else:
            if lora_name == "none" or not lora_name:   # pass through
                return (pipeline, _stack_settings(pipeline.get("lora_stack", [])),
                        "", prompt or "")
            lora_path = get_lora_full_path(lora_name)
            if lora_path is None:
                raise ValueError(f"LoRA file not found: {lora_name}")

        if not per_stage_weights:
            weight_s1 = weight_s2 = weight_s3 = strength

        # Copy the pipeline dict + stack list so chaining accumulates immutably and
        # ComfyUI sees a distinct output (the underlying pipe object stays shared).
        new_pipeline = derive_handle(pipeline)
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

        # Trigger words (cached lookup; network only on first sight of a file).
        triggers, merge_mode = [], add_triggers
        if fetch_triggers or add_triggers in ("catalog", "recipe", "full"):
            try:
                triggers, merge_mode, source = resolve_triggers(lora_path, add_triggers, fetch_triggers, force_refetch,
                                                                checkpoint=checkpoint_of(pipeline))
                if add_triggers in ("catalog", "recipe", "full"):
                    print(f"[EricKrea2-LoRA] {add_triggers} words ({source}): {triggers[0] if triggers else '(none)'}")
            except Exception as e:
                print(f"[EricKrea2-LoRA] trigger lookup failed (non-fatal): {e}")
        out_prompt = merge_triggers_into_prompt(
            prompt, triggers, merge_mode,
            pipe=pipeline.get("pipeline") if isinstance(pipeline, dict) else None, join=trigger_join)

        wtxt = (f"all={strength}" if not per_stage_weights
                else f"S1={weight_s1}, S2={weight_s2}, S3={weight_s3}")
        print(f"[EricKrea2-LoRA] queued '{os.path.basename(lora_path)}' as '{adapter_name}' "
              f"({wtxt}, ephemeral={ephemeral}); stack depth {len(stack)}. "
              "Realized at generation by the Multi-Stage Ultra node.")
        return (new_pipeline, _stack_settings(stack), ", ".join(triggers), out_prompt)


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
        new_pipeline = derive_handle(pipeline)
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
