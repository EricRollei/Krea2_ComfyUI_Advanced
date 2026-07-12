# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 Multi-LoRA Stack
===========================
A compact, growable alternative to chaining several Apply-LoRA nodes: one node
holds many LoRA slots but only shows the ones you use (plus one empty slot).

How the "grows as needed" UI works
----------------------------------
The node statically declares ``MAX_SLOTS`` rows (``lora_1``/``strength_1`` ...).
That keeps ComfyUI's native dropdowns (auto-populated from the loras folder) and
100% native serialization - nothing custom to save/restore. A small JS extension
(``web/js/krea2_lora_stack.js``) simply HIDES every row past the last used one +1,
so the graph stays small. Pick a LoRA in the last visible row and the next empty
row appears; clear a row and the tail collapses again.

Realization is identical to the single Apply-LoRA node: this node only appends
declarative entries onto ``pipeline["lora_stack"]``. The Multi-Stage Ultra node
loads the whole stack fresh at generation and (if any entry is ephemeral) clears
it afterward. Chain this after / before single Apply-LoRA nodes freely; adapter
names are de-duped across the whole stack.

Per-slot per-stage weights are intentionally omitted here to stay compact - use
the single "Apply LoRA" node for per-stage (S1/S2/S3) weighting.

Author: Eric Hiss (GitHub: EricRollei)
"""

import os

from .._lora_utils import get_lora_list, get_lora_full_path
from .krea2_lora import _sanitize_adapter_name, _stack_settings
from .. import _settings


class EricKrea2MultiLoRA:
    """Stack several LoRAs in one compact node. Only used slots (plus one empty)
    are shown on the graph. Nothing is loaded here - the Multi-Stage Ultra node
    realizes the whole stack fresh at generation and clears ephemeral ones after."""

    CATEGORY = "Eric/Krea2"
    FUNCTION = "apply"
    RETURN_TYPES = ("KREA2_PIPELINE", "STRING")
    RETURN_NAMES = ("pipeline", "settings")

    # Number of LoRA rows declared. The JS extension hides unused rows so this
    # can be generous without cluttering the graph.
    MAX_SLOTS = 10

    @classmethod
    def INPUT_TYPES(cls):
        loras = get_lora_list("krea2")
        optional = {
            "lora_preset": (_settings.list_preset_names("lora"), {"default": "custom",
                "tooltip": "Load a saved LoRA-stack recipe (slots / strengths / toggles) into the "
                           "panel for this run. 'custom' = use the panel as-is. Use the ★ Save Preset "
                           "button to add one; the list refreshes on graph reload."}),
            "ephemeral": ("BOOLEAN", {"default": True,
                "tooltip": "ON (recommended): load these LoRAs fresh each run and unload them "
                           "after generation (no stale/orphaned weights). OFF: leave them resident "
                           "(faster reuse, but Unload manually to free VRAM / before switching "
                           "workflows). Applies to every slot in this node."}),
        }
        for i in range(1, cls.MAX_SLOTS + 1):
            optional[f"on_{i}"] = ("BOOLEAN", {"default": True, "label_on": "on", "label_off": "off",
                "tooltip": f"Enable/disable LoRA slot {i} without clearing it. OFF keeps the file and "
                           "strength but skips loading it (distinct from 'none' = empty slot)."})
            optional[f"lora_{i}"] = (loras, {"default": "none",
                "tooltip": (f"LoRA slot {i} (from ComfyUI/models/loras/Krea2/). Pick a file to "
                            "activate it - a new empty slot appears below. 'none' = unused/off."
                            if i == 1 else
                            f"LoRA slot {i}. 'none' = unused/off.")})
            optional[f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 50000,
                "step": 0.05,
                "tooltip": f"Weight for LoRA slot {i} (applied to all stages)."})
        return {"required": {"pipeline": ("KREA2_PIPELINE",)}, "optional": optional}

    def apply(self, pipeline, ephemeral=True, lora_preset="custom", **kwargs):
        # Headless / API-run fallback: apply a named preset to the slot values (the
        # JS normally writes these into the visible panel at edit time).
        if lora_preset and lora_preset != "custom":
            _vals = dict(kwargs)
            _vals["ephemeral"] = ephemeral
            n = _settings.apply_named_preset("lora", lora_preset, _vals, None)
            ephemeral = bool(_vals.pop("ephemeral", ephemeral))
            kwargs = _vals
            print(f"[EricKrea2-LoRA] lora_preset '{lora_preset}': applied {n} field(s).")

        # Copy the pipeline dict + stack list so chaining accumulates immutably and
        # ComfyUI sees a distinct output (the underlying pipe object stays shared).
        new_pipeline = dict(pipeline)
        stack = list(pipeline.get("lora_stack", []))
        existing = {e["adapter_name"] for e in stack}

        added = 0
        for i in range(1, self.MAX_SLOTS + 1):
            name = kwargs.get(f"lora_{i}", "none")
            if not name or name == "none":
                continue
            if not bool(kwargs.get(f"on_{i}", True)):   # slot disabled via toggle
                continue
            path = get_lora_full_path(name)
            if path is None:
                print(f"[EricKrea2-LoRA] Multi-LoRA slot {i}: '{name}' not found, skipped.")
                continue
            strength = float(kwargs.get(f"strength_{i}", 1.0))

            adapter_name = _sanitize_adapter_name(path)
            if adapter_name in existing:   # de-dupe across the whole stack
                n = 2
                while f"{adapter_name}_{n}" in existing:
                    n += 1
                adapter_name = f"{adapter_name}_{n}"
            existing.add(adapter_name)

            stack.append({
                "path": path,
                "filename": os.path.basename(path),
                "lora_name": name,
                "adapter_name": adapter_name,
                "strength": strength,
                "weight_s1": strength,
                "weight_s2": strength,
                "weight_s3": strength,
                "ephemeral": bool(ephemeral),
            })
            added += 1

        new_pipeline["lora_stack"] = stack
        print(f"[EricKrea2-LoRA] Multi-LoRA queued {added} adapter(s) "
              f"(ephemeral={ephemeral}); stack depth {len(stack)}. "
              "Realized at generation by the Multi-Stage Ultra node.")
        return (new_pipeline, _stack_settings(stack))


NODE_CLASS_MAPPINGS = {"EricKrea2MultiLoRA": EricKrea2MultiLoRA}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2MultiLoRA": "Eric Krea2 Multi-LoRA Stack"}
