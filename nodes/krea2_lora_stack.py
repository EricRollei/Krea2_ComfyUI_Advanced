# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 Multi-LoRA Stack
===========================
A compact, growable alternative to chaining several Apply-LoRA nodes: one node
holds many LoRA slots but only shows the ones you use (plus one empty slot).

How the "grows as needed" UI works
----------------------------------
The node statically declares ``MAX_SLOTS`` rows (``lora_1``/``strength_1s1..s3`` ...).
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

Each slot carries three per-stage strengths (``strength_is1/is2/is3`` for
stages S1/S2/S3) instead of a single strength + a mode toggle - set all three
equal for a uniform weight, or shape them per stage (e.g. detail LoRA
0 / 0.7 / 1.0, style LoRA 1.0 / 0.6 / 0.3).

Word options (2026-10-20): some LoRAs are trained on alternatives (one keyword per colour grade, per style). The
LoRA Catalog keeps them as named word options; the ``word_options`` field picks one per slot, one line each,
``3: teal orange`` (slot 3 gets that option's words instead of its usual ones). The "Pick words" button fills
the field from the catalog. Lines for empty / switched-off slots are ignored.

Author: Eric Hiss (GitHub: EricRollei)
"""

from .._pipe_handles import derive_handle
import os

from .._lora_utils import get_lora_list, get_lora_full_path
from .._trigger_words import get_trigger_words, merge_triggers_into_prompt
from .krea2_lora import (TRIGGER_MODES, TRIGGER_TOOLTIP, _sanitize_adapter_name, _stack_settings, resolve_triggers,
                         checkpoint_of, canon_trigger_mode)
from .. import _settings


class EricKrea2MultiLoRA:
    """Stack several LoRAs in one compact node. Only used slots (plus one empty)
    are shown on the graph. Nothing is loaded here - the Multi-Stage Ultra node
    realizes the whole stack fresh at generation and clears ephemeral ones after."""

    CATEGORY = "Eric/Krea2"
    FUNCTION = "apply"
    # trigger_words / prompt outputs appended at the END so existing links
    # (by index) keep working in saved workflows.
    RETURN_TYPES = ("KREA2_PIPELINE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("pipeline", "settings", "trigger_words", "prompt")

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
            optional[f"strength_{i}s1"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 50000,
                "step": 0.05,
                "tooltip": f"Slot {i} weight during Stage-1 (composition). Set all three equal "
                           "for a uniform strength."})
            optional[f"strength_{i}s2"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 50000,
                "step": 0.05,
                "tooltip": f"Slot {i} weight during Stage-2 (refine)."})
            optional[f"strength_{i}s3"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 50000,
                "step": 0.05,
                "tooltip": f"Slot {i} weight during Stage-3 (final detail)."})
        # Node-level widgets appended AFTER all slot rows (positional
        # widgets_values safety; the row-hiding JS matches slot widgets by name
        # and never touches these). 'prompt' is a socket (forceInput), so it
        # does not occupy a widgets_values slot at all.
        optional["stack_enabled"] = ("BOOLEAN", {"default": True,
            "label_on": "on", "label_off": "off",
            "tooltip": "Master switch for the WHOLE stack. OFF = pass the pipeline through "
                       "untouched (no slot is queued, no triggers emitted) without clearing "
                       "any slot settings."})
        optional["fetch_triggers"] = ("BOOLEAN", {"default": True,
            "tooltip": "Look up trigger words for every ENABLED slot (safetensors metadata / "
                       "Civitai, cached locally in Eric_Krea2/data/lora_triggers.json - only "
                       "the first lookup per file touches the network). Fills the "
                       "trigger_words output (combined, de-duplicated, slot order)."})
        optional["add_triggers"] = (TRIGGER_MODES, {"default": "none",
            "tooltip": TRIGGER_TOOLTIP + "\nApplies to every enabled slot; the words go into the 'prompt' "
                       "output (connect your prompt to the prompt input)."})
        optional["force_refetch"] = ("BOOLEAN", {"default": False,
            "tooltip": "Bypass the local trigger cache and re-query Civitai for the enabled "
                       "slots this run. Leave OFF normally."})
        optional["prompt"] = ("STRING", {"default": "", "forceInput": True,
            "tooltip": "Optional prompt passthrough. Connect your prompt here and take the "
                       "'prompt' output onward; with add_triggers on, the enabled slots' "
                       "trigger words are merged in."})
        # Appended 2026-10-02 (a socket: no widgets_values slot, saved workflows untouched).
        optional["suggestions"] = ("LORA_SUGGESTIONS", {
            "tooltip": "Optional: LoRAs picked by the LoRA Suggester (video_prompter). They are added after "
                       "this panel's own slots with their per-stage weights; a file already in the stack is "
                       "not added twice. Their trigger words join the trigger merge."})
        # Appended LAST 2026-10-20 (widget values are saved by position; the two sockets above take none).
        optional["word_options"] = ("STRING", {"multiline": True, "default": "",
            "tooltip": "Pick one of a LoRA's word options (LoRA Catalog card > Prompt words > Word options) per "
                       "slot, one line each: '3: teal orange' (a name, its number, or the start of its name). "
                       "That slot's words become the option's words, in every add_triggers mode except none; "
                       "the trigger_words output carries them too. Use the 'Pick words' button to choose "
                       "from a list. '3: random' = a different option every run. An amber dot left of a slot = its "
                       "LoRA has word options and none is chosen (click it to pick). Empty = every slot uses its "
                       "usual words."})
        # Appended LAST 2026-10-29 (positional widget values).
        optional["trigger_join"] = (["comma", "period"], {"default": "comma", "tooltip": "How the trigger words join the prompt (when add_triggers is on). comma: 'trigA, trigB, your prompt' (the behaviour before 2026-10-29). period: each LoRA's words become their own sentence - 'trigA. trigB. Your prompt' - the way many Civitai prompts start; it may stop a trigger being read as text to draw (being tested with the Sweep Plan trigger_joins axis)."})
        return {"required": {"pipeline": ("KREA2_PIPELINE",)}, "optional": optional}

    @classmethod
    def IS_CHANGED(cls, word_options="", **kwargs):
        # '3: random' (2026-10-26) draws a new word option every queue, so never reuse the cached result then.
        if any(ln.partition(":")[2].strip().lower() == "random" for ln in str(word_options or "").splitlines()):
            return float("nan")
        return ""

    @staticmethod
    def parse_word_options(text):
        """'3: teal orange' lines -> {3: 'teal orange'}. Other lines are ignored (with a note)."""
        out = {}
        for ln in str(text or "").splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            slot, sep, option = ln.partition(":")
            if sep and slot.strip().isdigit() and option.strip():
                out[int(slot.strip())] = option.strip()
            else:
                print(f"[EricKrea2-LoRA] word_options: '{ln}' ignored (write 'slot: option', e.g. '3: teal orange')")
        return out

    def apply(self, pipeline, ephemeral=True, lora_preset="custom",
              stack_enabled=True, fetch_triggers=True, add_triggers="off",
              force_refetch=False, prompt="", suggestions=None, word_options="", trigger_join="comma", **kwargs):
        add_triggers = canon_trigger_mode(add_triggers)
        slot_options = self.parse_word_options(word_options)
        if not stack_enabled:
            print("[EricKrea2-LoRA] Multi-LoRA stack is toggled OFF - passing pipeline through.")
            return (pipeline, _stack_settings(pipeline.get("lora_stack", [])),
                    "", prompt or "")

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
        new_pipeline = derive_handle(pipeline)
        stack = list(pipeline.get("lora_stack", []))
        existing = {e["adapter_name"] for e in stack}

        added = 0
        triggers = []          # combined across enabled slots, slot order, de-duped
        _seen_trig = set()
        recipe_items = []      # recipe mode: every LoRA's words, merged once at the end (2026-10-16)
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
            # Per-stage strengths. An old-style preset / API prompt may still carry a
            # single strength_{i}: honour it as the fallback for all three stages.
            legacy = kwargs.get(f"strength_{i}", 1.0)
            s1 = float(kwargs.get(f"strength_{i}s1", legacy))
            s2 = float(kwargs.get(f"strength_{i}s2", legacy))
            s3 = float(kwargs.get(f"strength_{i}s3", legacy))

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
                "strength": s1,          # legacy field; realization uses weight_s1/s2/s3
                "weight_s1": s1,
                "weight_s2": s2,
                "weight_s3": s3,
                "ephemeral": bool(ephemeral),
            })
            added += 1

            # A word option for this slot (2026-10-20): its words replace the slot's usual ones.
            if i in slot_options:
                from .._catalog_lookup import find_option, word_options as _options_of
                if str(slot_options[i]).strip().lower() == "random":     # 2026-10-26: new option each run
                    import random as _random
                    _all = _options_of(path)
                    opt = _random.choice(_all) if _all else None
                else:
                    opt = find_option(path, slot_options[i])
                if opt is not None:
                    text = str(opt.get("text") or "").strip()
                    print(f"[EricKrea2-LoRA] slot {i}: word option '{opt.get('name')}' -> '{text}'")
                    stack[-1]["word_option"] = str(opt.get("name"))     # recorded in the image's LoRA metadata
                    if text and add_triggers in ("recipe", "full"):
                        recipe_items.append({"name": os.path.basename(path), "text": text, "weight": s1,
                                             "source": f"word option {opt.get('name')}"})
                    elif text and text.lower() not in _seen_trig:
                        _seen_trig.add(text.lower())
                        triggers.append(text)
                    continue
                have = [str(o.get("name")) for o in _options_of(path)]
                print(f"[EricKrea2-LoRA] slot {i}: word option '{slot_options[i]}' not found for "
                      f"{os.path.basename(path)} (" + ("has: " + ", ".join(have[:12]) if have else
                                                       "no options on its card") + ") - usual words")

            # Trigger words for this ENABLED slot (cached lookup; network only
            # the first time a file is seen). Disabled/'none' slots never reach
            # this point, so their triggers are never emitted.
            if fetch_triggers or add_triggers in ("catalog", "recipe", "full"):
                try:
                    found, _merge, _source = resolve_triggers(path, add_triggers, fetch_triggers, force_refetch,
                                                              checkpoint=checkpoint_of(pipeline))
                    if add_triggers in ("recipe", "full"):
                        if found:
                            recipe_items.append({"name": os.path.basename(path), "text": found[0],
                                                 "weight": s1, "source": _source})
                        found = []
                    for t in found:
                        k = t.lower()
                        if k not in _seen_trig:
                            _seen_trig.add(k)
                            triggers.append(t)
                except Exception as e:
                    print(f"[EricKrea2-LoRA] trigger lookup failed for slot {i} "
                          f"(non-fatal): {e}")

        # LoRAs from the LoRA Suggester (video_prompter), after the panel's own slots.
        stacked_paths = {os.path.normcase(e.get("path", "")) for e in stack}
        for entry in suggestions or []:
            path = entry.get("path") if isinstance(entry, dict) else None
            if not path or not os.path.isfile(path):
                print(f"[EricKrea2-LoRA] suggestion skipped, file not found: {path}")
                continue
            if os.path.normcase(path) in stacked_paths:
                continue
            stacked_paths.add(os.path.normcase(path))
            adapter_name = _sanitize_adapter_name(path)
            if adapter_name in existing:
                n = 2
                while f"{adapter_name}_{n}" in existing:
                    n += 1
                adapter_name = f"{adapter_name}_{n}"
            existing.add(adapter_name)
            s1 = float(entry.get("weight_s1", 1.0))
            stack.append({
                "path": path,
                "filename": os.path.basename(path),
                "lora_name": entry.get("lora_name") or os.path.basename(path),
                "adapter_name": adapter_name,
                "strength": s1,
                "weight_s1": s1,
                "weight_s2": float(entry.get("weight_s2", s1)),
                "weight_s3": float(entry.get("weight_s3", s1)),
                "ephemeral": bool(ephemeral),
            })
            added += 1
            if add_triggers in ("recipe", "full"):  # the catalog's words for this LoRA, like the panel slots
                try:
                    sugg, _m, _s = resolve_triggers(path, add_triggers, fetch_triggers, force_refetch,
                                                    checkpoint=checkpoint_of(pipeline))
                except Exception:
                    sugg, _s = (entry.get("triggers") or [])[:1], ""
                if sugg:
                    recipe_items.append({"name": os.path.basename(path), "text": sugg[0], "weight": s1,
                                         "source": _s})
                sugg = []
            else:
                sugg = (entry.get("triggers") or [])[:1 if add_triggers == "catalog" else None]
            for t in sugg:
                if t and t.lower() not in _seen_trig:
                    _seen_trig.add(t.lower())
                    triggers.append(t)
            print(f"[EricKrea2-LoRA] + suggested {entry.get('role', '')}: {entry.get('name', os.path.basename(path))} "
                  f"S1 {s1} / S2 {entry.get('weight_s2')} / S3 {entry.get('weight_s3')}")

        if recipe_items:
            # one tidy phrase list for the whole stack: repeats dropped, picks kept whole, a word budget by
            # weight, opposite words reported (_prompt_merge)
            from .._prompt_merge import merge_recipes, report_lines
            phrases, rep = merge_recipes(recipe_items, prompt or "")
            for t in phrases:
                if t.lower() not in _seen_trig:
                    _seen_trig.add(t.lower())
                    triggers.append(t)
            for line in report_lines(rep):
                print(f"[EricKrea2-LoRA] prompt words: {line}")
        new_pipeline["lora_stack"] = stack
        out_prompt = merge_triggers_into_prompt(
            prompt, triggers, "prepend" if add_triggers in ("catalog", "recipe", "full") else add_triggers,
            pipe=pipeline.get("pipeline") if isinstance(pipeline, dict) else None, join=trigger_join)
        print(f"[EricKrea2-LoRA] Multi-LoRA queued {added} adapter(s) "
              f"(ephemeral={ephemeral}); stack depth {len(stack)}. "
              "Realized at generation by the Multi-Stage Ultra node.")
        return (new_pipeline, _stack_settings(stack), ", ".join(triggers), out_prompt)


NODE_CLASS_MAPPINGS = {"EricKrea2MultiLoRA": EricKrea2MultiLoRA}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2MultiLoRA": "Eric Krea2 Multi-LoRA Stack"}
