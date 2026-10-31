# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
The words a sweep cell adds for its swept LoRAs (2026-10-16).

One function, used by Ultra V2 when it runs a cell and by the Sweep Plan's preview, so the preview shows exactly
what the run will send. Modes (nodes/krea2_lora.py canon_trigger_mode): off | catalog (trigger) | full | recipe
(saved pick) | prepend | append.

Word options (2026-10-20): an entry with "word_option" (a sweep line's 'name #option', the Multi-LoRA Stack's
word_options field) gets that option's words from the LoRA Catalog instead of its usual words, in every mode except
off. An option the card does not have: a note, and the usual words.
"""

from __future__ import annotations

import os


_SOURCE_NAMES = {
    "recipe (you)": "your pick for this checkpoint", "recipe (test)": "trigger test's pick for this checkpoint",
    "recipe (picked)": "your pick (all checkpoints)", "recipe (tested)": "a trigger test's pick (other checkpoint)",
    "recipe (recipe)": "no pick: trigger + style phrase", "recipe (trigger)": "no pick, no recipe: card trigger",
    "full (recipe full)": "full recipe", "full (recipe trigger+phrase)": "trigger + style phrase",
    "full (recipe official)": "official words", "full (recipe trigger)": "trigger only (no phrase)",
    "full (card triggers)": "all card triggers (no recipe)", "full (civitai words)": "Civitai words (no card)",
    "card": "card trigger", "lookup": "Civitai trigger", "none": "nothing found",
}


def source_name(src: str) -> str:
    return _SOURCE_NAMES.get(str(src), str(src))


def cell_words(entries, mode, checkpoint="", prompt=""):
    """entries: the cell's sweep-injected LoRAs, each {"path", "filename"?, "weight_s1"?}.
    Returns (words, sources, report): words = the list to merge into the prompt, sources = one line per LoRA saying
    where its words came from, report = the merge report for full / saved pick (conflicts etc.) or {}."""
    from .nodes.krea2_lora import canon_trigger_mode, resolve_triggers
    from ._trigger_words import get_trigger_words
    mode = canon_trigger_mode(mode)
    words, seen, sources, items, report = [], set(), [], [], {}
    if mode == "off":
        return words, sources, report
    for e in entries or []:
        path = e.get("path") or ""
        fname = e.get("filename") or os.path.basename(path)
        wanted = str(e.get("word_option") or "").strip()
        if wanted:
            from ._catalog_lookup import find_option, word_options
            opt = find_option(path, wanted)
            if opt is not None:
                text = str(opt.get("text") or "").strip()
                sources.append(f"{fname}: word option '{opt.get('name')}'" + ("" if text else " (no words)"))
                if text and mode in ("recipe", "full"):
                    items.append({"name": fname, "text": text, "weight": e.get("weight_s1", 1.0),
                                  "source": f"word option {opt.get('name')}"})
                elif text and text.lower() not in seen:
                    seen.add(text.lower())
                    words.append(text)
                continue
            have = [str(o.get("name")) for o in word_options(path)]
            sources.append(f"{fname}: word option '{wanted}' not found ("
                           + ("has: " + ", ".join(have[:12]) if have else "no options on its card")
                           + ") - usual words used")
        if mode in ("recipe", "full"):
            w, _m, src = resolve_triggers(path, mode, True, False, checkpoint=checkpoint)
            sources.append(f"{fname}: {source_name(src)}" + ("" if w else " (no words)"))
            if w:
                items.append({"name": fname, "text": w[0], "weight": e.get("weight_s1", 1.0), "source": src})
            cands = []
        elif mode == "catalog":
            # one trigger per LoRA: the card's first active trigger, else the first cached Civitai one
            from ._catalog_lookup import card_trigger
            t, src = card_trigger(path), "card"
            if not t:
                t = next((x for x in get_trigger_words(path) if x), "")
                src = "civitai" if t else "none"
            sources.append(f"{fname}: {source_name(src)}" + (f" '{t}'" if t else ""))
            cands = [t] if t else []
        else:
            cands = get_trigger_words(path)
            sources.append(f"{fname}: civitai words ({len(cands)})")
        for t in cands:
            if t and t.lower() not in seen:
                seen.add(t.lower())
                words.append(t)
    if items:
        from ._prompt_merge import merge_recipes
        phrases, report = merge_recipes(items, str(prompt or ""))
        for t in phrases:
            if t.lower() not in seen:
                seen.add(t.lower())
                words.append(t)
    return words, sources, report
