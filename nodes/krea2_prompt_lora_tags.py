# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 Prompt LoRA Tags (2026-10-15)
========================================
Reads <lora:...> tags out of a prompt, removes them from the text, and hands the LoRAs to the
Multi-LoRA Stack through its `suggestions` socket (the same socket the LoRA Suggester uses), so
the stack adds them after its own slots, never twice, with their trigger words in the merge.

    a woman in a red coat <lora:Shouji_Ink_KR2:0.66>, rain
    <lora:Shouji_Ink_KR2>                 weight 1.0
    <lora:Shouji_Ink_KR2:0.66>            0.66 on all three stages
    <lora:Shouji_Ink_KR2:1/0.7/0.4>       per stage S1/S2/S3
    <lora:Krea2/graphic-art/foo.safetensors:0.8>   a loras-folder path (or a full path)

A name is found like the Sweep Plan finds one: full path, loras-folder path (extension optional),
or the bare file name / stem in any subfolder. A bare name that matches several files, or no file,
stops the run with a clear message (on_missing = skip: dropped with a console note instead).
A1111-style '<lora:name:0.8:0.5>' (model:text-encoder) uses the first number.

Wiring: prompt in -> `prompt` out to the text encoder / Multi-LoRA Stack prompt input;
`loras` out -> the Multi-LoRA Stack's `suggestions` input.

Author: Eric Hiss (GitHub: EricRollei)
"""

from __future__ import annotations

import os
import re

_TAG = re.compile(r"<lora:([^<>]+)>", re.IGNORECASE)
_WEIGHTS = re.compile(r"^\s*-?\d*\.?\d+(\s*/\s*-?\d*\.?\d+){0,2}\s*$")


def _weights(text):
    ws = [float(x) for x in text.split("/")]
    if len(ws) == 1:
        ws = ws * 3
    if len(ws) != 3:
        raise ValueError(f"Prompt LoRA Tags: '{text}' - write one weight or three (s1/s2/s3)")
    return ws


def parse_tags(prompt: str):
    """-> (prompt without tags, [(name, (w1, w2, w3), tag text), ...])."""
    found = []
    for m in _TAG.finditer(str(prompt or "")):
        body = m.group(1).strip()
        name, ws = body, [1.0, 1.0, 1.0]
        if ":" in body:
            head, tail = body.rsplit(":", 1)
            if _WEIGHTS.match(tail):
                name, ws = head.strip(), _weights(tail.strip())
                if ":" in name:                      # A1111 <lora:name:unet:te> - keep the first weight
                    h2, t2 = name.rsplit(":", 1)
                    if _WEIGHTS.match(t2) and "/" not in t2:
                        name, ws = h2.strip(), _weights(t2.strip())
        found.append((name, tuple(ws), m.group(0)))
    clean = _TAG.sub(" ", str(prompt or ""))
    clean = re.sub(r"[ \t]+", " ", clean)
    clean = re.sub(r"\s*,\s*(,\s*)+", ", ", clean)        # ', ,' left where a tag was
    clean = re.sub(r" +([,.;:])", r"\1", clean)
    clean = "\n".join(ln.strip() for ln in clean.splitlines()).strip().strip(",").strip()
    return clean, found


class EricKrea2PromptLoraTags:
    """Pull <lora:name:weight> tags out of a prompt and feed them to the Multi-LoRA Stack."""

    CATEGORY = "Eric/Krea2"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "LORA_SUGGESTIONS", "STRING")
    RETURN_NAMES = ("prompt", "loras", "summary")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Prompt with <lora:name:weight> tags anywhere in it. The tags are removed "
                               "from the 'prompt' output. Weight: one value for all stages, or s1/s2/s3; "
                               "no weight = 1.0."}),
            },
            "optional": {
                "fetch_triggers": ("BOOLEAN", {"default": True,
                    "tooltip": "Attach each tagged LoRA's trigger words (its LoRA Catalog card trigger "
                               "first, then the cached Civitai/metadata lookup) so the Multi-LoRA Stack's "
                               "add_triggers can merge them like its own slots."}),
                "on_missing": (["error", "skip"], {"default": "error",
                    "tooltip": "A tag whose LoRA can't be found (or matches several files): error = stop "
                               "the run with the reason; skip = drop it with a console note."}),
            },
        }

    def run(self, prompt="", fetch_triggers=True, on_missing="error"):
        from .._lora_names import find_lora, _rel
        clean, tags = parse_tags(prompt)
        loras, lines, seen = [], [], set()
        for name, (w1, w2, w3), tag in tags:
            try:
                path = find_lora(name)
                if not path:
                    raise ValueError(f"Prompt LoRA Tags: {tag} - no LoRA named '{name}' "
                                     f"(loras folder, any subfolder, or a full path)")
            except ValueError as e:
                if on_missing == "skip":
                    print(f"[EricKrea2-PromptLoRA] skipped: {e}")
                    lines.append(f"skipped {tag}")
                    continue
                raise
            key = os.path.normcase(path)
            if key in seen:
                print(f"[EricKrea2-PromptLoRA] {tag}: already tagged once - the first tag wins")
                continue
            seen.add(key)
            triggers = []
            if fetch_triggers:
                try:
                    from .krea2_lora import resolve_triggers
                    from .._trigger_words import get_trigger_words
                    first, _m, _s = resolve_triggers(path, "catalog", True, False)
                    for t in list(first) + list(get_trigger_words(path) or []):
                        if t and t.lower() not in {x.lower() for x in triggers}:
                            triggers.append(t)
                except Exception as e:
                    print(f"[EricKrea2-PromptLoRA] trigger lookup failed for {os.path.basename(path)} "
                          f"(non-fatal): {e}")
            loras.append({"path": path, "lora_name": _rel(path),
                          "name": os.path.splitext(os.path.basename(path))[0],
                          "role": "prompt tag", "weight_s1": w1, "weight_s2": w2, "weight_s3": w3,
                          "triggers": triggers})
            wtxt = f"{w1:g}" if w1 == w2 == w3 else f"{w1:g}/{w2:g}/{w3:g}"
            lines.append(f"{os.path.basename(path)} @{wtxt}"
                         + (f"  triggers: {', '.join(triggers[:3])}" if triggers else ""))
        summary = "\n".join(lines) if lines else "no <lora:...> tags"
        print(f"[EricKrea2-PromptLoRA] {len(loras)} LoRA(s) from prompt tags"
              + (": " + "; ".join(lines) if lines else ""))
        return (clean, loras, summary)


NODE_CLASS_MAPPINGS = {"EricKrea2PromptLoraTags": EricKrea2PromptLoraTags}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2PromptLoraTags": "Eric Krea2 Prompt LoRA Tags"}
