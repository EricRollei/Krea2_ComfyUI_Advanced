# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 Merge Settings
=========================
Combine the `settings` strings from the Component Loader, Apply-LoRA, and
Multi-Stage Ultra V2 nodes into one KREA2_SETTINGS blob, so a single Save Image
node embeds the whole pipeline recipe (loader + lora + ultra) in the PNG.
Dict sections are merged; `lora` lists are concatenated. Empty/absent inputs
are ignored.

The optional `prompt` / `negative_prompt` inputs fold the actual prompt text
into a `prompt` section of the same blob, so the merged output carries the
whole recipe - loader + lora + ultra + sigmas + prompt - in one readable place.
"""

from .._settings import merge, wrap


class EricKrea2MergeSettings:
    CATEGORY = "Eric/Krea2"
    FUNCTION = "merge_settings"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("settings",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "settings_1": ("STRING", {"forceInput": True}),
                "settings_2": ("STRING", {"forceInput": True}),
                "settings_3": ("STRING", {"forceInput": True}),
                "settings_4": ("STRING", {"forceInput": True}),
                "prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"forceInput": True}),
            }
        }

    def merge_settings(self, settings_1="", settings_2="", settings_3="",
                       settings_4="", prompt="", negative_prompt=""):
        strings = [settings_1, settings_2, settings_3, settings_4]
        # Fold the raw prompt text into a structured `prompt` section so it rides
        # along in the same KREA2_SETTINGS blob (and thus the PNG metadata).
        section = {}
        if str(prompt or "").strip():
            section["positive"] = str(prompt).strip()
        if str(negative_prompt or "").strip():
            section["negative"] = str(negative_prompt).strip()
        if section:
            strings.append(wrap("prompt", section))
        return (merge(*strings),)


NODE_CLASS_MAPPINGS = {"EricKrea2MergeSettings": EricKrea2MergeSettings}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2MergeSettings": "Eric Krea2 Merge Settings"}
