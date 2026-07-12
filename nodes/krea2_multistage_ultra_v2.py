# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.

"""
Eric Krea2 Multi-Stage Ultra V2 (presets)
=========================================
A thin wrapper around :class:`EricKrea2MultistageUltra` that adds the shared
``KREA2_SETTINGS`` recipe system (see Krea2_Settings_Preset_Spec):

  * ``settings`` output - pretty-printed JSON of this node's ``ultra`` section,
    wrapped in the top-level schema. Feed it into a Save Image node to embed the
    recipe in the PNG, or into a merge node. The same serializer backs saved
    presets, so a PNG's recipe and a saved preset are interchangeable.
  * ``ultra_preset`` dropdown - load a named recipe from ``ultra_presets.json``
    and override the panel values for that run (partial presets supported).
  * ★ Save Preset button (JS) + ``/eric_krea2/save_preset`` endpoint write the
    current widget values into ``ultra_presets.json`` under a name.

The heavy multistage engine is inherited unchanged; nothing in the shipping v1
node is disturbed. The authoritative ``ultra`` field list is derived live from
the inherited ``INPUT_TYPES`` (never hardcoded), minus per-run / non-serializable
inputs (pipeline, VAE connections, prompt, seed).
"""

from __future__ import annotations

import json
import os

from .krea2_multistage_ultra import EricKrea2MultistageUltra

# ComfyUI primitive widget types that serialize cleanly to JSON.
_PRIMITIVE_TYPES = {"STRING", "INT", "FLOAT", "BOOLEAN"}

# Keys excluded from the recipe: object connections (not JSON), the preset
# control itself, and per-run values (prompt/seed are not part of a reusable
# recipe - this also keeps the metadata string == saved preset content).
_SETTINGS_EXCLUDE = {
    "krea2_pipeline", "ultra_preset",
    "prompt", "negative_prompt", "seed",
    "upscale_vae", "decode_vae",
}

KREA2_SETTINGS_VERSION = 1


def _ultra_presets_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "ultra_presets.json")


def _load_ultra_presets() -> dict:
    """Tolerant loader mirroring ``_load_cond_presets``: returns ``{}`` on any
    error and ignores non-dict entries."""
    try:
        with open(_ultra_presets_path(), "r", encoding="utf-8") as f:
            d = json.load(f)
        return {k: v for k, v in d.items() if isinstance(v, dict)} if isinstance(d, dict) else {}
    except Exception:
        return {}


class EricKrea2MultistageUltraV2(EricKrea2MultistageUltra):
    """Ultra multistage generate + save/load presets + a settings metadata output."""

    @classmethod
    def INPUT_TYPES(cls):
        # Start from the inherited schema so we always track field changes, then
        # inject the ultra_preset dropdown at the top of `optional`.
        base = EricKrea2MultistageUltra.INPUT_TYPES()
        preset_names = ["custom"] + sorted(_load_ultra_presets().keys())
        preset_widget = {
            "ultra_preset": (preset_names, {"default": "custom",
                "tooltip": "Load a saved recipe from ultra_presets.json and override the panel "
                           "values for this run (partial presets only override the fields they "
                           "contain). 'custom' = use the panel as-is. Use the ★ Save Preset button "
                           "to add one; the dropdown refreshes after a graph reload."}),
        }
        merged_optional = {}
        merged_optional.update(preset_widget)
        merged_optional.update(base.get("optional", {}))
        base["optional"] = merged_optional
        return base

    # Append `settings` as the LAST output so existing wired outputs keep index.
    RETURN_TYPES = EricKrea2MultistageUltra.RETURN_TYPES + ("STRING",)
    RETURN_NAMES = EricKrea2MultistageUltra.RETURN_NAMES + ("settings",)
    FUNCTION = "generate"
    CATEGORY = "Eric/Krea2"

    # ── settings schema ────────────────────────────────────────────────────
    @classmethod
    def ultra_setting_keys(cls):
        """The serializable ``ultra`` recipe keys, derived live from INPUT_TYPES."""
        keys = []
        it = cls.INPUT_TYPES()
        for section in ("required", "optional"):
            for name, spec in it.get(section, {}).items():
                if name in _SETTINGS_EXCLUDE:
                    continue
                t = spec[0] if isinstance(spec, (tuple, list)) and spec else spec
                if isinstance(t, list):            # combo widget
                    keys.append(name)
                elif isinstance(t, str) and t in _PRIMITIVE_TYPES:
                    keys.append(name)
        return keys

    @classmethod
    def serialize_settings(cls, values: dict) -> str:
        """Serialize the ``ultra`` section from a values dict (pretty JSON).
        The same function backs both the metadata output and saved presets."""
        keys = cls.ultra_setting_keys()
        ultra = {k: values[k] for k in keys if k in values}
        blob = {"krea2_settings_version": KREA2_SETTINGS_VERSION, "ultra": ultra}
        return json.dumps(blob, indent=2, ensure_ascii=False)

    # ── run ────────────────────────────────────────────────────────────────
    def generate(self, **kwargs):
        # 1) Apply a named preset (before anything reads the values).
        preset = str(kwargs.pop("ultra_preset", "custom") or "custom")
        if preset and preset != "custom":
            entry = _load_ultra_presets().get(preset, {})
            ultra = entry.get("ultra", entry) if isinstance(entry, dict) else {}
            valid = set(self.ultra_setting_keys())
            applied = 0
            for k, v in (ultra or {}).items():
                if k in valid:
                    kwargs[k] = v
                    applied += 1
            print(f"[EricKrea2-MS] ultra_preset '{preset}': applied {applied} field(s).")

        # 2) Build the settings string from the (possibly overridden) values,
        #    BEFORE the base pops cond_* out of kwargs.
        settings = self.serialize_settings(kwargs)

        # 2b) If a sigma-shape bundle is wired in, fold its enabled stages into the
        #     recipe as a "sigmas" section (so the master image's metadata carries
        #     the authored curves alongside the ultra fields).
        sig_bundle = kwargs.get("sigmas")
        if isinstance(sig_bundle, dict):
            section = {k: v for k, v in sig_bundle.items()
                       if k in ("s1", "s2", "s3") and isinstance(v, dict) and v.get("enabled", True)}
            if section:
                from .. import _settings
                settings = _settings.merge(settings, _settings.wrap("sigmas", section))

        # 3) Run the inherited engine and append settings as the last output.
        result = super().generate(**kwargs)
        if not isinstance(result, tuple):
            result = (result,)
        return result + (settings,)


NODE_CLASS_MAPPINGS = {"EricKrea2MultistageUltraV2": EricKrea2MultistageUltraV2}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2MultistageUltraV2": "Eric Krea2 Multi-Stage Ultra V2 (presets)"}
