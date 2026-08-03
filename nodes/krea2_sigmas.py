# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 Sigmas
=================
Author a per-stage sigma-curve *shape* and feed it into the Multi-Stage Ultra
node's optional ``sigmas`` input. The bundle is a SHAPE-SPEC, not a realized
array: each stage carries a curve + knobs and is realized at the Ultra node's
own ``s1/s2/s3_steps`` (so it's resolution- and step-count-independent, and one
saved shape works on any run).

Per stage (s1/s2/s3):
  * enable       - when off, that stage falls back to the Ultra node's own
                   schedule dropdown (mix authored + panel stages freely).
  * curve        - linear / balanced / karras / beta57 / beta / bong_tangent / linear_quadratic /
                   exponential (same library the Ultra dropdowns use).
  * detail_bias  - the single friendly knob (-1..+1). >0 packs more steps at low
                   sigma (fine detail); <0 favours high sigma (composition). Works
                   on EVERY curve: folded into Karras rho / bong_tangent slope, and
                   applied as a step redistribution on the others.
  * rho / alpha / beta - advanced overrides (rho 0 = auto from detail_bias;
                   alpha/beta only used by the 'beta' curve).

Shares the ``sigmas`` preset section with the generic preset UI (★ Save Preset /
dropdown), so an authored shape is reproducible the same way loader/lora/ultra
recipes are. When embedded via Ultra V2's settings output it also travels in the
master image's PNG metadata.
"""

from __future__ import annotations

from .. import _settings
from .._sigmas import CURVES

KREA2_SIGMAS_VERSION = 1

_STAGES = ("s1", "s2", "s3")
_STAGE_DEFAULT_CURVE = {"s1": "beta57", "s2": "linear", "s3": "linear"}


class EricKrea2Sigmas:
    """Author a per-stage sigma-curve shape (KREA2_SIGMAS) for the Ultra node."""

    CATEGORY = "Eric/Krea2"
    FUNCTION = "build"
    RETURN_TYPES = ("KREA2_SIGMAS",)
    RETURN_NAMES = ("sigmas",)

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "sigmas_preset": (_settings.list_preset_names("sigmas"), {"default": "custom",
                "tooltip": "Load a saved sigma-shape recipe into the panel for this run. 'custom' = "
                           "use the panel as-is. Use the ★ Save Preset button to add one; the list "
                           "refreshes on graph reload."}),
        }
        for n in _STAGES:
            label = n.upper()
            optional[f"{n}_enable"] = ("BOOLEAN", {"default": True, "label_on": "override", "label_off": "use panel",
                "tooltip": f"When ON, {label} uses this curve and OVERRIDES the Ultra node's {n}_schedule "
                           f"dropdown. When OFF, {label} falls back to that dropdown (mix authored + panel "
                           "stages)."})
            optional[f"{n}_curve"] = (CURVES, {"default": _STAGE_DEFAULT_CURVE[n],
                "tooltip": f"{label} sigma spacing. linear=composition; balanced/karras=more low-sigma "
                           "detail; beta57/beta=RES4LYF beta spacing; bong_tangent=hold-then-drop; "
                           "exponential=geometric."})
            optional[f"{n}_detail_bias"] = ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                "tooltip": f"{label} detail vs composition (-1..+1). >0 packs steps at low sigma (fine "
                           "detail); <0 favours high sigma (structure). Works on every curve "
                           "(Karras rho / bong_tangent slope, or a step redistribution otherwise)."})
            optional[f"{n}_rho"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.5,
                "tooltip": f"{label} advanced: explicit Karras rho for balanced/karras curves. 0 = auto "
                           "(derived from detail_bias). Higher = more low-sigma concentration."})
            optional[f"{n}_alpha"] = ("FLOAT", {"default": 0.5, "min": 0.05, "max": 6.0, "step": 0.05,
                "tooltip": f"{label} advanced: Beta-distribution alpha (only the 'beta' curve). "
                           "beta57 is fixed alpha=0.5."})
            optional[f"{n}_beta"] = ("FLOAT", {"default": 0.7, "min": 0.05, "max": 6.0, "step": 0.05,
                "tooltip": f"{label} advanced: Beta-distribution beta (only the 'beta' curve). "
                           "beta57 is fixed beta=0.7."})
        # APPENDED LAST (2026-07-28): never insert widgets mid-list - ComfyUI
        # saves widget values positionally and mid-list inserts scramble every
        # saved workflow's values below the insert point.
        optional["preset_notes"] = ("STRING", {"multiline": True, "default": "",
            "tooltip": "One or two sentences: what this curve is FOR (which sampler / recipe) "
                       "and what it does. Saved and loaded with the ★ preset."})
        return {"required": {}, "optional": optional}

    def build(self, sigmas_preset="custom", **kwargs):
        # Headless / API-run fallback: apply a named preset to the panel values
        # (the JS normally writes these into the visible widgets at edit time).
        if sigmas_preset and sigmas_preset != "custom":
            if not kwargs:
                # HEADLESS (e.g. the sweep's sigma_profiles axis): the preset
                # is the sole source of truth - apply every stored field
                # (valid_keys=None; the old set(kwargs.keys()) allow-list was
                # empty here, applied 0 fields, and silently built the widget
                # defaults - found 2026-07-27).
                _vals = {}
                n = _settings.apply_named_preset("sigmas", sigmas_preset, _vals, None)
                kwargs = _vals
                _note = str(_vals.get("preset_notes", "")).strip()
                print(f"[EricKrea2-Sigmas] sigmas_preset '{sigmas_preset}': "
                      f"applied {n} field(s) (headless)."
                      + (f' - "{_note}"' if _note else ""))
                if n == 0:
                    print(f"[EricKrea2-Sigmas] WARNING: preset '{sigmas_preset}' is "
                          f"empty or missing - bundle uses PANEL DEFAULTS.")
            else:
                # GUI: widgets RULE. The JS wrote the preset's values into the
                # widgets when it was selected, and anything the user changed
                # since (e.g. flipping a stage's enable off) is intent. Never
                # re-apply stored fields over live widgets at queue time -
                # doing so let a dropdown left on a preset silently re-enable
                # stages and stomp manual tweaks (found 2026-07-27: run-2
                # sweep built-ins flattened by a re-applied EXP6 s1_enable).
                pass

        bundle = {"krea2_sigmas_version": KREA2_SIGMAS_VERSION}
        for n in _STAGES:
            bundle[n] = {
                "enabled": bool(kwargs.get(f"{n}_enable", True)),
                "curve": str(kwargs.get(f"{n}_curve", _STAGE_DEFAULT_CURVE[n])),
                "bias": float(kwargs.get(f"{n}_detail_bias", 0.0)),
                "rho": float(kwargs.get(f"{n}_rho", 0.0)),
                "alpha": float(kwargs.get(f"{n}_alpha", 0.5)),
                "beta": float(kwargs.get(f"{n}_beta", 0.7)),
            }
        on = [n.upper() for n in _STAGES if bundle[n]["enabled"]]
        print(f"[EricKrea2-Sigmas] shape ready; overriding stages: {', '.join(on) or '(none)'}")
        return (bundle,)


NODE_CLASS_MAPPINGS = {"EricKrea2Sigmas": EricKrea2Sigmas}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2Sigmas": "Eric Krea2 Sigmas"}
