# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 Hybrid Sigma Handoff
===============================
Optional side panel for Multi-Stage Ultra / Ultra V2 (2026-10-23). Wire its `sampling`
output into Ultra's optional `sampling` input. Holds the sampler extras that would
otherwise crowd the Ultra panel:

  * sN_hybrid_handoff - for lcm_hybrid / lcm_hybrid2: switch from lcm to the tail
    sampler at this SIGMA instead of after a fixed step count, so one value means the
    same thing on every schedule, step count and shift. 0 = use sN_hybrid_steps.
  * sN_sde_hi / sN_sde_lo - SDE band: eta's churn only on steps whose next sigma is
    inside [lo, hi].
  * churn_form - how eta's noise is injected (ve classic / rf rectified-flow).

Unwired, Ultra behaves exactly as before. Sweep cells that set any of these keys
override the node for that cell.
"""

from __future__ import annotations

from .. import _sampling_extras as _sx

_BAND_TIP = ("SDE band {edge} (sigma) for stage {n}. eta's churn is applied only on steps whose NEXT "
             "sigma lies inside [s{n}_sde_lo, s{n}_sde_hi]; outside it the sampler runs deterministic. "
             "1.0 / 0.0 = whole window. Rough Krea2 guide: sigma 1.0-0.5 = composition, 0.5-0.15 = form "
             "and detail, below = micro texture. Does nothing at eta 0, and for euler / lcm / er_sde; on "
             "the lcm hybrids it gates the tail sampler only. The console shows how many steps churned.")

_HANDOFF_TIP = ("Stage {n} lcm-hybrid handoff, in sigma. The lcm head runs until the trajectory reaches "
                "this sigma; the tail (deis_3m for lcm_hybrid, res_2m for lcm_hybrid2) takes over from the "
                "first step at or below it. Same value = same switch point on every schedule, step count "
                "and shift - unlike s{n}_hybrid_steps, which it replaces when > 0. 0 = off (use "
                "s{n}_hybrid_steps). {extra}Higher values switch earlier (longer tail): more fine detail and "
                "texture, at the possible expense of composition and correct drawing. Lower values give "
                "more lcm steps: cleaner, more correct structure, softer detail. Rough guide on 12 linear "
                "steps: ~0.78 detailed, 0.65 balanced, 0.45-0.55 clean - the right point depends on the "
                "subject (architecture / glass / hands tend to want lower, textured or atmospheric scenes "
                "can take higher). The console prints the resulting step split.")


class EricKrea2HybridHandoff:
    @classmethod
    def INPUT_TYPES(cls):
        req = {}
        for n, (h_def, extra) in ((1, (0.65, "0.65 = 8 of 12 lcm steps on linear / simple (0.70 = 7). ")),
                                  (2, (0.0, "Refine windows usually start at sigma 0.5-0.76 - a handoff "
                                            "above the window start gives 1 lcm step. ")),
                                  (3, (0.0, "See s2. "))):
            req[f"s{n}_hybrid_handoff"] = ("FLOAT", {"default": h_def, "min": 0.0, "max": 1.0, "step": 0.01,
                                                     "tooltip": _HANDOFF_TIP.format(n=n, extra=extra)})
            req[f"s{n}_sde_hi"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                             "tooltip": _BAND_TIP.format(edge="TOP", n=n)})
            req[f"s{n}_sde_lo"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                             "tooltip": _BAND_TIP.format(edge="BOTTOM", n=n)})
        req["churn_form"] = (["rf", "ve"], {"default": "rf",
            "tooltip": "How eta's per-step noise is injected (all stages). rf: rectified-flow split "
                       "(as ComfyUI euler_ancestral_RF) - keeps the latent's signal level matched to the "
                       "sigma the model is told. ve: the classic k-diffusion split (Ultra's behaviour "
                       "before 2026-10-25) - over-weights signal on flow models (~2% at eta 0.3, ~20% at "
                       "eta 1), which shows as burnt contrast at high eta. Default rf: in the 2026-10-23 "
                       "res_2m sweep rf never clipped (all cells <1%) while ve reached 13-60% at eta "
                       "0.7-1.0. eta numbers are not interchangeable between the two."})
        return {"required": req}

    RETURN_TYPES = (_sx.BUNDLE_TYPE,)
    RETURN_NAMES = ("sampling",)
    FUNCTION = "build"
    CATEGORY = "Eric/Krea2"

    def build(self, **kwargs):
        bundle = {k: kwargs[k] for k in _sx.KEYS if k in kwargs}
        return (bundle,)


NODE_CLASS_MAPPINGS = {"EricKrea2HybridHandoff": EricKrea2HybridHandoff}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2HybridHandoff": "Eric Krea2 Hybrid Sigma Handoff"}
