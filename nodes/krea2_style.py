# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 Style Reference (training-free style transfer)
=========================================================
Bundles a style image with the style-transfer settings and per-stage multipliers.
Wire the KREA2_STYLE output into Multi-Stage Ultra's `style` socket. Nothing runs here;
Ultra installs the runtime (_style_ref.py) for one generation and removes it after.

Mechanism: nkxx188/ComfyUI-Krea2-StyleTransfer (MIT), ported - see _style_ref.py.
`recommended` locks the original author's tuned low-leakage preset (all custom widgets
are then ignored); `custom` uses the widgets below.

Cost: the model runs on [target, reference] every styled step (~2.3-2.6x per step) plus a
short reference-trajectory pass per styled stage. S2/S3 default to 0 (expensive at 8 MP).

Author: Eric Hiss (GitHub: EricRollei)
"""

from .._style_ref import PRESET_RECOMMENDED


class EricKrea2StyleReference:
    @classmethod
    def INPUT_TYPES(cls):
        R = PRESET_RECOMMENDED
        return {
            "required": {
                "style_image": ("IMAGE", {
                    "tooltip": "The style reference. Its palette / brushwork / texture / rendering "
                               "language are transferred; its content should not leak. It is cover-"
                               "cropped to each styled stage's size, so similar aspect ratios crop less."}),
            },
            "optional": {
                "preset": (["recommended", "custom"], {"default": "recommended",
                    "tooltip": "recommended: the original author's tuned low-leakage route (custom "
                               "widgets ignored). custom: use the widgets below."}),
                "style_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Overall style amount (applies in both modes). 0 = off."}),
                "s1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Stage 1 multiplier on style_strength. 0 = stock model for S1."}),
                "s2_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Stage 2 multiplier. Default 0: S2 refines S1's styled result at "
                               "full speed; raise to re-assert the style while refining (slow)."}),
                "s3_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Stage 3 multiplier (very slow at 8 MP)."}),
                "ref_k_strength": ("FLOAT", {"default": R["ref_k_strength"], "min": 0.0, "max": 5.0,
                    "step": 0.01, "tooltip": "custom: multiplier on the reference keys - how hard the "
                                             "target attends to the reference. Brings style back when "
                                             "low_scale_end is kept low."}),
                "low_scale_end": ("FLOAT", {"default": R["low_scale_end"], "min": -4.0, "max": 8.0,
                    "step": 0.01, "tooltip": "custom: end weight of the reference keys' low-frequency "
                                             "bands. Higher = more style, more content leakage / quality "
                                             "loss."}),
                "high_scale_start": ("FLOAT", {"default": R["high_scale_start"], "min": -4.0, "max": 8.0,
                    "step": 0.01, "tooltip": "custom: start weight of the high-frequency (positional) "
                                             "bands; they fade to high_scale_end over the run."}),
                "high_scale_end": ("FLOAT", {"default": R["high_scale_end"], "min": -4.0, "max": 8.0,
                    "step": 0.01, "tooltip": "custom: end weight of the high-frequency bands (0 = the "
                                             "target cannot copy reference positions late in the run)."}),
                "adain_strength": ("FLOAT", {"default": R["adain_strength"], "min": 0.0, "max": 1.0,
                    "step": 0.01, "tooltip": "custom: AdaIN of the target's Q/K toward the reference."}),
                "ref_value_mix": ("FLOAT", {"default": R["ref_value_mix"], "min": 0.0, "max": 1.0,
                    "step": 0.01, "tooltip": "custom: share of raw reference values (1 = raw)."}),
                "value_adain_strength": ("FLOAT", {"default": R["value_adain_strength"], "min": 0.0,
                    "max": 1.5, "step": 0.05, "tooltip": "custom: AdaIN of target values toward the "
                                                         "reference (only matters when ref_value_mix < 1)."}),
                "beta": ("FLOAT", {"default": R["beta"], "min": 0.01, "max": 20.0, "step": 0.01,
                    "tooltip": "custom: curve of the high->low frequency weighting."}),
                "blocks": ("STRING", {"default": R["blocks"],
                    "tooltip": "custom: transformer blocks that see the reference, e.g. 7-27 or 5,8,20-27."}),
                "trajectory": (["model_pc", "linear"], {"default": "model_pc",
                    "tooltip": "How the reference is noised to the target's level. model_pc: model-"
                               "guided (original route, costs ~2x trajectory_steps evals per styled "
                               "stage). linear: plain ref+noise mix, free but weaker."}),
                "gamma": ("FLOAT", {"default": R["gamma"], "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "model_pc blend: 1 = pure model trajectory, 0 = pure linear prior."}),
                "trajectory_steps": ("INT", {"default": 8, "min": 2, "max": 32,
                    "tooltip": "Points on the reference trajectory per styled stage."}),
            },
        }

    RETURN_TYPES = ("KREA2_STYLE", "IMAGE")
    RETURN_NAMES = ("style", "style_preview")
    FUNCTION = "build"
    CATEGORY = "Eric/Krea2"

    def build(self, style_image, preset="recommended", style_strength=1.0, s1_strength=1.0,
              s2_strength=0.0, s3_strength=0.0, ref_k_strength=1.06, low_scale_end=1.10,
              high_scale_start=1.04, high_scale_end=0.0, adain_strength=0.85, ref_value_mix=1.0,
              value_adain_strength=0.65, beta=2.5, blocks="7-27", trajectory="model_pc",
              gamma=0.5, trajectory_steps=8):
        if preset == "recommended":
            params = dict(PRESET_RECOMMENDED)
        else:
            params = {"ref_k_strength": ref_k_strength, "low_scale_end": low_scale_end,
                      "low_scale_start": 1.0, "high_scale_start": high_scale_start,
                      "high_scale_end": high_scale_end, "adain_strength": adain_strength,
                      "ref_value_mix": ref_value_mix, "value_adain_strength": value_adain_strength,
                      "beta": beta, "blocks": blocks, "trajectory": trajectory, "gamma": gamma}
        params["style_strength"] = float(style_strength)
        params["trajectory_steps"] = int(trajectory_steps)
        if preset == "recommended":
            params["trajectory"] = trajectory   # trajectory mode/steps stay user choices
        img = style_image[:1, :, :, :3].float().clamp(0, 1)
        bundle = {"image": img, "params": params, "s1": float(s1_strength),
                  "s2": float(s2_strength), "s3": float(s3_strength), "preset": preset}
        print(f"[EricKrea2-Style] bundle: {preset} | strength {style_strength} | S1/S2/S3 "
              f"x{s1_strength}/{s2_strength}/{s3_strength} | style image {img.shape[2]}x{img.shape[1]}")
        return (bundle, img)


NODE_CLASS_MAPPINGS = {"EricKrea2StyleReference": EricKrea2StyleReference}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2StyleReference": "Eric Krea2 Style Reference (training-free)"}
