# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 Edit + Eric Krea2 Negative Guidance
==============================================
Bundle builders for Multi-Stage Ultra's `edit` and `guidance` sockets. Nothing runs here;
Ultra encodes and installs _edit_guidance.EditGuidanceRuntime for one generation.

Edit = the Krea 2 Identity Edit recipe (krea2edit v1.2.5 nodes, Apache-2.0, by lbouaraba):
source image(s) as clean in-context tokens + image-grounded instruction encode.
Put the edit LoRA (krea2_identity_edit_v1_2) on the Multi-LoRA stack, S1 = 1.0.
Turbo 8-12 steps, guidance off for most edits; removals: Raw ~20 steps, s1_cfg ~2 (CFG 3)
with ground_negative on. Keep Stage 1 <= 2 MP.

Negative Guidance = NAG (positive-only negative prompting, works at guidance 0 / Turbo)
and NegPiP ((phrase:-1) in the prompt subtracts that concept). Both default OFF and can
be combined.

Author: Eric Hiss (GitHub: EricRollei)
"""

import torch
import torch.nn.functional as F

_VL_DEFAULT = r"H:\Testing\Qwen3-VL-4B-Instruct-heretic-7refusal"


class EricKrea2Edit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE", {"tooltip": "Image to edit (the scene / main reference, "
                                                      "RoPE frame 1)."}),
                "instruction": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Edit instruction, e.g. 'recolor the car to matte black'. NegPiP "
                               "(phrase:-1) works here when the Negative Guidance node has NegPiP on."}),
            },
            "optional": {
                "source_image_b": ("IMAGE", {"tooltip": "Second reference (the subject, RoPE frame "
                                                        "2) for person-into-scene edits. Place two "
                                                        "people in ONE pass rather than chaining."}),
                "fit_mode": (["fit", "crop (legacy)"], {"default": "fit",
                    "tooltip": "fit = v1.2 training geometry (resample to the target grid at a "
                               "centered offset; any aspect ratio). crop (legacy) = center-crop to "
                               "the output aspect - only for v1/v1.1 weights."}),
                "ref_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.05,
                    "tooltip": "Reference-fidelity dial on the LAST reference (the subject in two-ref "
                               "edits, else the source): multiplies target->reference attention. "
                               "1 = off, ~4 strong likeness (model card), >10 breaks removals, "
                               "<1 loosens."}),
                "ref_boost_a": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.05,
                    "tooltip": "Same dial for the FIRST reference (the scene) in two-ref edits. "
                               "No effect with one reference."}),
                "ref_boost_mask": ("MASK", {"tooltip": "Optional region of the LAST reference to "
                                                       "boost (e.g. the face), in that image's pixels."}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 2048, "step": 32,
                    "tooltip": "Longest side of the image(s) Qwen3-VL sees. Trained 384-768. Lower = "
                               "stronger instruction adherence, higher = stronger likeness. Lower it "
                               "if subjects duplicate. 0 = native size."}),
                "size_from": (["source", "ultra settings"], {"default": "source",
                    "tooltip": "source: Stage 1 takes the source aspect at Ultra's s1_megapixels "
                               "(width/height on Ultra still win if set). ultra settings: Ultra's "
                               "aspect_ratio decides; the fit geometry handles the mismatch."}),
                "ground_negative": ("BOOLEAN", {"default": True,
                    "tooltip": "With guidance (CFG) on: the negative is the EMPTY instruction "
                               "grounded on the same image(s) - the trained unconditional. Off = "
                               "Ultra's negative_prompt as plain text."}),
                "system_prompt": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Advanced: override the grounding system prompt (empty = Krea 2 / "
                               "training default)."}),
                "vision_processor_source": ("STRING", {"default": _VL_DEFAULT,
                    "tooltip": "Folder with the Qwen3-VL image-processor config (same as the Vision "
                               "Prompt node)."}),
            },
        }

    RETURN_TYPES = ("KREA2_EDIT", "IMAGE")
    RETURN_NAMES = ("edit", "grounding_preview")
    FUNCTION = "build"
    CATEGORY = "Eric/Krea2"
    DESCRIPTION = ("Instruction / identity edit (Krea 2 Identity Edit recipe) for Multi-Stage "
                   "Ultra's `edit` socket.")

    def build(self, source_image, instruction, source_image_b=None, fit_mode="fit",
              ref_boost=1.0, ref_boost_a=1.0, ref_boost_mask=None, grounding_px=768,
              size_from="source", ground_negative=True, system_prompt="",
              vision_processor_source=_VL_DEFAULT):
        images = [source_image[:1]] + ([source_image_b[:1]] if source_image_b is not None else [])
        mask = None
        if ref_boost_mask is not None:
            mask = ref_boost_mask[0] if ref_boost_mask.dim() == 3 else ref_boost_mask
            mask = mask.float().cpu()
        bundle = {
            "images": [im.float().cpu() for im in images],
            "instruction": str(instruction or ""),
            "fit_mode": "fit" if str(fit_mode).startswith("fit") else "crop",
            "ref_boost": float(ref_boost), "ref_boost_a": float(ref_boost_a),
            "ref_boost_mask": mask,
            "grounding_px": int(grounding_px), "size_from": str(size_from),
            "ground_negative": bool(ground_negative), "system_prompt": str(system_prompt or ""),
            "vision_processor_source": str(vision_processor_source or _VL_DEFAULT),
        }
        # preview = what the VLM sees (first image, grounding size)
        from .._edit_text import grounding_pils
        import numpy as np
        pil = grounding_pils([images[0]], grounding_px)[0]
        prev = torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0)[None]
        print(f"[EricKrea2-Edit] {len(images)} reference(s), fit={bundle['fit_mode']}, ref_boost "
              f"{ref_boost:g}/{ref_boost_a:g}{' (masked)' if mask is not None else ''}, grounding "
              f"{grounding_px}px, instruction: {bundle['instruction'][:120]!r}")
        return (bundle, prev)


class EricKrea2NegativeGuidance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nag": ("BOOLEAN", {"default": False,
                    "tooltip": "Normalized Attention Guidance: steer AWAY from nag_negative inside "
                               "attention - works at guidance 0 (Turbo), single pass + small overhead."}),
                "nag_negative": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "What to suppress, e.g. 'blurry, extra fingers, watermark'. Separate "
                               "from Ultra's negative_prompt (that one is the CFG negative)."}),
                "nag_phi": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Extrapolation strength (z+ + phi (z+ - z-))."}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Norm clamp: guided features may grow at most tau x the positive."}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Blend of the guided features into the positive ones."}),
                "nag_stages": (["s1", "s1_s2", "all"], {"default": "s1"}),
                "nag_sigma_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Apply NAG only while sigma is in [end, start]."}),
                "nag_sigma_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "negpip": ("BOOLEAN", {"default": False,
                    "tooltip": "NegPiP: (phrase:-1.0) in the prompt / edit instruction subtracts that "
                               "concept (its attention values are flipped)."}),
                "negpip_mode": (["lifted", "in_place"], {"default": "in_place",
                    "tooltip": "in_place (default, cleaner in tests): the phrase's tokens are flagged "
                               "where they stand. lifted: the phrase is removed from the prompt and "
                               "encoded on its own (the text encoder never reads it in context)."}),
                "negpip_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Multiplies every negative weight."}),
                "negpip_block_start": ("INT", {"default": 0, "min": 0, "max": 27}),
                "negpip_block_end": ("INT", {"default": 27, "min": 0, "max": 27}),
                "negpip_text_fusion": ("BOOLEAN", {"default": False,
                    "tooltip": "Also flip inside the text-fusion refiner attention (stronger)."}),
            },
        }

    RETURN_TYPES = ("KREA2_GUIDANCE",)
    RETURN_NAMES = ("guidance",)
    FUNCTION = "build"
    CATEGORY = "Eric/Krea2"
    DESCRIPTION = "NAG + NegPiP negative guidance for Multi-Stage Ultra's `guidance` socket."

    def build(self, nag, nag_negative, nag_phi, nag_tau, nag_alpha, nag_stages, nag_sigma_start,
              nag_sigma_end, negpip, negpip_mode, negpip_strength, negpip_block_start,
              negpip_block_end, negpip_text_fusion):
        st = {"s1": {1}, "s1_s2": {1, 2}, "all": {1, 2, 3}}[nag_stages]
        return ({"nag": bool(nag), "nag_negative": str(nag_negative or ""), "phi": float(nag_phi),
                 "tau": float(nag_tau), "alpha": float(nag_alpha), "nag_stages": st,
                 "sigma_start": float(nag_sigma_start), "sigma_end": float(nag_sigma_end),
                 "negpip": bool(negpip), "negpip_mode": str(negpip_mode),
                 "negpip_strength": float(negpip_strength),
                 "block_start": int(negpip_block_start), "block_end": int(negpip_block_end),
                 "negpip_text_fusion": bool(negpip_text_fusion)},)


NODE_CLASS_MAPPINGS = {"EricKrea2Edit": EricKrea2Edit,
                       "EricKrea2NegativeGuidance": EricKrea2NegativeGuidance}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2Edit": "Eric Krea2 Edit",
                              "EricKrea2NegativeGuidance": "Eric Krea2 Negative Guidance"}
