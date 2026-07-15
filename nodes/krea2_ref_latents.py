# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 Reference Latents
============================
VAE-encode 1-3 reference images into packed Krea2 latents for the
"index_timestep_zero" edit pathway (see ``_ref_latents.py`` for the mechanism
and provenance). Wire the output into the Multi-Stage Ultra node's optional
``ref_latents`` input.

Only useful with an edit-trained LoRA loaded (ai-toolkit ``krea2`` arch with
``model_kwargs.edit: true`` - e.g. the Civitai "Krea 2 Style Reference LoRA");
the base model was never trained to read reference tokens.

Recommended wiring for the Style Reference LoRA (matches its training setup):
  * this node's image(s) -> ref_latents            (pixel/structure detail, <=1 MP)
  * the SAME image(s) -> EricKrea2VisionPrompt with vision_megapixels ~= 0.15
    (ai-toolkit trains the VLM view at 384x384 total px) -> prompt_conditioning
  * the LoRA in the Multi-LoRA stack at strength 1.0
  * Turbo checkpoint, guidance 0 (with CFG on, refs condition both passes and
    partially cancel - see _ref_latents.py).

Author: Eric Hiss (GitHub: EricRollei)
"""

from __future__ import annotations


class EricKrea2RefLatents:
    CATEGORY = "Eric/Krea2"
    FUNCTION = "encode"
    RETURN_TYPES = ("KREA2_REF_LATENTS",)
    RETURN_NAMES = ("ref_latents",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "krea2_pipeline": ("KREA2_PIPELINE",),
                "image1": ("IMAGE", {"tooltip": "Reference image 1 (VAE-encoded, appended to the "
                                                "image token sequence at t=0)."}),
            },
            "optional": {
                "image2": ("IMAGE", {"tooltip": "Reference image 2 (optional)."}),
                "image3": ("IMAGE", {"tooltip": "Reference image 3 (optional; published edit LoRAs "
                                                "are mostly trained for 1-2 refs)."}),
                "max_ref_megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05,
                    "tooltip": "Per-reference pixel cap before VAE encode (downscale only, never "
                               "upscale). ai-toolkit edit training uses 1.0 MP - keep the default "
                               "unless VRAM or token count forces it down. Each 1 MP ref adds "
                               "~2048 tokens to every denoise step, so 2-3 refs at 1 MP is slow."}),
            },
        }

    def encode(self, krea2_pipeline, image1, image2=None, image3=None, max_ref_megapixels=1.0):
        pipe = krea2_pipeline["pipeline"]
        if pipe is None:
            from .krea2_multistage_ultra import EricKrea2MultistageUltra
            pipe = EricKrea2MultistageUltra._recover_unloaded_pipeline(krea2_pipeline)
        from .._ref_latents import prepare_ref_bundle
        images = [im for im in (image1, image2, image3) if im is not None]
        bundle = prepare_ref_bundle(pipe, images, max_megapixels=float(max_ref_megapixels))
        return (bundle,)


NODE_CLASS_MAPPINGS = {"EricKrea2RefLatents": EricKrea2RefLatents}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2RefLatents": "Eric Krea2 Reference Latents (Edit)"}
