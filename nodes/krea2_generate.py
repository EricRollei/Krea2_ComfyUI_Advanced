# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.

"""
Eric Krea2 Generate
==================
Text-to-image with a loaded Krea 2 pipeline. Emits both a decoded IMAGE and a
KREA2_LATENT (packed latents + pixel dims) so the result can be chained into the
2x upscale-decode node or a multi-stage node without regenerating.

With ``auto_settings`` on, picks the recommended sampler for the checkpoint:
distilled/Turbo -> 8 steps, guidance 0.0; base/Raw -> 28 steps, guidance 4.5.
Krea uses natural-language prompts; long, detailed visual/camera language works
best, and words to be rendered as text should be wrapped in quotes.
"""

from __future__ import annotations

import torch

from .._latent_utils import standard_decode

# The AutoencoderKLQwenImage used by Krea 2 is f8.
_VAE_SCALE_FACTOR = 8


class EricKrea2Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "krea2_pipeline": ("KREA2_PIPELINE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "auto_settings": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use the checkpoint's recommended steps/guidance "
                               "(Turbo: 8/0.0, Raw: 28/4.5). Off = use the manual values below.",
                }),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8}),
            },
        }

    RETURN_TYPES = ("IMAGE", "KREA2_LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "generate"
    CATEGORY = "Eric/Krea2"

    def generate(self, krea2_pipeline, prompt, negative_prompt, width, height, seed,
                 auto_settings=True, steps=28, guidance_scale=4.5, num_images=1):
        pipe = krea2_pipeline["pipeline"]
        is_distilled = krea2_pipeline.get("is_distilled", False)

        if auto_settings:
            steps, guidance_scale = (8, 0.0) if is_distilled else (28, 4.5)
            print(f"[EricKrea2] auto_settings: {'Turbo' if is_distilled else 'Raw'} "
                  f"-> steps={steps}, guidance={guidance_scale}")
        elif is_distilled and guidance_scale > 0:
            print(f"[EricKrea2] Turbo/distilled checkpoint: forcing guidance 0 "
                  f"(was {guidance_scale}). CFG is off-distribution for distilled models.")
            guidance_scale = 0.0

        # round to multiple of 16 (pipeline does this too; we mirror it for the latent dims)
        width = (int(width) // 16) * 16
        height = (int(height) // 16) * 16

        device = getattr(pipe, "_execution_device", None) or "cuda"
        generator = torch.Generator(device=device).manual_seed(int(seed))

        result = pipe(
            prompt=prompt,
            negative_prompt=(negative_prompt or None),
            height=height,
            width=width,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            num_images_per_prompt=int(num_images),
            generator=generator,
            output_type="latent",
        )
        packed = result.images  # [B, seq, 64] packed latents

        image = standard_decode(pipe, packed, height, width)
        latent = {
            "packed": packed.detach(),
            "height": height,
            "width": width,
            "vae_scale_factor": _VAE_SCALE_FACTOR,
        }
        return (image, latent)


NODE_CLASS_MAPPINGS = {"EricKrea2Generate": EricKrea2Generate}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2Generate": "Eric Krea2 Generate"}
