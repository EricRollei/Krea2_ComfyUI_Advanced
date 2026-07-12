# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.

"""
Eric Krea2 VAE Encode
=====================
Turn a ComfyUI IMAGE into a KREA2_LATENT (packed latents + pixel dims) - the
inverse of the Decode nodes, and the enabling primitive for img2img / refine
workflows.

The produced KREA2_LATENT has the exact structure Generate / Multi-Stage emit
(``{"packed", "height", "width", "vae_scale_factor"}``) in the same normalized
latent space, so it can be wired straight into the upscale-decode node, fed back
through a Multi-Stage refine pass, or (once img2img init inputs land) used as the
starting point for a partial-denoise generation.

Encoding is deterministic (uses the VAE distribution mode, no sampling noise);
variation for img2img comes from the downstream denoise/strength control.
"""

from __future__ import annotations

from .._latent_utils import standard_encode

# The AutoencoderKLQwenImage used by Krea 2 is f8 (matches Generate/Decode).
_VAE_SCALE_FACTOR = 8


class EricKrea2VAEEncode:
    """Encode an IMAGE into a KREA2_LATENT using the pipeline's Qwen-Image VAE
    (or an optional Wan-family decode VAE, which shares the same latent space)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "krea2_pipeline": ("KREA2_PIPELINE",),
                "image": ("IMAGE",),
            },
            "optional": {
                "encode_vae": ("KREA2_DECODE_VAE", {
                    "tooltip": "Optional alternate VAE to encode with instead of the pipeline's "
                               "Qwen-Image VAE. Krea2/Qwen-Image and Wan 2.1 share an identical "
                               "latent space (same latents_mean/std), so a Wan-family VAE encodes "
                               "into the same space - use it to match the grain of an alternate "
                               "decode VAE."}),
            },
        }

    RETURN_TYPES = ("KREA2_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "Eric/Krea2"

    def encode(self, krea2_pipeline, image, encode_vae=None):
        pipe = krea2_pipeline["pipeline"]
        if pipe is None:
            # Same unloaded-pipeline scenario the Multi-Stage node guards against
            # (EricKrea2UnloadModels emptied the loader's cached dict). Reuse its
            # recovery so encode heals the dict instead of throwing a cryptic error.
            from .krea2_multistage_ultra import EricKrea2MultistageUltra
            pipe = EricKrea2MultistageUltra._recover_unloaded_pipeline(krea2_pipeline)

        packed, height, width = standard_encode(pipe, image, encode_vae=encode_vae)
        latent = {
            "packed": packed,
            "height": height,
            "width": width,
            "vae_scale_factor": _VAE_SCALE_FACTOR,
        }
        tag = "alt VAE" if encode_vae is not None else "pipeline VAE"
        print(f"[EricKrea2] VAE-encoded image ({tag}) -> KREA2_LATENT "
              f"(packed {tuple(packed.shape)}, {height}x{width} px)")
        return (latent,)


NODE_CLASS_MAPPINGS = {"EricKrea2VAEEncode": EricKrea2VAEEncode}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2VAEEncode": "Eric Krea2 VAE Encode"}
