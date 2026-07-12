# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 Resolution
=====================
Pick a generation size once and reuse it. Computes width/height from the same
aspect-ratio + megapixels math the Multi-Stage Ultra node uses (so the numbers
match), and optionally derives the size from a source IMAGE - the enabling piece
for img2img, where you want Stage 1 sized to the input picture.

Outputs:
  * width / height (INT)  - wire into the Ultra node's width/height inputs to
                            override its aspect_ratio + megapixels sizing.
  * dims (STRING "WxH")   - for filenames / metadata / display.
  * latent (KREA2_LATENT) - a blank canvas at that size (zeros or unit-variance
                            noise) for img2img inits or the latent-resize node.

All dimensions are rounded to multiples of 16 (the Krea2/Qwen f8 x2-patch grid).
"""

from __future__ import annotations

import torch

from .._latent_utils import _pack_latents

# Krea2/Qwen-Image latent space: f8 VAE, 16 latent channels, 2x2 patch pack.
_VAE_SCALE_FACTOR = 8
_LATENT_CHANNELS = 16

# Aspect ratios (widest -> tallest) - mirrors the Multi-Stage Ultra node.
_ASPECT_RATIOS = {
    "21:9 ultrawide": (21, 9),
    "16:9 wide": (16, 9),
    "3:2 landscape": (3, 2),
    "4:3 landscape": (4, 3),
    "5:4 landscape": (5, 4),
    "1:1 square": (1, 1),
    "4:5 portrait": (4, 5),
    "3:4 portrait": (3, 4),
    "2:3 portrait": (2, 3),
    "9:16 tall": (9, 16),
    "9:21 tall": (9, 21),
}


def _round16(v: float) -> int:
    return max(16, int(round(v / 16)) * 16)


def _dims_from_ratio_mp(ratio_key: str, megapixels: float):
    w_r, h_r = _ASPECT_RATIOS.get(ratio_key, (1, 1))
    target = max(megapixels, 0.05) * 1_000_000.0
    aspect = w_r / h_r
    h = (target / aspect) ** 0.5
    w = h * aspect
    return _round16(w), _round16(h)


class EricKrea2Resolution:
    """Resolution / empty-latent helper: aspect+megapixels (or a source image) ->
    width / height / dims / a blank KREA2_LATENT canvas."""

    CATEGORY = "Eric/Krea2"
    FUNCTION = "compute"
    RETURN_TYPES = ("INT", "INT", "STRING", "KREA2_LATENT")
    RETURN_NAMES = ("width", "height", "dims", "latent")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (list(_ASPECT_RATIOS.keys()), {"default": "5:4 landscape"}),
                "megapixels": ("FLOAT", {"default": 3.00, "min": 0.05, "max": 16.0, "step": 0.05,
                    "tooltip": "Target size in megapixels; width/height are derived from the aspect "
                               "ratio. Matches the Ultra node's s1_megapixels math."}),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional source image (img2img): when connected, width/height are taken "
                               "from THIS image (rounded to /16) and the aspect_ratio + megapixels are "
                               "ignored. Wire Load Image -> here -> the Ultra node's width/height inputs "
                               "to size Stage 1 to your source picture."}),
                "swap_orientation": ("BOOLEAN", {"default": False, "label_on": "swapped", "label_off": "normal",
                    "tooltip": "Swap width and height (e.g. turn a landscape ratio into portrait) without "
                               "adding a mirrored entry to the ratio list."}),
                "fill": (["noise", "zeros"], {"default": "noise",
                    "tooltip": "Contents of the output KREA2_LATENT canvas. noise = unit-variance white "
                               "noise (a ready init for a full-denoise img2img/refine pass); zeros = an "
                               "empty canvas. The Ultra node re-noises any init at its s1_start_step, so "
                               "for start_step 0 the two are equivalent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Seed for the noise fill (reproducible). Ignored when fill=zeros."}),
            },
        }

    def compute(self, aspect_ratio, megapixels, image=None, swap_orientation=False,
                fill="noise", seed=0):
        if image is not None:
            # ComfyUI IMAGE is (B, H, W, C); derive the source size.
            h_px = _round16(int(image.shape[1]))
            w_px = _round16(int(image.shape[2]))
            src = f"image {int(image.shape[2])}x{int(image.shape[1])}"
        else:
            w_px, h_px = _dims_from_ratio_mp(aspect_ratio, float(megapixels))
            src = f"{aspect_ratio} @ {float(megapixels):.2f}MP"

        if swap_orientation:
            w_px, h_px = h_px, w_px

        # Build a blank packed canvas (CPU; the Ultra node moves it to device).
        h_lat = 2 * (h_px // (_VAE_SCALE_FACTOR * 2))
        w_lat = 2 * (w_px // (_VAE_SCALE_FACTOR * 2))
        if fill == "zeros":
            spatial = torch.zeros(1, _LATENT_CHANNELS, 1, h_lat, w_lat, dtype=torch.float32)
        else:
            g = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
            spatial = torch.randn(1, _LATENT_CHANNELS, 1, h_lat, w_lat,
                                  generator=g, dtype=torch.float32)
        packed = _pack_latents(spatial)
        latent = {"packed": packed, "height": h_px, "width": w_px,
                  "vae_scale_factor": _VAE_SCALE_FACTOR}

        dims = f"{w_px}x{h_px}"
        print(f"[EricKrea2] Resolution: {src} -> {dims} ({fill} latent, packed {tuple(packed.shape)})")
        return (w_px, h_px, dims, latent)


NODE_CLASS_MAPPINGS = {"EricKrea2Resolution": EricKrea2Resolution}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2Resolution": "Eric Krea2 Resolution"}
