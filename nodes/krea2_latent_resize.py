# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 Latent Resize
========================
Resize a KREA2_LATENT in latent space (no VAE round-trip) using the same
norm-preserving bislerp interpolation the Multi-Stage Ultra node uses between
stages. Works for both up- and down-scaling. Handy for building custom
multi-stage graphs: size a Stage output to the next stage yourself, downscale an
encoded image before a refine pass, or match a target resolution exactly.

Three sizing modes:
  * scale       - linear multiplier on both dimensions (keeps aspect).
  * dimensions  - explicit width/height (wire the Eric Krea2 Resolution node's
                  width/height outputs here; 0 on one axis = derive from aspect).
  * megapixels  - target area in MP, keeping the source aspect ratio.

All results are rounded to multiples of 16 (the Krea2/Qwen f8 x2-patch grid).
"""

from __future__ import annotations

from .._latent_utils import _upscale_latents


def _round16(v: float) -> int:
    return max(16, int(round(v / 16)) * 16)


class EricKrea2LatentResize:
    """Latent-space (bislerp) resize of a KREA2_LATENT to a scale / explicit size /
    target megapixels."""

    CATEGORY = "Eric/Krea2"
    FUNCTION = "resize"
    RETURN_TYPES = ("KREA2_LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "width", "height", "dims")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("KREA2_LATENT",),
                "mode": (["scale", "dimensions", "megapixels"], {"default": "scale",
                    "tooltip": "scale: multiply both dims by 'scale'. dimensions: use width/height "
                               "(0 on one axis = derive from the source aspect). megapixels: target "
                               "area in MP, keeping the source aspect."}),
            },
            "optional": {
                "scale": ("FLOAT", {"default": 2.0, "min": 0.05, "max": 8.0, "step": 0.05,
                    "tooltip": "Linear scale factor for 'scale' mode (2.0 = 2x each side = 4x area). "
                               "Values < 1 downscale."}),
                "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16,
                    "tooltip": "Target width for 'dimensions' mode (0 = derive from height + source "
                               "aspect). Wire the Eric Krea2 Resolution node here. Rounded to /16."}),
                "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16,
                    "tooltip": "Target height for 'dimensions' mode (0 = derive from width + source "
                               "aspect). Rounded to /16."}),
                "megapixels": ("FLOAT", {"default": 3.00, "min": 0.05, "max": 16.0, "step": 0.05,
                    "tooltip": "Target area in megapixels for 'megapixels' mode (source aspect kept)."}),
            },
        }

    def resize(self, latent, mode="scale", scale=2.0, width=0, height=0, megapixels=3.00):
        src_w = int(latent.get("width", 0))
        src_h = int(latent.get("height", 0))
        vsf = int(latent.get("vae_scale_factor", 8))
        packed = latent["packed"]
        if src_w <= 0 or src_h <= 0:
            raise ValueError("KREA2_LATENT is missing valid width/height metadata; cannot resize.")

        if mode == "dimensions":
            w, h = int(width), int(height)
            aspect = src_w / src_h
            if w <= 0 and h <= 0:
                w, h = src_w, src_h
            elif w <= 0:
                w = int(round(h * aspect))
            elif h <= 0:
                h = int(round(w / aspect))
            dst_w, dst_h = _round16(w), _round16(h)
        elif mode == "megapixels":
            aspect = src_w / src_h
            target = max(float(megapixels), 0.05) * 1_000_000.0
            h = (target / aspect) ** 0.5
            w = h * aspect
            dst_w, dst_h = _round16(w), _round16(h)
        else:  # scale
            s = max(float(scale), 1e-3)
            dst_w, dst_h = _round16(src_w * s), _round16(src_h * s)

        if dst_w == src_w and dst_h == src_h:
            out_packed = packed
            print(f"[EricKrea2] Latent Resize: {src_w}x{src_h} unchanged (target matches source).")
        else:
            out_packed = _upscale_latents(packed, src_h, src_w, dst_h, dst_w, vsf)
            print(f"[EricKrea2] Latent Resize ({mode}, bislerp): {src_w}x{src_h} -> {dst_w}x{dst_h}")

        out = {"packed": out_packed, "height": dst_h, "width": dst_w, "vae_scale_factor": vsf}
        return (out, dst_w, dst_h, f"{dst_w}x{dst_h}")


NODE_CLASS_MAPPINGS = {"EricKrea2LatentResize": EricKrea2LatentResize}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2LatentResize": "Eric Krea2 Latent Resize"}
