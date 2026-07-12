# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
#
# Krea 2 shares the AutoencoderKLQwenImage latent space with Qwen-Image and
# Wan2.1, so spacepxl/Wan2.1-VAE-upscale2x (a decoder-only finetune that emits
# 12 channels -> pixel_shuffle(2) -> 2x image) works directly on Krea 2 latents.
# This module is self-contained: loader node + decode helpers + cosine tile blend.
#
#  Upscale model: spacepxl/Wan2.1-VAE-upscale2x (Apache-2.0)
#    https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x
#  Independent implementation (no code copied from spacepxl repos).

from __future__ import annotations

import math
import types

import torch
import torch.nn.functional as F

from ._latent_utils import _unpack_latents, _pack_latents


# ═══════════════════════════════════════════════════════════════════════
#  Cosine tile-blend patch (C1-smooth tile seams)
# ═══════════════════════════════════════════════════════════════════════

def patch_cosine_blend_vae(vae) -> None:
    """Replace the VAE's linear tile-blend with C1-smooth cosine blending.

    When ``vae.enable_tiling()`` is active the decoder stitches overlapping
    tiles with ``blend_v``/``blend_h``; the default linear interpolation leaves
    a slope discontinuity (faint grid lines on smooth gradients). Cosine
    interpolation has zero derivative at both endpoints, so seams disappear.
    Idempotent; no effect when tiling is disabled.
    """
    if vae is None or getattr(vae, "_cosine_blend_patched", False):
        return

    def _cosine_blend_v(self, a, b, blend_extent):
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            alpha = (1.0 - math.cos(math.pi * y / blend_extent)) / 2.0
            b[..., y, :] = a[..., -blend_extent + y, :] * (1 - alpha) + b[..., y, :] * alpha
        return b

    def _cosine_blend_h(self, a, b, blend_extent):
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            alpha = (1.0 - math.cos(math.pi * x / blend_extent)) / 2.0
            b[..., x] = a[..., -blend_extent + x] * (1 - alpha) + b[..., x] * alpha
        return b

    vae.blend_v = types.MethodType(_cosine_blend_v, vae)
    vae.blend_h = types.MethodType(_cosine_blend_h, vae)
    vae._cosine_blend_patched = True


# ═══════════════════════════════════════════════════════════════════════
#  Upscale-VAE loader node
# ═══════════════════════════════════════════════════════════════════════

class EricKrea2UpscaleVAELoader:
    """Load the Wan2.1 2x upscale VAE (decoder-only finetune).

    Works on Krea 2 / Qwen-Image / Wan2.1 latents (shared latent space).
    Kept on CPU until decode is requested.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "spacepxl/Wan2.1-VAE-upscale2x",
                    "tooltip": "HuggingFace model ID or local path.",
                }),
            },
            "optional": {
                "subfolder": ("STRING", {
                    "default": "diffusers/Wan2.1_VAE_upscale2x_imageonly_real_v1",
                    "tooltip": "Subfolder containing config.json + weights. Blank if model_path is already the dir.",
                }),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "vae_tile_blend": (["cosine", "linear"], {
                    "default": "cosine",
                    "tooltip": "Tile-seam blending when tiling is active. cosine = C1-smooth (no grid lines).",
                }),
                "downsample_to_1x": ("BOOLEAN", {"default": False,
                    "tooltip": "Decode at 2x (this VAE's native output) then Lanczos-downsample back to the "
                               "generation's native 1x size, instead of returning the full 2x image. Trades the "
                               "literal resolution gain for this decoder's sharper/GAN-trained detail at your "
                               "original output size - a supersampling-style quality pass, not an upscale. "
                               "Applies to the FINAL decode only (upscale_vae_mode='final_decode'/'both' on "
                               "Ultra, or the standalone Upscale Decode node) - never to inter-stage upscaling, "
                               "which exists specifically to gain resolution between stages."}),
                "blur_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Gaussian blur sigma applied at 2x, BEFORE the Lanczos downsample (standard "
                               "supersampling order: blur to anti-alias, then resize). 0 = off - Lanczos' own "
                               "resampling filter already does reasonable anti-aliasing on its own, so this is "
                               "an extra taste knob, not a required step. Only used when downsample_to_1x is on."}),
            },
        }

    RETURN_TYPES = ("UPSCALE_VAE",)
    RETURN_NAMES = ("upscale_vae",)
    FUNCTION = "load_vae"
    CATEGORY = "Eric/Krea2"

    def load_vae(self, model_path, subfolder="diffusers/Wan2.1_VAE_upscale2x_imageonly_real_v1",
                 dtype="bfloat16", vae_tile_blend="cosine", downsample_to_1x=False, blur_sigma=0.0):
        from diffusers import AutoencoderKLWan
        dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(dtype, torch.bfloat16)
        kwargs = {"torch_dtype": dt}
        if subfolder and subfolder.strip():
            kwargs["subfolder"] = subfolder.strip()
        print(f"[EricKrea2] Loading upscale VAE from {model_path} ...")
        vae = AutoencoderKLWan.from_pretrained(model_path, **kwargs)
        vae.eval()
        vae._vae_tile_blend = vae_tile_blend
        # Stashed on the VAE object (same pattern as _vae_tile_blend above) so this travels
        # with the VAE into every consumer - Ultra's own internal final_decode call AND the
        # standalone Upscale Decode node - without needing matching widgets duplicated on each.
        vae._downsample_to_1x = bool(downsample_to_1x)
        vae._blur_sigma = float(blur_sigma)
        print(f"[EricKrea2] Upscale VAE loaded (out_channels={vae.config.out_channels}).")
        return (vae,)


# ═══════════════════════════════════════════════════════════════════════
#  Blur + Lanczos downsample (supersampling-style quality pass)
# ═══════════════════════════════════════════════════════════════════════

def _gaussian_blur_chw(image, sigma):
    """image: [B,3,H,W] float in [0,1]. Fails soft (returns image unchanged) if
    torchvision isn't available, rather than breaking the whole decode over an
    optional taste knob."""
    if not sigma or sigma <= 0:
        return image
    try:
        from torchvision.transforms.functional import gaussian_blur
        k = max(3, int(2 * round(3 * sigma) + 1))
        return gaussian_blur(image, kernel_size=[k, k], sigma=[float(sigma), float(sigma)])
    except Exception as e:
        print(f"[EricKrea2] blur_sigma={sigma} requested but torchvision.transforms.functional "
              f"unavailable ({e}); skipping blur, decode continues.")
        return image


def _lanczos_downsample_chw(image, target_h, target_w):
    """image: [B,3,H,W] float in [0,1] (CPU) -> [B,3,target_h,target_w] via true PIL LANCZOS.
    Per-image (PIL has no batch dim); fine at this stage since it's a one-shot final decode,
    not a per-step operation."""
    from PIL import Image as PILImage
    import numpy as np
    out = []
    for i in range(image.shape[0]):
        arr = (image[i].permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
        pil = PILImage.fromarray(arr, mode="RGB").resize((int(target_w), int(target_h)), PILImage.LANCZOS)
        out.append(torch.from_numpy(np.array(pil)).float() / 255.0)
    return torch.stack(out, dim=0).permute(0, 3, 1, 2)  # [B,3,target_h,target_w]


# ═══════════════════════════════════════════════════════════════════════
#  Decode helper: packed latents -> 2x image via upscale VAE
# ═══════════════════════════════════════════════════════════════════════

def decode_latents_with_upscale_vae(packed_latents, upscale_vae, pipe_vae,
                                    height, width, vae_scale_factor=8, downsample_override=None):
    """Decode packed Krea2/Qwen latents with the 2x upscale VAE -> [B, 2H, 2W, 3], unless
    downsampling is requested, in which case -> [B, H, W, 3] (blur optional, Lanczos
    downsample back to the generation's native size - a quality pass, not an upscale;
    see EricKrea2UpscaleVAELoader's tooltip). Downsampling is enabled by EITHER the loaded
    VAE's own downsample_to_1x flag OR an explicit ``downsample_override`` (True/False);
    the override wins when not None (lets Ultra's upscale_vae_mode request it per-run).
    This is the single shared decode path used by BOTH Ultra's own internal decode modes
    and the standalone Upscale Decode node - the downsample/blur options live on the VAE
    object precisely so both inherit them for free.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = next(upscale_vae.parameters()).dtype
    upscale_vae = upscale_vae.to(device)

    h_lat = 2 * (int(height) // (vae_scale_factor * 2))
    w_lat = 2 * (int(width) // (vae_scale_factor * 2))
    if h_lat > 128 or w_lat > 128:
        try:
            upscale_vae.enable_tiling()
        except Exception:
            pass
        if getattr(upscale_vae, "_vae_tile_blend", "cosine") == "cosine":
            patch_cosine_blend_vae(upscale_vae)
            print(f"[EricKrea2] Tiled VAE decode + cosine blend (latent {h_lat}x{w_lat})")
        else:
            print(f"[EricKrea2] Tiled VAE decode + linear blend (latent {h_lat}x{w_lat})")
    else:
        upscale_vae.use_tiling = False

    try:
        spatial = _unpack_latents(packed_latents, height, width, vae_scale_factor)
        spatial = spatial.to(device=device, dtype=dtype)

        z_dim = pipe_vae.config.z_dim
        latents_mean = torch.tensor(pipe_vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(device, dtype)
        latents_std = 1.0 / torch.tensor(pipe_vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(device, dtype)
        spatial = spatial / latents_std + latents_mean

        with torch.no_grad():
            decoded = upscale_vae.decode(spatial, return_dict=False)[0]  # [B, 12, 1, H, W]
        decoded = decoded.squeeze(2)                                     # [B, 12, H, W]
        image = F.pixel_shuffle(decoded, upscale_factor=2)              # [B, 3, 2H, 2W]
        image = torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)

        if getattr(upscale_vae, "_downsample_to_1x", False) if downsample_override is None \
                else bool(downsample_override):
            image = image.cpu()  # blur/resize happen on CPU either way (PIL for Lanczos)
            blur_sigma = getattr(upscale_vae, "_blur_sigma", 0.0)
            if blur_sigma and blur_sigma > 0:
                image = _gaussian_blur_chw(image, blur_sigma)
            image = _lanczos_downsample_chw(image, height, width)
            print(f"[EricKrea2] Upscale-VAE decode: 2x -> Lanczos downsample to {width}x{height} "
                  f"(blur_sigma={blur_sigma})")
            return image.permute(0, 2, 3, 1).float()

        return image.permute(0, 2, 3, 1).cpu().float()
    finally:
        upscale_vae.to("cpu")
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
#  Inter-stage helper: decode 2x via upscale VAE, re-encode to latents
# ═══════════════════════════════════════════════════════════════════════

def upscale_between_stages(packed_latents, upscale_vae, pipe_vae,
                           height, width, vae_scale_factor=8, target_h=None, target_w=None):
    """Decode at 2x via the upscale VAE, re-encode with the standard VAE.

    Returns ``(packed_latents, new_h, new_w)`` at 2x pixel resolution, latents in
    pipeline-normalized packed format ready for the next stage. If ``target_h``/
    ``target_w`` are given and smaller than the 2x decode, the 2x pixels are
    resampled DOWN to that target (bicubic + antialias) before re-encoding - this
    keeps the upscale VAE's learned detail but at a caller-chosen size instead of a
    forced 4x area (used by the 's2-s3 with downsample' modes).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    up_dtype = next(upscale_vae.parameters()).dtype
    vae_dtype = next(pipe_vae.parameters()).dtype
    z_dim = pipe_vae.config.z_dim

    mean_t = torch.tensor(pipe_vae.config.latents_mean).view(1, z_dim, 1, 1, 1)
    std_t = torch.tensor(pipe_vae.config.latents_std).view(1, z_dim, 1, 1, 1)

    upscale_vae = upscale_vae.to(device)
    if getattr(upscale_vae, "_vae_tile_blend", "cosine") == "cosine":
        patch_cosine_blend_vae(upscale_vae)
    try:
        spatial = _unpack_latents(packed_latents, height, width, vae_scale_factor)
        spatial = spatial.to(device=device, dtype=up_dtype)
        spatial = spatial * std_t.to(device, up_dtype) + mean_t.to(device, up_dtype)
        with torch.no_grad():
            decoded = upscale_vae.decode(spatial, return_dict=False)[0]
        decoded = decoded.squeeze(2)
        pixels_2x = F.pixel_shuffle(decoded, upscale_factor=2)  # [B, 3, 2H, 2W], [-1, 1]
        del decoded, spatial
        if target_h and target_w and (int(target_h) < pixels_2x.shape[2]
                                      or int(target_w) < pixels_2x.shape[3]):
            # Free the upscale VAE weights off the GPU BEFORE the resample: the
            # antialias bicubic needs a float32 copy of the (2x-resolution) pixel
            # tensor, which momentarily doubles it - keeping the VAE weights resident
            # at the same time is a needless OOM risk on large S2->S3 jumps.
            upscale_vae.to("cpu")
            torch.cuda.empty_cache()
            _pd = pixels_2x.dtype
            pixels_2x = F.interpolate(pixels_2x.float(), size=(int(target_h), int(target_w)),
                                      mode="bicubic", align_corners=False, antialias=True).to(_pd)
    finally:
        upscale_vae.to("cpu")
        torch.cuda.empty_cache()

    new_h, new_w = pixels_2x.shape[2], pixels_2x.shape[3]

    pipe_vae = pipe_vae.to(device)
    pixels_5d = pixels_2x.unsqueeze(2).to(dtype=vae_dtype)
    del pixels_2x
    with torch.no_grad():
        posterior = pipe_vae.encode(pixels_5d).latent_dist
        raw_latents = posterior.mode()
    del pixels_5d
    norm_latents = (raw_latents - mean_t.to(device, vae_dtype)) / std_t.to(device, vae_dtype)
    new_packed = _pack_latents(norm_latents)

    print(f"[EricKrea2] Inter-stage VAE upscale: {height}x{width} -> {new_h}x{new_w}")
    return new_packed, new_h, new_w


NODE_CLASS_MAPPINGS = {"EricKrea2UpscaleVAELoader": EricKrea2UpscaleVAELoader}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2UpscaleVAELoader": "Eric Krea2 Upscale VAE Loader"}
