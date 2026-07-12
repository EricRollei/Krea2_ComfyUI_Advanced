# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
#
# Self-contained latent helpers for Krea 2. Krea 2 and Qwen-Image share the
# AutoencoderKLQwenImage latent space and use identical pack/unpack logic, so
# these are the same routines proven in the Qwen UltraGen pipeline, copied here
# so Eric_Krea2 has no dependency on any other node pack.

from __future__ import annotations

import torch


def _unpack_latents(latents: torch.Tensor, height: int, width: int,
                    vae_scale_factor: int) -> torch.Tensor:
    """Unpack flow-packed latents back to spatial (B, C, 1, H_lat, W_lat)."""
    batch_size, num_patches, channels = latents.shape
    h_lat = 2 * (int(height) // (vae_scale_factor * 2))
    w_lat = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, h_lat // 2, w_lat // 2,
                           channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // 4, 1, h_lat, w_lat)
    return latents


def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack spatial latents (B, C, 1, H, W) back to flow format (B, seq, C*4)."""
    batch_size, c, _one, h, w = latents.shape
    latents = latents.squeeze(2)                        # (B, C, H, W)
    latents = latents.view(batch_size, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)        # (B, H/2, W/2, C, 2, 2)
    latents = latents.reshape(batch_size, (h // 2) * (w // 2), c * 4)
    return latents


def _upscale_latents(latents_packed: torch.Tensor,
                     src_h: int, src_w: int,
                     dst_h: int, dst_w: int,
                     vae_scale_factor: int) -> torch.Tensor:
    """Unpack -> bislerp upscale -> repack latents.

    Bislerp (slerp interpolation) preserves vector norms and angular
    relationships between latent channels, giving sharper, more coherent
    upscaled latents than bicubic's independent-channel averaging.
    """
    import comfy.utils as comfy_utils
    spatial = _unpack_latents(latents_packed, src_h, src_w, vae_scale_factor)
    dst_h_lat = 2 * (int(dst_h) // (vae_scale_factor * 2))
    dst_w_lat = 2 * (int(dst_w) // (vae_scale_factor * 2))
    spatial_4d = spatial.squeeze(2)  # (B, C, H, W)
    upscaled = comfy_utils.bislerp(spatial_4d, dst_w_lat, dst_h_lat)
    upscaled = upscaled.unsqueeze(2)  # (B, C, 1, H', W')
    return _pack_latents(upscaled)


def _add_noise_flowmatch(latents: torch.Tensor, noise: torch.Tensor,
                         sigma: float) -> torch.Tensor:
    """Flow-matching noising: x_noisy = (1 - sigma) * x + sigma * noise."""
    return (1.0 - sigma) * latents + sigma * noise


def _check_cancelled():
    """Raise InterruptProcessingException if the user hit Cancel in ComfyUI."""
    import comfy.model_management
    comfy.model_management.throw_exception_if_processing_interrupted()


def standard_decode(pipe, packed: torch.Tensor, height: int, width: int,
                    decode_vae=None) -> torch.Tensor:
    """Decode packed Krea2 latents to a ComfyUI IMAGE at 1x.

    Replicates the Krea2Pipeline decode path (unpack -> latent de-normalization
    -> vae.decode -> drop frame dim) and returns a [B, H, W, 3] float tensor in
    [0, 1]. Used so a Generate node can emit both an image and a chainable latent
    from a single ``output_type="latent"`` run.

    decode_vae: optional alternate VAE (e.g. base Wan 2.1) to decode with instead
    of the pipeline's Qwen-Image VAE. Valid because both share the same latent
    space (identical latents_mean/std), so the same de-normalized latent feeds
    either decoder - this swaps the *grain/texture* of the reconstruction
    (Wan = natural grain vs Qwen = smoother 'plastic' skin) without an upscale.
    """
    vae = decode_vae if decode_vae is not None else pipe.vae
    z = vae.config.z_dim
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype

    # Use the pipeline's own unpack so vae_scale_factor / patch_size always match.
    lat = pipe._unpack_latents(packed.to(device), height, width).to(dtype)

    lm = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(device, dtype)
    ls = 1.0 / torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(device, dtype)
    lat = lat / ls + lm

    # Tile large decodes. A single non-tiled decode of a big latent through the
    # Wan-family VAE leaves a boundary artifact (a noisy band at the bottom edge);
    # tiling with overlap + cosine seam-blending matches ComfyUI's behaviour and
    # removes it. Small images decode in one shot.
    tiled = (lat.shape[-2] > 128 or lat.shape[-1] > 128) and hasattr(vae, "enable_tiling")
    if tiled:
        try:
            vae.enable_tiling()
            try:
                from ._upscale_vae import patch_cosine_blend_vae
                patch_cosine_blend_vae(vae)
            except Exception:
                pass
            print(f"[EricKrea2] standard_decode: tiled decode engaged (latent "
                  f"{lat.shape[-2]}x{lat.shape[-1]}, vae={type(vae).__name__})")
        except Exception as e:
            tiled = False
            print(f"[EricKrea2] standard_decode: tiling requested but enable_tiling() failed "
                  f"({e}); falling back to a single non-tiled decode - this is the exact "
                  f"condition the bottom-edge-band comment above warns about.")
    else:
        print(f"[EricKrea2] standard_decode: non-tiled decode (latent {lat.shape[-2]}x{lat.shape[-1]}, "
              f"vae={type(vae).__name__}, has_enable_tiling={hasattr(vae, 'enable_tiling')})")
    try:
        with torch.no_grad():
            img = vae.decode(lat, return_dict=False)[0][:, :, 0]  # (B, 3, H, W) in [-1, 1]
    finally:
        if tiled and hasattr(vae, "disable_tiling"):
            try:
                vae.disable_tiling()
            except Exception:
                pass
    img = ((img + 1.0) / 2.0).clamp(0.0, 1.0)
    return img.permute(0, 2, 3, 1).float().cpu()


def standard_encode(pipe, image: torch.Tensor, encode_vae=None):
    """Encode a ComfyUI IMAGE into packed Krea2 latents (the inverse of standard_decode).

    Pixels ``[B, H, W, 3]`` in ``[0, 1]`` -> ``vae.encode`` -> latent-space
    normalization -> flow-pack, producing a packed tensor ``[B, seq, z*4]`` in the
    SAME normalized space that Generate/Multi-Stage emit (so it can drive img2img
    or a refine pass without regenerating from scratch).

    Mirrors ``standard_decode`` exactly:
      decode does   raw = norm * std + mean   ->   vae.decode
      so encode does raw = vae.encode(...)     ->   norm = (raw - mean) / std

    The deterministic distribution mode (mean) is used - no VAE sampling noise -
    so the init latent is clean and reproducible; variation comes from the
    denoise/strength lever on the downstream generate, not from the encode.

    encode_vae: optional alternate VAE (e.g. a Wan 2.1 KREA2_DECODE_VAE). Valid
    because Krea2/Qwen-Image and Wan 2.1 share identical latents_mean/std, so its
    output lands in the same normalized latent space.

    Returns ``(packed, height, width)`` where height/width are the input pixel
    dims rounded down to a multiple of 16 (the pack requires even latent dims;
    Generate rounds identically). The image is bilinearly resized to those dims
    if it isn't already a clean multiple of 16.
    """
    vae = encode_vae if encode_vae is not None else pipe.vae
    z = vae.config.z_dim
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype

    if image.dim() != 4 or image.shape[-1] != 3:
        raise ValueError(
            f"[EricKrea2] standard_encode expects an IMAGE [B, H, W, 3]; got {tuple(image.shape)}")
    b, h, w, _ = image.shape
    tgt_h = (int(h) // 16) * 16
    tgt_w = (int(w) // 16) * 16
    if tgt_h < 16 or tgt_w < 16:
        raise ValueError(
            f"[EricKrea2] image too small to encode ({h}x{w}px); need at least 16px per side.")

    px = image.permute(0, 3, 1, 2).to(device=device, dtype=dtype)   # (B, 3, H, W) in [0, 1]
    if (tgt_h, tgt_w) != (int(h), int(w)):
        px = torch.nn.functional.interpolate(
            px, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
        print(f"[EricKrea2] standard_encode: resized {h}x{w} -> {tgt_h}x{tgt_w} (multiple of 16)")
    px = px * 2.0 - 1.0                     # (B, 3, H, W) in [-1, 1]
    px = px.unsqueeze(2)                    # (B, 3, 1, H, W) - Qwen/Wan video-VAE frame dim

    # Tile large encodes, symmetric with standard_decode's >128-latent rule, to avoid
    # the same boundary artifacts a single big non-tiled pass can leave.
    lat_h, lat_w = tgt_h // 8, tgt_w // 8
    tiled = (lat_h > 128 or lat_w > 128) and hasattr(vae, "enable_tiling")
    if tiled:
        try:
            vae.enable_tiling()
            print(f"[EricKrea2] standard_encode: tiled encode engaged "
                  f"(latent {lat_h}x{lat_w}, vae={type(vae).__name__})")
        except Exception as e:
            tiled = False
            print(f"[EricKrea2] standard_encode: tiling requested but enable_tiling() failed ({e}); "
                  f"falling back to a single non-tiled encode.")
    try:
        with torch.no_grad():
            enc = vae.encode(px)
            dist = getattr(enc, "latent_dist", None)
            if dist is not None:
                raw = dist.mode()                          # (B, z, 1, H_lat, W_lat), un-normalized
            else:
                raw = enc[0] if isinstance(enc, (tuple, list)) else enc
    finally:
        if tiled and hasattr(vae, "disable_tiling"):
            try:
                vae.disable_tiling()
            except Exception:
                pass

    lm = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(device, dtype)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(device, dtype)
    norm = (raw - lm) * inv_std            # normalized z-latent (matches Generate's packed space)

    packed = _pack_latents(norm)           # (B, seq, z*4)
    return packed.detach(), tgt_h, tgt_w
