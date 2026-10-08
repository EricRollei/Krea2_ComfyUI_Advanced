# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md - EXCEPT the filter core marked below.
#
# The notch-filter core (extract_grid, lattice_amp, auto_limit, the degrid math)
# is ported from lunaaispace-eng/ComfyUI-DeGrid (degrid_core.py), licensed under
# the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
# Changes: adapted to BCHW tensors in either [0,1] or [-1,1] range, no zoom/vis
# framing helpers, stats trimmed to what the Krea2 nodes report.
"""
Eric Krea2 DeGrid - VAE 2-pixel lattice removal
===============================================
The Qwen-Image VAE (Krea2's VAE) leaves a fixed-phase 2px pixel grid in decoded
images, strongest in flat/dark areas. The spacepxl 2x upscale VAE builds its
output with pixel_shuffle(2) - also a 2x2-cell structure. When the inter-stage
VAE hop decodes, upsamples and RE-ENCODES, that lattice is fed back into the next
stage's latent, which then "develops" it as texture - a plausible source of the
faint weave / lines that appeared only after S2/S3.

Filter: separable 9-tap alternating binomial -> response sin^8(w/2): exact zero at
any 2px-period pattern (checker or stripes), unity at DC with an 8th-order flat
zero (no banding). The correction is amplitude-LIMITED so real edges pass, and an
image with no phase-locked lattice is passed through untouched.

Author: Eric Hiss (GitHub: EricRollei); filter core (c) lunaaispace-eng, Apache-2.0
"""

import torch
import torch.nn.functional as F

# ---- ported core (Apache-2.0, lunaaispace-eng/ComfyUI-DeGrid) ----------------

_KERNEL = [1.0, -8.0, 28.0, -56.0, 70.0, -56.0, 28.0, -8.0, 1.0]
_NORM = 256.0
_PAD = 4
NEGLIGIBLE_AMP = 0.5 / 255.0   # raw Qwen-VAE grid sits at 1-5/255; below = clean


def extract_grid(x):
    """2px-grid component (Bx + By - Bxy) of x [B,C,H,W] in 0..1 units."""
    b, c, h, w = x.shape
    if h <= 2 * _PAD or w <= 2 * _PAD:
        return torch.zeros_like(x)
    k = torch.tensor(_KERNEL, dtype=x.dtype, device=x.device) / _NORM
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    bx = F.conv2d(F.pad(x, (_PAD, _PAD, 0, 0), mode="reflect"), kx, groups=c)
    by = F.conv2d(F.pad(x, (0, 0, _PAD, _PAD), mode="reflect"), ky, groups=c)
    bxy = F.conv2d(F.pad(bx, (0, 0, _PAD, _PAD), mode="reflect"), ky, groups=c)
    return bx + by - bxy


def lattice_amp(corr):
    """Phase-locked lattice amplitude: means of the four (y%2, x%2) sublattices.
    Incoherent detail averages away over millions of pixels; the grid does not.
    Returns peak-to-peak [B] (max over channels), 0..1 units."""
    h = corr.shape[2] // 2 * 2
    w = corr.shape[3] // 2 * 2
    if h < 2 or w < 2:
        return torch.zeros(corr.shape[0], dtype=corr.dtype, device=corr.device)
    c = corr[:, :, :h, :w]
    m = torch.stack([c[:, :, i::2, j::2].mean(dim=(2, 3)) for i in (0, 1) for j in (0, 1)], dim=-1)
    return (m.amax(-1) - m.amin(-1)).amax(-1)


def _subsample(flat, max_samples=1_000_000):
    n = flat.shape[-1]
    return flat[..., :: n // max_samples + 1] if n > max_samples else flat


def auto_limit(corr, floor=0.004, ceil=0.05, mult=3.0):
    """Per-image clamp limit from the 75th percentile of |corr| (smooth regions
    dominate, so that approximates the artifact amplitude; edges are outliers)."""
    flat = _subsample(corr.abs().reshape(corr.shape[0], -1))
    q = torch.quantile(flat.float(), 0.75, dim=1)
    return (q * mult).clamp(floor, ceil).to(corr.dtype)


# ---- Krea2 wrappers ------------------------------------------------------------

def degrid_bchw(x01, limit="auto", skip_when_clean=True):
    """Remove the 2px lattice from x01 [B,C,H,W] float in 0..1.
    Returns (cleaned [same shape, clamped 0..1], stats list[dict])."""
    x = x01.float()
    corr = extract_grid(x)
    amp = lattice_amp(corr)
    lim = auto_limit(corr) if limit == "auto" else torch.full(
        (x.shape[0],), float(limit), dtype=corr.dtype, device=corr.device)
    lim_b = lim.view(-1, 1, 1, 1)
    clipped = (corr.abs() > lim_b).float().mean(dim=(1, 2, 3)) * 100.0
    corr = corr.clamp(-lim_b, lim_b)
    skipped = amp < NEGLIGIBLE_AMP
    if skip_when_clean:
        corr = torch.where(skipped.view(-1, 1, 1, 1), torch.zeros_like(corr), corr)
    cleaned = (x - corr).clamp(0.0, 1.0).to(x01.dtype)
    stats = [{"amp_255": float(amp[i]) * 255.0, "limit": float(lim[i]),
              "edges_protected_pct": float(clipped[i]),
              "skipped": bool(skipped[i]) and skip_when_clean}
             for i in range(x.shape[0])]
    return cleaned, stats


def degrid_signed_bchw(x_pm1, **kw):
    """Same as degrid_bchw for a [-1,1] tensor (the VAE pixel domain)."""
    cleaned, stats = degrid_bchw((x_pm1.float() + 1.0) * 0.5, **kw)
    return (cleaned * 2.0 - 1.0).to(x_pm1.dtype), stats


def degrid_image(image_bhwc, **kw):
    """ComfyUI IMAGE [B,H,W,C] 0..1 -> (cleaned IMAGE, stats)."""
    x = image_bhwc.permute(0, 3, 1, 2).contiguous()
    cleaned, stats = degrid_bchw(x, **kw)
    return cleaned.permute(0, 2, 3, 1).contiguous(), stats


def format_stats(stats, tag=""):
    parts = []
    for i, s in enumerate(stats):
        if s["skipped"]:
            parts.append(f"img{i}: grid {s['amp_255']:.2f}/255 - none detected, untouched")
        else:
            parts.append(f"img{i}: grid {s['amp_255']:.2f}/255 removed (limit {s['limit']:.3f}, "
                         f"edges protected {s['edges_protected_pct']:.1f}%)")
    return f"[EricKrea2-DeGrid]{(' ' + tag) if tag else ''} " + " | ".join(parts)
