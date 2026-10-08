# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 edit geometry - krea2edit v1.2.5 "fit" / "crop (legacy)" reference placement
=====================================================================================
Pixel-space source preparation for identity/instruction edit LoRAs (Krea 2 Identity Edit
v1.2) and where the source tokens sit on the target RoPE grid. Must be byte-identical to
the geometry the LoRA was trained on - a different ref size moves the centered offset and
shows up as a seam / doubling band (krea2edit v1.2.4 RCA).

Ported from lbouaraba/comfyui-krea2edit `_fit_encode_image` + `_imgids_offset`
(v1.2.5, Apache-2.0) and the trainer's `_fit_prep` (lbouaraba/krea2edit-trainer,
Apache-2.0). Token = 16 px (VAE /8 x patch 2).

fit:
  * near-matched AR (fit-inside covers >= 92% of both target sides): minimal center-crop
    to the exact target AR, resize to the target px -> ref grid == target grid.
  * genuine AR mismatch: fit-inside, size floor-snapped to /16 and capped at the target's
    /16 floor; the source is first center-cropped so the fitted axis lands on the /16 grid
    with ZERO squash ("crop-to-grid"), then resized. Tokens are placed stride-1 at a
    FRACTIONAL centered offset ((th-gh)/2, (tw-gw)/2) on the target grid.
crop (legacy, v1/v1.1 weights): center-crop to the target AR, resize to the target px,
  offset 0.
Resampling: bicubic + antialias (the inference node; the trainer used bilinear+antialias
on the same geometry - the node is the inference reference).

Author: Eric Hiss (GitHub: EricRollei)
Geometry credit: lbouaraba (conradlocke) - krea2edit / krea2edit-trainer, Apache-2.0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

CROP_TOL = 0.08
TOKEN_PX = 16


def plan_fit(ih, iw, px_h, px_w, mode="fit"):
    """Pure geometry (no tensors). ih, iw: source px; px_h, px_w: target px (multiples
    of 16). Returns dict(crop=(y0, x0, ch, cw), size=(nh, nw), grid=(gh, gw),
    offset=(off_h, off_w) in tokens, branch=str)."""
    ih, iw, px_h, px_w = int(ih), int(iw), int(px_h), int(px_w)
    th, tw = px_h // TOKEN_PX, px_w // TOKEN_PX
    if mode == "fit":
        sc = min(px_h / ih, px_w / iw)
        if ih * sc >= px_h * (1 - CROP_TOL) and iw * sc >= px_w * (1 - CROP_TOL):
            s = max(px_h / ih, px_w / iw)
            ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
            y0, x0 = (ih - ch) // 2, (iw - cw) // 2
            nh, nw, branch = px_h, px_w, "fit-fill"
        else:
            nh = min(max(16, int(ih * sc) // 16 * 16), max(16, px_h // 16 * 16))
            nw = min(max(16, int(iw * sc) // 16 * 16), max(16, px_w // 16 * 16))
            ch, cw = min(ih, max(1, int(round(nh / sc)))), min(iw, max(1, int(round(nw / sc))))
            y0, x0 = (ih - ch) // 2, (iw - cw) // 2
            branch = "fit-inside"
        gh, gw = nh // TOKEN_PX, nw // TOKEN_PX
        off = (max(0.0, (th - gh) / 2), max(0.0, (tw - gw) / 2))
    else:  # crop (legacy)
        s = max(px_h / ih, px_w / iw)
        ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
        y0, x0 = (ih - ch) // 2, (iw - cw) // 2
        nh, nw, branch = px_h, px_w, "crop"
        gh, gw = th, tw
        off = (0.0, 0.0)
    return {"crop": (y0, x0, ch, cw), "size": (nh, nw), "grid": (gh, gw),
            "offset": off, "branch": branch, "target_grid": (th, tw)}


def fit_source(image, px_w, px_h, mode="fit"):
    """image: ComfyUI IMAGE [B, H, W, C] 0..1 (frame 0 used). Returns (fitted [1, nh, nw, 3]
    float32 0..1, plan)."""
    img = image[:1, :, :, :3].float()
    ih, iw = int(img.shape[1]), int(img.shape[2])
    p = plan_fit(ih, iw, px_h, px_w, mode)
    y0, x0, ch, cw = p["crop"]
    x = img.permute(0, 3, 1, 2)[..., y0:y0 + ch, x0:x0 + cw]
    nh, nw = p["size"]
    if (ch, cw) != (nh, nw):
        x = F.interpolate(x, size=(nh, nw), mode="bicubic", antialias=True)
    return x.clamp(0, 1).permute(0, 2, 3, 1).contiguous(), p


def ref_position_ids(plan, frame, device=None):
    """[gh*gw, 3] float rotary rows for a fitted source: axis0 = frame (1..N), h/w stride-1
    at the plan's fractional centered offset on the target grid."""
    gh, gw = plan["grid"]
    oh, ow = plan["offset"]
    ids = torch.zeros(gh, gw, 3, device=device, dtype=torch.float32)
    ids[..., 0] = float(frame)
    ids[..., 1] = (torch.arange(gh, device=device, dtype=torch.float32) + oh)[:, None]
    ids[..., 2] = (torch.arange(gw, device=device, dtype=torch.float32) + ow)[None, :]
    return ids.reshape(gh * gw, 3)


def placement_preview(fitted, plan, px_w, px_h):
    """Visual check: the fitted source drawn where its tokens sit on the target canvas
    (grey outside). Returns IMAGE [1, px_h, px_w, 3]."""
    canvas = torch.full((1, int(px_h), int(px_w), 3), 0.5)
    oh, ow = plan["offset"]
    y, x = int(round(oh * TOKEN_PX)), int(round(ow * TOKEN_PX))
    nh, nw = fitted.shape[1], fitted.shape[2]
    canvas[:, y:y + nh, x:x + nw] = fitted[:, :px_h - y, :px_w - x]
    return canvas
