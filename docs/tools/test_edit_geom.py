"""Parity: _edit_geom vs krea2edit v1.2.5 (_fit_encode_image geometry + _imgids_offset).
Reference functions below are copied from lbouaraba/comfyui-krea2edit (Apache-2.0) with
vae.encode replaced by identity. Run: python test_edit_geom.py PACK_DIR"""

import importlib.util
import itertools
import os
import sys

import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location(
    "eg", os.path.join(sys.argv[1], "_edit_geom.py")
)
EG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EG)


def ref_fit(image, H, W, fit_mode):  # krea2edit _fit_encode_image (H, W = latent dims)
    px_h, px_w = H * 8, W * 8
    img = image.movedim(-1, 1)
    ih, iw = img.shape[-2:]
    if fit_mode == "fit":
        sc = min(px_h / ih, px_w / iw)
        CROP_TOL = 0.08
        if ih * sc >= px_h * (1 - CROP_TOL) and iw * sc >= px_w * (1 - CROP_TOL):
            s = max(px_h / ih, px_w / iw)
            ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
            y0, x0 = (ih - ch) // 2, (iw - cw) // 2
            img = img[..., y0 : y0 + ch, x0 : x0 + cw]
            nh, nw = px_h, px_w
        else:
            nh = min(max(16, int(ih * sc) // 16 * 16), max(16, px_h // 16 * 16))
            nw = min(max(16, int(iw * sc) // 16 * 16), max(16, px_w // 16 * 16))
            ch2, cw2 = (
                min(ih, max(1, int(round(nh / sc)))),
                min(iw, max(1, int(round(nw / sc)))),
            )
            y0, x0 = (ih - ch2) // 2, (iw - cw2) // 2
            img = img[..., y0 : y0 + ch2, x0 : x0 + cw2]
        img = F.interpolate(img.float(), size=(nh, nw), mode="bicubic", antialias=True)
        return img.movedim(1, -1)[..., :3].clamp(0, 1)
    s = max(px_h / ih, px_w / iw)
    ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
    y0, x0 = (ih - ch) // 2, (iw - cw) // 2
    img = img[..., y0 : y0 + ch, x0 : x0 + cw]
    img = F.interpolate(img.float(), size=(px_h, px_w), mode="bicubic", antialias=True)
    return img.movedim(1, -1)[..., :3].clamp(0, 1)


def ref_ids(frame, gh, gw, th, tw):  # krea2edit _imgids_offset (bs=1)
    off_h, off_w = max(0.0, (th - gh) / 2), max(0.0, (tw - gw) / 2)
    ids = torch.zeros(gh, gw, 3)
    ids[..., 0] = frame
    ids[..., 1] = (torch.arange(gh, dtype=torch.float32) + off_h)[:, None]
    ids[..., 2] = (torch.arange(gw, dtype=torch.float32) + off_w)[None, :]
    return ids.reshape(gh * gw, 3)


torch.manual_seed(0)
srcs = [
    (754, 1000),
    (753, 1000),
    (1000, 754),
    (1024, 1024),
    (768, 1365),
    (1536, 640),
    (900, 1200),
    (1200, 900),
    (517, 389),
    (2048, 1152),
    (640, 1536),
    (1100, 1000),
]
tgts = [
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1344, 768),
    (768, 1344),
    (1408, 1056),
    (1536, 1536),
]
n = bad = 0
for (ih, iw), (px_h, px_w), mode in itertools.product(srcs, tgts, ["fit", "crop"]):
    img = torch.rand(1, ih, iw, 3)
    ref = ref_fit(img, px_h // 8, px_w // 8, mode)
    ours, plan = EG.fit_source(img, px_w, px_h, mode)
    n += 1
    ok = ref.shape == ours.shape and torch.equal(ref, ours)
    if mode == "fit":
        gh, gw = ours.shape[1] // 16, ours.shape[2] // 16
        ok = ok and torch.equal(
            ref_ids(1, gh, gw, px_h // 16, px_w // 16), EG.ref_position_ids(plan, 1)
        )
    if not ok:
        bad += 1
        print(
            "MISMATCH",
            (ih, iw),
            (px_h, px_w),
            mode,
            tuple(ref.shape),
            tuple(ours.shape),
            plan["branch"],
        )
print(f"{n} cases, {bad} mismatches")
print("PASS" if bad == 0 else "FAIL")
