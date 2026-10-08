# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Multi-Stage Ultra glue for the `edit` / `guidance` sockets (2026-10-08).
prepare_edit_guidance() encodes everything ONCE before sampling (grounded instruction,
grounded empty negative, refine prompt, NAG negative, NegPiP weights, fitted VAE sources)
and returns an installed EditGuidanceRuntime - or None when nothing is active.
Mutates Ultra kwargs: prompt_conditioning, width/height (edit size_from=source),
crop_bottom (edit: 0 - a guard band would shift the source registration).

Author: Eric Hiss (GitHub: EricRollei)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_LOG = "[EricKrea2-EditGuide]"


def _round16(v):
    return max(16, int(round(float(v) / 16.0)) * 16)


def _boost_mask_tokens(mask, plan):
    """MASK [H, W] in source px -> bool [gh*gw] on the fitted source grid (crop first)."""
    y0, x0, ch, cw = plan["crop"]
    m = mask[y0:y0 + ch, x0:x0 + cw].float()
    gh, gw = plan["grid"]
    m = F.interpolate(m[None, None], size=(gh, gw), mode="area")[0, 0]
    return (m > 0.5).reshape(-1)


@torch.no_grad()
def prepare_edit_guidance(pipe, kwargs, edit, guide, resolve_s1_dims, log=print):
    from ._edit_text import grounded_encode, negpip_encode, plain_encode
    from ._edit_geom import fit_source, ref_position_ids
    from ._latent_utils import standard_encode
    from ._edit_guidance import EditGuidanceRuntime

    device = getattr(pipe, "_execution_device", None) or "cuda"
    g = dict(guide) if isinstance(guide, dict) else {}
    use_negpip = bool(g.get("negpip"))
    npm = g.get("negpip_mode", "lifted") if use_negpip else None
    nps = float(g.get("negpip_strength", 1.0))

    if g.get("nag"):
        neg_txt = str(g.get("nag_negative", "")).strip()
        if not neg_txt:
            log(f"{_LOG} NAG is on but nag_negative is empty - NAG disabled")
            g["nag"] = False
        else:
            g["neg_cond"] = plain_encode(pipe, neg_txt, device)
            log(f"{_LOG} NAG negative encoded: {neg_txt[:120]!r}")

    vmul_by_len = {}
    e = None
    pc_connected = isinstance(kwargs.get("prompt_conditioning"), dict) and \
        kwargs["prompt_conditioning"].get("embeds") is not None

    if isinstance(edit, dict) and edit.get("images"):
        e = dict(edit)
        if int(kwargs.get("crop_bottom", 0) or 0) > 0:
            log(f"{_LOG} crop_bottom disabled for this edit run (keeps the source registration)")
            kwargs["crop_bottom"] = 0
        explicit = int(kwargs.get("width", 0) or 0) > 0 and int(kwargs.get("height", 0) or 0) > 0
        from_init = kwargs.get("init_latent") is not None and kwargs.get("init_match_size", True)
        if e.get("size_from", "source") == "source" and not explicit and not from_init:
            img = e["images"][0]
            ih, iw = int(img.shape[1]), int(img.shape[2])
            mp = float(kwargs.get("s1_megapixels", 3.0))
            h = (mp * 1_000_000.0 * ih / iw) ** 0.5
            kwargs["width"], kwargs["height"] = _round16(h * iw / ih), _round16(h)
            log(f"{_LOG} Stage 1 sized to the source aspect ({iw}x{ih}) at {mp:g} MP -> "
                f"{kwargs['width']}x{kwargs['height']}")
        s1_w, s1_h = resolve_s1_dims(
            kwargs.get("aspect_ratio", "5:4 landscape"), kwargs.get("s1_megapixels", 3.0),
            kwargs.get("width", 0), kwargs.get("height", 0), kwargs.get("init_latent"),
            kwargs.get("init_match_size", True), verbose=False)
        if s1_w * s1_h > 2.2e6:
            log(f"{_LOG} WARNING: Stage 1 is {s1_w * s1_h / 1e6:.2f} MP - the edit LoRA is trained "
                "<= 2 MP (above it sources bleed / subjects duplicate). Lower s1_megapixels and let "
                "S2/S3 upscale.")
        refs = []
        for i, im in enumerate(e["images"]):
            fitted, plan = fit_source(im, s1_w, s1_h, e["fit_mode"])
            packed, eh, ew = standard_encode(pipe, fitted)
            r = {"packed": packed.detach().to(device), "pos": ref_position_ids(plan, i + 1),
                 "plan": plan}
            if i == len(e["images"]) - 1 and e.get("ref_boost_mask") is not None:
                r["boost_mask"] = _boost_mask_tokens(e["ref_boost_mask"], plan)
            refs.append(r)
            log(f"{_LOG} source {i + 1}: {im.shape[2]}x{im.shape[1]} -> {plan['branch']} "
                f"{plan['size'][1]}x{plan['size'][0]} ({plan['grid'][1]}x{plan['grid'][0]} tokens) "
                f"at offset {plan['offset'][1]:g},{plan['offset'][0]:g} on the "
                f"{s1_w // 16}x{s1_h // 16} target grid")
        e["refs"] = refs
        e["target_grid"] = (s1_h // 16, s1_w // 16)
        proc = e.get("vision_processor_source")
        pos = grounded_encode(pipe, e["instruction"], e["images"], device, e["grounding_px"],
                              e.get("system_prompt", ""), proc, negpip_mode=npm,
                              negpip_strength=nps, log=log)
        if pos["vmul"] is not None:
            vmul_by_len[int(pos["embeds"].shape[1])] = pos["vmul"]
        if e.get("ground_negative", True):
            e["neg_cond"] = grounded_encode(pipe, "", e["images"], device, e["grounding_px"],
                                            e.get("system_prompt", ""), proc, log=log)
        ptxt = str(kwargs.get("prompt", "") or "").strip()
        if ptxt:
            rc = negpip_encode(pipe, ptxt, device, npm, nps, log) if use_negpip else \
                plain_encode(pipe, ptxt, device)
            e["refine_cond"] = rc
            if rc["vmul"] is not None:
                vmul_by_len.setdefault(int(rc["embeds"].shape[1]), rc["vmul"])
            log(f"{_LOG} Stage 2/3 refine with Ultra's prompt text; Stage 1 uses the grounded "
                "instruction")
        else:
            log(f"{_LOG} Ultra prompt empty - all stages use the grounded instruction")
        if pc_connected:
            log(f"{_LOG} NOTE: prompt_conditioning input is ignored while an edit is connected")
        kwargs["prompt_conditioning"] = {"embeds": pos["embeds"], "mask": pos["mask"]}
    elif use_negpip:
        if pc_connected:
            log(f"{_LOG} NegPiP: prompt_conditioning is precomputed (Vision Prompt) - its text "
                "can't be parsed here; NegPiP inactive. Put (phrase:-1) in Ultra's prompt instead.")
        else:
            pos = negpip_encode(pipe, str(kwargs.get("prompt", "") or ""), device, npm, nps, log)
            kwargs["prompt_conditioning"] = {"embeds": pos["embeds"], "mask": pos["mask"]}
            if pos["vmul"] is not None:
                vmul_by_len[int(pos["embeds"].shape[1])] = pos["vmul"]
    g["vmul_by_len"] = vmul_by_len
    if use_negpip and not vmul_by_len:
        log(f"{_LOG} NegPiP on but no (phrase:-w) groups found - NegPiP inactive")
        g["negpip"] = False
    if not (e or g.get("nag") or g.get("negpip")):
        return None
    return EditGuidanceRuntime(pipe, edit=e, guide=g)   # Ultra installs it inside its try
