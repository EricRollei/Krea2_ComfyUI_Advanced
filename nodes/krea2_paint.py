# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 Paint (inpaint / outpaint) + Paint Composite
=======================================================
Builds the KREA2_PAINT bundle for Multi-Stage Ultra's `paint` socket. Nothing runs
here; Ultra installs _paint.PaintRuntime for one generation and removes it after.

Adapter LoRA: put it on the Multi-LoRA stack (S1 = 1, S2/S3 = 0):
  AnyPaint  - yijunwang2/krea2-anypaint  (arbitrary masks, inpaint + outpaint, one pass)
  Outpaint  - yijunwang2/krea2-outpaint  (rectangular outpaint only, older)
Both: Krea 2 TURBO, 8 steps, guidance 0. `none` = training-free (restore only).
For best adherence wire `vlm_image` into Eric Krea2 Vision Prompt image1 (picture_n) -
the adapter was run with the condition image visible to Qwen3-VL.

Author: Eric Hiss (GitHub: EricRollei)
"""

import torch
import torch.nn.functional as F

from .._paint import median_rgb, resize_max_edge, resize_mask, round16

_RESTORE = ["auto", "on", "structure", "off"]


class EricKrea2Paint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Source image. Kept regions are pinned to it "
                                               "at full resolution."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "1 = generate / edit (ComfyUI mask-editor convention). "
                                             "Source-sized or canvas-sized. Optional for pure outpaint."}),
                "adapter": (["anypaint", "outpaint", "none"], {"default": "anypaint",
                    "tooltip": "anypaint: arbitrary masks, inpaint+outpaint (needs the AnyPaint LoRA "
                               "on the LoRA stack). outpaint: older rectangular-only adapter. none: "
                               "training-free - only pins the kept region (fine for small inpaints, "
                               "weak for outpaint)."}),
                "pad_left": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16,
                                     "tooltip": "Outpaint: pixels added on the left (source scale)."}),
                "pad_right": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16}),
                "pad_top": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16}),
                "pad_bottom": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16}),
                "seam_px": ("INT", {"default": 32, "min": 0, "max": 256, "step": 4,
                    "tooltip": "Band around the mask that is regenerated (not pinned) so the model "
                               "can blend. Canvas pixels; scaled per stage. Adapter default 32."}),
                "reference_max_edge": ("INT", {"default": 384, "min": 128, "max": 1024, "step": 16,
                    "tooltip": "Condition image size. 384 is the adapter's training contract - "
                               "change only to experiment."}),
                "restore_s1": (_RESTORE, {"default": "on",
                    "tooltip": "Pin kept tokens to the source at Stage 1."}),
                "restore_s2": (_RESTORE, {"default": "auto",
                    "tooltip": "auto: 'on' when the source has at least this stage's resolution, else "
                               "'structure'. on: pin kept regions every step (exact source pixels). "
                               "structure: pin while sigma > structure_release_sigma (keeps composition "
                               "and identity), then let the low-noise steps add detail - avoids forcing "
                               "an upsampled small source's blur. off: S2 may rewrite kept regions."}),
                "restore_s3": (_RESTORE, {"default": "auto"}),
                "adapter_s1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 1.0,
                    "tooltip": "Adapter reference conditioning at Stage 1 (1 = on, 0 = off)."}),
                "adapter_s2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 1.0,
                    "tooltip": "At Stage 2. Default off: the adapter was trained at <=1024 px targets."}),
                "adapter_s3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 1.0}),
                "structure_release_sigma": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "restore 'structure' (and auto on upscale stages): kept regions are pinned "
                               "while sigma is above this, free below. Lower = closer to the source, "
                               "higher = more fresh detail (and more drift)."}),
            },
        }

    RETURN_TYPES = ("KREA2_PAINT", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("paint", "vlm_image", "canvas_preview", "generated_mask")
    FUNCTION = "build"
    CATEGORY = "Eric/Krea2"

    def build(self, image, mask=None, adapter="anypaint", pad_left=0, pad_right=0, pad_top=0,
              pad_bottom=0, seam_px=32, reference_max_edge=384, restore_s1="on", restore_s2="auto",
              restore_s3="auto", adapter_s1=1.0, adapter_s2=0.0, adapter_s3=0.0,
              structure_release_sigma=0.4):
        src = image[:1, :, :, :3].float().clamp(0, 1).cpu()
        _, sh, sw, _ = src.shape
        # canvas = source + pads, snapped to /16 by trimming/growing the right/bottom pad
        cw, ch = round16(sw + pad_left + pad_right), round16(sh + pad_top + pad_bottom)
        x0, y0 = int(pad_left), int(pad_top)
        if pad_left + pad_right == 0 and pad_top + pad_bottom == 0 and (cw, ch) != (sw, sh):
            # pure inpaint on a non-/16 source: resize the source onto the canvas
            src = F.interpolate(src.permute(0, 3, 1, 2), size=(ch, cw), mode="bicubic",
                                align_corners=False).clamp(0, 1).permute(0, 2, 3, 1)
            sh, sw = ch, cw
        x1, y1 = min(cw, x0 + sw), min(ch, y0 + sh)
        placed = src[:, : y1 - y0, : x1 - x0]

        # generated mask on the canvas (1 = generate); outside the source box always generated
        gen = torch.ones(ch, cw)
        inner = torch.zeros(y1 - y0, x1 - x0)
        if mask is not None:
            m = mask[0] if mask.dim() == 3 else mask
            m = m.float().cpu()
            if tuple(m.shape) == (ch, cw) and (ch, cw) != (sh, sw):
                gen = (m > 0.5).float()
                gen[:y0] = 1; gen[y1:] = 1; gen[:, :x0] = 1; gen[:, x1:] = 1
                inner = None
            else:
                if tuple(m.shape) != (sh, sw):
                    print(f"[EricKrea2-Paint] mask {tuple(m.shape[::-1])} != source {sw}x{sh}: "
                          "resized (nearest)")
                inner = resize_mask(m, sh, sw)[: y1 - y0, : x1 - x0]
        if inner is not None:
            gen[y0:y1, x0:x1] = inner
        if gen.sum() < 1:
            raise ValueError("[EricKrea2-Paint] nothing to generate: the mask is empty and there "
                             "is no outpaint padding.")

        known_px = placed.reshape(-1, 3)
        fill = median_rgb(known_px)
        known = fill.view(1, 1, 1, 3).expand(1, ch, cw, 3).clone()
        known[:, y0:y1, x0:x1] = placed
        keep_px = known[0][gen < 0.5]
        cond_canvas = known.clone()
        cond_canvas[0][gen > 0.5] = median_rgb(keep_px)

        if adapter == "anypaint":
            condition = resize_max_edge(cond_canvas, reference_max_edge)
            bbox = [0.0, 0.0, 1.0, 1.0]
        elif adapter == "outpaint":
            if mask is not None and float(inner.sum() if inner is not None else 0) > 0:
                print("[EricKrea2-Paint] WARNING: the outpaint adapter is rectangular-only; "
                      "the interior mask is still regenerated but the adapter wasn't trained for it "
                      "- use anypaint for interior edits.")
            condition = resize_max_edge(placed, reference_max_edge)
            bbox = [x0 / cw, y0 / ch, x1 / cw, y1 / ch]
        else:
            condition, bbox = None, [0.0, 0.0, 1.0, 1.0]

        preview = known.clone()
        tint = torch.tensor([1.0, 0.25, 0.6]).view(1, 1, 3)
        g = gen[None, :, :, None]
        preview = preview * (1 - 0.55 * g) + tint * 0.55 * g

        bundle = {
            "adapter": adapter, "known": known, "generated": gen, "condition": condition,
            "bbox_norm": bbox, "canvas": (cw, ch), "source_px": (sw, sh),
            "src_box_norm": [x0 / cw, y0 / ch, x1 / cw, y1 / ch], "seam_px": int(seam_px),
            "restore": {1: restore_s1, 2: restore_s2, 3: restore_s3},
            "adapter_mult": {1: float(adapter_s1), 2: float(adapter_s2), 3: float(adapter_s3)},
            "cond_packed": None, "structure_release_sigma": float(structure_release_sigma),
        }
        frac = float(gen.mean())
        print(f"[EricKrea2-Paint] bundle: {adapter} | canvas {cw}x{ch} (source {sw}x{sh} at "
              f"{x0},{y0}) | generate {frac:.0%} of canvas | condition "
              f"{'-' if condition is None else f'{condition.shape[2]}x{condition.shape[1]}'}")
        if frac > 0.6 and adapter != "none":
            print("[EricKrea2-Paint] note: >60% of the canvas is generated - the adapter author "
                  "warns large generated fractions can add a second subject.")
        vlm = condition if condition is not None else resize_max_edge(cond_canvas, reference_max_edge)
        return (bundle, vlm, preview, gen[None])


class EricKrea2PaintComposite:
    """Paste the source back outside the mask with a feathered inward edge (pixel-exact
    preservation; the Outpaint adapter's post step). Optional - AnyPaint does not need it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The generated result (any stage size)."}),
                "paint": ("KREA2_PAINT",),
            },
            "optional": {
                "feather_px": ("INT", {"default": 32, "min": 0, "max": 512, "step": 2,
                    "tooltip": "Inward feather from the kept-region edge, canvas pixels "
                               "(scaled to the result)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"
    CATEGORY = "Eric/Krea2"

    def composite(self, image, paint, feather_px=32):
        b, H, W, _ = image.shape
        cw, ch = paint["canvas"]
        scale = W / float(cw)
        sw, sh = paint["source_px"]
        bx0, _, bx1, _ = paint["src_box_norm"]
        if sw < 0.9 * (bx1 - bx0) * W:
            print(f"[EricKrea2-Paint] composite note: the source ({sw}px wide) is smaller than its "
                  f"area in the {W}px result - pasted pixels are upsampled and may look softer.")
        known = F.interpolate(paint["known"].permute(0, 3, 1, 2), size=(H, W), mode="bicubic",
                              align_corners=False).clamp(0, 1).permute(0, 2, 3, 1)
        gen = resize_mask(paint["generated"], H, W)
        r = max(0, int(round(feather_px * scale)))
        if r > 0:
            grown = F.max_pool2d(gen[None, None], 2 * r + 1, 1, r)
            alpha = 1.0 - F.avg_pool2d(grown, 2 * r + 1, 1, r, count_include_pad=False)[0, 0]
        else:
            alpha = 1.0 - gen
        alpha = alpha.clamp(0, 1)[None, :, :, None].to(image.device, image.dtype)
        out = image * (1 - alpha) + known.to(image.device, image.dtype) * alpha
        print(f"[EricKrea2-Paint] composite: {W}x{H}, feather {r}px")
        return (out,)


NODE_CLASS_MAPPINGS = {"EricKrea2Paint": EricKrea2Paint,
                       "EricKrea2PaintComposite": EricKrea2PaintComposite}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2Paint": "Eric Krea2 Paint (inpaint / outpaint)",
                              "EricKrea2PaintComposite": "Eric Krea2 Paint Composite"}
