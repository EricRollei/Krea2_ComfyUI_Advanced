# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.

"""
Eric Krea2 debug nodes
======================
Utility nodes for diagnosing latent-space issues.

  EricKrea2SaveLatentDebug -> write a KREA2_LATENT (packed tensor + dims) to a
                              .safetensors file on disk, and pass it through
                              unchanged so it can sit inline in a graph.
"""

from __future__ import annotations

import os
import re


class EricKrea2SaveLatentDebug:
    """Save a KREA2_LATENT to disk for offline inspection. Pass-through node.

    auto_index=True (default) never overwrites: it writes
    <stem>_<NNNN>[_<label>]_<W>x<H>.safetensors, auto-incrementing NNNN by
    scanning the folder. Batch-friendly (saves on every queue)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("KREA2_LATENT",),
                "path": ("STRING", {
                    "default": "A:\\_dbg_latent.safetensors",
                    "tooltip": "Base path. With auto_index ON this is a prefix: the file written is "
                               "<stem>_<NNNN>[_<label>]_<W>x<H>.safetensors next to it."}),
            },
            "optional": {
                "auto_index": ("BOOLEAN", {"default": True,
                    "tooltip": "ON: never overwrite - append an auto-incrementing 4-digit index (+ dims "
                               "and label) so you can batch-save many. OFF: write exactly 'path'."}),
                "label": ("STRING", {"default": "",
                    "tooltip": "Optional tag folded into the filename (e.g. 'banded', 'clean') to make "
                               "batches easy to identify."}),
            },
        }

    RETURN_TYPES = ("KREA2_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Eric/Krea2"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always run, so batch queues save every item even on identical inputs.
        return float("nan")

    def _next_path(self, path, label, w, h):
        d = os.path.dirname(path) or "."
        stem, ext = os.path.splitext(os.path.basename(path))
        ext = ext or ".safetensors"
        os.makedirs(d, exist_ok=True)
        rx = re.compile(r"^" + re.escape(stem) + r"_(\d{4})(?:_.*)?" + re.escape(ext) + r"$")
        mx = -1
        for fn in os.listdir(d):
            m = rx.match(fn)
            if m:
                mx = max(mx, int(m.group(1)))
        lab = ""
        if label.strip():
            lab = "_" + re.sub(r"[^A-Za-z0-9]+", "-", label.strip()).strip("-")
        return os.path.join(d, f"{stem}_{mx + 1:04d}{lab}_{w}x{h}{ext}")

    def save(self, latent, path, auto_index=True, label=""):
        from safetensors.torch import save_file
        packed = latent["packed"].detach().cpu().contiguous().float()
        h = int(latent["height"])
        w = int(latent["width"])
        meta = {
            "height": str(h),
            "width": str(w),
            "vae_scale_factor": str(int(latent.get("vae_scale_factor", 8))),
            "label": label,
        }
        if auto_index:
            out_path = self._next_path(path, label, w, h)
        else:
            out_path = path
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        save_file({"packed": packed}, out_path, metadata=meta)
        print(f"[EricKrea2] Saved latent {tuple(packed.shape)} -> {out_path} "
              f"(h={h} w={w} label='{label}')")
        return (latent,)


NODE_CLASS_MAPPINGS = {
    "EricKrea2SaveLatentDebug": EricKrea2SaveLatentDebug,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EricKrea2SaveLatentDebug": "Eric Krea2 Save Latent (debug)",
}
