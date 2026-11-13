# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""Eric Krea2 Sweep Sheets (2026-10-16): make (or remake) share sheets for any finished sweep.

Wire Ultra V2's manifest_path output in, or paste a sweep folder / sweep_manifest.json path. Writes JPEGs into
<sweep>/share/ - see _sweep_sheets.py for the layout. Star ratings and notes come from the LoRA Catalog's Sweeps page.
"""

import os


class EricKrea2SweepSheets:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sweep": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "A sweep folder, or its sweep_manifest.json (Ultra V2's manifest_path output "
                               "plugs in here)."}),
                "mode": (["strips", "strips + singles", "singles"], {"default": "strips",
                    "tooltip": "strips = one labelled strip per prompt; singles = each image with its own labels."}),
                "background": (["black", "white"], {"default": "black"}),
                "group_by": (["auto", "prompt", "lora", "seed", "none"], {"default": "auto",
                    "tooltip": "What one strip holds. auto = one strip per prompt when the sweep has several "
                               "prompts, otherwise one strip for everything."}),
                "ratings": (["none", "stars", "stars + notes"], {"default": "none",
                    "tooltip": "Print your star ratings (and notes) from the catalog's Sweeps page under each "
                               "image."}),
                "tile_px": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64,
                    "tooltip": "Long side of each image in a strip. 768 lets people pixel-peep."}),
                "max_px": ("INT", {"default": 6000, "min": 1024, "max": 16000, "step": 100,
                    "tooltip": "No strip grows wider or taller than this; extra images wrap to new rows, "
                               "then new parts."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("share_folder",)
    FUNCTION = "make"
    OUTPUT_NODE = True
    CATEGORY = "Eric/Krea2"

    def make(self, sweep, mode="strips", background="black", group_by="auto", ratings="none",
             tile_px=768, max_px=6000):
        from .._sweep_sheets import make_sheets
        path = str(sweep or "").strip().strip('"')
        if not path:
            raise ValueError("Sweep Sheets: give a sweep folder or a sweep_manifest.json path")
        folder = os.path.dirname(path) if os.path.isfile(path) else path
        res = make_sheets(folder, mode=mode, background=background, group_by=group_by, tile_px=int(tile_px),
                          max_px=int(max_px), ratings=ratings,
                          log=lambda *a: print("[EricKrea2-Sheets]", *a))
        if not res.get("ok"):
            raise ValueError(f"Sweep Sheets: {res.get('error', 'failed')}")
        text = f"{len(res['files'])} file(s) in {res['folder']}"
        return {"ui": {"text": [text]}, "result": (res["folder"],)}


NODE_CLASS_MAPPINGS = {"EricKrea2SweepSheets": EricKrea2SweepSheets}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2SweepSheets": "Eric Krea2 Sweep Sheets"}
