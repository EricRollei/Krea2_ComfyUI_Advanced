# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 Settings from Image
==============================
Load a "master" PNG that was saved with an embedded KREA2_SETTINGS recipe (e.g.
by your Save Image node) and fan it back out:

    IMAGE  + MASK          - the decoded picture (also works as a plain loader)
    settings               - the full recipe blob (feed straight into Merge)
    loader / ultra / lora  - per-section strings for reproduction

The recipe is read from the PNG's text chunks. By default every chunk is scanned
for the ``krea2_settings_version`` marker, so it does not matter which key the
Save Image node used; set ``metadata_key`` to read one specific chunk.

A JS button (``web/js/krea2_settings_from_image.js``) adds "★ Save recipe →
presets", which asks the server to extract the recipe and drop it into the
loader / ultra / lora preset libraries under a name you choose - the image →
named-preset bridge, so any master image becomes selectable in the node panels.

Author: Eric Hiss (GitHub: EricRollei)
"""

import hashlib
import json
import os

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths

from .. import _settings


class EricKrea2SettingsFromImage:
    """Read an embedded KREA2_SETTINGS recipe from a PNG and fan it out (image +
    mask + full blob + per-section strings). Use the ★ button to save the recipe
    into the preset libraries."""

    CATEGORY = "Eric/Krea2"
    FUNCTION = "load"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "settings", "loader", "ultra", "lora")

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        try:
            files = [f for f in os.listdir(input_dir)
                     if os.path.isfile(os.path.join(input_dir, f))]
        except Exception:
            files = []
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True,
                    "tooltip": "A PNG saved with an embedded KREA2_SETTINGS recipe. The recipe is "
                               "read from the PNG's text chunks; the picture is decoded too."}),
            },
            "optional": {
                "metadata_key": ("STRING", {"default": "",
                    "tooltip": "Optional: the exact PNG text-chunk key holding the recipe. Leave "
                               "blank to auto-scan every chunk for the KREA2_SETTINGS marker."}),
            },
        }

    def load(self, image, metadata_key=""):
        path = folder_paths.get_annotated_filepath(image)
        img = Image.open(path)

        out_images, out_masks = [], []
        for frame in ImageSequence.Iterator(img):
            frame = ImageOps.exif_transpose(frame)
            if frame.mode == "I":
                frame = frame.point(lambda x: x * (1 / 255))
            rgb = frame.convert("RGB")
            arr = np.array(rgb).astype(np.float32) / 255.0
            out_images.append(torch.from_numpy(arr)[None, ])
            if "A" in frame.getbands():
                m = np.array(frame.getchannel("A")).astype(np.float32) / 255.0
                m = 1.0 - torch.from_numpy(m)
            else:
                m = torch.zeros((64, 64), dtype=torch.float32)
            out_masks.append(m.unsqueeze(0))

        if len(out_images) > 1:
            image_t = torch.cat(out_images, dim=0)
            mask_t = torch.cat(out_masks, dim=0)
        else:
            image_t = out_images[0]
            mask_t = out_masks[0]

        blob = _settings.extract_from_png(path, str(metadata_key or "").strip())
        full_s = loader_s = ultra_s = lora_s = ""
        if isinstance(blob, dict) and blob:
            full_s = json.dumps(blob, indent=2, ensure_ascii=False)
            if isinstance(blob.get("loader"), dict):
                loader_s = _settings.wrap("loader", blob["loader"])
            if isinstance(blob.get("ultra"), dict):
                ultra_s = _settings.wrap("ultra", blob["ultra"])
            if "lora" in blob:
                lora_s = _settings.wrap("lora", blob["lora"])
            print(f"[EricKrea2-FromImage] recipe found in '{os.path.basename(path)}': "
                  f"loader={'y' if loader_s else '-'} ultra={'y' if ultra_s else '-'} "
                  f"lora={'y' if lora_s else '-'}.")
        else:
            print(f"[EricKrea2-FromImage] no KREA2_SETTINGS recipe embedded in "
                  f"'{os.path.basename(path)}'.")

        return (image_t, mask_t, full_s, loader_s, ultra_s, lora_s)

    @classmethod
    def IS_CHANGED(cls, image, metadata_key=""):
        try:
            path = folder_paths.get_annotated_filepath(image)
            m = hashlib.sha256()
            with open(path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()
        except Exception:
            return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, image, metadata_key=""):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


NODE_CLASS_MAPPINGS = {"EricKrea2SettingsFromImage": EricKrea2SettingsFromImage}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2SettingsFromImage": "Eric Krea2 Settings from Image"}
