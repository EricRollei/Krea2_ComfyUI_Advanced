# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 Control (Control-LoRA / "ControlNet" for Krea 2)
===========================================================
Bundles a control image (depth / pose / canny / lineart ... already preprocessed by
any ComfyUI preprocessor node) with a Krea2 Control-LoRA checkpoint and per-stage
strengths. Wire the KREA2_CONTROL output into Multi-Stage Ultra's `control` socket.
Nothing is loaded here; Ultra installs the control hooks for one generation (see
_control.py) and removes them afterwards.

Public weights: Patil/Krea-2-depth-controlnet (depth; trained on Krea-2-Raw, works on
Raw and Turbo; ~1 MP buckets). The control TYPE is decided by the checkpoint - a depth
LoRA needs a depth map (near = white), pose/canny LoRAs need their matching map.

Author: Eric Hiss (GitHub: EricRollei)
"""

from .._lora_utils import get_lora_list, get_lora_full_path


def _control_choices():
    names = get_lora_list("krea2")
    ctrl = [n for n in names if "control" in n.lower().replace("\\", "/")]
    rest = [n for n in names if n not in ctrl and n != "none"]
    return (ctrl + rest) or ["none"]


class EricKrea2Control:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_image": ("IMAGE", {
                    "tooltip": "Preprocessed control map (e.g. Depth Anything / DepthPro output for the "
                               "depth LoRA). Any size - it is cover-cropped to each stage's resolution, "
                               "so keep it at the generation's aspect ratio."}),
                "control_lora": (_control_choices(), {
                    "tooltip": "Krea2 Control-LoRA checkpoint (expanded input layer + block LoRA). Files "
                               "with 'control' in their path are listed first."}),
            },
            "optional": {
                "channel_mode": (["grayscale", "rgb"], {"default": "grayscale",
                    "tooltip": "grayscale: luminance repeated to RGB (depth maps). rgb: keep colors "
                               "(pose / canny / normal / lineart LoRAs)."}),
                "normalize": (["per_image_minmax", "none"], {"default": "per_image_minmax",
                    "tooltip": "per_image_minmax stretches the map to 0..1 - matches the depth LoRA's "
                               "training (inverse depth, min-max). Use none for pose/canny."}),
                "invert": ("BOOLEAN", {"default": False,
                    "tooltip": "Flip the map. The depth LoRA expects NEAR = WHITE; turn this on only if "
                               "your depth preview shows near objects dark."}),
                "s1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Stage 1 control strength (= the reference's --lora-scale on the block "
                               "LoRA; the expanded input layer is always full). 0 = stage runs the stock "
                               "model. ~0.6 = weaker structure adherence, more freedom."}),
                "s2_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Stage 2 strength. Structure is set in S1; a moderate value keeps the "
                               "refine from drifting. 0 = stock model for this stage."}),
                "s3_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Stage 3 strength (detail pass). Usually 0: the trained buckets are "
                               "~1 MP and S3 runs far above that."}),
            },
        }

    RETURN_TYPES = ("KREA2_CONTROL", "IMAGE")
    RETURN_NAMES = ("control", "control_preview")
    FUNCTION = "build"
    CATEGORY = "Eric/Krea2"

    def build(self, control_image, control_lora, channel_mode="grayscale",
              normalize="per_image_minmax", invert=False,
              s1_strength=1.0, s2_strength=0.5, s3_strength=0.0):
        from .._control import preprocess_control_image
        path = get_lora_full_path(control_lora)
        if not path:
            raise ValueError(f"[EricKrea2-Control] control LoRA not found: {control_lora}")
        img = preprocess_control_image(control_image, channel_mode, normalize, bool(invert))
        bundle = {"path": path, "image": img, "s1": float(s1_strength), "s2": float(s2_strength),
                  "s3": float(s3_strength), "name": control_lora}
        print(f"[EricKrea2-Control] bundle: {control_lora} | {channel_mode}/{normalize}"
              f"{'/inverted' if invert else ''} | S1/S2/S3 {s1_strength}/{s2_strength}/{s3_strength} "
              f"| control image {img.shape[2]}x{img.shape[1]}")
        return (bundle, img)


NODE_CLASS_MAPPINGS = {"EricKrea2Control": EricKrea2Control}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2Control": "Eric Krea2 Control (Depth/Pose ControlNet-LoRA)"}
