# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
End-to-end check of the Control-LoRA runtime (_control.py) on the real model.

Loads the Krea-2 Turbo diffusers pipeline, builds a SYNTHETIC depth map (near =
white: receding floor + sphere + box), renders the same seed with and without
ControlRuntime installed (8 steps, guidance 0 - the reference Turbo recipe), and
writes depth | baseline | controlled strip. Also asserts hooks are fully removed
(post-removal render must equal the baseline bit-for-bit).

Usage: python test_control.py [--lora PATH] [--out DIR] [--size 1024] [--strength 1.0]
Author: Eric Hiss (GitHub: EricRollei)
"""

import argparse
import importlib
import os
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.abspath(os.path.join(HERE, "..", ".."))
COMFY = os.path.abspath(os.path.join(PACK, "..", ".."))


def load_pack():
    sys.path.insert(0, COMFY)
    pkg = types.ModuleType("k2pack")
    pkg.__path__ = [PACK]
    sys.modules["k2pack"] = pkg
    return importlib.import_module("k2pack._control")


def synthetic_depth(n):
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    d = 0.15 + 0.6 * np.clip((yy - 0.35) / 0.65, 0, 1)  # floor: far top, near bottom
    d[yy < 0.35] = 0.08  # back wall
    cx, cy, r = 0.62, 0.55, 0.17  # sphere
    rr = ((xx - cx) ** 2 + (yy - cy) ** 2) / r**2
    sph = rr < 1
    d[sph] = 0.75 + 0.25 * np.sqrt(1 - rr[sph])
    box = (xx > 0.14) & (xx < 0.38) & (yy > 0.42) & (yy < 0.74)  # box, front face
    d[box] = 0.62
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lora",
        default=r"L:/Models/loras/Krea2/control/krea2_depth_control_lora_patil.safetensors",
    )
    ap.add_argument("--base", default=r"H:/Training/Krea-2-Turbo")
    ap.add_argument("--out", default=r"A:/ai-tools/scratch/control_test")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    C = load_pack()
    from diffusers import Krea2Pipeline
    from PIL import Image

    t0 = time.perf_counter()
    pipe = Krea2Pipeline.from_pretrained(a.base, torch_dtype=torch.bfloat16).to(
        "cuda:0"
    )
    import importlib as _il

    _cp = _il.import_module(C.__name__.rsplit(".", 1)[0] + "._compat")
    _cp.apply_attention_backend(
        pipe.transformer, os.environ.get("K2_ATTN", "sdpa"), log=print
    )
    print(
        f"pipeline loaded in {time.perf_counter() - t0:.0f}s on {torch.cuda.get_device_name(0)}"
    )

    depth = synthetic_depth(a.size)
    img = torch.from_numpy(depth)[None, :, :, None].repeat(1, 1, 1, 3)
    img = C.preprocess_control_image(img, "grayscale", "per_image_minmax", False)
    prompt = (
        "a glossy red ceramic sphere and a weathered wooden crate on a stone floor in a dim "
        "vaulted hall, soft window light, photograph"
    )

    def render():
        g = torch.Generator("cuda:0").manual_seed(a.seed)
        return pipe(
            prompt=prompt,
            height=a.size,
            width=a.size,
            num_inference_steps=8,
            guidance_scale=0.0,
            generator=g,
        ).images[0]

    base = render()
    rt = C.ControlRuntime(
        pipe, {"path": a.lora, "image": img, "s1": a.strength, "s2": 0, "s3": 0}
    )
    rt.install()
    try:
        t0 = time.perf_counter()
        ctrl = render()
        print(
            f"controlled render {time.perf_counter() - t0:.1f}s, control calls {rt.n_ctrl_calls}"
        )
    finally:
        rt.remove()
    again = render()
    same = np.array_equal(np.asarray(again), np.asarray(base))
    print(f"hooks removed cleanly (post-removal render == baseline): {same}")

    dimg = Image.fromarray((depth / depth.max() * 255).astype(np.uint8)).convert("RGB")
    strip = Image.new("RGB", (a.size * 3, a.size))
    for i, im in enumerate((dimg, base, ctrl)):
        strip.paste(im.resize((a.size, a.size)), (i * a.size, 0))
    p = os.path.join(a.out, f"control_strip_s{a.strength}_seed{a.seed}.png")
    strip.save(p)
    strip.resize((a.size * 3 // 2, a.size // 2)).save(
        p.replace(".png", "_small.jpg"), quality=90
    )
    print(f"saved {p}")
    print("RESULT", "PASS" if same and rt.n_ctrl_calls > 0 else "CHECK")


if __name__ == "__main__":
    main()
