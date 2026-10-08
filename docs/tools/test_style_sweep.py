# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Style-reference strength sweep (headless, diffusers Krea2Pipeline + _style_ref.StyleRuntime).
Rows = reference images, columns = ref | baseline | one column per setting level, from the
author's recommended preset up to deliberately overdriven settings, to find where signature
style improves vs where content starts leaking.

Usage: python test_style_sweep.py REF [REF ...] [--out DIR] [--size 1024] [--steps 8] [--seed 11]
Pick the GPU with CUDA_DEVICE_ORDER=PCI_BUS_ID + CUDA_VISIBLE_DEVICES.
Author: Eric Hiss (GitHub: EricRollei)
"""

import argparse
import importlib
import json
import os
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.abspath(os.path.join(HERE, "..", ".."))
COMFY = os.path.abspath(os.path.join(PACK, "..", ".."))

# label -> overrides on top of PRESET_RECOMMENDED (custom route)
CONFIGS = [
    ("R recommended", {}),
    ("A strength1.5", {"style_strength": 1.5}),
    ("B k1.25 low1.6", {"ref_k_strength": 1.25, "low_scale_end": 1.6}),
    (
        "C k1.5 low2.2 ad1",
        {"ref_k_strength": 1.5, "low_scale_end": 2.2, "adain_strength": 1.0},
    ),
    (
        "D C+blocks3-27",
        {
            "ref_k_strength": 1.5,
            "low_scale_end": 2.2,
            "adain_strength": 1.0,
            "blocks": "3-27",
        },
    ),
    (
        "E k1.8 low3 hi1.2",
        {
            "ref_k_strength": 1.8,
            "low_scale_end": 3.0,
            "adain_strength": 1.0,
            "high_scale_start": 1.2,
        },
    ),
]


def load_pack():
    sys.path.insert(0, COMFY)
    pkg = types.ModuleType("k2pack")
    pkg.__path__ = [PACK]
    sys.modules["k2pack"] = pkg
    return importlib.import_module("k2pack._style_ref")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("refs", nargs="+")
    ap.add_argument(
        "--prompt",
        default="an old lighthouse keeper standing on a rocky coast at dusk, "
        "waves breaking, a lighthouse behind him",
    )
    ap.add_argument("--base", default=r"H:/Training/Krea-2-Turbo")
    ap.add_argument("--out", default=r"A:/ai-tools/scratch/style_sweep")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument(
        "--configs",
        default=None,
        help="JSON list of [label, {overrides}] replacing the built-in CONFIGS",
    )
    a = ap.parse_args()
    global CONFIGS
    if a.configs:
        CONFIGS = [(l, dict(o)) for l, o in json.loads(a.configs)]
    os.makedirs(a.out, exist_ok=True)
    S = load_pack()
    from diffusers import Krea2Pipeline
    from PIL import Image, ImageDraw, ImageFont

    t0 = time.perf_counter()
    pipe = Krea2Pipeline.from_pretrained(a.base, torch_dtype=torch.bfloat16).to(
        "cuda:0"
    )
    print(
        f"pipeline loaded {time.perf_counter() - t0:.0f}s on {torch.cuda.get_device_name(0)}",
        flush=True,
    )

    def render():
        g = torch.Generator("cuda:0").manual_seed(a.seed)
        t = time.perf_counter()
        im = pipe(
            prompt=a.prompt,
            height=a.size,
            width=a.size,
            num_inference_steps=a.steps,
            guidance_scale=0.0,
            generator=g,
        ).images[0]
        return im, time.perf_counter() - t

    def load_ref(path):
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr)[None]

    render()
    base, tb = render()
    base.save(os.path.join(a.out, "baseline.png"))
    print(f"baseline {tb:.1f}s", flush=True)

    log = {
        "prompt": a.prompt,
        "seed": a.seed,
        "size": a.size,
        "configs": dict(CONFIGS),
        "times": {},
    }
    results = {}
    for path in a.refs:
        ref = load_ref(path)
        rname = os.path.splitext(os.path.basename(path))[0]
        for label, ov in CONFIGS:
            p = dict(S.PRESET_RECOMMENDED)
            p.update(ov)
            p["trajectory_steps"] = a.steps
            rt = S.StyleRuntime(
                pipe, {"image": ref, "params": p, "s1": 1.0, "s2": 0.0, "s3": 0.0}
            )
            rt.install()
            try:
                im, ts = render()
            finally:
                rt.remove()
            tag = label.split()[0]
            im.save(os.path.join(a.out, f"{rname}__{tag}.png"))
            results[(path, label)] = im
            log["times"][f"{rname}/{tag}"] = round(ts, 1)
            print(f"{rname} {label}: {ts:.1f}s", flush=True)

    n = a.size
    cols = 2 + len(CONFIGS)
    head = 64
    grid = Image.new("RGB", (n * cols, head + n * len(a.refs)), (20, 20, 20))
    d = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except Exception:
        font = ImageFont.load_default()
    for c, t in enumerate(["reference", "baseline"] + [l for l, _ in CONFIGS]):
        d.text((c * n + 16, 12), t, fill=(235, 235, 235), font=font)
    for r, path in enumerate(a.refs):
        ref_im = Image.open(path).convert("RGB")
        ref_im.thumbnail((n, n))
        y = head + r * n
        grid.paste(ref_im, ((n - ref_im.width) // 2, y + (n - ref_im.height) // 2))
        grid.paste(base, (n, y))
        for c, (label, _) in enumerate(CONFIGS):
            grid.paste(results[(path, label)], ((2 + c) * n, y))
    gp = os.path.join(a.out, "style_sweep_grid.png")
    grid.save(gp)
    grid.resize((grid.width // 4, grid.height // 4)).save(
        gp.replace(".png", "_small.jpg"), quality=88
    )
    with open(os.path.join(a.out, "sweep_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    print(f"saved {gp}\nDONE", flush=True)


if __name__ == "__main__":
    main()
