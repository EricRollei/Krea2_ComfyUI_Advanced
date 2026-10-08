# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Headless check of the style-reference runtime (_style_ref.py) on Krea-2 Turbo.

  1. baseline render (no style)
  2. runtime installed at strength 0 -> must equal baseline bit-for-bit
  3. one styled render per reference image (same prompt / seed)
  4. runtime removed -> render must equal baseline bit-for-bit
Writes ref | baseline | styled strips + a combined grid, and timings.

Usage: python test_style.py REF [REF ...] [--prompt P] [--size 1024] [--steps 8] [--out DIR]
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
    ap.add_argument("--out", default=r"A:/ai-tools/scratch/style_test")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--trajectory", default="model_pc")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = load_pack()
    from diffusers import Krea2Pipeline
    from PIL import Image

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

    def bundle(img, strength):
        p = dict(S.PRESET_RECOMMENDED)
        p["style_strength"] = strength
        p["trajectory"] = a.trajectory
        p["trajectory_steps"] = a.steps
        return {"image": img, "params": p, "s1": 1.0, "s2": 0.0, "s3": 0.0}

    render()  # warm-up (kernels / allocator)
    base, tb = render()
    print(f"baseline {tb:.1f}s", flush=True)

    def load_ref(path):
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr)[None]

    refs = [load_ref(p) for p in a.refs]
    rt = S.StyleRuntime(pipe, bundle(refs[0], 0.0))
    rt.install()
    try:
        s0, _ = render()
    finally:
        rt.remove()
    same0 = np.array_equal(np.asarray(s0), np.asarray(base))
    print(f"strength 0 == baseline: {same0}", flush=True)

    rows = []
    for path, ref in zip(a.refs, refs):
        rt = S.StyleRuntime(pipe, bundle(ref, 1.0))
        rt.install()
        try:
            st, ts = render()
        finally:
            rt.remove()
        print(
            f"{os.path.basename(path)}: styled {ts:.1f}s (x{ts / tb:.2f} baseline)",
            flush=True,
        )
        rows.append((path, st))

    after, _ = render()
    same_after = np.array_equal(np.asarray(after), np.asarray(base))
    print(f"after removal == baseline: {same_after}", flush=True)

    n = a.size
    grid = Image.new("RGB", (n * 3, n * len(rows)), (20, 20, 20))
    for r, (path, st) in enumerate(rows):
        ref_im = Image.open(path).convert("RGB")
        ref_im.thumbnail((n, n))
        grid.paste(ref_im, ((n - ref_im.width) // 2, r * n + (n - ref_im.height) // 2))
        grid.paste(base, (n, r * n))
        grid.paste(st, (2 * n, r * n))
        st.save(
            os.path.join(
                a.out, f"styled_{os.path.splitext(os.path.basename(path))[0]}.png"
            )
        )
    gp = os.path.join(a.out, f"style_grid_{a.trajectory}.png")
    grid.save(gp)
    grid.resize((grid.width // 4, grid.height // 4)).save(
        gp.replace(".png", "_small.jpg"), quality=85
    )
    print(f"saved {gp}")
    print("RESULT", "PASS" if same0 and same_after else "CHECK")


if __name__ == "__main__":
    main()
