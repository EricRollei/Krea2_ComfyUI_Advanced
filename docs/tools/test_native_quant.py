# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Native quantized compute: speed + parity + PEFT-LoRA check (headless, Krea-2 Turbo).

For bf16 and each native format (fresh pipeline per format, same seed, flash_varlen +
mask-safe attention as in the pack): render at 1 MP and 3 MP (8 steps), report s/step
(steady state), PSNR vs bf16, peak VRAM; then with one PEFT LoRA loaded via the pack's
loader: PSNR(native+LoRA vs bf16+LoRA) and the LoRA's effect size on each.

Usage: python test_native_quant.py [--formats int8,int8_convrot,mxfp8,nvfp4,fp8] [--lora PATH]
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
from PIL import Image

PACK = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\custom_nodes\Eric_Krea2"
OUT = r"A:\ai-tools\scratch\native_quant_test"
TURBO = r"H:\Training\Krea-2-Turbo"
PROMPT = (
    "an old lighthouse keeper standing on a rocky coast at dusk, waves breaking, a lighthouse "
    "behind him, wet stones and tide pools in the foreground"
)
SIZES = {"1mp": (1024, 1024), "3mp": (2048, 1536)}


def load_pack(nq_path):
    sys.path.insert(0, os.path.abspath(os.path.join(PACK, "..", "..")))
    pkg = types.ModuleType("k2pack")
    pkg.__path__ = [PACK]
    sys.modules["k2pack"] = pkg
    spec = importlib.util.spec_from_file_location("k2pack._native_quant", nq_path)
    NQ = importlib.util.module_from_spec(spec)
    sys.modules["k2pack._native_quant"] = NQ
    spec.loader.exec_module(NQ)
    return (
        NQ,
        importlib.import_module("k2pack._compat"),
        importlib.import_module("k2pack._lora_utils"),
    )


def psnr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    m = float(((a - b) ** 2).mean())
    return round(99.0 if m == 0 else 10 * np.log10(255**2 / m), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq", default=os.path.join(PACK, "_native_quant.py"))
    ap.add_argument("--formats", default="int8,int8_convrot,mxfp8,nvfp4,fp8")
    ap.add_argument(
        "--lora",
        default=r"L:\Models\loras\Krea2\art-style\2650650-Oil_Painting_-_Thomas_Gainsborough\3098462-Krea2_v1\3098462_tgainsborough_000001800.safetensors",
    )
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    NQ, CP, LU = load_pack(a.nq)
    from diffusers import Krea2Pipeline

    res = {"gpu": torch.cuda.get_device_name(0)}

    def fresh():
        p = Krea2Pipeline.from_pretrained(TURBO, torch_dtype=torch.bfloat16).to(
            "cuda:0"
        )
        CP.apply_attention_backend(p.transformer, "auto", log=lambda *_: None)
        return p

    def render(pipe, size, tag):
        w, h = SIZES[size]
        times = []

        def cb(_p, i, _t, kw):
            torch.cuda.synchronize()
            times.append(time.perf_counter())
            return kw

        g = torch.Generator("cpu").manual_seed(11)
        torch.cuda.reset_peak_memory_stats()
        im = pipe(
            prompt=PROMPT,
            width=w,
            height=h,
            num_inference_steps=8,
            guidance_scale=0.0,
            generator=g,
            callback_on_step_end=cb,
        ).images[0]
        im.save(os.path.join(OUT, f"{tag}_{size}.png"))
        st = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        sps = sorted(st)[len(st) // 2] if st else float("nan")
        return im, round(sps, 3), round(torch.cuda.max_memory_allocated() / 1e9, 1)

    fmts = ["bf16"] + [f for f in a.formats.split(",") if f]
    ref = {}
    for fmt in fmts:
        pipe = fresh()
        entry = {}
        if fmt != "bf16":
            t = time.perf_counter()
            n = NQ.convert_transformer(pipe.transformer, fmt, 256, log=print)
            entry["convert_s"] = round(time.perf_counter() - t, 1)
            entry["converted"] = n
        render(pipe, "1mp", f"warm_{fmt}")
        for size in SIZES:
            im, sps, peak = render(pipe, size, fmt)
            entry[size] = {"s_per_step": sps, "peak_gb": peak}
            if fmt == "bf16":
                ref[size] = im
            else:
                entry[size]["psnr_vs_bf16"] = psnr(im, ref[size])
        # PEFT LoRA check at 1 MP
        try:
            LU.load_lora_with_key_fix(
                pipe, a.lora, "nqtest", log_prefix="[nq-LoRA]", weight=1.0
            )
            iml, _, _ = render(pipe, "1mp", f"{fmt}_lora")
            if fmt == "bf16":
                ref["lora"] = iml
                entry["lora_effect_psnr_vs_nolora"] = psnr(iml, ref["1mp"])
            else:
                entry["lora_psnr_vs_bf16lora"] = psnr(iml, ref["lora"])
                entry["lora_effect_psnr_vs_nolora"] = psnr(
                    iml, Image.open(os.path.join(OUT, f"{fmt}_1mp.png"))
                )
        except Exception as e:
            entry["lora_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        res[fmt] = entry
        print(fmt, json.dumps(entry), flush=True)
        del pipe
        torch.cuda.empty_cache()
    b = res["bf16"]
    for fmt in fmts[1:]:
        e = res[fmt]
        for size in SIZES:
            if size in e and size in b:
                e[size]["speedup"] = round(
                    b[size]["s_per_step"] / e[size]["s_per_step"], 2
                )
    json.dump(res, open(os.path.join(OUT, "native_quant_results.json"), "w"), indent=1)
    print("SUMMARY", json.dumps(res, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
