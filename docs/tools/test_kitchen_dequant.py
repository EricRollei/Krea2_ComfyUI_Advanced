# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Headless test for the component loader's comfy_kitchen dequant path.

  phase 1 (fast): dequantize sample layers of a quantized checkpoint with the
                  loader's own _dequant_comfy_quant and compare each against the
                  same layer of the Turbo bf16 base (cosine + rel-RMS). A correct
                  decode of a fine-tune is close to its base (cos ~0.9+); a scale
                  layout error collapses cosine toward 0.
  phase 2 (--full): full _override_transformer load into the meta skeleton,
                  asserting no missing/unexpected keys and no meta params left.

Usage: python test_kitchen_dequant.py CKPT [--full] [--device cuda:0]
Loads the loader module through a stub package so the node pack's __init__
(server routes) is never imported. Author: Eric Hiss (GitHub: EricRollei)
"""

import argparse
import importlib
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.abspath(os.path.join(HERE, "..", ".."))
COMFY = os.path.abspath(os.path.join(PACK, "..", ".."))
BASE = r"H:/Training/Krea-2-Turbo"
REF = r"A:/Models/diffusion_models/Krea2/krea2_turbo_bf16.safetensors"
SAMPLES = [
    "blocks.0.attn.wq.weight",
    "blocks.5.mlp.down.weight",
    "blocks.13.attn.wo.weight",
    "blocks.27.mlp.up.weight",
    "txtfusion.layerwise_blocks.0.attn.wk.weight",
]


def load_loader():
    sys.path.insert(0, COMFY)
    pkg = types.ModuleType("k2pack")
    pkg.__path__ = [PACK]
    sys.modules["k2pack"] = pkg
    sub = types.ModuleType("k2pack.nodes")
    sub.__path__ = [os.path.join(PACK, "nodes")]
    sys.modules["k2pack.nodes"] = sub
    return importlib.import_module("k2pack.nodes.krea2_component_loader")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    import torch
    from safetensors import safe_open

    L = load_loader()
    print(
        f"loader imported; torch {torch.__version__}; device {a.device} = "
        f"{torch.cuda.get_device_name(torch.device(a.device))}"
    )

    # ---- phase 1: sample layers vs bf16 base ----
    if a.ckpt.endswith(".gguf"):
        a.full = True
        print("GGUF: phase 1 n/a (numpy dequant path) - full load only")
    else:
        phase1(a, L, torch, safe_open)
    if not a.full:
        return
    _phase2(a, L, torch)


def phase1(a, L, torch, safe_open):
    with safe_open(a.ckpt, framework="pt") as f:
        keys = list(f.keys())
        pre = (
            "model.diffusion_model."
            if any(k.startswith("model.diffusion_model.") for k in keys)
            else ""
        )
        want = set()
        for s in SAMPLES:
            for suf in ("", "_scale", "_scale_2"):
                if pre + s + suf in keys:
                    want.add(pre + s + suf)
            m = pre + s[: -len("weight")] + "comfy_quant"
            if m in keys:
                want.add(m)
        sd = {k: f.get_tensor(k) for k in want}
    confs = L._layer_quant_confs(sd, path=a.ckpt)
    t0 = time.perf_counter()
    out, n = L._dequant_comfy_quant(
        sd, torch.bfloat16, log=print, device=a.device, quant_confs=confs
    )
    print(f"dequantized {n} sample weights in {time.perf_counter() - t0:.2f}s")
    worst = 1.0
    with safe_open(REF, framework="pt") as r:
        rkeys = set(r.keys())
        for s in SAMPLES:
            k = pre + s
            if k not in out or s not in rkeys:
                print(f"  skip {s}")
                continue
            w = out[k].double()
            ref = r.get_tensor(s).double()
            cos = torch.nn.functional.cosine_similarity(
                w.flatten(), ref.flatten(), dim=0
            ).item()
            rel = ((w - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
            worst = min(worst, cos)
            print(
                f"  {s:48s} {tuple(w.shape)} fmt={confs.get(s[:-7], {}).get('format', '-'):6s} "
                f"cos={cos:.4f} relRMS={rel:.3f} std={w.std():.4g}/{ref.std():.4g}"
            )
    print(f"PHASE1 {'PASS' if worst > 0.8 else 'FAIL'} (worst cosine {worst:.4f})")


def _phase2(a, L, torch):
    # ---- phase 2: full transformer load ----
    t0 = time.perf_counter()
    loader = L.EricKrea2ComponentLoader()
    m = loader._override_transformer(
        BASE, a.ckpt, torch.bfloat16, print, device=a.device
    )
    metas = [n for n, p in m.named_parameters() if p.is_meta]
    nonfinite = [n for n, p in m.named_parameters() if not torch.isfinite(p).all()]
    print(
        f"full load {time.perf_counter() - t0:.1f}s | params {sum(p.numel() for p in m.parameters()) / 1e9:.2f}B "
        f"| meta left {len(metas)} | non-finite {len(nonfinite)}"
    )
    print(f"PHASE2 {'PASS' if not metas and not nonfinite else 'FAIL'}")


if __name__ == "__main__":
    main()
