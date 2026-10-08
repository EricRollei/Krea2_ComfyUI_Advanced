# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
DeGrid checks:
  1) synthetic: smooth gradient + edges + known 2px checker (2/255) -> the lattice
     must be detected, removed (residual << injected) and the edge preserved;
     a clean image must be passed through bit-exact (skip_when_clean).
  2) real: measure the phase-locked lattice amplitude in PNGs (e.g. Ultra outputs).

Usage: python test_degrid.py [png ...]
Author: Eric Hiss (GitHub: EricRollei)
"""

import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "k2degrid", os.path.join(HERE, "..", "..", "_degrid.py"))
dg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dg)


def synthetic():
    h, w = 512, 768
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    base = (0.15 + 0.5 * xx * yy).expand(3, h, w).clone()
    base[:, 200:320, 300:500] += 0.3                      # a hard-edged block
    clean = base.clamp(0, 1).unsqueeze(0)
    checker = ((torch.arange(h)[:, None] + torch.arange(w)[None, :]) % 2).float() * 2 - 1
    grid = checker * (2.0 / 255.0)                        # +-2/255 lattice
    dirty = (clean + grid).clamp(0, 1)

    out, st = dg.degrid_bchw(dirty)
    resid = (out - clean)[..., 8:-8, 8:-8]
    print(f"synthetic: detected {st[0]['amp_255']:.2f}/255 (injected p2p 4.00), "
          f"skipped={st[0]['skipped']}")
    print(f"  residual vs clean: mean|.| {resid.abs().mean() * 255:.3f}/255, "
          f"max {resid.abs().max() * 255:.2f}/255 (lattice was 2.00)")
    edge = (out - clean)[..., 199:202, 300:500].abs().max() * 255
    print(f"  edge rows max deviation {edge:.2f}/255")
    out2, st2 = dg.degrid_bchw(clean)
    print(f"clean image: detected {st2[0]['amp_255']:.3f}/255, skipped={st2[0]['skipped']}, "
          f"bit-exact={torch.equal(out2, clean)}")
    ok = (not st[0]["skipped"]) and resid.abs().mean() * 255 < 0.3 and st2[0]["skipped"]
    print("SYNTHETIC", "PASS" if ok else "FAIL")


def real(paths):
    from PIL import Image
    for p in paths:
        im = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        x = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0)
        if torch.cuda.is_available():
            x = x.cuda()
        _, st = dg.degrid_bchw(x)
        s = st[0]
        verdict = "none (passed through)" if s["skipped"] else \
            f"REMOVED, limit {s['limit']:.3f}, edges protected {s['edges_protected_pct']:.1f}%"
        print(f"{os.path.basename(p)} {im.shape[1]}x{im.shape[0]}: lattice "
              f"{s['amp_255']:.2f}/255 -> {verdict}")


if __name__ == "__main__":
    synthetic()
    if len(sys.argv) > 1:
        real(sys.argv[1:])
