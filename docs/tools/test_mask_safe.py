"""Verify _compat mask-safe attention: same seed/prompt, Turbo 8 steps.
  A: native SDPA (stock processor, reference)    B: flash_varlen + mask-safe processor
  C: flash_varlen stock processor (the bug)
Reports PSNR/MAD vs A (whole + bottom 10%), timings; saves a side-by-side.
Usage: python test_mask_safe.py COMPAT_PATH [W H]"""
import importlib.util
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

spec = importlib.util.spec_from_file_location("k2compat", sys.argv[1])
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
W = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
H = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
OUT = r"A:\ai-tools\scratch\mask_safe_test"; os.makedirs(OUT, exist_ok=True)
PROMPT = ("an old lighthouse keeper standing on a rocky coast at dusk, waves breaking, a lighthouse behind him, "
          "wet stones and tide pools in the foreground")
from diffusers import Krea2Pipeline
pipe = Krea2Pipeline.from_pretrained(r"H:\Training\Krea-2-Turbo", torch_dtype=torch.bfloat16).to("cuda:0")
tr = pipe.transformer
stock = {id(m): m.processor for m in tr.modules() if getattr(m, "processor", None) is not None}


def restore_stock():
    for m in tr.modules():
        if id(m) in stock:
            m.processor = stock[id(m)]


def render(tag):
    g = torch.Generator("cpu").manual_seed(11)
    torch.cuda.synchronize(); t = time.perf_counter()
    im = pipe(prompt=PROMPT, height=H, width=W, num_inference_steps=8, guidance_scale=0.0, generator=g).images[0]
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    im.save(os.path.join(OUT, f"{tag}.png"))
    return im, dt


def metr(a, b, frac=None):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    if frac:
        h = a.shape[0]; a = a[int(h * (1 - frac)):]; b = b[int(h * (1 - frac)):]
    mse = float(((a - b) ** 2).mean())
    return round(99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse), 2), round(float(np.abs(a - b).mean()), 2)


tr.set_attention_backend("native"); render("warm")
A, ta = render("A_native_stock")
tr.set_attention_backend("flash_varlen")
n = C.install_mask_safe_attention(tr, log=print)
render("warm2")
B, tb = render("B_flashvarlen_masksafe")
restore_stock(); tr.set_attention_backend("flash_varlen")
Cc, tc = render("C_flashvarlen_stock")
print(f"swapped {n} processors | times: native {ta:.1f}s, flash+fix {tb:.1f}s, flash stock {tc:.1f}s")
print("B vs A (fix vs reference): whole", metr(B, A), "bottom10%", metr(B, A, 0.10))
print("C vs A (bug vs reference): whole", metr(Cc, A), "bottom10%", metr(Cc, A, 0.10))
g = Image.new("RGB", (W * 3 // 2, H // 2))
for i, im in enumerate((A, B, Cc)):
    g.paste(im.resize((W // 2, H // 2)), (i * W // 2, 0))
g.save(os.path.join(OUT, "A_B_C.jpg"), quality=88)
print("DONE")
