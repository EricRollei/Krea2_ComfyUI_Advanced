"""End-to-end A/B: attention_backend auto (flash-attn 2) vs sage_fp8 (SageAttention 2 int8-QK /
fp8-PV dense) on Krea 2 Turbo bf16, same seeds. Reports s/step (callback timing, steady-state
steps only) and PSNR sage vs flash per image; saves PNG pairs. 2026-10-08.
Usage: python test_sage_fp8_ab.py [MODEL_DIR] [OUT_DIR]"""
import importlib, os, sys, time, types, json
import numpy as np
import torch
PACK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.abspath(os.path.join(PACK, "..", "..")))
pkg = types.ModuleType("k2p"); pkg.__path__ = [PACK]; sys.modules["k2p"] = pkg
CP = importlib.import_module("k2p._compat")
MODEL = sys.argv[1] if len(sys.argv) > 1 else r"H:\Training\Krea-2-Turbo"
OUT = sys.argv[2] if len(sys.argv) > 2 else r"A:\ai-tools\scratch\sage_ab"
os.makedirs(OUT, exist_ok=True)
STEPS = 8
SIZES = [("1.5MP", 1408, 1088), ("3MP", 2048, 1536), ("8.4MP", 3328, 2496)]
PROMPTS = [
    ("portrait", "close-up portrait of an elderly fisherman with a weathered face and white beard, "
                 "wool cap, overcast harbour light, shallow depth of field, photo"),
    ("text", "a hand-painted shop sign reading \"BEAR & CO. FINE INSTRUMENTS\" above a brass-framed "
             "window full of antique microscopes, late afternoon sun, photo"),
    ("detail", "macro photo of frost crystals on a dark green leaf, intricate dendritic ice, "
               "tiny water droplets, black background"),
]
from diffusers import Krea2Pipeline
pipe = Krea2Pipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload(gpu_id=0)   # transformer resident during denoise; TE / VAE swap
try:
    pipe.vae.enable_tiling()
except Exception:
    pass
T = []


def cb(p, i, t, kw):
    torch.cuda.synchronize(); T.append(time.perf_counter()); return kw


def render(tag, prompt, w, h, seed=7):
    T.clear()
    g = torch.Generator("cpu").manual_seed(seed)
    im = pipe(prompt=prompt, width=w, height=h, num_inference_steps=STEPS, guidance_scale=0.0,
              generator=g, callback_on_step_end=cb).images[0]
    im.save(os.path.join(OUT, tag + ".png"))
    d = np.diff(T)[1:] if len(T) > 2 else np.array([float("nan")])   # drop step-0 warmup
    return np.asarray(im, dtype=np.float32), float(np.median(d))


def psnr(a, b):
    m = float(((a - b) ** 2).mean()); return 99.0 if m == 0 else round(10 * np.log10(255 ** 2 / m), 2)


res = []
for sz, w, h in SIZES:
    for pn, pr in PROMPTS:
        if sz == "8.4MP" and pn != "portrait":
            continue                      # one 8.4 MP pair (time)
        row = {"size": sz, "prompt": pn}
        try:
            CP.apply_attention_backend(pipe.transformer, "auto", log=lambda *_: None)
            A, ta = render(f"{sz}_{pn}_flash", pr, w, h)
            got = CP.apply_attention_backend(pipe.transformer, "sage_fp8", log=print)
            B, tb = render(f"{sz}_{pn}_sagefp8", pr, w, h)
            row.update(backend=got, flash_s_step=round(ta, 3), sage_s_step=round(tb, 3),
                       speedup=round(ta / tb, 3), psnr=psnr(A, B))
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        torch.cuda.empty_cache()
        print(json.dumps(row), flush=True)
        res.append(row)
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
print("DONE")
