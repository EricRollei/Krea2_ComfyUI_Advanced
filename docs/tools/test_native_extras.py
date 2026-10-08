"""Native compute loose ends (2026-10-08): LoKr on a native (int8_convrot) transformer
(PEFT path AND forced direct-merge path), unload -> exact baseline, transformer .to('cpu') and
back, Unload-node style teardown (.to('meta')) frees VRAM. Turbo 1 MP, 8 steps, seed 11.
Usage: python test_native_extras.py [LOKR_PATH]"""
import importlib, os, sys, types, time
import numpy as np
import torch
PACK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.abspath(os.path.join(PACK, "..", "..")))
pkg = types.ModuleType("k2p"); pkg.__path__ = [PACK]; sys.modules["k2p"] = pkg
NQ = importlib.import_module("k2p._native_quant")
CP = importlib.import_module("k2p._compat")
LU = importlib.import_module("k2p._lora_utils")
LOKR = sys.argv[1] if len(sys.argv) > 1 else r"L:\Models\loras\Krea2\nsfw\Positions\2777586-Bondage_Bench_for_Krea2\3127858-v1_0\3127858_bondagebenchkrea2lokr.safetensors"
OUT = r"A:\ai-tools\scratch\native_extras"; os.makedirs(OUT, exist_ok=True)
PROMPT = "a wooden bench in a sunlit stone courtyard, ivy on the walls, photo"
from diffusers import Krea2Pipeline
pipe = Krea2Pipeline.from_pretrained(r"H:\Training\Krea-2-Turbo", torch_dtype=torch.bfloat16).to("cuda:0")
CP.apply_attention_backend(pipe.transformer, "auto", log=lambda *_: None)


def render(tag):
    g = torch.Generator("cpu").manual_seed(11)
    im = pipe(prompt=PROMPT, width=1024, height=1024, num_inference_steps=8, guidance_scale=0.0,
              generator=g).images[0]
    im.save(os.path.join(OUT, tag + ".png"))
    return np.asarray(im, dtype=np.float32)


def psnr(a, b):
    m = float(((a - b) ** 2).mean()); return 99.0 if m == 0 else round(10 * np.log10(255 ** 2 / m), 2)


res = {}
A = render("a_bf16")
LU.load_lora_with_key_fix(pipe, LOKR, "lk", log_prefix="[t]", weight=1.0)
B = render("b_bf16_lokr"); LU.unload_all_loras(pipe)
res["bf16 lokr effect (psnr vs base, lower = stronger)"] = psnr(B, A)
n = NQ.convert_transformer(pipe.transformer, "int8_convrot", 256, log=print)
C = render("c_native")
res["native vs bf16 base"] = psnr(C, A)
LU.load_lora_with_key_fix(pipe, LOKR, "lk", log_prefix="[t]", weight=1.0)
peft_like = sum(1 for m in pipe.transformer.modules() if type(m).__name__.lower().startswith("lokr"))
D = render("d_native_lokr_peft")
res["native+lokr(PEFT) vs bf16+lokr"] = psnr(D, B)
res["native lokr effect vs native base"] = psnr(D, C)
LU.unload_all_loras(pipe)
E = render("e_native_after_unload")
res["after unload == native base (bit-exact)"] = bool((E == C).all())
# forced direct-merge path (the PEFT-failure fallback): side-term deltas on QuantLinear
from safetensors.torch import load_file
sd = LU._remap_krea_lycoris_keys(load_file(LOKR), "[t]")
try:
    LU._load_lokr_adapter_direct(pipe, sd, "lkd", "[t]", weight=1.0)
    nd = sum(1 for m in pipe.transformer.modules() if m.__dict__.get("_eric_wdelta") is not None)
    F_ = render("f_native_lokr_direct")
    res["direct-merge modules with side term"] = nd
    res["native+lokr(direct) vs native+lokr(PEFT)"] = psnr(F_, D)
    LU.unload_all_loras(pipe)
    G = render("g_native_after_direct_unload")
    res["after direct unload == native base"] = bool((G == C).all())
except Exception as e:
    res["direct-merge path"] = f"FAILED {type(e).__name__}: {str(e)[:200]}"
# .to('cpu') and back
torch.cuda.synchronize(); a0 = torch.cuda.memory_allocated() / 1e9
try:
    pipe.transformer.to("cpu"); torch.cuda.empty_cache()
    a1 = torch.cuda.memory_allocated() / 1e9
    pipe.transformer.to("cuda:0")
    H = render("h_native_after_cpu_roundtrip")
    res[".to(cpu) frees GB"] = round(a0 - a1, 1)
    res["after cpu round-trip == native base"] = bool((H == C).all())
except Exception as e:
    res[".to(cpu) round-trip"] = f"FAILED {type(e).__name__}: {str(e)[:200]}"
# Unload-node teardown
tr = pipe.transformer
try:
    tr.to("meta")
except Exception as e:
    res["to(meta)"] = f"raised {type(e).__name__} (Unload node swallows it)"
pipe.transformer = None; del tr
import gc; gc.collect(); torch.cuda.empty_cache()
res["VRAM after teardown GB (pipe VAE+TE remain)"] = round(torch.cuda.memory_allocated() / 1e9, 1)
for k, v in res.items():
    print(f"{k}: {v}")
print("DONE")
