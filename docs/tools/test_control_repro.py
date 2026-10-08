"""Repro: control mosaic in Ultra's stock pipe(...) S1 path. Real depth map (cb_depth from the
ComfyUI run), knight prompt, Turbo 10 steps, attention auto. Renders: square 1024 pipe, 1376x1088
pipe, 1376x1088 pipe with sigmas=linear list (as Ultra passes). Usage: python test_control_repro.py"""
import importlib, os, sys, types, glob
import numpy as np, torch
from PIL import Image
PACK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.abspath(os.path.join(PACK, "..", "..")))
pkg = types.ModuleType("k2r"); pkg.__path__ = [PACK]; sys.modules["k2r"] = pkg
C = importlib.import_module("k2r._control"); CP = importlib.import_module("k2r._compat")
OUT = r"A:\ai-tools\scratch\control_repro"; os.makedirs(OUT, exist_ok=True)
dep = sorted(glob.glob(r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\output\cb_depth_*.png"))[-1]
img = torch.from_numpy(np.asarray(Image.open(dep).convert("RGB"), dtype=np.float32) / 255.0)[None]
img = C.preprocess_control_image(img, "grayscale", "per_image_minmax", False)
from diffusers import Krea2Pipeline
pipe = Krea2Pipeline.from_pretrained(r"H:\Training\Krea-2-Turbo", torch_dtype=torch.bfloat16).to("cuda:0")
CP.apply_attention_backend(pipe.transformer, "auto", log=print)
P = ("a knight in silver armor standing on dark wet rocks by a stormy sea, a stone castle tower on "
     "the cliff behind him, dusk, waves crashing, photo")
rt = C.ControlRuntime(pipe, {"path": r"L:\Models\loras\Krea2\control\krea2_depth_control_lora_patil.safetensors",
                             "image": img, "s1": 1.0, "s2": 0.0, "s3": 0.0})
rt.install()
for tag, w, h, sig in (("sq1024", 1024, 1024, None), ("1376x1088", 1376, 1088, None),
                       ("1376x1088_sigmas", 1376, 1088, [1.0 - i / 10 for i in range(10)])):
    g = torch.Generator("cuda:0").manual_seed(11)
    kw = dict(sigmas=sig) if sig else dict(num_inference_steps=10)
    im = pipe(prompt=P, width=w, height=h, guidance_scale=0.0, generator=g, **kw).images[0]
    im.save(os.path.join(OUT, tag + ".png")); print("saved", tag, flush=True)
rt.remove()
print("DONE")
