# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Headless check of the inpaint/outpaint runtime (_paint.py) on Krea-2 Turbo.

  1. bit-exact: runtime installed with adapter=none + restore off == stock; after removal == stock
  2. per showcase case: our runtime (stock diffusers pipeline + our LoRA loader + the AUTHOR's
     prompt embeddings, saved by paint_ref_run.py) vs the author's pipeline output
     -> PSNR / mean abs diff (whole image and kept region) + kept-token latent RMSE vs known
  3. grid: source | canvas preview | author | ours | showcase result

Usage: python test_paint.py CASE_ID [CASE_ID ...]
Author: Eric Hiss (GitHub: EricRollei)
"""

import importlib
import json
import os
import sys
import time
import types

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.abspath(os.path.join(HERE, "..", ".."))
COMFY = os.path.abspath(os.path.join(PACK, "..", ".."))
TEST = r"A:\ai-tools\scratch\paint_test"
OUT = os.path.join(TEST, "out")
TURBO = r"H:\Training\Krea-2-Turbo"
LORA = r"L:\Models\loras\Krea2\paint\krea2_anypaint_rank32.safetensors"


def load_pack():
    sys.path.insert(0, COMFY)
    pkg = types.ModuleType("k2pack")
    pkg.__path__ = [PACK]
    sys.modules["k2pack"] = pkg
    nodes = types.ModuleType("k2pack.nodes")
    nodes.__path__ = [os.path.join(PACK, "nodes")]
    sys.modules["k2pack.nodes"] = nodes
    return (importlib.import_module("k2pack._paint"), importlib.import_module("k2pack.nodes.krea2_paint"),
            importlib.import_module("k2pack._lora_utils"), importlib.import_module("k2pack._latent_utils"))


def F_cos(a, b):
    return torch.nn.functional.cosine_similarity(a[None], b[None])[0]


def to_t(im):
    return torch.from_numpy(np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0)[None]


def psnr(a, b, m=None):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    d = (a - b) ** 2
    mad = np.abs(a - b)
    if m is not None:
        d, mad = d[m], mad[m]
    mse = float(d.mean())
    return (99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)), float(mad.mean())


def main():
    cases = sys.argv[1:]
    P, N, LU, LT = load_pack()
    from diffusers import Krea2Pipeline
    man = {c["id"]: c for c in json.load(open(os.path.join(TEST, "manifest.json")))}
    t0 = time.perf_counter()
    pipe = Krea2Pipeline.from_pretrained(TURBO, torch_dtype=torch.bfloat16).to("cuda:0")
    print(f"pipeline loaded {time.perf_counter() - t0:.0f}s on {torch.cuda.get_device_name(0)}", flush=True)
    node = N.EricKrea2Paint()
    res = {}

    # 1) bit-exact pass-through (no LoRA yet)
    c0 = man[cases[0]]
    src0 = to_t(Image.open(os.path.join(TEST, c0["source"])))
    m0 = torch.zeros(src0.shape[1], src0.shape[2]); m0[:64, :64] = 1
    b_none = node.build(src0, m0[None], adapter="none", restore_s1="off", restore_s2="off",
                        restore_s3="off")[0]

    def t2i(seed=5):
        g = torch.Generator("cpu").manual_seed(seed)
        return pipe(prompt=c0["prompt"], height=1024, width=1024, num_inference_steps=8,
                    guidance_scale=0.0, generator=g).images[0]
    base = t2i()
    rt = P.PaintRuntime(pipe, b_none); rt.install()
    try:
        inst = t2i()
    finally:
        rt.remove()
    after = t2i()
    res["bitexact_installed"] = bool(np.array_equal(np.asarray(base), np.asarray(inst)))
    res["bitexact_after_remove"] = bool(np.array_equal(np.asarray(base), np.asarray(after)))
    print(f"bit-exact: installed(none/off)={res['bitexact_installed']} "
          f"after remove={res['bitexact_after_remove']}", flush=True)

    # 1b) our Vision Prompt node on our condition image vs the author's embeddings
    try:
        VP = importlib.import_module("k2pack.nodes.krea2_vision_prompt").EricKrea2VisionPrompt()
        for cid in cases:
            c = man[cid]
            W, H = c["canvas_size"]
            x0, y0, x1, y1 = c["bbox"]
            src = to_t(Image.open(os.path.join(TEST, c["source"])))
            msk = torch.from_numpy(np.asarray(Image.open(os.path.join(TEST, c["mask"])).convert("L"),
                                              dtype=np.float32) / 255.0)
            vlm = node.build(src, msk[None], pad_left=x0, pad_top=y0, pad_right=W - x1,
                             pad_bottom=H - y1)[1]
            cond = VP.encode({"pipeline": pipe, "is_distilled": True}, c["prompt"], image1=vlm,
                             print_prompt=False)[0]
            ours_e = cond["embeds"].float().cpu()
            ref_e = torch.load(os.path.join(OUT, f"{cid}_embeds.pt"))["embeds"].float()
            if ours_e.shape == ref_e.shape:
                cos = float(F_cos(ours_e.flatten(), ref_e.flatten()))
                res[f"{cid}_vp_embed"] = {"shape": list(ours_e.shape), "cosine": round(cos, 5),
                                          "max_abs": round(float((ours_e - ref_e).abs().max()), 4)}
            else:
                res[f"{cid}_vp_embed"] = {"shape_ours": list(ours_e.shape), "shape_author": list(ref_e.shape)}
            print(cid, "vision-prompt embeds vs author:", res[f"{cid}_vp_embed"], flush=True)
    except Exception as e:
        res["vp_embed_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        print("vision-prompt comparison skipped:", res["vp_embed_error"], flush=True)

    # 2) adapter cases (precomputed embeddings -> the text encoder can leave the GPU)
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    # test-only: keep the pipeline executing on the GPU with the encoder parked on the CPU
    pipe.__class__ = type("K2TestPipe", (pipe.__class__,),
                          {"_execution_device": property(lambda s: torch.device("cuda:0"))})
    LU.load_lora_with_key_fix(pipe, LORA, "anypaint", log_prefix="[test-LoRA]", weight=1.0)
    rows = []
    for cid in cases:
        c = man[cid]
        W, H = c["canvas_size"]
        src = to_t(Image.open(os.path.join(TEST, c["source"])))
        msk = torch.from_numpy(np.asarray(Image.open(os.path.join(TEST, c["mask"])).convert("L"),
                                          dtype=np.float32) / 255.0)
        x0, y0, x1, y1 = c["bbox"]
        bundle, vlm, prev, gen = node.build(src, msk[None], adapter="anypaint", pad_left=x0, pad_top=y0,
                                            pad_right=W - x1, pad_bottom=H - y1)
        ref_cond = Image.open(os.path.join(OUT, f"{cid}_condition.png"))
        ours_cond = Image.fromarray((vlm[0].numpy() * 255).round().astype(np.uint8))
        cond_psnr = (psnr(ref_cond, ours_cond) if ref_cond.size == ours_cond.size else ("size", ref_cond.size,
                                                                                        ours_cond.size))
        e = torch.load(os.path.join(OUT, f"{cid}_embeds.pt"))
        emb = e["embeds"].to("cuda:0", torch.bfloat16)
        emask = e["mask"].to("cuda:0")
        rt = P.PaintRuntime(pipe, bundle); rt.install()
        g = torch.Generator("cpu").manual_seed(int(c["edit_seed"]))
        t = time.perf_counter()
        try:
            lat = pipe(prompt_embeds=emb, prompt_embeds_mask=emask, height=H, width=W,
                       num_inference_steps=8, guidance_scale=0.0, generator=g, output_type="latent").images
            known, keep = rt._known_for(H // 16, W // 16, lat.device)
            k_rmse = float(((lat.float() - known)[:, keep] ** 2).mean().sqrt())
            k_scale = float(known[:, keep].float().pow(2).mean().sqrt())
        finally:
            rt.remove()
        dt = time.perf_counter() - t
        ours = LT.standard_decode(pipe, lat, H, W)
        ours_im = Image.fromarray((ours[0].float().cpu().numpy() * 255).round().astype(np.uint8))
        ours_im.save(os.path.join(OUT, f"{cid}_ours.png"))
        author = Image.open(os.path.join(OUT, f"{cid}_author.png")).convert("RGB")
        show = Image.open(os.path.join(TEST, c["result"])).convert("RGB")
        keep_px = (gen[0].numpy() < 0.5)
        r = {"time_s": round(dt, 1), "cond_psnr_mad": cond_psnr,
             "ours_vs_author_psnr_mad": psnr(ours_im, author),
             "ours_vs_author_kept_psnr_mad": psnr(ours_im, author, keep_px),
             "author_vs_showcase_psnr_mad": psnr(author, show),
             "ours_vs_showcase_psnr_mad": psnr(ours_im, show),
             "ours_vs_source_kept_psnr_mad": psnr(ours_im, Image.fromarray(
                 (bundle["known"][0].numpy() * 255).round().astype(np.uint8)), keep_px),
             "kept_token_latent_rmse": round(k_rmse, 5), "known_latent_rms": round(k_scale, 4)}
        res[cid] = r
        print(cid, json.dumps(r), flush=True)
        prev_im = Image.fromarray((prev[0].numpy() * 255).round().astype(np.uint8))
        rows.append([Image.open(os.path.join(TEST, c["source"])).convert("RGB"), prev_im, author, ours_im, show])

    # 3) grid (each cell fit to 512 px)
    cell = 512
    grid = Image.new("RGB", (cell * 5, cell * len(rows)), (24, 24, 24))
    for r_i, row in enumerate(rows):
        for c_i, im in enumerate(row):
            im = im.copy(); im.thumbnail((cell, cell))
            grid.paste(im, (c_i * cell + (cell - im.width) // 2, r_i * cell + (cell - im.height) // 2))
    grid.save(os.path.join(OUT, "paint_parity_grid.jpg"), quality=88)
    json.dump(res, open(os.path.join(OUT, "paint_parity.json"), "w"), indent=1)
    print("RESULT", "PASS" if res["bitexact_installed"] and res["bitexact_after_remove"] else "CHECK")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
