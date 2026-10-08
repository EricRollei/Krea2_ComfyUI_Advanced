"""Single-forward parity diagnostic: author's transformer (ref_kv_cache) vs ours (PaintRuntime)
on IDENTICAL inputs. Two phases (two processes, the 48 GB card can't hold both models):

  python paint_diag.py author CASE   -> saves noise, embeds, cond packed latent, known latent,
                                        velocity at the first timestep (+ one mid timestep)
  python paint_diag.py ours CASE     -> loads those, runs our runtime on the same tensors,
                                        reports cosine / rel-error of the velocity on generated
                                        tokens, noise equality, and condition-latent difference.
"""
import importlib.util
import json
import os
import sys
import types

import numpy as np
import torch
from PIL import Image

SRC = r"A:\ai-tools\scratch\paint_adapters_src"
TEST = r"A:\ai-tools\scratch\paint_test"
OUT = os.path.join(TEST, "out")
TURBO = r"H:\Training\Krea-2-Turbo"
LORA = r"L:\Models\loras\Krea2\paint\krea2_anypaint_rank32.safetensors"
PACK = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\custom_nodes\Eric_Krea2"
TS = [1.0, 0.6]


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def case(cid):
    man = {c["id"]: c for c in json.load(open(os.path.join(TEST, "manifest.json")))}
    return man[cid]


def author(cid):
    pl = load_mod("anypaint_pipeline", os.path.join(SRC, "krea2-anypaint__pipeline.py"))
    ap = load_mod("anypaint_helpers", os.path.join(SRC, "krea2-anypaint__anypaint.py"))
    c = case(cid)
    W, H = c["canvas_size"]
    pipe = pl.Krea2OstrisEditPipeline.from_pretrained(TURBO, torch_dtype=torch.bfloat16).to("cuda:0")
    pipe.load_lora_weights(LORA, adapter_name="anypaint")
    pipe.set_adapters(["anypaint"], weights=[1.0])
    dev = torch.device("cuda:0")
    prep = ap.prepare_anypaint(Image.open(os.path.join(TEST, c["source"])),
                               Image.open(os.path.join(TEST, c["mask"])), (W, H), tuple(c["bbox"]))
    e = torch.load(os.path.join(OUT, f"{cid}_embeds.pt"))
    emb, emask = e["embeds"].to(dev, torch.bfloat16), e["mask"].to(dev)
    g = torch.Generator("cpu").manual_seed(int(c["edit_seed"]))
    ch = pipe.transformer.config.in_channels // 4
    noise = pipe.prepare_latents(1, ch, H, W, torch.float32, dev, g, None)
    cond_t = pipe._to_chw_tensor(prep.condition)
    with torch.no_grad():
        ref_lat = pipe._encode_reference_latents([cond_t], 384 * 384, None, dev)  # generator None
        ref_tok, ref_pos = pipe._pack_reference_latents(ref_lat, dev, torch.bfloat16,
                                                        placements=[prep.reference_placement],
                                                        target_grid_size=(H // 16, W // 16))
        kv = pipe.transformer.precompute_ref_kv(ref_tok, ref_pos)
        pos = pipe.prepare_position_ids(emb.shape[1], H // 16, W // 16, dev)
        out = {"noise": noise.cpu(), "ref_tok": ref_tok.float().cpu(), "ref_pos": ref_pos.cpu(), "v": {}}
        for t in TS:
            ts = torch.full((1,), t, device=dev, dtype=torch.bfloat16)
            v = pipe.transformer(hidden_states=noise.to(torch.bfloat16), encoder_hidden_states=emb,
                                 timestep=ts, position_ids=pos, encoder_attention_mask=emask,
                                 ref_kv_cache=kv, return_dict=False)[0]
            out["v"][t] = v.float().cpu()
        # also: the same call WITHOUT the reference (to scale the diff)
        ts = torch.full((1,), TS[0], device=dev, dtype=torch.bfloat16)
        out["v_noref"] = pipe.transformer(hidden_states=noise.to(torch.bfloat16), encoder_hidden_states=emb,
                                          timestep=ts, position_ids=pos, encoder_attention_mask=emask,
                                          return_dict=False)[0].float().cpu()
    torch.save(out, os.path.join(OUT, f"{cid}_diag_author.pt"))
    print("author diag saved", {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)}, flush=True)


def ours(cid):
    sys.path.insert(0, os.path.abspath(os.path.join(PACK, "..", "..")))
    pkg = types.ModuleType("k2pack"); pkg.__path__ = [PACK]; sys.modules["k2pack"] = pkg
    nodes = types.ModuleType("k2pack.nodes"); nodes.__path__ = [os.path.join(PACK, "nodes")]
    sys.modules["k2pack.nodes"] = nodes
    import importlib
    P = importlib.import_module("k2pack._paint")
    N = importlib.import_module("k2pack.nodes.krea2_paint")
    LU = importlib.import_module("k2pack._lora_utils")
    from diffusers import Krea2Pipeline
    c = case(cid)
    W, H = c["canvas_size"]
    dev = torch.device("cuda:0")
    pipe = Krea2Pipeline.from_pretrained(TURBO, torch_dtype=torch.bfloat16).to(dev)
    LU.load_lora_with_key_fix(pipe, LORA, "anypaint", log_prefix="[diag-LoRA]", weight=1.0)
    A = torch.load(os.path.join(OUT, f"{cid}_diag_author.pt"))
    e = torch.load(os.path.join(OUT, f"{cid}_embeds.pt"))
    emb, emask = e["embeds"].to(dev, torch.bfloat16), e["mask"].to(dev)
    g = torch.Generator("cpu").manual_seed(int(c["edit_seed"]))
    ch = pipe.transformer.config.in_channels // 4
    noise = pipe.prepare_latents(1, ch, H, W, torch.float32, dev, g, None)
    rep = {"noise_equal": bool(torch.equal(noise.cpu(), A["noise"]))}
    src = torch.from_numpy(np.asarray(Image.open(os.path.join(TEST, c["source"])).convert("RGB"),
                                      dtype=np.float32) / 255.0)[None]
    msk = torch.from_numpy(np.asarray(Image.open(os.path.join(TEST, c["mask"])).convert("L"),
                                      dtype=np.float32) / 255.0)
    x0, y0, x1, y1 = c["bbox"]
    bundle = N.EricKrea2Paint().build(src, msk[None], pad_left=x0, pad_top=y0, pad_right=W - x1,
                                      pad_bottom=H - y1, restore_s1="off")[0]
    rt = P.PaintRuntime(pipe, bundle)
    rt.install()
    try:
        tok, posr = rt._ref_tokens_and_pos(H // 16, W // 16, dev, torch.bfloat16)
        rep["ref_pos_max_abs_diff"] = float((posr.cpu() - A["ref_pos"]).abs().max())
        rt_tok = tok.float().cpu()
        rep["ref_tok_rel_err"] = float((rt_tok - A["ref_tok"]).norm() / A["ref_tok"].norm())
        pos = pipe.prepare_position_ids(emb.shape[1], H // 16, W // 16, dev)
        with torch.no_grad():
            for t in TS:
                ts = torch.full((1,), t, device=dev, dtype=torch.bfloat16)
                v = pipe.transformer(hidden_states=A["noise"].to(dev, torch.bfloat16), encoder_hidden_states=emb,
                                     timestep=ts, position_ids=pos, encoder_attention_mask=emask,
                                     return_dict=False)[0].float().cpu()
                va = A["v"][t]
                cos = float(torch.nn.functional.cosine_similarity(v.flatten()[None], va.flatten()[None])[0])
                rep[f"t{t}_cos_vs_author"] = round(cos, 5)
                rep[f"t{t}_rel_err"] = round(float((v - va).norm() / va.norm()), 5)
            # also: our run with the AUTHOR's exact ref tokens (isolates encode differences)
            rt.b["cond_packed"] = (A["ref_tok"].to(torch.bfloat16), rt.b["cond_packed"][1], rt.b["cond_packed"][2])
            rt.kv_cache.clear()
            ts = torch.full((1,), TS[0], device=dev, dtype=torch.bfloat16)
            v2 = pipe.transformer(hidden_states=A["noise"].to(dev, torch.bfloat16), encoder_hidden_states=emb,
                                  timestep=ts, position_ids=pos, encoder_attention_mask=emask,
                                  return_dict=False)[0].float().cpu()
            va = A["v"][TS[0]]
            rep["author_reftok_cos"] = round(float(torch.nn.functional.cosine_similarity(
                v2.flatten()[None], va.flatten()[None])[0]), 5)
            rep["author_reftok_rel_err"] = round(float((v2 - va).norm() / va.norm()), 5)
            rep["ref_effect_rel (author with vs without ref)"] = round(float(
                (A["v"][TS[0]] - A["v_noref"]).norm() / A["v"][TS[0]].norm()), 5)
    finally:
        rt.remove()
    print("DIAG", json.dumps(rep, indent=1), flush=True)

    # full 8-step trajectory with an fp32 latent accumulator (the author's loop), our runtime
    import numpy as _np
    from PIL import Image as _Image
    LT = importlib.import_module("k2pack._latent_utils")
    bundle = N.EricKrea2Paint().build(src, msk[None], pad_left=x0, pad_top=y0, pad_right=W - x1,
                                      pad_bottom=H - y1)[0]
    rt = P.PaintRuntime(pipe, bundle)
    rt.install()
    try:
        sched = pipe.scheduler
        sched.set_timesteps(sigmas=_np.linspace(1.0, 1 / 8, 8), device=dev, mu=1.15)
        sched.set_begin_index(0)
        lat = A["noise"].to(dev).float()
        pos = pipe.prepare_position_ids(emb.shape[1], H // 16, W // 16, dev)
        with torch.no_grad():
            for t in sched.timesteps:
                ts = (t / sched.config.num_train_timesteps).expand(1).to(torch.bfloat16)
                v = pipe.transformer(hidden_states=lat.to(torch.bfloat16), encoder_hidden_states=emb,
                                     timestep=ts, position_ids=pos, encoder_attention_mask=emask,
                                     return_dict=False)[0]
                lat = sched.step(v.float(), t, lat, return_dict=False)[0]
    finally:
        rt.remove()
    img = LT.standard_decode(pipe, lat.to(torch.bfloat16), H, W)
    im = _Image.fromarray((img[0].numpy() * 255).round().astype(_np.uint8))
    im.save(os.path.join(OUT, f"{cid}_ours_fp32.png"))
    au = _np.asarray(_Image.open(os.path.join(OUT, f"{cid}_author.png")).convert("RGB"), dtype=_np.float32)
    ou = _np.asarray(im, dtype=_np.float32)
    mse = float(((au - ou) ** 2).mean())
    print("FULL_FP32", json.dumps({"psnr_vs_author": round(10 * _np.log10(255 ** 2 / mse), 2),
                                   "mad_vs_author": round(float(_np.abs(au - ou).mean()), 3)}), flush=True)


if __name__ == "__main__":
    {"author": author, "ours": ours}[sys.argv[1]](sys.argv[2])
