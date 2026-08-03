# Copyright (c) 2026 Eric Hiss. All rights reserved.
"""
Euler S2/S3 parity A/B (2026-07-25)
===================================
Settles the question: does the post-2026-07-22 solver path change euler
refine-window results vs the old diffusers-pipe loop at DEFAULT settings
(fixed mu 1.15, eta 0), and if so, is it the fp32-vs-bf16 accumulator?

Static audit already verified identical: update rule, timestep encoding
(shifted sigma), raw sigma ladder (linspace(1, 1/N, N)), fixed-policy mu,
cfg gating convention, and the guidance formula. The ONE structural
difference left is the accumulator: the pipe loop holds latents in bf16 and
re-quantizes every step (scheduler.step upcasts->steps->downcasts); the
solver holds fp32 across the window. This script isolates that.

Three arms, same model, same window (steps 7->20 of a 20-step fixed-mu
schedule), same re-noised latent, same everything:
  A  old pipe-style loop: bf16 latents + scheduler.step per step
  B  current solver path: _res_solver rk euler (fp32 accumulator)
  B2 solver formulation but bf16-requantized each step (isolates precision
     from formulation: B2==A within noise -> formulation identical, and
     A-vs-B is purely the precision dither)

Run with ComfyUI CLOSED (needs the VRAM):
  A:\\Comfy25\\ComfyUI_windows_portable\\python_embeded\\python.exe ^
      A:\\...\\Eric_Krea2\\docs\\tools\\euler_parity_ab.py [--base <pipeline dir>]

--base defaults to the loader.base_pipeline_path found in the newest
output/sweeps cell PNG (written since the provenance update). Outputs land in
docs/tools/euler_parity_out/: A.png, B.png, B2.png, diff stats on stdout.
"""

import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np
import torch


def find_base_from_sweeps():
    out = os.path.join(os.path.dirname(os.path.dirname(REPO)), "output", "sweeps")
    # repo = .../ComfyUI/custom_nodes/Eric_Krea2 -> output is .../ComfyUI/output
    out = os.path.abspath(os.path.join(REPO, "..", "..", "output", "sweeps"))
    from PIL import Image
    for d in sorted(glob.glob(os.path.join(out, "*")), reverse=True):
        for f in sorted(glob.glob(os.path.join(d, "cell_*.png"))):
            try:
                doc = json.loads(Image.open(f).text.get("krea2_settings", "{}"))
                p = doc.get("loader", {}).get("base_pipeline_path")
                if p and os.path.isdir(p):
                    return p
            except Exception:
                continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="diffusers Krea2 pipeline folder")
    ap.add_argument("--prompt", default="a lighthouse on a rocky coast at golden hour, "
                                        "crashing waves, dramatic clouds")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mp", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--start", type=int, default=7)
    args = ap.parse_args()

    base = args.base or find_base_from_sweeps()
    if not base:
        sys.exit("no --base given and none found in recent sweep PNGs")
    print(f"[parity] pipeline: {base}")

    from diffusers import Krea2Pipeline
    import _res_solver as rs

    dev = "cuda"
    pipe = Krea2Pipeline.from_pretrained(base, torch_dtype=torch.bfloat16).to(dev)
    pipe.set_progress_bar_config(disable=True)

    # geometry for args.mp at 1:1
    side = int(round((args.mp * 1_000_000) ** 0.5 / 16)) * 16
    H = W = side
    print(f"[parity] {W}x{H}, steps {args.start}->{args.steps} of {args.steps}, "
          f"mu fixed 1.15, guidance 0 (single-cond - guidance formula already "
          f"verified identical; keeping it off halves eval count)")

    emb, mask = pipe.encode_prompt(prompt=args.prompt, device=dev, num_images_per_prompt=1)
    dtype = emb.dtype
    num_ch = pipe.transformer.config.in_channels // (pipe.patch_size ** 2)
    g0 = torch.Generator(device=dev).manual_seed(args.seed)
    x0_lat = pipe.prepare_latents(1, num_ch, H, W, dtype, dev, g0, None)
    grid_h = H // (pipe.vae_scale_factor * pipe.patch_size)
    grid_w = W // (pipe.vae_scale_factor * pipe.patch_size)
    pos = pipe.prepare_position_ids(emb.shape[1], grid_h, grid_w, dev)

    def call_model(lat, sigma):
        ts = sigma.to(dtype).expand(lat.shape[0])
        return pipe.transformer(hidden_states=lat.to(dtype), encoder_hidden_states=emb,
                                timestep=ts, position_ids=pos,
                                encoder_attention_mask=mask, attention_kwargs=None,
                                return_dict=False)[0]

    # one shared "stage output" to refine: quick 8-step euler full pass
    raw8 = np.linspace(1.0, 1.0 / 8, 8).tolist()
    pipe.scheduler.set_timesteps(sigmas=raw8, device=dev, mu=1.15)
    sig8 = pipe.scheduler.sigmas.to(torch.float32)
    lat = x0_lat.to(torch.float32)
    with torch.no_grad():
        for i in range(len(sig8) - 1):
            v = call_model(lat, sig8[i]).to(torch.float32)
            lat = lat + (sig8[i + 1] - sig8[i]) * v
    clean = lat  # fp32 "previous stage" latent

    # shared window + shared re-noise
    rawN = np.linspace(1.0, 1.0 / args.steps, args.steps).tolist()
    pipe.scheduler.set_timesteps(sigmas=rawN, device=dev, mu=1.15)
    sigN = pipe.scheduler.sigmas.to(torch.float32)
    window = sigN[args.start:].clone()
    s0 = window[0]
    gn = torch.Generator(device=dev).manual_seed(args.seed + 1)
    noise = torch.randn(clean.shape, device=dev, dtype=torch.float32, generator=gn)
    noised = (1.0 - s0) * clean + s0 * noise

    # Chaos-control arm: the SAME pipe-style loop as A, but with one tiny
    # bf16-ulp-scale perturbation at entry. After 13 steps through a 12B
    # transformer, its distance from A is the pure chaos-amplification floor:
    # if it lands at the same distance as the solver arms, precision-seeded
    # divergence explains everything and the formulation is exonerated.
    results = {}
    with torch.no_grad():
        # A: old pipe-style loop - bf16 latents, scheduler.step each step
        pipe.scheduler.set_timesteps(sigmas=rawN, device=dev, mu=1.15)
        pipe.scheduler.set_begin_index(args.start)
        latA = noised.to(torch.bfloat16)
        tsteps = pipe.scheduler.timesteps[args.start:]
        for t in tsteps:
            v = pipe.transformer(hidden_states=latA, encoder_hidden_states=emb,
                                 timestep=(t / pipe.scheduler.config.num_train_timesteps)
                                 .expand(latA.shape[0]).to(latA.dtype),
                                 position_ids=pos, encoder_attention_mask=mask,
                                 attention_kwargs=None, return_dict=False)[0]
            latA = pipe.scheduler.step(v, t, latA, return_dict=False)[0]
        results["A_pipe_bf16"] = latA.to(torch.float32)

        # A_perturbed: chaos control - identical loop, one-shot 1e-3 nudge
        pipe.scheduler.set_timesteps(sigmas=rawN, device=dev, mu=1.15)
        pipe.scheduler.set_begin_index(args.start)
        gp = torch.Generator(device=dev).manual_seed(args.seed + 99)
        latP = (noised + 1e-3 * torch.randn(noised.shape, device=dev,
                                            dtype=torch.float32, generator=gp)
                ).to(torch.bfloat16)
        for t in tsteps:
            v = pipe.transformer(hidden_states=latP, encoder_hidden_states=emb,
                                 timestep=(t / pipe.scheduler.config.num_train_timesteps)
                                 .expand(latP.shape[0]).to(latP.dtype),
                                 position_ids=pos, encoder_attention_mask=mask,
                                 attention_kwargs=None, return_dict=False)[0]
            latP = pipe.scheduler.step(v, t, latP, return_dict=False)[0]
        results["A2_chaos_control"] = latP.to(torch.float32)

        # B: current solver path - fp32 accumulator, x0-form euler
        def denoise_fn(xx, sigma):
            v = call_model(xx, sigma)
            return xx.to(dtype) - sigma.to(dtype) * v  # x0 form, as in the node

        results["B_solver_fp32"] = rs.rk_explicit_sample(
            denoise_fn, noised.clone(), window.clone(), tableau="euler", eta=0.0)

        # B2: solver formulation, bf16-requantized accumulator each step
        latB2 = noised.clone()
        for i in range(len(window) - 1):
            x0p = denoise_fn(latB2, window[i]).to(torch.float32)
            d = (latB2 - x0p) / window[i]
            latB2 = latB2 + (window[i + 1] - window[i]) * d
            latB2 = latB2.to(torch.bfloat16).to(torch.float32)  # per-step quantization
        results["B2_solver_bf16steps"] = latB2

    print("\n[parity] pairwise latent deltas (mean-relative):")
    keys = list(results.keys())
    denom = results["A_pipe_bf16"].abs().mean()
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            d = (results[ka] - results[kb]).abs()
            print(f"  {ka:20s} vs {kb:20s} max {d.max().item():.6f}  "
                  f"rel {(d.mean() / denom).item():.6f}")
    print("\n[parity] verdict guide: if A2_chaos_control sits at the SAME rel "
          "distance from A as the solver arms (~all pairs similar), the gap is "
          "pure chaos amplification of rounding noise - formulation exonerated. "
          "If A-vs-A2 is MUCH smaller than A-vs-B, a systematic difference is "
          "real - report the numbers.")

    outdir = os.path.join(REPO, "docs", "tools", "euler_parity_out")
    os.makedirs(outdir, exist_ok=True)
    from PIL import Image
    for k, v in results.items():
        # Mirror of Krea2Pipeline.__call__'s decode block (Qwen image VAE:
        # per-channel latents_mean/std denorm, 5D (B,C,T,H,W), take frame 0).
        try:
            lat_un = pipe._unpack_latents(v.to(dtype), H, W)
            lat_un = lat_un.to(pipe.vae.dtype)
            lm = (torch.tensor(pipe.vae.config.latents_mean)
                  .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                  .to(lat_un.device, lat_un.dtype))
            ls = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
                1, pipe.vae.config.z_dim, 1, 1, 1).to(lat_un.device, lat_un.dtype)
            lat_un = lat_un / ls + lm
            img = pipe.vae.decode(lat_un, return_dict=False)[0][:, :, 0]
        except Exception as e:
            print(f"  (decode skipped for {k}: {e})")
            continue
        arr = ((img[0].float().permute(1, 2, 0).cpu().numpy().clip(-1, 1) + 1) * 127.5)
        Image.fromarray(arr.astype("uint8")).save(os.path.join(outdir, f"{k}.png"))
        print(f"  wrote {k}.png")
    print(f"\n[parity] interpretation: B2~=A (rel < ~1e-3) means the formulation is "
          f"identical and any A-vs-B gap is purely the per-step bf16 dither the old "
          f"path added. Large B2-vs-A gaps would mean a real formulation bug - report it.")


if __name__ == "__main__":
    main()
