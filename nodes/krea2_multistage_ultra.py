# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.

"""
Eric Krea2 Multi-Stage Generate
==============================
Progressive high-res generation: draft -> bislerp latent upscale -> partial
re-noise -> re-denoise, up to 3 stages. Ports the proven Qwen UltraGen multistage
to Krea 2 (same FlowMatchEulerDiscreteScheduler with dynamic shifting).

Per-stage control of step *spacing* (schedule) and re-noise *spectrum* (noise):
stage 1 forms composition (linear schedule, low-frequency noise), later stages
add progressive detail only (karras schedule, high-frequency noise that perturbs
fine detail while leaving the established structure intact).

Intended for the **Raw/base** checkpoint (full CFG, resolution-aware shift). The
distilled **Turbo** checkpoint uses a fixed mu=1.15 and is distilled for guidance
0, so guidance is off on Turbo by default; the `turbo_guidance` toggle
(off / s1_only / all_stages) can re-enable low-g CFG - and with it the negative
prompt - per stage. For the final high-res step on Turbo, the spacepxl 2x
upscale-decode is the cleanest path: decode at 2x for a real resolution gain, or
enable the loader's `downsample_to_1x` (optionally with `blur_sigma`) to hold the
generation's native size while taking the upscaler's sharper/GAN-trained detail as
a supersampling pass - which lands even crisper than a plain Wan2.1-VAE decoder
finetune, just without the resolution boost. Official samplers: Turbo 8 steps /
guidance 0; Raw 28-52 steps / cfg ~3.5.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

from .._latent_utils import (
    _pack_latents, _upscale_latents, _add_noise_flowmatch, _check_cancelled, standard_decode,
)
from .._upscale_vae import upscale_between_stages, decode_latents_with_upscale_vae
from .._res_solver import (res_multistep_sample, res_2s_sample, deis_sample,
                           rk_explicit_sample, lcm_sample, lcm_hybrid_sample,
                           abnorsett_sample, lawson4_sample, etdrk4_sample,
                           er_sde_sample)
from .. import _sigmas
from .. import _telemetry

_VAE_SCALE_FACTOR = 8  # AutoencoderKLQwenImage is f8
_SCHEDULES = ["linear", "balanced", "karras", "beta57", "beta", "bong_tangent",
              "exponential", "linear_quadratic", "simple"]
_SAMPLERS = ["euler", "res_2m", "res_2s", "deis_3m", "abnorsett_3m", "abnorsett_4m",
             "lawson4_4s", "etdrk4_4s", "er_sde", "rk6_7s", "lcm", "lcm_hybrid",
             "lcm_hybrid2"]
_NOISE_TYPES = ["white", "low_freq", "high_freq", "pink"]
_NOISE_STRENGTH = 0.30  # how far non-white noise leans from white (kept low so the
                        # flow-match denoiser, trained on white noise, can still clean it)

# Upscale-VAE routing. `upscale_vae_mode` maps to explicit per-transition flags so the
# behaviour is unambiguous: (s2s3_vae, s2s3_downsample, final_vae, final_downsample).
#  - s2s3_vae         : S2->S3 jump uses the trained 2x VAE (decode-2x + re-encode).
#  - s2s3_downsample  : after the 2x decode, resample down to the upscale_to_stage3
#                       target (keeps the VAE's detail at YOUR chosen size instead of a
#                       forced 4x area). Off => forced 2x linear (4x area).
#  - final_vae        : final image decoded with the 2x upscale VAE.
#  - final_downsample : that final 2x decode is Lanczos-downsampled back to native
#                       (a supersample quality pass, not a resolution gain).
# S1->S2 has its OWN switch (s1_s2_upscale_vae) - it is deliberately NOT part of this
# dropdown (using the VAE that early is a specialist move; see its tooltip).
# Legacy aliases (inter_stage/final_decode) keep older saved graphs working.
_UPSCALE_VAE_MODE_FLAGS = {
    # mode                                (s2s3_vae, s2s3_down, final_vae, final_down)
    "disabled":                           (False, False, False, False),
    "s2-s3":                              (True,  False, False, False),
    "s2-s3 with downsample":              (True,  True,  False, False),
    "final decode":                       (False, False, True,  False),
    "final decode with downsample":       (False, False, True,  True),
    "both":                               (True,  False, True,  False),
    "both with downsample":               (True,  True,  True,  True),
    "both with final decode downsample":  (True,  False, True,  True),
    # legacy aliases
    "inter_stage":                        (True,  False, False, False),
    "final_decode":                       (False, False, True,  False),
}
_UPSCALE_VAE_MODES = ["disabled", "s2-s3", "s2-s3 with downsample",
                      "final decode", "final decode with downsample",
                      "both", "both with downsample", "both with final decode downsample"]

# Aspect ratios, widest -> tallest.
_ASPECT_RATIOS = {
    "21:9 ultrawide": (21, 9),
    "16:9 wide": (16, 9),
    "3:2 landscape": (3, 2),
    "4:3 landscape": (4, 3),
    "5:4 landscape": (5, 4),
    "1:1 square": (1, 1),
    "4:5 portrait": (4, 5),
    "3:4 portrait": (3, 4),
    "2:3 portrait": (2, 3),
    "9:16 tall": (9, 16),
    "9:21 tall": (9, 21),
}


def _round16(v: float) -> int:
    return max(16, int(round(v / 16)) * 16)


def _dims_from_ratio_mp(ratio_key: str, megapixels: float):
    w_r, h_r = _ASPECT_RATIOS.get(ratio_key, (1, 1))
    target = max(megapixels, 0.05) * 1_000_000.0
    aspect = w_r / h_r
    h = (target / aspect) ** 0.5
    w = h * aspect
    return _round16(w), _round16(h)


def _resolve_s1_dims(aspect_ratio, s1_megapixels, width, height, init_latent, init_match_size,
                     verbose: bool = True):
    """Resolve Stage 1's target (w, h), same precedence used inside ``_generate_inner``:
    aspect_ratio+s1_megapixels < explicit width/height < init_latent (when init_match_size).
    Factored out so ``generate()`` can resolve it early (before the heavy denoise loop) to
    size-match ``ref_latents``."""
    s1_w, s1_h = _dims_from_ratio_mp(aspect_ratio, s1_megapixels)
    if int(width) > 0 and int(height) > 0:
        s1_w, s1_h = _round16(int(width)), _round16(int(height))
        if verbose:
            print(f"[EricKrea2-MS] Stage 1 size overridden by width/height inputs -> {s1_w}x{s1_h} "
                  "(aspect_ratio + s1_megapixels ignored)")
    if init_latent is not None and init_match_size:
        s1_w = _round16(int(init_latent.get("width", s1_w)))
        s1_h = _round16(int(init_latent.get("height", s1_h)))
        if verbose:
            print(f"[EricKrea2-MS] img2img: Stage 1 size matched to init latent -> {s1_w}x{s1_h}")
    return s1_w, s1_h


def _scaled_dims(w: int, h: int, area_factor: float):
    """New WxH for an area scale factor, rounded to multiples of 16 (keeps aspect)."""
    s = math.sqrt(max(area_factor, 1e-6))
    return _round16(w * s), _round16(h * s)


# ── spectral noise ─────────────────────────────────────────────────────────
# Re-noise spectrum control. We shape noise in the spatial latent domain, then
# renormalize to unit variance (so the flow-match level is preserved) and pack.
# white     - flat spectrum (standard Gaussian).
# low_freq  - energy concentrated at low spatial frequency: biases large coherent
#             structure. Good for the composition stage.
# high_freq - energy at high spatial frequency: perturbs fine detail while leaving
#             low-frequency structure (the composition) intact. Good for refine.
# pink      - 1/f amplitude falloff: natural-image-like, broadband but low-weighted.

def _freq_filter(n: torch.Tensor, noise_type: str) -> torch.Tensor:
    B, C, H, W = n.shape
    fy = torch.fft.fftfreq(H, device=n.device)
    fx = torch.fft.fftfreq(W, device=n.device)
    rad = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    r = rad / rad.max().clamp(min=1e-8)  # 0 (DC) .. 1
    if noise_type == "low_freq":
        filt = torch.exp(-((r / 0.35) ** 2))
    elif noise_type == "high_freq":
        filt = 1.0 - torch.exp(-((r / 0.20) ** 2))
    elif noise_type == "pink":
        filt = 1.0 / (r + 1e-2)  # amplitude ~ 1/f -> power ~ 1/f², the actual natural-image
                                # statistic (Field 1987). The previous 1/sqrt(r) was amplitude
                                # ~ 1/f^0.5 -> power ~ 1/f, which is classic 1D "pink noise"
                                # (audio), not the 2D natural-image statistic the docstring
                                # actually claims.
    else:
        return n
    F = torch.fft.fft2(n) * filt[None, None]
    shaped = torch.fft.ifft2(F).real
    shaped = shaped - shaped.mean(dim=(-2, -1), keepdim=True)
    shaped = shaped / shaped.std(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    # Blend toward white. Pure colored noise is off-distribution for the
    # flow-match denoiser and survives denoising as structured chroma (rainbow
    # ribbons), so we keep mostly white and only lean the spectrum.
    out = (1.0 - _NOISE_STRENGTH) * n + _NOISE_STRENGTH * shaped
    out = out - out.mean(dim=(-2, -1), keepdim=True)
    out = out / out.std(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    return out


def _shaped_packed_noise(b, h_lat, w_lat, noise_type, generator, device):
    """Packed [b, seq, 64] noise with the requested spatial-frequency spectrum."""
    n = torch.randn(b, 16, h_lat, w_lat, generator=generator, device=device, dtype=torch.float32)
    if noise_type and noise_type != "white":
        n = _freq_filter(n, noise_type)
    return _pack_latents(n.unsqueeze(2))


# ── scheduler-aware sigma helpers (mirror Krea2Pipeline.__call__) ──────────

def _packed_seq_len(height: int, width: int, vsf: int) -> int:
    h_lat = 2 * (int(height) // (vsf * 2))
    w_lat = 2 * (int(width) // (vsf * 2))
    return (h_lat // 2) * (w_lat // 2)


def _renorm_hop(dst, ref, mode, tag):
    """Inter-stage hop statistics repair + flatness telemetry (2026-07-23).

    Latent interpolation is a lowpass: it shrinks latent variance, and decoded
    variance IS contrast/saturation - each hop drains a little, and multi-stage
    chains compound it into the progressive flattening/desaturation Eric
    observed. Modes: "off" (previous behavior, telemetry only), "global" (match
    the pre-hop latent's overall mean/std), "per_channel" (match per-channel
    mean/std - the strongest color/contrast hold). Handles packed (B, seq, C;
    channel LAST) and unpacked (B, C, H, W) layouts. Fail-silent by design.
    """
    try:
        def _stats(t, per_ch):
            if not per_ch:
                return t.mean(), t.std()
            if t.dim() == 4:                    # (B, C, H, W)
                return (t.mean(dim=(0, 2, 3), keepdim=True),
                        t.std(dim=(0, 2, 3), keepdim=True))
            return (t.mean(dim=(0, 1), keepdim=True),   # (B, seq, C)
                    t.std(dim=(0, 1), keepdim=True))

        pre = float(ref.float().std())
        post = float(dst.float().std())
        m = str(mode or "off").lower()
        if m in ("global", "per_channel"):
            rmu, rsd = _stats(ref.float(), m == "per_channel")
            dmu, dsd = _stats(dst.float(), m == "per_channel")
            out = ((dst.float() - dmu) / (dsd + 1e-8) * rsd + rmu).to(dst.dtype)
            print(f"[EricKrea2-MS]   latent std {tag}: {pre:.4f} -> hop {post:.4f} -> "
                  f"renorm({m}) {float(out.float().std()):.4f}")
            return out
        print(f"[EricKrea2-MS]   latent std {tag}: {pre:.4f} -> hop {post:.4f} (renorm off)")
        return dst
    except Exception:
        return dst


def _compute_mu(seq_len: int, scheduler, is_distilled: bool, policy=None, stage=None) -> float:
    """Flow-matching shift exponent (mu) for one stage.

    Raw/base: the scheduler-config linear rule in seq_len (resolution-aware, as
    Krea2Pipeline.__call__ does) - the `policy` is ignored.

    Distilled/Turbo: distillation baked a FIXED mu=1.15 (calibrated <=2K /
    ~max_image_seq_len tokens). Above that band a fixed shift under-shifts the
    schedule and upscaled stages keep mid-frequency mottling (splotch bug,
    diagnosed 2026-07-21). `policy` comes from the distilled_shift widgets:
      {"mode": "fixed"|"resolution"|"manual", "mu": {"s1":f, "s2":f, "s3":f}}
    with `stage` naming the current stage. When no policy is threaded (external
    caller), the KREA2_DISTILLED_SHIFT env var is honored as a debug fallback
    ("fixed"/"resolution"/<number>)."""
    base_seq = scheduler.config.get("base_image_seq_len", 256)
    max_seq = scheduler.config.get("max_image_seq_len", 6400)
    base_shift = scheduler.config.get("base_shift", 0.5)
    max_shift = scheduler.config.get("max_shift", 1.15)
    m = (max_shift - base_shift) / (max_seq - base_seq)
    if is_distilled:
        mu_map = {}
        if isinstance(policy, dict):
            mode = str(policy.get("mode", "fixed") or "fixed").lower()
            mu_map = policy.get("mu") or {}
        else:
            mode = os.environ.get("KREA2_DISTILLED_SHIFT", "fixed").strip().lower()
        if mode in ("", "fixed"):
            return 1.15
        if mode == "resolution":
            mu = max(1.15, 1.15 + m * (seq_len - max_seq))
            print(f"[EricKrea2-MS] distilled shift 'resolution' ({stage or '?'}): "
                  f"seq {seq_len} -> mu {mu:.3f}")
            return mu
        if mode == "manual":
            mu = float(mu_map.get(stage or "s1", 1.15))
            print(f"[EricKrea2-MS] distilled shift manual ({stage or 's1'}): mu {mu:.3f}")
            return mu
        try:  # env-var numeric override (debug fallback path only)
            mu = float(mode)
            print(f"[EricKrea2-MS] distilled shift env override: mu {mu:.3f}")
            return mu
        except ValueError:
            print(f"[EricKrea2-MS] KREA2_DISTILLED_SHIFT='{mode}' not understood; using fixed 1.15")
            return 1.15
    b = base_shift - m * base_seq
    return m * seq_len + b


def _compute_actual_start_sigma(scheduler, raw_sigmas: list, mu: float) -> float:
    sigmas = np.array(raw_sigmas, dtype=np.float64)
    if scheduler.config.get("use_dynamic_shifting", False):
        shift_type = scheduler.config.get("time_shift_type", "exponential")
        if shift_type == "exponential":
            exp_mu = math.exp(mu)
            sigmas = exp_mu / (exp_mu + (1.0 / sigmas - 1.0))
        else:
            sigmas = mu / (mu + (1.0 / sigmas - 1.0))
    else:
        static_shift = scheduler.config.get("shift", 1.0)
        sigmas = static_shift * sigmas / (1.0 + (static_shift - 1.0) * sigmas)
    shift_terminal = scheduler.config.get("shift_terminal", None)
    if shift_terminal:
        one_minus_z = 1.0 - sigmas
        scale_factor = one_minus_z[-1] / (1.0 - shift_terminal)
        sigmas = 1.0 - (one_minus_z / scale_factor)
    return float(sigmas[0])


def build_sigma_schedule(num_steps: int, denoise: float, schedule: str = "linear") -> list:
    """Flow-matching sigma schedule (raw, pre-shift). `schedule` sets step *spacing*
    within the active denoise range; all schedules start at the same sigma for a
    given denoise so they're interchangeable. Delegates to the shared curve library
    in :mod:`_sigmas` (linear/balanced/karras/beta57/beta/bong_tangent/exponential)."""
    return _sigmas.build_sigmas(num_steps, denoise, schedule)


def _keep(num_steps: int, denoise: float) -> int:
    return num_steps if denoise >= 1.0 else max(1, int(round(num_steps * denoise)))


def _rebalance_krea_embeds(embeds, layer_weights, multiplier):
    """Rescale Krea2 conditioning to counter its built-in prompt dampening.

    Krea2's ``encode_prompt`` returns the 12 Qwen3-VL taps UNFLATTENED as
    ``(B, seq, n_layers, dim)`` (the taps live on axis -2; ``text_fusion``
    collapses them via its projector). Apply a per-layer gain on that axis plus
    a global multiplier. A flattened ``(B, seq, n_layers*dim)`` layout (ComfyUI
    CONDITIONING style) is also handled. This is the native equivalent of the
    community ``ConditioningKrea2Rebalance`` node.
    """
    if embeds is None or embeds.dim() < 1:
        return embeds
    n = len(layer_weights)
    if n == 0:
        return embeds
    mult = float(multiplier)
    w = torch.tensor(layer_weights, dtype=embeds.dtype, device=embeds.device)

    # Unflattened layout: 12 taps on axis -2  (B, seq, n_layers, dim).
    if embeds.dim() >= 3 and embeds.shape[-2] == n:
        w = w.view(*([1] * (embeds.dim() - 2)), n, 1)
        return embeds * w * mult

    # Flattened layout: (..., n_layers*dim).
    D = int(embeds.shape[-1])
    if D % n == 0:
        per = D // n
        lead = tuple(embeds.shape[:-1])
        w = w.view(*([1] * len(lead)), n, 1)
        return (embeds.reshape(*lead, n, per) * w * mult).reshape(*lead, D)

    print(f"[EricKrea2-MS] cond_rebalance: embeds shape {tuple(embeds.shape)} has no "
          f"{n}-layer axis; left unchanged.")
    return embeds


def _jitter_embeds(embeds, strength, generator):
    """Seeded Gaussian perturbation of the conditioning ("text-encoder seed").

    The TE forward is deterministic - there is nothing to seed there - so per-seed
    conditioning variance is created here instead: e' = e + strength * std * N(0,1).
    Distilled models map (noise, conditioning) -> image so contractively that varying
    the init noise barely moves the layout, but small conditioning perturbations move
    it a lot - this attacks diversity collapse in conditioning space (complementary
    to the delta-LoRA weight-space and high-res noise-space approaches).

    std is computed PER LAYER TAP on the (B, seq, n_layers, dim) Krea2 layout (the 12
    Qwen3-VL taps have very different scales, especially after cond_rebalance), or
    globally per batch for a flattened layout. Noise is drawn in fp32 from the given
    generator and the result cast back to the input dtype."""
    e = embeds.float()
    dims = (1, e.dim() - 1) if e.dim() >= 3 else None
    std = e.std(dim=dims, keepdim=True) if dims else e.std()
    n = torch.randn(e.shape, generator=generator, device=e.device, dtype=torch.float32)
    return (e + float(strength) * std * n).to(embeds.dtype)


@torch.no_grad()
def _res_denoise_packed(pipe, prompt, neg, height, width, x_start, raw_sigmas,
                        is_distilled, guidance_scale, eta, generator, progress_cb=None,
                        sigma_window=None, method="res_2m", cond_override=None, noise_type="white",
                        shift_policy=None, shift_stage=None, hybrid_lcm_steps=0):
    """Run the RES multistep (res_2m) sampler over the Krea2 transformer directly,
    bypassing pipe.__call__. Returns the packed latent (same format as pipe(...).images).

    x_start: packed start latent already at raw_sigmas[0] (the re-noised upscaled latent
             for partial stages; spectrum-shaped noise for S1). If None, white noise is
             sampled (S1 white-noise path).
    cond_override: optional precomputed (prompt_embeds, prompt_embeds_mask) tuple (e.g. from
             EricKrea2VisionPrompt) - when given, skips pipe.encode_prompt(prompt=...) entirely
             and uses these directly, so a vision-informed conditioning flows through exactly
             like a text-only one. `prompt` is ignored for the positive side in that case.
    noise_type: shapes the SDE ANCESTRAL CHURN noise only (the per-step noise re-injected when
             eta>0) - NOT the primary re-noise/init, which is always plain white (matches what
             the flow-match denoiser was actually trained to invert; see the module docstring's
             noise-type note for why colored noise as the DOMINANT substitution survives
             denoising as visible artifacts). This mirrors RES4LYF's own `noise_type_sde` -
             their colored-noise types are for exactly this per-step SDE role, not initial/re-
             noise. No effect when eta=0 or method='euler' (no SDE churn hook either way).
    The shifted sigma grid is taken straight from the scheduler so it is bit-identical to
    what pipe.__call__ would step through; timestep == shifted sigma for flow matching.
    """
    device = getattr(pipe, "_execution_device", None) or "cuda"
    if cond_override is not None:
        prompt_embeds, prompt_mask = cond_override
        prompt_embeds = prompt_embeds.to(device)
        prompt_mask = prompt_mask.to(device)
    else:
        prompt_embeds, prompt_mask = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1)
    dtype = prompt_embeds.dtype
    do_cfg = bool(guidance_scale) and guidance_scale > 0
    neg_embeds = neg_mask = None
    if do_cfg:
        neg_embeds, neg_mask = pipe.encode_prompt(prompt=[neg or ""], device=device, num_images_per_prompt=1)

    # starting latent
    if x_start is None:
        num_ch = pipe.transformer.config.in_channels // (pipe.patch_size ** 2)
        x = pipe.prepare_latents(1, num_ch, height, width, dtype, device, generator, None)
    else:
        x = x_start.to(device=device, dtype=dtype)

    # rotary position ids for this stage's latent grid
    grid_h = height // (pipe.vae_scale_factor * pipe.patch_size)
    grid_w = width // (pipe.vae_scale_factor * pipe.patch_size)
    position_ids = pipe.prepare_position_ids(prompt_embeds.shape[1], grid_h, grid_w, device)

    # shifted sigma grid. A pre-built window (start/end-step refinement) is used as-is and
    # may terminate above 0 (early stop -> noisy handoff); otherwise build it from raw_sigmas
    # straight off the scheduler (identical to pipe.__call__).
    if sigma_window is not None:
        sigmas = sigma_window.to(device=device, dtype=torch.float32)
    else:
        mu = _compute_mu(x.shape[1], pipe.scheduler, is_distilled,
                         policy=shift_policy, stage=shift_stage)
        pipe.scheduler.set_timesteps(sigmas=list(raw_sigmas), device=device, mu=mu)
        sigmas = pipe.scheduler.sigmas.to(device=device, dtype=torch.float32)  # [keep+1], ends at 0

    def denoise_fn(xx, sigma):
        xin = xx.to(dtype)                      # model dtype (e.g. bf16); accumulator stays fp32
        ts = sigma.to(dtype).expand(xin.shape[0])   # timestep == shifted sigma (t / num_train)
        pipe.transformer._eric_cfg_pass = "pos"      # read by _edit_guidance (edit/NAG/NegPiP)
        v = pipe.transformer(
            hidden_states=xin, encoder_hidden_states=prompt_embeds, timestep=ts,
            position_ids=position_ids, encoder_attention_mask=prompt_mask,
            attention_kwargs=None, return_dict=False,
        )[0]
        if do_cfg:
            pipe.transformer._eric_cfg_pass = "neg"
            vneg = pipe.transformer(
                hidden_states=xin, encoder_hidden_states=neg_embeds, timestep=ts,
                position_ids=position_ids, encoder_attention_mask=neg_mask,
                attention_kwargs=None, return_dict=False,
            )[0]
            pipe.transformer._eric_cfg_pass = "pos"
            v = v + guidance_scale * (v - vneg)
        return xin - sigma.to(dtype) * v          # velocity -> x0

    noise_sampler = None
    # er_sde injects noise EVERY step regardless of eta (eta is ignored by contract), so it
    # always needs the seeded sampler - with eta 0 it used to fall back to the solver's
    # unseeded randn_like and a fixed seed gave a different image every run (2026-10-08).
    if (eta and eta > 0) or method == "er_sde":
        if noise_type and noise_type != "white":
            h_lat, w_lat = 2 * (int(height) // 16), 2 * (int(width) // 16)
            b = x.shape[0]
            def noise_sampler(s, sn):
                return _shaped_packed_noise(b, h_lat, w_lat, noise_type, generator, device)
        else:
            noise_sampler = lambda s, sn: torch.randn(
                x.shape, device=device, dtype=torch.float32, generator=generator)

    def _cb(i, sigma, denoised, xx):
        if progress_cb is not None:
            progress_cb()

    if method == "res_2s":
        out = res_2s_sample(denoise_fn, x, sigmas, eta=float(eta or 0.0),
                            noise_sampler=noise_sampler, callback=_cb)
    elif method == "deis_3m":
        out = deis_sample(denoise_fn, x, sigmas, max_order=3, eta=float(eta or 0.0),
                          noise_sampler=noise_sampler, callback=_cb)
    elif method in ("abnorsett_3m", "abnorsett_4m"):
        # Fixed-coefficient Norsett exponential AB (2026-07-27): uniform-step
        # phi weights (RES4LYF-faithful), bounded on EVERY schedule - unlike
        # the exact variable-step order-4 (former deis_4m alias) whose weights
        # blew up ~269x on linear_quadratic and amplified prediction noise
        # into garbage. deis_4m was dropped for that reason (Eric's call).
        out = abnorsett_sample(denoise_fn, x, sigmas,
                               order=(3 if method == "abnorsett_3m" else 4),
                               eta=float(eta or 0.0),
                               noise_sampler=noise_sampler, callback=_cb)
    elif method == "er_sde":
        # Intrinsically stochastic (its own churn law) - eta is IGNORED by
        # contract; the stage's noise_sampler shapes the churn (colored noise
        # works, unlike stock ComfyUI's er_sde).
        out = er_sde_sample(denoise_fn, x, sigmas,
                            noise_sampler=noise_sampler, callback=_cb)
    elif method == "lawson4_4s":
        out = lawson4_sample(denoise_fn, x, sigmas, eta=float(eta or 0.0),
                             noise_sampler=noise_sampler, callback=_cb)
    elif method == "etdrk4_4s":
        out = etdrk4_sample(denoise_fn, x, sigmas, eta=float(eta or 0.0),
                            noise_sampler=noise_sampler, callback=_cb)
    elif method == "rk6_7s":
        out = rk_explicit_sample(denoise_fn, x, sigmas, tableau="rk6_7s",
                                 eta=float(eta or 0.0), noise_sampler=noise_sampler,
                                 callback=_cb)
    elif method == "euler":
        # eta CLAMPED to 0 for euler (2026-07-25, Eric's ruling): the GUI
        # tooltips promise "euler ignores eta" and Stage 1 euler bypasses this
        # dispatch entirely (pipe path, no eta), so refine-stage euler honoring
        # churn here would make the same widget mean different things per
        # stage. One contract everywhere: euler is the deterministic baseline;
        # for 1st-order + churn use res_2m at low eta instead.
        out = rk_explicit_sample(denoise_fn, x, sigmas, tableau="euler",
                                 eta=0.0, noise_sampler=noise_sampler,
                                 callback=_cb)
    elif method in ("lcm", "lcm_hybrid", "lcm_hybrid2"):
        # lcm-family draws are ALWAYS plain seeded WHITE, regardless of eta or
        # noise_type (2026-07-23): the full re-noise replaces the latent
        # wholesale, and the model was trained on WHITE noise at every sigma -
        # a full replacement with shaped/colored noise is out-of-distribution
        # and decodes as chromatic blotches over the image ("rainbow").
        # noise_type shaping stays with the FRACTIONAL SDE churn samplers
        # (res/deis/rk at eta>0), where the injected deviation is small.
        _ns = (lambda s, sn: torch.randn(
            x.shape, device=device, dtype=torch.float32, generator=generator))
        if method == "lcm":
            out = lcm_sample(denoise_fn, x, sigmas, noise_sampler=_ns, callback=_cb)
        else:
            # lcm_hybrid: lcm churn -> deis_3m tail; lcm_hybrid2: lcm -> res_2m
            # tail (2026-07-25 - res_2m was testing better than deis_3m as the
            # detail-development half). hybrid_lcm_steps = exact churn steps
            # (0 = auto half); replaced the old fraction 2026-07-26 because at
            # low step counts small % changes were silent no-ops.
            out = lcm_hybrid_sample(denoise_fn, x, sigmas, eta=float(eta or 0.0),
                                    noise_sampler=_ns, callback=_cb,
                                    second=("res_2m" if method == "lcm_hybrid2"
                                            else "deis_3m"),
                                    lcm_steps=int(hybrid_lcm_steps or 0))
    else:
        out = res_multistep_sample(denoise_fn, x, sigmas, eta=float(eta or 0.0),
                                   noise_sampler=noise_sampler, callback=_cb)
    return out.to(dtype)


def _report_prompt_tokens(pipe, prompt, max_sequence_length=512):
    """Exact Krea2 prompt-token count (same template/tokenizer as the pipeline's
    get_text_hidden_states). Budget = max_sequence_length - template suffix tokens (507);
    diffusers truncates anything beyond it silently. Logs only; never raises. 2026-10-08."""
    try:
        tok = pipe.tokenizer
        prefix = pipe.prompt_template_encode_prefix
        start = int(pipe.prompt_template_encode_start_idx)
        budget = int(max_sequence_length) - int(pipe.prompt_template_encode_num_suffix_tokens)
        ids = tok(prefix + str(prompt), truncation=False).input_ids
        n = len(ids) - start
        if n <= budget:
            print(f"[EricKrea2-MS] prompt: {n}/{budget} tokens")
            return n
        dropped = ids[start + budget:]
        head = tok.decode(dropped[:24]).strip().replace("\n", " ")
        print(f"[EricKrea2-MS] \u26a0 prompt: {n}/{budget} tokens - the last {n - budget} token(s) "
              f"are TRUNCATED by the Krea 2 encoder (fixed 512-token budget). Dropped text "
              f"starts: \"{head}{'...' if len(dropped) > 24 else ''}\"")
        return n
    except Exception as e:
        print(f"[EricKrea2-MS] prompt token count unavailable ({type(e).__name__}: {e})")
        return None


class EricKrea2MultistageUltra:
    # PATH B (work in progress): exact copy of EricKrea2Multistage, kept separate so the
    # shipping node is never disturbed. This is the base onto which the RES multistep
    # (res_2m / res_2s) + beta57 + optional SDE eta denoise loop will be grafted,
    # replacing the diffusers pipe.__call__ Euler stepping for the partial-denoise stages.
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "krea2_pipeline": ("KREA2_PIPELINE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "prompt_conditioning": ("KREA2_CONDITIONING", {
                    "tooltip": "Optional precomputed conditioning (from Eric Krea2 Vision Prompt), used INSTEAD "
                               "of encoding 'prompt' as text. Lets image-grounded conditioning flow through every "
                               "stage exactly like a text-only prompt would (same tensor shape). cond_rebalance "
                               "still applies. 'prompt' is ignored for the positive side when this is connected, "
                               "but still used for the negative side and console logging."}),
                "ref_latents": ("KREA2_REF_LATENTS", {
                    "tooltip": "Optional reference latents (from Eric Krea2 Reference Latents). That "
                               "node's edit_recipe picks the mechanism: ostris_t0 appends refs at t=0 "
                               "modulation (style/reference LoRAs, ~0.4-0.5 strength - 1.0 is reported "
                               "to break/fragment the image); edit_frame prepends the source on the "
                               "target grid at shared timestep (identity/instruction edit LoRAs at 1.0, "
                               "pixel-fit to this run's S1 size, conditions S1 only). ONLY does "
                               "something useful with a matching edit-trained LoRA loaded; the base "
                               "model ignores what it can't read. Turbo at guidance 0 for most use; "
                               "edit_frame removals want Raw at CFG ~3 with an EMPTY negative."}),
                "ref_match_size": ("BOOLEAN", {"default": True,
                    "tooltip": "ostris_t0 recipe only (edit_frame always pixel-fits + re-encodes at S1 "
                               "size instead). Rescale ref_latents (in latent token space, no re-encode) "
                               "to this run's resolved Stage 1 size before denoising. NOTE: the ostris "
                               "node/training never matches refs to the target (own grid, <=1MP cap), so "
                               "for the Style Reference LoRA try OFF first - ON deviates from the trained "
                               "geometry and the token-space resample slightly softens the reference. ON "
                               "remains available for the corner-bias failure mode (reference pasted in a "
                               "shrunken corner / fragmented partial copy) if OFF misbehaves at your "
                               "sizes."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "cond_preset": (["custom"] + sorted(cls._load_cond_presets().keys()), {"default": "custom",
                    "tooltip": "Preset library of rebalance profiles (from cond_presets.json). 'custom' "
                               "uses the manual cond_multiplier + cond_layer_weights below; a named preset "
                               "overrides both (only when cond_rebalance is ON). Edit the JSON + restart to "
                               "add sets."}),
                "cond_rebalance": ("BOOLEAN", {"default": False,
                    "tooltip": "Rescale the positive conditioning's 12 Qwen3-VL layer taps "
                               "(per-layer gains x global multiplier) - the native equivalent of "
                               "the community 'ConditioningKrea2Rebalance' node used to counter "
                               "Krea2's built-in prompt dampening. Off = untouched conditioning."}),
                "cond_multiplier": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10000.0, "step": 0.1,
                    "tooltip": "Global gain on the positive conditioning when cond_rebalance is on "
                               "(reference default 4.0)."}),
                "cond_layer_weights": ("STRING", {"default": "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0",
                    "tooltip": "Comma-separated per-layer gains for the 12 Qwen3-VL taps (applied "
                               "before the global multiplier). Count must divide the conditioning "
                               "width (12 for Krea2). Reference emphasizes taps 8/9/11."}),
                "cond_jitter": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Seeded Gaussian perturbation of the POSITIVE conditioning, as a fraction "
                               "of its per-layer-tap std (0 = off). The 'text-encoder seed' diversity "
                               "trick: distilled/Turbo models barely respond to init-noise changes but "
                               "respond strongly to conditioning changes, so this restores seed-to-seed "
                               "variety in conditioning space. Applied ONCE, identically for every stage "
                               "(after cond_rebalance). Useful range ~0.01-0.10; high values drift off "
                               "prompt. Negative conditioning is untouched."}),
                "cond_jitter_seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Seed for cond_jitter noise. -1 (default) = derived from the main seed "
                               "(seed+777), so changing the main seed changes the jitter too - the "
                               "diversity behaviour you usually want. >=0 = fixed independent seed: hold "
                               "the jitter constant while sweeping noise seeds (or vice versa) to explore "
                               "conditioning space and noise space separately."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "seed_mode": (["same_all_stages", "offset_per_stage", "random_per_stage"], {
                    "default": "offset_per_stage",
                    "tooltip": "same_all_stages: one seed threaded through (reproducible). "
                               "offset_per_stage: S2=seed+1, S3=seed+2 (reproducible, decorrelated). "
                               "random_per_stage: S1 seeded, S2/S3 fresh random each run."}),
                # framing / output prep (widget order only; crop applies to the FINAL stage)
                "crop_bottom": ("INT", {"default": 0, "min": 0, "max": 96, "step": 2,
                    "tooltip": "Crop N latent rows off the bottom of the FINAL executed stage (1 latent "
                               "row = 8 px at the final stage's resolution). 0 = off (default). Not needed "
                               "since the Oct 2026 mask-safe attention fix - the bottom band was a "
                               "flash_varlen bug, not the model. Kept for old workflows / special cases. "
                               "Snapped to even (2x2 patch stride)."}),
                "crop_overgen": ("BOOLEAN", {"default": False,
                    "tooltip": "ON (recommended): the final stage makes disposable extra bottom rows "
                               "(S1 generates taller; S2/S3 pad the upscaled latent) so the band forms "
                               "in them and is trimmed - your target size/aspect is preserved and the "
                               "real content is never stretched. OFF: trim the final latent directly "
                               "(output ends up shorter by the crop)."}),
                "init_match_size": ("BOOLEAN", {"default": True,
                    "tooltip": "img2img only: match Stage 1 size to the init latent's own dimensions "
                               "(preserves source resolution/aspect). OFF = use aspect_ratio + "
                               "s1_megapixels and resize the init to fit (can reframe/stretch if the "
                               "aspect differs)."}),
                "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16,
                    "tooltip": "Explicit Stage-1 width override (0 = derive from aspect_ratio + "
                               "s1_megapixels). Wire the Eric Krea2 Resolution node here to size Stage 1 "
                               "to a source image (img2img) or a fixed resolution. Both width AND height "
                               "must be > 0 to take effect; rounded to /16. init_match_size still wins "
                               "when an init_latent is connected."}),
                "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16,
                    "tooltip": "Explicit Stage-1 height override (0 = derive from aspect_ratio + "
                               "s1_megapixels). See 'width'. Both must be > 0 to take effect."}),
                "aspect_ratio": (list(_ASPECT_RATIOS.keys()), {"default": "5:4 landscape"}),
                "s1_megapixels": ("FLOAT", {"default": 3.00, "min": 0.25, "max": 16.0, "step": 0.05,
                    "tooltip": "Stage-1 size in megapixels (dimensions derived from the aspect ratio)."}),
                # Stage 1 - composition. Official steps: Turbo 8, Raw 28-52.
                "s1_steps": ("INT", {"default": 10, "min": 1, "max": 200}),
                "s1_cfg": ("FLOAT", {"default": 1.9, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Guidance for this stage. Krea2 convention: this is 'g', where standard "
                               "CFG = 1 + g (g=0 -> no guidance/CFG 1.0; g=0.2 -> CFG 1.2; g=3.5 -> CFG "
                               "4.5 for Raw). g>0 activates the negative prompt but adds an uncond pass "
                               "(~2x slower this stage). On Turbo it's forced to 0 unless turbo_guidance."}),
                "s1_schedule": (_SCHEDULES, {"default": "beta57"}),
                "s1_sampler": (_SAMPLERS, {"default": "res_2m",
                    "tooltip": "Denoise solver. euler: 1st-order (fastest, A/B baseline). "
                               "res_2m: RES 2nd-order exponential MULTIstep (sharper, same model-call "
                               "cost as euler). res_2s: RES 2nd-order SINGLE-step predictor/corrector "
                               "(strongest at low steps, ~2x slower). deis_3m: DEIS 3rd-order "
                               "multistep, exact variable-step weights (very smooth, euler cost). "
                               "abnorsett_3m / abnorsett_4m: Norsett exponential Adams-Bashforth "
                               "3rd/4th order, 1 call/step, FIXED uniform-step weights - bounded "
                               "and stable on every schedule incl. linear_quadratic (the kreamania "
                               "pass-2 style pick). lawson4_4s / etdrk4_4s: single-step exponential "
                               "RK order 4, 4 calls/step, no history - the most noise-robust "
                               "solvers here, stable everywhere; strong on refine windows. er_sde: "
                               "ER-SDE-Solver-3 (ComfyUI-faithful port) - stochastic, re-injects "
                               "scaled noise every step; strong at LOW step counts (community "
                               "recipe: er_sde + simple schedule, 8 steps); eta ignored (own "
                               "churn), noise type shapes the churn. All but "
                               "euler honour a noisy early end_step. rk6_7s: 6th-order 7-stage "
                               "explicit Runge-Kutta - highest solver accuracy, ~7x model calls "
                               "per step (the kreamania finetuner pass-1 recipe with "
                               "linear_quadratic sigmas). Recommended: res_2s + "
                               "beta57 for Stage 1. lcm: full ancestral - every step jumps to "
                               "x0 and fully re-noises to the next sigma (ignores eta, always "
                               "seeded). The artifact-launderer: strongest choice for "
                               "post-upscale refine windows (the native euler->lcm two-pass "
                               "trick); too churny for Stage 1. lcm_hybrid: lcm churn for the "
                               "first half of the window, then deis_3m for the second half - "
                               "launders entry damage, then develops detail trajectory-faithfully "
                               "(the detail-preserving alternative when plain lcm eats fine "
                               "detail)."}),
                "s1_noise": (_NOISE_TYPES, {"default": "white",
                    "tooltip": "Shapes the SDE ancestral-churn noise only (eta>0, non-euler samplers) - "
                               "NOT the primary init, which is always plain white. white is recommended; "
                               "low_freq/high_freq/pink are a subtle stylistic lean, not a composition "
                               "control (that's denoise strength / start_step). No effect at eta=0 or "
                               "with sampler=euler."}),
                "s1_eta": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "SDE ancestral churn for Stage 1 (res_2m/res_2s/deis_* only; euler ignores "
                               "it). 0 = deterministic ODE. S1 lays down composition, so keep it low "
                               "(0-0.1); higher adds variation but can roughen structure."}),
                # img2img - optional Stage-1 init latent (from Eric Krea2 VAE Encode or another
                # stage's latent). Reuses the same start/end-step window S2/S3 use, so denoise
                # strength is set by s1_start_step (no separate denoise float).
                "init_latent": ("KREA2_LATENT", {
                    "tooltip": "Optional img2img source (from Eric Krea2 VAE Encode, or any stage's "
                               "KREA2_LATENT output). When connected, Stage 1 becomes an img2img pass: "
                               "the init latent is re-noised at s1_start_step and denoised over "
                               "[s1_start_step, s1_end_step] instead of generating from pure noise. "
                               "Leave unconnected for normal text-to-image."}),
                "s1_start_step": ("INT", {"default": 0, "min": 0, "max": 199,
                    "tooltip": "img2img only (needs init_latent): inject the init latent at this step of "
                               "the s1_steps schedule = the re-noise / denoise-strength control. 0 = full "
                               "noise (ignores the source, ~text-to-image); higher preserves more of the "
                               "source (lighter change). No effect for text-to-image."}),
                "s1_end_step": ("INT", {"default": 0, "min": 0, "max": 200,
                    "tooltip": "img2img only: stop Stage 1 at this step of s1_steps. 0 (default) = run to "
                               "the end (full denoise, recommended for the first stage). < s1_steps leaves "
                               "the latent noisy for the next stage (res_2m only)."}),
                # Stage 2 - refine
                "s1_s2_upscale_vae": ("BOOLEAN", {"default": False,
                    "tooltip": "S1->S2 jump: use the trained 2x VAE (decode-2x + re-encode = real detail) "
                               "instead of bislerp. Forces a 2x linear step (4x area), OVERRIDING "
                               "upscale_to_stage2. Requires upscale_vae connected. Separate from "
                               "upscale_vae_mode because using the VAE this early is a specialist move - "
                               "it injects detail right after composition and the re-encode softening "
                               "gets refined away by S2/S3 (unlike at the final stage, where it looks soft)."}),
                "upscale_to_stage2": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 8.0, "step": 0.5,
                    "tooltip": "S1->S2 AREA upscale factor (0 = stop after S1). This is MEGAPIXELS, "
                               "not linear: 2.0 = 2x the pixels ~= 1.41x per side; 4.0 = 2x per side. "
                               "IGNORED for this jump when s1_s2_upscale_vae is on (the VAE forces a "
                               "2x linear step = 4x area). Auto-rounded to a multiple of 16px."}),
                "s2_steps": ("INT", {"default": 20, "min": 1, "max": 200,
                    "tooltip": "Length of the S2 sigma schedule. The stage runs only the "
                               "[start_step, end_step] window of it (reference-workflow style)."}),
                "s2_cfg": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Guidance 'g' (standard CFG = 1 + g). g>0 enables the negative prompt "
                               "and costs ~2x time this stage. Turbo: forced 0 unless turbo_guidance."}),
                "s2_start_step": ("INT", {"default": 7, "min": 0, "max": 199,
                    "tooltip": "Inject the upscaled latent at this step of the s2_steps schedule "
                               "(re-noise level; replaces denoise-strength). Higher = preserves more "
                               "S1 structure / lighter refinement."}),
                "s2_end_step": ("INT", {"default": 20, "min": 1, "max": 200,
                    "tooltip": "Stop S2 at this step. < s2_steps leaves the latent slightly noisy for "
                               "S3 to continue (res_2m only; euler always runs to sigma 0)."}),
                "s2_schedule": (_SCHEDULES, {"default": "linear"}),
                "s2_sampler": (_SAMPLERS, {"default": "euler",
                    "tooltip": "Denoise solver for Stage 2 (see s1_sampler). Recommended for the "
                               "refine pass: deis_3m or abnorsett_4m + bong_tangent schedule."}),
                "s2_noise": (_NOISE_TYPES, {"default": "white",
                    "tooltip": "Shapes the SDE ancestral-churn noise only (eta>0, non-euler samplers) - "
                               "NOT the primary re-noise. See s1_noise."}),
                "s2_eta": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "SDE ancestral churn for the Stage 2 refine (res_2m/res_2s/deis_* only; "
                               "euler ignores it). 0 = clean deterministic refine (best for avoiding "
                               "stray-hair / edge artifacts); ~0.1 adds a little fine detail. Keep low on "
                               "refine passes - high eta re-injects high-frequency noise every step."}),
                # Stage 3 - final detail
                "upscale_to_stage3": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 0.5,
                    "tooltip": "S2->S3 AREA upscale factor (0 = stop after S2). MEGAPIXELS, not linear: "
                               "2.0 = 2x the pixels ~= 1.41x per side. IGNORED when upscale_vae_mode does "
                               "a plain 's2-s3' VAE jump (forced 4x area); but RESPECTED by the "
                               "'s2-s3 with downsample' modes (VAE 2x detail resampled to THIS factor). "
                               "Auto-rounded to a multiple of 16px."}),
                "s3_steps": ("INT", {"default": 20, "min": 1, "max": 200,
                    "tooltip": "Length of the S3 sigma schedule. The stage runs only the "
                               "[start_step, end_step] window of it."}),
                "s3_cfg": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Guidance 'g' (standard CFG = 1 + g). g>0 enables the negative prompt "
                               "and costs ~2x time this stage. Turbo: forced 0 unless turbo_guidance."}),
                "s3_start_step": ("INT", {"default": 9, "min": 0, "max": 199,
                    "tooltip": "Inject the upscaled latent at this step of the s3_steps schedule "
                               "(re-noise level). Lower = deeper final refinement."}),
                "s3_end_step": ("INT", {"default": 20, "min": 1, "max": 200,
                    "tooltip": "Stop S3 at this step. For the FINAL stage keep this = s3_steps so the "
                               "image fully denoises to sigma 0; ending early leaves visible noise."}),
                "s3_schedule": (_SCHEDULES, {"default": "linear"}),
                "s3_sampler": (_SAMPLERS, {"default": "euler",
                    "tooltip": "Denoise solver for Stage 3 (see s1_sampler). deis_3m / abnorsett_4m + "
                               "bong_tangent works well for the final refine."}),
                "s3_noise": (_NOISE_TYPES, {"default": "white",
                    "tooltip": "Shapes the SDE ancestral-churn noise only (eta>0, non-euler samplers) - "
                               "NOT the primary re-noise. See s1_noise. Denoise strength (start/end step) "
                               "is still the real detail control."}),
                "s3_eta": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "SDE ancestral churn for the Stage 3 final refine (res_2m/res_2s/deis_* "
                               "only; euler ignores it). 0 = cleanest; small values (~0.1) add fine detail. "
                               "High eta here re-introduces high-frequency artifacts on the final image."}),
                "turbo_guidance": (["off", "s1_only", "all_stages"], {"default": "off",
                    "tooltip": "Guidance on the Turbo/distilled checkpoint (no effect on Raw, which "
                               "always uses its cfg).\n"
                               "off: force all cfg=0 - no CFG, negative prompt ignored, fastest.\n"
                               "s1_only: use s1_cfg for composition (where guidance matters most) and "
                               "force s2_cfg=s3_cfg=0, so refinement stays single-pass/fast. RECOMMENDED.\n"
                               "all_stages: use every per-stage cfg (negative active throughout; each "
                               "guided stage ~2x slower).\n"
                               "Krea2 cfg is g (standard CFG = 1+g); Turbo is distilled for g=0, so keep "
                               "it LOW (g~0.1-0.3 == CFG 1.1-1.3). Any g>0 adds an uncond pass (~2x that "
                               "stage). We use plain CFG, not CFG++."}),
                # Optional per-stage sigma-curve override (from Eric Krea2 Sigmas).
                "sigmas": ("KREA2_SIGMAS", {
                    "tooltip": "Optional per-stage sigma-curve shape (from Eric Krea2 Sigmas). Each "
                               "enabled stage OVERRIDES that stage's schedule dropdown with its curve "
                               "+ detail_bias; disabled stages fall back to the panel dropdown. The "
                               "shape is resolution-independent - it's realized at THIS node's own "
                               "s1/s2/s3_steps. Leave unconnected to use the schedule dropdowns."}),
                # Trained upscale VAE (optional) - higher-quality between-stage upscale.
                "upscale_vae": ("UPSCALE_VAE",),
                "upscale_vae_mode": (_UPSCALE_VAE_MODES, {
                    "default": "disabled",
                    "tooltip": "Trained 2x upscale VAE routing (requires upscale_vae connected). This "
                               "dropdown covers the S2->S3 jump and the final decode ONLY - the S1->S2 "
                               "jump has its own toggle (s1_s2_upscale_vae).\n"
                               "disabled: bislerp latent interpolation + standard decode (no VAE).\n"
                               "s2-s3: S2->S3 via VAE decode-2x + re-encode (real detail; forces 4x area, "
                               "overrides upscale_to_stage3). Needs 3 stages.\n"
                               "s2-s3 with downsample: same VAE detail, then resampled DOWN to your "
                               "upscale_to_stage3 factor (respects the field instead of forcing 4x).\n"
                               "final decode: final image via VAE 2x decode (2x larger output).\n"
                               "final decode with downsample: VAE 2x decode then Lanczos ->native "
                               "(supersample quality pass, same output size).\n"
                               "both: s2-s3 (4x area) + final VAE 2x decode.\n"
                               "both with downsample: s2-s3 downsampled + final downsampled.\n"
                               "both with final decode downsample: s2-s3 (4x area) + final downsampled."}),
                # Alternate decode VAE (optional) - swap grain/texture at decode.
                "decode_vae": ("KREA2_DECODE_VAE", {
                    "tooltip": "Optional base Wan 2.1 (or other Wan-family) VAE for the 1x "
                               "final decode and stage previews, instead of the Qwen VAE "
                               "(natural grain vs 'plastic' skin). Ignored when upscale_vae_mode "
                               "does the final 2x decode."}),
                "preview_stages": ("BOOLEAN", {"default": False,
                    "tooltip": "Also decode stage 1 and stage 2 to images (stage1_image / "
                               "stage2_image outputs) for parameter tuning. Costs one extra "
                               "1x decode per stage; off by default."}),
                # -- distilled shift policy (added 2026-07-21; appended LAST so saved --
                # -- workflows keep their positional widget values intact           --
                "distilled_shift": (["fixed", "resolution", "manual"], {"default": "fixed",
                    "tooltip": "Per-stage flow-matching shift (mu) policy for DISTILLED/Turbo "
                               "models (Raw is already resolution-aware and ignores this).\n"
                               "fixed: mu=1.15 everywhere - Krea's distillation value, calibrated "
                               "for <=2K. Above that the schedule is under-shifted and upscale "
                               "error can survive as mottling/splotches on many fine tunes.\n"
                               "resolution: keep 1.15 up to the ~1.6MP calibration band, extend "
                               "mu automatically above it (recommended starting point).\n"
                               "manual: dial each stage with shift_mu_s1/s2/s3 below.\n"
                               "NOTE: fixed and resolution IGNORE the shift_mu fields below "
                               "(manual only). Stages set to the euler sampler skip this policy "
                               "entirely and run the stock diffusers flow-match shift instead.\n"
                               "Rule of thumb: LOWER mu = more low-sigma time = more fine "
                               "detail/texture (but mottling risk at high res); HIGHER mu = more "
                               "high-sigma time = smoother, cleaner structure (too high = "
                               "soft/waxy)."}),
                "shift_mu_s1": ("FLOAT", {"default": 1.15, "min": 0.1, "max": 10.0, "step": 0.05,
                    "tooltip": "Manual mu for Stage 1 - used ONLY when distilled_shift=manual "
                               "(fixed/resolution ignore it, and euler stages ignore it too). "
                               "1.15 = Turbo default; usually leave it there for <=2MP S1. "
                               "Lower = more detail, higher = smoother."}),
                "shift_mu_s2": ("FLOAT", {"default": 1.15, "min": 0.1, "max": 10.0, "step": 0.05,
                    "tooltip": "Manual mu for Stage 2 - used ONLY when distilled_shift=manual "
                               "(fixed/resolution ignore it, and euler stages ignore it too). "
                               "Try 2.0-3.0 at 4-8MP if splotches persist. Lower = more fine "
                               "detail/texture, higher = smoother/cleaner (too high = soft/waxy)."}),
                "shift_mu_s3": ("FLOAT", {"default": 1.15, "min": 0.1, "max": 10.0, "step": 0.05,
                    "tooltip": "Manual mu for Stage 3 - used ONLY when distilled_shift=manual "
                               "(fixed/resolution ignore it, and euler stages ignore it too). "
                               "Usually matches or slightly exceeds shift_mu_s2. Lower = more "
                               "detail, higher = smoother."}),
                "upscale_renorm": (["off", "global", "per_channel"], {"default": "off",
                    "tooltip": "Repair inter-stage latent statistics after each upscale hop. "
                               "Latent interpolation is a lowpass that shrinks variance, and "
                               "decoded variance IS contrast/saturation - each hop drains a "
                               "little and stages compound it into progressive flattening / "
                               "desaturation. per_channel: re-match each channel's mean/std to "
                               "the pre-hop latent (strongest color+contrast hold; recommended). "
                               "global: overall mean/std only. off: previous behavior. A "
                               "telemetry line prints latent std pre-hop / post-hop / "
                               "post-renorm in every mode, so the drain is measurable."}),
                "s1_hybrid_steps": ("INT", {"default": 0, "min": 0, "max": 98,
                    "tooltip": "lcm_hybrid / lcm_hybrid2 on Stage 1: EXACT number of lcm "
                               "full-churn steps at the head of the stage's window before the "
                               "tail sampler takes over (lcm_hybrid: deis_3m; lcm_hybrid2: "
                               "res_2m). 0 = auto (half the window, ceil - the historical "
                               "behavior). Clamped so both halves always run at least one "
                               "step. Fewer lcm steps = more trajectory-faithful detail; more "
                               "= more laundering. Ignored by every other sampler."}),
                "s2_hybrid_steps": ("INT", {"default": 0, "min": 0, "max": 98,
                    "tooltip": "Stage 2 hybrid handoff in exact lcm steps (0 = auto half). "
                               "See s1_hybrid_steps."}),
                "s3_hybrid_steps": ("INT", {"default": 0, "min": 0, "max": 98,
                    "tooltip": "Stage 3 hybrid handoff in exact lcm steps (0 = auto half). "
                               "See s1_hybrid_steps."}),
                # Appended 2026-10-07 at the END of optional (positional widgets_values safety).
                "degrid": (["off", "hops", "final", "hops+final"], {"default": "off",
                    "tooltip": "Remove the VAE's 2-pixel lattice (DeGrid notch filter; auto-calibrated, "
                               "edge-protected, untouched when no grid is detected).\n"
                               "hops: on the inter-stage VAE upscale (s1_s2_upscale_vae / S2->S3 VAE "
                               "modes) BEFORE re-encoding, so the upscale VAE's pixel_shuffle lattice "
                               "is never baked into the next stage's latent. No effect on plain "
                               "latent (bislerp) hops - they have no pixel step.\n"
                               "final: on the decoded output image.\n"
                               "Console prints the measured grid amplitude per pass."}),
                # Appended 2026-10-07: a SOCKET (no widgets_values slot) - positionally safe.
                "control": ("KREA2_CONTROL", {
                    "tooltip": "Optional Control-LoRA bundle (from Eric Krea2 Control): depth / pose / "
                               "canny ... structure control. Per-stage strengths live on that node "
                               "(0 = stage runs the stock model). The control image is re-encoded at "
                               "each controlled stage's own resolution."}),
                # Appended 2026-10-07: a SOCKET (no widgets_values slot) - positionally safe.
                "style": ("KREA2_STYLE", {
                    "tooltip": "Optional training-free style reference (from Eric Krea2 Style "
                               "Reference). Per-stage multipliers live on that node (default S1 "
                               "only). ~2.3-2.6x per styled step. Not combinable with ref_latents."}),
                # Appended 2026-10-08: a SOCKET (no widgets_values slot) - positionally safe.
                "paint": ("KREA2_PAINT", {
                    "tooltip": "Optional inpaint / outpaint bundle (from Eric Krea2 Paint). Stage 1 "
                               "follows the canvas aspect at s1_megapixels unless width/height are "
                               "set. Kept regions are pinned to the source (per-stage on that node). "
                               "Put the AnyPaint LoRA on the LoRA stack (S1 only). Not combinable "
                               "with style or ref_latents."}),
                # Appended 2026-10-08: SOCKETS (no widgets_values slot) - positionally safe.
                "edit": ("KREA2_EDIT", {
                    "tooltip": "Optional instruction / identity edit (from Eric Krea2 Edit; Krea 2 "
                               "Identity Edit recipe). Put the edit LoRA on the LoRA stack, S1 = 1. "
                               "Stage 1 only (<= 2 MP); S2/S3 refine with this node's prompt text "
                               "(or the instruction when empty). Not combinable with ref_latents, "
                               "style or paint."}),
                "guidance": ("KREA2_GUIDANCE", {
                    "tooltip": "Optional negative guidance (from Eric Krea2 Negative Guidance): NAG "
                               "(negative prompt that works at guidance 0) and/or NegPiP "
                               "((phrase:-1) in the prompt). Combinable with each other and with "
                               "edit; skipped with style / paint / ref_latents."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "KREA2_LATENT", "IMAGE", "IMAGE", "KREA2_LATENT", "KREA2_LATENT")
    RETURN_NAMES = ("image", "latent", "stage1_image", "stage2_image", "stage1_latent", "stage2_latent")
    FUNCTION = "generate"
    CATEGORY = "Eric/Krea2"

    @staticmethod
    def _recover_unloaded_pipeline(krea2_pipeline):
        """ComfyUI may hand us the loader's cached KREA2_PIPELINE dict whose "pipeline"
        was set to None by EricKrea2UnloadModels (that node mutates the shared cached
        dict in place to release VRAM). If the loader's module-level cache still holds a
        live pipeline (e.g. Unload ran with also_clear_module_caches=False), heal the
        dict and reuse it. Otherwise raise a clear, actionable error instead of the
        cryptic `'NoneType' object has no attribute 'encode_prompt'`.

        Note: Unload always nulls the module-cache entry after tearing a pipe down to
        'meta', so any pipeline recovered here is guaranteed fully live (never meta)."""
        recovered = None
        try:
            from .krea2_component_loader import _COMP_CACHE
            recovered = _COMP_CACHE.get("pipeline")
        except Exception:
            recovered = None
        if recovered is None:
            try:
                from .krea2_loader import _PIPELINE_CACHE
                recovered = _PIPELINE_CACHE.get("pipeline")
            except Exception:
                recovered = None
        if recovered is None:
            raise RuntimeError(
                "Krea2 pipeline is None: the model was freed by 'Eric Krea2 Unload Models' "
                "and no live pipeline remains in the loader cache. Force the loader to reload "
                "(e.g. toggle any loader widget, or run the queue once more so IS_CHANGED "
                "rebuilds it) before generating. If this keeps happening, set the Unload node's "
                "'also_clear_module_caches' to True so the loader reliably rebuilds next queue."
            )
        krea2_pipeline["pipeline"] = recovered  # heal the shared dict for this + future runs
        print("[EricKrea2] recovered live pipeline from loader cache "
              "(cached dict had been unloaded); healed for reuse")
        return recovered

    def generate(self, **kwargs):
        # LoRA: realize the declared stack fresh, ephemeral-clear in a finally (cancel-safe).
        krea2_pipeline = kwargs["krea2_pipeline"]
        pipe = krea2_pipeline["pipeline"]
        if pipe is None:
            pipe = self._recover_unloaded_pipeline(krea2_pipeline)
        from .._lora_utils import (realize_lora_stack, unload_all_loras,
                                   assert_clean_baseline)
        lora_stack = krea2_pipeline.get("lora_stack", []) or []
        self._lora_runtime = None
        if lora_stack:
            self._lora_runtime = (lora_stack, realize_lora_stack(pipe, lora_stack))
        else:
            # No stack declared (stack node absent OR toggled off) -> the expected
            # state is NO LoRA effects. Enforce it instead of assuming it: a prior
            # run's failed/partial unload (or non-ephemeral leftovers) must not
            # leak into this run silently - that was exactly the 2026-07-21
            # "stack off but gens still contaminated" bug. Cheap no-op on a clean
            # pipe; assert_clean_baseline prints ONLY if residue survives.
            unload_all_loras(pipe)
            assert_clean_baseline(pipe)
        # Conditioning rebalance (opt-in): emphasize the positive prompt's
        # Qwen3-VL layer taps natively (community anti-dampening trick). Pops its
        # own kwargs so _generate_inner's signature is untouched. Also returns the
        # resolved (layer_weights, multiplier) so a precomputed `prompt_conditioning`
        # (e.g. from EricKrea2VisionPrompt) gets the SAME rebalance applied directly -
        # the encode_prompt wrap below only fires for the plain prompt-string path,
        # since a vision conditioning never calls encode_prompt at all.
        # Edit / negative guidance (2026-10-08): encode once (grounded instruction, NAG
        # negative, NegPiP weights, fitted sources) BEFORE cond_rebalance so the rebalance
        # reaches the precomputed positive. Installed inside the try below.
        edit_bundle = kwargs.pop("edit", None)
        guide_bundle = kwargs.pop("guidance", None)
        self._eg_rt = None
        _has_edit = isinstance(edit_bundle, dict) and bool(edit_bundle.get("images"))
        _busy = [k for k in ("ref_latents", "style", "paint") if isinstance(kwargs.get(k), dict)
                 and (kwargs.get(k).get("packed") or kwargs.get(k).get("image") is not None
                      or kwargs.get(k).get("known") is not None)]
        if _has_edit and _busy:
            raise ValueError(f"[EricKrea2-Edit] edit cannot be combined with {_busy} (each "
                             "wraps the transformer forward) - use one per run.")
        if isinstance(guide_bundle, dict) and any(k in _busy for k in ("style", "paint", "ref_latents")):
            print("[EricKrea2-EditGuide] negative guidance is skipped with style / paint / ref_latents "
                  "(each owns the transformer forward) - disconnect one of them.")
            guide_bundle = None
        if _has_edit or isinstance(guide_bundle, dict):
            from .._edit_install import prepare_edit_guidance
            try:
                self._eg_rt = prepare_edit_guidance(pipe, kwargs, edit_bundle if _has_edit else None,
                                                    guide_bundle, _resolve_s1_dims)
            except Exception:
                if lora_stack and any(e.get("ephemeral", True) for e in lora_stack):
                    unload_all_loras(pipe)
                raise
        reb_orig, reb_lw, reb_mult = self._install_cond_rebalance(pipe, kwargs)
        pc = kwargs.get("prompt_conditioning")
        if reb_lw is not None and isinstance(pc, dict) and pc.get("embeds") is not None:
            new_embeds = _rebalance_krea_embeds(pc["embeds"], reb_lw, reb_mult)
            kwargs["prompt_conditioning"] = {"embeds": new_embeds, "mask": pc.get("mask")}
            print(f"[EricKrea2-MS] cond_rebalance applied directly to precomputed prompt_conditioning "
                  f"{tuple(new_embeds.shape)}")
        # Reference latents (opt-in edit pathway): wrap transformer.forward so packed
        # reference tokens ride along at t=0 modulation (see _ref_latents.py). Popped
        # here so _generate_inner's signature stays untouched; the wrap covers BOTH
        # call paths (pipe(...) and _res_denoise_packed's direct transformer calls).
        ref_bundle = kwargs.pop("ref_latents", None)
        ref_match_size = kwargs.pop("ref_match_size", True)
        ref_orig = None
        if isinstance(ref_bundle, dict) and ref_bundle.get("packed"):
            from .._ref_latents import (install_ref_latents, resize_ref_bundle_to_dims,
                                        encode_refs_at_dims)
            ref_recipe = str(ref_bundle.get("recipe", "ostris_t0") or "ostris_t0")
            s1_w, s1_h = _resolve_s1_dims(
                kwargs.get("aspect_ratio", "5:4 landscape"), kwargs.get("s1_megapixels", 3.0),
                kwargs.get("width", 0), kwargs.get("height", 0), kwargs.get("init_latent"),
                kwargs.get("init_match_size", True), verbose=False)
            if ref_recipe == "edit_frame":
                # Edit recipe: the source grid MUST equal the generation grid
                # (trained on same-size pairs). Pixel-space fit + re-encode at the
                # resolved S1 size; ref_match_size (token-space) does not apply.
                ref_bundle = encode_refs_at_dims(pipe, ref_bundle, s1_w, s1_h)
            elif ref_match_size:
                ref_bundle = resize_ref_bundle_to_dims(pipe, ref_bundle, s1_w, s1_h)
            ref_orig = install_ref_latents(pipe, ref_bundle, recipe=ref_recipe)
        # Control-LoRA (opt-in): forward hooks only (expanded img_in + block LoRA
        # deltas), switched per stage from _lora_stage, removed in finally.
        control_bundle = kwargs.pop("control", None)
        style_bundle = kwargs.pop("style", None)
        paint_bundle = kwargs.pop("paint", None)
        self._control_rt = None
        self._style_rt = None
        self._paint_rt = None
        self._te_offload = None
        try:
            if isinstance(control_bundle, dict) and control_bundle.get("path"):
                from .._control import ControlRuntime
                self._control_rt = ControlRuntime(pipe, control_bundle)
                self._control_rt.install()
            if isinstance(style_bundle, dict) and style_bundle.get("image") is not None:
                if ref_orig is not None:
                    raise ValueError("[EricKrea2-Style] style reference and ref_latents both wrap the "
                                     "transformer - use one of them per run.")
                from .._style_ref import StyleRuntime
                self._style_rt = StyleRuntime(pipe, style_bundle)
                self._style_rt.install()
                self._te_offload = self._offload_text_encoder(pipe, krea2_pipeline, kwargs, "style")
            if isinstance(paint_bundle, dict) and paint_bundle.get("known") is not None:
                self._paint_rt = self._install_paint(pipe, paint_bundle, kwargs, lora_stack,
                                                     ref_orig is not None or self._style_rt is not None)
            if self._eg_rt is not None:
                self._eg_rt.install()
                if self._eg_rt.edit and self._te_offload is None:
                    self._te_offload = self._offload_text_encoder(pipe, krea2_pipeline, kwargs, "edit")
            return self._generate_inner(**kwargs)
        finally:
            if getattr(self, "_eg_rt", None) is not None:
                self._eg_rt.remove()
                self._eg_rt = None
            if self._te_offload is not None:
                self._restore_text_encoder(pipe, self._te_offload)
                self._te_offload = None
            if self._paint_rt is not None:
                self._paint_rt.remove()
                self._paint_rt = None
            if self._style_rt is not None:
                self._style_rt.remove()
                self._style_rt = None
            if self._control_rt is not None:
                self._control_rt.remove()
                self._control_rt = None
            if ref_orig is not None:
                pipe.transformer.forward = ref_orig
            if reb_orig is not None:
                pipe.encode_prompt = reb_orig
            if lora_stack and any(e.get("ephemeral", True) for e in lora_stack):
                unload_all_loras(pipe)
            self._lora_runtime = None

    @staticmethod
    def _offload_text_encoder(pipe, krea2_pipeline, kwargs, why, vram_limit_gb=64.0):
        """2026-10-08: style doubles the transformer batch, which overflows 48 GB cards with the
        ~8 GB Qwen3-VL encoder resident. On cards below vram_limit_gb, when no stage uses CFG
        (Turbo + turbo_guidance=off), encode the positive prompt ONCE (through the possibly
        rebalance-wrapped encode_prompt - identical to what every stage would compute), feed it
        as prompt_conditioning, and park the encoder on the CPU for the run. The pipeline's
        execution device is pinned to the transformer's device while the encoder is away
        (diffusers infers it from an unordered component set). Returns restore state or None."""
        try:
            tr_dev = next(pipe.transformer.parameters()).device
            te = getattr(pipe, "text_encoder", None)
            if te is None or tr_dev.type != "cuda":
                return None
            total_gb = torch.cuda.get_device_properties(tr_dev).total_memory / 1e9
            if total_gb >= float(vram_limit_gb):
                return None
            distilled = bool(krea2_pipeline.get("is_distilled", False))
            if not (distilled and str(kwargs.get("turbo_guidance", "off")) == "off"):
                print(f"[EricKrea2-MS] {why}: {total_gb:.0f} GB card but guidance is active - the "
                      "text encoder stays on the GPU (negative prompt needs it); may run out of VRAM.")
                return None
            pc = kwargs.get("prompt_conditioning")
            if not (isinstance(pc, dict) and pc.get("embeds") is not None):
                emb, msk = pipe.encode_prompt(prompt=kwargs.get("prompt", ""), device=tr_dev,
                                              num_images_per_prompt=1)
                kwargs["prompt_conditioning"] = {"embeds": emb, "mask": msk}
            te_dev = next(te.parameters()).device
            orig_cls = pipe.__class__
            pipe.__class__ = type(orig_cls.__name__, (orig_cls,),
                                  {"_execution_device": property(lambda s, _d=tr_dev: _d)})
            te.to("cpu")
            torch.cuda.empty_cache()
            print(f"[EricKrea2-MS] {why} on a {total_gb:.0f} GB card: prompt encoded once, text "
                  "encoder parked on the CPU for this run (restored after).")
            return (te, te_dev, orig_cls)
        except Exception as e:
            print(f"[EricKrea2-MS] text-encoder offload skipped ({type(e).__name__}: {e})")
            return None

    @staticmethod
    def _restore_text_encoder(pipe, state):
        te, te_dev, orig_cls = state
        try:
            pipe.__class__ = orig_cls
            te.to(te_dev)
        except Exception as e:
            print(f"[EricKrea2-MS] WARNING: text encoder restore failed ({type(e).__name__}: {e})")

    @staticmethod
    def _install_paint(pipe, paint_bundle, kwargs, lora_stack, other_wrap):
        """Inpaint / outpaint (2026-10-08): size S1 to the canvas, sanity-check the LoRA,
        install _paint.PaintRuntime. Mutates kwargs (width/height, crop_bottom)."""
        if other_wrap:
            raise ValueError("[EricKrea2-Paint] paint cannot be combined with style or ref_latents "
                             "(each wraps the transformer forward) - use one per run.")
        cw, ch = paint_bundle["canvas"]
        if int(kwargs.get("crop_bottom", 0) or 0) > 0:
            print("[EricKrea2-Paint] crop_bottom disabled for this run (a guard band would stretch "
                  "the canvas registration).")
            kwargs["crop_bottom"] = 0
        explicit = int(kwargs.get("width", 0) or 0) > 0 and int(kwargs.get("height", 0) or 0) > 0
        from_init = kwargs.get("init_latent") is not None and kwargs.get("init_match_size", True)
        if not explicit and not from_init:
            mp = float(kwargs.get("s1_megapixels", 3.0))
            a = cw / float(ch)
            h = (mp * 1_000_000.0 / a) ** 0.5
            kwargs["width"], kwargs["height"] = _round16(h * a), _round16(h)
            print(f"[EricKrea2-Paint] Stage 1 sized to the canvas aspect ({cw}x{ch}) at {mp:g} MP "
                  f"-> {kwargs['width']}x{kwargs['height']}")
        else:
            w, h = int(kwargs.get("width", 0) or cw), int(kwargs.get("height", 0) or ch)
            if abs((w / float(h)) / (cw / float(ch)) - 1.0) > 0.03:
                print(f"[EricKrea2-Paint] WARNING: Stage 1 aspect {w}x{h} differs from the canvas "
                      f"{cw}x{ch} - the canvas will be stretched to fit.")
        ad = paint_bundle.get("adapter", "none")
        if ad != "none":
            names = " ".join(str(e.get("path", "")).lower() for e in (lora_stack or []))
            if ad not in names:
                print(f"[EricKrea2-Paint] WARNING: adapter={ad} but no LoRA with '{ad}' in its name "
                      "is on the LoRA stack - the reference K/V will be built from the plain model "
                      "(add krea2_{ad}_rank32 to the Multi-LoRA stack, S1 = 1).")
        from .._paint import PaintRuntime
        rt = PaintRuntime(pipe, paint_bundle)
        rt.install()
        return rt

    @staticmethod
    def _load_cond_presets():
        """Load the rebalance preset library from cond_presets.json (package root).
        Tolerant: returns {} on any error, ignores unknown keys per entry."""
        import os, json
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cond_presets.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            return {k: v for k, v in d.items() if isinstance(v, dict)} if isinstance(d, dict) else {}
        except Exception:
            return {}

    def _install_cond_rebalance(self, pipe, kwargs):
        """Wrap ``pipe.encode_prompt`` so the POSITIVE prompt's conditioning is
        rescaled per the Krea2 layer-tap rebalance. Returns
        ``(orig_encode_prompt_or_None, layer_weights_or_None, multiplier)`` - the
        original bound method to restore in ``finally`` (or ``None`` when disabled),
        plus the resolved weights/multiplier so a precomputed ``prompt_conditioning``
        (which never calls encode_prompt) can have the same rebalance applied
        directly by the caller."""
        enabled = bool(kwargs.pop("cond_rebalance", False))
        multiplier = float(kwargs.pop("cond_multiplier", 4.0))
        weights_str = kwargs.pop("cond_layer_weights", "")
        preset = str(kwargs.pop("cond_preset", "custom"))
        if not enabled:
            return None, None, multiplier
        if preset and preset != "custom":
            p = self._load_cond_presets().get(preset)
            if p:
                if "multiplier" in p:
                    multiplier = float(p["multiplier"])
                if "weights" in p:
                    weights_str = ",".join(str(float(x)) for x in p["weights"])
                print(f"[EricKrea2-MS] cond_rebalance preset '{preset}': multiplier {multiplier}, "
                      f"{len(p.get('weights', []))} weights")
            else:
                print(f"[EricKrea2-MS] cond_rebalance: preset '{preset}' not in cond_presets.json; "
                      "using manual values.")
        try:
            lw = [float(x) for x in str(weights_str).replace(";", ",").split(",")
                  if x.strip() != ""]
        except ValueError:
            print("[EricKrea2-MS] cond_rebalance: could not parse layer weights; disabled.")
            return None, None, multiplier
        if not lw:
            print("[EricKrea2-MS] cond_rebalance: empty layer weights; disabled.")
            return None, None, multiplier

        def _norm(p):
            if isinstance(p, (list, tuple)):
                p = p[0] if p else ""
            return (p or "").strip()

        pos_n = _norm(kwargs.get("prompt", ""))
        orig = pipe.encode_prompt
        self._reb_hits = 0

        def _wrapped(*a, **k):
            p = k.get("prompt", a[0] if a else None)
            out = orig(*a, **k)
            if _norm(p) == pos_n and isinstance(out, tuple) and len(out) == 2:
                emb, mask = out
                new = _rebalance_krea_embeds(emb, lw, multiplier)
                self._reb_hits += 1
                if new is not emb:
                    print(f"[EricKrea2-MS] cond_rebalance applied #{self._reb_hits} "
                          f"to positive embeds {tuple(emb.shape)}")
                return new, mask
            return out

        pipe.encode_prompt = _wrapped
        print(f"[EricKrea2-MS] cond_rebalance ON: {len(lw)} layer gains x{multiplier} "
              f"global (positive conditioning, every stage).")
        return orig, lw, multiplier

    def _generate_inner(self, krea2_pipeline, prompt, prompt_conditioning=None, negative_prompt="", seed=0,
                 seed_mode="offset_per_stage",
                 aspect_ratio="5:4 landscape", s1_megapixels=3.00, width=0, height=0,
                 s1_steps=10, s1_cfg=1.9, s1_schedule="beta57", s1_sampler="res_2m", s1_noise="white",
                 init_latent=None, init_match_size=True, s1_start_step=0, s1_end_step=0,
                 crop_bottom=0, crop_overgen=False,
                 upscale_to_stage2=3.0, s2_steps=20, s2_cfg=1.5, s2_start_step=7, s2_end_step=20,
                 s2_schedule="linear", s2_sampler="euler", s2_noise="white",
                 upscale_to_stage3=2.0, s3_steps=20, s3_cfg=1.3, s3_start_step=9, s3_end_step=20,
                 s3_schedule="linear", s3_sampler="euler", s3_noise="white",
                 s1_eta=0.1, s2_eta=0.1, s3_eta=0.1, eta=None,
                 cond_jitter=0.0, cond_jitter_seed=-1,
                 turbo_guidance="off",
                 upscale_vae=None, upscale_vae_mode="disabled", s1_s2_upscale_vae=False,
                 decode_vae=None, preview_stages=False, sigmas=None,
                 distilled_shift="fixed", shift_mu_s1=1.15, shift_mu_s2=1.15, shift_mu_s3=1.15,
                 upscale_renorm="off", s1_hybrid_steps=0, s2_hybrid_steps=0,
                 s3_hybrid_steps=0, degrid="off", _s1_reuse=None):
        pipe = krea2_pipeline["pipeline"]
        _degrid_mode = str(degrid or "off")
        _degrid_hops = _degrid_mode in ("hops", "hops+final")
        _degrid_final = _degrid_mode in ("final", "hops+final")
        is_distilled = krea2_pipeline.get("is_distilled", False)
        # Distilled shift policy (2026-07-21): per-stage mu for Turbo models - see
        # _compute_mu. Always announce the resolved mode so the active policy is
        # visible in the console (a silent 'fixed' hid the pre-GUI env-var hook).
        shift_policy = {"mode": str(distilled_shift or "fixed").lower(),
                       "mu": {"s1": float(shift_mu_s1), "s2": float(shift_mu_s2),
                              "s3": float(shift_mu_s3)}}
        if is_distilled:
            _sp = shift_policy["mode"]
            if _sp == "manual":
                print(f"[EricKrea2-MS] distilled_shift=manual: mu s1={shift_mu_s1:.2f} "
                      f"s2={shift_mu_s2:.2f} s3={shift_mu_s3:.2f}")
            elif _sp == "resolution":
                print("[EricKrea2-MS] distilled_shift=resolution "
                      "(mu grows past 1.15 above the ~1.6MP calibration band)")
            else:
                print("[EricKrea2-MS] distilled_shift=fixed (mu=1.15 every stage)")
            if _sp != "fixed" and s1_sampler == "euler":
                print("[EricKrea2-MS] NOTE: Stage 1 TEXT-TO-IMAGE with sampler=euler runs "
                      "through the stock diffusers pipe (its own fixed mu=1.15), so the "
                      "policy reaches that one path only via a non-euler S1 sampler. All "
                      "WINDOWED stages (img2img S1, S2, S3) honor the policy with every "
                      "sampler, euler included (2026-07-22 shift-mismatch fix).")
        # Legacy fallback: older graphs saved a single global 'eta'. If explicitly passed,
        # it seeds all three per-stage etas (per-stage widgets override on new graphs).
        if eta is not None:
            s1_eta = s2_eta = s3_eta = float(eta)
        vsf = _VAE_SCALE_FACTOR

        # Precomputed conditioning (from Eric Krea2 Vision Prompt) - when connected, every
        # stage uses this instead of re-deriving from the `prompt` string. cond_rebalance was
        # already applied to it (if enabled) by generate() before this function was called.
        _cond_override = None
        if isinstance(prompt_conditioning, dict) and prompt_conditioning.get("embeds") is not None:
            _cond_override = (prompt_conditioning["embeds"], prompt_conditioning["mask"])
            print(f"[EricKrea2-MS] using precomputed prompt_conditioning {tuple(_cond_override[0].shape)} "
                  "instead of encoding 'prompt' as text (every stage).")
        elif isinstance(prompt, str) and prompt.strip():
            _report_prompt_tokens(pipe, prompt)

        # LoRA per-stage weights (no-op unless generate() realized a stack)
        def _lora_stage(idx):
            rt = getattr(self, "_lora_runtime", None)
            if rt and rt[0]:
                from .._lora_utils import set_stack_stage_weights
                set_stack_stage_weights(pipe, rt[0], rt[1], idx)
            crt = getattr(self, "_control_rt", None)
            if crt is not None:
                crt.set_stage(idx)
            srt = getattr(self, "_style_rt", None)
            if srt is not None:
                srt.set_stage(idx)
            prt = getattr(self, "_paint_rt", None)
            if prt is not None:
                prt.set_stage(idx)
            ert = getattr(self, "_eg_rt", None)
            if ert is not None:
                ert.set_stage(idx)

        if is_distilled:
            if turbo_guidance == "off":
                if max(s1_cfg, s2_cfg, s3_cfg) > 0:
                    print("[EricKrea2-MS] Distilled/Turbo: turbo_guidance=off, forcing all guidance=0 "
                          "(negative prompt inactive). Set turbo_guidance=s1_only or all_stages to use it.")
                s1_cfg = s2_cfg = s3_cfg = 0.0
            elif turbo_guidance == "s1_only":
                s2_cfg = s3_cfg = 0.0
                if s1_cfg > 0:
                    print(f"[EricKrea2-MS] turbo_guidance=s1_only: S1 guidance g={s1_cfg} active "
                          "(negative prompt on for composition, ~2x S1 time); S2/S3 forced g=0 (fast, "
                          "single-pass). Krea2 g: standard CFG = 1+g, Turbo distilled for g=0 so keep low "
                          "(~0.1-0.3).")
                    if s1_cfg > 1.5:
                        print(f"[EricKrea2-MS]   WARNING: s1_cfg g={s1_cfg} (== CFG {1+s1_cfg:.1f}) is "
                              "Raw-level on Turbo; likely over-saturated. Lower toward g~0.2.")
            else:  # all_stages
                if max(s1_cfg, s2_cfg, s3_cfg) > 0:
                    print("[EricKrea2-MS] turbo_guidance=all_stages: guidance on a distilled checkpoint "
                          "(negative active throughout; each guided stage ~2x slower). Krea2 g: standard "
                          "CFG = 1+g; Turbo distilled for g=0, keep low (~0.1-0.3).")
                    if max(s1_cfg, s2_cfg, s3_cfg) > 1.5:
                        print(f"[EricKrea2-MS]   WARNING: cfg up to g={max(s1_cfg, s2_cfg, s3_cfg):.1f} "
                              "(== CFG > 2.5) is Raw-level on Turbo; likely over-saturated.")

        neg = (negative_prompt or "").strip() or None

        if any(nt != "white" for nt in (s1_noise, s2_noise, s3_noise)):
            print("[EricKrea2-MS] NOTE: non-white noise now shapes the SDE ancestral-churn noise "
                  f"only (per-step re-injection when eta>0, leaning {int(_NOISE_STRENGTH*100)}% off "
                  "white), NOT the primary re-noise/init - the primary substitution is always plain "
                  "white now (matches RES4LYF's own noise_type_sde: colored types are for that "
                  "per-step SDE role, not the dominant re-noise, which is why using it there "
                  "produced color-swirl artifacts instead of a composition bias). Has NO effect "
                  "with eta=0 or sampler=euler (no SDE churn hook in either case).")

        s1_w, s1_h = _resolve_s1_dims(aspect_ratio, s1_megapixels, width, height, init_latent,
                                      init_match_size)
        # bottom-band crop. overgen grows S1 by a REAL guard band (extra rows the model generates
        # into), carries it through both upscales, and trims the accumulated margin off the final
        # stage - so the boundary band forms in the discarded margin at EVERY stage and the target
        # region stays interior (clean). 1 latent row = 8 px; even count (2x2 patch stride).
        _crop_rows = max(0, int(crop_bottom))
        if _crop_rows % 2:
            _crop_rows += 1
        _crop_px = _crop_rows * 8
        do_s2 = upscale_to_stage2 > 0
        do_s3 = do_s2 and upscale_to_stage3 > 0
        # S1->S2 via the trained 2x VAE forces a 2x linear step (4x area), overriding upscale_to_stage2.
        use_s1s2_vae = bool(s1_s2_upscale_vae) and (upscale_vae is not None) and do_s2
        _overgen = _crop_rows > 0 and crop_overgen

        def _chain_dims(_s1h):
            if do_s2 and use_s1s2_vae:
                _s2 = (_round16(s1_w * 2), _round16(_s1h * 2))
            elif do_s2:
                _s2 = _scaled_dims(s1_w, _s1h, upscale_to_stage2)
            else:
                _s2 = (0, 0)
            _s3 = _scaled_dims(_s2[0], _s2[1], upscale_to_stage3) if do_s3 else (0, 0)
            return _s2[0], _s2[1], _s3[0], _s3[1]

        s2_w, s2_h, s3_w, s3_h = _chain_dims(s1_h)
        last_target_h = s3_h if do_s3 else (s2_h if do_s2 else s1_h)
        if _overgen:                       # grow S1 + rebuild the chain proportionally
            s1_h = _round16(s1_h + _crop_px)
            s2_w, s2_h, s3_w, s3_h = _chain_dims(s1_h)

        # -- upscale VAE mode (needs a connected upscale_vae) --
        mode = upscale_vae_mode
        if mode != "disabled" and upscale_vae is None:
            print(f"[EricKrea2-MS] upscale_vae_mode={mode} but no upscale_vae connected; "
                  "falling back to bislerp + standard decode.")
            mode = "disabled"
        _s2s3_vae, _s2s3_down, _final_vae, _final_down = _UPSCALE_VAE_MODE_FLAGS.get(
            mode, (False, False, False, False))
        if _s2s3_vae and not do_s3:
            print("[EricKrea2-MS] s2-s3 VAE upscale needs 3 stages; S2->S3 will use bislerp instead.")
        if s1_s2_upscale_vae and upscale_vae is None:
            print("[EricKrea2-MS] s1_s2_upscale_vae set but no upscale_vae connected; S1->S2 uses bislerp.")
        use_inter_stage = _s2s3_vae and do_s3
        inter_downsample = use_inter_stage and _s2s3_down
        use_final_decode = _final_vae
        final_downsample = _final_vae and _final_down

        device = getattr(pipe, "_execution_device", None) or "cuda"

        # Conditioning jitter (opt-in): perturb the POSITIVE conditioning once, then
        # thread it through the existing cond_override path so every stage uses the
        # SAME jittered conditioning (per-stage jitter would make S2/S3 refine toward
        # a different point than S1 composed). Encoding here goes through the (possibly
        # rebalance-wrapped) encode_prompt, so the order is rebalance -> jitter.
        if cond_jitter and float(cond_jitter) > 0:
            _jseed = int(cond_jitter_seed)
            if _jseed < 0:
                _jseed = (int(seed) + 777) if int(seed) > 0 else int(
                    torch.randint(0, 2**31 - 1, (1,)).item())
            _jgen = torch.Generator(device=device).manual_seed(int(_jseed))
            if _cond_override is None:
                _je, _jm = pipe.encode_prompt(prompt=prompt, device=device,
                                              num_images_per_prompt=1)
            else:
                _je, _jm = _cond_override
                _je = _je.to(device)
            _je = _jitter_embeds(_je, float(cond_jitter), _jgen)
            _cond_override = (_je, _jm)
            print(f"[EricKrea2-MS] cond_jitter: sigma={float(cond_jitter):.3f} x per-tap std, "
                  f"seed={_jseed}, applied once to positive conditioning "
                  f"{tuple(_je.shape)} (all stages)")

        # -- per-stage seeds --
        def _make_generator(s):
            return torch.Generator(device=device).manual_seed(int(s)) if s > 0 else None
        if seed_mode == "offset_per_stage":
            gen_s1 = _make_generator(seed)
            gen_s2 = _make_generator(seed + 1 if seed > 0 else 0)
            gen_s3 = _make_generator(seed + 2 if seed > 0 else 0)
            seed_info = f"offset S1={seed}, S2={seed+1 if seed>0 else 'rand'}, S3={seed+2 if seed>0 else 'rand'}"
        elif seed_mode == "random_per_stage":
            gen_s1, gen_s2, gen_s3 = _make_generator(seed), None, None
            seed_info = f"random S1={seed if seed>0 else 'rand'}, S2/S3=rand"
        else:  # same_all_stages (one generator threaded through)
            gen_s1 = _make_generator(seed)
            gen_s2 = gen_s3 = gen_s1
            seed_info = f"same {seed if seed>0 else 'rand'}"

        n_stages = 3 if do_s3 else (2 if do_s2 else 1)
        _s2_disp = f"{s2_w}x{s2_h} ({s2_w*s2_h/1e6:.2f} MP){' VAE2x' if use_s1s2_vae else ''}"
        if not do_s3:
            _s3_disp = ""
        elif use_inter_stage and inter_downsample:
            _s3_disp = f"VAE2x->ds {s3_w}x{s3_h} ({s3_w*s3_h/1e6:.2f} MP)"
        elif use_inter_stage:
            _s3_disp = "VAE2x (4x area)"
        else:
            _s3_disp = f"{s3_w}x{s3_h} ({s3_w*s3_h/1e6:.2f} MP)"
        _dec_disp = ("VAE2x-ds->native" if final_downsample else
                     ("VAE2x decode" if use_final_decode else "standard decode"))
        print(f"[EricKrea2-MS] {aspect_ratio}, {n_stages} stage(s), seed_mode={seed_mode} ({seed_info}), "
              f"upscale_vae_mode={mode}, decode={_dec_disp}:")
        print(f"[EricKrea2-MS]   S1 {s1_w}x{s1_h} ({s1_w*s1_h/1e6:.2f} MP) [{s1_schedule}/{s1_noise}]"
              + (f" -> S2 {_s2_disp} [{s2_schedule}/{s2_noise}]" if do_s2 else "")
              + (f" -> S3 {_s3_disp} [{s3_schedule}/{s3_noise}]" if do_s3 else ""))

        import comfy.utils
        def _win_len(steps, start, end):
            steps = max(1, int(steps))
            start = max(0, min(int(start), steps - 1))
            end = max(start + 1, min(int(end), steps))
            return end - start
        _s1_run = (_win_len(s1_steps, s1_start_step, (int(s1_end_step) or int(s1_steps)))
                   if init_latent is not None else int(s1_steps))
        total = (_s1_run
                 + (_win_len(s2_steps, s2_start_step, s2_end_step) if do_s2 else 0)
                 + (_win_len(s3_steps, s3_start_step, s3_end_step) if do_s3 else 0))
        pbar = comfy.utils.ProgressBar(total)
        _telem = _telemetry.StageTracker(device)

        def make_cb():
            def on_step_end(_pipe, step_idx, _t, cb_kwargs):
                pbar.update(1)
                _telem.step()
                _check_cancelled()
                return cb_kwargs
            return on_step_end

        def _res_progress():
            pbar.update(1)
            _telem.step()
            _check_cancelled()

        def _resolve_sched(stage_key, steps, default_sched):
            """Per-stage sigma-curve resolution. Returns (sched_arg, label): a plain
            schedule STRING when the optional `sigmas` bundle doesn't override this
            stage (unchanged default path), or a precomputed raw sigma LIST + starred
            label when it does."""
            curve, params = _sigmas.resolve_stage(sigmas, stage_key, default_sched)
            if params:  # bundle override active for this stage
                return _sigmas.build_from_spec(int(steps), 1.0, curve, params), f"{curve}*"
            return default_sched, default_sched

        def _denoise_stage(up_lat, dst_h, dst_w, steps, cfg, start_step, end_step,
                           sched, noise_type, gen, tag, sampler, sched_label=None, stage_eta=0.0,
                           stage_key=None):
            """Re-noise an upscaled latent to a chosen step of a `steps`-length schedule,
            then denoise the [start_step, end_step] window (reference-workflow handoff).
            start_step sets the injection / re-noise level (replaces denoise-strength);
            end_step < steps stops early and hands a still-noisy latent to the next stage
            (honoured by every sampler; euler runs as a 1-stage RK). `sched` may be a schedule
            name (str) or a precomputed raw sigma list (from a KREA2_SIGMAS override)."""
            _check_cancelled()
            steps = max(1, int(steps))
            start_step = max(0, min(int(start_step), steps - 1))
            end_step = max(start_step + 1, min(int(end_step), steps))
            # full schedule with the chosen spacing, shifted exactly as pipe.__call__ would
            full_raw = build_sigma_schedule(steps, 1.0, sched) if isinstance(sched, str) else list(sched)
            label = sched_label or (sched if isinstance(sched, str) else "custom")
            mu = _compute_mu(_packed_seq_len(dst_h, dst_w, vsf), pipe.scheduler, is_distilled,
                             policy=shift_policy, stage=stage_key)
            pipe.scheduler.set_timesteps(sigmas=list(full_raw), device=device, mu=mu)
            full_shifted = pipe.scheduler.sigmas.to(device=device, dtype=torch.float32)  # [steps+1], ->0
            window = full_shifted[start_step:end_step + 1].clone()   # shifted sigmas for this window
            start_sigma = float(window[0])
            end_sigma = float(window[-1])
            ends_clean = end_sigma <= 1e-6
            h_lat, w_lat = 2 * (dst_h // 16), 2 * (dst_w // 16)
            # Primary re-noise is ALWAYS plain white - matches what the flow-match denoiser
            # was trained to invert. `noise_type` now only shapes the SDE ancestral-churn
            # noise passed to _res_denoise_packed below (eta>0, non-euler samplers only) -
            # see that function's docstring for why colored noise as the dominant
            # re-noise/init substitution produced the color-swirl artifacts this used to.
            noise = _shaped_packed_noise(up_lat.shape[0], h_lat, w_lat, "white", gen, device).to(up_lat.dtype)
            noised = _add_noise_flowmatch(up_lat, noise, start_sigma)
            print(f"[EricKrea2-MS] -- {tag}: {dst_w}x{dst_h}, steps {start_step}->{end_step} of {steps} "
                  f"({end_step - start_step} denoise steps), guidance={cfg}, {label}/{noise_type}, "
                  f"sampler={sampler}, start_sigma={start_sigma:.3f}, end_sigma={end_sigma:.3f}"
                  f"{'' if ends_clean else ' (NOISY handoff)'} --")
            _telem.begin(tag)
            # ALL samplers - euler included (as a 1-stage RK tableau) - go through our
            # solver with the SAME shifted window used for the re-noise above. The old
            # euler special-case handed `noised` to pipe(...), which re-shifts the raw
            # window with its own FIXED mu=1.15; under a per-stage shift policy the
            # re-noise level and the solver's first sigma then disagreed, and the
            # mismatch grew with the policy delta - mild noise at S2, full garble at
            # S3 (the 2026-07-22 shift-mismatch bug). One path, one sigma space.
            return _res_denoise_packed(pipe, prompt, neg, dst_h, dst_w, noised, None,
                                       is_distilled, float(cfg), float(stage_eta), gen,
                                       progress_cb=_res_progress, sigma_window=window,
                                       method=sampler, cond_override=_cond_override,
                                       noise_type=noise_type,
                                       hybrid_lcm_steps={"s1": s1_hybrid_steps,
                                                         "s2": s2_hybrid_steps,
                                                         "s3": s3_hybrid_steps}[stage_key],
                                       shift_policy=shift_policy, shift_stage=stage_key)

        def _latent_dict(lat, h, w):
            return {"packed": lat.detach(), "height": h, "width": w, "vae_scale_factor": vsf}

        # crop the accumulated guard band off the LAST executed stage.
        last_stage = 3 if do_s3 else (2 if do_s2 else 1)

        def _final_crop(lat, cur_h, w):
            if _crop_rows <= 0:
                return lat, cur_h
            target_h = max(16, last_target_h if _overgen else cur_h - _crop_px)
            drop = (cur_h - target_h) // 8      # even (both heights are /16)
            if drop <= 0:
                return lat, cur_h
            keep = ((cur_h // 16) - drop // 2) * (w // 16)
            print(f"[EricKrea2-MS]   final crop: {w}x{cur_h} -> {w}x{target_h} (-{drop} rows)")
            return lat[:, :keep, :].contiguous(), target_h

        stage_latents = {1: None, 2: None}

        # Stage 1 - img2img (re-noise a supplied init latent) or draft from noise
        _check_cancelled()
        _lora_stage(1)
        # Sweep S1-latent reuse (2026-07-24, sweep v1.1): when the sweep driver
        # supplies a captured post-S1 latent (identical panel S1, fixed seed,
        # cell overrides strictly s2_*/s3_*), skip the Stage 1 denoise entirely.
        # Refused under same_all_stages seed_mode: S2 shares S1's generator
        # OBJECT there, so skipping S1 would leave it un-advanced and change
        # S2's noise draw vs a full run (cells would stop being comparable).
        if _s1_reuse is not None and seed_mode == "same_all_stages":
            print("[EricKrea2-MS] S1 reuse disabled: seed_mode=same_all_stages would "
                  "change Stage 2 noise vs a full run - running Stage 1 normally.")
            _s1_reuse = None
        if _s1_reuse is not None:
            cur_lat = _s1_reuse["packed"].to(device)
            print(f"[EricKrea2-MS] -- Stage 1: REUSED sweep latent "
                  f"{int(_s1_reuse['width'])}x{int(_s1_reuse['height'])} (denoise skipped) --")
        elif init_latent is not None:
            # img2img: resize the init latent to the S1 grid and run it through the SAME
            # re-noise/denoise-window mechanic S2/S3 use. s1_start_step is the denoise-strength
            # control (0 = ~pure noise/text-to-image; higher preserves more of the source).
            _il_h = int(init_latent.get("height", s1_h))
            _il_w = int(init_latent.get("width", s1_w))
            init_up = _upscale_latents(init_latent["packed"].to(device), _il_h, _il_w,
                                       s1_h, s1_w, vsf)
            _s1_end = int(s1_end_step) or int(s1_steps)
            _dn = max(0, int(s1_steps) - int(s1_start_step)) / max(1, int(s1_steps))
            print(f"[EricKrea2-MS] Stage 1 img2img: init {_il_w}x{_il_h} -> {s1_w}x{s1_h}, "
                  f"re-noise at step {s1_start_step}/{s1_steps} (denoise ~{_dn:.0%})")
            _s1_sched, _s1_label = _resolve_sched("s1", int(s1_steps), s1_schedule)
            cur_lat = _denoise_stage(init_up, s1_h, s1_w, s1_steps, s1_cfg, s1_start_step, _s1_end,
                                     _s1_sched, s1_noise, gen_s1, "Stage 1 (img2img)", s1_sampler,
                                     sched_label=_s1_label, stage_eta=s1_eta, stage_key="s1")
        else:
            _s1_sched, _s1_label = _resolve_sched("s1", int(s1_steps), s1_schedule)
            s1_sigmas = (build_sigma_schedule(int(s1_steps), 1.0, _s1_sched)
                         if isinstance(_s1_sched, str) else list(_s1_sched))
            # Primary init is ALWAYS plain white (let the sampler/pipe draw its own) - matches
            # what the flow-match denoiser was trained on. s1_noise now only shapes the SDE
            # ancestral-churn noise (eta>0, non-euler samplers) - see _res_denoise_packed's
            # docstring for why colored noise as the dominant starting substitution produced
            # visible color-swirl artifacts instead of a composition bias.
            s1_init = None
            _crop_msg = ""
            if _crop_rows > 0:
                _crop_msg = (f", +{_crop_rows}row guard band (overgen)" if _overgen
                             else f", trim {_crop_rows}row at final")
            print(f"[EricKrea2-MS] -- Stage 1: {s1_w}x{s1_h}, {s1_steps} steps, "
                  f"guidance={s1_cfg}, {_s1_label}/{s1_noise}, sampler={s1_sampler}{_crop_msg} --")
            _telem.begin("Stage 1")
            # Paint (2026-10-08): the stock pipe(...) Euler loop keeps latents in bf16; the paint
            # adapters' reference behaviour is an fp32 trajectory (parity 17 dB -> 36 dB vs the
            # author's pipeline). With paint active, Euler runs on our solver (fp32 accumulator,
            # Euler as a 1-stage RK). Non-paint runs are unchanged (seed-reproducible).
            if (s1_sampler != "euler" or getattr(self, "_paint_rt", None) is not None
                    or getattr(self, "_eg_rt", None) is not None):
                cur_lat = _res_denoise_packed(pipe, prompt, neg, s1_h, s1_w, s1_init, s1_sigmas,
                                              is_distilled, float(s1_cfg), float(s1_eta), gen_s1,
                                              progress_cb=_res_progress, method=s1_sampler,
                                              cond_override=_cond_override, noise_type=s1_noise,
                                              hybrid_lcm_steps=s1_hybrid_steps,
                                              shift_policy=shift_policy, shift_stage="s1")
            else:
                if s1_noise != "white" and s1_eta and s1_eta > 0:
                    print(f"[EricKrea2-MS]    note: s1_noise={s1_noise} has no effect with "
                          "sampler=euler (no SDE churn hook). Use res_2m/res_2s/deis_* to apply it.")
                _prompt_kwargs = ({"prompt_embeds": _cond_override[0], "prompt_embeds_mask": _cond_override[1]}
                                 if _cond_override is not None else {"prompt": prompt})
                s1 = pipe(
                    negative_prompt=neg, height=s1_h, width=s1_w,
                    num_inference_steps=int(s1_steps), sigmas=s1_sigmas,
                    guidance_scale=float(s1_cfg), generator=gen_s1, latents=s1_init,
                    callback_on_step_end=make_cb(), output_type="latent",
                    **_prompt_kwargs,
                )
                cur_lat = s1.images
        if last_stage == 1:
            cur_lat, s1_hf = _final_crop(cur_lat, s1_h, s1_w)
            cur_h, cur_w = s1_hf, s1_w
        else:
            cur_h, cur_w = s1_h, s1_w
        stage_latents[1] = _latent_dict(cur_lat, cur_h, cur_w)

        if do_s2:
            _lora_stage(2)
            if use_s1s2_vae:
                print("[EricKrea2-MS]   Inter-stage VAE upscale (S1->S2, trained 2x) ...")
                up2, s2_h, s2_w = upscale_between_stages(cur_lat, upscale_vae, pipe.vae, cur_h, cur_w, vsf,
                                                         degrid=_degrid_hops)
            else:
                up2 = _upscale_latents(cur_lat, cur_h, cur_w, s2_h, s2_w, vsf)
            up2 = _renorm_hop(up2, cur_lat, upscale_renorm, "S1->S2")
            _s2_sched, _s2_label = _resolve_sched("s2", s2_steps, s2_schedule)
            cur_lat = _denoise_stage(up2, s2_h, s2_w, s2_steps, s2_cfg, s2_start_step, s2_end_step,
                                     _s2_sched, s2_noise, gen_s2, "Stage 2", s2_sampler,
                                     sched_label=_s2_label, stage_eta=s2_eta, stage_key="s2")
            if last_stage == 2:
                cur_lat, s2_h = _final_crop(cur_lat, s2_h, s2_w)
            cur_h, cur_w = s2_h, s2_w
            stage_latents[2] = _latent_dict(cur_lat, cur_h, cur_w)
        if do_s3:
            _lora_stage(3)
            if use_inter_stage:
                if inter_downsample:
                    print(f"[EricKrea2-MS]   Inter-stage VAE upscale (S2->S3, 2x then downsample to "
                          f"{s3_w}x{s3_h}) ...")
                    up3, cur_h3, cur_w3 = upscale_between_stages(
                        cur_lat, upscale_vae, pipe.vae, cur_h, cur_w, vsf,
                        target_h=s3_h, target_w=s3_w, degrid=_degrid_hops)
                else:
                    print("[EricKrea2-MS]   Inter-stage VAE upscale (S2->S3, trained 2x) ...")
                    up3, cur_h3, cur_w3 = upscale_between_stages(cur_lat, upscale_vae, pipe.vae, cur_h, cur_w, vsf,
                                                                 degrid=_degrid_hops)
            else:
                cur_h3, cur_w3 = s3_h, s3_w
                up3 = _upscale_latents(cur_lat, cur_h, cur_w, cur_h3, cur_w3, vsf)
            up3 = _renorm_hop(up3, cur_lat, upscale_renorm, "S2->S3")
            _s3_sched, _s3_label = _resolve_sched("s3", s3_steps, s3_schedule)
            cur_lat = _denoise_stage(up3, cur_h3, cur_w3, s3_steps, s3_cfg, s3_start_step, s3_end_step,
                                     _s3_sched, s3_noise, gen_s3, "Stage 3", s3_sampler,
                                     sched_label=_s3_label, stage_eta=s3_eta, stage_key="s3")
            cur_lat, cur_h3 = _final_crop(cur_lat, cur_h3, cur_w3)
            cur_h, cur_w = cur_h3, cur_w3

        _telem.close()
        if use_final_decode:
            print(f"[EricKrea2-MS]   Final upscale-VAE decode "
                  f"({'2x then downsample to native' if final_downsample else '2x'}) ...")
            image = decode_latents_with_upscale_vae(cur_lat, upscale_vae, pipe.vae, cur_h, cur_w, vsf,
                                                    downsample_override=(True if final_downsample else None))
        else:
            image = standard_decode(pipe, cur_lat, cur_h, cur_w, decode_vae=decode_vae)
        if _degrid_final:
            from .._degrid import degrid_image, format_stats
            try:
                image, _dg_stats = degrid_image(image)
                print(format_stats(_dg_stats, "final image"))
            except Exception as _e:  # cleanup pass must never kill a finished generation
                print(f"[EricKrea2-DeGrid] final pass skipped ({type(_e).__name__}: {_e})")

        final_latent = _latent_dict(cur_lat, cur_h, cur_w)
        s1_latent = stage_latents[1] or final_latent
        s2_latent = stage_latents[2] or final_latent

        # Optional per-stage preview images (1x decode with the same decode VAE)
        if preview_stages:
            print("[EricKrea2-MS]   Decoding stage previews (1x) ...")
            s1_img = standard_decode(pipe, s1_latent["packed"], s1_latent["height"],
                                     s1_latent["width"], decode_vae=decode_vae)
            if stage_latents[2] is not None:
                s2_img = standard_decode(pipe, s2_latent["packed"], s2_latent["height"],
                                         s2_latent["width"], decode_vae=decode_vae)
            else:
                s2_img = image  # stage 2 didn't run -> S2 == final
        else:
            s1_img = s2_img = torch.zeros(1, 1, 1, 3)

        print(f"[EricKrea2-MS] Done: final image {image.shape[2]}x{image.shape[1]} px"
              + (" (+ stage previews)" if preview_stages else ""))
        return (image, final_latent, s1_img, s2_img, s1_latent, s2_latent)


NODE_CLASS_MAPPINGS = {"EricKrea2MultistageUltra": EricKrea2MultistageUltra}
NODE_DISPLAY_NAME_MAPPINGS = {"EricKrea2MultistageUltra": "Eric Krea2 Multi-Stage Ultra (res)"}
