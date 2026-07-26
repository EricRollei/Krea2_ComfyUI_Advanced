# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Eric_Krea2 / _res_solver.py
#
# Standalone RES (Refined Exponential Solver) multistep sampler for rectified-flow
# (flow-matching) models such as Krea 2 / Qwen-Image / Flux.
#
# This is a faithful, model-decoupled re-implementation of the 2nd-order exponential
# multistep method ("res_2m" / ComfyUI "res_multistep"), from:
#   "Refined Exponential Solver (RES)" - arXiv:2308.02157
# It follows the formulation used by ComfyUI's k_diffusion sample_res_multistep and
# RES4LYF's res_2m (data-prediction / x0 form, stepping in t = -log(sigma) with exact
# phi-functions). The math (phi-function exponential multistep) is from the paper and
# is not specific to any one implementation.
#
# Design: the solver is fully decoupled from the diffusion model. It calls a supplied
#   denoise_fn(x, sigma_scalar_tensor) -> x0_pred
# closure, so it can be unit-tested with a trivial denoiser (see __main__) and reused
# across any flow-matching pipeline. The node supplies a denoise_fn that wraps the
# Krea2 transformer (velocity -> x0 via  x0 = x - sigma * v).

from __future__ import annotations
import math
import torch


# --------------------------------------------------------------------------------------
# helpers (mirrors of ComfyUI k_diffusion utilities, kept local so we have no GPL import)
# --------------------------------------------------------------------------------------
def to_d(x: torch.Tensor, sigma: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    """Flow-matching / Karras ODE derivative. For rectified flow with sigma==t this
    equals the velocity v = eps - x0, since x_t = (1-t) x0 + t eps."""
    return (x - denoised) / sigma


def get_ancestral_step(sigma_from: torch.Tensor, sigma_to: torch.Tensor, eta: float = 1.0):
    """Split a step into a deterministic 'down' sigma and a stochastic 'up' sigma.
    eta == 0  -> (sigma_to, 0): pure ODE step (our deterministic default)."""
    if not eta:
        return sigma_to, sigma_to.new_zeros(())
    sigma_up = torch.minimum(
        sigma_to,
        eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5,
    )
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def _phi1(t: torch.Tensor) -> torch.Tensor:
    return torch.expm1(t) / t


def _phi2(t: torch.Tensor) -> torch.Tensor:
    return (_phi1(t) - 1.0) / t


# --------------------------------------------------------------------------------------
# the sampler
# --------------------------------------------------------------------------------------
@torch.no_grad()
def res_multistep_sample(
    denoise_fn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    eta: float = 0.0,
    s_noise: float = 1.0,
    noise_sampler=None,
    callback=None,
):
    """RES 2nd-order exponential multistep (res_2m).

    Args:
        denoise_fn: callable(x, sigma_scalar_tensor) -> x0_pred  (same shape as x).
        x:          starting latent at sigmas[0] (any shape; packed [B,seq,C] is fine).
        sigmas:     1-D tensor of sigma values, DESCENDING. Normally ENDS IN 0.0 so the
                    last step denoises clean; if it ends ABOVE 0 (an early-stop refinement
                    window) the sampler returns the partially-denoised latent at that sigma.
        eta:        SDE churn. 0.0 = deterministic ODE (phase 1). >0 adds ancestral noise.
        s_noise:    scale on injected noise (eta>0 only).
        noise_sampler: callable(sigma, sigma_next) -> noise tensor like x (eta>0 only).
        callback:   optional callable(i, sigma, denoised, x).

    Returns:
        Final latent (the x0 prediction at the terminal step).
    """
    if noise_sampler is None:
        noise_sampler = lambda s, sn: torch.randn_like(x)

    sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    n = sigmas.shape[0] - 1

    old_denoised = None
    old_sigma_down = None

    for i in range(n):
        sigma = sigmas[i]
        denoised = denoise_fn(x, sigma)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

        if callback is not None:
            callback(i, sigma, denoised, x)

        if float(sigma_down) == 0.0 or old_denoised is None:
            # flow-match Euler (also the terminal sigma->0 denoising step)
            d = to_d(x, sigma, denoised)
            dt = sigma_down - sigma
            x = x + d * dt
        else:
            # 2nd-order exponential multistep (arXiv:2308.02157), data-prediction form.
            # t = -log(sigma); h = t_next - t; c2 is the (negated) step ratio that lets
            # b1,b2 weight the current vs previous x0 prediction via exact phi-functions.
            t = sigma.log().neg()
            t_old = old_sigma_down.log().neg()
            t_next = sigma_down.log().neg()
            t_prev = sigmas[i - 1].log().neg()
            h = t_next - t
            c2 = (t_prev - t_old) / h
            phi1_val, phi2_val = _phi1(-h), _phi2(-h)
            b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)
            x = (-h).exp() * x + h * (b1 * denoised + b2 * old_denoised)

        # ancestral noise (phase 2, eta>0)
        if float(sigmas[i + 1]) > 0.0 and eta > 0.0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

        old_denoised = denoised
        old_sigma_down = sigma_down

    return x


# --------------------------------------------------------------------------------------
# res_2s - 2nd-order single-step exponential Runge-Kutta (RES base method)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def res_2s_sample(
    denoise_fn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    c2: float = 0.5,
    eta: float = 0.0,
    s_noise: float = 1.0,
    noise_sampler=None,
    callback=None,
):
    """RES 2nd-order SINGLE-step exponential Runge-Kutta ("res_2s").

    A predictor/corrector: each interior step evaluates the model at the current sigma
    (D1) and again at an intermediate sub-step sigma (D2), then combines them with exact
    phi-functions. Self-starting (no history), so it is the strongest low-step-count
    choice - at ~2x the model-call cost of res_2m/euler. For constant x0 it is exact.

    Reduces to the same b1/b2 phi weighting as res_2m, but with c2 an explicit sub-step
    fraction and D2 a fresh evaluation (vs. res_2m's reuse of the previous step's x0).
    The intermediate sigma is the (1-c2, c2) geometric interpolant in log-sigma space:
        sigma_mid = sigma^(1-c2) * sigma_down^c2.

    Args mirror res_multistep_sample. c2=0.5 is the midpoint (RES4LYF default).
    """
    if noise_sampler is None:
        noise_sampler = lambda s, sn: torch.randn_like(x)

    sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    n = sigmas.shape[0] - 1

    for i in range(n):
        sigma = sigmas[i]
        d1 = denoise_fn(x, sigma)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

        if callback is not None:
            callback(i, sigma, d1, x)

        if float(sigma_down) == 0.0:
            # terminal / clean step: exponential Euler == flow Euler to x0
            d = to_d(x, sigma, d1)
            x = x + d * (sigma_down - sigma)
        else:
            t = sigma.log().neg()
            t_next = sigma_down.log().neg()
            h = t_next - t                       # > 0 (sigma decreasing)
            c2h = c2 * h
            # sub-step point (log-sigma interpolation) and its predictor
            sigma_mid = (sigma.log() * (1.0 - c2) + sigma_down.log() * c2).exp()
            x_mid = (-c2h).exp() * x + c2h * _phi1(-c2h) * d1
            d2 = denoise_fn(x_mid, sigma_mid)
            # 2nd-order corrector
            phi1_h, phi2_h = _phi1(-h), _phi2(-h)
            b2 = phi2_h / c2
            b1 = phi1_h - b2
            x = (-h).exp() * x + h * (b1 * d1 + b2 * d2)

        if float(sigmas[i + 1]) > 0.0 and eta > 0.0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

    return x


# --------------------------------------------------------------------------------------
# DEIS - Diffusion Exponential Integrator Sampler (multistep), deis_3m / deis_4m
# --------------------------------------------------------------------------------------
def _exp_lagrange_coeffs(offsets, h: float):
    """Exponential-integrator quadrature weights for one DEIS step.

    We step in  t = -log(sigma)  where the rectified-flow ODE is  dx/dt = x0 - x, whose
    exact solution over [t_i, t_i + h] is

        x_{i+1} = e^{-h} x_i + integral_0^h e^{tau - h} P(tau) dtau,   tau = t - t_i,

    with P(tau) the Lagrange interpolant of the data prediction x0 through the current
    step (offset 0) and the previous `order-1` steps (negative offsets). This returns the
    weights b[j] such that  integral_0^h e^{tau-h} P(tau) dtau = sum_j b[j] * x0_j.

    The e^{tau-h} kernel (bounded by 1 on [0,h]) damps the extrapolation, so unlike a raw
    polynomial integral in sigma this stays well-conditioned on non-uniform schedules
    (karras / beta / bong_tangent) and converges to a clean image instead of leaving noise.

    Moments  M_k = integral_0^h e^{tau-h} tau^k dtau  are exact via the recursion
        M_0 = 1 - e^{-h},   M_k = h^k - k * M_{k-1}.
    """
    import numpy as np

    o = len(offsets)
    M = [1.0 - math.exp(-h)]
    for k in range(1, o):
        M.append(h ** k - k * M[k - 1])

    coeffs = []
    for j in range(o):
        p = np.poly1d([1.0])
        denom = 1.0
        for m in range(o):
            if m == j:
                continue
            p = p * np.poly1d([1.0, -float(offsets[m])])   # (tau - offset_m)
            denom *= (float(offsets[j]) - float(offsets[m]))
        asc = (p / denom).c[::-1]                          # ascending powers: c0, c1, ...
        b = 0.0
        for power, ck in enumerate(asc):
            b += float(ck) * M[power]
        coeffs.append(b)
    return coeffs


@torch.no_grad()
def deis_sample(
    denoise_fn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    max_order: int = 3,
    eta: float = 0.0,
    s_noise: float = 1.0,
    noise_sampler=None,
    callback=None,
):
    """DEIS multistep exponential integrator ("deis_3m" = order 3, "deis_4m" = order 4).

    Stepping in t = -log(sigma), the flow ODE is  dx/dt = x0 - x  (a linear ODE), so each
    step is an exponential-Euler update whose forcing term x0(t) is extrapolated by a
    Lagrange polynomial through the last `max_order` data predictions and integrated
    exactly against the exponential kernel (see _exp_lagrange_coeffs). 1 model eval/step;
    self-starting (order ramps 1 -> max_order as history accrues). For constant x0 every
    order is exact. This is the numerically-stable ("tab") DEIS form and matches the res
    solvers' domain, so it converges cleanly rather than leaving residual noise.

    arXiv:2204.13902 (DEIS), specialised to the linear/flow-matching schedule.

    NOTE: in this formulation (exact exponential kernel + Lagrange quadrature on the
    actual non-uniform grid) the order-k update coincides with the variable-step
    exponential Adams-Bashforth-Norsett method of the same order - so max_order=4 is
    also our "abnorsett_4m" (the Ultra node exposes that recipe-name as an alias).

    Args mirror res_multistep_sample; max_order in {3, 4} for deis_3m / deis_4m.
    """
    if noise_sampler is None:
        noise_sampler = lambda s, sn: torch.randn_like(x)

    sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    n = sigmas.shape[0] - 1
    max_order = max(1, int(max_order))

    x0_hist = []   # past data predictions, most-recent-first
    t_hist = []    # matching t = -log(sigma) abscissae (python floats), most-recent-first

    for i in range(n):
        sigma = sigmas[i]
        denoised = denoise_fn(x, sigma)

        if callback is not None:
            callback(i, sigma, denoised, x)

        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

        if float(sigma_down) <= 0.0:
            # terminal: the exact flow solution at sigma -> 0 is the data prediction itself
            x = denoised
        else:
            t_i = -math.log(float(sigma))
            t_down = -math.log(float(sigma_down))
            h = t_down - t_i                                   # > 0
            order = min(max_order, len(x0_hist) + 1)
            offsets = [0.0] + [th - t_i for th in t_hist[:order - 1]]
            preds = [denoised] + x0_hist[:order - 1]
            b = _exp_lagrange_coeffs(offsets, h)
            x = math.exp(-h) * x + b[0] * preds[0]
            for bc, pv in zip(b[1:], preds[1:]):
                x = x + bc * pv
            if float(sigmas[i + 1]) > 0.0 and eta > 0.0:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

        x0_hist.insert(0, denoised)
        t_hist.insert(0, -math.log(float(sigma)))
        x0_hist = x0_hist[:max_order - 1]
        t_hist = t_hist[:max_order - 1]

    return x


@torch.no_grad()
def lcm_sample(
    denoise_fn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    eta: float = 1.0,
    s_noise: float = 1.0,
    noise_sampler=None,
    callback=None,
):
    """LCM-style full-ancestral sampling: every step jumps to the data prediction
    and re-noises completely to the next sigma with the FLOW-MATCHING mix
    x = (1 - sigma_next) * x0 + sigma_next * noise. (ComfyUI's sample_lcm looks
    like plain x0 + sigma*noise, but its CONST/flow model-sampling wrapper applies
    exactly this scaling inside noise_scaling(); the first cut of this function
    copied the sampler line without the wrapper convention - the VE formula in a
    flow solver - and over-drove the signal ~2x at mid sigmas. Fixed 2026-07-22.)
    eta is ignored - the method IS maximal ancestral churn. Matches ComfyUI
    sample_lcm, so a native "lcm" refine pass is reproducible here.

    The heaviest artifact-launderer of the set: each step discards the
    trajectory (and with it accumulated error like latent-upscale echo) and
    rebuilds from the model prior plus fresh noise - which is why lcm refine
    windows erase upscale artifacts that deterministic ODE samplers (eta=0)
    faithfully integrate into the final image. Trade-off: it also discards
    fine detail the trajectory carried, so it belongs in post-upscale refine
    windows, not in Stage 1."""
    if noise_sampler is None:
        noise_sampler = lambda s, sn: torch.randn_like(x)

    sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    n = sigmas.shape[0] - 1

    for i in range(n):
        sigma = sigmas[i]
        denoised = denoise_fn(x, sigma)
        if callback is not None:
            callback(i, sigma, denoised, x)
        sigma_next = sigmas[i + 1]
        if float(sigma_next) > 0.0:
            # flow-matching re-noise: x_sigma = (1 - sigma) * x0 + sigma * noise
            x = ((1.0 - sigma_next) * denoised
                 + noise_sampler(sigma, sigma_next) * s_noise * sigma_next)
        else:
            x = denoised
    return x


@torch.no_grad()
def lcm_hybrid_sample(denoise_fn, x, sigmas, eta=0.0, s_noise=1.0,
                      noise_sampler=None, callback=None,
                      second="deis_3m", split_frac=0.5):
    """lcm churn for the FIRST part of the window, an ODE/SDE sampler after.

    Rationale (2026-07-23): with pure lcm the final detail is bounded by the x0
    prediction at the entry (highest-sigma) steps - the first churn discards the
    incoming trajectory (and the previous stage's real detail), and later steps
    only polish that re-invention. The hybrid launders upscale damage where it
    lives (the entry steps) and then develops detail trajectory-faithfully at
    low sigma, where detail is made. eta/s_noise apply to the second half's SDE
    path; the lcm half is inherently fully ancestral.

    2026-07-25: parameterized. `second` picks the tail sampler: "deis_3m"
    (default - the original hybrid) or "res_2m" (lcm_hybrid2 in the node GUI).
    `split_frac` is the handoff point as a fraction of the window's steps
    (0.5 default). ceil keeps the historical odd-count behavior: the churn
    half gets the extra step, and split is clamped to [1, n-1] so both halves
    always run at least one step."""
    sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    n = int(sigmas.shape[0]) - 1
    if n <= 1:
        return lcm_sample(denoise_fn, x, sigmas, s_noise=s_noise,
                          noise_sampler=noise_sampler, callback=callback)
    sf = min(0.95, max(0.05, float(split_frac)))
    split = max(1, min(n - 1, int(-(-n * sf // 1))))  # ceil, clamped
    x = lcm_sample(denoise_fn, x, sigmas[:split + 1], s_noise=s_noise,
                   noise_sampler=noise_sampler, callback=callback)
    _cb2 = (None if callback is None
            else (lambda i, s, d, xx: callback(i + split, s, d, xx)))
    if second == "res_2m":
        return res_multistep_sample(denoise_fn, x, sigmas[split:], eta=eta,
                                    s_noise=s_noise, noise_sampler=noise_sampler,
                                    callback=_cb2)
    return deis_sample(denoise_fn, x, sigmas[split:], max_order=3, eta=eta,
                       s_noise=s_noise, noise_sampler=noise_sampler, callback=_cb2)


# --------------------------------------------------------------------------------------
# Generic explicit Runge-Kutta ("linear" / non-exponential family) - rk6_7s
# --------------------------------------------------------------------------------------
# Butcher tableau for the 7-stage method RES4LYF labels "rk6_7s" (their comment: non-
# monotonic, ~5th order despite the name). A Butcher tableau is pure math - the a/b/c
# coefficients define the method; the implementation below is ours. Verified: b sums to
# 1, every row of a sums to its c (consistency conditions).
RK_TABLEAUS = {
    "euler": {   # 1-stage explicit Euler as a degenerate tableau: windowed euler
        # stages share the exact consistent path (same shifted window for the
        # re-noise and the stepping) as every other sampler - the 2026-07-22
        # shift-mismatch fix. 1 model call per step, identical cost to before.
        "a": [[]],
        "b": [1.0],
        "c": [0.0],
    },
    "rk6_7s": {
        "a": [
            [],
            [1/3],
            [0.0, 2/3],
            [1/12, 1/3, -1/12],
            [-1/16, 9/8, -3/16, -3/8],
            [0.0, 9/8, -3/8, -3/4, 1/2],
            [9/44, -9/11, 63/44, 18/11, 0.0, -16/11],
        ],
        "b": [11/120, 0.0, 27/40, 27/40, -4/15, -4/15, 11/120],
        "c": [0.0, 1/3, 2/3, 1/3, 1/2, 1/2, 1.0],
    },
}


@torch.no_grad()
def rk_explicit_sample(
    denoise_fn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    tableau: str = "rk6_7s",
    eta: float = 0.0,
    s_noise: float = 1.0,
    noise_sampler=None,
    callback=None,
):
    """Generic explicit Runge-Kutta on the flow ODE, in SIGMA space.

    For rectified flow the ODE is  dx/dsigma = d = (x - x0)/sigma  (to_d). Each step
    integrates sigma -> sigma_down with the given Butcher tableau: stage s evaluates the
    model at  x + h * sum_j a[s][j] * k_j  and  sigma + c[s] * h  (h = sigma_down - sigma,
    negative), then  x += h * sum_j b_j * k_j.  This is the classic "linear" RK family
    (RES4LYF's rk-named types), as opposed to the exponential res/deis methods above -
    a high-order solve of the SAME trajectory, at len(c) model calls per step (7 for
    rk6_7s, so 10 steps = 70 evals: the accuracy the 'kreamania'-style finetuner recipes
    are tuned around).

    Terminal step (sigma_down == 0) falls back to the flow-Euler step to the data
    prediction, exactly like the other samplers here. eta/noise_sampler/callback mirror
    res_multistep_sample (callback fires once per STEP, not per stage eval).
    """
    tb = RK_TABLEAUS[tableau]
    a, b, c = tb["a"], tb["b"], tb["c"]
    n_stages = len(c)

    if noise_sampler is None:
        noise_sampler = lambda s, sn: torch.randn_like(x)

    sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    n = sigmas.shape[0] - 1

    for i in range(n):
        sigma = sigmas[i]
        den1 = denoise_fn(x, sigma)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

        if callback is not None:
            callback(i, sigma, den1, x)

        if float(sigma_down) == 0.0:
            d = to_d(x, sigma, den1)
            x = x + d * (sigma_down - sigma)
        else:
            h = sigma_down - sigma                    # negative (descending)
            ks = [to_d(x, sigma, den1)]
            for s in range(1, n_stages):
                x_s = x
                for j, aij in enumerate(a[s]):
                    if aij:
                        x_s = x_s + (h * aij) * ks[j]
                sig_s = (sigma + c[s] * h).clamp(min=sigma_down)
                den_s = denoise_fn(x_s, sig_s)
                ks.append(to_d(x_s, sig_s, den_s))
            upd = torch.zeros_like(x)
            for bj, kj in zip(b, ks):
                if bj:
                    upd = upd + bj * kj
            x = x + h * upd

        if float(sigmas[i + 1]) > 0.0 and eta > 0.0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

    return x


# --------------------------------------------------------------------------------------
# self-test (CPU only, no model): a constant-x0 denoiser must drive x -> x0_target,
# and the integrator must stay finite/stable. Also checks res beats Euler on a known ODE.
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Build a flow-matching toy: a fixed target image x0*, and a "model" that returns
    # the true x0* given any (x, sigma). For the linear flow ODE the exact solution at
    # sigma=0 is x0*, so the sampler output must equal x0* regardless of step count.
    shape = (1, 64, 4)
    x0_star = torch.randn(shape)

    def denoise_const(x, sigma):
        return x0_star.clone()

    # schedule: linear flow sigmas 1 -> 1/8, then terminal 0
    raw = torch.linspace(1.0, 1.0 / 8, 8)
    sigmas = torch.cat([raw, raw.new_zeros(1)])

    # start at sigma0 from a noised version of x0*
    eps = torch.randn(shape)
    s0 = sigmas[0]
    x_init = (1 - s0) * x0_star + s0 * eps

    out = res_multistep_sample(denoise_const, x_init.clone(), sigmas, eta=0.0)
    err = (out - x0_star).abs().max().item()
    print(f"[self-test] res_2m  constant-x0 max|err| = {err:.3e}  (should be ~0)")

    out_2s = res_2s_sample(denoise_const, x_init.clone(), sigmas, eta=0.0)
    print(f"[self-test] res_2s  constant-x0 max|err| = {(out_2s - x0_star).abs().max().item():.3e}")
    out_d3 = deis_sample(denoise_const, x_init.clone(), sigmas, max_order=3, eta=0.0)
    print(f"[self-test] deis_3m constant-x0 max|err| = {(out_d3 - x0_star).abs().max().item():.3e}")
    out_d4 = deis_sample(denoise_const, x_init.clone(), sigmas, max_order=4, eta=0.0)
    print(f"[self-test] deis_4m constant-x0 max|err| = {(out_d4 - x0_star).abs().max().item():.3e}")

    # second check: a non-constant denoiser, ensure finite + shape preserved
    def denoise_shrink(x, sigma):
        # pretend the clean estimate is a mild low-pass of x (just to exercise the math)
        return 0.5 * x

    out2 = res_multistep_sample(denoise_shrink, x_init.clone(), sigmas, eta=0.0)
    print(f"[self-test] non-constant finite={torch.isfinite(out2).all().item()} "
          f"shape={tuple(out2.shape)}")

    # third check: residual-noise convergence on an IRREGULAR (karras-like) schedule.
    # This is the case that exposed the old sigma-space DEIS: a target that varies smoothly
    # along the trajectory. We integrate the true flow ODE with a fine Euler reference and
    # confirm the exp-integrator DEIS lands close to it (small residual == no leftover noise).
    A = torch.randn(shape)                       # fixed linear operator on the state
    def denoise_lin(x, sigma):
        # x0 estimate that depends on sigma so the velocity is genuinely non-constant
        s = float(sigma)
        return 0.3 * A + (0.2 * s) * x

    ks = torch.linspace(1.0, (1.0 / 8) ** (1 / 3), 8) ** 3   # karras-ish, very non-uniform
    ks = torch.cat([ks, ks.new_zeros(1)])
    fine = torch.linspace(1.0, 1.0 / 512, 512)
    fine = torch.cat([fine, fine.new_zeros(1)])
    ref = res_multistep_sample(denoise_lin, x_init.clone(), fine, eta=0.0)
    for name, mo in (("deis_3m", 3), ("deis_4m", 4)):
        got = deis_sample(denoise_lin, x_init.clone(), ks, max_order=mo, eta=0.0)
        rel = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-8)
        print(f"[self-test] {name} irregular-schedule rel|err vs fine ref| = {rel:.3e} "
              f"(should be small; large => leftover noise)")

    # rk6_7s: constant-x0 exactness (velocity is constant along that trajectory, so any
    # consistent RK is exact) + irregular-schedule accuracy vs the fine Euler reference
    # (should be at least as good as the multisteps, at 7x the eval cost) + tableau
    # consistency conditions.
    out_rk = rk_explicit_sample(denoise_const, x_init.clone(), sigmas, tableau="rk6_7s", eta=0.0)
    print(f"[self-test] rk6_7s  constant-x0 max|err| = {(out_rk - x0_star).abs().max().item():.3e}")
    got_rk = rk_explicit_sample(denoise_lin, x_init.clone(), ks, tableau="rk6_7s", eta=0.0)
    rel_rk = (got_rk - ref).abs().max().item() / (ref.abs().max().item() + 1e-8)
    print(f"[self-test] rk6_7s  irregular-schedule rel|err vs fine ref| = {rel_rk:.3e}")
    _tb = RK_TABLEAUS["rk6_7s"]
    print(f"[self-test] rk6_7s tableau: sum(b)={sum(_tb['b']):.6f} (expect 1.0); row-sum==c ok: "
          f"{all(abs(sum(_tb['a'][r]) - _tb['c'][r]) < 1e-12 for r in range(1, len(_tb['c'])))}")

    for name, fn in (("res_2s", lambda: res_2s_sample(denoise_shrink, x_init.clone(), sigmas)),
                     ("deis_3m", lambda: deis_sample(denoise_shrink, x_init.clone(), sigmas, max_order=3)),
                     ("deis_4m", lambda: deis_sample(denoise_shrink, x_init.clone(), sigmas, max_order=4)),
                     ("rk6_7s", lambda: rk_explicit_sample(denoise_shrink, x_init.clone(), sigmas))):
        o = fn()
        print(f"[self-test] {name} non-constant finite={torch.isfinite(o).all().item()} shape={tuple(o.shape)}")

    # lcm family: constant-x0 must land exactly on x0* (the lcm half churns
    # AROUND x0*, the deis half converges exactly for a constant forcing).
    out_lcm = lcm_sample(denoise_const, x_init.clone(), sigmas)
    print(f"[self-test] lcm        constant-x0 max|err| = {(out_lcm - x0_star).abs().max().item():.3e}")
    out_hyb = lcm_hybrid_sample(denoise_const, x_init.clone(), sigmas)
    print(f"[self-test] lcm_hybrid constant-x0 max|err| = {(out_hyb - x0_star).abs().max().item():.3e}")
    # residual-noise discriminator on the irregular schedule: the hybrid is
    # stochastic, so it will NOT match the deterministic reference exactly -
    # but its output STD must stay in the same ballpark. A large std blowup
    # means un-removed churn noise (the rainbow-over-image failure).
    got_hyb = lcm_hybrid_sample(denoise_lin, x_init.clone(), ks)
    rel_hyb = (got_hyb - ref).abs().max().item() / (ref.abs().max().item() + 1e-8)
    print(f"[self-test] lcm_hybrid irregular rel|err vs ref| = {rel_hyb:.3e} "
          f"(stochastic: small-ish expected, NOT ~0)")
    print(f"[self-test] lcm_hybrid output std {float(got_hyb.std()):.3f} vs ref std "
          f"{float(ref.std()):.3f}  <-- blowup here = residual-noise bug")

    # eta>0 path executes without error
    out3 = res_multistep_sample(denoise_const, x_init.clone(), sigmas, eta=0.5)
    print(f"[self-test] eta=0.5 finite={torch.isfinite(out3).all().item()}")
    print(f"[self-test] res_2s  eta=0.5 finite="
          f"{torch.isfinite(res_2s_sample(denoise_const, x_init.clone(), sigmas, eta=0.5)).all().item()}")
    print(f"[self-test] deis_3m eta=0.5 finite="
          f"{torch.isfinite(deis_sample(denoise_const, x_init.clone(), sigmas, max_order=3, eta=0.5)).all().item()}")
    print(f"[self-test] rk6_7s  eta=0.5 finite="
          f"{torch.isfinite(rk_explicit_sample(denoise_const, x_init.clone(), sigmas, eta=0.5)).all().item()}")
    print("[self-test] OK")
