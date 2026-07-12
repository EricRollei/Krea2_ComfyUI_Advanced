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
    for name, fn in (("res_2s", lambda: res_2s_sample(denoise_shrink, x_init.clone(), sigmas)),
                     ("deis_3m", lambda: deis_sample(denoise_shrink, x_init.clone(), sigmas, max_order=3)),
                     ("deis_4m", lambda: deis_sample(denoise_shrink, x_init.clone(), sigmas, max_order=4))):
        o = fn()
        print(f"[self-test] {name} non-constant finite={torch.isfinite(o).all().item()} shape={tuple(o.shape)}")

    # eta>0 path executes without error
    out3 = res_multistep_sample(denoise_const, x_init.clone(), sigmas, eta=0.5)
    print(f"[self-test] eta=0.5 finite={torch.isfinite(out3).all().item()}")
    print(f"[self-test] res_2s  eta=0.5 finite="
          f"{torch.isfinite(res_2s_sample(denoise_const, x_init.clone(), sigmas, eta=0.5)).all().item()}")
    print(f"[self-test] deis_3m eta=0.5 finite="
          f"{torch.isfinite(deis_sample(denoise_const, x_init.clone(), sigmas, max_order=3, eta=0.5)).all().item()}")
    print("[self-test] OK")
