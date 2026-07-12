# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Shared flow-matching sigma-schedule math
========================================
One place for every Krea2 sigma *curve*. Everything here returns a RAW
(pre-shift) descending sigma list in ``[sigma_min, sigma_start]`` where
``sigma_min = 1/num_steps`` and ``sigma_start = 1.0`` for a full-denoise run -
exactly what ``pipe.scheduler.set_timesteps(sigmas=...)`` expects (it reapplies
the resolution-aware dynamic shift and appends the terminal 0). The *schedule*
only sets step spacing; the pipeline owns the shift, so a curve authored once is
valid at any resolution.

Curves:
  linear        - uniform spacing (composition).
  balanced      - Karras rho=3 (moderate low-sigma detail).
  karras        - Karras rho=7 (heavy low-sigma / fine texture).
  beta57        - Beta(0.5, 0.7) CDF spacing (RES4LYF default).
  beta          - Beta(alpha, beta) CDF spacing (configurable).
  bong_tangent  - RES4LYF two-stage arctan descent (hold-then-drop).
  exponential   - geometric spacing.

`detail_bias` (-1..+1) is the single friendly knob: it packs more sampling
steps at low sigma (bias>0 -> fine detail) or high sigma (bias<0 -> composition).
It applies to EVERY curve: karras/balanced fold it into the rho, bong_tangent
into its drop slope, and the remaining curves (linear/beta/beta57/exponential)
get an equivalent monotonic step redistribution. Explicit `rho` overrides it on
the Karras curves.

Ported from RES4LYF (ClownsharkBatwing/RES4LYF) sigmas.py for beta57/bong_tangent.
"""

import math

import numpy as np

CURVES = ["linear", "balanced", "karras", "beta57", "beta", "bong_tangent", "exponential"]


def _rho_from_bias(base_rho: float, bias: float, rho) -> float:
    if rho is not None and float(rho) > 0:
        return float(rho)
    b = max(-1.0, min(1.0, float(bias)))
    # bias +1 -> 4x base rho (more low-sigma detail); -1 -> 0.25x (more composition).
    return max(1.0, base_rho * (2.0 ** (2.0 * b)))


def _karras(t, sigma_start, sigma_min, rho):
    inv = 1.0 / rho
    return (sigma_start ** inv + t * (sigma_min ** inv - sigma_start ** inv)) ** rho


def _tan_sigmas(steps, slope, pivot, start, end):
    """RES4LYF arctan descent from `start` to `end` over `steps` points."""
    steps = max(1, int(steps))
    two_over_pi = 2.0 / math.pi
    smax = (two_over_pi * math.atan(-slope * (0 - pivot)) + 1) / 2
    smin = (two_over_pi * math.atan(-slope * ((steps - 1) - pivot)) + 1) / 2
    srange = (smax - smin) or 1e-12
    sscale = start - end
    return [((two_over_pi * math.atan(-slope * (x - pivot)) + 1) / 2 - smin) / srange * sscale + end
            for x in range(steps)]


def _bong_tangent(keep, sigma_start, sigma_min, pivot_1=0.6, pivot_2=0.6,
                  slope_1=0.2, slope_2=0.2, middle=0.5):
    """RES4LYF two-stage arctan schedule, scaled into [sigma_min, sigma_start]
    and resampled to exactly `keep` points."""
    steps = keep + 2
    mid_val = sigma_min + (sigma_start - sigma_min) * float(middle)
    p1 = max(1, int(steps * pivot_1))
    p2 = max(1, int(steps * pivot_2))
    midpoint = int((steps * pivot_1 + steps * pivot_2) / 2)
    s1 = slope_1 / (steps / 40.0)
    s2 = slope_2 / (steps / 40.0)
    stage_2_len = max(1, steps - midpoint)
    stage_1_len = max(1, steps - stage_2_len)
    t1 = _tan_sigmas(stage_1_len, s1, p1, sigma_start, mid_val)[:-1]
    t2 = _tan_sigmas(stage_2_len, s2, p2 - stage_1_len, mid_val, sigma_min)
    sig = np.asarray(t1 + t2, dtype=np.float64)
    if sig.size == 0:
        sig = np.linspace(sigma_start, sigma_min, keep)
    elif sig.size != keep:
        sig = np.interp(np.linspace(0.0, 1.0, keep),
                        np.linspace(0.0, 1.0, sig.size), sig)
    return sig


def _apply_detail_warp(sig, bias):
    """Redistribute the sample points along an already-built descending sigma
    curve WITHOUT moving its endpoints. ``bias`` > 0 packs more steps at LOW sigma
    (fine detail); ``bias`` < 0 packs them at HIGH sigma (composition). This is the
    universal ``detail_bias`` for curves that have no native rho/slope knob
    (linear / beta / beta57 / exponential) - karras/balanced fold bias into rho and
    bong_tangent into its slope, so those are left untouched by the caller.

    Mechanism: reinterpret ``sig`` as sigma(t) on a uniform t in [0,1], then resample
    at warped positions ``u = t**k`` with ``k = 2**bias`` (k>1 -> sample density
    concentrates near t=1 = low sigma). Monotonic in, monotonic out; u[0]=0 and
    u[-1]=1 keep sigma_start / sigma_min pinned.
    """
    b = max(-1.0, min(1.0, float(bias)))
    if abs(b) < 1e-6:
        return sig
    a = np.asarray(sig, dtype=np.float64)
    n = a.size
    if n < 3:
        return sig
    t = np.linspace(0.0, 1.0, n)
    u = t ** (2.0 ** b)
    return np.interp(u, t, a).tolist()


def build_sigmas(num_steps, denoise=1.0, curve="linear", *, bias=0.0, rho=None,
                 alpha=0.5, beta=0.7, pivot=0.6, slope=0.2, middle=0.5):
    """Raw descending sigma list for one stage. `denoise` < 1 trims the high-sigma
    head (re-noise level); the pipeline reapplies its resolution shift downstream."""
    num_steps = max(1, int(num_steps))
    sigma_max, sigma_min = 1.0, 1.0 / num_steps
    if denoise >= 1.0:
        keep, sigma_start = num_steps, sigma_max
    else:
        keep = max(1, int(round(num_steps * float(denoise))))
        full_linear = np.linspace(sigma_max, sigma_min, num_steps)
        sigma_start = float(full_linear[num_steps - keep])

    t = np.linspace(0.0, 1.0, keep)
    curve = (curve or "linear").lower()

    if curve in ("karras", "balanced"):
        base = 3.0 if curve == "balanced" else 7.0
        sig = _karras(t, sigma_start, sigma_min, _rho_from_bias(base, bias, rho))
    elif curve in ("beta", "beta57"):
        a, b = (0.5, 0.7) if curve == "beta57" else (float(alpha), float(beta))
        try:
            from scipy.stats import beta as _beta_dist
            bvals = _beta_dist.ppf(1.0 - np.linspace(0.0, 1.0, keep), a, b)
            bvals = np.clip(np.nan_to_num(bvals, nan=0.0), 0.0, 1.0)
            sig = sigma_min + bvals * (sigma_start - sigma_min)
        except Exception as _e:
            print(f"[EricKrea2-sigmas] {curve} needs scipy ({_e}); using linear spacing.")
            sig = np.linspace(sigma_start, sigma_min, keep)
    elif curve == "bong_tangent":
        # detail_bias nudges the drop slope (bias>0 -> steeper late drop = more detail hold early).
        s = float(slope) * (2.0 ** (max(-1.0, min(1.0, float(bias)))))
        sig = _bong_tangent(keep, sigma_start, sigma_min, pivot, pivot, s, s, middle)
    elif curve == "exponential":
        sig = sigma_start * (sigma_min / sigma_start) ** t
    else:  # linear
        sig = np.linspace(sigma_start, sigma_min, keep)

    # Universal detail knob: curves without a native bias parameter get detail_bias
    # applied as a monotonic step redistribution (endpoints pinned). karras/balanced
    # already fold bias into rho; bong_tangent into its slope - don't double-apply.
    if curve not in ("karras", "balanced", "bong_tangent"):
        sig = _apply_detail_warp(sig, bias)

    sig = np.clip(np.asarray(sig, dtype=np.float64), sigma_min, sigma_start)
    # enforce strictly-descending (guards tangent/beta overshoot from breaking the solver)
    sig = np.maximum.accumulate(sig[::-1])[::-1]
    return sig.tolist()


# ── per-stage override resolution (KREA2_SIGMAS bundle) ──────────────────────

_STAGE_PARAM_KEYS = ("bias", "rho", "alpha", "beta", "pivot", "slope", "middle")


def resolve_stage(bundle, stage_key, fallback_curve):
    """Return ``(curve, params)`` for a stage. If `bundle` overrides this stage
    (present + enabled), use its curve/params; otherwise the widget `fallback_curve`
    with empty params (so the default build path is unchanged)."""
    if isinstance(bundle, dict):
        st = bundle.get(stage_key)
        if isinstance(st, dict) and st.get("enabled", True):
            curve = st.get("curve", fallback_curve) or fallback_curve
            params = {k: st[k] for k in _STAGE_PARAM_KEYS if k in st}
            return curve, params
    return fallback_curve, {}


def build_from_spec(num_steps, denoise, curve, params):
    """Build a raw sigma list from a resolved ``(curve, params)`` pair."""
    p = params or {}
    return build_sigmas(
        num_steps, denoise, curve,
        bias=float(p.get("bias", 0.0)),
        rho=p.get("rho"),
        alpha=float(p.get("alpha", 0.5)),
        beta=float(p.get("beta", 0.7)),
        pivot=float(p.get("pivot", 0.6)),
        slope=float(p.get("slope", 0.2)),
        middle=float(p.get("middle", 0.5)),
    )
