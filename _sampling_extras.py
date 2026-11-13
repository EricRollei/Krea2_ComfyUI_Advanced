# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Eric_Krea2 / _sampling_extras.py
#
# The sampling-extras bundle (2026-10-23): hybrid handoff by sigma, per-stage SDE band,
# churn form. Authored on the "Eric Krea2 Hybrid Sigma Handoff" node and fed into the
# Ultra / Ultra V2 optional `sampling` input, so the Ultra panel stays small.
#
# Precedence for every key, resolved in one place (resolve()):
#     explicit value (a sweep cell override)  >  bundle (the node)  >  ENGINE_DEFAULTS
# ENGINE_DEFAULTS reproduce the behaviour from before these features existed - EXCEPT churn_form,
# which defaults to "rf" since 2026-10-25 (Eric's call after the 8-sampler screen: ve burnt every
# sampler at eta 0.8, 44-69% clip; rf held 1-6%). At eta 0 nothing changes; with eta > 0 an unwired
# Ultra now renders differently from older images - set churn_form ve on the node to reproduce one.

from __future__ import annotations

KEYS = ("s1_hybrid_handoff", "s2_hybrid_handoff", "s3_hybrid_handoff",
        "s1_sde_hi", "s1_sde_lo", "s2_sde_hi", "s2_sde_lo", "s3_sde_hi", "s3_sde_lo",
        "churn_form")

ENGINE_DEFAULTS = {
    "s1_hybrid_handoff": 0.0, "s2_hybrid_handoff": 0.0, "s3_hybrid_handoff": 0.0,   # 0 = use hybrid_steps
    "s1_sde_hi": 1.0, "s1_sde_lo": 0.0, "s2_sde_hi": 1.0, "s2_sde_lo": 0.0,
    "s3_sde_hi": 1.0, "s3_sde_lo": 0.0,                                            # whole window
    "churn_form": "rf",                                       # rectified-flow split (was "ve" before 2026-10-25)
}

BUNDLE_TYPE = "KREA2_SAMPLING"


def resolve(explicit: dict | None, bundle: dict | None) -> dict:
    """Effective value of every key: explicit (not None) > bundle > engine default."""
    explicit = explicit or {}
    bundle = bundle if isinstance(bundle, dict) else {}
    out = {}
    for k in KEYS:
        v = explicit.get(k)
        if v is None:
            v = bundle.get(k)
        if v is None:
            v = ENGINE_DEFAULTS[k]
        out[k] = v
    out["churn_form"] = "rf" if str(out["churn_form"]).lower() == "rf" else "ve"
    for k in KEYS[:-1]:
        out[k] = float(out[k])
    return out


def bands(resolved: dict) -> dict:
    """Per-stage SDE band for the solver: None = whole window (legacy, bit-identical path)."""
    b = {}
    for st in ("s1", "s2", "s3"):
        hi, lo = resolved[f"{st}_sde_hi"], resolved[f"{st}_sde_lo"]
        if hi >= 1.0 and lo <= 0.0:
            b[st] = None
        else:
            if lo > hi:
                print(f"[EricKrea2-MS] WARNING: {st}_sde_lo {lo:g} > {st}_sde_hi {hi:g} - "
                      f"empty SDE band, {st} runs without churn")
            b[st] = (lo, hi)
    return b


def handoffs(resolved: dict) -> dict:
    return {st: resolved[f"{st}_hybrid_handoff"] for st in ("s1", "s2", "s3")}


def describe(resolved: dict) -> str:
    """One console line listing only what differs from the engine defaults."""
    diff = [f"{k}={resolved[k]:g}" if isinstance(resolved[k], float) else f"{k}={resolved[k]}"
            for k in KEYS if resolved[k] != ENGINE_DEFAULTS[k]]
    return ", ".join(diff) if diff else "all at defaults"
