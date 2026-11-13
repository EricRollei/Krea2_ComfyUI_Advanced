# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Eric_Krea2 / _traj_probe.py
#
# Trajectory probe + SDE-band telemetry (2026-10-22,
# docs/spec_sde_band_and_trajectory_probe_2026-10-22.md, part B).
#
# A DIAGNOSTIC ONLY - it never changes the image. At every model evaluation the
# samplers already report (step i, sigma_i, D_i) through their callback, where
# D = the model's clean-image (x0) prediction. From a rolling 3-deep history we
# compute, in t = -log(sigma) (the variable all our exponential solvers step in):
#
#   drift_i : rms(D_i - D_{i-1}) / (t_i - t_{i-1}) / rms(D_i)
#             how fast the model's idea of the final image is changing per unit
#             log-sigma. Peaks = sigma bands where the model is "deciding".
#
#   lte_i   : Milne-device local-error estimate for the step sigma_{i-1}->sigma_i:
#             rms(C - P) / rms(D_i), with P = order-2 exponential Adams-Bashforth
#             (nodes: step start + previous step) and C = order-2 exponential
#             Adams-Moulton (nodes: step start + step end), both integrated
#             against the same exact e^{tau-h} kernel the deis/abnorsett solvers
#             use (_exp_lagrange_coeffs). Zero when D is constant or linear in t;
#             scales ~h^3 otherwise. It is PROPORTIONAL to the local error of a
#             2nd-order multistep step (~5/6 x for AB2) - read shapes and rankings,
#             not absolute values.
#
# Both are computed from the logged predictions only, so they measure the
# trajectory the model actually took, whatever sampler drove it. Steps whose
# inputs were touched by injected churn noise (eta / lcm / er_sde) are flagged
# NOT clean and excluded from the summaries (read the probe at eta 0, or
# outside the SDE band). Every public function is fail-silent: a probe problem
# must never break a generation.

from __future__ import annotations

import math

try:
    from ._res_solver import _exp_lagrange_coeffs, hybrid_split
except ImportError:                      # direct script run (self-test)
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from _res_solver import _exp_lagrange_coeffs, hybrid_split

# Samplers whose eta churn the SDE band gates (euler: eta clamped 0; lcm: full
# churn by design; er_sde: intrinsic churn - none of those read the band).
BAND_METHODS = ("res_2m", "res_2s", "deis_3m", "abnorsett_3m", "abnorsett_4m",
                "lawson4_4s", "etdrk4_4s", "rk6_7s", "lcm_hybrid", "lcm_hybrid2")

LATE_SIGMA = 0.3                         # "late" = steps below this sigma

# Last generate()'s per-stage summaries; reset at the start of every run.
LAST: dict = {}


def reset():
    LAST.clear()


def _in_band(s, band):
    if band is None:
        return True
    return float(band[0]) <= float(s) <= float(band[1])


def churn_mask(sigmas, eta, band, method, hybrid_lcm_steps=0, handoff=0.0):
    """mask[i] = True when noise is injected after step i (so D_{i+1} is not a
    pure-trajectory prediction). Mirrors the dispatch in _res_denoise_packed."""
    try:
        sig = [float(s) for s in sigmas]
        n = len(sig) - 1
        method = str(method or "res_2m")
        e = float(eta or 0.0)
        if method in ("lcm", "er_sde"):
            return [sig[i + 1] > 0.0 for i in range(n)]
        if method == "euler":
            return [False] * n
        split = 0
        if method in ("lcm_hybrid", "lcm_hybrid2") and n > 1:
            split = hybrid_split(sig, hybrid_lcm_steps, handoff)   # same rule as the solver
        out = []
        for i in range(n):
            if i < split:
                out.append(sig[i + 1] > 0.0)
            else:
                out.append(e > 0.0 and sig[i + 1] > 0.0 and _in_band(sig[i + 1], band))
        return out
    except Exception:
        return []


def print_band(stage, sigmas, eta, band, form, method, mask):
    """Always-on console line for eta-churn stages: how many steps actually churned."""
    try:
        if not (eta and float(eta) > 0.0) or str(method) not in BAND_METHODS:
            return
        sig = [float(s) for s in sigmas]
        n = len(sig) - 1

        idx = [i for i in range(n) if mask and i < len(mask) and mask[i]]
        tail = f" (lcm head included)" if str(method).startswith("lcm_hybrid") else ""
        if band is None:
            where = "whole window"
        else:
            where = f"band \u03c3[{float(band[1]):.2f}\u2192{float(band[0]):.2f}]"
        span = (f", \u03c3_next {sig[idx[0] + 1]:.2f}\u2026{sig[idx[-1] + 1]:.2f}"
                if idx else "")
        print(f"[EricKrea2-MS]    {str(stage or 's?').upper()} SDE churn ({form}, eta {float(eta):g}): "
              f"{where}: {len(idx)}/{n} steps churned{span}{tail}")
    except Exception:
        pass


def print_handoff(stage, sigmas, method, hybrid_lcm_steps=0, handoff=0.0):
    """Console line for lcm-hybrid stages: the step split and the sigma it lands on."""
    try:
        if not str(method).startswith("lcm_hybrid"):
            return
        sig = [float(s) for s in sigmas]
        n = len(sig) - 1
        if n <= 1:
            return
        k = hybrid_split(sig, hybrid_lcm_steps, handoff)
        h = float(handoff or 0.0)
        if h > 0.0:
            how = f"handoff \u03c3{h:.2f} requested"
            if int(hybrid_lcm_steps or 0) > 0:
                how += f"; hybrid_steps {int(hybrid_lcm_steps)} ignored"
        else:
            how = f"hybrid_steps {int(hybrid_lcm_steps or 0) or 'auto'}"
        print(f"[EricKrea2-MS]    {str(stage or 's?').upper()} hybrid handoff: {k}/{n} lcm steps "
              f"-> tail from \u03c3{sig[k]:.2f} ({how})")
    except Exception:
        pass


def _rms(t):
    return float(t.pow(2).mean().sqrt())


class Probe:
    def __init__(self, stage, mask):
        self.stage = str(stage or "s?")
        self.mask = list(mask or [])
        self.hist = []          # [(i, t, D)] last two evaluations
        self.rows = []

    def _m(self, i):
        return bool(self.mask[i]) if 0 <= i < len(self.mask) else False

    def observe(self, i, sigma, denoised):
        try:
            s = float(sigma)
            if not s > 0.0:
                return
            t = -math.log(s)
            D = denoised.detach().float()
            row = {"i": int(i), "sigma": s, "drift": None, "drift_clean": None,
                   "lte": None, "lte_sigma": None, "clean": None}
            if self.hist:
                scale = _rms(D) + 1e-8
                i1, t1, D1 = self.hist[-1]
                dt = t - t1
                if dt > 0.0:
                    row["drift"] = _rms(D - D1) / dt / scale
                    # first interval starts from pure noise (sigma ~1): its "drift" is the jump out of
                    # the noise mean, not a decision - kept in the curve, excluded from the peak (2026-10-23)
                    row["drift_clean"] = (not self._m(i1)) and math.exp(-t1) < 0.97
                if len(self.hist) >= 2:
                    i2, t2, D2 = self.hist[-2]
                    h = t - t1
                    if h > 0.0 and t1 > t2:
                        bP = _exp_lagrange_coeffs([0.0, t2 - t1], h)
                        bC = _exp_lagrange_coeffs([0.0, h], h)
                        diff = (bC[0] - bP[0]) * D1 + bC[1] * D - bP[1] * D2
                        row["lte"] = _rms(diff) / scale
                        row["lte_sigma"] = math.sqrt(s * math.exp(-t1))   # step mid (geometric)
                        row["clean"] = not (self._m(i2) or self._m(i1))
            self.hist.append((int(i), t, D))
            self.hist = self.hist[-2:]
            self.rows.append(row)
        except Exception:
            pass

    def summary(self):
        rows = self.rows
        lte_rows = [r for r in rows if r["lte"] is not None]
        clean = [r for r in lte_rows if r["clean"]]
        out = {"steps": len(rows), "lte_steps": len(lte_rows), "clean": len(clean)}
        if clean:
            tot = sum(r["lte"] for r in clean)
            pk = max(clean, key=lambda r: r["lte"])
            late = sum(r["lte"] for r in clean if r["lte_sigma"] < LATE_SIGMA)
            out.update({"lte_sum": tot, "lte_peak": pk["lte"], "lte_peak_sigma": pk["lte_sigma"],
                        "late_pct": (100.0 * late / tot) if tot > 0 else 0.0})
        dr = [r for r in rows if r["drift"] is not None and r["drift_clean"]]
        if dr:
            pk = max(dr, key=lambda r: r["drift"])
            out.update({"drift_peak": pk["drift"], "drift_peak_sigma": pk["sigma"]})
        out["curves"] = {"sigma": [r["sigma"] for r in rows],
                         "drift": [r["drift"] for r in rows],
                         "lte": [r["lte"] for r in rows],
                         "lte_sigma": [r["lte_sigma"] for r in rows],
                         "clean": [r["clean"] for r in rows],
                         "churn": self.mask}
        return out


def start(stage, mask):
    try:
        return Probe(stage, mask)
    except Exception:
        return None


def finish(probe):
    try:
        if probe is None or not probe.rows:
            return
        s = probe.summary()
        LAST[probe.stage] = s
        tag = probe.stage.upper()
        if "lte_sum" in s:
            line = (f"lte \u03a3{s['lte_sum']:.4f} peak {s['lte_peak']:.4f} "
                    f"@\u03c3{s['lte_peak_sigma']:.2f} | late(<{LATE_SIGMA:g}) {s['late_pct']:.0f}%")
        else:
            line = "lte n/a (no churn-free 3-step run - read the probe at eta 0)"
        if "drift_peak" in s:
            line += f" | drift peak {s['drift_peak']:.3f} @\u03c3{s['drift_peak_sigma']:.2f}"
        print(f"[EricKrea2-Probe] {tag}: {line} | clean {s['clean']}/{s['lte_steps']} steps")
    except Exception:
        pass


def sweep_fields():
    """Flat probe_* scalars for the sweep CSV + per-stage curves for the manifest."""
    out = {}
    try:
        curves = {}
        for st in sorted(LAST):
            s = LAST[st]
            for k, nd in (("lte_sum", 5), ("lte_peak", 5), ("lte_peak_sigma", 3),
                          ("late_pct", 1), ("drift_peak", 4), ("drift_peak_sigma", 3)):
                if k in s and s[k] is not None:
                    out[f"probe_{st}_{k}"] = round(float(s[k]), nd)
            out[f"probe_{st}_clean"] = int(s.get("clean", 0))
            curves[st] = s.get("curves")
        if curves:
            out["trajectory"] = curves
    except Exception:
        pass
    return out


# ------------------------------------------------------------------------------
# self-test (CPU, no model): synthetic x0 trajectories with known shape.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    A = torch.randn(1, 64, 8)
    B = torch.randn(1, 64, 8)
    C = torch.randn(1, 64, 8)

    def run(fn, sig):
        p = Probe("s1", [False] * (len(sig) - 1))
        for i, s in enumerate(sig[:-1]):
            p.observe(i, torch.tensor(s), fn(-math.log(s)))
        return p

    sig = [1.0 - 1e-4] + [0.9 * (0.82 ** k) for k in range(12)] + [0.0]
    const = run(lambda t: A, sig)
    lin = run(lambda t: A + B * t, sig)
    quad = run(lambda t: A + C * t * t, sig)
    mc = max(r["lte"] or 0.0 for r in const.rows)
    md = max(r["drift"] or 0.0 for r in const.rows)
    ml = max(r["lte"] or 0.0 for r in lin.rows)
    print(f"[probe self-test] const: max lte {mc:.2e}, max drift {md:.2e}  (expect ~0)")
    print(f"[probe self-test] linear-in-t: max lte {ml:.2e}  (expect ~0)")
    print(f"[probe self-test] quadratic: max lte {max(r['lte'] or 0 for r in quad.rows):.3e} (expect >0)")

    # order check: halve the step in t, error per step should drop ~8x (h^3)
    def lte_at(h):
        s0 = 0.5
        ss = [s0 * math.exp(h), s0, s0 * math.exp(-h), 0.0]
        p = run(lambda t: A + C * t * t, ss)
        return [r["lte"] for r in p.rows if r["lte"] is not None][0]
    r1, r2 = lte_at(0.4), lte_at(0.2)
    print(f"[probe self-test] h-halving ratio {r1 / r2:.2f}  (expect ~8 for O(h^3))")

    # churn flagging
    p = Probe("s2", [False, True, False, False, False])
    for i, s in enumerate([0.8, 0.6, 0.45, 0.33, 0.24]):
        p.observe(i, torch.tensor(s), A + C * (-math.log(s)) ** 2)
    print(f"[probe self-test] clean flags {[r['clean'] for r in p.rows]} "
          f"(expect [None, None, False, False, True])")
    finish(p)
    print(f"[probe self-test] sweep_fields keys: {sorted(k for k in sweep_fields() if k != 'trajectory')}")
    ok = (mc < 1e-5 and md < 1e-5 and ml < 1e-4 and 6.0 < r1 / r2 < 10.0
          and [r['clean'] for r in p.rows] == [None, None, False, False, True])
    print("[probe self-test] " + ("OK" if ok else "FAILED"))
