# Spec: SDE band (A) + trajectory error probe (B)
2026-10-22 - status: PROPOSED, no code written. Diagnose-first: agree on this doc before any edit.

Origin: reading ComfyUI's SA-Solver (`comfy/k_diffusion/sa_solver.py`, `sample_sa_solver` in
`sampling.py`, `SamplerSASolver` in `comfy_extras/nodes_custom_sampler.py`) and the
YakovSava/adaptive-step-samplers repo. Two ideas are worth having independent of SA-Solver itself.

---

## 0. Background facts this spec rests on (verified in source)

- SA-Solver's noise window is set in model-time percent and converted via
  `model_sampling.percent_to_sigma`, then tested against **sigma_next**, inclusive.
  For Krea2 (FLUX-type, exponential shift mu = 1.15, sigma(t) = e^mu / (e^mu + 1/t - 1)):

  | percent | 0.2  | 0.5  | 0.8  | 0.9  |
  |---------|------|------|------|------|
  | sigma   | 0.93 | 0.76 | 0.44 | 0.26 |

  So the default "middle band" on Krea2 is sigma 0.93 -> 0.44: the composition / mid-structure half.
  Everything below 0.44 is deterministic. Percent windows do NOT transfer across models with
  different shifts -> **our band is specified in sigma, not percent.**

- Our eta today (`_res_solver.py`): every churn-capable sampler calls
  `get_ancestral_step(sigma, sigma_next, eta)` every step and adds
  `noise_sampler(...) * s_noise * sigma_up` when `eta > 0`. Eta is uniform across the whole window.

- **Finding (needs your ruling, see A.5):** that ancestral step is the VE form. Comfy itself uses an
  RF-specific form for flow models in `sample_euler_ancestral_RF` and the dpmpp_2s RF branch
  (rescale by alpha_next/alpha_down, renoise_coeff = sqrt(s_next^2 - s_down^2 * a_next^2 / a_down^2)),
  but comfy's own `res_multistep` ancestral uses the VE form too - so we match comfy's
  res_multistep_ancestral, and comfy is internally inconsistent. On a flow model the VE form
  gets the noise variance right but leaves the **signal coefficient too high**: after the step the
  state carries (1 - sigma_down)·x0 while the model is told sigma_next, which expects (1 - sigma_next)·x0.
  Worked example (one step 0.76 -> 0.69):
  - eta 0.3: signal +2% over what the label implies - negligible.
  - eta 1.0: signal +21% - large.
  Plausible contributor to "high eta breaks things" (the Jul-23 rainbow on lcm_hybrid's deis tail
  cleared at eta 0). Hypothesis, not proven.

---

## A. SDE band - churn only inside a sigma band

### A.1 Behaviour
Per stage, two floats define a band [lo, hi] in sigma. On each step the effective eta is
`eta_i = eta if lo <= sigma_next <= hi else 0.0` (same test as SA-Solver: sigma_next, inclusive).
Defaults hi = 1.0, lo = 0.0 => band covers everything => **bit-identical to today**.

### A.2 Widgets (Ultra; V2 inherits via `EricKrea2MultistageUltra.INPUT_TYPES()`)
Appended at the TRUE END of the optional dict, after everything currently there (positional safety):

| widget | type | default | range | tooltip gist |
|---|---|---|---|---|
| s1_sde_hi | FLOAT | 1.0 | 0-1, step 0.01 | churn only while sigma_next <= this |
| s1_sde_lo | FLOAT | 0.0 | 0-1, step 0.01 | churn only while sigma_next >= this |
| s2_sde_hi / s2_sde_lo | FLOAT | 1.0 / 0.0 | same | per stage |
| s3_sde_hi / s3_sde_lo | FLOAT | 1.0 / 0.0 | same | per stage |
| churn_form | COMBO [ve, rf] | ve | - | ONLY if you approve A.5; default = current behaviour |

lo > hi => warn once, treat as empty band (stage runs eta 0). New keys flow into presets / PNG chunks
automatically (same as the Jul-21 shift widgets). Old presets lack the keys => defaults => unchanged.

### A.3 Solver change (`_res_solver.py`)
- New helper `_eta_at(eta, sigma_next, band) -> float`; `band=None` => return eta (legacy).
- Each churn-capable sampler gains `eta_band=None` kwarg; inside the loop compute `eta_i` once and use
  it in BOTH places eta is used today: `get_ancestral_step(..., eta=eta_i)` and the `eta_i > 0` noise
  gate. Applies to: res_multistep_sample, res_2s_sample, deis_sample, abnorsett_sample,
  lawson4_sample, etdrk4_sample, rk_explicit_sample, and the hybrid tails (lcm_hybrid_sample passes
  eta_band through to its res/deis tail).
- **Excluded:** euler (eta clamped to 0 by contract), lcm (eta ignored, full churn by design),
  er_sde (its noise scaler also shapes the drift - zeroing noise outside a band without the matching
  drift change would make it inconsistent; revisit only if wanted).
- Multistep history (old_sigma_down etc.) is unaffected: steps outside the band just have
  sigma_down = sigma_next, exactly as at eta 0.
- noise_sampler creation in `_res_denoise_packed` (line ~455) is unchanged: it is created when eta > 0,
  and the band only decides per step whether it is called. Seeded draw order changes only when the
  band is narrower than the window (expected - different experiment).

### A.4 Dispatch, telemetry, sweep
- `_res_denoise_packed` gets `eta_band=(lo, hi)` per stage, threaded the same way as the
  shift-policy keys were (`_generate_inner` -> `_denoise_stage` -> `_res_denoise_packed`).
- Always-on, fail-silent console line per stage when eta > 0:
  `[EricKrea2] S2 SDE band σ[0.76→0.26]: 5/12 steps churned (σ_next 0.69...0.39)`
  - this matters because how many steps fall in a band depends on schedule, mu and window.
- Sweep: new axis `sde_bands` = lines of `hi-lo` (e.g. `1.0-0.0`, `0.93-0.44`, `0.5-0.15`),
  stage-scoped via stage_scope like steps/eta. Inert-dedupe: drop the band from the effective key when
  the cell's stage eta == 0 or its sampler is euler / lcm / er_sde. Caption shorthand: `b.93-.44`.

### A.5 Open decision - churn form (ve vs rf)
Recommendation: add `churn_form` [ve | rf] with default ve (no change to any saved workflow), implement
rf exactly as comfy's `sample_euler_ancestral_RF` (downstep_ratio = 1 + (s_next/s - 1)*eta;
sigma_down = s_next*downstep_ratio; integrate to sigma_down; x = (a_next/a_down)*x + noise*renoise_coeff),
then A/B in a sweep at eta 0.3 / 0.8 / 1.5. If rf wins, flip the default later (your call).
Alternatives: (2) leave it, document only; (3) switch silently (not recommended - changes every saved eta>0 image).
Note: eta **values** are not interchangeable between forms (rf's eta is a downstep ratio, ve's a
variance fraction), so sweep both forms rather than comparing at one number.

### A.6 Suggested first experiments
1. Composition-diversity knob for turbo: S1 res_2m or hybrid tail, eta 1.0, band 1.0-0.5 vs 0.5-0.15
   vs full, seeds_axis x8, read mean uniqueness per sweep (same protocol as VarSeed).
2. SA-Solver's default band on Krea2 = `0.93-0.44`; compare against your lcm_hybrid2 S1 recipe.
3. Refine stages: S2 deis_3m eta 0.8 band 0.5-0.2 (detail band only) vs full window.

### A.7 Tests (container + `__main__` self-tests)
- band (0,1) == legacy bit-for-bit for every sampler at eta 0.5 with a shared seeded noise_sampler.
- empty band == eta 0 bit-for-bit.
- churned-step count matches the telemetry line on a known ladder.
- rf form at eta 0 == ve form at eta 0 (both reduce to the ODE).

---

## B. Trajectory probe - free per-step error / curvature measurement

### B.1 What it measures
A diagnostic only: it never changes the image. At every model evaluation we already have
(sigma_i, x_i, D_i) where D = the model's clean-image prediction (`callback(i, sigma, denoised, x)`).
From a rolling history of the last 3 D's in lambda = log((1 - sigma)/sigma) (half-log-SNR, same variable
SA-Solver uses):

1. **x0 drift** `drift_i = rms(D_i - D_{i-1}) / |lambda_i - lambda_{i-1}|` - how fast the model's
   idea of the final image changes per unit log-SNR. Cheap and intuitive; sharp peaks = sigma bands
   where the model is "deciding" something.
2. **LTE estimate** (Milne device): predictor = order-2 exponential Adams-Bashforth from (D_{i-1}, D_i),
   corrector = order-2 exponential Adams-Moulton from (D_i, D_{i+1}), both over the same step, both
   via the existing `_exp_lagrange_coeffs` machinery. `lte_i = rms(C - P) / rms(x)`. This is the
   free error estimate the adaptive-samplers repo found tracks true local error at a ratio of 0.93-0.98.
   Computed one step late (needs D_{i+1}); computed from the logged D's only, so it is
   **sampler-agnostic** - it measures the trajectory the model actually took, whatever solver drove it.

Both are relative RMS numbers (per element, fp32), so cells at different resolutions compare.

### B.2 Caveats (to be printed in the README, not hidden)
- Churn contaminates both metrics: noise injected between evals changes D for reasons other than
  curvature. Each step records `eta_i` (from A) and the summaries are computed over churn-free steps
  by default; read LTE at eta 0 or outside the SDE band for clean numbers.
- Multi-eval samplers (res_2s, rk6_7s, lawson4, etdrk4): the callback fires once per step and
  `res_2s_sample` passes `d1` rather than `denoised` (line ~190) - **verify at implementation** what
  d1 is; if it is not an x0 prediction the probe skips that sampler (records "n/a") rather than lie.
- lcm / lcm hybrids: the churn half is maximal ancestral by design - probe values there are
  meaningless and flagged; the hybrid tail is fine.
- It is a diagnostic to inform curve design, NOT a step controller (the repo's adaptive-step result:
  error-equidistributed schedules lost to the plain `normal` scheduler).

### B.3 Where it lives
- New `_traj_probe.py` (package-agnostic, like `_sweep_core.py`): class `TrajectoryProbe` with
  `observe(i, sigma, denoised, x, eta_i)` and `summary()`; holds only a 3-deep GPU history + per-step
  python floats (memory negligible).
- Hook: `_res_denoise_packed._cb` (already wraps progress) calls `probe.observe(...)` inside try/except
  (fail-silent, same doctrine as `_telemetry`).
- Always-on (cost: a handful of tensor ops per step) - **or** behind a widget; your call (B.6).

### B.4 Outputs
- Console, one line per stage: `[EricKrea2] S1 probe: lte Σ 0.041 peak 0.012 @σ0.71, drift peak @σ0.84, late(<0.3) 18%`
- Sweep manifest per cell: per-stage arrays `sigma[], drift[], lte[], eta[]` + summaries.
- sweep_summary.csv new columns: `s1_lte_sum, s1_lte_peak, s1_lte_peak_sigma, s1_drift_peak_sigma,
  s1_lte_late_frac` (and s2/s3 when those stages ran) - so they land in the CSV you rate in Excel.
- Optional `probe_curves.png` in the sweep folder: lte and drift vs sigma, one line per cell, log y.
  This is the curve-design view (where should EXP-style curves put their steps).

### B.5 Suggested experiments
1. Same prompt/seed, `linear` 24 steps, eta 0, S1 only, across your ~20 checkpoints (checkpoint sweep
   or repeated runs) -> per-fine-tune LTE profile. Prediction to test: Dark Beast (clean) vs the
   splotchy fine-tunes differ in late-band (σ<0.3) LTE / drift.
2. Overlay the measured LTE profile on the step density of EXP1-EXP6 and the best built-ins: do the
   winning curves put steps where LTE is high?
3. Correlate `lte_late_frac` / `lte_sum` with your Excel rating columns across a 192-cell sweep -
   an internal, physics-based metric to set beside clip% and sharp_ten.

### B.6 Open decisions
- Always-on vs `trajectory_probe` BOOLEAN appended at the end of Ultra (recommend always-on, matching
  the telemetry precedent).
- Include `probe_curves.png` now or later.
- Order: order-2 Milne is the robust choice; order 3 is possible but rides the same Vandermonde
  conditioning risk diagnosed in abnorsett_4m - recommend order 2 only.

### B.7 Tests
- Synthetic D linear in lambda -> lte == 0 (to 1e-7); constant D -> drift == 0 and lte == 0.
- Quadratic D -> lte scales ~h^3 under step halving (order check).
- Calibration: on a smooth synthetic, ratio lte_est / true local error within ~0.8-1.2.

---

## Implementation order (after approval)
1. A without churn_form (pure gate, provably legacy-identical by default) + telemetry + tests.
2. B probe core + console line + manifest/CSV columns.
3. churn_form rf (if approved) + A/B sweep preset.
4. sweep axis `sde_bands`, dedupe, captions; README sections; probe_curves.png.
Backups to docs/backup/ before each base-file edit, as usual.

## Appendix - Krea2 native `simple` ladder, 12 steps, mu 1.15
σ: 1.000, 0.972, 0.940, 0.905, 0.863, 0.816, 0.760, 0.693, 0.612, 0.513, 0.387, 0.223, 0
SA-Solver default band (σ_next in [0.44, 0.93]) => steps 3-9 churn (7 of 12); first two and last three
are deterministic.

---
## Decisions (2026-10-22, Eric)
- A.5: build `churn_form` [ve | rf], default ve, compare via sweep.
- B probe: value questioned ("if a mod is better we'd always use it") - B is PARKED; build A first.
- probe_curves.png: later.
- Probe error estimate: order 2 only - OK.
- Native SA-Solver test (Krea2 turbo, cfg 1): simple good at eta 0 / 0.5 / 1.0; beta good at eta 0 / 0.5 only;
  eta 1.7 bad on both. Gains over our samplers "very subtle" -> SA-Solver port (D) and PC corrector variant (C) dropped.

## Implementation (2026-10-22) - BUILT, deployed, NOT yet live-tested in ComfyUI
- Eric reversed the B parking after the explanation: A + churn_form + B built together.
- Probe variable: t = -log(sigma) (the solvers' own stepping variable), not half-log-SNR as drafted
  above; order-2 Milne with the _exp_lagrange_coeffs kernel. Self-test: const/linear-in-t -> ~1e-8,
  h-halving ratio 6.2 (pre-asymptotic O(h^3)), churn flags correct.
- New file _traj_probe.py; patched _res_solver.py (churn_step/_in_band/_rescale, eta_band +
  churn_form on 7 samplers + lcm_hybrid tail), nodes/krea2_multistage_ultra.py (7 widgets appended
  at the END, threading, band console line, probe hook), ultra_v2 (probe fields into sweep cells),
  _sweep_core.py (probe_* CSV columns, caption b<hi>-<lo>, churn_form), krea2_sweep.py (sde_bands +
  churn_forms widgets appended last, axes, inert dedupe), _sweep_review.py (probe columns in xlsx),
  README (What's New + section). Backups: docs/backup/*_pre_sdeband_2026-10-22.*
- docs/tools/test_sde_band.py: all 6 files compile; legacy vs new BIT-IDENTICAL on all 10 sampler
  paths (band None, explicit (0,1)); empty band == eta 0; rf eta 0 == ve eta 0; rf finite/different;
  one-step signal coefficient 0.760->0.693: ve +1.4% (eta 0.3) / +19.8% (eta 1.0), rf -0.1% / +0.1%;
  band count 7/12 matches appendix.
- Not covered by tests: the GUI widgets, SweepPlan axis parsing inside ComfyUI, the live probe on the model.
