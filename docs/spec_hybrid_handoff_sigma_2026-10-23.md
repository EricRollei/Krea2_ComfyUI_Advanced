# Spec: hybrid handoff by sigma (lcm_hybrid / lcm_hybrid2)
2026-10-23 - status: PROPOSED, no code written. Agree on this before any edit.

## Problem
`sN_hybrid_steps` sets the lcm -> tail handoff as a STEP COUNT. The sigma where the handoff
actually happens then depends on the schedule, the step count, the shift (mu) and the stage
window. That is why the "7 of 12" sweet spot has to be re-found per curve and per fine-tune,
and why a sweep over hybrid_steps on two schedules is not comparing like with like.
Example, 12 steps, Krea2 shift 1.15, handoff after 7 lcm steps: `simple` hands off at sigma ~0.69;
an exponential-family curve that packs steps high hands off much lower. Same widget value,
different recipe.

## Proposal
New per-stage widget **`sN_hybrid_handoff`** (FLOAT, 0-1, default **0 = off**), appended at the END
of Ultra's widgets (same positional rule; covered by the 2026-10-23 blank-value repair).

- 0 (default): unchanged - `sN_hybrid_steps` decides, exactly as today.
- > 0: the lcm head runs until the trajectory reaches that sigma; the tail takes over from the
  first ladder sigma at or below it. Concretely, over the stage's actual (shifted, windowed)
  ladder `sigmas[0..n]`: `k = first j in 1..n-1 with sigmas[j] <= handoff`; head = k steps;
  if no ladder sigma is <= handoff, k = n-1; clamp to [1, n-1] as today (both halves always run).
  `sN_hybrid_steps` is ignored for that stage when the handoff is set (console says so).
- Console, every hybrid stage, both modes:
  `S1 hybrid handoff: 7/12 lcm steps -> tail from sigma 0.61 (handoff 0.65 requested)` /
  `(hybrid_steps 7)` - so the step-count and sigma views are always both visible.
- Probe and SDE band need no change: they already follow the real split (the churn mask uses the
  same k - _traj_probe.churn_mask must call the shared split helper, see Implementation).

## Why sigma, and which value
- One number means the same thing on every schedule, step count and mu - presets travel between
  curves and fine-tunes.
- The probe gives a principled starting point: hand off around the drift peak or just below it
  (your 2026-10-23 reading on this model: drift peak sigma ~0.69). lcm launders while the model is
  still deciding composition; the trajectory-faithful tail takes over once structure is set.
- Refine stages (S2/S3) start at sigma ~0.5-0.76: a handoff above the window start clamps to
  1 lcm step (console notes it) - set refine-stage handoffs inside the window (e.g. 0.35).

## Implementation (for review)
1. `_res_solver.py`: `hybrid_split(sigmas, lcm_steps=0, handoff_sigma=0.0) -> k` (single source of
   truth); `lcm_hybrid_sample(..., handoff_sigma=0.0)` uses it. `handoff_sigma=0` -> today's code path
   bit-for-bit.
2. `_traj_probe.churn_mask(...)` calls the same `hybrid_split` (no duplicated logic).
3. Ultra: widgets `s1/s2/s3_hybrid_handoff` appended last; `_generate_inner` params; threaded to
   `_res_denoise_packed(hybrid_handoff=...)` next to `hybrid_lcm_steps`; handoff console line.
4. Sweep Plan: new field `hybrid_handoffs` appended last (e.g. `0.75, 0.65, 0.55, 0.45`) ->
   `<stage>_hybrid_handoff` for the scoped stages. Dedupe: inert on non-hybrid samplers; when a cell
   sets a handoff, its `<stage>_hybrid_steps` is dropped as inert. Caption `hs0.65` (vs `h7`).
5. Tests: handoff 0 == legacy bit-identical (all hybrid paths); known ladder -> expected k; clamps
   at both ends; probe mask == solver split.
6. README: hybrids section + "Field-tested Stage-1 recipe" gets the sigma form once you have a value.

## Suggested first experiment (after build)
S1 lcm_hybrid2, your panel curve, `hybrid_handoffs: 0.8, 0.7, 0.6, 0.5, 0.4` x two schedules
(e.g. simple + your best EXP). If the best handoff sigma is the same on both schedules while the best
hybrid_steps differed, the sigma form is the better knob and becomes the recipe.

## Open decisions
- Widget name: `sN_hybrid_handoff` (proposed) vs `sN_handoff_sigma`.
- Default stays 0 (off) - recommended, so saved workflows and presets are unchanged.
- Rule when the handoff falls between ladder points: "first sigma at or below" (proposed, never hands
  off early) vs "nearest".

## Decisions + implementation (2026-10-23) - BUILT, deployed, NOT yet live-tested in ComfyUI
- Eric: separate node "Eric Krea2 Hybrid Sigma Handoff" (EricKrea2HybridHandoff, nodes/krea2_hybrid_handoff.py)
  holding the handoff AND the SDE band + churn form; the 7 Ultra widgets added 2026-10-22 were REMOVED
  from Ultra (values saved since then are trailing extras and are ignored on load). Rounding rule: first
  step at or below the handoff sigma. Widget name: sN_hybrid_handoff.
- Ultra gets an optional `sampling` socket (KREA2_SAMPLING); in V2 it sits AFTER `sweep` so sweep keeps its slot.
- Precedence (one place, _sampling_extras.resolve): sweep-cell value > node bundle > legacy defaults
  (handoff 0, band 1/0, churn ve) - unwired Ultra is bit-identical to before.
- Node defaults: s1_hybrid_handoff 0.70 (= 7/12 on linear/simple), s2/s3 0 (off), bands 1/0, churn_form rf
  (from the res_2m sweep: rf never clipped, ve 13-60% clip at eta 0.7-1.0).
- V2 settings output gains a "sampling" section (server _ALLOWED_SECTIONS updated); sweep cells record the
  effective extras. Sweep: hybrid_handoffs axis (appended last), dedupe (inert on non-hybrids; handoff set ->
  hybrid_steps inert), caption hs0.70.
- Tests docs/tools/test_hybrid_handoff.py ALL OK: handoff 0 bit-identical to legacy (both tails, lcm_steps
  0/3/7/11, eta 0/0.5); split cases; handoff 0.70 == lcm_steps 7 bit-identical; probe mask uses the same
  split; precedence. test_sde_band.py still ALL OK.
- Probe: drift peak now excludes the first interval out of pure noise (was always the peak at sigma ~0.94).
