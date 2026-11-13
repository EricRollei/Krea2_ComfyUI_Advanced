# Spec: history gain, final-sigma floor, normalized churn noise
2026-11-02 - status: DROPPED 2026-11-03 after a live sweep - see dev_notes_2026-11-03_history_gain_floor_noisenorm_DROPPED.md. Code removed.

Source of the ideas: envy-ai/ComfyUI-DPMpp-2M-Sharp (GPL-3.0; its res.py also carries the
RES4LYF commercial-service restriction). NOTHING is copied - all three are re-derived here in our
own solver's terms. The ideas are trivial; the code would be ours.

All three live on the **Eric Krea2 Hybrid Sigma Handoff** node (Ultra panel untouched), all default
OFF, and an unwired / all-off Ultra stays bit-identical to today.

---------------------------------------------------------------------------------------------------
## Honest expected value (read first)

| Feature | What it changes | Expected gain | Cost |
|---|---|---|---|
| A. history gain | late-step x0 target undershoot in multistep tails -> model re-commits detail | **modest, uncertain** - a 4th detail lever next to detail bias, handoff sigma and eta. Its distinct trait: adds detail with NO added noise and NO extra evals, in the deterministic tail. Maybe 1-in-3 it becomes a keeper. | ~40 lines solver + node |
| B. final-sigma floor | stop the last stage just above 0 -> a little of the trajectory's own residual noise survives into the decode | **low-moderate, taste-dependent** - VAE-rendered micro-texture, not pixel grain; the controllable form of the "old bf16 dither looked better" effect from the euler parity thread | ~20 lines |
| C. normalized churn noise | per-channel zero-mean / unit-std on WHITE churn draws (shaped noise already is) | **low, probably subtle** - raw draws carry a per-channel mean error ~1/sqrt(seq) (~1% at 2 MP). Most plausible place to see it: the lcm head (full re-noise every step) and high-eta res_2m. Color-cast hygiene, cheap to A/B | ~10 lines |

None of these is a likely breakthrough. Together they are cheap, sweepable, and default-off - worth
one build + one characterization sweep, then keep or drop each on evidence.

---------------------------------------------------------------------------------------------------
## Current code (verified on disk 2026-11-02)
- `_res_solver.py`: res_multistep (one `old_denoised`), deis_sample / abnorsett_sample (`x0_hist`
  list), er_sde (old_denoised + derivative history). All terminal steps go fully clean (sigma 0:
  `x = denoised` or flow-Euler to 0).
- `krea2_multistage_ultra.py::_res_denoise_packed`: white churn = raw `torch.randn(x.shape, generator)`;
  shaped noise (`_freq_filter`) IS mean/std normalized per channel; lcm family forces raw white `_ns`.
- `_sampling_extras.py`: KEYS / ENGINE_DEFAULTS / resolve() precedence (sweep cell > node > default).

---------------------------------------------------------------------------------------------------
## A. History gain (`sN_history_gain`, `history_gain_mode`)

### Mechanism (re-derived)
Multistep samplers form the step's x0 forcing as `b0*D_now + sum_j b_j*D_hist_j`. Consistency forces
the history coefficients to sum NEGATIVE (they encode the extrapolation). Multiplying the history
terms by `k > 1` therefore subtracts a little extra prediction from the target: the step undershoots,
the latent is slightly under-signaled for its sigma, and the model's next prediction re-commits
structure/edges. It is a bias, not higher order: it does not vanish with more steps.

### Definition
At step i of a supported sampler, with current sigma `s_i`:
    k_i = 1 + g * (1 - s_i)**2          # ramp by SIGMA: 0 at sigma 1, -> g near sigma 0
Every HISTORY prediction used in that step's update (not D_now) is replaced by H':
- `full`   : H' = k_i * H                                  (their behaviour; also scales DC -> can
                                                             shift brightness/color)
- `detail` : H' = H + (k_i - 1) * (H - mean_seq(H))        (per-channel mean over the token axis of
                                                             the packed [B, seq, 64] latent preserved;
                                                             only deviations scaled -> color/DC safe)
Terminal step (sigma_down == 0) untouched (first-order exit), as in their code.

Why sigma-ramped instead of their `(i/len(sigmas))**2`: step-index ramps restart at 0 in our S2/S3
windows (barely engage on a 5-6 step refine) and change meaning with step count/schedule - the same
trap as the old hybrid_split fraction. Sigma = same meaning everywhere (same doctrine as handoff).

Magnitude: g 0.15, last history step sigma ~0.15 -> k-1 ~0.11 -> res_2m target undershoot ~5%.
Suggested sweep: 0, 0.1, 0.2, 0.4, 0.8 (they allow up to 100; we cap at 2.0).

### Applicability (stated up front)
- HONORED: res_2m, deis_3m, abnorsett_3m, abnorsett_4m, and the TAILS of lcm_hybrid (deis_3m tail)
  and lcm_hybrid2 (res_2m tail). On the hybrids the lcm head is unaffected.
- INERT: euler (incl. S1 pipe path), lcm, res_2s, lawson4_4s, etdrk4_4s, rk6_7s (single-step, no
  history), er_sde (deliberately excluded - its history terms are derivative estimates inside a
  line-faithful port; changing them breaks the faithfulness we verified).
- Console per stage: `S2 history gain 0.20 (detail): honored by res_2m` or `... INERT for rk6_7s`.

### Widgets (node, appended AFTER churn_form - positional rule)
- `s1_history_gain`, `s2_history_gain`, `s3_history_gain`: FLOAT 0-2.0 step 0.01, default 0 (off).
- `history_gain_mode`: [detail | full], default detail.

---------------------------------------------------------------------------------------------------
## B. Final-sigma floor (`final_sigma_floor`)

### Definition
Applies to the FINAL stage only (the stage whose output is decoded); earlier stages' outputs are
upscaled and re-noised anyway, so a floor there is meaningless.
- If the final stage's ladder ends `[..., s_last, 0]` and `s_last > f`: the ladder end becomes
  `[..., s_last, f]` - same number of model evals, no clean exit step.
- If `s_last <= f`: the trailing 0 is dropped (stops at s_last; console notes the floor was below
  the last ladder point).
- Output rescale: x / (1 - f), so the signal level is restored (rf: x_f = (1-f)x0 + f*eps ->
  x0 + f/(1-f)*eps). Without this, f 0.02 would also flatten contrast 2%.
- The residual is the trajectory's OWN remaining noise (coherent with the image), except after an
  lcm step or a churned step, where it is that step's fresh draw at sigma f.

### Applicability
- All solver-path samplers support ending above 0 already (early-stop windows use it).
- INERT: S1 euler pipe path (S1-only euler runs) - console note; and any final stage that already
  early-stops via end_step (it already ends above 0).

### Widget
- `final_sigma_floor`: FLOAT 0-0.05 step 0.001, default 0 (off). Suggested 0.005, 0.01, 0.02, 0.03.

---------------------------------------------------------------------------------------------------
## C. Normalized churn noise (`churn_noise_norm`)

### Definition
Wrap the WHITE noise samplers built in `_res_denoise_packed` (the eta/er_sde sampler AND the
lcm-family `_ns`): per draw, per batch item, per packed channel:
    n = (n - mean_seq(n)) / std_seq(n)
Shaped noise is already normalized (no change). RNG consumption unchanged -> same seed stays
comparable across on/off. Primary init noise (S1 prepare_latents, stage re-noise) NOT touched -
see open decision 3.

### Applicability
Everything that draws churn: res/deis/abnorsett/rk/lawson/etdrk at eta > 0, er_sde (always), lcm
and the lcm-hybrid heads (always). INERT at eta 0 for the non-lcm, non-er_sde samplers.

### Widget
- `churn_noise_norm`: [off | per_channel], default off.

---------------------------------------------------------------------------------------------------
## Implementation (for review)
1. `_sampling_extras.py`: add KEYS `s1/s2/s3_history_gain`, `history_gain_mode`, `final_sigma_floor`,
   `churn_noise_norm`; ENGINE_DEFAULTS 0/"detail"/0/"off"; resolve() type coercion; describe()
   picks them up automatically; helper `history_gains(resolved)`.
2. `_res_solver.py`: helper `_apply_history_gain(H, sigma, g, mode)` (returns H unchanged when g==0 -
   bit-identical path); new kwargs `history_gain=0.0, history_gain_mode="detail"` on
   res_multistep_sample, deis_sample, abnorsett_sample, lcm_hybrid_sample (forwarded to the tail
   only). Applied at use time to `old_denoised` / each `x0_hist` entry in the update - the stored
   history stays raw (no compounding across steps).
3. `krea2_multistage_ultra.py`:
   - `_res_denoise_packed(..., history_gain=0.0, history_gain_mode="detail", sigma_floor=0.0,
     noise_norm="off")`; noise wrapper for C; ladder edit + rescale for B (only when sigma_floor > 0
     and the window ends at 0); per-stage console line for A/B/C (honored / inert).
   - generate(): new explicit kwargs (None defaults) into `_sxm.resolve`; pass history gain per
     stage; pass sigma_floor only to the final active stage (S1 in S1-only runs, else last enabled).
   - Ultra V2 inherits via the bundle; V2 sweep adapter needs no change if kwargs flow through
     (verify during build).
4. Node `krea2_hybrid_handoff.py`: the 6 widgets appended at the END of the required dict, tooltips
   carrying the applicability lists above. Consider renaming display to "Eric Krea2 Sampling
   Extras" (display name only - NODE id unchanged, saved workflows safe) - open decision 4.
5. Sweep Plan: ONE new field `history_gains` appended last (stage-scoped like hybrid_handoffs;
   caption `hg0.2`); dedupe drops sN_history_gain on cells whose stage sampler is inert. Floor and
   noise-norm go through combos_json first (captions `fl0.01`, `nn`); real fields only if they earn it.
6. Tests (docs/tools/test_sampling_extras_2026-11.py, CPU, synthetic denoiser):
   - all off == current outputs bit-identical, every sampler;
   - detail mode: target per-channel mean equals g=0 target's mean;
   - g>0 breaks constant-x0 exactness by the expected small amount (documented, not a bug);
   - floor: output == ladder ending at f, rescaled; eval count unchanged;
   - norm: per-channel mean 0 / std 1 of draws; same-seed draw sequence unchanged.
7. README: Samplers section - "History gain", "Final-sigma floor", "Normalized churn" with the
   applicability lists; What's New entry.

## Suggested first experiments (after build)
1. S1 lcm_hybrid2 + your best EXP curve, fixed seed, 2 prompts:
   `history_gains: 0, 0.1, 0.2, 0.4, 0.8` x mode {detail, full} (mode via combos_json). Watch the
   vectorscope on `full` - if detail mode matches its sharpness without the color move, full can go.
2. Same base, final stage: `final_sigma_floor` 0 / 0.005 / 0.01 / 0.02 - judge at 100% crop on skin,
   sky, hair.
3. churn_noise_norm off/on on the lcm_hybrid2 recipe and on res_2m eta 0.6 - same seeds; look for
   color-cast differences between seeds (norm should reduce seed-to-seed cast variance if it matters).

## Open decisions
1. history_gain_mode default `detail` (proposed) vs `full` (source behaviour).
2. Ramp shape (1 - sigma)^2 (proposed) vs linear (1 - sigma) or a start-sigma knob - keep simple first.
3. Should churn_noise_norm also normalize the primary init / stage re-noise? Proposed NO for now
   (changes every image, not just churn; separate experiment).
4. Rename the node's display name to "Eric Krea2 Sampling Extras" (id unchanged)?
5. Include er_sde in history gain? Proposed no (faithful-port contract).
