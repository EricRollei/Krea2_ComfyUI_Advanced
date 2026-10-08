# Spec: training-free style reference for Eric_Krea2 (port of nkxx188/ComfyUI-Krea2-StyleTransfer)

Status: DRAFT for Eric's review - no code written yet. Source: MIT (c) 2026 jieg9341-lab, port with attribution.

## 1. What the original actually does (read from nodes.py)

Per sampling call the model runs a 2x batch: [target, reference] at the same timestep.

1. Reference trajectory ("RF cache"): the clean reference latent (VAE-encoded at the target's exact
   size, cover-cropped) is taken UP the noise schedule once per run before step 1. flowturbo_pc =
   Heun steps integrating the model's own velocity (target prompt) from sigma 0 to each sampler sigma,
   each state blended gamma=0.5 with the straight-line prior (1-s)*ref + s*eps (fixed eps, seed 42).
   The reference branch therefore sits at an on-manifold noisy version of the reference.
   Cost ~2x step count in single-batch evals.
2. Reference branch runs normally; it only supplies K/V.
3. Target branch, blocks 7-27 only (blocks 0-6 untouched):
   - target queries attend to [target K ; reference image K] / [target V ; reference image V]
     in one joint softmax.
   - reference K reweighted per RoPE frequency band: high-freq positional bands fade x1.04 -> x0
     over the run, low-freq bands x1.0 -> x1.10 (low_scale_end), frame axis at the low scale; then
     x ref_k_strength 1.06. Early: target can align to the reference layout; late: it cannot copy
     positions (no content leakage) but still pulls texture/palette/stroke statistics.
   - AdaIN of target image Q,K toward the reference's per-channel token stats (x0.85).
   - value: at the preset ref_value_mix=1.0 -> raw reference V (value_adain then inert).
   - output = styled attention (at strength 1 the native attention it computes is x0 - wasted).
4. Recommended preset: strength 1.0, ref_k 1.06, ref_value_mix 1.0, value_adain 0.65,
   flowturbo_pc, gamma 0.5, beta 2.5, high 1.04->0, low 1.0->1.10, adain 0.85, blocks 7-27.
   Their sampler: Turbo, 8 steps, euler_ancestral/simple, cfg 1, same positive prompt as ref cond.

No LoRA, no training, no ostris reference recipe.

## 2. Port design (same pattern as _control.py)

_style_ref.py + node EricKrea2StyleReference + Ultra socket `style` (appended last; socket = no
widgets_values slot).
- Forward wrapper on pipe.transformer per generation: append reference row, duplicate text
  states/mask/timestep, run, return target row. Sigma = diffusers timestep.
- Styled attention processor swapped onto transformer_blocks[7..27].attn for the run, originals
  restored; attention backend copied from the original processor. Diffusers [B,L,H,D]; RoPE pairs
  interleaved like comfy; axes 32/48/48 match -> scale_vec ports 1:1.
- Per stage: reference cover-cropped + re-encoded at the stage grid (from position_ids), own
  trajectory up to that stage's top sigma. Strength per stage S1/S2/S3.

Deliberate deviations:
- D1 trajectory on an internal grid (default 12 pts) + linear interpolation for any requested sigma,
  instead of exact sampler sigmas - our RK/hybrid/deis samplers evaluate at intermediate sigmas.
  trajectory_steps widget; `linear` mode = free (prior only).
- D2 skip the native attention when mix >= 1 (theirs multiplies it by 0).
- D3 reference branch uses each call's own text states (identical at Turbo g=0).

## 3. Cost (estimate, to be measured)
~2.3-2.6x per controlled step at ~3 MP + trajectory (~12 evals) => S1-only style run ~3x a plain S1.
8 MP stages worse (~3x/step) -> S2/S3 default 0. Later optimisation: cache reference K/V from the
trajectory pass to drop the 2x batch (~1.6 GB per cached sigma at 3 MP).

## 4. Node
reference_image; preset [recommended|custom]; strength 0-2; s1/s2/s3 multipliers (1/0/0);
custom: ref_k_strength, low_scale_end, high_scale_start, adain_strength, ref_value_mix,
value_adain_strength, beta, blocks, trajectory [model_pc|linear], gamma, trajectory_steps.
Out: KREA2_STYLE + fitted-reference preview. Two-reference mode deferred.

## 5. Interactions
LoRA stack: both branches (as original). Control: restrict to target row (small _control.py change).
ref_latents edit pathway: refused in the same run in v1. Sweep: unaffected.

## 6. Validation
1. strength 0 == stock processor bit-exact; hooks removed -> baseline identical.
2. Headless strip, Turbo, euler 8 steps, 1 MP: reference | no style | style.
3. Optional parity A/B vs the original pack in ComfyUI (same ref/prompt/seed; close, not identical).
4. Eric's real references, S1-only ~3 MP, hybrids/EXP curves.
