# History gain / final-sigma floor / normalized churn - built, tested, DROPPED (2026-11-03)

Built 2026-11-02 from docs/spec_history_gain_floor_noisenorm_2026-11-02.md (ideas from
envy-ai/ComfyUI-DPMpp-2M-Sharp, re-derived, no code copied). Removed 2026-11-03 at Eric's call:
limited gains, and most settings produced unusable images - he does not want knobs that blow up
the image across most of their range.

Revert: docs/tools/revert_histgain_2026-11-03.py restored the `_pre_histgain_2026-11-02` backups
after verifying each file was byte-identical to what the patchers produced (nothing else lost).
Kept for reference: docs/tools/patch_histgain_floor_norm_2026-11-02.py,
patch_histgain_readme_2026-11-02.py, test_histgain_floor_norm.py (re-applying the patcher to the
current files would rebuild the feature if it is ever wanted again).

## Test
Sweep `20261103_004628_trigger-test-pilot-5-edited_...` (checkpoint darkbeast5csep 3365025),
60 cells, combos mode, S1 only: lcm_hybrid2, 10 steps, handoff 0.65, eta 0, 2.4 MP, cfg 0.1,
2 prompts x sigma profiles OPT-AB412 / OPT-DEIS12 x 15 lines (baseline, gain 0.4-2.0 detail/full,
floor 0.01/0.025/0.05, noise norm, two stacks). Review sheet rated in Excel.

## What we learned
1. **OPT-DEIS12 > OPT-AB412** on this checkpoint/recipe: rated 5/5 vs 5/4 at baseline; the
   trajectory probe agrees (s1 lte_sum ~0.021 / 0.016 vs 0.031 / 0.024 - smoother path).
   Best cells: DEIS12 baseline, DEIS12 + per-channel noise, DEIS12 + gain 0.4 (full) - a tie.
2. **History gain does NOT sharpen on Krea2 lcm_hybrid2 tails - it softens.** Every ladder was
   monotonic: sharp_ten and clip fall with gain (DEIS12 p1: 14024 / 1.58% at 0 -> 6914 / 0.01%
   at 2.0). Eye: contrast drops, blocked shadows open up, but colour shifts and other areas go
   soft - net not desirable; high values unusable. `full` lowered clip slightly more than
   `detail` at equal gain. The repo's "sharper" claim (made for Qwen Image 2.1 + DPM++ 2M) did
   not transfer. Only plausible use: a highlight/clip tamer on hot renders (AB412 p2 5.55% ->
   ~4% at 0.4) - not worth a knob. Negative gain (overshoot) was considered and judged likely to
   be the mirror (contrast/clip up, colour shift, ringing), not a new capability.
   CPU toy test: `detail` mode keeps per-step channel means exactly, but the final image's channel
   means drift about as much as `full` - the model's response feeds the shift back.
3. **Final-sigma floor as built was broken:** smaller floor = WORSE (0.01: clip 6-13%, sharp_ten
   x3; 0.05 milder). Cause: the floor step became an ordinary 2nd-order res_2m step down to a
   tiny sigma = a huge step in log-sigma, and the multistep extrapolation blew up (same
   ill-conditioning family as linear_quadratic). A correct version would take the floor step as a
   first-order exit (or re-noise a clean result at the floor level); expected payoff = subtle
   grain only, so not pursued. LESSON: any change that ends a multistep solver above 0 must use a
   first-order final step, never the solver's own high-order update.
4. **Normalized churn noise: harmless, invisible.** DEIS12 clip 1.58 -> 1.38 (p1), 2.86 -> 2.87
   (p2), same ratings and uniqueness. With eta 0 only the lcm head draws churn; raw per-channel
   mean error ~1/sqrt(seq) is too small to see at 2.4 MP.
5. The stacked combo cells (clip 18-47%, uniqueness 34-50) were dominated by the floor damage.

## If revisited
Only worth it with a new idea, not a direction flip: e.g. a frequency-selective (high-pass)
history adjustment, or the first-order floor exit tested alone as a grain option.
