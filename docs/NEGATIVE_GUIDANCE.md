# Negative guidance on Turbo: NAG & NegPiP

[<- Back to the README](../README.md) | [All docs](../README.md#documentation)

## Negative guidance (NAG & NegPiP)

```
Eric Krea2 Negative Guidance -> (guidance) Multi-Stage Ultra
```

Both are **off by default**, both work at **guidance 0** (Turbo), and they can be combined
with each other and with an edit. Ultra's `negative_prompt` stays the CFG negative.

**NAG - Normalized Attention Guidance.** A negative text stream (`nag_negative`) runs
through the blocks next to the positive one. For the image queries the attention output is
pushed away from the negative and renormalised:
`z = z+ + phi (z+ - z-)`, L1 ratio clamped to `tau`, blended `alpha` into the positive.
Text and edit-source tokens keep their positive attention.

- `nag_phi` 4.0, `nag_tau` 2.5, `nag_alpha` 0.25 (the published defaults),
  `nag_stages` s1 / s1_s2 / all, `nag_sigma_start/end` limit it to a sigma window.
- In tests it suppressed an unwanted feature (wings on a llama-bird) cleanly with a small
  remnant. Cost: ~+23% per step on text-to-image, ~+12% on edits (single pass - CFG would
  be +100%).

**NegPiP.** Put what to subtract in the node's **`negpip_negative`** field (2026-11-03), e.g.
`beads of sweat, water droplets, wet skin, dripping` - each comma/newline item becomes its own
group at weight -1.0; write `(wet ground:-0.5)` for a different weight. You can also still write
`(phrase:-1.0)` groups directly in Ultra's prompt or the Edit instruction; both sources are used
when `negpip` is on. The phrase's attention **values** are multiplied by its weight inside the
joint blocks, so a negative weight subtracts the concept instead of adding it.
- To use NAG and NegPiP together, give them different concepts: `nag_negative` for one set,
  `negpip_negative` for another (the same concept in both over-suppresses).

- `negpip_mode` - `in_place` (default; the phrase stays where it is, its tokens are flagged -
  cleanest in tests) or `lifted` (the phrase is removed and encoded on its own, appended).
- `negpip_strength` multiplies every negative weight; `negpip_block_start/end` limit the
  blocks; `negpip_text_fusion` also flips inside the text-fusion attention (stronger).
- Positive weights are stripped (Krea 2 conditioning has no prompt weighting). With a
  precomputed `prompt_conditioning` (Vision Prompt) the text can't be parsed - put the
  groups in Ultra's prompt instead. Cost ~0.

**Using both** on the same concept over-suppressed in tests (dark patches) - pick one per
concept.

**Field use (Nov 2026).** NAG and NegPiP both work well in practice. The standout NAG use is
suppressing Turbo's wet / drippy artifacts at higher step counts (sweat, droplets, wet ground,
"spilled jam" skin) - see the field update under the Stage-1 recipe.

**Tuning NAG** (what each setting does - from the code, `_edit_guidance.nag_combine`):
for every image token, NAG builds `g = z+ + phi (z+ - z-)`, shrinks `g` if its size (L1 norm)
exceeds `tau` x the positive's, then outputs `z+ + alpha (g - z+)`.
- `nag_alpha` (0.25) - the **dose**: how much of the guided result is mixed in. The most linear,
  predictable strength control; 0 = off. Lower it first if NAG feels too strong (try 0.15, 0.10).
- `nag_phi` (4.0) - **how hard it pushes away** from the negative. Bigger = a stronger push, but
  only until the `tau` cap bites: at the defaults the cap often limits it, so raising phi may change
  little, while lowering phi (2-3) does soften it.
- `nag_tau` (2.5) - the **safety cap** on how far the guided features may grow. Lower (1.5-2)
  = gentler and less risk of side effects (tone / texture shifts); higher lets phi act fully.
- `nag_sigma_start / end` - **when** it acts. Wet details are fine-detail features, formed in the
  middle and late sigmas; a window like start 0.7, end 0 leaves composition (sigma 1.0-0.7) untouched
  and may keep the suppression while reducing side effects. Untested - a good first experiment.
- `nag_stages` - s1 by default. If refine stages (S2/S3) add the drips back, use s1_s2 or all.
Suggested gentle starting point (untested): alpha 0.15, phi 3, tau 2.0, sigma window 0.7-0.

All three features (edit sources + ref_boost, NAG, NegPiP) share one exact attention core:
keys are attended group by group and merged by log-sum-exp, so ref_boost needs no dense
L x L bias (which would force slow SDPA and run out of memory at 3 MP) and NAG's positive
and negative passes share the expensive image attention. Verified against dense reference
implementations (docs/tools/test_edit_guidance_math.py).

### Long prompts and NegPiP `in_place` (Nov 2026)

Krea 2's text block holds about 507 prompt tokens. With `in_place`, NegPiP phrases - including everything in
the `negpip_negative` field, which is appended to the end of the prompt - count toward that budget. When
prompt + negatives run over it, the phrases that no longer fit are moved to `lifted` placement automatically
(encoded on their own and appended as extra rows), so the whole negative list still applies. The console
prints a NOTE listing them, and in a sweep each cell's `negpip` column / manifest entry records it.
Before this fix, prompts of roughly 460-507 tokens with NegPiP on could produce pure-noise images.
