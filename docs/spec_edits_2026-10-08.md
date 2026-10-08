# Spec - Edits: krea2edit v1.2 geometry + ref_boost, NAG, NegPiP (2026-10-08)

Status: APPROVED 2026-10-08 (decisions below) - BUILT, GPU tests in progress. i2i program item 4 (last).

## 0. Sources read (at source, 2026-10-08)
| repo | version / date | license | what we take |
|---|---|---|---|
| lbouaraba/comfyui-krea2edit | v1.2.5 (2026-07-29), full history restored (21 commits; the July "deleted+recreated" state is resolved) | Apache-2.0 | fit geometry, ref_boost, grounded-encode template (port with attribution) |
| conradlocke/krea2-identity-edit (HF) | v1.2 weights (+ r128 0.91 GB, r64 0.46 GB SVD versions) | Krea 2 Community License | the LoRA these nodes serve - **not on disk yet** |
| iljung1106/ComfyUI-Krea2-NAG | 2026-08-12 | MIT | NAG math + Krea2 block placement + edit-aware "target-only" rule |
| ChenDarYen/ComfyUI-NAG | original NAG (Flux/Wan/SD3...) | MIT | reference equations / defaults |
| blue-pen5805/ComfyUI-krea2-negpip | 2026-10-02 | **AGPL-3.0** | README-level mechanism only - **no code copied** (clean reimplementation) |
| crt-nodes `_minimaxh3_negpip.py` (installed) | - | - | mechanism notes (lifted-phrase variant) |

## 1. What we have today and what's wrong with it
`_ref_latents.py` recipe `edit_frame` (built Jul 21, never live-tested) implements the v1/v1.1 mechanism:
`[text | source(frame 1..N) | target(frame 0)]`, shared timestep, live-slice. Gaps vs v1.2.5:

1. **Geometry is wrong for v1.2 weights.** `encode_refs_at_dims` *stretches* a mismatched-AR source
   to the target size (not even the v1 center-crop). v1.2 weights were trained on the `fit` geometry
   below - stretch is out of distribution (the "stretched people" bug krea2edit fixed in v1.2).
2. **Stage gating by token count breaks with `fit`.** The wrapper passes through when
   `gh*gw != img_len`; a fitted source is legitimately *smaller* than the target grid, so the edit
   would silently switch itself off. Needs explicit stage gating instead.
3. **No ref_boost** (the v1.2 fidelity dial; HF card recommends ~4 for strong likeness).
4. **Grounded encode not byte-matched.** Our Vision Prompt `bare_edit` builds the vision blocks but
   (a) sizes by megapixels, not krea2edit's longest-side `grounding_px` (default 768, trained
   384-768), and (b) system prompt / template must be verified identical to krea2edit's
   `KREA2_EDIT_TEMPLATE` (system "Describe the image by detailing the color, shape, size, texture,
   quantity, text, spatial relationships of the objects and background:", vision blocks BEFORE
   the instruction, one block per image, order scene then subject).

## 2. krea2edit v1.2 mechanism (exact, for the port)
**fit geometry** (pixel space, `_fit_encode_image`, target px = latent dims x 8):
- `sc = min(px_h/ih, px_w/iw)` (fit-inside scale).
- *Near-matched AR* (`ih*sc >= px_h*(1-0.08)` and same for w): minimal center-crop to the target AR,
  resize to exactly the target px -> ref grid == target grid (fit == crop at matched AR).
- *Genuine AR mismatch*: ref size `nh = min(max(16, floor16(ih*sc)), floor16(px_h))` (same for w) -
  must be byte-identical to the trainer's `_fit_prep`; then **crop-to-grid**: center-crop the source to
  `round(nh/sc) x round(nw/sc)` before resizing so there is zero squash (v1.2.4 seam/doubling fix).
- Resize bicubic + antialias, VAE-encode at that size.
- **RoPE**: ref tokens stride-1 at a **fractional centered offset** `((th-gh)/2, (tw-gw)/2)` on the
  target grid, frame index i+1 (v1.2.4: integer floor put odd-gap refs half a token off).
- `crop (legacy)` mode: center-crop to target AR, resize to target px, positions = plain grid
  (for v1/v1.1 weights).

**ref_boost**: additive attention-logit bias `log(b)` on **target-query -> ref-key** entries only,
every block (text/ref queries untouched). `ref_boost` = last ref (the subject), `ref_boost_a` = first
ref (scene, two-ref edits). Optional `ref_boost_mask` (ref-image space) limits the last ref's boost
to a region (e.g. a face). Equivalent to multiplying those keys' post-softmax weight by b, then
renormalising. HF: ~4 for strong likeness, >10 breaks removals, <1 loosens.

**Usage contract** (README/HF): <=2 MP; Turbo 8-12 steps CFG 1 for most edits; removals need
Raw 20 steps CFG 3 with a *grounded* empty negative; prefer ODE samplers (er_sde breaks outpaint
coherence); two distinct people in ONE pass (scene -> a, subject -> b).

## 3. Proposed design

### 3a. One shared attention core: split-key attention with LSE merge (`_split_attn.py`, new)
krea2edit implements ref_boost as a dense `L x L` bias, which forces SDPA off flash (slow, and the
dense-mask OOM we already hit with style at 3 MP). The Krea2-NAG reference runs a *second full*
attention per block (+~40% step time). Both, plus NegPiP, are expressible as one primitive:

    out = merge_g( attn(Q, K_g, V_g * vmul_g) , lse_g + log w_g )

i.e. attend separately to key groups (pos-text | neg-text | ref1 | ref2 | image), each returning
its log-sum-exp, and merge with per-group log-weights. Then:
- **ref_boost** = `log w_ref = log b` for target query rows.
- **NAG** = two merges sharing the expensive `image`(+refs) group: `[pos-text | image]` and
  `[neg-text | image]`. The image-key attention is computed **once**; only the 512-token text groups
  are extra -> expected overhead a few %, not +40%. (Valid because NAG's two passes share identical
  image Q/K/V - the property the Krea2-NAG author relies on.)
- **NegPiP** = `vmul` on the flagged text-key rows (negative -> flipped value).
- Compaction of padded text keys (the flash_varlen fix) falls out for free: groups are built from
  valid keys only.
Kernel: flash-attn varlen with `return_attn_probs` (softmax_lse) - same package the loader already
uses; fallback SDPA math path with explicit bias when unavailable. Installed as an extension of
`_compat.Krea2MaskSafeProcessor` only when edit/NAG/NegPiP are active; stock path untouched otherwise.
Correctness gate: unit test vs dense-bias SDPA and vs a naive two-pass NAG (bf16 noise level).

### 3b. Edit node - new `Eric Krea2 Edit` (KREA2_EDIT) + Ultra socket `edit` (appended)
Inputs (all new node, so order is free):
`krea2_pipeline`, `source_image`, `instruction` (multiline),
optional `source_image_b`, `fit_mode` [fit, crop (legacy)] = fit, `ref_boost` 1.0,
`ref_boost_a` 1.0, `ref_boost_mask` MASK, `grounding_px` 768, `system_prompt` "" (empty = training
default), `ground_negative` (bool, on: the CFG>1 negative is the empty instruction grounded on
the same image - the trained unconditional).
Outputs: `KREA2_EDIT`, `grounded_preview` (what the VLM saw), `fit_preview` (the fitted source
as placed on the target canvas - lets you see crop/margins before generating).
- The node does the grounded Qwen3-VL encode itself (reusing the Vision Prompt encoder, bare
  template, krea2edit system prompt, longest-side `grounding_px`); the bundle carries
  conditioning + source images. Ultra encodes the fitted source at the resolved S1 size
  (pixel path, before sampling - same as krea2edit's `target_latent` pre-encode).
- **Stage gating**: conditions S1 only (trained <=2 MP), S2/S3 refine stock - explicit stage flag,
  not token-count equality. Warn when S1 > 2.2 MP.
- **Exclusivity v1**: `edit` vs `style` vs `paint` vs `ref_latents` - one at a time (clear error).
- Warn when an SDE sampler / eta > 0 is used at S1 with an edit (krea2edit advisory).
- **Existing `RefLatents` `edit_frame`**: fix the stretch bug by routing it through the same
  fit geometry (it is a bug for any edit LoRA); keep it for other edit_frame LoRAs (zoom, Anything2Real
  in your Vision folder). Widget list unchanged.

### 3c. Negative guidance node - new `Eric Krea2 Negative Guidance` (KREA2_GUIDANCE) + Ultra socket `guidance` (appended)
Everything default **off**.
- **NAG**: `nag` (bool, off), `nag_negative` (text), `nag_phi` 4.0, `nag_tau` 2.5,
  `nag_alpha` 0.25, `nag_sigma_start` 1.0, `nag_sigma_end` 0.0, `nag_stages` [s1, s1_s2, all] = s1.
  Math: `z = z+ + phi (z+ - z-)`; L1-norm ratio clamp to tau; `alpha*z + (1-alpha)*z+`; norms in
  fp32; applied to the **raw attention output of live target queries only** (before gate/to_out),
  so text and ref tokens keep positive-path attention (the Krea2-NAG edit rule). The negative text
  stream is carried through the blocks (its own text_fusion + per-block text update) exactly as the
  reference does. Works at CFG off (Turbo) - the point of it; with CFG on it guides the positive
  pass only.
- **NegPiP**: `negpip` (bool, off), `negpip_mode` [lifted, in_place] = lifted,
  `negpip_strength` 1.0, `negpip_block_start` 0, `negpip_block_end` 27, `negpip_text_fusion`
  (bool, off - also flip inside the text-fusion attention; "stronger" path).
  Syntax in the Ultra prompt / Edit instruction: `(phrase:-1.0)`. `lifted` removes the phrase from
  the prompt, encodes it on its own and appends its rows (so Qwen never reads the word in the
  positive context); `in_place` flags the tokens where they stand. Value rows multiplied by the
  weight in the selected blocks. Positive weights are left alone (Krea2 has no weighting today).
  Token limit: lifted rows count toward 512; overflow -> warn + drop the phrase.
- **Exclusivity v1**: guidance works with plain t2i and edit; with `style`/`paint` it is skipped
  with a console note (their runtimes own the attention processors). Extendable later.

## 4. What I am NOT proposing
- No copied NegPiP code (AGPL vs your CC BY-NC dual license). krea2edit (Apache) and NAG (MIT)
  are ported with attribution in the module headers + README credits.
- No ref_boost/NAG at S2/S3 by default (untrained there); stage selectors exist for NAG only.

## 5. Test plan
1. **Geometry unit test (CPU)**: our fit vs krea2edit `_fit_encode_image` sizes/crop boxes/offsets
   over a grid of source ARs and S1 sizes, incl. the 754-vs-753 px doubling case. Must match exactly.
2. **Attention core**: split-LSE vs dense-bias SDPA (ref_boost 0.5/2/4/8), vs naive two-pass NAG,
   vs stock (all off): max-abs error at bf16 noise; speed at 2 MP: target <=5% overhead for
   NAG and ref_boost.
3. **Edits (needs the v1.2 LoRA)**, Turbo 10 steps CFG off, ~1.5-2 MP S1-only, fixed seeds:
   recolor, add object, restage/pose, outpaint (mismatched AR - the fit path), two-ref person into
   scene, ref_boost 1/2/4/8 strip; removal: Raw CFG 3 (reference recipe) vs **Turbo + NAG**
   (the interesting experiment - can NAG make removals work at CFG off?).
   Cross-check a few cases against krea2edit in a native ComfyUI workflow (visual, not bitwise:
   different noise RNG / stack).
4. **NAG / NegPiP on plain t2i**: suppression cases (the "llama-bird, negative: big wings" style
   example; "(blurry:-1)"; a color you don't want), phi 2/4/6, lifted vs in_place.
5. **Multistage**: edit at S1 then S2/S3 refine - check identity holds (paint showed S2/S3 can drift).

## 6. Decisions for Eric
1. **Download the Identity Edit v1.2 LoRA?** (Krea 2 Community License; full or r128 0.91 GB /
   r64 0.46 GB "near-identical"). Recommended: full v1.2 + r128 to `L:\Models\loras\Krea2\edit\`.
2. **New Edit node + `edit` socket on Ultra** (3b), with the old RefLatents `edit_frame` kept but
   geometry-fixed - OK? (Alternative: extend the RefLatents node with appended widgets - I
   recommend the new node: the RefLatents widget set is ostris-shaped.)
3. **One combined Negative Guidance node** (NAG + NegPiP, both off) on a `guidance` socket - or
   two separate nodes?
4. **NAG negative text**: own field in the guidance node (recommended - Ultra's `negative_prompt`
   stays the CFG negative), or reuse Ultra's `negative_prompt` when CFG is off?
5. **NegPiP default mode**: ship both `lifted` / `in_place` and pick the default from test 4?
   (recommended)

## 7. Eric's decisions (2026-10-08)
1. Download Identity Edit v1.2 - yes: full + r128 in L:\Models\loras\Krea2\edit\2026-identity-edit-v1_2\ (SHA-256 verified).
2. New Edit node + dit socket - yes; RefLatents edit_frame kept, geometry fixed.
3. One combined Negative Guidance node; NAG and NegPiP usable together - yes.
4. NAG negative text in its own field - yes.
5. Ship both NegPiP modes, pick the default from tests - yes.

## 8. As built (2026-10-08)
- _edit_geom.py - fit / crop geometry; parity test docs/tools/test_edit_geom.py: 168/168 cases byte-identical
  to krea2edit v1.2.5 (pixels + RoPE offsets).
- _split_attn.py - per-group flash-attn (softmax_lse) + log-sum-exp merge; fp32 math fallback.
  Contiguous key segments with the same weight are attended in ONE call, so plain/NegPiP-only calls
  are a single kernel; splits only for ref_boost and NAG's text groups.
- _edit_guidance.py - EditGuidanceRuntime: custom forward over the model's own modules (LoRA /
  native-quant / control hooks all apply) with sources, ref_boost (+mask), NAG, NegPiP; stock
  forward when nothing is active. docs/tools/test_edit_guidance_math.py: vs independent dense
  references (stock, dense-bias ref_boost, V-scaled NegPiP, naive two-sequence NAG) - all ~1e-7 fp32.
- _edit_text.py - grounded encode at NATURAL length (trainer recipe; 512 padding would overflow with
  two 768 px references), LANCZOS grounding cap; NegPiP parse (lifted / in_place); plain encode.
- _edit_install.py - Ultra glue: encodes once before sampling, sizes S1 to the source aspect
  (size_from=source), disables crop_bottom for edits, refine prompt for S2/S3 = Ultra prompt text
  (or the grounded instruction when empty), TE offload on <64 GB cards when guidance is off.
- Ultra: dit + guidance sockets appended; CFG pass marked on the transformer; S1 euler runs
  through our solver when edit/guidance is active; exclusivity (edit vs ref_latents/style/paint).
- RefLatents dit_frame: v1.2 fit geometry + explicit target-grid gating (was stretch + token count).
- Note: cond_rebalance reaches the S1 positive (grounded / NegPiP) conditioning; the S2/S3 refine
  conditioning and the NAG negative are not rebalanced.