# Eric_Krea2 research — refusal LoRA vs cond_rebalance, prompt structure, vision/edit gleanings
_2026-07-13 · sources: Krea2_TextFusion_Refusal_Reduction.safetensors (inspected on disk), Civitai 2775340 & 2764349 (API), ethanfel/ComfyUI-Krea2TextEncoder and ostris/ComfyUI-Krea2-Ostris-Edit (cloned, full source read), HF krea/Krea-2-Turbo discussion #4, Civitai Fedor Filter Bypass writeup_

## 1. Krea2 TextFusion Refusal-Reduction LoRA vs our cond_rebalance

### What we have now
`_rebalance_krea_embeds()` in `krea2_multistage_ultra.py`: static per-tap gains (12 values)
x global multiplier applied to the `(B, seq, 12, dim)` conditioning tensor BEFORE the DiT.
Mathematically this is a diagonal rescale of text_fusion's INPUT — the same lever the
"knob-overwrite" bypass LoRAs pull (Fedor Filter Bypass family: they overwrite values in
the 1x12 tap weighting; community ablation attributes refusal mainly to taps 9/10, with
11 as a secondary refusal that also affects how naturally humans render; taps 1-8 and 12
carry style/anatomy priors — 1-based indexing per that writeup). Our presets boost taps
labeled 8/9/11. Amplitude-based: it out-shouts the dampening, and amplifies EVERYTHING in
a boosted tap, refusal features included. Known side effect: oversaturation / "plastic"
look at high gain (same artifact reported for the all-12-knob bypass files).

### What the LoRA actually is (file inspected at L:\Models\loras\Krea2\tools\)
- Rank-64 trained LoRA, ai-toolkit 0.10.20, step 7900 / epoch 83, BF16, 64 tensors, ~26 MB.
- Targets ONLY `txtfusion.layerwise_blocks.{0,1}` and `txtfusion.refiner_blocks.{0,1}`:
  attn wq/wk/wv/wo/gate + mlp gate/up/down in each (8 linears x 4 blocks x A/B = 64).
- Metadata: "Retained txtfusion layerwise_blocks 0/1 and refiner_blocks 0/1; removed
  txtfusion.projector and external txtmlp" — filtered down from a broader Targeted_v3.
- Author's description (Civitai 2775340): trained with the single objective of reducing
  refusal while preserving visual knowledge; deliberately NO adapter on the 1x12 tap
  projector, txtmlp, or any image-transformer block; route isolated by layer-by-layer
  ablation "instead of broadly amplifying activations or directly altering the projector"
  — i.e. explicitly positioned against BOTH the rebalance approach and the
  projector-overwrite bypasses. Recommended strength 1.0.

### How it differs (the mechanism gap)
| | cond_rebalance (ours) | Refusal-Reduction LoRA |
|---|---|---|
| Where | conditioning embeds, pre-DiT | text_fusion attention/MLP weights |
| What | 12 scalars + global gain | learned rank-64 residuals, 4 blocks x 8 linears |
| Selectivity | per-tap amplitude only | direction-selective within taps (can suppress the refusal feature while leaving co-resident content features alone) |
| Side effects | boosts everything in a tap; saturation at high gain | claims minimal; leaves tap scales + projector untouched |
| Composability | runtime, preset-driven | stacks with rebalance; both can be active |

Bottom line: not a better version of the same trick — an orthogonal, LEARNED one. The most
interesting experiment: whether it lets us BACK OFF cond_multiplier. If the refusal route
is neutralized in weight space, the amplitude boost no longer has to carry that job —
one suspected contributor to the saturation/color complaints.

### Loader compatibility — CONFIRMED, zero code needed
Keys are Comfy layout `diffusion_model.txtfusion.layerwise_blocks.N.attn.gate.lora_A.weight`.
`_lora_utils._map_krea_module()` already matches `txtfusion_(layerwise|refiner)_blocks_N_<leaf>`
and all 8 leaves map via `_KREA_LEAF` (gate->to_gate, mlp.down->ff.down, etc.). Loads through
Apply LoRA / Multi-LoRA stack today, per-stage strengths included. No alpha keys in the file
(PEFT alpha=r convention) — our fold-into-lora_B default is correct for it.

### Suggested A/B protocol
Fixed seed, cond_jitter off, 4 arms: (a) enhancer preset alone, (b) LoRA@1.0 + flat,
(c) LoRA@1.0 + gentle, (d) LoRA@1.0 + enhancer. Judge adherence AND saturation/skin.
Optional 5th arm: LoRA per-stage 1.0/0.5/0 — composition is decided in stage 1, where
refusal most likely acts, so stage-1-only may suffice.

## 2. Prompt structure — Magic Prompt & Structured Prompt vs the suggested outline

### Provenance of the outline
The `[Subject+Pose] -> ... -> [Aesthetic & Medium]` ordering and the "one or two cohesive,
flowing paragraphs" guidance come from the community system prompt on HF
(krea/Krea-2-Turbo discussion #4) — an image->prompt recreation prompt. It also stresses:
flowing prose, start directly with the subject (no "In this image..."), and explicitly
avoid mechanical keyword lists / comma-stuffed style.

### Do our nodes follow it? Mostly no.
- **EricKrea2Prompt** comma-joins fields in order: subject, style_medium, scene,
  composition, camera, lighting, color, mood, extra. Deviations: (1) it IS a comma list,
  which the guidance argues against; (2) style/medium sits 2nd instead of LAST; (3)
  environment sits 3rd instead of after composition; (4) no pose/action,
  appearance/clothing, or props/materials fields.
- **EricKrea2MagicPrompt** default system prompt asks for "a single paragraph" and lists
  detail categories in arbitrary order — no sequencing, no two-paragraph option.

### Proposed changes (pending go-ahead)
1. Rewrite `_DEFAULT_EXPANDER_SYSTEM` to encode the ordering + the two-paragraph split
   (P1: each subject in turn — who -> pose/action -> clothing/props — then environment
   woven in; P2: composition/framing, lighting, color palette/mood, aesthetic/medium,
   optional technical). Keep the preserve-intent and quote-text-to-render clauses.
2. EricKrea2Prompt: add optional `pose_action`, `appearance_clothing`, `props_materials`
   fields; reorder the join to subject -> pose -> appearance -> props -> composition ->
   camera -> scene -> lighting -> color -> mood -> style_medium (LAST) -> extra; emit two
   sentence-groups ("Paragraph 1. Paragraph 2.") instead of one comma chain. Additive +
   reorder only — no input removed, old workflows still load (output text changes).

### Token budget (512) — what's mechanically true
- Krea2's text budget is fixed: `max_sequence_length` (512) Qwen3-VL-tokenizer tokens
  minus template prefix/suffix; the pipeline pads in the middle and TRUNCATES the rest.
  Exceeding budget silently loses the tail — that's the real cliff.
- Below budget there is no mechanical penalty for length: padding is attention-masked.
  "Shorter adheres better" is anecdotal; Krea's own guidance says long, specific prompts
  give best fidelity, while very short prompts trade adherence for variety
  (aesthetic-first behavior). The long-prompt failure mode is concept DILUTION (too many
  competing subjects), not tokenization.
- Gotcha: Magic Prompt's `max_tokens=512` counts the EXPANDER LLM's tokens, not Qwen3-VL
  tokens. Worth adding a cheap length check (chars/~3.5 heuristic) that warns in `status`
  when the expansion likely exceeds the encode budget.

## 3. Gleanings from ethanfel + ostris for Vision Prompt / edit ambitions

### ethanfel/ComfyUI-Krea2TextEncoder (MIT — safe to adapt with attribution)
Features it has that EricKrea2VisionPrompt lacks:
1. **Per-image masks with bbox crop + `mask_padding`** — the VLM sees only the masked
   region (box grown by a padding fraction per side; mask computes the crop only, is not
   sent to the VLM). Cheap to add before `_prep_pil` (pure tensor crop). High leverage
   for "take THE JACKET from this photo" workflows.
2. **Connectable `system_prompt` override + shipped instruct-style system prompt** that
   tells the VLM to describe the reference, then explain how the user's instruction
   should combine with / alter it — the trick that makes prompt and reference INTERACT
   (edit-like) instead of sitting side by side. For us: make the descriptor system text
   an optional override instead of always using `pipe.prompt_template_encode_prefix`
   verbatim (swap only the system message inside the prefix, keep the chat scaffolding
   and the pad-in-the-middle tokenization). Out-of-distribution — A/B vs the descriptor.
3. Unbounded auto-growing image+mask slots via a JS extension (nice-to-have; we have 3).
4. Their default is 1.0 MP/image with NO cross-image budget sharing — our 0.40 MP
   auto-share is more correct for the fixed 512 budget; keep ours.

### ostris/ComfyUI-Krea2-Ostris-Edit (no LICENSE file — reimplement, don't copy code)
This is the real style-transfer/img2img mechanism, and it is NOT a text-encoder trick:
- Text encode: VLM refs downscaled to **384x384 total px** (coarse semantics only), refs
  also **VAE-encoded to <=1 MP** (snapped /16) and attached as `reference_latents`.
- Model patch: patchifies ref latents into tokens, appends them to the image token
  sequence with RoPE axis-0 index i+1 (own y/x grid from 0), modulates the ref span with
  the **t=0** timestep vector while the live span uses real t ("index_timestep_zero"),
  and slices refs out of the prediction. Optional `kv_cache` mode precomputes ref K/V in
  one t=0 pass (valid only for LoRAs trained with ai-toolkit's kv_cache
  isolate-ref-attention kwarg).
- **Requires an edit-trained LoRA** (ai-toolkit `krea2` + `model_kwargs.edit: true`);
  the base model was never trained to read those tokens. The Civitai "Krea 2 Style
  Reference LoRA" (2764349) is exactly such a LoRA: trained with the Turbo training
  adapter on thousands of curated style pairs, handles 1-2 refs, no trigger word.
- Community reality check (Detail Enhancer edit-LoRA page): highly experimental, alters
  the image, occasional horizontal-AR issues, slight lighting/color shifts.

### What this means for Eric_Krea2
Our VisionPrompt = semantic grounding through the vision path (no pixels survive the VLM
bottleneck). Ref-latents = pixel/structure detail conditioned at t=0. Complementary — and
the edit LoRAs NEED the latter. To run the Style Reference LoRA in OUR diffusers pipeline:
- New node (e.g. `EricKrea2RefLatents`): VAE-encode 1-2 refs (<=1 MP, /16 snap), output a
  KREA2_REF_LATENTS bundle.
- Ultra node: optional `ref_latents` input; wrap the transformer forward to pack refs,
  extend RoPE ids (axis-0 = i+1), per-span modulation (real t vs t=0), slice output.
  Same surgery ostris does on comfy's SingleStreamDiT, mapped to the diffusers module
  names we already know from `_lora_utils` (`text_fusion`, `transformer_blocks`,
  `time_embed`, `time_mod_proj`, `img_in`, `final_layer`).
- VLM sizing when an edit LoRA is loaded: match training (384x384 total px ~= 0.147 MP) —
  add a "match ostris edit training" preset/toggle on vision_megapixels.
- Skip kv_cache in v1 (optimization for one specific LoRA training mode).

Effort: the forward-wrap with per-span modulation is the only genuinely tricky part;
the rest is plumbing we've done before.
