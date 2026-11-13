# What's New - full history

[<- Back to the README](../README.md) | [All docs](../README.md#documentation)

## What's New

**Late October 2026 - LoRA picks by creator, one pick per prompt, random word options.**

- **`creator=`** in Sweep Plan catalog picks (`random: catalog creator=Adel_Ai x4`), alias `author=`;
  combines with `tag=`, `kind=`, `role=` and `arch=` (default Krea2, `arch=any` = every base). The
  creator is now part of the searchable card text too, as in the catalog's search box.
- **`per prompt`** - each prompt line gets its own pick, paired instead of crossed: bakeoff
  `random: catalog creator=Adel_Ai per prompt` = one LoRA per prompt; stacks
  `random: catalog creator=Adel_Ai x2 per prompt @0.8` = one 2-LoRA stack per prompt.
- **`#random`** word option in sweeps (`name #random`), **`3: random`** in the LoRA Stack's
  `word_options` (a new option every queue), and an amber dot on stack slots whose LoRA has word
  options but none chosen. See [LoRA picks in the Sweep Plan](SWEEPS_REFERENCE.md#random-lora-picks).

**Late October 2026 - SDE band, rectified-flow churn, trajectory probe.**

- **New node: Eric Krea2 Hybrid Sigma Handoff** (2026-10-23) - wire its `sampling` output into
  Ultra's optional `sampling` input. It holds the hybrid handoff, SDE band and churn form, so the
  Ultra panel stays small; unwired, Ultra behaves exactly as before. **Hybrid handoff by sigma:**
  `sN_hybrid_handoff` switches lcm_hybrid / lcm_hybrid2 from lcm to the tail at a sigma instead of
  a step count (first step at or below it), so one value works on every schedule and step count;
  0.70 reproduces 7 of 12 on linear. 0 = use `sN_hybrid_steps`.
- **SDE band per stage** - `s1/s2/s3_sde_hi` and `_sde_lo` (on the Hybrid Sigma Handoff node) limit eta's
  churn to a sigma range, e.g. noise only in the composition band (1.0-0.5) or only in the detail
  band (0.5-0.15). Defaults (1 / 0) = previous behaviour. The console says how many steps churned.
- **`churn_form` ve | rf** - `rf` injects eta's noise in the rectified-flow form; the classic `ve`
  form over-weights the image signal on flow models (measured +1.4% at eta 0.3, +19.8% at eta 1.0
  on one mid-sigma step). Default `rf` since 2026-10-25, with or without the node (ve burnt all 8 samplers in the screen at eta 0.8); set `ve` on the node to reproduce older eta > 0 images.
- **Trajectory probe** - every stage now prints how fast the model's predicted image changes and a
  free per-step error estimate; sweeps get `probe_*` columns in the CSV and review workbook.
  Diagnostic only, never changes the image.
  See [SDE band, churn form & the trajectory probe](SAMPLERS.md#sde-band-churn-form--the-trajectory-probe).

**Mid-October 2026 - sweeps that explain themselves, random LoRA picks, prompt LoRA tags.**

- **Sweep folders explain themselves** - the Sweep Plan has a new `run_notes` field (last
  widget: why you are running this sweep today). Every sweep folder now gets
  `sweep_info.txt` (preset name, preset notes, run notes, checkpoint, what varied) and
  `sweep_review.xlsx`, a review workbook with thumbnails, what varied, empty columns for your
  own ratings, and the UniPercept scores and metrics. Works for any sweep (LoRA, sampler,
  model...), and is written even when you cancel. See [Parameter sweeps](SWEEPS_REFERENCE.md#parameter-sweeps).
- **Random LoRA picks in the Sweep Plan** - `random: graphic-art x6` in `lora_files` picks 6
  LoRAs from a folder; in stacks mode a slot can be random
  (`random: graphic-art x4 @0.8 | V0id @0.6`). A fresh draw every queue by default, `seed=7` for a
  repeatable one. See [Random LoRA picks](SWEEPS_REFERENCE.md#random-lora-picks).
- **Prompt LoRA Tags** - new **Eric Krea2 Prompt LoRA Tags** node: write
  `<lora:Shouji_Ink_KR2:0.66>` in a prompt; the node removes the tag and feeds the LoRA to the
  Multi-LoRA Stack's `suggestions` socket. See [Prompt LoRA tags](LORAS.md#prompt-lora-tags).
- **LoRA names forgive Civitai download names** - a bare name like `Shouji_Ink_KR2` now also
  finds `3317450_shouji_ink_kr2_bf16.safetensors` (version-id prefix and `_bf16` / `_fp16` /
  `_fp8` suffix ignored), as long as exactly one file matches. Used by the Prompt LoRA Tags
  node and the random picks.
- **Sweep names in folders and files** - a sweep folder is now
  `<date_time>_<sweep name>_<checkpoint>` and its files are `sweep_manifest_<name>.json`,
  `sweep_summary_<name>.csv`, `sweep_review_<name>.xlsx`, `sweep_info_<name>.txt`. The name is
  the preset, `custom`, or `<preset>-edited` when you changed a loaded preset - `sweep_info`
  then says which preset and which fields. Older folders keep their old names and still work
  everywhere.
- **Every cell's LoRAs in the CSV and review workbook** - a `LoRAs used` column with each
  cell's actual stack and weights, whatever was swept (rebuilding an older sweep's workbook
  reads it from the cell PNGs).
- **Stack screens: `cross:`, stack_parts, Start 12 / 13** (2026-10-16) - a stack line starting with `cross:` makes
  every combination of its slots' picks (`cross: random: illustration x4 | random: graphic-art x4 | random: treatment
  x4` = 64 stacks). The new Sweep Plan field `stack_parts` (appended last) picks what goes next to each stack:
  each LoRA alone (as before), reference only, build up (A, A+B, A+B+C) or leave one out. Start 12 screens 64
  trios; Start 13 builds up the winners, which the LoRA Catalog's Sweeps page lists for you (Follow-up button).
  See [docs/SWEEPS.md](SWEEPS.md#picking-loras-for-you).
- **Share sheets for posting** (2026-10-16) - a finished sweep also writes ready-to-post JPEGs into
  `<sweep>/share/`: one labelled strip per prompt (prompt and shared settings on top; each image's LoRA short
  name, weight, creator and Civitai model number underneath; 768 px images, wrapping at 6000 px) and/or every
  image labelled on its own. New Sweep Plan fields `share_sheets` / `share_background` (appended last). The new
  **Eric Krea2 Sweep Sheets** node, and the LoRA Catalog's Sweeps page (with your star ratings and notes), make
  them again for any sweep. See [docs/SWEEPS.md](SWEEPS.md#share-sheets-for-civitai-reddit-).
- **Sweep guide, Show plan, Start presets** (2026-10-16) - [docs/SWEEPS.md](SWEEPS.md) is a
  practical guide to the Sweep Plan with examples. *Show plan* (under the node) lists every cell before you
  queue: its LoRAs and weights, its settings and the exact words added to its prompt. Eleven *Start*
  presets cover the common questions (strength ladder, folder bake-off, pairs, samplers, steps, seeds...).
- **Trigger words: four plain choices** (2026-10-16) - on Apply LoRA, Multi-LoRA Stack and Sweep Plan:
  `none`, `trigger` (the card's main trigger), `full` (official words + trigger + style phrase) and
  `saved pick` (the words picked for the loaded checkpoint, else trigger + style phrase; the Sweep Plan's
  new default). Older workflows' off / catalog / recipe load as none / trigger / saved pick.
- **Paste LoRA button on the Multi-LoRA Stack** (2026-10-16) - copy a LoRA in the LoRA Catalog
  (*copy name*, *copy as prompt tag*, or the path) and press *Paste LoRA from clipboard*: it goes into
  the next empty slot at its card weight (a tag's weight wins). Reminder: clicking the middle of a
  slot's dropdown opens a search over names, triggers and catalog cards (Shift+click = plain list).
- **Sweep Plan: Clear button, empty sampler lists** (2026-10-16) - a *Clear sweep* button at the
  bottom of the node puts every field back to its default (keeps `output_folder` and `enabled`,
  empties the notes; asks first). `samplers` and `schedulers` now start empty on a new node; the
  preset *All samplers + schedulers (lists only)* fills both with every option and touches nothing
  else. Nodes already in your workflows keep what they have.
- **`all:` and catalog picks in `lora_files`** (2026-10-16) - `all: illustration` = every LoRA in
  the folder; `random: catalog watercolor x4` and `all: catalog tag="art nouveau"` pick from the
  LoRA Catalog by words and filters. Names typed by hand now resolve like Civitai downloads
  (version prefix and `_bf16` may be left off). See [Random LoRA picks](SWEEPS_REFERENCE.md#random-lora-picks).
- **Prompt words merged for a stack** (2026-10-16) - with `recipe` triggers, the words of all the
  LoRAs in a stack are merged once: repeats dropped, picks kept whole, a word budget shared by
  weight, opposite words ("muted" vs "vibrant") reported. See
  [LoRA trigger words](LORAS.md#lora-trigger-words-base-model-vs-fine-tunes-measured-oct-2026).
- **`<lora:...>` tags in sweep prompts load** - in a sweep, tags in a cell's prompt are taken out
  of the text and loaded for that cell (a tag for a LoRA already in your stack sets its weight).
  Before this they were encoded as plain words and nothing loaded.
- **No more over-cap surprises** - `max_combos` now defaults to 0 = no cap, and the Sweep Plan
  shows the cell count under the node as you edit (counted by the same code the queue runs).
  Nodes already in your workflows keep their saved cap (64): set it to 0 there.
- **`recipe` trigger mode** - on Apply LoRA, Multi-LoRA Stack and the Sweep Plan: each LoRA gets
  the prompt words chosen in the LoRA Catalog for the loaded checkpoint (your pick, else a
  trigger test's pick, else trigger + style phrase). See
  [LoRA trigger words](LORAS.md#lora-trigger-words-base-model-vs-fine-tunes-measured-oct-2026).

**October 2026 - image-to-image suite, negative guidance, native quantized compute.**

- **Control (ControlNet-LoRA)** - new **Eric Krea2 Control** node + Ultra `control` socket:
  depth / pose / canny ... structure control via Krea 2 Control-LoRA checkpoints (expanded
  input layer + block LoRA), per-stage strengths. See [Control](IMAGE_TO_IMAGE.md#control-controlnet-lora).
- **Style Reference (training-free)** - new **Eric Krea2 Style Reference** node + Ultra
  `style` socket: transfers palette / brushwork / texture from one image with no LoRA.
  See [Style Reference](IMAGE_TO_IMAGE.md#style-reference-training-free).
- **Inpaint / outpaint** - new **Eric Krea2 Paint** + **Paint Composite** nodes + Ultra
  `paint` socket: AnyPaint / Outpaint adapters, kept regions pinned to the source at every
  stage. See [Inpaint / outpaint](IMAGE_TO_IMAGE.md#inpaint--outpaint-paint).
- **Edits** - new **Eric Krea2 Edit** node + Ultra `edit` socket: the Krea 2 Identity Edit
  recipe (recolor, restyle, re-pose, add / replace objects, outpaint, person-into-scene),
  with the v1.2 `fit` reference geometry and the `ref_boost` fidelity dial.
  See [Edits](IMAGE_TO_IMAGE.md#edits-krea-2-identity-edit).
- **Negative guidance** - new **Eric Krea2 Negative Guidance** node + Ultra `guidance`
  socket: NAG and NegPiP, both off by default, combinable, work at guidance 0.
  See [Negative guidance](NEGATIVE_GUIDANCE.md#negative-guidance-nag--negpip).
- **Native quantized compute** - Component Loader `compute` / `native_format` (appended):
  run quantized checkpoints on their native kernels. See
  [Native quantized compute](MODELS_AND_QUANT.md#native-quantized-compute).
- **More checkpoint formats** - the Component Loader now loads MXFP8, NVFP4, ConvRot-W4A4,
  W4A8 and W6A8 single files (ComfyUI `comfy_kitchen` layouts) next to fp8, INT8 and
  INT8-ConvRot. See [Quantized checkpoints](MODELS_AND_QUANT.md#quantized-checkpoints-the-component-loader-reads).
- **DeGrid** - removes the VAE's 2-pixel lattice: new **Eric Krea2 DeGrid** node, a
  `degrid` toggle on both decode nodes, and `degrid` on Ultra (inter-stage VAE hops and/or
  the final image). See [DeGrid](MODELS_AND_QUANT.md#degrid-vae-2-pixel-lattice).
- **LoRA search** - click the middle of any LoRA dropdown (Multi-LoRA Stack, Apply LoRA,
  Diagnose LoRA) for a search popup over filenames, folders, trigger words and LoRA
  Catalog cards; Multi-LoRA Stack also gained a `suggestions` socket for the LoRA
  Suggester. See [LoRA search](LORAS.md#lora-search--suggestions).
- **Fixes** - the `flash_varlen` bottom-band bug (all Krea 2 attention is now mask-safe;
  existing seeds render slightly differently, edge to edge correct); VRAM pinned by LoRA
  nodes and by the loader on model switches; the Reference Latents `edit_frame` recipe used
  a stretch instead of the trained fit geometry. See [Notes](../README.md#notes).

**Late July 2026 (cont.) - field-calibrated scoring, preset UX, er_sde.**

- **Field-calibrated metrics & captions** - clip% flags at absolute 4%/6%
  thresholds, Tenengrad is the displayed/gated sharpness, splotch and noise
  are demoted to CSV-only, contact-sheet captions use stage-grouped
  shorthand with a compact wrapped metric line, and Auto Pick gained the
  `most_typical` objective (consensus render) plus a uniqueness
  consensus-outlier gate. Calibrated from two full 192-cell runs.
- **Folder scoring chain** - the new `Eric Sweep Folder Scorer` (in
  Eric-image-classification) grades a whole sweep with Blur V7 / Noise V4 /
  composition / exposure / UniPercept and writes a merged `scores.csv` that
  Auto Pick reads via `best_external`: sweep -> unload -> score -> pick ->
  promote in one queue.
- **Sweep presets that show themselves** - the Sweep Plan node joined the ★
  preset system: selecting a preset writes its values into the panel,
  editing flips back to `custom`, and a `preset_notes` field (also on the
  Sigmas node) keeps a saved sentence about what each preset is for.
- **`er_sde` sampler + `simple` scheduler** - a ComfyUI-faithful ER-SDE
  Solver-3 port (verified step-identical): intrinsically stochastic, and
  the only 1-eval/step sampler that is schedule-proof (amplification 0.048
  on EVERY schedule incl. linear_quadratic). eta is ignored; the noise TYPE
  shapes its churn (colored-noise er_sde is unique to this package).
  `simple` = uniform-in-sigma, an alias of `linear`.
- **Swept-schedule precedence** - swept schedule cells now disable a wired
  sigmas bundle for that stage, cell-locally: a swept axis is never
  silently inert.

**Late July 2026 - parameter sweeps, hybrid samplers, inter-stage renorm.**

- **Parameter sweep system** - wire an `Eric Krea2 Sweep Plan` into Ultra V2's
  new `sweep` input and one queue item runs a whole grid (or explicit combo
  list) of sampler / scheduler / steps / window / eta / noise / mu / cfg /
  megapixels settings against a fixed seed: per-cell PNGs with full embedded
  recipes, a labeled contact sheet, `sweep_summary.csv`, a JSON manifest,
  automatic S1-latent reuse for s2/s3-only cells, and one-click promotion of a
  winning cell to a preset. See [Parameter sweeps](SWEEPS_REFERENCE.md#parameter-sweeps).
- **`lcm_hybrid` / `lcm_hybrid2` samplers + `hybrid_split`** - lcm full-churn
  entry (launders upscale damage) handing off to a trajectory-faithful tail
  (`deis_3m` / `res_2m`) at a selectable fraction of the window.
- **`upscale_renorm`** - repairs the latent variance each upscale hop drains
  (variance is contrast/saturation), stopping the progressive flattening of
  multi-hop runs. `per_channel` recommended.
- **Sweep v2: LoRA, prompt & sigma-profile axes** - bake a whole style-LoRA
  library off against the bare checkpoint (or layered on your stack), ladder
  one stack entry through strengths, cross everything with a prompt list and
  with saved sigma-shape presets - all in one queue item, all standalone-
  reproducible per cell.

**July 2026 - multi-stage sampling, presets & img2img toolkit.**

- **Custom flow-matching samplers** - `res_2m`, `res_2s`, `deis_3m`, `deis_4m`
  re-implemented from the RES/DEIS papers (no external solver import), selectable
  per stage. Recommended: `res_2s` + `beta57` for the Stage-1 draft, `deis_3m` /
  `deis_4m` + `bong_tangent` for the refine passes.
- **Shared schedule library + Sigmas node** - `linear`, `balanced`, `karras`,
  `beta57`, `beta`, `bong_tangent`, `exponential`, with a per-stage `detail_bias`
  control driven from a dedicated **Sigmas** node.
- **img2img** - a **VAE Encode** node plus an `init_latent` input on the
  Multi-Stage node; denoise strength is set by the re-noise step window.
- **Vision Prompt** - image-grounded conditioning via Krea2's own Qwen3-VL vision
  path (no training, no VAE/reference-latent pathway needed). Auto-shares its
  megapixel budget across however many reference images are connected. See
  [Vision Prompt](IMAGE_TO_IMAGE.md#vision-prompt-image-grounded-conditioning).
- **Resolution & latent helpers** - a **Resolution** node (aspect+MP or
  image-derived size → `width`/`height`/blank latent) and a latent-space
  **Latent Resize** (bislerp, no VAE round-trip).
- **Presets & settings** - named JSON presets per section (loader / component loader /
  Apply LoRA / LoRA stack / Ultra / sigmas) with a ★ Save button, a **Merge Settings**
  node, and **Settings from Image** for one-click reproduction.
- **LoRA tooling** - apply / unload / diagnose plus a **Multi-LoRA Stack**, both with
  **per-stage weights** (S1/S2/S3) and named-preset save/load (★ Save).
- **Reference Latents (edit LoRAs)** - a **Reference Latents** node VAE-encodes 1-3
  reference images into the ai-toolkit "index_timestep_zero" edit pathway, fed via a new
  optional `ref_latents` input on the Multi-Stage Ultra (res) node. Works *only* with an
  edit-trained LoRA loaded (e.g. the Krea 2 Style Reference LoRA on Turbo at guidance 0);
  the base model can't read reference tokens. See
  [Reference Latents](IMAGE_TO_IMAGE.md#reference-latents-edit-lora-image-conditioning).
- **Upscale VAE quality-pass modes** - the trained spacepxl Wan 2× decoder can drive
  the S2→S3 jump and/or the final decode, each with a **downsample** variant that keeps
  its sharp detail but resamples down to your chosen size instead of forcing 4× area.
  The `… with downsample` modes are usually **cleaner/sharper than the Wan 2.1 VAE** at
  the same output size and avoid the resolution blow-up. An on-node **resolution-chain
  readout** shows the real S1→S2→S3→decode sizes and flags any field the VAE overrides.
  See [Upscale VAE: modes & precedence](HIGH_RES.md#upscale-vae-modes--precedence).

*Initial release (June 2026):* Krea 2 Raw + Turbo support, multi-stage high-res,
2× upscale-decode, selectable attention backend, natural-language prompt builder +
LLM Magic-Prompt expander.
