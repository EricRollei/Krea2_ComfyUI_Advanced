# Sweeps - a practical guide

A sweep renders many versions of one image in a single queue item - different LoRAs, strengths, samplers, step
counts, seeds, prompts - and lays them out so you can compare them. The model and LoRAs stay loaded the whole time,
so it is much faster than queueing the same runs by hand.

This guide covers the **Eric Krea2 Sweep Plan** node. For the full reference of every field, hover over it in
ComfyUI; the README's [Parameter sweeps](SWEEPS_REFERENCE.md#parameter-sweeps) section has the technical details.

---

## 1. Wiring

```
Component Loader -> (LoRA stack, optional) -> Multi-Stage Ultra V2 <- Sweep Plan
                                                       |
                                              sweep_sheet / sweep_manifest
```

- Connect the Sweep Plan's `sweep` output to Ultra V2's `sweep` input.
- **Ultra V2's panel is the base recipe.** Every cell starts from it and changes only what the sweep changes.
  Set up the panel first (sampler, steps, size, prompt) and make one normal render you like.
- Turn the sweep off with the Sweep Plan's `enabled` switch to go back to single renders without unwiring it.

## 2. The three buttons under the node

| | |
|---|---|
| **cell counter** | How many images the panel will make, updated as you type. Check it before you queue. |
| **Show plan** | A window listing every cell in order: its LoRAs and weights, its settings, **the exact words added to its prompt**, and the start of its prompt. Filter box, copy-as-text. Use it every time you sweep LoRAs. |
| **Clear sweep** | Puts every field back to its default (keeps `output_folder` and `enabled`, empties the notes). Start here when a sweep from last week is still on the panel. |

## 3. Start from a preset

Pick one from `sweep_preset`; its values are written into the panel so you can see and edit them. The **Start**
presets are made to be copied and adapted:

| Preset | What it answers | Cells |
|---|---|---|
| Start 01 - one LoRA, strength ladder | How strong should this LoRA be? Does it work on other subjects? | 15 |
| Start 02 - bake-off, every LoRA in a folder (quick) | Which LoRAs in this folder deserve a closer look? | 1 + folder |
| Start 03 - bake-off, 8 random LoRAs from the catalog | A quick look at LoRAs matching some words | 9 |
| Start 04 - one LoRA paired with each in a folder | Which partners suit this LoRA? | ~2 x folder |
| Start 05 - hand-written stacks | Do these specific combinations work? | 12 |
| Start 06 - sampler x scheduler shoot-out (quick) | Which sampler / schedule suits this checkpoint? | 12 |
| Start 07 - step count | Where do more steps stop helping? | 15 |
| Start 08 - seed variety | How much does this recipe vary from seed to seed? | 8 |
| Start 09 - image size (megapixels) | How does the image change with size? | 8 |
| Start 10 - LoRAs x prompts (do the words work?) | Do these LoRAs show with their words? | 12 |
| Start 11 - stage 2 refine window | How late should Stage 2 start? (2-stage panel) | 4 |
| Start 12 - stack screen, every combination (quick) | Which 3-LoRA stacks are worth a closer look? | 65 |
| Start 13 - build up the winning stacks | What does each layer of a good stack add? | 4 per stack |
| All samplers + schedulers (lists only) | Fills only the two lists with every option | - |

Change a field and the dropdown shows `custom`; the sweep folder still records which preset it was based on.
Save your own with the ★ button and write a line in `preset_notes` saying what it is for.

## 4. What can be swept

Each filled field is an **axis**; the sweep makes **every combination** of the axes (a grid). Blank = not swept
(the panel value is used).

| Field | Example | Notes |
|---|---|---|
| `samplers` | `lcm_hybrid2, euler, er_sde` | |
| `schedulers` | `bong_tangent, simple` | also saved sigma-shape preset names |
| `pair_lock` | on | pairs samplers and schedulers line by line instead of crossing them |
| `steps` | `8, 12, 16` | |
| `windows` | `6-12, 8-12` | where the scoped stage starts and ends |
| `etas`, `noise_types`, `cfgs`, `mus`, `shift_modes`, `hybrid_steps` | | sampler details |
| `megapixels` | `1, 2, 3` | Stage 1 size |
| `seeds` (with `seed_policy` = seeds_axis) | `101, 102, 103` | |
| `prompts` | one prompt per line | always the slowest axis: the sheet reads prompt by prompt |
| LoRAs | see section 5 | |

`stage_scope` says which stage the sampler / steps / window fields apply to (`s1`, `s2`, `s3`, or `all_stages` =
the same value on all three). To list exact runs instead of a grid, see **4b. combos mode**.

**Count before you queue:** 4 samplers x 3 schedules x 3 prompts = 36 images. At ~2 minutes each that is over an
hour. `quick_screen` renders Stage 1 only at `screen_megapixels` (1 MP) - use it for wide first looks, then re-run
the short list properly.

## 4b. Hand-written cells: combos mode

Grid mode makes every combination of its axes. When you want an exact list of runs that does not form a grid -
three different sampler recipes, a handful of refine windows, a few stacks each with its own settings - set
`mode` = `combos` and write one line per cell.

### Where the lines go

| Field | Format |
|---|---|
| `combos_json` | One JSON object per line. Blank lines and lines starting with `#` are skipped; a trailing comma is tolerated. |
| `combos_file` | A path to a file. **Wins over `combos_json`.** Accepts a JSON array `[{...}, {...}]`, `{"combos": [...]}`, or the same one-object-per-line format. Use it for campaigns you want to keep, and when you want pretty multi-line JSON. |

### The rule

**Each line changes only the keys it names; everything else comes from the Ultra V2 panel.** The keys are the
Ultra V2 widget names - the same names a saved ultra preset and a cell PNG's `krea2_settings` chunk (`ultra`
section) use, so you can copy key/value pairs straight out of either.

```json
{"s1_sampler": "euler", "s1_schedule": "bong_tangent", "s1_steps": 8}
{"s1_sampler": "lcm_hybrid2", "sigma_profile": "OPT-DEIS12", "s1_steps": 9}
{"s1_sampler": "er_sde", "s1_schedule": "simple", "s1_steps": 9}
```

- The stage is in the key (`s1_`, `s2_`, `s3_`). **`stage_scope` does nothing in combos mode.**
- JSON spelling: double quotes around keys and text, numbers bare (`8`, `0.75`), `true` / `false` lowercase. In
  `combos_json` each object must sit on one line.
- **The plan does not check names or values.** A misspelled key makes that cell fail (red tile with the error;
  the rest of the sweep runs). A misspelled value may fail or quietly not do what you meant. Read the cell list
  in **Show plan** before you queue.

### Which Sweep Plan fields still apply

| Still applies | Ignored in combos mode (grid fields) |
|---|---|
| `prompts` - every line runs with every prompt (prompt blocks, slowest) | `samplers`, `schedulers`, `pair_lock` |
| `sigma_profiles` - every line x every profile | `steps`, `windows`, `etas`, `noise_types`, `cfgs` |
| LoRA fields (`lora_mode` + `lora_files` ...) - every line x every LoRA cell | `mus`, `shift_modes`, `hybrid_steps`, `megapixels` |
| `lora_triggers`, `keep_panel_stack`, `include_baseline` | `sde_bands`, `churn_forms`, `hybrid_handoffs`, `trigger_joins` |
| `quick_screen` / `screen_megapixels` (see the notes below) | `seed_policy` / `seeds` (use `_seed` in the line instead) |
| `max_combos`, scoring, sorting, sheets, output folder, notes | |

### Key reference - per stage

Replace `X` with `1`, `2` or `3`.

| Key | Values | Notes |
|---|---|---|
| `sX_sampler` | `euler`, `res_2m`, `res_2s`, `deis_3m`, `abnorsett_3m`, `abnorsett_4m`, `lawson4_4s`, `etdrk4_4s`, `er_sde`, `rk6_7s`, `lcm`, `lcm_hybrid`, `lcm_hybrid2` | |
| `sX_schedule` | `linear`, `balanced`, `karras`, `beta57`, `beta`, `bong_tangent`, `exponential`, `linear_quadratic`, `simple` | Built-in schedules only. For a saved ★ curve use `sigma_profile` (below). A line that sets `sX_schedule` switches that stage of a wired Sigmas node off for that cell. |
| `sX_steps` | 1-200 | |
| `sX_start_step`, `sX_end_step` | 0-199 / 0-200 | The window. `s1_end_step` 0 = run to the end. S2/S3 windows are where the refine starts and stops. |
| `sX_cfg` | 0-20 | Diffusers convention (0.8 here = classic 1.8). On a distilled model it only acts where `turbo_guidance` allows it. |
| `sX_eta` | 0-1 | Churn. No effect on `euler`, `lcm`, `er_sde`. |
| `sX_noise` | `white`, `low_freq`, `high_freq`, `pink` | Shapes the churn. Forced to `white` for the lcm family; no effect on `euler` / `lcm`. Live on `er_sde`. |
| `sX_hybrid_steps` | 0-98 | lcm hybrids only: number of lcm steps before the tail sampler; 0 = half. |
| `sX_hybrid_handoff` | 0-1 (a sigma) | lcm hybrids only: hand off at this sigma instead; above 0 it replaces `sX_hybrid_steps`. Beats the Hybrid Sigma Handoff node. |
| `sX_sde_hi`, `sX_sde_lo` | 0-1 (sigmas) | SDE band: churn only between these sigmas. Needs eta churn to matter. Beats the Hybrid Sigma Handoff node. |
| `shift_mu_sX` | 0.1-10 | Note the order: `shift_mu_s1`, not `s1_shift_mu`. Only acts with `distilled_shift` = `manual`. |

### Key reference - whole image

| Key | Values | Notes |
|---|---|---|
| `_seed` | integer | **The cell's seed.** Do not write `seed` - the sweep overwrites it with the panel seed. |
| `seed_mode` | `same_all_stages`, `offset_per_stage`, `random_per_stage` | |
| `s1_megapixels` | 0.25-16 | Stage 1 size. Beats `screen_megapixels` under quick_screen. |
| `aspect_ratio` | `21:9 ultrawide`, `16:9 wide`, `3:2 landscape`, `4:3 landscape`, `5:4 landscape`, `1:1 square`, `4:5 portrait`, `3:4 portrait`, `2:3 portrait`, `9:16 tall`, `9:21 tall` | Exact text, spaces included. |
| `width`, `height` | 0-8192 (step 16) | 0 = from aspect ratio + megapixels. |
| `upscale_to_stage2`, `upscale_to_stage3` | 0-8 | 0 = stop after the previous stage. |
| `s1_s2_upscale_vae`, `upscale_vae_mode` | true/false; `disabled`, `s2-s3`, `s2-s3 with downsample`, `final decode`, `final decode with downsample`, `both`, `both with downsample`, `both with final decode downsample` | Need an upscale VAE wired. |
| `upscale_renorm` | `off`, `global`, `per_channel` | |
| `degrid` | `off`, `hops`, `final`, `hops+final` | |
| `distilled_shift` | `fixed`, `resolution`, `manual` | |
| `turbo_guidance` | `off`, `s1_only`, `all_stages` | |
| `churn_form` | `rf`, `ve` | Beats the Hybrid Sigma Handoff node. |
| `crop_bottom`, `crop_overgen`, `init_match_size` | 0-96 (step 2); true/false; true/false | |
| `prompt` | text | Replaces the panel prompt for the cell (re-encoded, even when a conditioning node is wired). `<lora:name:w>` tags work as in section 5. |
| `negative_prompt` | text | |
| `cond_rebalance`, `cond_multiplier`, `cond_layer_weights` | true/false; 0-10000; 12 comma-separated numbers as one string | |
| `cond_jitter`, `cond_jitter_seed` | 0-1; -1 or an integer | |
| `ref_match_size` | true/false | |

### Key reference - sweep-only keys

These are not panel widgets; the sweep handles them before the cell runs.

| Key | Value | What it does |
|---|---|---|
| `sigma_profile` | a saved ★ sigmas preset name, e.g. `"OPT-DEIS12"`, `"EXP6"` | Builds that preset's curve for the cell and replaces any wired Sigmas node. An unknown name warns in the console and the cell runs the panel schedule. Check which stages the preset enables (EXP1 has all enables off). |
| `lora_stack` | a stack line: `"Shouji_Ink_KR2 @0.7 \| michaelparkes @1/0.5/0.7"` | The cell's LoRAs. Same name rules as `lora_files`; `@w` = all stages, `@s1/s2/s3` per stage, `name #option` for a word option. **No `@` = 1.0 here** (the card weight is only looked up for `lora_files` lines). `random:` / `all:` / `cross:` pickers are `lora_files`-only. |
| `lora_mult` | number | Optional multiplier on every weight of the cell's `lora_stack`. |
| `lora_file` + `lora_strength` | a LoRA name + a number | One LoRA at one strength (bakeoff cell). `lora_strength` must be a number - `card` is a `lora_strengths` word, not a combos value. |
| `lora_baseline` | `true` | The reference cell: no LoRA, or the panel stack alone with `keep_panel_stack` on. |
| `lora_target` + `lora_strength` | stack slot (`"1"`, or part of the name) + a number | Strength-ladder cell on the panel stack. Needs `lora_mode` = `strength_ladder` on the node. |
| `trigger_join` | `comma`, `period` | (temporary test key) how added LoRA words join the prompt. |

LoRA cells pick up their prompt words from `lora_triggers` like any LoRA sweep, and by default run without the
panel's LoRAs (`keep_panel_stack` adds them). With `lora_mode` = `none` the plan sees the LoRA keys and switches
itself to stacks (if any line has `lora_stack`) or bakeoff. **Don't mix `lora_file` and `lora_stack` lines in one
list** - the plan switches to stacks and the `lora_file` cells fail; write a single LoRA as a one-entry stack line
instead (`{"lora_stack": "Shouji_Ink_KR2 @0.8"}`).

### What the sweep does to your lines

- **quick_screen** forces `upscale_to_stage2` / `upscale_to_stage3` to 0 even if a line sets them, and sets
  `s1_megapixels` to `screen_megapixels` unless the line has its own.
- **lcm noise:** any lcm-family line with a coloured `sX_noise` runs white (marked in the console).
- **Duplicates are dropped:** two lines that only differ by a key with no effect (eta on euler, hybrid keys on a
  non-hybrid, ...) run once. The cell counter shows the real number.
- **Stage 1 reuse:** lines that only touch `s2_` / `s3_` keys share one Stage 1 render (fixed seed, no
  quick_screen, `seed_mode` not `same_all_stages`), so refine-window comparisons cost Stage 2 + 3 only.

### Examples

Stage 2 refine windows on one Stage 1 (only the first cell renders Stage 1):

```json
{"s2_sampler": "lcm_hybrid2", "s2_start_step": 5, "s2_end_step": 12, "s2_steps": 12}
{"s2_sampler": "lcm_hybrid2", "s2_start_step": 7, "s2_end_step": 12, "s2_steps": 12}
{"s2_sampler": "res_2m", "s2_start_step": 7, "s2_end_step": 12, "s2_steps": 12, "s2_eta": 0.4}
```

Hybrid handoff vs fixed lcm steps:

```json
{"s1_sampler": "lcm_hybrid2", "sigma_profile": "EXP6", "s1_steps": 12, "s1_hybrid_steps": 7}
{"s1_sampler": "lcm_hybrid2", "sigma_profile": "EXP6", "s1_steps": 12, "s1_hybrid_handoff": 0.65}
{"s1_sampler": "lcm_hybrid2", "sigma_profile": "EXP6", "s1_steps": 12, "s1_hybrid_handoff": 0.45}
```

Seeds and sizes by hand:

```json
{"_seed": 101, "s1_megapixels": 2}
{"_seed": 102, "s1_megapixels": 2}
{"_seed": 101, "s1_megapixels": 3, "aspect_ratio": "2:3 portrait"}
```

Shift:

```json
{"distilled_shift": "fixed"}
{"distilled_shift": "manual", "shift_mu_s1": 1.7}
{"distilled_shift": "manual", "shift_mu_s1": 2.1}
```

LoRA stacks, each with its own settings, plus a reference cell:

```json
{"lora_baseline": true}
{"lora_stack": "Shouji_Ink_KR2 @0.7 | michaelparkes @1/0.5/0.7", "s1_sampler": "lcm_hybrid2"}
{"lora_stack": "Shouji_Ink_KR2 @0.7", "lora_mult": 1.2, "s1_sampler": "er_sde", "s1_schedule": "simple"}
```

Different prompts per line (the `prompts` field would cross them with every line instead):

```json
{"prompt": "a lighthouse in fog, ink wash", "s1_sampler": "euler", "s1_schedule": "bong_tangent"}
{"prompt": "a fox in snow, gouache", "s1_sampler": "er_sde", "s1_schedule": "simple"}
```

The same cells as a `combos_file` (array form - multi-line objects allowed):

```json
[
  {"s1_sampler": "euler",       "s1_schedule": "bong_tangent", "s1_steps": 8},
  {"s1_sampler": "lcm_hybrid2", "sigma_profile": "OPT-DEIS12", "s1_steps": 9},
  {"s1_sampler": "er_sde",      "s1_schedule": "simple",       "s1_steps": 9}
]
```

## 4c. Sweeping checkpoints

Compare several Turbo checkpoints in **one** sweep: one folder, one contact sheet, one spreadsheet.

1. Add **Eric Krea2 Sweep Checkpoints** and wire its output into the Sweep Plan's `checkpoints` input.
2. Click **Checkpoints: N of M** on the node. The list shows every file in
   `models/diffusion_models/krea2/` (the same list as the Component Loader's dropdown). Tick the ones
   you want; the filter box narrows the list. The ticked names appear in the node's text box.
3. Set up the Sweep Plan as usual (grid or combos). Every cell runs on every ticked checkpoint.

How it runs:
- Checkpoints run **in the order of the pick list**, each loaded once. A checkpoint that is already
  loaded is used as is. To switch, the sweep unloads the current model and loads the next with your
  Component Loader settings, changing only the transformer.
- **Turbo only**: all checkpoints use the loader's base folder, and the sweep refuses to start if that
  base is not Turbo.
- A checkpoint that fails to load gets red tiles with the error; the sweep carries on.
- **All models are unloaded when the sweep ends** (also after Stop or an error), so the next queue
  loads fresh.
- Combos lines can name a checkpoint themselves: `{"checkpoint": "krea2\\myModel.safetensors", "s1_steps": 10}`.
  Such a line runs only on that checkpoint; lines without one are crossed with the ticked list.
- The cell counter under the Sweep Plan shows `cells x checkpoints = total`. `max_combos` applies to the
  total.

What you get:
- `contact_sheet.png` with one block per checkpoint under its own header, plus
  `contact_sheet_<checkpoint>.png` for each checkpoint.
- A `checkpoint` column in the spreadsheet and the review workbook; cell file names start with a short
  checkpoint name; each cell PNG records the checkpoint it really ran in its loader settings.
- Two difference scores: **U** (`uniqueness`) compares a cell with the other cells of the same
  checkpoint, as before. **X** (`uniq_ckpt`) compares it with the *same* cell (same settings and seed)
  on the other checkpoints - a high X means that checkpoint renders that setting most differently
  from the rest.
- Stage 1 reuse works within each checkpoint.

## 5. LoRAs

### lora_mode

| Mode | Each cell | Use it to |
|---|---|---|
| `bakeoff` | ONE LoRA (from `lora_files`) at one strength (from `lora_strengths`) | compare LoRAs, find a strength |
| `stacks` | a whole STACK, one line of `lora_files` per stack | test combinations |
| `strength_ladder` | the panel's own stack, with one entry's strength stepped | tune one LoRA inside a stack you already use |
| `none` | no LoRA sweeping; the panel stack is used as is | sampler / steps / seed / size sweeps |

By default a swept LoRA runs **alone** (the panel's LoRAs are removed) so it is judged against the bare checkpoint.
`keep_panel_stack` puts it on top of the panel stack instead. `include_baseline` adds a reference cell with no LoRA;
in stacks mode `stack_parts` says what else goes next to each stack:

| stack_parts | Added next to a stack A + B + C | Use it to |
|---|---|---|
| `each LoRA alone` (default) | A, B, C alone | judge a stack against its single LoRAs |
| `reference only` | nothing but the no-LoRA reference | screen many stacks fast |
| `build up` | A, then A + B (in the order written) | see what each layer adds |
| `leave one out` | A + B, A + C, B + C | find the LoRA a stack can do without |

A cell that two stacks share is made once. `include_baseline` off = no extra cells at all.

### Writing lora_files

```
# bakeoff: one LoRA per line
Shouji_Ink_KR2
3396751_michaelparkes_000001700_bf16

# stacks: one stack per line, LoRAs joined with |
Shouji_Ink_KR2 @0.7 | 3104259_arthurrackham_000001750 @0.6
Shouji_Ink_KR2 | michaelparkes @1/0.5/0.7
```

- **Names:** the file name with or without extension (the LoRA Catalog's *copy name* button), a loras-folder path,
  a full path, or the Civitai name with the version number and `_bf16` left off. A name that matches two files stops
  the plan and lists both.
- **Weights:** `@0.7` = all three stages; `@1/0.5/0.7` = Stage 1 / 2 / 3; no `@` = the LoRA Catalog card weight
  (1.0 when there is no card).
- **lora_strengths:** bakeoff: every LoRA at every value (`card` = its card weight, `card, 1.0` = both).
  stacks: optional multipliers on the whole line (blank = weights as written).
- Lines starting with `#` are notes.

### Picking LoRAs for you

```
all: illustration                          every LoRA in loras/Krea2/illustration
random: graphic-art x6                     6 random ones from that folder
random: catalog watercolor x8              8 whose catalog card mentions watercolor
all: catalog tag="art nouveau"             every LoRA tagged art nouveau
all: catalog kind=style_medium ink         filters: kind= role= arch= (default Krea2)
Shouji_Ink_KR2 | all: illustration         stacks: Shouji with each other LoRA in turn
random: graphic-art x4 @0.8 | V0id @0.6    stacks: 4 stacks, a new random pick in each
```

Random picks repeat for the same text (so a re-run is the same sweep); add `seed=7` for another draw, `seed=new`
for a fresh draw every queue. **Show plan** lists the picked files; so do the sweep folder's files.

In a stack line, the slots' picks are **lined up**: `random: illustration x4 | random: graphic-art x4` makes 4 stacks
(pick 1 with pick 1, pick 2 with pick 2 ...). Start the line with **`cross:`** to get **every combination** instead:

```
cross: random: illustration x4 | random: graphic-art x4 | random: treatment x4     4 x 4 x 4 = 64 stacks
cross: Shouji_Ink_KR2 @0.7 | random: graphic-art x6 | all: treatment                1 x 6 x every treatment LoRA
```

Each slot draws its LoRAs once, so every LoRA shows up in many stacks - after you rate them you can see which LoRA
lifts or sinks its stacks. A combination holding the same file twice is skipped. Watch the cell counter: crosses grow
fast.

### The words added to the prompt (`lora_triggers`)

Most LoRAs need words in the prompt to show, and on fine-tuned checkpoints (Lox, FinalCut) they need **more** words
than on the base model. `lora_triggers` decides what each swept LoRA adds; the words go in front of the cell's
prompt. Baseline cells get none.

| Choice | What each LoRA adds | When |
|---|---|---|
| `none` | nothing | testing the LoRA alone, or your prompts already carry the words |
| `trigger` | its main trigger from the card - one name, sometimes a few words (`arackham`, `retro artstyle illustration`) | base model; lean stacks |
| `full` | its whole recipe: official trained words + trigger + style phrase | fine-tunes; when a LoRA "does nothing" |
| `saved pick` (default) | the words picked for the loaded checkpoint - yours (card editor, Sweeps page) or a trigger test's - else trigger + style phrase | normal use |

With two or more LoRAs in a cell, `full` and `saved pick` merge the words: repeats dropped, picks kept whole, a
length budget shared by weight, and opposite words ("muted" vs "vibrant") reported in Show plan and the console.

Where the words come from: the LoRA Catalog writes each LoRA a *recipe* (`run.py triggers`); a *trigger test*
(`run.py triggers --plan-test`, then `--check`) scores the variants on a checkpoint and saves the best as that
checkpoint's pick; you can pick by hand in the card editor (Prompt words panel) or with "use words" on the Sweeps
page. When the recipe itself is wrong (a bad trigger or style phrase), correct it under **Recipe** in the same panel:
your trigger, style phrase, official words or Full line replace the model's in `full` and in `saved pick`'s default,
and survive re-running `run.py triggers`. **Show plan** says, for every LoRA, which of these it used.

A real example: a stacks sweep on Lox with `trigger` gave `arackham, ...` and `inkrelic, ...` - one word each in
front of a 390-token prompt, and the LoRAs barely showed. With `saved pick` the same cells get
`arackham, pen and ink linework with heavy cross-hatching ..., muted sepia-toned palette, ...`.

`<lora:name:weight>` tags inside a sweep prompt are taken out of the text and loaded for that cell (a tag for a LoRA
already in the cell sets its weight). They get no words added - write the words in the prompt yourself.

## 6. What a sweep writes

A folder under `output/sweeps/` named `<date>_<sweep name>_<checkpoint>`:

- every cell as a PNG with its full recipe embedded, including the prompt as sent and the words added (drag it
  into ComfyUI to reproduce it),
- a labelled contact sheet,
- `sweep_review_<name>.xlsx` - thumbnails, what varied, **LoRAs used**, empty columns for your
  ratings, the metrics and (after the Sweep Folder Scorer) the UniPercept scores,
- `sweep_summary_<name>.csv`, `sweep_manifest_<name>.json` (per cell, also the words added), `sweep_info_<name>.txt`
  (preset, notes, checkpoint, what varied, the random picks).

Write `run_notes` before you queue: why you are running this one today. It lands in the info file
and the manifest, so the folder explains itself next month.

### Share sheets (for Civitai, Reddit ...)

With `share_sheets` on (default `strips`), the sweep also writes ready-to-post JPEGs into `<sweep>/share/`:

- **strips** - one picture per prompt. On top: the prompt (3 lines at most) and the settings every image in it
  shares (checkpoint, sampler, schedule, steps, seed, stages, LoRA words). Under each image: the LoRA's short name
  and weight, then `by <creator> - civitai <model id>`, then any setting that differs between images. Images are
  768 px so people can pixel-peep; a strip wraps into more rows instead of growing past 6000 px, and splits into
  parts if it would get taller than that. The footer credits the creators.
- **singles** - every image at full size on a border: settings on top, its LoRAs (name, weight, creator, model id)
  underneath, no prompt. Named `single_cell_###_<lora>.jpg`.

`share_background` picks black or white. Changing these two never flips the preset to `custom`, and Clear keeps them.

To **add your star ratings and notes**, rate the cells on the LoRA Catalog's Sweeps page, then press **Share
sheets** there (choose strips / singles, background, ratings: stars or stars + notes, one strip per prompt / LoRA /
seed). The **Eric Krea2 Sweep Sheets** node does the same inside ComfyUI for any old sweep: plug in Ultra V2's
`manifest_path`, or paste a sweep folder.

## 7. Recipes for common questions

**"Is this new LoRA any good?"** - Start 01 with its name. Look at the reference cell first, then 0.7 and 1.0.
If nothing changes, switch `lora_triggers` to `full` and queue again.

**"Which of my 35 illustration LoRAs should I keep?"** - Start 02 (quick screen), rate in the workbook, then Start 01
on the short list.

**"What goes well with Shouji?"** - Start 04. Compare each pair with its two single-LoRA cells next to it.

**"Why does this LoRA show on base turbo but not on Lox?"** - Start 10 on each checkpoint, or a trigger test in the
LoRA Catalog (`run.py triggers --plan-test --names <name>`), which tries none / trigger / official / trigger+phrase /
full and saves the winner as the pick for that checkpoint.

**"Which illustration + graphic-art + treatment stacks work?"** - Start 12: 64 stacks and one reference, quick
screen. Mark them on the LoRA Catalog's Sweeps page; its **Follow-up** button lists the stacks you marked good as
lines. Paste them into Start 13 (`stack_parts` = build up) to see none -> A -> A + B -> A + B + C side by side (Space
on a stack in the Sweeps page shows the same row). Put the LoRA you want as the first layer first in each line.

**"Best sampler for this fine-tune?"** - Start 06, then *All samplers + schedulers* at quick-screen size on the
two best rows.

## 8. Troubleshooting

| Symptom | Look at |
|---|---|
| "plan error" in the counter | the message names the field; usually a misspelled LoRA name or sampler |
| LoRA barely shows | Show plan: which words did it get? Try `full`; check the weight |
| cell counter huge | an `all:` folder or several crossed axes; use quick_screen or trim |
| same picks every time | add `seed=new` to the `random:` line |
| a name matches two files | write the folder path, or remove the duplicate file |
| combos cell fails with "unexpected keyword" | a misspelled key in that line - compare with section 4b |
| combos seed ignored | write `_seed`, not `seed` |
| the sweep stops on cancel | the folder is still written with what finished |
| no `share` folder | `share_sheets` was off, or the console says "share sheets skipped" and why |
