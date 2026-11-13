# Parameter sweeps - full reference

[<- Back to the README](../README.md) | [All docs](../README.md#documentation)

## Parameter sweeps

A fine tune's preferred sampler, scheduler, step count, mu and cfg tolerance
are all proxies for one variable - how far its training eroded the turbo
distillation. The sweep system turns that from folklore into measurement:
characterize a model in one queue item, keep the evidence, promote the
winners to presets.

**Wiring:** `Eric Krea2 Sweep Plan` → Ultra V2's `sweep` input (last socket).
The Ultra panel is the *base recipe*; each cell overrides only its swept keys.
All existing machinery (LoRA stack, cond rebalance, ref latents, telemetry,
renorm) runs untouched per cell.

**Grid mode:** fill any of the axis fields - `samplers`, `schedulers`
(`pair_lock` zips them into fixed pairs), `steps`, `windows` (`5-12` →
start/end step), `etas`, `noise_types`, `cfgs`, `shift_modes`/`mus`,
`hybrid_steps`, `megapixels` - blank = not swept (panel value used). The
`schedulers` field takes built-in schedules AND saved sigma-shape ★ preset
names mixed freely (each entry resolves as a schedule first, then as a
sigmas preset applied as that cell's full profile; unknown names are
skipped with a console warning, never an abort). Both fields start empty
(2026-10-16); the sweep preset *All samplers + schedulers (lists only)* fills them with
everything available - delete what you don't want. *Clear sweep* (the button at the bottom
of the node) resets every field to its default. Full cross product;
a `max_combos` above 0 hard-refuses bigger plans with an axis-size breakdown (0, the default
since 2026-10-15, = no cap - the readout under the node shows the cell count as you edit). Cells that differ
only by inert knobs (eta/noise on `euler`/`lcm`) are deduped automatically.

**Combos mode:** one JSON object per line in `combos_json`, or a file via
`combos_file` (JSON array, `{"combos": [...]}`, or JSON-lines) - keys are the
same widget keys as presets and PNG recipes, so combos are copy-pasteable
from any stamped image. Sweep campaigns become versionable files.

**Workflow intent:** `quick_screen` forces S1-only at `screen_megapixels` for
cheap wide grids - perfect Stage 1 first, then sweep `stage_scope=s2`/`s3`
refinement on the short list. For s2/s3-only sweeps the post-S1 latent is
computed once and **reused** across cells (`s1_reused` in the manifest;
automatically disabled per cell for anything that changes Stage 1, and
globally under `seed_mode=same_all_stages`).

**Outputs** (to `output/sweeps/<timestamp>_<sweep name>_<finetune>/`, or `output_folder`). The
sweep name is the preset name, `custom`, or `<preset>-edited`, and it is also in the file names
below - `sweep_manifest_<name>.json` and so on (2026-10-15; older sweeps keep
`sweep_manifest.json` etc., and every reader accepts both):
- `cell_NNN_<values>.png` - every cell at full res with the complete resolved
  recipe (`ultra` + `loader` + `lora` sections) embedded in a
  `krea2_settings` chunk - each file is standalone-reproducible.
- `contact_sheet.png` - downscaled overview (`tile_px` tiles), captions show
  only the diffs vs base, green line = per-cell generation time + scores.
- `sweep_summary.csv` - one row per cell: swept columns, `gen_time_s`,
  rounded scores - sortable in any spreadsheet.
- `sweep_manifest.json` - full resolved recipes, timings, metrics, errors, and an `about`
  block (preset name, preset notes, run notes, random LoRA picks).
- `sweep_info.txt` (2026-10-15) - what was run and why, in plain text: preset name, preset
  notes, run notes, checkpoint, date and duration, cell count, what varied, the random LoRA
  picks, and the panel settings every cell started from.
- `sweep_review.xlsx` (2026-10-15, needs `openpyxl`) - one row per cell: a thumbnail, the
  swept values, `LoRAs used` (each cell's actual stack and weights, swept or not), empty
  columns for your own ratings (rating, fine detail, render quality,
  contrast, composition, cohesiveness, notes), then the UniPercept scores from `scores.csv`
  and the automatic metrics; panes frozen at the thumbnail; an About sheet with
  `sweep_info.txt`. Works for any kind of sweep. When the workbook is rebuilt (e.g. after
  scoring), the ratings and notes you typed are carried over cell by cell and the previous
  file is kept as `sweep_review.prev.xlsx`; if the workbook is open in Excel the rebuild is
  skipped with a note instead of failing. Older sweep folders can be backfilled:
  `python _sweep_review.py <sweep folder> [...]`.

The extra files are written even when you cancel a sweep, before the queue item stops.

**Scores** (optional, CPU, milliseconds - failure detectors, not beauty
judges): Laplacian + Tenengrad sharpness, wavelet noise sigma, flat-region
chroma variance, clip fraction. All land in the CSV/manifest; `sort_by`
pre-sorts the sheet; scores never hide a cell - the eye stays the judge.

**Field calibration** (two full 192-cell characterization runs, 2026-07-29):
**clip%** is the best bad-image predictor - above 3-4% suspect, above 6%
almost always bad (the sheet flags `C4.2%!` / `C7.1%!!` at exactly those
absolute thresholds, and Auto Pick's gates use them). **sharp_ten**
(Tenengrad) proved more trustworthy than Laplacian - it is the sharpness
shown on the sheet and gated by Auto Pick - though neither is immune to one
strong edge dominating; subject/tiled sharpness via the external `scores.csv`
route is the real answer there. **noise sigma** and **splotch** showed no
quality discrimination in the field - they remain CSV columns but gate
nothing and no longer appear on the sheet.

**Sheet captions** are stage-grouped shorthand - one line per swept stage
(`s1: 2.5mp res_2m+beta57 h7 e0.3`), ★profile / seed / prompt on a trailing
line - plus a compact metric line: `T88s  S11.3k  C0.5%  N1.0  U159`
(T time, S sharp_ten/1000, C clip% with the !/!! flags, N noise, U
uniqueness).

**Uniqueness** (model-free, automatic): every cell gets a
composition-uniqueness score - its mean distance from the other cells'
64x64 luma thumbnails, grouped per (prompt, S1 megapixels). READ THE
DIRECTION BY EXPERIMENT TYPE: in a fixed-seed SETTINGS sweep, high
uniqueness means "deviates from the consensus" and broken cells deviate
hardest - the LOW end is the consensus render and the high end is an
outlier list; in a SEED sweep, divergence is the point and high = more
compositional variety. Auto Pick has an objective for each direction.

**Promotion:** `Eric Krea2 Sweep → Preset` reads a manifest cell into
`ultra_presets.json` (swept keys only, or the full recipe), auto-tagged with
the checkpoint name and an `objective` note - per-fine-tune preset libraries
by construction.

**LoRA, prompt & sigma-profile axes (sweep v2):**

- **LoRA bake-off** (`lora_mode=bakeoff`) - one LoRA per line in
  `lora_files` (loras-folder names or full paths) x `lora_strengths`
  (one value drives all three stage weights). Default: each swept LoRA runs
  ALONE, so every style is measured against the bare checkpoint on equal
  footing; `keep_panel_stack` layers it on top of your declared stack
  instead. `include_baseline` adds the reference cell (no-LoRA, or the
  panel stack alone when layering). Sweep-injected entries are ephemeral -
  fused per cell, cleaned per cell, nothing persists.
- **Strength ladder** (`lora_mode=strength_ladder`) - keeps the
  panel-declared stack and steps only the `lora_target` entry (1-based
  index, or a name substring) through `lora_strengths`.
- **Stacks** (`lora_mode=stacks`, 2026-10-09) - tests LoRAs TOGETHER. One
  whole stack per line in `lora_files`, LoRAs separated by `|`, each
  `name @weight` (all three stages) or `name @s1/s2/s3` (no `@` = the LoRA's LoRA Catalog card weight,
  else 1.0 - since 2026-10-15; the cell text, CSV and workbook show the number used); a name is a
  loras-folder path, a full path, or just the file name (found in any subfolder, so a
  line survives moving the LoRA);
  lines starting with `#` are ignored:
  `comic_book_v2 @0.8 | V0id @0.6`
  `comic_book_v2 @0.8 | halftone_print @0.5 | Detail Slider @1.5`
  `include_baseline` adds the no-LoRA reference cell PLUS each distinct
  LoRA of the lines alone at its line weight, so every combination is judged
  against its parts (duplicates dropped). `lora_strengths` is optional here:
  multipliers that scale every weight of every stack and single (blank = as
  written). `keep_panel_stack` layers each stack on your panel stack; trigger
  words merge for every LoRA in the stack. Captions read
  `comic_book_v2 @0.8 + V0id @0.6`. The LoRA Catalog writes these sweeps for you
  (`run.py sweep-plan`, saved as a sweep preset) and marks the results on its Sweeps
  page. Spec: docs/spec_lora_stack_sweeps_2026-10-09.md.
- **Random LoRA picks** (2026-10-15) - see [Random LoRA picks](#random-lora-picks) below.
- **Prompt axis** - one prompt per line in `prompts`; each prompt cell
  re-encodes (conditioning cleared automatically, ~seconds/cell). Prompts
  expand as the SLOWEST axis, so the contact sheet and CSV read in
  prompt-major blocks - metric comparisons are only meaningful within a
  block.
- **Prompt files (2026-10-26)** - a line in `prompts` can pull prompts from a
  text file (one prompt per line; blank and `#` lines skipped), worded like
  the LoRA picks:

  ```
  file: D:\prompts\portraits.txt              (every prompt in the file; all: works too)
  random: D:\prompts\portraits.txt x20        (20 different prompts, drawn at random)
  random: D:\prompts\portraits.txt x20 seed=7     (the same 20 every run)
  A lighthouse at dawn, oil on canvas          (typed prompts still work, mixed in)
  ```

  No seed = a fresh draw every queue, `seed=7` = the same draw every run. A
  relative path starts from the ComfyUI folder. Fewer lines than asked = all
  of them (noted). Every cell records the actual prompt text, so results stay
  reproducible if the file changes; editing the file rebuilds the plan on the
  next queue. Combines with everything, e.g. `random: prompts.txt x20` plus
  `random: catalog comic | random: catalog illustration per prompt` = 20
  prompts, each with its own random comic + illustration stack.
- **Sigma-profile axis** - `sigma_profiles` takes saved ★ preset NAMES from
  the Eric Krea2 Sigmas node (validated at plan time); each cell rebuilds
  the full bundle from that preset, overriding any wired sigmas node for
  that cell.
- **Trigger words** (`lora_triggers`, default `catalog`) - each bake-off
  cell merges its swept LoRA''s trigger words into that cell''s prompt.
  `catalog` = ONE trigger per LoRA from its LoRA Catalog card (your editor
  priority first; switched-off triggers skipped), else the first cached
  Civitai trigger, always prepended; `prepend` / `append` = every cached
  Civitai trigger word; `off` = none (use it when the cells carry their own
  words, as the trigger-test presets do). On fine-tuned checkpoints one short
  trigger is often not enough - see *LoRA trigger words: base model vs
  fine-tunes* above. Only sweep-injected entries contribute; baseline
  and ladder cells are untouched; a changed prompt re-encodes automatically;
  the effective prompt + triggers land in the cell''s metadata.

#### Random LoRA picks

In bake-off and stacks mode, a line (or a stack slot) in `lora_files` can draw LoRAs at
random from a folder:

```
random: graphic-art x6                          (bakeoff: 6 different LoRAs)
random: graphic-art x4 @0.8 | V0id @0.6         (stacks: 4 stacks, a new pick in each)
random: graphic-art @0.7 | random: photo @0.5   (stacks: two random slots)
```

- **Folder:** a subfolder of `loras/Krea2` (`graphic-art`), a loras-folder path, or a full
  path; subfolders are included (the Civitai `id-Name/id-version/` layout).
- **`xN`** = how many (default 1). Picks are distinct while the folder has enough files;
  two random slots in one stack never draw the same file, and a random slot never repeats a
  LoRA written by hand in the same line.
- **Seeds (changed 2026-11-03):** no seed = a **fresh draw every time the sweep is queued**;
  `seed=7` (any word or number) = the same picks every run - use it to compare checkpoints or settings
  on identical LoRAs. Each slot has its own seed: `random: a seed=7 | random: b` keeps `a` and redraws `b`.
- **Recorded:** the picks are printed in the console, written as loras-folder paths into
  every cell (caption, CSV, PNG recipe, the workbook's `LoRAs used`), and listed under
  "Random LoRAs" in `sweep_info` and the manifest's `about` block.
- Mixes freely with ordinary lines, `lora_strengths` (including `card`), baselines and
  `#` comment lines (now skipped in bake-off mode too, as the tooltip always said).

**`all:` (2026-10-16)** takes every LoRA of the source instead of a random few:

```
all: illustration                               (bakeoff: one line per LoRA in the folder)
Shouji_Ink_KR2 | all: illustration              (stacks: Shouji with each of the others in turn)
```

In stacks mode an `all:` slot makes one stack per LoRA in the source (minus LoRAs written by
hand in the same line); with two `all:` slots the larger one sets the count.

**Catalog sources (2026-10-16; creator filter and per-prompt picks 2026-10-26).** Instead of a
folder, `random:` and `all:` can pick from the LoRA Catalog: `catalog` followed by words and filters.

```
random: catalog watercolor x4                     (4 LoRAs whose card mentions watercolor)
all: catalog tag="art nouveau"                    (quotes for a phrase)
all: catalog kind=style_medium ink                (kind = part of the card kind)
random: catalog role=style noir x3                (role exact)
random: catalog creator=Adel_Ai x4                (4 of Adel_Ai's Krea2 LoRAs)
random: catalog creator=Adel_Ai tag=watercolor    (filters combine)
all: catalog creator=Adel_Ai arch=Flux            (his Flux LoRAs instead of Krea2)
```

| Filter | Matches | Notes |
|---|---|---|
| plain words | anywhere on the card: name, file, folder, kind, effect, facets, artists, triggers, creator | every word must match |
| `tag=` | the same card text, as one phrase | quotes for spaces: `tag="art nouveau"` |
| `creator=` / `author=` | the card's creator | case, `_` and `-` ignored: `Adel_Ai` = `adel ai` |
| `kind=` | part of the card's kind | `kind=style_medium` |
| `role=` | the card's role, exactly | `role=style` |
| `arch=` | the top folder under your loras folder (`Krea2`, `Flux`, `Z_Image`, `Wan`) | **default `Krea2`**; `arch=any` = every base |

Only LoRAs whose file exists are picked. A query that finds nothing stops the plan with a clear
message; **Show plan** lists the picks before you queue.

**One pick per prompt - `per prompt` (2026-10-26).** Normally the prompts axis crosses with the LoRA
list (10 prompts x 10 LoRAs = 100 cells). Add `per prompt` to a `random:` line and every prompt line
gets its own draw instead, paired with it:

```
random: catalog creator=Adel_Ai per prompt                    (bakeoff: one LoRA per prompt)
random: catalog creator=Adel_Ai x2 per prompt                 (bakeoff: two single-LoRA cells per prompt)
random: catalog creator=Adel_Ai x2 per prompt @0.8            (stacks: ONE stack of two per prompt)
random: catalog creator=Adel_Ai x2 per prompt | V0id @0.6     (stacks: two picks + a fixed LoRA in every stack)
random: comic | random: illustration per prompt   (stacks: one from each folder - one 'per prompt' covers the line)
```

- **`xN` means different things by mode:** bakeoff = N separate cells per prompt, one LoRA each;
  stacks = one stack holding N picks.
- Needs the `prompts` field (one prompt per line), and the `per prompt` line must be the only line
  in `lora_files`. Only `random:` works per prompt (not `all:`).
- Picks are distinct within a stack, never repeat a LoRA written by hand in the line, and don't
  repeat across prompts while enough LoRAs match; otherwise the plan notes that some prompts share.
- No `@` = each pick gets its card weight. `seed=` works as for other picks.
- `include_baseline` and `stack_parts` work within each prompt: its own no-LoRA reference, its own
  LoRAs alone, build up, or leave one out.

**Proven partners - `partner:` (2026-10-27).** In stacks mode a `partner:` slot adds a partner for the line's
first LoRA, chosen from what you have already rated:

```
random: catalog comic x4 | partner: best           (4 stacks: each pick + its best partner)
Frazetta Style | partner: random                   (one of Frazetta's good partners, a new one per seed)
random: catalog creator=Adel_Ai per prompt | partner: best @0.6   (per prompt, partner at 0.6)
```

Partners come from the LoRA Catalog's **stack synergy** - how much better your rated stacks of the two scored
than the two LoRAs' own ratings predict (prompt and checkpoint taken out; shown on each card as "Pairs well
with"). With no rated stacks for that LoRA yet, it falls back to what Civitai users stack it with (owned LoRAs
only, from `run.py harvest`). `best` (the default) takes the top one; `random` any of your positive partners;
`@w` and `seed=` work as elsewhere. A LoRA already in the line is never its own partner; with no partner known
the slot is dropped and Show plan says so. Rate stacks with `stack_parts` "each LoRA alone" so the catalog can
compare the stack with its parts - that is what makes synergy reliable.

**Word options in sweeps (2026-10-20; `#random` 2026-10-26).** For LoRAs whose card lists word
options (Prompt words > Word options - e.g. one keyword per colour grade):

```
cinematic_grade #teal orange @0.8      (that option's words)
cinematic_grade #each                  (one cell per option)
cinematic_grade #random                (one option drawn at random)
random: catalog creator=Adel_Ai x2 per prompt #random   (every pick draws its own option)
```

`#random` draws a new option every queue. A card can mark a default option with a leading `*`; it is used when you don't pick.
In the LoRA Stack node the same choice lives in `word_options` (`3: teal orange`, or `3: random` for
a new option every queue); an amber dot left of a slot means its LoRA has options and none is
chosen - click it to pick.

**Names typed by hand** in `lora_files` (and in stacks lines) now resolve like the Prompt LoRA
Tags node: the exact file name or stem, or the Civitai name with the version-id prefix and the
`_bf16` suffix left off. A name that fits two files stops with both listed. A LoRA written
without `@` in a stack gets its card weight.

All v2 axes cross with every classic axis, in grid mode AND on top of
explicit combos, under the same cap and dedupe rules. LoRA/prompt/profile
cells always run a full Stage 1 (S1-latent reuse correctly sits out). Each
cell PNG records the ACTUAL stack, bundle, and prompt that cell ran - not
the panel's - so every file stays a complete standalone recipe.

Failed cells never kill a sweep (red tile + manifest error), and interrupts
finalize the files on disk before stopping the queue item.

**Auto Pick:** `Eric Krea2 Sweep Auto Pick` closes the loop from sweep to
preset: manifest in, `cell_index` + a human-readable report out. Objectives:
`fastest_acceptable` (fastest cell passing the field-calibrated gates:
clip% ABSOLUTE - reject above 6%, flag 4-6% as suspect-but-passing;
sharp_ten relative to the sweep median or an eye-picked `reference_cell`
with `tolerance` slack; and a uniqueness consensus-outlier gate - cells far
above the reference are rejected as deviants, with 2x tolerance so style
variation passes), `most_unique` (highest composition-uniqueness - the SEED
sweep objective; recomputed from cell PNGs for older manifests),
`most_typical` (LOWEST uniqueness - the consensus render, the right pick in
settings sweeps where high uniqueness flags broken outliers), `best_metric`
(any built-in column), and `best_external` (any column of a `scores.csv` in
the sweep folder, joined by filename - blank metric = first numeric column).
Wire `cell_index` into Sweep -> Preset (convert its widget to an input) and
one queue runs sweep -> pick -> promote. The picker proposes and explains
its choice with runners-up; promotion only happens because you queued the
preset node - metrics never silently write your preset library.

**Folder scoring chain:** the `Eric Sweep Folder Scorer` node (in the
Eric-image-classification package) consumes the manifest path and grades
every cell with the analyzer suite - Blur V7, Noise V4, composition,
exposure, UniPercept (with a `device` picker for the second GPU) - and
writes/merges `scores.csv` in the sweep folder, one column per numeric
output (`comp.composition_score`, `blur7.variance`, ...). Merging is by
filename, so multiple grading passes stack columns instead of clobbering.
UniPercept runs in its `direct_logits` mode here (2026-10-15): continuous 0-100 scores
(`unip.iaa` aesthetics, `unip.iqa` quality, `unip.ista` structure & texture) plus a spread
for each (`unip.iaa_sd` ...: how torn the model is, in score points; the workbook calls these
columns *aesthetics doubt*, *quality doubt*, *structure doubt*), about 4x faster than
the old whole-number mode. A grader that fails is recorded as a failure, never as placeholder
scores. After scoring, the scorer rebuilds `sweep_review.xlsx` so the scores appear next to
your ratings (ratings kept).
The full hands-off chain is: Ultra V2 `sweep_manifest` -> `Krea2 Unload`
(frees the generator's VRAM; its wildcard passthrough keeps the ordering) ->
`Sweep Folder Scorer` -> `Auto Pick` (`best_external`) -> `Sweep -> Preset`:
one queue from characterization to a machine-proposed, human-ratified
preset.

**Sweep presets, the toggle & notes:** the Sweep Plan node carries an
`enabled` switch (off = the wired node is inert and normal generation runs)
and a `sweep_preset` dropdown backed by `sweep_presets.json`. Selecting a
preset WRITES its values into the panel - you always see exactly what will
run - and editing any field flips the dropdown back to `custom`; the ★ Save
Preset button saves the panel under a new name. Headless/API runs apply the
named preset as the sole source. Both the Sweep Plan and Sigmas nodes carry
a `preset_notes` field saved with each ★ preset - one sentence on what the
preset is FOR - written into the panel on select and printed by headless
runs, so preset libraries stay self-documenting. The Sweep Plan's `run_notes`
field (the last widget, 2026-10-15) is for THIS run - what you are checking, what changed
since last time. It is never saved into a preset, typing in it does not flip the dropdown to
`custom`, and it lands in `sweep_info.txt` and the manifest with the preset name and notes.

**Swept schedules vs a wired sigmas bundle:** a cell that explicitly sweeps
a stage's schedule disables the wired bundle for that stage, cell-locally -
a swept axis is never silently inert (the bundle previously overrode swept
schedules, which produced identical cells and only a warning). Unswept
stages keep the bundle; `sigma_profiles` cells replace it entirely; each
cell's PNG chunk records what actually applied.
