# Sweeps - a practical guide

A sweep renders many versions of one image in a single queue item - different LoRAs, strengths, samplers, step
counts, seeds, prompts - and lays them out so you can compare them. The model and LoRAs stay loaded the whole time,
so it is much faster than queueing the same runs by hand.

This guide covers the **Eric Krea2 Sweep Plan** node. For the full reference of every field, hover over it in
ComfyUI; the README's [Parameter sweeps](../README.md#parameter-sweeps) section has the technical details.

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
the same value on all three).

**Count before you queue:** 4 samplers x 3 schedules x 3 prompts = 36 images. At ~2 minutes each that is over an
hour. `quick_screen` renders Stage 1 only at `screen_megapixels` (1 MP) - use it for wide first looks, then re-run
the short list properly.

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
| the sweep stops on cancel | the folder is still written with what finished |
| no `share` folder | `share_sheets` was off, or the console says "share sheets skipped" and why |
