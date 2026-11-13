# LoRA stack sweeps: testing which LoRAs work together (spec draft 2026-10-09)

Status (2026-10-09): steps 1 and 2 of section 4 BUILT - `lora_mode = stacks` in SweepPlan + Ultra V2 (verify: docs/tools/verify_stacks_mode.py), and the click-to-mark Sweeps page + `stacks` table in the LoRA Catalog card editor. Step 3 BUILT the same day (`run.py sweep-plan`, folder-based per Eric's decisions; names in lines are file names, resolved by a new file-name fallback in Ultra V2). Step 4 (Suggester pair bonus) not started. Connects the Krea2 sweep system to the LoRA Catalog (families, harvested
Civitai pairs, the good/bad stacks table) so combinations can be tested on our own renders, not only taken from
Civitai popularity.

## 1. What already works today (no code)

The sweep's `lora_mode = bakeoff` with **keep_panel_stack ON** already tests "anchor + each candidate":
put the anchor LoRA (e.g. a comic style) in the Multi-LoRA panel, list candidates in `lora_files`, set
`lora_strengths` (e.g. `0.5, 0.8`), `include_baseline` ON (= anchor alone). One cell per candidate × strength, the
anchor present in every cell. Every cell PNG already records its full LoRA stack in the krea2_settings chunk.

What it cannot do: arbitrary stacks per cell (A+B, A+C, B+C, A+B+C ...), different strengths per LoRA inside a cell,
or take its candidate list from the catalog.

## 2. Additions

### 2.1 SweepPlan: `lora_mode = stacks` (new option in the existing combo; position-safe)
`lora_files` then holds one STACK per line:

```
comic_book_v2t2 @0.8 | V0id @0.6
comic_book_v2t2 @0.8 | halftone_print @0.5
comic_book_v2t2 @0.8 | V0id @0.6 | Detail Slider @1.5
```
- `name @weight` (weight → S1=S2=S3), or `name @s1/s2/s3` for per-stage weights; names resolved like bakeoff (loras
  folder name or full path).
- `keep_panel_stack` ON = each line is layered on top of the panel stack; OFF = the line is the whole stack.
- `include_baseline` = one cell per DISTINCT LoRA in all lines, alone at its line weight (so every pair is judged
  against its parts). Can be large; a widget-free rule: baselines are deduped across lines.
- `lora_strengths` in stacks mode = optional global multipliers (e.g. `0.75, 1.0`): each stack line × each multiplier.
- Plan vocabulary: reserved key `lora_stack` (list of entries) handled in V2 `_apply_v2_keys` next to `lora_file`.
  Stack entries use the existing schema (path/filename/lora_name/adapter_name "sweep_"/strength/weight_s1-3/
  ephemeral True). Trigger merge (`lora_triggers`) applies to every swept entry; sliders skipped (no triggers).

### 2.2 Catalog: `run.py sweep-plan` writes the stacks for you
```
python run.py sweep-plan --anchor "KreaComics" --family comic_illustration --top 12
python run.py sweep-plan --family comic_illustration --pairs 15            (best pairs inside a family)
python run.py sweep-plan --anchor "Vallejo" --from harvest --top 10        (Civitai co-use partners you own)
```
Candidate sources, mixable: same / friendly family (card family rules), harvested pairs (lora_pairs, owned only),
your ratings (★4-5 first, ★1-2 never), role fit (style + treatment + detailer, not style + style unless same family).
Output: a ready **sweep preset** written into Eric_Krea2's `sweep_presets.json`
(`{"name": "stacks comic x12", "sweep": {"lora_mode": "stacks", "lora_files": "...", "prompts": "...", ...}}`) - select
it in the SweepPlan dropdown. Weights start at each card's typical weight (blend ratio 0.6 for a second style).

### 2.3 Verdicts back into the catalog
After looking at the contact sheet:
- copy (or drag) the cell PNGs you like into `A:\LoRA_Catalog\stacks\good`, ones that failed into `stacks\bad`;
- `python run.py stacks-import` reads each PNG's krea2_settings chunk (the exact stack, weights, prompt, seed,
  checkpoint) - and the .txt files you already write - into the catalog `stacks` table (verdict, source "sweep" or
  "render", sha256 of each LoRA, base model);
- `lora_pairs` gets a third source beside Civitai: your verdicts, which outrank it (a bad stack = strong penalty for
  each pair in it, a good one = strong bonus).
Optional later: the sweep's existing Folder Scorer / `best_external` contract (ArtiMuse / UniPercept scores.csv) as a
first-pass ranking, the eye still decides.

## 3. Experiment design (defaults the generator writes)
- 3 prompts per family, chosen to stress it: e.g. comic = a portrait, an action scene, a city establishing shot.
- 2 seeds (seeds_axis), fixed sampler recipe (the field-tested S1 recipe), S1 2 mp for screening
  (quick_screen for first passes).
- Size check: 12 stacks × 2 multipliers + ~8 baselines = 32 configs × 3 prompts × 2 seeds = 192 cells
  (≈ your full characterization sweeps; ~1-2 h at 2 mp). The generator prints the count and refuses above max_combos
  as the sweep already does.
- Metrics stay failure detectors: clip% flags blown stacks (stacked LoRAs over-saturate - the commonest failure);
  most_typical / uniqueness is NOT a quality signal here.

## 4. Order of work
1. `stacks` mode in SweepPlan + V2 adapter (Eric_Krea2; append-only, backups to docs/backup).
2. `stacks-import` (PNG chunk + .txt reader) and the `stacks` table in the catalog.
3. `sweep-plan` generator (family rules move into the catalog so both sides share them).
4. Suggester: pair bonus from your verdicts > harvested pairs.

## 5. Decisions (Eric, 2026-10-09)
1. First tests: Krea2 **graphic-art** and **art-style** - 2 LoRAs within each folder, and 1 graphic-art + 1
   art-style. Later: 1 graphic-art + 1 surreal. (graphic-art is due to be split into illustration / comic /
   graphic-art first - LoRA Catalog TODO A.3 - so the families are clean.)
2. Judging: by eye first; UniPercept / ArtiMuse pre-ranking maybe later.
3. Verdicts: **click-to-mark** on the contact sheet (good / bad / unsure per cell), written straight into the catalog's
   stacks table. Replaces the copy-to-folder route in 2.3 as the main path; the folders stay for renders made outside
   a sweep. Sketch: a small page served by the catalog's editor server, reading the sweep dir's manifest + cell PNGs;
   keyboard G / B / U per cell; each mark stores the cell's stack (from its PNG chunk), prompt, seed and checkpoint.
