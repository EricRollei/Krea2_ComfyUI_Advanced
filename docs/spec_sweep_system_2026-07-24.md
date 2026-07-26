# Spec: Krea2 Sweep System ("Auto-Step") - DRAFT for review, no code yet
Date: 2026-07-24
Status: awaiting Eric's approval / edits

## Goal
Automatically step Ultra V2 through combinations of sampler / scheduler / steps /
refine-window / shift-mu / eta-noise settings against a fixed seed + prompt, so a
new fine tune can be characterized quickly, results compared side-by-side
(optionally ArtiMuse-scored), and winning combos promoted into the existing V2
preset library.

Motivation (from today's research): a fine tune's preferred sampler, scheduler,
step count, mu, and CFG tolerance are all proxies for one variable - how far its
training eroded the turbo distillation (trajectory straightness + sigma-band
coverage). A sweep is therefore a *characterization instrument*, not just a
convenience.

## Architecture
Single queue item; no graph re-execution; model + LoRA state stays loaded.

### 1. New node: `EricKrea2SweepPlan`
Emits a `KREA2_SWEEP` plan object. No model access; pure config.

Widgets:
- `mode`: `grid` | `combos`
- `stage_scope`: `s1` | `s2` | `s3` | `all_stages`
  - Swept keys apply to the selected stage(s); everything else comes from
    Ultra's live widget values (base config).
  - `steps/window` axis maps to full steps on s1, to (steps, start->end window)
    on s2/s3.
- Grid-mode multiline fields (comma/newline separated values; blank field =
  axis not swept, base value used):
  - `samplers`  e.g. `euler, deis_3m, lcm_hybrid, rk6_7s`
  - `schedulers` e.g. `linear, karras, beta, linear_quadratic`
  - `pair_lock` (bool): OFF = full samplers x schedulers cross;
    ON = zip line-by-line into fixed pairs (for testing known recipes).
  - `steps` e.g. `8, 10, 12`
  - `windows` e.g. `5-12, 6-12, 7-12` (s2/s3 scope only; ignored for s1)
  - `etas` e.g. `0, 0.5, 1.0`
  - `noise_types` e.g. `white, pink` (lcm-family combos force white per the
    Jul 23 fix - plan generator drops/marks colored-noise x lcm cells as
    auto-coerced rather than silently diverging)
  - `shift_modes` / `mus` e.g. `fixed, resolution` / `1.15, 2.1, 3.0`
  - `cfgs` e.g. `1.0, 1.5, 2.0` (for characterizing RAW-drifted fine tunes)
- Combos-mode multiline field: one combo per line, JSON object using the same
  keys as KREA2_SETTINGS / PNG metadata, so combos are copy-pasteable from
  existing presets and stamped images. Per-stage keys allowed
  (e.g. `{"s2_sampler": "lcm_hybrid", "s2_steps": 12, "s2_window": "6-12"}`).
- `seed_policy`: `fixed` (default; comparison requires it) | `seeds_axis`
  (extra `seeds` field becomes another grid axis)
- `max_combos` (int, default 64): hard cap. Plan node prints the computed count
  and *refuses* (raises) if over cap, listing the axis sizes so it's obvious
  what to trim. No silent truncation.

### 2. Ultra V2: optional `sweep` input (appended at END of optional inputs -
positionally safe, same pattern as the shift widgets)
When connected, `generate()` loops over the plan:
- Deep-copy resolved base settings, overlay combo overrides, run the normal
  pipeline (all existing machinery: LoRA stack, cond_rebalance, ref-latents,
  telemetry, renorm, VAE tiling prefs - untouched and active).
- Interrupt-safe: `processing_interrupted()` checked between combos; partial
  results still emitted, manifest marked `"complete": false`.
- Failure isolation: per-combo try/except; failed cell -> flat dark tile with
  error text on the sheet + `"error"` field in manifest; sweep continues.
- Progress: ComfyUI ProgressBar over combos + the existing per-stage console
  banner, prefixed `[sweep 7/24]`.
- Per-combo wall time + the existing telemetry deltas recorded in manifest.

### 3. Outputs & files
Directory: `output/sweeps/<sweep_id>/` where sweep_id = timestamp + checkpoint
short-name.
- `cell_000.png ...` - every cell saved through the existing PNG-metadata
  stamping (full resolved settings embedded -> any cell is reproducible and
  preset-promotable from the file alone).
- `contact_sheet.png` - labeled grid. Caption per tile: cell index + ONLY the
  params that differ from base (full settings live in per-cell metadata).
  Optional score annotation. `sort_by`: `plan_order` | `score`.
- `sweep_manifest.json` - base settings, per-cell override dict, resolved full
  settings, seed, checkpoint + LoRA stack info, timings, scores, errors.
Node outputs: IMAGE batch (all cells), IMAGE contact sheet, STRING manifest
path.

### 4. Scoring (optional toggle, default off)
- `score_with_artimuse` on Ultra's sweep group (or on the plan node - see open
  Q3).
- Runs at END of sweep: load ArtiMuse once (import-guarded against
  Eric-image-classification being absent -> fail silent to plan order, console
  note), score all cells, unload via existing unload infrastructure. No
  mid-sweep VRAM churn.
- Scores land in manifest + optional sheet annotation/sort. Score never deletes
  or hides a cell - the eye stays the judge; score is a pre-sort only.

### 5. Preset promotion: new node `EricKrea2SweepToPreset`
- Inputs: manifest path (STRING, wire from sweep output or paste), `cell_index`,
  `preset_name`, `sections` (which preset sections to write, default the swept
  keys only).
- Reads the cell's resolved settings from the manifest, writes into the existing
  V2 preset JSON library, auto-tagged with the checkpoint name (so presets are
  per-fine-tune, matching the whole premise).
- v1 is this small node; a JS "click a tile -> save preset" flow on the sheet is
  a later polish item.

## Non-goals (v1)
- Sweeping prompt text, LoRA strengths, cond_rebalance taps, resolution
  (all possible later via the same combos-mode JSON path; grid axes for them
  can come after the harness is proven).
- Per-stage intermediate image saving (later toggle).
- Any change to existing widget order or preset schema semantics.

## Open questions for Eric
1. Sweep loop location: inside Ultra's generate() via the optional input
   (recommended - reuses everything), or a separate driver node that imports
   and calls Ultra (keeps Ultra untouched but duplicates its input surface)?
2. Default screening recipe: add a `quick_screen` toggle that forces S1-only at
   reduced resolution for cheap first passes, then re-run short-list at full
   3-stage - worth having in v1?
3. ArtiMuse: which metric(s) do you want as the sort key, and is the
   load-once-at-end / unload pattern acceptable on the 6000?
4. Combos DSL: JSON-lines as spec'd, or do you also want the terse
   `key=value key=value` form parsed?
5. Cap default 64 okay? (3-stage 8MP runs at ~10-17 min make full-pipeline
   grids expensive; S1-only screening is where big grids belong.)
6. Any objection to `output/sweeps/<id>/` as the save location?
