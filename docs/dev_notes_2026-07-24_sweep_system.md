# Dev note: Sweep system implementation (2026-07-24)

Spec: docs/spec_sweep_system_2026-07-24_v2_final.md (approved). This note covers
what was built, how it hangs together, and the deliberate trade-offs.

## Files
- `_sweep_core.py` (NEW, package root) - the portable engine. Zero Krea2
  imports; talks to the host through three callables (run_one /
  resolve_values / settings_string) plus optional interrupt/progress hooks.
  THIS is the template for porting sweeps to qwen or any other node set:
  reuse the file verbatim, write a new plan node + a ~60-line adapter in the
  host node.
- `nodes/krea2_sweep.py` (NEW) - `EricKrea2SweepPlan` (grid + JSON-lines combo
  modes, stage scoping, validation, hard cap refusal) and
  `EricKrea2SweepToPreset` (manifest cell -> ultra_presets.json, checkpoint-
  tagged, atomic write).
- `nodes/krea2_multistage_ultra_v2.py` (MODIFIED, +126 lines) - `sweep` input
  socket appended LAST in optional (connection type, no widget value ->
  saved-workflow positions untouched); `sweep_sheet` + `sweep_manifest`
  outputs appended AFTER `settings` (all existing output indices unchanged);
  `_run_sweep()` adapter. Pre-edit backup:
  docs/backup/krea2_multistage_ultra_v2_pre_sweep_2026-07-24.py
- `__init__.py` (MODIFIED) - registers `krea2_sweep`.
- Base `krea2_multistage_ultra.py`: UNTOUCHED (shipping-node rule).

## Architecture decisions
1. **Loop calls the inherited `generate()` per combo** (full wrapper incl.
   LoRA realize + ephemeral clear, rebalance install, ref-latents install,
   finally-teardown). Correctness by construction - a sweep cell is
   byte-for-byte the same code path as queueing runs by hand, and interrupts
   tear down cleanly mid-cell. Cost: the per-run setup (~seconds when a LoRA
   stack is declared) repeats per cell. For full-pipeline cells (minutes)
   it's noise; for 10s quick-screen cells it's ~10-20% overhead. If that ever
   matters, the v2 optimization is a base-class refactor exposing the
   install/teardown as a context manager entered once around the loop -
   deliberately NOT done now to keep the shipping engine untouched.
2. **Defaults + recipe keys derived live** from `_generate_inner`'s signature
   (inspect) and `ultra_setting_keys()` - no hardcoded field lists to drift.
3. **Interrupt = finalize-then-reraise.** Stop button: partial cells + contact
   sheet + manifest (marked `"incomplete": true`) are already on disk; the
   re-raise lets ComfyUI cancel the queue item normally (node outputs are
   discarded by Comfy on interrupt - the disk artifacts are the product).
4. **Failed cells never kill the sweep**: red tile on the sheet + `"error"`
   in the manifest; the IMAGE batch output contains successful cells only.
5. **Metrics are classical + CPU-only** (numpy, ms per cell, no model loads):
   variance-of-Laplacian + Tenengrad (the pair disagreeing flags
   oversharpening), Haar-HH median/0.6745 noise sigma computed at NATIVE res
   (decimation would alias it), flat-region chroma variance (the
   splotch/turbo-residue detector), clip fraction. Scores annotate and can
   sort; they never hide cells.
6. **Checkpoint naming**: pipeline dict currently guarantees only
   `model_path`; `transformer_path` (the actual fine-tune identity when using
   the component loader's override) lives in loader_settings, not the dict.
   `_run_sweep` tries transformer_path/model_name/model_path in order.
   FUTURE (one line in component loader): add `"transformer_path"` to the
   returned pipeline dict so sweep dirs/preset tags name the fine tune rather
   than the base pipeline folder.

## Behaviour notes
- Every cell PNG carries the full resolved recipe in a `krea2_settings` text
  chunk (same schema as presets / Settings From Image reads) + a `sweep_cell`
  chunk with just the overrides. Any single cell file is reproducible and
  promotable on its own.
- quick_screen forces `upscale_to_stage2/3 = 0` and `s1_megapixels =
  screen_megapixels` per cell; the manifest records `"quick_screen": true` so
  promoted presets are honest about provenance.
- seeds: `fixed` policy uses the panel seed everywhere (fair comparison);
  `seeds_axis` adds seeds as one more grid axis via a `_seed` pseudo-key.
- lcm-family x colored-noise cells are coerced to white AND labeled (plan
  node), per the 2026-07-23 rainbow fix.
- SweepToPreset stores `_sweep_meta` (checkpoint, objective, sweep id,
  metrics, time) inside the preset's ultra dict - V2's apply loop filters to
  valid widget keys, so the meta rides along harmlessly as provenance.
- Line endings: nodes/krea2_multistage_ultra_v2.py was CRLF and is now LF
  (MCP write). Python is indifferent; if git flags a whole-file EOL diff,
  autocrlf or one VSCode re-save with CRLF restores it.

## Testing done (container, stubbed engine - NOT yet run inside ComfyUI)
- Engine: expansion, per-cell failure isolation, interrupt finalize path,
  metric computation + sorting, PNG chunk round-trip, contact sheet render.
- Plan node: axis mapping per scope, all_stages mu -> distilled_shift=manual
  forcing, window parsing, pair_lock mismatch guard, sampler/scheduler/noise
  validation, over-cap refusal with axis breakdown, combos JSON-lines, lcm
  noise coercion.
- V2 patch: input appended last / outputs appended after settings, sweep run
  end-to-end with stub base (batch stacking, quick_screen + fixed-seed
  reaching the engine, checkpoint naming from model_path, manifest contents),
  non-sweep path returns the 9-tuple with legacy indices intact.
- SweepToPreset: promotion shape, checkpoint tag, failed-cell guard.

## First live test suggestion
Restart ComfyUI, wire Sweep Plan -> Ultra V2 `sweep`, quick_screen ON at 1MP,
samplers `euler, deis_3m, res_2m, lcm_hybrid` x schedulers `linear, beta57`
(8 cells, ~1-2 min), sort_by `noise_sigma`. Check output/sweeps/<id>/ and the
sweep_sheet output, then promote one cell with Sweep -> Preset and load it
from the ultra_preset dropdown after a graph reload.

## v1.1 (same day, post-live-load)
Changes after Eric's first look, all deployed + compile-verified with the
embedded python:
1. **Blank grid defaults** - samplers/schedulers now default empty. Rule is
   uniform: blank axis = not swept (panel value used). Sweeping only eta no
   longer requires clearing demo values first.
2. **combos_file** on SweepPlan - path to a JSON file of runs; accepts a JSON
   array of override objects, {"combos": [...]}, or JSON-lines. Takes
   precedence over the combos_json textbox. Sweep campaigns are now
   versionable files.
3. **S1 latent reuse** (the one deliberate base-engine edit, diagnosed +
   approved): `_generate_inner` gained a private `_s1_reuse=None` kwarg that
   skips the Stage 1 denoise and starts from a supplied post-S1 latent.
   The sweep driver (V2._run_sweep) captures `stage1_latent` from the first
   cell whose overrides leave S1 invariant and injects it into later
   invariant cells. Eligibility, all enforced per cell: override keys
   strictly s2_*/s3_* (seed ignored - constant under the required fixed
   policy), fixed seed_policy, quick_screen off, and seed_mode !=
   same_all_stages (S2 shares S1's generator OBJECT there; skipping S1 would
   leave it un-advanced and silently change S2's noise vs a full run - the
   engine double-guards this and falls back to a full S1 with a console
   note). Manifest records `"s1_reused": true|false` per cell. Cells with
   any global key (distilled_shift, upscale_to_stage2, shift_mu_s1, cfgs
   via s1, seeds axis) automatically run full - mixed plans partially
   benefit. Expected saving on scope-s2/s3 sweeps: roughly S1's share of the
   pipeline (~15-25%) on every cell after the first.
   Engine backup: docs/backup/krea2_multistage_ultra_pre_s1reuse_2026-07-24.py
4. `_sweep_core.run_one` may now return `(image, extra_dict)`; extras merge
   into that cell's manifest record (how s1_reused travels without the core
   knowing Krea2 semantics - stays portable).
5. Deploy note: files written by the MCP write tool carry CRLF on Windows;
   the PowerShell patcher converts pattern EOLs per target file. Base ultra
   remains LF, sweep-era files are CRLF - cosmetic only.
