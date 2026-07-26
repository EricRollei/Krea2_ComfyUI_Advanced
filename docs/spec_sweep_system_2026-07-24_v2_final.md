# Spec: Krea2 Sweep System - v2 FINAL (approved 2026-07-24)
Supersedes spec_sweep_system_2026-07-24.md draft. Eric's answers incorporated.

## Decisions
1. Sweep loop lives inside Ultra V2 via optional `sweep` input (appended at end
   of optional inputs, positionally safe). PORTABILITY REQUIREMENT: the engine
   must be reusable as a template for other node sets (qwen, etc.) -
   therefore split:
   - `sweep_core.py` - package-agnostic engine: plan parsing/expansion,
     combo iteration harness, manifest writer, contact-sheet renderer,
     metric scoring, interrupt/failure handling. No Krea2 imports. Talks to
     the host node through a small adapter protocol:
       adapter.apply_overrides(base_settings, combo) -> settings
       adapter.run(settings, seed) -> image tensor (+ telemetry dict)
       adapter.describe(settings) -> dict for manifest/metadata
   - `sweep_nodes.py` - Krea2-facing nodes (SweepPlan, SweepToPreset) +
     the Krea2 adapter. Porting to another package = new adapter + key map.
2. `quick_screen` toggle in v1: forces S1-only at reduced resolution
   (screen_megapixels widget, default 1.0) for cheap wide grids; workflow is
   perfect S1 first, then short-list full-pipeline passes. Sweeps serve
   multiple objectives (fastest clean run / highest quality / per-finetune
   presets), so the manifest records per-cell wall time alongside quality
   metrics, and sheet/manifest sorting can use ANY recorded column, including
   time. Promoted presets take a free-text `objective` tag ("fast",
   "quality", finetune name, ...).
3. Scoring = lightweight technical metrics, scores only, no ArtiMuse, no
   descriptions, no model loads. Computed inline per cell (CPU, ms-scale):
   - sharpness: variance-of-Laplacian + Tenengrad (both recorded; they
     disagree usefully on oversharpened halos)
   - noise: wavelet-based sigma estimate (skimage.restoration.estimate_sigma
     if available, else a robust MAD-of-HH1 fallback in numpy - no hard dep)
   - flat_region_chroma_var: chroma variance in low-gradient regions - the
     splotch/residue detector, directly relevant to fine tunes with turbo
     residue
   - clip_fraction: % pixels at 0/255 per channel (blowout/crush indicator)
   `sort_by`: plan_order | any metric | time. Metrics annotate the sheet and
   land in the manifest. They pre-sort; the eye judges.
4. Combos mode = JSON-lines with KREA2_SETTINGS keys (copy-pasteable from
   presets / PNG metadata). Per-stage prefixed keys allowed.
5. max_combos default 64, hard refusal over cap with axis-size breakdown.
   (Timing reality per Eric: 4.5MP 3-stage ≈ 3 min, so 64 full cells ≈ 3.2 h;
   1MP S1-only screening cells are seconds-to-tens-of-seconds - grids belong
   there.)
6. Output to `output/sweeps/<sweep_id>/` (timestamp + checkpoint short-name):
   metadata-stamped cell PNGs, labeled contact_sheet.png (captions show only
   diffs vs base), sweep_manifest.json.

## Unchanged from draft
- SweepPlan node: grid|combos, stage_scope s1|s2|s3|all_stages, pair_lock,
  axes = samplers, schedulers, steps, windows, etas, noise_types (lcm-family
  x colored-noise auto-coerced to white and marked), shift_modes/mus, cfgs;
  seed_policy fixed|seeds_axis.
- Execution: single queue item, deep-copied settings per combo, all existing
  Ultra machinery active, interrupt-safe between combos, per-combo failure
  isolation (error tile + manifest error), ProgressBar + [sweep i/N] banners.
- SweepToPreset: manifest path + cell_index + preset_name (+ objective tag),
  writes resolved settings into existing V2 preset library, auto-tagged with
  checkpoint name.
- Non-goals v1: prompt/LoRA-strength/rebalance/resolution axes (combos-mode
  JSON can already carry them if keys exist; official axes later), per-stage
  intermediate saves, any widget reorder.
