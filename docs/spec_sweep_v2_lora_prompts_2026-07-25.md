# Spec: Sweep v2 - LoRA + prompt axes (2026-07-25)
Status: APPROVED IN PRINCIPLE (Eric: "implement all 3 modes at one time").
Three semantic defaults chosen below are flagged ⚑ for veto before build.

## Goal
Extend the sweep system with the axes that matter most for Eric's library:
LoRA bake-offs across his style-LoRA collection, strength ladders on the
current stack, and prompt-list matrices - all crossable with the existing
sampler/scheduler/steps/eta/mu/cfg/megapixels axes under the same cap,
manifest, CSV, sheet, and preset machinery.

## Modes (all on the existing EricKrea2SweepPlan node; widgets APPENDED at the
## end of optional for position safety)

New widgets:
- `lora_mode`: `none` | `bakeoff` | `strength_ladder` (default none)
- `lora_files` (multiline): bake-off list, one LoRA per line - path or name
  resolved against the ComfyUI loras folder the same way the stack node does
- `lora_strengths`: e.g. `0.6, 0.8, 1.0` - bakeoff: each file at each
  strength; ladder: the strength values for the target entry
- `lora_target`: ladder mode - 1-based index into the panel's declared stack
  (`1` default) or a name substring match
- `include_baseline` (BOOLEAN, default true): add one no-LoRA (bakeoff) /
  panel-strength (ladder) reference cell
- `keep_panel_stack` (BOOLEAN, default false): bakeoff only - see ⚑1
- `prompts` (multiline): one prompt per line; blank = not swept

### Mode semantics
1. **bakeoff**: axis = lora_files x lora_strengths. ⚑1 DEFAULT: the swept
   LoRA runs ALONE (panel stack replaced) so every style is measured against
   the bare checkpoint - comparability first. `keep_panel_stack=true` instead
   APPENDS the swept LoRA on top of the declared stack (for "which style
   layers over my base look" questions).
2. **strength_ladder**: the panel-declared stack is kept; only the target
   entry's strength steps through lora_strengths. Multiple entries -> run the
   ladder per target via combos-file lines.
3. **prompts**: independent axis, crossable with everything. Each prompt cell
   sets prompt=<line> AND prompt_conditioning=None so the engine re-encodes
   (~seconds/cell). ⚑2 DEFAULT: prompts expand as the SLOWEST axis so the
   contact sheet and CSV read in prompt-major blocks; metric comparisons are
   only meaningful within a prompt block and the manifest tags each cell's
   prompt for downstream grouping.

## Plan/engine representation
Cells carry reserved non-widget keys: `lora_file`, `lora_strength`,
`lora_target`, `prompt`. They are visible everywhere a human looks (labels,
filenames, CSV swept columns, manifest overrides) because they are not
underscore-prefixed; the V2 adapter POPS them before calling the engine (they
are not widget kwargs). ⚑3 DEFAULT: they also make a cell S1-VARIANT
automatically (not s2_/s3_-prefixed), so S1 latent reuse correctly sits out for
LoRA/prompt cells - no exceptions needed.

## Adapter work (V2._run_sweep)
- Read the stack-entry schema from nodes/krea2_lora_stack.py at build time
  (implementation step 1: read that source; never guess the dict keys).
- Per cell: shallow-copy the pipeline dict, deep-copy lora_stack, apply
  bakeoff/ladder patch, pass the clone. The per-cell realize+teardown the
  engine already does per combo is exactly the required mechanism (~2s/cell
  fuse cost, accepted since v1).
- Per-cell metadata: the `lora` section of the settings chunk becomes
  per-cell (current-stack holder written by run_one, read by
  settings_string - sequential within one iteration, documented).
- `prompt` cells: recorded in manifest + chunk (`prompt` field), so a cell
  PNG remains fully reproducible standalone.

## Interactions & guards
- Cap unchanged (64 default): 12 LoRAs x 3 strengths x 2 prompts = 72 ->
  refusal with axis breakdown; raising the cap stays a deliberate act.
- quick_screen composes freely (LoRA screening at 1MP S1-only is the
  intended cheap first pass for a large library).
- SweepToPreset: promotes ultra fields as today; lora/prompt land in
  _sweep_meta provenance (presets do not apply LoRAs - the stack is declared
  upstream by design).
- Uniqueness metric (planned, separate task): pairwise distance grouped
  per prompt block.

## Open item logged separately
Euler S2/S3 parity audit vs diffusers pipe path (sigma window / timestep
encoding / guidance formula / dtype) - Eric observed quality loss post
2026-07-22 unification; affects all samplers if real. Own task.
