# Spec: checkpoint sweeps (one folder, one report)

Status: BUILT 2026-11-08 (compile-checked, core logic tested offline; not yet live-tested in ComfyUI).
Eric's answers: both sheet kinds; uniqueness within AND across checkpoints (uniq_ckpt); krea2 models
folder only; pick-list order. Deferred: Auto Pick 'best cell per checkpoint'; transformer-only swap;
the Sweep Plan cell counter does not multiply by the checkpoint count.

## Goal

Run one sweep across several Krea 2 Turbo checkpoints and get **one sweep folder, one contact sheet,
one spreadsheet**, so checkpoints can be compared directly. Today each checkpoint is queued as its own
run, which gives one folder per checkpoint and makes comparison hard.

## Decisions already made (Eric, 2026-11-08)

- Checkpoints are picked from a **dropdown with check boxes**, not typed.
- **Turbo only.** All checkpoints run on the loader's Turbo base folder; Raw and mixed bases are out.
- After the sweep, **all models are unloaded**, so the next queue loads fresh.
- Build it as a node that feeds the existing Sweep Plan. It must stay usable by a future Sweep Plan V2
  (not started - only discussed on 2026-11-03; no spec exists yet).

## What exists in the code (read 2026-11-08)

- The Component Loader already lists checkpoints for its `transformer_pick` dropdown:
  `_list_models("diffusion_models", "krea2")` -> every single file under `models/diffusion_models/krea2/`.
- The loader's output handle carries `loader_settings` (base path, transformer path, TE, VAE, precision,
  attention, compute ...), so Ultra V2 can rebuild the exact load with only the transformer changed.
- The loader's `_override_transformer` handles single files (incl. INT8-ConvRot / fp8 / MXFP8 / NVFP4
  dequant), GGUF and diffusers folders.
- The Unload node's `_teardown_pipe` + cache clearing is the proven way to actually free VRAM (ComfyUI's
  node-output cache otherwise pins the old pipeline).
- `_sweep_core.run_sweep` takes one `checkpoint_name`; lead axes (prompt -> profile -> LoRA) already
  expand slowest-first and cross with grid and combos modes.

## Design

### 1. New node: Eric Krea2 Sweep Checkpoints

- One widget that looks like a dropdown: **"Checkpoints: 5 of 20 v"**. Clicking opens a checklist of
  every file the loader can see under `models/diffusion_models/krea2/`, with check boxes, a filter box,
  and Select all / None.
- The ticked names are stored in a hidden STRING widget (one name per line), so they save with the
  workflow, survive reloads, and work headless.
- Output: `KREA2_CHECKPOINTS` (an ordered list of names). Optional `enabled` toggle (off = output nothing,
  the sweep runs on the loader's checkpoint as today).
- A name that no longer exists on disk is shown greyed out in the list and skipped at run time with a
  console warning (same rule as unknown schedule names).

### 2. Sweep Plan: new optional input `checkpoints`

- An optional **input socket** (not a widget), so no saved widget values move.
- Checkpoint becomes the **outermost axis**: checkpoint -> prompt -> sigma profile -> LoRA -> grid or
  combos. Every checkpoint gets the identical cell list. The cell counter shows `N checkpoints x M cells`.
- Combos lines may also name `"checkpoint": "<name>"`; such lines run only on that checkpoint, and the
  plan groups lines by checkpoint so a checkpoint is never loaded twice.

### 3. Ultra V2: running the groups

For each checkpoint, in order:
1. If it is the checkpoint already loaded, use it as is (it is put first in the order).
2. Otherwise tear down the current pipeline (frees VRAM - only one transformer resident on the
   RTX 5000), then call the Component Loader's own `load()` with the original `loader_settings` and
   `transformer_path` set to this checkpoint. Reusing the loader's code path means every format,
   quantization, attention and compute setting behaves exactly as when loading by hand.
3. Run that checkpoint's cells. LoRAs already load and unload per cell, so LoRA axes work on every
   checkpoint.
4. A checkpoint that fails to load marks its cells failed (red tiles with the error); the sweep goes on.

After the last group, or on interrupt (finalize first, as now): tear down the pipeline, clear the
loader cache and ComfyUI's models - **everything unloaded**.

Before the first load, check the base folder's `is_distilled` flag and refuse with a clear message if it
is not Turbo.

### 4. One folder, one report

- Folder: `<timestamp>_<sweep name>_<N>ckpts`.
- Every cell's manifest entry, CSV row and PNG gets the checkpoint. The PNG's `krea2_settings` loader
  section records the cell's real transformer path, so any cell reproduces on its own.
- Contact sheet: one sheet with a header band per checkpoint and its cells in a block underneath
  (captions show a short checkpoint name). Optionally also one sheet per checkpoint.
- Spreadsheet: a `checkpoint` column, so your rating columns can be sorted and filtered by checkpoint.
- Stage 1 reuse, uniqueness and Auto Pick work per checkpoint; Auto Pick gains "best cell per
  checkpoint" in its report.

### 5. Cost

About 30 s per checkpoint load plus dequantization for quantized files. Version 1 reloads the text
encoder and VAE with each checkpoint (simplest, uses the loader unchanged). A later optimisation can
swap only the transformer and keep the text encoder and VAE resident.

## Questions for Eric

1. **Contact sheet:** one combined sheet with a block per checkpoint, per-checkpoint sheets as well, or both?
2. **Uniqueness across checkpoints:** compare cells within each checkpoint only, or across checkpoints
   too (which would flag the checkpoint that renders a prompt most differently from the rest)?
3. **Sources:** only files in `models/diffusion_models/krea2/` (what the loader's dropdown shows), or
   also a box for pasted paths (diffusers folders, GGUF elsewhere)?
4. **Order:** ticked order, alphabetical, or loaded checkpoint first then alphabetical?

## Out of scope

Raw and mixed bases; other model families; the Sweep Plan V2 layout rework (separate, still to plan).