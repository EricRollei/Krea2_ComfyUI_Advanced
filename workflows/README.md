# Workflows

Ready-to-load ComfyUI graphs for Eric_Krea2. ComfyUI embeds the full node graph
in each PNG's metadata, so you don't need a separate `.json`.

## How to load

**Drag** `example-workflow.png` onto the ComfyUI canvas (or **Workflow → Open**
and pick the PNG). The graph loads with every node and wire in place.

## Files

- **`example-workflow.png`** - the main, multi-purpose graph. It wires the whole
  pipeline into one canvas so you flip parts on/off with group bypass instead of
  loading different workflows:
  - **Loaders** - Component Loader (Raw/Turbo) + Upscale/Decode VAE loaders +
    Multi-LoRA Stack.
  - **txt2img** - prompt → Multi-Stage Ultra V2 → Upscale Decode.
  - **img2img** - Load Image → Krea2 VAE Encode → S1 `init_latent` (bypass the
    group for pure txt2img).
  - **Vision Prompt** - reference image → Qwen3-VL vision path →
    `KREA2_CONDITIONING`.
  - **Sampling** - optional Sigmas node feeding per-stage schedules.
  - **Save** - Merge Settings → metadata-aware Save Image (recipe embedded in the
    PNG, readable back via Settings from Image).
- **`workflow.png`** - a trimmed reference graph / thumbnail of the layout.

## Notes

- The graphs reference models by their standard ComfyUI paths (no embedded
  weights), so they load on any install - just point the loader widgets at your
  local model folders.
- Bypass (Ctrl+B) the img2img and Vision groups for a plain txt2img run;
  re-enable them to add image conditioning.
