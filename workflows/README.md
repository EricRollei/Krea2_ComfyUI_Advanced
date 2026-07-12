# Workflows

Ready-to-load ComfyUI workflow JSON files for Eric_Krea2.

Drag any `.json` here onto the ComfyUI canvas (or use **Workflow → Open**) to
load it. Suggested files to add before publishing:

- `krea2_turbo_fast_4k.json` - Loader (Turbo) → Generate → Upscale Decode 2×.
- `krea2_raw_multistage.json` - Loader (Raw) → Multi-Stage Ultra (3 stages).
- `krea2_lora_stack.json` - Apply-LoRA chain → Multi-Stage Ultra V2.
- `krea2_presets.json` - Multi-Stage Ultra V2 showing the ★ Save Preset flow.

Keep these lightweight (no embedded weights). Reference models by their
standard ComfyUI paths so they load on any install.
