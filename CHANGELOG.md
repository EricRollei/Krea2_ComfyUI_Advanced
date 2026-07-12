# Changelog

All notable changes to Eric_Krea2 are documented here. Format loosely based on
[Keep a Changelog](https://keepachangelog.com/).

## Unreleased

### Added - INT8 ConvRot checkpoint support (component loader)
- **INT8 / INT8-ConvRot dequantization** - the component loader's single-file
  paths (transformer and text encoder) now load native-ComfyUI INT8 checkpoints,
  including the **ConvRot** variant that is becoming the community-standard
  8-bit format (natively supported by ComfyUI since v0.27.0, quality ≈ GGUF Q8).
  `_dequant_comfy_fp8` grew into `_dequant_comfy_quant` (old name aliased):
  per-row `[out, 1]` and tensorwise scalar `weight_scale` tensors are both
  applied, and layers flagged `convrot` get the group-wise **regular-Hadamard
  un-rotation** after the scale multiply. The rotation matrix (Kronecker powers
  of the 4x4 regular Hadamard, / sqrt(n)) is symmetric-orthogonal - its own
  inverse - mirroring `comfy_kitchen.tensor.int8_utils` exactly; verified
  round-trip recovery corr 0.99997 vs 0.06 without it (the "pure noise" a
  ConvRot file previously decoded to). Runs on GPU in fp32 alongside the fp8
  dequant.
- **Per-layer quant-config discovery** - which layers are rotated is read from
  BOTH the file-level safetensors `__metadata__._quantization_metadata` (parsed
  from the raw header, since `safetensors.load_file` drops it) and per-layer
  `<module>.comfy_quant` marker tensors, including markers mis-saved as bf16
  byte values by some converters. Names are prefix-stripped
  (`model.diffusion_model.` etc.) so lookups line up with either key style; the
  nested `params` fallback the native ComfyUI loader accepts is honoured too.
- **Loud failure on broken quants** - an INT8 weight with no companion
  `weight_scale`, or a ConvRot layer whose width is incompatible with the group
  size, now logs a clear warning instead of silently loading garbage.

### Changed - ~10x faster component-loader model init
- **Meta-device instantiation** - the single-file transformer / text-encoder /
  VAE overrides (and the GGUF path in `_gguf_utils`) now build the model
  skeleton under accelerate's `init_empty_weights(include_buffers=False)` and
  load with `load_state_dict(assign=True)`. Previously `from_config` /
  `Qwen3VLModel(cfg)` ran a full random-weight initialization on CPU in fp32 -
  kaiming / trunc-normal passes over every tensor that `load_state_dict`
  immediately overwrote - costing ~50s for the 12.8B transformer and ~215s for
  the 4B Qwen3-VL TE (≈90% of a ~298s cold load, with the CPU/RAM churn to
  match). Params now start on the meta device (zero alloc, zero init) while
  buffers stay real so computed values (rotary tables) remain correct; cold
  component loads drop to roughly plain-loader speed (~30s). Falls back to the
  old full init if accelerate is missing.
- **Missing-key guard** - with meta init a key absent from the checkpoint is
  left as a meta tensor (crash or garbage at inference) rather than "randomly
  initialized", so the loader now verifies every parameter was materialized and
  raises with the offending names instead of limping on.
- **Dropped a redundant full-state-dict dtype cast** after the transformer key
  remap (dequant already emits target-dtype tensors); saves a transient
  ~25 GB RAM copy on the 12.8B model.

### Added - upscale-VAE downsample modes & clarity
- **Per-stage `eta`** - the single global `eta` widget on the Multi-Stage Ultra
  node is replaced by `s1_eta` / `s2_eta` / `s3_eta` (each 0-1, default 0.1) so
  SDE churn can be tuned independently per stage. Legacy graphs with a saved
  `eta` still load (it fans out to all three).
- **`upscale_vae_mode` downsample variants** - the dropdown gains
  `s2-s3`, `s2-s3 with downsample`, `final decode`, `final decode with
  downsample`, `both`, `both with downsample`, and `both with final decode
  downsample` (old `inter_stage` / `final_decode` still load as aliases). The
  "with downsample" S2->S3 modes keep the trained 2x VAE's learned detail but
  resample down to your `upscale_to_stage3` factor instead of forcing 4x area;
  `upscale_between_stages` takes an optional `target_h/target_w` (GPU-native
  bicubic+antialias) to do this. The final-decode downsample routes through
  `decode_latents_with_upscale_vae`'s new `downsample_override`.
- **Resolution-chain readout & forced-field labels** - a new
  `POST /eric_krea2/resolution_chain` endpoint realizes the real
  S1->S2->S3->decode sizes with the same math `generate()` runs, drawn under the
  Ultra node (`krea2_upscale_hints.js`). The same helper relabels
  `upscale_to_stage2/3` when a connected VAE overrides them (e.g.
  "[VAE 2x -> 4x area, ignored]"), exposing the cascade footgun where
  `both` + `s1_s2_upscale_vae` silently balloons the final size.

### Changed
- `upscale_to_stage2/3` tooltips now spell out that they are **area** (megapixel)
  factors and note exactly when a connected upscale VAE ignores or respects them.

## 2026-07 - Multi-stage sampling, presets & img2img toolkit

A large development cycle that turned the pack from a text-to-image generator into
a full production toolkit: img2img, a reusable preset/settings system, a shared
scheduler library, extra RES/DEIS samplers, and resolution/latent helpers.

### Added - samplers (custom flow-matching solvers, `_res_solver.py`)
- **`res_2m`** - RES 2nd-order exponential *multistep* solver, re-implemented from
  the exponential-integrator math (arXiv:2308.02157) in data-prediction form.
  Same model-call cost as Euler but sharper, and it honours a *noisy early-stop*
  (`end_step < steps`) so a stage can hand a still-noisy latent to the next.
- **`res_2s`** - RES 2nd-order *single-step* exponential Runge-Kutta (midpoint
  predictor + corrector, 2 model evals/step). Self-starting, so it is the
  strongest choice at very low step counts. *Why:* the RES4LYF community reports
  `res_2s` + a `beta`/`beta57` schedule as the best Stage-1 draft combo.
- **`deis_3m` / `deis_4m`** - DEIS 3rd/4th-order *multistep* (arXiv:2204.13902),
  specialised to rectified flow as an **exponential integrator in $t=-\log\sigma$**:
  the flow ODE is `dx/dt = x0 - x`, so each step is `x_{i+1} = e^{-h} x_i + ∫ e^{τ-h}
  P(τ) dτ` with `P` the Lagrange fit of the data prediction through the last
  `max_order` steps (1 model eval/step, same cost as Euler, very smooth). *Why:*
  reported best for the Stage-2 refine paired with a `bong_tangent` schedule. The
  bounded `e^{τ-h}` kernel keeps the extrapolation stable on non-uniform schedules
  (an earlier raw-σ velocity-integration form left residual noise on karras/beta/
  bong_tangent curves and was replaced).
- All four solvers are model-decoupled (take a `denoise_fn` closure), support the
  ancestral **`eta`** SDE path, fall back to a clean flow-Euler at the terminal
  σ→0 step, and ship with a CPU self-test (`python _res_solver.py`) that verifies
  constant-x0 convergence to ~machine-epsilon. No RES4LYF/k-diffusion code is
  imported - the algorithms are recreated from the papers.

### Added - schedules (shared curve library, `_sigmas.py`)
- Central `build_sigmas(...)` used by both the sampler nodes and the new Sigmas
  node, exposing curves: `linear`, `balanced`, `karras`, `beta57`, `beta`,
  `bong_tangent`, `exponential`. *Why:* one source of truth so a schedule looks
  identical whether picked on the stage or driven from the Sigmas node.
- **`EricKrea2Sigmas`** node - per-stage schedule overrides (S1/S2/S3) with a
  `detail_bias` control (shifts sigma spacing toward fine detail vs. structure)
  and Beta α/β shape knobs, emitted as a `KREA2_SIGMAS` bundle that the Ultra
  nodes fold into their recipe.
- **Live sigma-curve preview** on the Sigmas node (web JS + `POST
  /eric_krea2/preview_sigmas`): draws all three stage curves (y = noise level,
  x = step) realized by the same `build_sigmas` used at sample time, so the
  abstract knobs become a picture; disabled stages are dimmed/dashed.

### Added - img2img
- **`EricKrea2VAEEncode`** - encodes an `IMAGE` into a `KREA2_LATENT` (the img2img
  primitive), matching the packed 16-channel f8 latent format.
- Stage 1 of the Ultra node now accepts an optional **`init_latent`**: it is
  re-noised at `s1_start_step` and denoised over `[s1_start_step, s1_end_step]`,
  so denoise-strength is set by the step window (no separate float). `init_match_size`
  preserves the source resolution/aspect.

### Added - resolution & latent helpers
- **`EricKrea2Resolution`** - pick a size once from `aspect_ratio` + `megapixels`
  (mirrors the Ultra node's math) *or* derive width/height from a source `image`
  for img2img. Outputs `width`/`height`/`dims` (to drive the Ultra node's new
  `width`/`height` inputs) plus a blank `KREA2_LATENT` canvas (noise or zeros).
- **`EricKrea2LatentResize`** - resize a `KREA2_LATENT` in latent space (no VAE
  round-trip) via the norm-preserving bislerp interpolation used between stages;
  `scale` / `dimensions` / `megapixels` modes, up or down.
- The Ultra base node gained optional **`width`/`height`** inputs (0 = derive from
  aspect+MP; both > 0 override Stage 1; `init_match_size` still wins). *Why:* lets
  the Resolution node size Stage 1 exactly, including to a source image.

### Added - presets & settings system
- Generalised preset backend (`server.py`, `_settings.py`) with named JSON presets
  per **section** (`ultra`, `loader`, `lora`, `sigmas`), a ★ Save Preset button
  (web JS), and a `<section>_presets.json` store. *Why:* reusable, shareable
  configs without retyping widget values.
- **`EricKrea2ComponentLoader`** and **`EricKrea2MultiLoRA`** gained preset
  dropdowns; the LoRA stack applies presets headlessly.
- **`EricKrea2MergeSettings`** - merges section recipe strings (dict sections
  updated, `lora` lists concatenated) into one `KREA2_SETTINGS`.
- **`EricKrea2SettingsFromImage`** - reads an image's embedded generation settings
  back into a `KREA2_SETTINGS` recipe for one-click reproduction.
- **`EricKrea2MultistageUltraV2`** - preset-driven variant of the Ultra node that
  serialises its widgets (including `sigmas`, `width`/`height`) into a single
  reproducible recipe.

### Added - LoRA & utility nodes
- **`EricKrea2ApplyLoRA` / `EricKrea2UnloadLoRA` / `EricKrea2DiagnoseLoRA`** -
  apply, cleanly remove, and inspect LoRA adapters on the cached pipeline.
- **`EricKrea2UnloadModels`** - free VRAM on demand.
- **`EricKrea2SaveLatentDebug`** and **`EricKrea2LatentToComfy`** - debug/save a
  packed latent and bridge `KREA2_LATENT` → ComfyUI `LATENT`.

### Changed
- The Ultra node's per-stage `*_sampler` widgets now list all five solvers with
  tooltips documenting the recommended pairings (`res_2s`+`beta57` for S1;
  `deis_3m`/`deis_4m`+`bong_tangent` for S2/S3).

### Notes
- `res_2s` runs 2 model evals per step, so that stage takes ~2x the wall-clock the
  step-count progress bar implies (cosmetic only).
- DEIS with `eta > 0` uses the standard ancestral approximation (integrate the
  deterministic part to `sigma_down`, then inject `sigma_up` noise).

## 2026-06 - Initial release

### Added
- **`EricKrea2Loader`** - loads `Krea2Pipeline` (Raw or Turbo) via
  `from_pretrained`, auto-detects `is_distilled` from `model_index.json`, applies
  a selectable `attention_backend` (`auto`/`flash`/`sage`/`sdpa`, with safe
  fallback to SDPA), and caches the pipeline in VRAM.
- **`EricKrea2Generate`** - text-to-image; emits `IMAGE` and a chainable
  `KREA2_LATENT` from a single `output_type="latent"` pass. `auto_settings`
  selects 8 steps / guidance 0 for Turbo and 28 / 4.5 for Raw.
- **`EricKrea2Multistage`** - up to 3-stage progressive high-res (draft → bislerp
  latent upscale → partial re-noise → re-denoise). Resolution-aware via the
  flow-matching scheduler's dynamic shift; `mu` fixed to 1.15 on distilled
  checkpoints. Intended for the Raw checkpoint; warns on Turbo.
- **`EricKrea2UpscaleVAELoader` / `EricKrea2Decode`** - the spacepxl Wan 2×
  upscale decoder applied to Krea 2 latents (shared `AutoencoderKLQwenImage`
  latent space), with tiled decode + cosine tile-seam blending for large outputs.
- **`EricKrea2VAEDecode`** - plain 1× decode of a `KREA2_LATENT`.
- **`EricKrea2Prompt`** - assembles structured fields into a dense, comma-joined
  natural-language prompt; auto-quotes text-to-render.
- **`EricKrea2MagicPrompt`** - LLM prompt expander against any OpenAI-compatible
  endpoint (LM Studio / Ollama), with model auto-detection and support for
  loading Krea's `expansion.txt` as the system prompt.

### Notes
- Self-contained: the latent pack/unpack, bislerp upscale, flow-match re-noise,
  sigma-shift math, and upscale-decode helpers are bundled, so the pack has no
  dependency on Eric_Qwen_Edit or any other node pack.
- Requires a diffusers build that includes `Krea2Pipeline` (later than
  `0.39.0.dev0`); install from source with `--force-reinstall` since the version
  string is unchanged.
