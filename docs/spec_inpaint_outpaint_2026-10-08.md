# Spec - Inpaint / Outpaint for Multi-Stage Ultra (2026-10-08)

Status: DRAFT for Eric's review. Nothing built yet.

## 1. What exists in the community (read at source, 2026-10-08)

| Adapter | Base / run | Mechanism | Masks |
|---|---|---|---|
| **yijunwang2/krea2-anypaint** (rank 32, weights: Krea 2 Community License; pipeline: Apache-2.0) | trained on Raw, run on **Turbo, 8 steps, guidance 0** | registered reference + isolated kv_cache + per-step known-token restore | arbitrary (inpaint, outpaint, mixed, disconnected) |
| **yijunwang2/krea2-outpaint** (same author, older) | same | registered reference + kv_cache, pixel composite afterwards | rectangular outpaint only |
| **Cierpliwy/krea2-inpaint-edit** (default / mild / strong) | **Raw** | ostris Edit pack, kv_cache ON, region painted solid black in the reference | arbitrary |

The two yijunwang2 pipelines are byte-identical apart from the restore step, so **one runtime
covers both**. The source is at `A:\ai-tools\scratch\paint_adapters_src\`.

### AnyPaint mechanism, exactly as in its `pipeline.py` / `anypaint.py`
1. **Condition image:** the full canvas, with the source placed in its box and every
   pixel to be generated set to the **median colour of the known pixels**. Downscaled to a
   **384 px max edge** (snapped to multiples of 16), then VAE-encoded. This is one reference.
2. **Registered RoPE:** reference tokens sit on frame axis 1. Their h/w coordinates are
   mapped into the *target* grid: `y = y0*H + (i+0.5)*(y1-y0)*H/h - 0.5`. These are fractional
   coordinates, unlike our `ostris_t0` integer grid from 0. AnyPaint uses
   `bbox = [0,0,1,1]`, so the reference covers the whole canvas. Outpaint uses the source's box.
3. **Isolated kv_cache:** the reference tokens alone run through all 28 blocks once at
   **t = 0**, with no text and no target. Each block's post-RoPE K/V is captured. On every
   denoising step the target and text attend to `[own K/V + cached ref K/V]`. The reference
   tokens are not in the sequence, and nothing attends back to them.
4. **VLM sees the condition:** `encode_reference_in_prompt=True` puts
   `Picture 1: <vision>` plus the prompt into Qwen3-VL, at 384² px. Our Vision Prompt node
   (`picture_n` template) already does exactly this.
5. **Known-region restore:**
   - `keep_mask = NOT dilate(generated_mask, 32 px)`; a token is kept only if all 16×16 of
     its pixels are kept.
   - After each step, kept tokens are set to `known + σ_next·(initial_noise − known)`.
   - The known latent is the full-resolution canvas encode.
   - The 32 px band around the mask is left for the model to blend.
6. No pixel composite (AnyPaint). The Outpaint adapter pastes the source back with a 32 px
   inward feather.

## 2. Proposed design

### 2.1 Known-region restore in velocity space (sampler-agnostic)
Instead of a per-step callback in each sampler, the forward wrapper replaces the velocity of
kept tokens:

`v_keep = (x_t − x0_known) / σ`, which forces `x0_pred = x0_known` exactly.

- For Euler starting from pure noise, this reproduces AnyPaint's restore exactly. Starting
  at `x_1 = n`, each step gives `x0_known + σ_next·(n − x0_known)`, the same formula.
- It works unchanged for res_2m / res_2s / deis / ancestral eta (their x0 is pinned), for CFG
  (both passes get the same v, so the CFG delta is 0 on kept tokens), and for both Ultra call
  paths (`pipe(...)` Euler and `_res_denoise_packed`).
- No sampler code changes.

### 2.2 New module `_paint.py` (runtime) + node `nodes/krea2_paint.py`
`PaintRuntime(pipe, bundle)` follows the same install/remove pattern as Style/Control:
- **Forward wrapper:** reads the stage grid from `position_ids`. For each grid it lazily builds:
  - the known packed latent (known canvas resized to the stage size, VAE-encoded);
  - the keep-token mask;
  - the kv cache, if an adapter is selected. The cache is recomputed per grid because the
    registered coordinates scale with the target grid, and on `set_stage` because the LoRA
    scale changes.
- **Attention processors:** all 28 blocks get a `KVCacheAttnProcessor`. It appends the cached
  reference K/V and extends the text padding mask. It falls through to the original processor
  when inactive.
- **Precompute:** sets `tr._eric_paint_precompute`, so the Control hooks skip the reference
  tokens (the same pattern as `_eric_style_inversion`).
- **`set_stage(n)`:** per-stage switches for adapter conditioning and restore.

### 2.3 Node: "Eric Krea2 Paint (inpaint / outpaint)" → `KREA2_PAINT`, `IMAGE`, `IMAGE`, `MASK`
| input | default | note |
|---|---|---|
| `image` (IMAGE) | - | source |
| `mask` (MASK, optional) | none | ComfyUI convention: 1 = generate. Optional for pure outpaint. |
| `adapter` | `anypaint` | `anypaint` / `outpaint` / `none (training-free)` |
| `pad_left/right/top/bottom` | 0 | outpaint, in px; snapped so the canvas is a multiple of 16 |
| `seam_px` | 32 | regenerated band around the mask, scaled per stage |
| `reference_max_edge` | 384 | adapter contract; tooltip warns against changing it |
| `restore_s1/s2/s3` | on/on/on | keep unmasked tokens pinned at each stage |
| `adapter_s1/s2/s3` | 1/0/0 | adapter kv conditioning per stage (trained at ≤1024 px) |

Outputs:
- the bundle;
- `vlm_image`: the 384 px condition, to wire into Vision Prompt `image1`;
- `canvas_preview`;
- `generated_mask` (canvas size, after the bbox), used by the composite node.

The adapter LoRA itself goes on the **Multi-LoRA stack** (append it, S1 = 1, S2/S3 = 0) so
loading and format conversion stay in one place. The node checks that a LoRA whose name
contains "anypaint"/"outpaint" is loaded and warns if not.

### 2.4 Ultra integration (append-only, same as control/style)
- New **socket** `paint` (KREA2_PAINT), appended at the end. No widget changes.
- When `paint` is connected, the S1 dims follow the canvas aspect at `s1_megapixels`, unless
  explicit width/height is set, like `init_match_size`.
- `init_latent` + `s1_start_step` still work. The masked region then starts from a partially
  noised source instead of pure noise ("recolour the coat" edits keep the structure). Kept
  tokens are pinned either way.
- Mutually exclusive with `ref_latents` and `style` in the first build (each wraps the same
  forward). Compatible with `control`.
- **S2/S3:** restore re-encodes the **full-resolution source** at the stage size. Preserved
  regions therefore come back as the original pixels at high resolution, not a 3 MP→8 MP
  re-imagining, while the generated region gets refined normally. This is the main thing we
  add over the reference pipeline.
- `degrid`, `upscale_vae`, LoRAs and sweeps are unaffected.

### 2.5 Optional post node: "Eric Krea2 Paint Composite"
`(generated IMAGE, paint bundle or source + generated_mask, feather_px=32)` pastes original
pixels back outside the mask with a feathered edge, for pixel-exact preservation (the
Outpaint-adapter behaviour). It is a separate node, so it stays opt-in.

## 3. Limits to state in tooltips
- The adapters were trained on targets ≤1024 px with a 384 px condition. S1 at ~1-1.5 MP
  matches training best. 3 MP S1 should work (coordinates are normalized) but is out of
  distribution, so test before relying on it. S2/S3 refine without the adapter.
- Large generated fractions can introduce a second subject (stated by the author).
- Thin mask features are coarsened to 16 px tokens.
- `none` (training-free) is fine for small inpaints but weak for outpaint and large regions.

## 4. Phase 2 (not in the first build)
- **Cierpliwy inpaint-edit** (Raw): needs the ostris Edit pack's kv_cache placement read at
  source (index placement, black-painted reference). It would reuse the same KV processor
  with a different reference builder.
- Two-pass interior placement for the old Outpaint adapter. Not needed if AnyPaint is
  preferred, since it covers everything in one pass.
- Soft / differential masks.

## 5. Test plan (headless first, then Ultra)
1. **Parity:** for 3 AnyPaint showcase cases (internal edit, outpaint, mixed), run the author's
   `pipeline.py` against our runtime at the same seed, 1024 px, Euler 8 steps. The images
   should be near-identical (the same equations). Their recorded showcase results are a
   second reference.
2. **Bit-exact:** runtime installed with an empty mask and adapter `none` gives an identical
   image; after removal, identical again.
3. **Restore check:** the kept-token RMSE vs the known latent is ~0 at S1, S2 and S3.
4. **Ultra:** a 1.5 MP S1 → S2 run on one case. Check that the preserved region keeps its
   original detail at S2.

## 6. Needed from Eric
- **Downloads:** the AnyPaint LoRA (`krea2_anypaint_rank32.safetensors`) to
  `L:\Models\loras\Krea2\paint\`, plus the 3 showcase cases for parity. Optionally the old
  Outpaint LoRA.
- **Design calls:**
  1. AnyPaint as the default adapter.
  2. The S2/S3 full-resolution restore defaulting to on.
  3. The composite as a separate node.
  4. Cierpliwy deferred to phase 2.
