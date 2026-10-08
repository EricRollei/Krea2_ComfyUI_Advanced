# Eric_Krea2 - roadmap / to-do

Living list of agreed-but-not-built work. Newest decisions at the top of each item.

## Native quantized compute toggle - BUILT 2026-10-08 (_native_quant.py; Component Loader compute / native_format, appended)
Re-quantizes the 224 joint-block Linears after the normal bf16 load onto comfy_kitchen kernels
(same QuantizedTensor path as ComfyUI core). keep = checkpoint format; int8 = ConvRot; mxfp8; nvfp4
(dynamic activation scale). PEFT LoRA works on top (verified: LoRA effect identical to bf16 under
int8-convrot); direct-merge deltas (diff / LoKr / LoHa / fallback) become exact bf16 side terms.
Bench (RTX PRO 5000, 3 MP, s/step; bf16 2.93 / 48.2 GB peak): int8-convrot 2.06 (1.42x, closest
to bf16), int8 2.02, mxfp8 2.11, fp8 2.14, nvfp4 1.75 (1.67x, 30.8 GB). docs/tools/test_native_quant.py.
Original plan notes (kept for reference):

**Why:** the loader always dequantizes to bf16 (best IQ). Measured on the RTX PRO 6000
(`docs/tools/bench_quant_matmul.py`, per sampling step, one Krea2 block x28):

| mode  | 3.3 MP (S1) | 8.5 MP (S3) |
|-------|-------------|-------------|
| bf16  | 6.8 s (x1.00) | 27.0 s (x1.00) |
| fp8 per-tensor | x1.10 | x1.06 |
| mxfp8 | **x1.32** | x1.16 |
| nvfp4 | **x1.67** | x1.32 |

Attention (unaffected by quant) is ~60% of an 8 MP step, so the gain is largest for
**S1-only runs at ~3 MP** - a common workflow (Eric runs many S1-only 3 MP jobs), where
~30% is worthwhile.

**Scope when built:**
- User toggle on the component loader: `compute: bf16 | native_quant` (append at END).
- Formats to support natively: INT8 tensorwise (+ConvRot) FIRST (by far the most common
  Krea2 fine-tune format in the library), MXFP8, NVFP4, fp8-scaled. Add INT8 to the bench
  before building (Blackwell INT8 tensor cores ~ fp8 rate).
- GGUF: lower-quant GGUFs are common -> consider keeping GGUF quantized via ComfyUI-GGUF's
  GPU dequant-on-the-fly ops (VRAM win), and at minimum move GGUF dequant to GPU (load is
  ~2x slower than safetensors today because the numpy dequant runs on CPU).
- Design problem to solve: LoRA. PEFT wraps the base Linear (OK with quantized weights),
  but the direct-merge fallback writes into weights - must be disabled or routed to an
  unmerged delta path in native_quant mode.
- IQ caveat to document: native quant also quantizes ACTIVATIONS (extra loss vs dequant).

## Attention speed (idea, 2026-10-07)
At 8 MP attention is ~60% of the step and measured kernel efficiency is low. Evaluate
SageAttention / FA-for-Blackwell in the component loader's attention backend list -
likely a bigger high-res win than any quant mode.

## i2i program (priority order agreed 2026-10-07)
1. ControlNet (Control LoRA, facok method) - BUILT 2026-10-07 (_control.py, nodes/krea2_control.py; Ultra `control` socket)
2. Training-free style transfer (nkxx188 port) - BUILT 2026-10-07 (_style_ref.py, nodes/krea2_style.py; Ultra `style` socket); strength sweep + Ultra 3 MP check 2026-10-08
3. Inpaint / outpaint - BUILT 2026-10-08 (_paint.py, nodes/krea2_paint.py: Eric Krea2 Paint + Paint Composite; Ultra `paint` socket). AnyPaint/Outpaint mechanism (registered ref + isolated kv_cache + velocity-space restore). Single-forward parity vs the author's pipeline: identical velocity given identical ref tokens. Spec: docs/spec_inpaint_outpaint_2026-10-08.md. Phase 2: Cierpliwy inpaint-edit (Raw, ostris placement), soft masks.
4. Edits - BUILT 2026-10-08 (see section below + docs/spec_edits_2026-10-08.md)

## Smaller items
- Krea2T-Enhancer-style per-token relative-delta cap for cond_rebalance (`rebalance_cap`).
- LoRA layer-group strip / image-token-only LoRA option on the Multi-LoRA stack.
- TeaCache-style step skipping for S2/S3 refine windows (measure with the sweep system).

## flash_varlen prefix-mask bug (found 2026-10-08) - FIXED 2026-10-08 (approved by Eric)
diffusers `_flash_varlen_attention` keeps `key[b, :valid_len]` (assumes valid keys are a PREFIX).
Krea2 always pads text to 512 and the sequence is [text(padded) | image], so with the loader's
`attention_backend=auto` (-> flash_varlen) every generation attends to ~(512-prompt) garbage padded
text keys and DROPS the last ~(512-prompt) image keys = bottom token rows. Confirmed same-seed 3 MP
A/B (crop_bottom 0): flash shows the bottom smear band (what crop_bottom hides); SDPA clean edge to
edge; composition/prompt adherence also differed. SDPA is ~3.5x slower + ~80 GB reserved at 3 MP, so
the fix is to compact the valid keys before the kernel (as _paint.compact_kv does for paint/style),
applied to the stock Krea2 attention path. Changes existing seeds' outputs.
FIX: _compat.Krea2MaskSafeProcessor installed by apply_attention_backend on all 32 Krea2 attention
modules (both loaders). Verified (docs/tools/test_mask_safe.py, 1 MP same seed): flash+fix vs SDPA
31.8 dB whole / 45.1 dB bottom 10% (kernel noise); stock flash vs SDPA 15.2 / 11.5 dB. flash+fix 7.3 s
= stock flash 7.3 s; SDPA 21.1 s. Also removes SDPA's dense-mask OOM at 3 MP batch-2 (style).
Follow-up: crop_bottom should no longer be needed (widget kept; verify on real workflows).

## Native compute - ComfyUI end-to-end (2026-10-08, RTX PRO 5000, Ultra S1-only 3 MP, euler 8 steps)
| checkpoint (keep ->) | bf16 s/step / alloc | native s/step / alloc | PSNR vs bf16 |
|---|---|---|---|
| INT8-ConvRot 3091496 (int8_convrot) | 2.9 / 34.8 GB | 2.0 / 22.9 GB | 29.5 dB |
| same + Gainsborough PEFT LoRA | 3.1 / 35.3 GB | 2.2 / 22.9 GB | 29.2 dB |
| Kreamania mxfp8 (mxfp8) | 2.9 / 35.0 GB | 2.1 / 23.3 GB | 26.0 dB |
| sinoxedit fp8-scaled (fp8) | 2.9 / 35.1 GB | 2.0 / 22.9 GB | 18.6 dB (composition shifts; quality equal) |
Note: darkBeastINT8Convrot2_darkBeastKREA2FP8.safetensors is actually a bf16 file (keep -> mxfp8).

## Pipeline-handle VRAM leak - FIXED 2026-10-08 (_pipe_handles.py)
LoRA / LoRA-stack / Unload-LoRA nodes returned plain dict copies of the pipeline that ComfyUI's
node cache kept alive; ComfyUI also keeps OLDER loader outputs. A checkpoint switch after a LoRA
run left the old model resident -> OOM. Every loader output and every derived copy is now
registered; _free_current nulls all handles pointing at the outgoing pipeline. Verified: 8-job
mixed queue (3 checkpoints, bf16/native, LoRA) all succeed on the 48 GB card.

## Edits + negative guidance - BUILT 2026-10-08 (spec: docs/spec_edits_2026-10-08.md)
Nodes: Eric Krea2 Edit (KREA2_EDIT) + Eric Krea2 Negative Guidance (KREA2_GUIDANCE, NAG + NegPiP,
both off by default, combinable) -> new Ultra sockets dit, guidance (appended).
LoRA: krea2_identity_edit_v1_2 (+ r128) in L:\Models\loras\Krea2\edit\2026-identity-edit-v1_2\.
Verified: geometry 168/168 byte-identical to krea2edit v1.2.5; attention core vs dense references ~1e-7.
ComfyUI e2e (5000, Turbo 10 steps, 1.5 MP): recolor / pose (identity kept) / add object / replace
object / mismatched-AR outpaint all good; ref_boost 4 holds the reference silhouette closer.
Removal ("the person in the background") removed BOTH people on Turbo, Turbo+NAG and Raw CFG 3 -
model limitation (card: removals unreliable); try wording that names what to keep.
NAG (wings test): suppresses well. NegPiP in_place: cleanest -> default; lifted left a dark flap;
NAG+NegPiP together over-suppresses (dark patches) - use one.
Cost per step at 1.5 MP: edit 3.1 s (vs 1.3 t2i - 2x tokens); ref_boost +6%; NAG +23% t2i / +12% edit;
NegPiP ~0. TODO: fuse the LSE merge (fp32 elementwise) to cut NAG overhead.
Also fixed: RefLatents edit_frame used a stretch -> now v1.2 fit geometry + explicit grid gating.

## Model-switch OOM (pipeline survived release) - FIXED 2026-10-08
_free_current kept the outgoing pipeline in a local list while calling gc.collect(); the pipeline
holds reference cycles (forward-wrapper bound methods), so it was only freed by a LATER collection,
after the next model was already loading -> OOM on a 48 GB card (Turbo -> Raw switch). Release now
runs in a helper (no surviving locals); EditGuidanceRuntime restores forward without a cycle; the
loader logs the referrers if a released pipeline still survives (self-diagnosis).
## Overnight run 2026-10-08 (items 1, 2, 3, 7)
- **Native extras - VERIFIED.** CPU round-trip of a native-quant pipeline is bit-exact and frees
  13.5 GB; Unload is exact; teardown frees VRAM. LoKr on native: effect 18 dB vs no LoRA,
  native+LoKr vs bf16+LoKr 24.7 dB (same as plain native-vs-bf16).
- **LoKr loading - FIXED.** ai-toolkit Krea 2 LoKr (original key names) applied 0/256 modules.
  Fixes in _lora_utils: `_remap_krea_lycoris_keys`, PEFT factor from the file
  (`_lokr_file_factor`, 8 for ai-toolkit), purge of half-injected PEFT layers before the direct
  fallback, `base_layer.weight` lookup, ComfyUI scale rule (full w1/w2 -> strength).
- **Control combos - FIXED.** Control was silently dropped whenever reference tokens were in
  img_in (hook needs exactly gh*gw tokens). Target and reference spans now go through img_in
  separately, refs flagged `_eric_ctrl_skip`. control+edit, control+NAG clean; control applied
  10/10 calls. control+ref_latents: control applies; full visual run with the Style Ref LoRA not
  finished (VRAM spill at 1.5 MP with 12 GB taken by displays) - rerun at 1 MP.
- **ref_latents without its LoRA = mosaic**, not "ignored" (README corrected). With the Style
  Reference LoRA at 0.5 the image is clean but style transfer was weak (Van Gogh ref, photo
  result) - try 1.0 as the README recommends.
- **NAG overhead** +23% -> +14% (2-part closed-form lerp merge, fp32 norms only).
- **Attention backends (bench_attention_backends.py, RTX PRO 5000, H48/KV12/D128, bf16):**
  | backend | 1.5 MP | 3 MP | 8.4 MP | rel err | cos |
  |---|---|---|---|---|---|
  | flash_attn2 (current) | 5.39 ms | 20.2 ms | 146 ms | 2.3e-3 | 0.999997 |
  | SDPA cuDNN | x0.92 | x0.93 | x0.94 | same | same |
  | SDPA mem-efficient | x0.39 | x0.39 | x0.42 | same | same |
  | SDPA flash | not compiled in this torch | | | | |
  | xformers | x0.99 | x0.93 | x0.99 | same | same |
  | sage2 `sageattn` auto | x1.60 | x1.61 | x1.73 | 3.9e-2 | 0.99923 |
  | sage2 int8 QK / fp8 PV (cuda) | x1.73 | x1.82 | x1.74 | 3.9e-2 | 0.99924 |
  | sage2 int8 QK / fp16 PV (triton) | x1.17 | x1.15 | x1.13 | 1.3e-2 | 0.99991 |
  | sage2 int8 QK / fp16 PV (cuda) | INVALID - 0.2-1.3 ms timings and NaN at 3 MP: no working sm120 kernel, output not computed |
  Notes: diffusers `sage_varlen` = the Triton kernel, so the loader's `sage` choice (varlen
  first) gets only ~1.15x. Dense `sageattn` (fp8 PV) is the real win: with attention ~60% of
  an 8 MP step, roughly -25% step time at 8 MP, ~-10-15% at 3 MP. Its ~4% per-call error is the
  usual SageAttention tradeoff; needs an end-to-end A/B (PSNR + eyeball at 3 and 8 MP)
  before any default change. _split_attn (edit/NAG/ref_boost) needs LSE: sage2's
  `sageattn_qk_int8_pv_fp8_cuda(..., return_lse=True)` could serve it.
  **DECISION FOR ERIC:** (a) leave as is; (b) add a `sage_fp8` choice that calls dense
  `sageattn_qk_int8_pv_fp8_cuda` behind the mask-safe key compaction (append-only option);
  (c) also use it in _split_attn. Recommend (b) then an A/B, (c) only if (b) looks clean.
  -> Eric chose (b). BUILT 2026-10-08: `sage_fp8` appended to both loaders' attention_backend
  (candidates `_sage_qk_int8_pv_fp8_cuda` -> sage -> flash_varlen -> flash; diffusers already
  registers that backend, mask-safe compaction makes dense OK). E2E A/B
  (docs/tools/test_sage_fp8_ab.py, Turbo bf16, 8 steps, 5000): s/step flash -> sage_fp8
  1.5 MP 1.32-1.36 -> 1.26-1.29 (x1.05); 3 MP 2.96-3.02 -> 2.68-2.73 (x1.09-1.12);
  8.4 MP 10.20 -> 8.30 (x1.23). PSNR vs flash 14-36 dB = trajectory divergence (sibling
  images / reframing), no visible artefacts at 1:1. Kernel-only 1.7x shrinks because MLP,
  norms, RoPE and the QKV/out projections dominate below ~8 MP. (c) still open: _split_attn
  via sage `return_lse=True` - worth it mainly for 8 MP edit runs.

## Reproducibility + small fixes (2026-10-08, after v1.6.0 commit)
- **er_sde was not seed-reproducible - FIXED.** Ultra built its seeded noise_sampler only when
  eta > 0; er_sde churns every step regardless of eta, so at eta 0 it drew UNSEEDED randn_like
  (different image every run at a fixed seed). Now er_sde always gets the seeded sampler.
  _res_solver's unseeded fallback now warns once when used.
- **LoRA search** works without the LoRA Catalog (tested: 568 LoRAs, filename/folder/trigger
  search); note now reads "LoRA Catalog not installed (optional)" when none is configured.
- **Mask-safe attention kept** after Eric's A/B (no quality difference; fixes the bottom edge).
  Observed: with the fix, prompt/LoRA concepts express more strongly (e.g. Dali-style figures
  more often nude) - consistent with the old path diluting the prompt with ~400 padding keys.
  Diagnostic switch ERIC_KREA2_MASK_SAFE=0 (env, read at startup) remains for future A/Bs.
- **Launcher review** (hunyuan_speed_launcher.bat): dead TORCH_CUDNN_* / XFORMERS_DISABLED env
  vars removed, --fast autotune added, CUDA_DEVICE_ORDER=FASTEST_FIRST explicit,
  expandable_segments unsupported on Windows (left off). Backups in docs/backup.

## Prompt token budget (2026-10-08)
- Ultra prints `prompt: N/507 tokens` every run (exact, pipeline tokenizer/template) and warns with
  the dropped words on overflow.
- Trigger merge (_trigger_words.merge_triggers_into_prompt, pipe=...) is budget-aware via
  _prompt_budget.py: shortens the prompt body (drop sentences before the last, else cut the end)
  so appended/prepended triggers survive. Whole-word trigger dedup (rust != rusted).
- video_prompter Platform Prompt Rewriter: length_guard (retry+trim / trim / off, appended);
  core/profiles/token_budget.py (Qwen tokenizer from local files, else estimate); target =
  limit - 8% headroom (Krea 2: 471 of 512). Backups: video_prompter/dev/backup_2026-10-08_token_budget.
- Not covered: video_prompter LoRA Suggester `prompt_with_triggers` (pre-rewrite merge, substring
  dedup) and the other expander nodes - they use their own length rules.
