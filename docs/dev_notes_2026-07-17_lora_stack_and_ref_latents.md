# Dev notes 2026-07-17 — Multi-LoRA stack quality issue + reference-latents CFG investigation

Status: **unresolved, no code changes made**. Written up for handoff/second-opinion review.
Symptom reported by user: images produced with the Krea 2 Style Reference LoRA
(Civitai 2764349, ai-toolkit `krea2` edit LoRA, rank 64) are lower quality than
expected, both via the single `EricKrea2ApplyLoRA` node and via
`EricKrea2MultiLoRA` with all other slots toggled off. Additional LoRAs in the
stack are not simply inert — they visibly affect output, "just not what's
expected" (not yet characterized more precisely — see Open Questions).

## 1. Confirmed bug: `_live_lora_layer_count()` adapter-collision blind spot

**File:** `_lora_utils.py`

`_live_lora_layer_count(transformer, adapter_name=None)` (defined line 621) counts
how many modules currently carry LoRA layers. When called **without**
`adapter_name`, it can't tell "a new adapter was added" apart from "an existing
adapter already covered this module" — if adapter B targets the same modules as
already-loaded adapter A, the total live-layer count doesn't change after
injecting B, so the before/after comparison reads as "0 modules matched" and the
caller deletes the adapter it just added.

Confirmed empirically with a standalone PEFT repro (two `LoraConfig`s targeting
the same `nn.Linear`, injected under different adapter names):
```
before any adapter: 0
after adapter A (count, no filter): 1
after adapter A (count, filter=lora_A): 1
after adapter B added on SAME module (count, no filter): 1   # <-- unchanged!
after adapter B (count, filter=lora_B): 1                     # correct if filtered
```

**Three call sites currently omit the `adapter_name` filter** (all in
`_lora_utils.py`):
1. `_load_lora_adapter()` — lines 680 / 682 (pipeline-load try, inside the
   multi-fallback single-adapter loader).
2. `load_lora_with_key_fix()` — lines 1222 / 1224 (fast path,
   `pipe.load_lora_weights(lora_path, adapter_name=...)`).
3. `load_lora_with_key_fix()` — lines 1260 / 1267 (fallback path, after Krea
   original-format key conversion).

**Contrast:** `realize_lora_stack()` (line 1557) already does this correctly —
line 1582 calls `_live_lora_layer_count(transformer, adapter_name)` **with** the
filter. So the codebase has both the buggy and correct pattern side by side;
the fix (not yet applied) would be to pass `adapter_name=adapter_name` at the
three sites above, matching line 1582's pattern.

**Effect if uncorrected:** the 2nd/3rd+ LoRA in a stack that targets any module
already covered by an earlier-loaded LoRA gets falsely flagged as "0 modules
matched" and is deleted via `_safe_delete_adapter()` right after being injected.

## 2. Why this bug does NOT fully explain the user's symptoms

- User tested `EricKrea2MultiLoRA` with **only the style LoRA slot on**, all
  others off. Confirmed by reading `nodes/krea2_lora_stack.py`: when a slot's
  `on_i` is `False`, that slot is **never appended** to `pipeline["lora_stack"]`
  — it's excluded entirely, not loaded at weight 0. So this is a **1-entry
  stack**, structurally identical to the single-node path. The bug above only
  fires for the 2nd+ adapter colliding with an earlier one — it cannot explain
  a 1-entry-stack result being worse than the single-node result.
- User also reports the 2nd/3rd LoRAs are not simply "ignored" (which the bug
  would cause — full deletion, zero effect) but have visible, unexpected
  effects. That's inconsistent with "adapter got deleted" as the sole
  explanation. Needs reconciling — possibly two separate issues, or the
  deletion isn't as clean as `_safe_delete_adapter` assumes (e.g. PEFT layer
  wrapper left behind with a dangling/zeroed adapter still active — not yet
  checked).

## 3. Reference-latents / CFG asymmetry — investigated, likely NOT the cause (needs confirmation)

Hypothesis considered: `_ref_latents.py`'s `install_ref_latents()` monkey-patches
`pipe.transformer.forward` globally and cannot distinguish a positive/cond call
from a negative/uncond call, so under CFG the reference conditioning would ride
both passes and partially cancel in the CFG delta (this caveat is in the
module's own docstring already).

However, `nodes/krea2_multistage_ultra.py`'s `_generate_inner()` already gates
this: `turbo_guidance` defaults to `"off"`, and when `is_distilled` (Turbo
checkpoint) and `turbo_guidance == "off"`, **all stage CFG values are forced to
0.0** regardless of the `cfg`/`s1_cfg`/`s2_cfg`/`s3_cfg` inputs (see ~line
871-876) — meaning no negative encode, no uncond pass, single forward call per
step. In that (default) configuration there is nothing for the ref-latents
forward wrap to be asymmetric *about* — only one pass exists.

This theory only becomes live if the user's tests used `turbo_guidance` set to
`"s1_only"` or `"all_stages"`, or if they're on the **Raw/base** checkpoint
(which always runs full CFG, `turbo_guidance` only gates the `is_distilled`
branch). **Not yet confirmed which the user actually used** — this is the key
open question before pursuing this thread further.

## 4. Open questions (need answers before further action)

1. Checkpoint used for the bad-quality tests: Turbo/distilled or Raw/base?
2. `turbo_guidance` setting: default `off`, or `s1_only`/`all_stages`?
3. Were `ref_latents` (reference image conditioning) actually connected/used in
   the tests being compared, or was this plain txt2img with just the style
   LoRA and no reference image?
4. Concrete description of "effects, just not what's expected" for the 2nd/3rd
   LoRA case — garbled/noisy output, wrong style dominance, weak/diluted
   effect, color/lighting shift, structural distortion? This distinguishes
   between candidate mechanisms (partial-delete leftover vs. a genuinely
   different bug vs. expected LoRA-interaction behavior).
5. Were strength / per-stage-weight values identical between the single-node
   test and the 1-entry-stack test? (Should be, but not independently verified.)

## 5. Suggested next investigation steps (not yet done)

- Add `adapter_name` filtering at the three call sites in section 1 (low-risk,
  matches existing correct pattern elsewhere in the same file) — but note this
  alone won't explain the 1-entry-stack symptom, so verify against real
  generations before assuming it's "the fix."
- Instrument `_safe_delete_adapter` / the fast-path loaders to log actual PEFT
  adapter state (e.g. `transformer.peft_config.keys()`, `active_adapters()`)
  before and after each stack entry loads, to see directly whether a "deleted"
  adapter is truly gone or partially resident.
- Once Q1/Q2/Q3 above are answered, re-evaluate whether the ref-latents CFG
  caveat is even reachable in the user's actual test configuration.

---

## RESOLUTION UPDATE (2026-07-17, later session)

Section 1 fix **applied** after independent verification: `adapter_name` filter
added at all three call sites in `_lora_utils.py` (lines ~680/682, ~1222/1224,
~1260/1267), matching the correct pattern in `realize_lora_stack()`.

Additional finding from tracing the full fallback cascade: the colliding 2nd
LoRA was likely not deleted outright - after the double false-negative delete
it lands in `_load_lora_adapter_peft` (raw PEFT injection, rank from first
lora_A tensor, alpha from first `.alpha` key or alpha=r), i.e. loaded but with
potentially wrong scaling and possibly invisible to diffusers'
`pipe.set_adapters()`. Since `set_stack_stage_weights()` used one all-or-nothing
`set_adapters(names, weights)` call, a single unrecognised name silently killed
per-stage scaling for the ENTIRE stack. Hardened: batch -> on failure, retry
the diffusers-known subset as a batch, and set the rest directly per-layer via
PEFT `set_scale` (preserves alpha/r). Console log fingerprint of the old bug:
"Fast-path load matched 0 transformer modules" -> "using direct PEFT injection".

Sections 2-3 (1-entry-stack quality complaint, ref-latents CFG asymmetry)
remain open pending the Section 4 answers (checkpoint, turbo_guidance,
ref_latents connected).

---

## SECOND ROUND (2026-07-17, same day): three more bugs found from live testing

User-reported symptoms after the Section-1 fix:
  (a) multi-stack loads only ONE LoRA - whichever enabled slot comes first,
      regardless of position; the same LoRA works fine in the single node;
  (b) an ephemeral slider LoRA left its effect behind: slider ON -> long hair,
      slider OFF next run -> hair STILL long (weights leaked into the model).

Evidence gathered: safetensors header key dump of all four LoRAs from the test
run showed IDENTICAL key style (`diffusion_model.blocks.N.attn.wk.lora_A.weight`;
mohrbacher additionally has `.alpha` keys). Yet mohrbacher validated 256/256
targets and the other three 0/256, seconds apart, same transformer. The only
state change between validations: mohrbacher's adapter had just been injected.

### Bug A - converter validation blind to PEFT wrapping (the "only first LoRA" bug)
When PEFT injects an adapter into a Linear, the parameter is RENAMED:
`X.weight` -> `X.base_layer.weight`. `_convert_krea_lora_to_diffusers`
validated targets via `param_shapes.get(f"{diff_mod}.weight")` over
`named_parameters()` - so every module covered by an earlier-loaded adapter
reads "(missing)" and the whole LoRA is rejected as unmapped. First loadable
LoRA wraps the modules and wins; all later ones fail. Also explains why the
weightslider works in the single node (no prior wrapper) but not stacked.
Note `_apply_krea_direct_deltas` ALREADY handled this rename (its lines carry
a comment about it); the validation path never got the same treatment.
FIX: `_param_shape_wrapped()` helper (tries `X.weight` then
`X.base_layer.weight`) used at the low-rank, `.diff`, and `.diff_b` validation
sites. Same base_layer fallback added to the merge-target lookup in
`_load_lora_adapter_direct` and the LoKR/LoHa direct path.

### Bug B - unload restore misses renamed params and DESTROYS the backup (the ephemeral leak)
`unload_all_loras` restores `_lora_backup_<adapter>` snapshots by the key
recorded at LOAD time. If a direct merge happened on an unwrapped module and a
low-rank adapter wrapped that module later in the same realize pass, the
recorded key (`X.weight`) no longer exists at unload (`X.base_layer.weight`
now) -> restore silently skipped -> `delattr` destroyed the only pristine
copy -> delta permanently baked until model reload. Exactly reproduces the
long-hair symptom, and taints EVERY later generation (single or stack).
Likely also contaminated earlier quality comparisons (Section 2's open
"1-entry stack looks worse" complaint should be RE-TESTED after a fresh model
load before further investigation).
FIX: `_alt_param_keys()` - restore tries both namings (wrap and unwrap
direction); any entry that still cannot be restored is KEPT on the transformer
(never discarded) with a loud WARNING naming the adapter and sample keys.

### Bug C - `x or y` on tensors crashed the direct-merge fallback
`_load_lora_adapter_direct` used `params.get("lora_A.weight") or
params.get("lora_A.default.weight")` - `or` calls `bool(tensor)` which raises
"Boolean value of Tensor with more than one value is ambiguous" whenever the
FIRST key exists. The last-resort fallback was therefore unreachable (seen
three times in the user's console log). FIX: explicit `is None` checks.

### Post-fix state / test protocol
- IMPORTANT: any model loaded during a leaky session has baked deltas and its
  backups are gone - fully reload the checkpoint (or restart ComfyUI) before
  judging output.
- Test 1 (stack): two Comfy-format LoRAs (e.g. mohrbacher + vision_vanguard)
  in the stack -> expect NO "(missing)" unmapped lines, both adapters in the
  `verify:` line with per-stage weight lines listing both.
- Test 2 (ephemeral): slider + style LoRA stacked, run, toggle slider off,
  run -> effect must disappear; console shows "restored N direct-merged
  param(s)" and must NOT show the new "backup KEPT" warning.
- If the "backup KEPT" WARNING ever appears, capture the sample keys - it
  means a third naming variant exists that _alt_param_keys doesn't cover yet.
- Helper-level tests (key parsing, wrap-tolerant lookups, real key styles)
  pass offline; not yet validated against a live generation.
