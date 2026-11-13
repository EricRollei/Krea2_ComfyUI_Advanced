# High-res, upscale VAE, renorm & turbo guidance

[<- Back to the README](../README.md) | [All docs](../README.md#documentation)

## Two high-res strategies

Krea 2's two checkpoints suit two different paths:

**Turbo → fast 4k (no re-denoise).** Turbo is distilled (8 steps, guidance 0) and
resists partial re-denoise, but the 2× decode sidesteps that entirely:

```
Loader (Krea-2-Turbo) → Generate (2048², auto) → [Upscale VAE Loader] → Upscale Decode 2× → 4096²
```

**Raw → controlled high-res (multi-stage).** Raw keeps the full-CFG,
resolution-aware trajectory that partial re-denoise needs:

```
Loader (Krea-2-Raw) → Multi-Stage Ultra (1024² → 2× → 2×) → 4k image  (+ optional Upscale Decode 2× → 8k-class)
```

## Upscale VAE: modes & precedence

On the Multi-Stage Ultra node, `upscale_vae_mode` (requires `upscale_vae` connected)
governs the **S2→S3** jump and the **final decode**. Each transition can be a plain
2× VAE step (real resolution gain, 4× area) or a **downsample** variant that keeps the
upscale decoder's sharp/GAN-trained detail but resamples down to your chosen size:

| Mode | S2→S3 | Final image |
|------|-------|--------------|
| `disabled` | bislerp latent resize | `decode_vae` (or the base Qwen VAE if none connected) |
| `s2-s3` | **upscale VAE** 2× (4× area; ignores `upscale_to_stage3`) | `decode_vae` / base |
| `s2-s3 with downsample` | **upscale VAE** 2× → downsample to `upscale_to_stage3` | `decode_vae` / base |
| `final decode` | bislerp latent resize | **upscale VAE** 2× |
| `final decode with downsample` | bislerp latent resize | **upscale VAE** 2× → downsample to native |
| `both` | **upscale VAE** 2× (4× area) | **upscale VAE** 2× |
| `both with downsample` | **upscale VAE** 2× → downsample | **upscale VAE** 2× → downsample to native |
| `both with final decode downsample` | **upscale VAE** 2× (4× area) | **upscale VAE** 2× → downsample to native |

> **Recommended: prefer the `… with downsample` variants.** In practice the upscale
> VAE decoded at 2× *then* resampled down is cleaner and sharper than the Wan 2.1 VAE
> at the same output size - you get its detail without the literal 4× area blow-up (and
> without the re-encode softening compounding across stages). The plain `s2-s3` / `both`
> modes force 4× area and **ignore** `upscale_to_stage3`, which is the usual cause of a
> run ballooning to 20 MP+; the downsample variants respect your numeric factor instead.

The old `inter_stage` / `final_decode` values still load (aliased to `s2-s3` /
`final decode`) so existing graphs keep working.

**S1→S2 has its own switch.** `s1_s2_upscale_vae` (a separate boolean) routes the
*first* jump through the upscale VAE and forces 2× (4× area), overriding
`upscale_to_stage2`. It's deliberately not in the dropdown - using the VAE that early is
a specialist move (the re-encode softening gets refined away by S2/S3). Combining it
with a `both …` mode is what cascades to enormous final sizes - watch the on-node
**resolution chain** readout, which shows the real S1→S2→S3→decode sizes and flags every
field the VAE is overriding.

**`decode_vae` is ignored outright for the final image whenever the mode does a VAE
final decode** (`final decode*` or `both*`) - not blended, not overridden, simply not
passed to that call at all. It still governs the **stage preview** images
(`preview_stages=True` → `stage1_image`/`stage2_image`) regardless of what the final
image uses, since previews always go through `standard_decode(..., decode_vae=decode_vae)`
independent of `upscale_vae_mode`. So: want the upscale VAE for your final image but a
specific grain/texture VAE for previews? Keep both connected - they don't conflict.
Want the upscale VAE and don't care about previews? `decode_vae` can stay disconnected
with no effect either way.

**Two ways to get the downsample quality pass on the final decode.** The mode's
`… final decode downsample` variants request it per-run; the Upscale VAE Loader's own
`downsample_to_1x` toggle bakes it into the loaded VAE for *every* consumer (including
the standalone Upscale Decode node). Either enables the same thing - decode at 2× then
Lanczos-downsample back to native, a supersampling-style pass rather than a literal
resolution gain. `blur_sigma=0` is a good default (Lanczos already anti-aliases); raise
it only as a taste knob. Inter-stage (S2→S3) downsampling reads the mode flags, not the
loader boolean.

## Inter-stage renorm (`upscale_renorm`)

Latent interpolation is a lowpass: every upscale hop shrinks latent variance,
and decoded variance *is* contrast/saturation - so multi-hop runs compound a
little drain per hop into progressive flattening and desaturation.
`upscale_renorm` repairs the statistics right after each hop:

- **`per_channel`** (recommended) - re-match each latent channel's mean/std to
  the pre-hop latent; strongest color + contrast hold.
- **`global`** - overall mean/std only.
- **`off`** - previous behavior.

A telemetry line prints latent std pre-hop / post-hop / post-renorm in every
mode, so the drain (and the repair) is measurable per run.

## Turbo guidance & the negative prompt

Turbo is distilled for guidance 0, so most wrappers give you no CFG - and therefore no
**negative prompt** - on the Turbo checkpoint. Multi-Stage Ultra re-enables it. The
`turbo_guidance` dropdown decides where classifier-free guidance (and the negative pass
it needs) is allowed on a distilled model:

| `turbo_guidance` | S1 | S2 | S3 |
|------------------|----|----|----|
| `off` *(default)* | g = 0 | g = 0 | g = 0 |
| `s1_only` | uses `s1_cfg` | forced g = 0 | forced g = 0 |
| `all_stages` | `s1_cfg` | `s2_cfg` | `s3_cfg` |

- **Guidance convention** is Krea's `g`, where standard **CFG = 1 + g** (`g=0` → CFG 1.0,
  `g=0.2` → CFG 1.2). Turbo was distilled for `g=0`, so keep it low (**~0.1-0.3**); the node
  warns above `g=1.5` (Raw-level, over-saturates Turbo).
- **The negative prompt is only active on a stage whose `g > 0`.** Under `off` it is inert.
- Each guided stage runs a second (uncond) pass, so it is **~2× slower** for that stage -
  which is why `s1_only` exists: put guidance + negatives on the composition, then let
  S2/S3 refine fast and single-pass (it force-zeros `s2_cfg`/`s3_cfg` for you - you don't
  set them yourself).

Using a real negative prompt on a distilled/Turbo checkpoint at all is unusual - most
Turbo pipelines simply can't feed one.

For negatives **without** a second pass, see [NAG / NegPiP](NEGATIVE_GUIDANCE.md#negative-guidance-nag--negpip) -
both work at g = 0.
