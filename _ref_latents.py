# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 reference-latent (edit) support
==========================================
Diffusers-side implementation of the "index_timestep_zero" reference method that
ai-toolkit's Krea2 edit LoRAs (``model_kwargs.edit: true``) are trained with, and
that ostris/ComfyUI-Krea2-Ostris-Edit implements for the ComfyUI-core model.
Reimplemented from the published mechanism against OUR diffusers
``Krea2Transformer2DModel`` (verified line-by-line against the installed
``transformer_krea2.py``) - not copied from the ostris repo (which ships no license).

Mechanism (what an edit LoRA expects at inference):
  * Each reference image is VAE-encoded, normalized, and flow-packed exactly like
    a generation latent (we reuse ``_latent_utils.standard_encode``).
  * The packed reference tokens are APPENDED to the image token sequence.
  * Rotary position ids: reference i gets axis-0 index ``i + 1`` with its own
    h/w grid from 0 (the live image sits at axis-0 index 0).
  * Modulation: the live span (text + noisy image) is modulated with the real
    timestep; the reference span is modulated with **t = 0** (clean data) -
    that is the "index_timestep_zero" part.
  * The velocity prediction is sliced to the live image tokens only; reference
    tokens are dropped before the final layer.

The base model was NEVER trained to read these tokens - this pathway only does
something useful with an edit-trained LoRA loaded (e.g. the Civitai "Krea 2
Style Reference LoRA", trained on style pairs with the Turbo adapter).

CFG caveat: the transformer-forward wrapper cannot tell a positive call from a
negative one, so with guidance enabled the references condition BOTH passes and
their influence partially cancels in the CFG delta. Intended for Turbo at
guidance 0 (which is what the published edit LoRAs target anyway).

kv_cache (ai-toolkit's isolate-ref-attention optimization) is deliberately not
implemented in v1 - it only applies to LoRAs trained with that kwarg.

UNTESTED ON A LIVE MODEL as of writing - built against the installed diffusers
source (module names, modulation layout, rotary/mask shapes all verified by
reading ``transformer_krea2.py`` + ``pipeline_krea2.py``) and compiles clean.
Turn on the console prints for the first run.

Author: Eric Hiss (GitHub: EricRollei)
Mechanism credit: ostris / ai-toolkit (index_timestep_zero reference method).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ── reference bundle preparation (encode side) ──────────────────────────────

def prepare_ref_bundle(pipe, images, max_megapixels: float = 1.0, encode_vae=None):
    """ComfyUI IMAGEs -> packed, normalized reference latents + rotary grids.

    images: list of ComfyUI IMAGE tensors (B, H, W, C in 0..1); frame 0 of each
    batch is used. Each is downscaled (never upscaled) to fit ``max_megapixels``,
    then encoded via ``_latent_utils.standard_encode`` (deterministic dist.mode,
    (raw - mean) / std normalization, flow-pack) - byte-identical latent space to
    what Generate/Multi-Stage step through. standard_encode snaps dims to /16
    (vae 8x * patch 2), matching ai-toolkit's edit-training snap.

    Returns {"packed": [(1, seq_i, C) cpu fp32 ...], "grids": [(gh, gw) ...],
             "sizes": [(h, w) ...]}.
    """
    from ._latent_utils import standard_encode

    packed_list, grids, sizes = [], [], []
    cap_px = float(max_megapixels) * 1_000_000.0
    for idx, image in enumerate(images):
        if image is None:
            continue
        img = image[:1]  # first frame of the batch
        h, w = int(img.shape[1]), int(img.shape[2])
        if h * w > cap_px:
            scale = (cap_px / (h * w)) ** 0.5
            nh, nw = max(16, int(round(h * scale))), max(16, int(round(w * scale)))
            img = torch.nn.functional.interpolate(
                img.permute(0, 3, 1, 2), size=(nh, nw), mode="bilinear",
                align_corners=False).permute(0, 2, 3, 1)
            print(f"[EricKrea2-Ref] reference {idx + 1}: downscaled {w}x{h} -> {nw}x{nh} "
                  f"(cap {max_megapixels:.2f} MP)")
        packed, tgt_h, tgt_w = standard_encode(pipe, img, encode_vae=encode_vae)
        p = pipe.patch_size
        gh = (tgt_h // pipe.vae_scale_factor) // p
        gw = (tgt_w // pipe.vae_scale_factor) // p
        packed_list.append(packed.detach().to("cpu", torch.float32))
        grids.append((gh, gw))
        sizes.append((tgt_h, tgt_w))
        print(f"[EricKrea2-Ref] reference {idx + 1}: {tgt_w}x{tgt_h}px -> grid {gh}x{gw} "
              f"({gh * gw} tokens)")
    return {"packed": packed_list, "grids": grids, "sizes": sizes}


# ── transformer forward wrapper (denoise side) ──────────────────────────────

def install_ref_latents(pipe, bundle, verbose: bool = True):
    """Wrap ``pipe.transformer.forward`` so reference tokens ride along at t=0.

    Returns the ORIGINAL bound forward - restore it in a ``finally`` with
    ``pipe.transformer.forward = original``. The wrapper covers BOTH Ultra call
    paths (pipe(...) euler loop and _res_denoise_packed's direct transformer
    calls), since both go through transformer.forward.

    The wrapper re-runs the model's own submodules (text_fusion, txt_in, img_in,
    transformer_blocks, final_layer) so any loaded LoRA - including edit LoRAs
    on those linears - applies exactly as in the stock forward; only the
    per-span modulation and the extended sequence/rotary/mask differ.
    """
    tr = pipe.transformer
    orig_forward = tr.forward
    packed_list = [p for p in bundle.get("packed", []) if p is not None]
    grids = list(bundle.get("grids", []))
    if not packed_list:
        return None

    try:
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
    except Exception:  # very old diffusers layouts
        Transformer2DModelOutput = None

    state = {"printed": False}

    def wrapped(hidden_states, encoder_hidden_states, timestep, position_ids,
                encoder_attention_mask=None, attention_kwargs=None, return_dict=True,
                **extra):
        device, dtype = hidden_states.device, hidden_states.dtype
        B, img_len, _ = hidden_states.shape
        txt_len = encoder_hidden_states.shape[1]

        # 1) reference tokens + their rotary rows (axis-0 = i+1, own h/w grid)
        ref_toks, ref_rows = [], []
        for i, (pk, (gh, gw)) in enumerate(zip(packed_list, grids)):
            ref_toks.append(pk.to(device=device, dtype=dtype).expand(B, -1, -1))
            rid = torch.zeros(gh, gw, 3, device=device)
            rid[..., 0] = float(i + 1)
            rid[..., 1] = torch.arange(gh, device=device, dtype=torch.float32)[:, None]
            rid[..., 2] = torch.arange(gw, device=device, dtype=torch.float32)[None, :]
            ref_rows.append(rid.reshape(gh * gw, 3))
        ref_tok = torch.cat(ref_toks, dim=1)
        ref_len = int(ref_tok.shape[1])
        pos = torch.cat([position_ids.to(device=device, dtype=torch.float32),
                         torch.cat(ref_rows, dim=0)], dim=0)

        # 2) dual timestep modulation: real t for the live span, t=0 for refs
        temb = tr.time_embed(timestep, dtype=dtype)
        temb0 = tr.time_embed(torch.zeros_like(timestep), dtype=dtype)
        mod = tr.time_mod_proj(F.gelu(temb, approximate="tanh"))
        mod0 = tr.time_mod_proj(F.gelu(temb0, approximate="tanh"))

        # 3) masks: refs are always-valid keys, like image tokens
        text_mask = None
        attn_mask = None
        if encoder_attention_mask is not None:
            text_mask = encoder_attention_mask[:, None, None, :]
            live = encoder_attention_mask.new_ones((B, img_len + ref_len))
            attn_mask = torch.cat([encoder_attention_mask, live], dim=1)[:, None, None, :]

        # 4) embed: [text | noisy image | clean refs] through the model's own modules
        ctx = tr.text_fusion(encoder_hidden_states, attention_mask=text_mask)
        ctx = tr.txt_in(ctx)
        img = tr.img_in(torch.cat([hidden_states, ref_tok], dim=1))
        h = torch.cat([ctx, img], dim=1)
        split = txt_len + img_len  # first ref token

        rot = tr.rotary_emb(pos)

        if verbose and not state["printed"]:
            state["printed"] = True
            print(f"[EricKrea2-Ref] active: {len(packed_list)} reference(s), {ref_len} ref tokens "
                  f"appended (seq {txt_len}+{img_len}+{ref_len}), t=0 modulation on ref span")

        # 5) blocks with per-span modulation (mirrors Krea2TransformerBlock.forward,
        #    indices: 0 prescale, 1 preshift, 2 pregate, 3 postscale, 4 postshift, 5 postgate)
        for block in tr.transformer_blocks:
            m = (mod.unflatten(-1, (6, -1)) + block.scale_shift_table).unbind(-2)
            r = (mod0.unflatten(-1, (6, -1)) + block.scale_shift_table).unbind(-2)

            def span_mod(t_, si, hi):
                return torch.cat([(1.0 + m[si]) * t_[:, :split] + m[hi],
                                  (1.0 + r[si]) * t_[:, split:] + r[hi]], dim=1)

            def span_gate(t_, gi):
                return torch.cat([m[gi] * t_[:, :split], r[gi] * t_[:, split:]], dim=1)

            attn_out = block.attn(span_mod(block.norm1(h), 0, 1),
                                  attention_mask=attn_mask, image_rotary_emb=rot)
            h = h + span_gate(attn_out, 2)
            ff_out = block.ff(span_mod(block.norm2(h), 3, 4))
            h = h + span_gate(ff_out, 5)

        # 6) live image tokens only -> final layer at real t
        out = tr.final_layer(h[:, txt_len:split], temb)
        if not return_dict:
            return (out,)
        if Transformer2DModelOutput is not None:
            return Transformer2DModelOutput(sample=out)
        return (out,)

    tr.forward = wrapped
    if verbose:
        n_tok = sum(int(p.shape[1]) for p in packed_list)
        print(f"[EricKrea2-Ref] installed reference-latent forward wrap: "
              f"{len(packed_list)} ref(s), {n_tok} tokens total. Needs an edit-trained LoRA "
              f"(ai-toolkit edit:true) to do anything useful; intended for Turbo/guidance 0.")
    return orig_forward
