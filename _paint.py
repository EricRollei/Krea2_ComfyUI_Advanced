# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 inpaint / outpaint runtime (registered reference + isolated kv_cache + restore)
=========================================================================================
Installed by Multi-Stage Ultra for one generation when its ``paint`` socket is connected
(bundle built by nodes/krea2_paint.py). Removed in ``finally``.

Mechanism (yijunwang2/krea2-anypaint + krea2-outpaint, Apache-2.0 pipeline code; the
contract the adapter LoRA was trained against - reimplemented on OUR diffusers model,
see docs/spec_inpaint_outpaint_2026-10-08.md):

  1. Condition image: the whole canvas with every to-be-generated pixel set to the median
     colour of the known pixels, <= 384 px max edge (AnyPaint), or the source alone
     (Outpaint). VAE-encoded + packed like a generation latent.
  2. Registered RoPE: the condition's tokens sit on frame axis 1 with FRACTIONAL h/w
     coordinates mapped into the target grid via its normalized bbox:
        y = y0*H + (i + 0.5) * (y1 - y0) * H / h - 0.5      (x likewise)
  3. Isolated kv_cache: the condition tokens alone run through all 28 blocks once at t=0
     (no text, no target); each block's post-RoPE K/V is captured. Every denoising call
     the text+target attend to [own K/V | cached ref K/V]; nothing attends back to refs.
  4. Known-region restore, done in VELOCITY space so it is sampler-agnostic: for kept
     tokens   v := (x_t - x0_known) / sigma   =>  every sampler's x0 = x0_known exactly.
     For Euler from pure noise this reproduces the reference pipeline's per-step
     "known + sigma_next * (noise - known)" restore exactly; it also holds under
     res_2m/res_2s/deis, ancestral eta and CFG (both passes get the same v).
     keep = NOT dilate(generated, seam_px); a 16x16 token is kept only if all its
     pixels are kept (matches the reference token rule).

Multi-stage (our addition): restore re-encodes the FULL-RES known canvas at each
stage's size, so preserved regions come back as the original pixels at S2/S3 when the
source has that resolution (`auto` restores only then). Adapter conditioning is per
stage (default S1 only - trained at <=1024 px targets).

Author: Eric Hiss (GitHub: EricRollei)
Mechanism credit: yijunwang2 (Krea 2 AnyPaint / Outpaint functional adapters,
pipeline code Apache-2.0); kv_cache isolated reference attention: ostris / ai-toolkit.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_LOG = "[EricKrea2-Paint]"


# -- pixel-side helpers (used by the node and the runtime) ----------------------------

def round16(v):
    return max(16, int(round(float(v) / 16.0)) * 16)


def median_rgb(pixels):
    """pixels: [N, 3] float 0..1 -> [3] median (grey if empty), like the reference helper."""
    if pixels.numel() == 0:
        return torch.tensor([127.0 / 255.0] * 3)
    q = (pixels * 255.0).round()
    return q.median(dim=0).values / 255.0


def resize_max_edge(img, max_edge):
    """img [1, H, W, 3] -> aspect-kept downscale to max edge, dims snapped DOWN to /16
    (reference: round(scale*side)//16*16, min 16). Antialiased bicubic ~ PIL LANCZOS."""
    _, h, w, _ = img.shape
    s = min(1.0, float(max_edge) / max(h, w))
    nh = max(16, int(round(h * s)) // 16 * 16)
    nw = max(16, int(round(w * s)) // 16 * 16)
    if (nh, nw) == (h, w):
        return img
    x = img.permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(nh, nw), mode="bicubic", antialias=True, align_corners=False)
    return x.clamp(0, 1).permute(0, 2, 3, 1)


def resize_image(img, h, w):
    _, ih, iw, _ = img.shape
    if (ih, iw) == (h, w):
        return img
    x = img.permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(h, w), mode="bicubic", antialias=(h < ih or w < iw),
                      align_corners=False)
    return x.clamp(0, 1).permute(0, 2, 3, 1)


def resize_mask(m, h, w):
    """m [H, W] in 0..1 -> nearest resize, binarized."""
    if tuple(m.shape) == (h, w):
        return (m > 0.5).float()
    x = F.interpolate(m[None, None].float(), size=(h, w), mode="nearest")[0, 0]
    return (x > 0.5).float()


def dilate(m, r):
    """binary [H, W] dilation with a (2r+1)^2 square (reference uses MORPH_RECT)."""
    r = int(r)
    if r <= 0:
        return m
    return F.max_pool2d(m[None, None], kernel_size=2 * r + 1, stride=1, padding=r)[0, 0]


def keep_token_mask(generated_px, seam_px, patch_px=16):
    """generated [H, W] (1 = generate, H/W multiples of 16) -> [L] bool keep per token."""
    gen = dilate(generated_px, seam_px)
    keep = 1.0 - gen
    h, w = keep.shape
    # exactly the reference rule: nearest-resize to latent res (1/8), then all-of-2x2 per token
    lat = F.interpolate(keep[None, None], size=(h // 8, w // 8), mode="nearest")[0, 0]
    k = lat.view(h // patch_px, 2, w // patch_px, 2).amin(dim=(1, 3))
    return (k > 0.5).reshape(-1)


def compact_kv(k, v, mask):
    """Drop masked-out keys explicitly and return (k, v, None) when every batch row shares
    the same key mask. Needed because diffusers' flash_varlen backend keeps key[:valid_len]
    (assumes the valid keys are a PREFIX); Krea2's [text(padded) | image | refs] layout has
    the padding in the MIDDLE, so that backend silently drops the LAST keys (image tail /
    appended reference keys) and keeps padded text instead. Returns the inputs unchanged
    (mask kept) when rows differ - the caller then forces SDPA."""
    if mask is None:
        return k, v, None, False
    m = mask.bool()
    m2 = m.reshape(m.shape[0], m.shape[-1]) if m.dim() >= 2 else m[None]
    if m2.shape[-1] != k.shape[1]:
        return k, v, mask, True
    if bool(m2.all()):
        return k, v, None, False
    if m2.shape[0] > 1 and not bool((m2 == m2[:1]).all()):
        return k, v, mask, True
    idx = m2[0].nonzero(as_tuple=False).squeeze(1)
    return k.index_select(1, idx), v.index_select(1, idx), None, False


# -- attention processor with an optional cached-KV extension ----------------------------

class PaintKVProcessor:
    """Replaces one block's Krea2AttnProcessor for the run.
    capture mode (precompute): records post-RoPE K/V, then attends normally.
    inject mode: appends the cached reference K/V as extra keys/values.
    Otherwise falls through to the original processor (bit-exact)."""

    def __init__(self, orig, runtime, idx):
        self.orig = orig
        self.rt = runtime
        self.idx = idx
        self._attention_backend = getattr(orig, "_attention_backend", None)
        self._parallel_config = getattr(orig, "_parallel_config", None)

    def _attend(self, q, k, v, mask, gqa):
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        k, v, mask, force_sdpa = compact_kv(k, v, mask)
        try:
            if force_sdpa:
                raise RuntimeError("per-row key masks differ - SDPA required")
            return dispatch_attention_fn(q, k, v, attn_mask=mask, enable_gqa=gqa,
                                         backend=self._attention_backend,
                                         parallel_config=self._parallel_config)
        except Exception as e:
            if not self.rt._warned_fallback:
                self.rt._warned_fallback = True
                print(f"{_LOG} attention backend refused the extended call ({type(e).__name__}: "
                      f"{str(e)[:120]}); using torch SDPA for these calls")
            qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
            if gqa and kt.shape[1] != qt.shape[1]:
                rep = qt.shape[1] // kt.shape[1]
                kt, vt = kt.repeat_interleave(rep, 1), vt.repeat_interleave(rep, 1)
            m = mask.bool() if mask is not None and mask.dtype != torch.bool else mask
            return F.scaled_dot_product_attention(qt, kt, vt, attn_mask=m).transpose(1, 2)

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        rt = self.rt
        capture = rt._capture is not None
        kv = rt._inject[self.idx] if (rt._inject is not None and not capture) else None
        if not capture and kv is None:
            return self.orig(attn, hidden_states, attention_mask, image_rotary_emb)
        from diffusers.models.embeddings import apply_rotary_emb
        gqa = attn.num_heads != attn.num_kv_heads
        q = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        k = attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        v = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        gate = attn.to_gate(hidden_states)
        q, k = attn.norm_q(q), attn.norm_k(k)
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
            k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
        if capture:
            rt._capture.append((k.detach(), v.detach()))
        else:
            B = q.shape[0]
            rk, rv = kv
            rk = rk.to(k.dtype).expand(B, -1, -1, -1)
            rv = rv.to(v.dtype).expand(B, -1, -1, -1)
            k = torch.cat([k, rk], dim=1)
            v = torch.cat([v, rv], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones(
                    tuple(attention_mask.shape[:-1]) + (rk.shape[1],))], dim=-1)
        out = self._attend(q, k, v, attention_mask, gqa)
        out = out.flatten(2, 3) * torch.sigmoid(gate)
        return attn.to_out[0](out)


# -- runtime ---------------------------------------------------------------------------

class PaintRuntime:
    """rt = PaintRuntime(pipe, bundle); rt.install(); rt.set_stage(n) ...; rt.remove()"""

    def __init__(self, pipe, bundle):
        self.pipe = pipe
        self.tr = pipe.transformer
        self.b = bundle
        self.stage = 1
        self.orig_forward = None
        self.orig_procs = {}
        self._capture = None
        self._inject = None
        self._warned_fallback = False
        self.known_cache = {}     # (gh, gw) -> (packed known [1, L, C] fp32, keep [L] bool)
        self.kv_cache = {}        # (stage, gh, gw) -> list of (k, v) per block
        self.n_calls = 0
        self.stats = {}

    # ---- per-stage switches ----
    def adapter_on(self):
        return (self.b.get("adapter", "none") != "none"
                and self.b.get("condition") is not None
                and float(self.b.get("adapter_mult", {}).get(self.stage, 0.0)) > 0)

    def restore_mode(self):
        return str(self.b.get("restore", {}).get(self.stage, "auto"))

    def install(self):
        if self.b.get("adapter", "none") != "none":
            for i, blk in enumerate(self.tr.transformer_blocks):
                self.orig_procs[i] = blk.attn.processor
                blk.attn.processor = PaintKVProcessor(blk.attn.processor, self, i)
        self.orig_forward = self.tr.forward
        orig = self.orig_forward
        rt = self

        def wrapped(hidden_states, encoder_hidden_states, timestep, position_ids,
                    encoder_attention_mask=None, attention_kwargs=None, return_dict=True, **extra):
            return rt._forward(orig, hidden_states, encoder_hidden_states, timestep, position_ids,
                               encoder_attention_mask, attention_kwargs, return_dict, extra)
        self.tr.forward = wrapped
        cw, ch = self.b["canvas"]
        print(f"{_LOG} installed: adapter={self.b.get('adapter')} canvas {cw}x{ch}, "
              f"seam {self.b.get('seam_px')}px, restore S1/S2/S3="
              f"{self.b['restore'].get(1)}/{self.b['restore'].get(2)}/{self.b['restore'].get(3)}, "
              f"adapter S1/S2/S3 x{self.b['adapter_mult'].get(1)}/{self.b['adapter_mult'].get(2)}/"
              f"{self.b['adapter_mult'].get(3)}")
        self.set_stage(1)

    def remove(self):
        if self.orig_forward is not None:
            self.tr.forward = self.orig_forward
            self.orig_forward = None
        for i, p in self.orig_procs.items():
            self.tr.transformer_blocks[i].attn.processor = p
        self.orig_procs = {}
        self._inject = None
        self.kv_cache.clear()
        self.known_cache.clear()
        if self.stats:
            print(f"{_LOG} removed after {self.n_calls} model calls; restore stats: "
                  + "; ".join(f"{k}: {v}" for k, v in self.stats.items()))

    def set_stage(self, n):
        self.stage = int(n)
        # LoRA weights change per stage -> cached ref K/V are stale
        self.kv_cache = {k: v for k, v in self.kv_cache.items() if k[0] == self.stage}
        print(f"{_LOG} stage {self.stage}: adapter {'ON' if self.adapter_on() else 'off'}, "
              f"restore {self.restore_mode()}")

    # ---- builders ----
    def _encode(self, img, device):
        from ._latent_utils import standard_encode
        packed, h, w = standard_encode(self.pipe, img.to(device))
        return packed.float(), h, w

    def _known_for(self, gh, gw, device):
        key = (gh, gw)
        if key not in self.known_cache:
            H, W = gh * 16, gw * 16
            known = self.b["known"]                       # [1, Hc, Wc, 3] full-res canvas
            kimg = resize_image(known, H, W)
            packed, _, _ = self._encode(kimg, device)
            gen = resize_mask(self.b["generated"], H, W)
            seam = max(0, int(round(float(self.b.get("seam_px", 32)) * W / float(self.b["canvas"][0]))))
            keep = keep_token_mask(gen, seam).to(device)
            self.known_cache[key] = (packed, keep)
            print(f"{_LOG} known canvas encoded at {W}x{H} ({gw}x{gh} tokens): "
                  f"{int(keep.sum())}/{keep.numel()} tokens kept, seam {seam}px")
        return self.known_cache[key]

    def _source_scale_ok(self, gh, gw):
        """auto restore: only when the source pixels cover the stage resolution."""
        sw, sh = self.b.get("source_px", self.b["canvas"])
        bx0, by0, bx1, by1 = self.b.get("src_box_norm", [0.0, 0.0, 1.0, 1.0])
        need_w = (bx1 - bx0) * gw * 16
        need_h = (by1 - by0) * gh * 16
        return sw >= 0.9 * need_w and sh >= 0.9 * need_h

    def _ref_tokens_and_pos(self, gh, gw, device, dtype):
        cond = self.b["condition"]
        if "cond_packed" not in self.b or self.b["cond_packed"] is None:
            packed, h, w = self._encode(cond, device)
            self.b["cond_packed"] = (packed.cpu(), h, w)
        packed, h, w = self.b["cond_packed"]
        rh, rw = h // 16, w // 16
        x0, y0, x1, y1 = (float(v) for v in self.b["bbox_norm"])
        ys = y0 * gh + (torch.arange(rh, device=device, dtype=torch.float32) + 0.5) * ((y1 - y0) * gh / rh) - 0.5
        xs = x0 * gw + (torch.arange(rw, device=device, dtype=torch.float32) + 0.5) * ((x1 - x0) * gw / rw) - 0.5
        ids = torch.zeros(rh, rw, 3, device=device)
        ids[..., 0] = 1.0
        ids[..., 1] = ys[:, None]
        ids[..., 2] = xs[None, :]
        return packed.to(device=device, dtype=dtype), ids.reshape(-1, 3)

    @torch.no_grad()
    def _precompute_kv(self, gh, gw, device, dtype):
        tok, pos = self._ref_tokens_and_pos(gh, gw, device, dtype)
        tr = self.tr
        ts = torch.zeros(tok.shape[0], device=device, dtype=dtype)
        temb = tr.time_embed(ts, dtype=dtype)
        mod = tr.time_mod_proj(F.gelu(temb, approximate="tanh"))
        rot = tr.rotary_emb(pos)
        kv = []
        tr._eric_paint_precompute = True   # control hooks skip these tokens
        try:
            h = tr.img_in(tok)
            for blk in tr.transformer_blocks:
                self._capture = []
                h = blk(h, mod, rot, None)
                kv.append(self._capture[0])
        finally:
            self._capture = None
            try:
                del tr._eric_paint_precompute
            except AttributeError:
                pass
        print(f"{_LOG} stage {self.stage}: reference K/V cached for {len(kv)} blocks "
              f"({tok.shape[1]} ref tokens registered on the {gw}x{gh} target grid)")
        return kv

    # ---- the wrapped forward ----
    def _forward(self, orig, hidden_states, ehs, timestep, pos, mask, attention_kwargs, return_dict, extra):
        self.n_calls += 1
        L = hidden_states.shape[1]
        img = pos[-L:]
        gh, gw = int(img[:, 1].max().item()) + 1, int(img[:, 2].max().item()) + 1
        if gh * gw != L:
            print(f"{_LOG} could not read the image grid ({gh}x{gw} != {L}); passing through")
            return orig(hidden_states=hidden_states, encoder_hidden_states=ehs, timestep=timestep,
                        position_ids=pos, encoder_attention_mask=mask,
                        attention_kwargs=attention_kwargs, return_dict=return_dict, **extra)
        device, dtype = hidden_states.device, hidden_states.dtype
        if self.adapter_on():
            key = (self.stage, gh, gw)
            if key not in self.kv_cache:
                self.kv_cache[key] = self._precompute_kv(gh, gw, device, dtype)
            self._inject = self.kv_cache[key]
        else:
            self._inject = None
        try:
            out = orig(hidden_states=hidden_states, encoder_hidden_states=ehs, timestep=timestep,
                       position_ids=pos, encoder_attention_mask=mask,
                       attention_kwargs=attention_kwargs, return_dict=return_dict, **extra)
        finally:
            self._inject = None

        mode = self.restore_mode()
        if mode == "auto":
            # full pin when the source has the pixels (S1 always); otherwise pin STRUCTURE only
            # (high-noise steps: composition / identity), then free the low-noise detail steps -
            # pinning to an upsampled small source all the way to sigma 0 forces blur, and no
            # pin at all lets a 0.8+ re-noise rewrite the preserved region (identity drift).
            mode = "on" if (self.stage == 1 or self._source_scale_ok(gh, gw)) else "structure"
        sig_now = float(timestep.reshape(-1)[0].float())
        rel = float(self.b.get("structure_release_sigma", 0.4))
        do_restore = mode == "on" or (mode == "structure" and sig_now > rel)
        sk = f"S{self.stage}:{mode}"
        if sk not in self.stats:
            self.stats[sk] = {"on": "pinned every step", "off": "not pinned",
                              "structure": f"pinned while sigma > {rel:g}, free below"}[mode]
            print(f"{_LOG} stage {self.stage}: restore {mode} ({self.stats[sk]})")
        if not do_restore:
            return out
        known, keep = self._known_for(gh, gw, device)
        v = out[0] if not return_dict else out.sample
        sigma = timestep.reshape(-1).float().to(device)
        sigma = sigma.expand(v.shape[0]) if sigma.numel() == 1 else sigma
        x = hidden_states.float()
        v_keep = (x - known.to(device)) / sigma.clamp_min(1e-6).view(-1, 1, 1)
        km = keep.view(1, -1, 1)
        v_new = torch.where(km, v_keep, v.float()).to(v.dtype)
        if not return_dict:
            return (v_new,) + tuple(out[1:])
        out.sample = v_new
        return out
