# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md - EXCEPT the ported mechanism noted below.
#
# Mechanism ported from nkxx188/ComfyUI-Krea2-StyleTransfer (nodes.py, "controlled"
# single-reference route + flowturbo_pc reference trajectory), MIT License,
# Copyright (c) 2026 jieg9341-lab. Re-implemented for the diffusers Krea2 transformer.
"""
Eric Krea2 Style Reference runtime (training-free style transfer)
=================================================================
Per transformer call the batch becomes [target, reference]:

  * the reference is the style image, VAE-encoded at THIS call's grid (cover-crop),
    taken up the noise schedule along a model-guided trajectory (Heun predictor-
    corrector on the model's own velocity, blended `gamma` with the straight-line prior
    (1-s)*ref + s*eps) so it sits on-manifold at the target's noise level;
  * in the active blocks (default 7-27) the target's queries attend to
    [target K ; reference-image K] / [target V ; reference-image V] in one softmax;
  * the reference K is re-weighted per RoPE frequency band: high-frequency (positional)
    bands fade 1.04 -> 0 over the run, low-frequency bands 1.0 -> 1.10, then x ref_k.
    Early the target can align to the reference layout; late it can no longer copy
    positions (no content leakage) but still pulls palette / stroke / texture;
  * target image Q,K are AdaIN-matched toward the reference's token statistics;
  * value path: target-V AdaIN toward ref-V, mixed with raw ref-V (`ref_value_mix`).

Deliberate deviations from the original (see docs/spec_style_transfer_2026-10-07.md):
  D1 trajectory on an internal grid + linear interpolation (our RK/hybrid samplers
     evaluate at sigmas the wrapper can't know in advance);
  D2 the native attention is skipped when attention_mix >= 1 (theirs computes it x0);
  D3 the reference branch uses each call's own text states;
  D4 run progress estimated from sigma via the inverse Turbo shift (theirs: index in
     the sampler's sigma list, which the wrapper doesn't have).

Everything is installed for ONE generation and removed in `finally` (forward wrapper +
per-block attention processors). Strength 0 for a stage = stock model for that stage.

Author: Eric Hiss (GitHub: EricRollei)
"""

import math

import torch
import torch.nn.functional as F

_LOG = "[EricKrea2-Style]"

PRESET_RECOMMENDED = {
    "style_strength": 1.0, "value_adain_strength": 0.65, "ref_value_mix": 1.0,
    "ref_k_strength": 1.06, "trajectory": "model_pc", "gamma": 0.5, "beta": 2.5,
    "high_scale_start": 1.04, "high_scale_end": 0.0, "low_scale_start": 1.0,
    "low_scale_end": 1.10, "adain_strength": 0.85, "blocks": "7-27",
}


# -- small helpers (ported) -------------------------------------------------------

def parse_blocks(spec):
    out = set()
    for raw in str(spec or "").replace(";", ",").split(","):
        p = raw.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(p))
    return out


def _axes_dims(head_dim):
    hd = int(head_dim)
    axes = [hd - 12 * (hd // 16), 6 * (hd // 16), 6 * (hd // 16)]
    return axes if sum(axes) == hd and all(v > 0 for v in axes) else [hd]


def freq_scale_vector(head_dim, high, low, beta, device, dtype):
    """Per-channel K scale over the interleaved RoPE pairs: frame axis at `low`; the
    h/w axes run high-frequency (first pairs, `high`) -> low-frequency (`low`) on a
    power-`beta` curve. Ported 1:1 (diffusers uses the same interleaved pairs)."""
    axes = _axes_dims(head_dim)
    pieces = []
    for i, d in enumerate(axes):
        pairs = d // 2
        if len(axes) >= 2 and i == 0:
            s = torch.full((pairs,), float(low), device=device, dtype=torch.float32)
        else:
            x = (torch.linspace(0.0, 1.0, pairs, device=device, dtype=torch.float32)
                 if pairs > 1 else torch.zeros(1, device=device))
            s = float(high) + (float(low) - float(high)) * x.pow(float(beta))
        pieces.append(s.repeat_interleave(2))
        if d % 2:
            pieces.append(torch.ones(1, device=device))
    v = torch.cat(pieces)[: int(head_dim)]
    if v.numel() < head_dim:
        v = F.pad(v, (0, int(head_dim) - v.numel()), value=1.0)
    return v.to(dtype)


def _adain_seq(target, style, eps=1e-6):
    """AdaIN over the token dim (dim 1 of [B, L, H, D])."""
    tm, sm = target.mean(dim=1, keepdim=True), style.mean(dim=1, keepdim=True)
    ts = target.float().var(dim=1, keepdim=True, unbiased=False).add(eps).sqrt().to(target.dtype)
    ss = style.float().var(dim=1, keepdim=True, unbiased=False).add(eps).sqrt().to(target.dtype)
    return (target - tm) / ts * ss + sm


def _t_unshift(sigma, mu=1.15):
    """Inverse of the Krea2 exponential time shift: sigma -> linear t (D4)."""
    a = math.exp(mu)
    s = min(max(float(sigma), 0.0), 1.0)
    return s / max(s + a * (1.0 - s), 1e-9)


def _shift(t, mu=1.15):
    a = math.exp(mu)
    t = min(max(float(t), 0.0), 1.0)
    if t <= 0:
        return 0.0
    return a / (a + 1.0 / t - 1.0)


# -- styled attention processor ---------------------------------------------------

class StyledAttnProcessor:
    """Replaces one block's Krea2AttnProcessor for the run. Falls through to the
    original processor whenever the style batch is not active."""

    def __init__(self, orig, runtime, block_idx):
        self.orig = orig
        self.rt = runtime
        self.block_idx = block_idx
        self._attention_backend = getattr(orig, "_attention_backend", None)
        self._parallel_config = getattr(orig, "_parallel_config", None)

    def _attend(self, q, k, v, mask, gqa):
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from ._paint import compact_kv   # flash_varlen prefix-mask bug: see _paint.compact_kv
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
                print(f"{_LOG} attention backend refused the styled call ({type(e).__name__}: "
                      f"{str(e)[:120]}); using torch SDPA for styled attention")
            qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
            if gqa and kt.shape[1] != qt.shape[1]:
                rep = qt.shape[1] // kt.shape[1]
                kt, vt = kt.repeat_interleave(rep, 1), vt.repeat_interleave(rep, 1)
            m = mask.bool() if mask is not None and mask.dtype != torch.bool else mask
            return F.scaled_dot_product_attention(qt, kt, vt, attn_mask=m).transpose(1, 2)

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        rt = self.rt
        c = rt.cur
        if rt.bypass or c is None or hidden_states.shape[0] != 2 * c["tb"]:
            return self.orig(attn, hidden_states, attention_mask, image_rotary_emb)
        from diffusers.models.embeddings import apply_rotary_emb
        tb, T = c["tb"], c["text_len"]
        gqa = attn.num_heads != attn.num_kv_heads

        q = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        k = attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        v = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        gate = attn.to_gate(hidden_states)
        q, k = attn.norm_q(q), attn.norm_k(k)
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
            k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)

        # target Q/K: AdaIN of the image tokens toward the reference's statistics
        qt, kt = q[:tb], k[:tb]
        a = c["adain"]
        if a > 0:
            qt = qt.clone()
            kt = kt.clone()
            qt[:, T:] = qt[:, T:] * (1 - a) + _adain_seq(qt[:, T:], q[tb:, T:]) * a
            kt[:, T:] = kt[:, T:] * (1 - a) + _adain_seq(kt[:, T:], k[tb:, T:]) * a

        # reference image K, re-weighted per RoPE band; reference V (controlled)
        sv = freq_scale_vector(attn.head_dim, c["high"], c["low"], c["beta"], k.device, k.dtype)
        ref_k = k[tb:, T:] * sv.view(1, 1, 1, -1) * c["ref_k"]
        ref_v_raw = v[tb:, T:]
        tv = v[:tb, T:]
        va = c["value_adain"]
        base = tv * (1 - va) + _adain_seq(tv, ref_v_raw) * va
        mixv = c["ref_value_mix"]
        ref_v = base * (1 - mixv) + ref_v_raw * mixv

        k_cat = torch.cat([kt, ref_k], dim=1)
        v_cat = torch.cat([v[:tb], ref_v], dim=1)
        m_t = m_r = None
        if attention_mask is not None:
            m_t = torch.cat([attention_mask[:tb], attention_mask.new_ones(
                (tb,) + tuple(attention_mask.shape[1:-1]) + (ref_k.shape[1],))], dim=-1)
            m_r = attention_mask[tb:]
        out_t = self._attend(qt, k_cat, v_cat, m_t, gqa)
        mix = c["mix"]
        if mix < 1.0:                                   # D2: native only when it matters
            native = self._attend(qt, kt, v[:tb], attention_mask[:tb]
                                  if attention_mask is not None else None, gqa)
            out_t = native * (1.0 - mix) + out_t * mix
        out_r = self._attend(q[tb:], k[tb:], v[tb:], m_r, gqa)
        out = torch.cat([out_t, out_r], dim=0).flatten(2, 3)
        out = out * torch.sigmoid(gate)
        rt.n_styled += 1
        return attn.to_out[0](out)


# -- runtime ------------------------------------------------------------------------

class StyleRuntime:
    """Use as: rt = StyleRuntime(pipe, bundle); rt.install(); rt.set_stage(n) ...; rt.remove()"""

    def __init__(self, pipe, bundle):
        self.pipe = pipe
        self.tr = pipe.transformer
        self.b = bundle
        self.p = dict(bundle["params"])
        self.mults = {1: float(bundle.get("s1", 1.0)), 2: float(bundle.get("s2", 0.0)),
                      3: float(bundle.get("s3", 0.0))}
        self.stage = 1
        self.strength = 0.0
        self.cur = None
        self.bypass = False
        self.orig_forward = None
        self.orig_procs = {}
        self.clean = {}        # (h, w) -> clean packed reference latent [1, L, C]
        self.traj = {}         # (stage, h, w) -> (sigmas list asc, states list)
        self.n_styled = 0
        self.n_calls = 0
        self._warned_fallback = False

    # ---- install / remove ----
    def install(self):
        blocks = parse_blocks(self.p.get("blocks", "7-27"))
        n = len(self.tr.transformer_blocks)
        for i in sorted(blocks):
            if 0 <= i < n:
                attn = self.tr.transformer_blocks[i].attn
                self.orig_procs[i] = attn.processor
                attn.processor = StyledAttnProcessor(attn.processor, self, i)
        self.orig_forward = self.tr.forward
        orig = self.orig_forward
        rt = self

        def wrapped(hidden_states, encoder_hidden_states, timestep, position_ids,
                    encoder_attention_mask=None, attention_kwargs=None, return_dict=True):
            return rt._forward(orig, hidden_states, encoder_hidden_states, timestep,
                               position_ids, encoder_attention_mask, attention_kwargs, return_dict)
        self.tr.forward = wrapped
        print(f"{_LOG} installed on {len(self.orig_procs)} blocks ({self.p.get('blocks')}); "
              f"strength {self.p['style_strength']}, S1/S2/S3 x{self.mults[1]}/{self.mults[2]}/"
              f"{self.mults[3]}, trajectory {self.p['trajectory']} ({self.p['trajectory_steps']} pts)")
        self.set_stage(1)

    def remove(self):
        if self.orig_forward is not None:
            self.tr.forward = self.orig_forward
            self.orig_forward = None
        for i, proc in self.orig_procs.items():
            self.tr.transformer_blocks[i].attn.processor = proc
        self.orig_procs.clear()
        self.clean.clear()
        self.traj.clear()
        self.cur = None
        for attr in ("_eric_style_target_b", "_eric_style_inversion"):
            try:
                delattr(self.tr, attr)
            except AttributeError:
                pass
        if self.n_calls:
            print(f"{_LOG} removed (styled {self.n_calls} transformer calls, "
                  f"{self.n_styled} block-attentions)")

    def set_stage(self, idx):
        self.stage = int(idx)
        self.strength = float(self.p["style_strength"]) * self.mults.get(self.stage, 0.0)
        print(f"{_LOG} stage {self.stage}: "
              + (f"style ON, strength {self.strength:.2f}" if self.strength > 0 else "OFF (stock model)"))

    # ---- reference preparation ----
    def _clean_ref(self, gh, gw, device, dtype):
        key = (gh, gw)
        z = self.clean.get(key)
        if z is None:
            from ._latent_utils import standard_encode
            from ._control import _cover_crop
            px = _cover_crop(self.b["image"], gh * 16, gw * 16)
            packed, h, w = standard_encode(self.pipe, px)
            if packed.shape[1] != gh * gw:
                raise RuntimeError(f"{_LOG} reference encode gave {packed.shape[1]} tokens, "
                                   f"expected {gh}x{gw}")
            z = packed[:1].to(device, dtype)
            self.clean[key] = z
            print(f"{_LOG} reference encoded at {w}x{h} px ({gh}x{gw} tokens)")
        return z

    def _velocity(self, orig, z, sigma, ehs, mask, pos):
        ts = torch.full((z.shape[0],), float(sigma), device=z.device, dtype=z.dtype)
        self.bypass = True
        self.tr._eric_style_inversion = True     # control hooks stay off for the reference path
        try:
            with torch.no_grad():
                return orig(hidden_states=z, encoder_hidden_states=ehs, timestep=ts,
                            position_ids=pos, encoder_attention_mask=mask,
                            attention_kwargs=None, return_dict=False)[0].to(z.dtype)
        finally:
            self.bypass = False
            self.tr._eric_style_inversion = False

    def _build_traj(self, orig, ref0, s_top, ehs, mask, pos):
        """States of the reference at increasing sigmas 0 .. s_top (D1: own grid)."""
        n = max(2, int(self.p["trajectory_steps"]))
        t_top = _t_unshift(s_top)
        grid = sorted({0.0} | {_shift(t_top * i / n) for i in range(1, n + 1)} | {float(s_top)})
        g = torch.Generator(device=ref0.device)
        g.manual_seed(42)
        eps = torch.randn(ref0.shape, generator=g, device=ref0.device, dtype=torch.float32).to(ref0.dtype)
        gamma = float(self.p["gamma"])
        prior = lambda s: (1.0 - s) * ref0 + s * eps
        states = [ref0.clone()]
        if self.p["trajectory"] == "linear":
            states = [prior(s) for s in grid]
            return grid, states
        # model-guided predictor-corrector on a grid with midpoints (flowturbo_pc analog)
        fine = [grid[0]]
        for a, b in zip(grid[:-1], grid[1:]):
            fine += [0.5 * (a + b), b]
        keep = set(round(s, 7) for s in grid)
        z = ref0.clone()
        v0 = self._velocity(orig, z, fine[0], ehs, mask, pos)
        out = {round(grid[0], 7): z.clone()}
        for sp, sc in zip(fine[:-1], fine[1:]):
            d = sc - sp
            zp = z + d * v0
            v1 = self._velocity(orig, zp, sc, ehs, mask, pos)
            zm = z + 0.5 * d * (v0 + v1)
            z = gamma * zm + (1.0 - gamma) * prior(sc)
            v0 = v1
            if round(sc, 7) in keep:
                out[round(sc, 7)] = z.clone()
        states = [out[round(s, 7)] for s in grid]
        print(f"{_LOG} stage {self.stage} reference trajectory: {len(grid)} states to "
              f"sigma {s_top:.3f} ({2 * (len(grid) - 1) + 1} model evals)")
        return grid, states

    def _ref_at(self, key, sigma):
        grid, states = self.traj[key]
        s = min(max(float(sigma), grid[0]), grid[-1])
        for i in range(1, len(grid)):
            if s <= grid[i]:
                a, b = grid[i - 1], grid[i]
                w = 0.0 if b <= a else (s - a) / (b - a)
                return states[i - 1] * (1.0 - w) + states[i] * w
        return states[-1]

    # ---- the wrapped forward ----
    def _forward(self, orig, hidden_states, ehs, timestep, pos, mask, attention_kwargs, return_dict):
        if self.strength <= 0 or self.bypass:
            return orig(hidden_states=hidden_states, encoder_hidden_states=ehs, timestep=timestep,
                        position_ids=pos, encoder_attention_mask=mask,
                        attention_kwargs=attention_kwargs, return_dict=return_dict)
        B, L, _ = hidden_states.shape
        img = pos[-L:]
        gh, gw = int(img[:, 1].max().item()) + 1, int(img[:, 2].max().item()) + 1
        if gh * gw != L:
            raise RuntimeError(f"{_LOG} could not read the image grid ({gh}x{gw} != {L} tokens)")
        sigma = float(timestep.flatten()[0].float().item())
        key = (self.stage, gh, gw)
        if key not in self.traj:
            ref0 = self._clean_ref(gh, gw, hidden_states.device, hidden_states.dtype)
            self.traj[key] = self._build_traj(orig, ref0, max(sigma, 1e-3), ehs[:1],
                                              None if mask is None else mask[:1], pos)
            self.top = getattr(self, "top", {})
            self.top[key] = sigma
        ref = self._ref_at(key, sigma).expand(B, -1, -1)

        s = self.strength
        p = self.p
        t_top = _t_unshift(self.top[key])
        prog = 1.0 - _t_unshift(sigma) / max(t_top, 1e-9)            # D4
        prog = min(max(prog, 0.0), 1.0)
        lerp = lambda x, y: x + (y - x) * prog
        high = lerp(1.0 + (p["high_scale_start"] - 1.0) * min(s, 1.5), p["high_scale_end"])
        low = lerp(p["low_scale_start"], 1.0 + (p["low_scale_end"] - 1.0) * s)
        self.cur = {"tb": B, "text_len": ehs.shape[1], "high": high, "low": low,
                    "beta": p["beta"], "ref_k": p["ref_k_strength"],
                    "adain": max(0.0, min(1.0, p["adain_strength"] * min(s, 1.25))),
                    "value_adain": p["value_adain_strength"], "ref_value_mix": p["ref_value_mix"],
                    "mix": max(0.0, min(1.0, s))}
        self.tr._eric_style_target_b = B     # lets the control hooks act on target rows only
        ts = timestep.reshape(-1)
        ts = ts.expand(B) if ts.numel() == 1 else ts
        try:
            out = orig(hidden_states=torch.cat([hidden_states, ref.to(hidden_states.dtype)], 0),
                       encoder_hidden_states=torch.cat([ehs, ehs], 0),
                       timestep=torch.cat([ts, ts], 0), position_ids=pos,
                       encoder_attention_mask=None if mask is None else torch.cat([mask, mask], 0),
                       attention_kwargs=attention_kwargs, return_dict=False)[0]
        finally:
            self.cur = None
            try:
                delattr(self.tr, "_eric_style_target_b")
            except AttributeError:
                pass
        self.n_calls += 1
        out = out[:B]
        if return_dict:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput
            return Transformer2DModelOutput(sample=out)
        return (out,)
