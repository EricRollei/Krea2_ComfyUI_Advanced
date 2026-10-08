# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 edit + negative-guidance runtime (installed by Multi-Stage Ultra per run)
====================================================================================
One forward wrapper on ``pipe.transformer`` that can, per model call:

* EDIT (krea2edit v1.2 / Krea 2 Identity Edit recipe): prepend the clean VAE-encoded
  source(s) as  [text | source_1(frame 1) | source_2(frame 2) | target(frame 0)],
  shared real timestep, source tokens placed by the v1.2 `fit` geometry (_edit_geom),
  output = target tokens only. Stage 1 only (trained <= 2 MP).
  ref_boost: + log(b) on target-query -> source-key logits (all blocks), optional mask.
* NAG (Normalized Attention Guidance, Chen et al. 2025; Krea2 placement after
  iljung1106/ComfyUI-Krea2-NAG, MIT): a negative text stream runs through the blocks
  next to the positive one; for the TARGET image queries
      z = z+ + phi (z+ - z-);  L1 ratio clamp to tau;  z = alpha z + (1 - alpha) z+
  on the raw attention output (before gate / to_out). Text and source queries keep the
  positive attention (edit-safe "target-only" rule).
* NegPiP: value rows of flagged text tokens multiplied by their (negative) weight in the
  selected joint blocks (optionally also in the text-fusion refiner blocks).

All three run on _split_attn: keys are attended group by group (pos-text | neg-text |
sources | image) and merged by log-sum-exp - ref_boost is a per-group log-weight, NAG's
positive and negative attentions share the image/source groups (only the small text
groups are extra), NegPiP is a value multiplier on the text group. Calls where nothing
is active go to the original forward untouched.

CFG: Ultra marks each transformer call with transformer._eric_cfg_pass ("pos"/"neg").
NAG + NegPiP act on the positive pass only. For edits, the negative pass also gets the
sources and (ground_negative) the grounded EMPTY instruction - the trained unconditional.

Author: Eric Hiss (GitHub: EricRollei)
Credits: krea2edit (lbouaraba, Apache-2.0); NAG (ChenDarYen, MIT; Krea2 port iljung1106,
MIT); NegPiP idea (hako-mikan).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import _split_attn as SA

_LOG = "[EricKrea2-EditGuide]"


def nag_combine(z_pos, z_neg, phi, tau, alpha):
    """NAG eqs. 7-10 on [B, L, H*D] (L1 norms over the full feature vector, accumulated in
    fp32; the elementwise work stays in the activation dtype - same result to bf16 rounding,
    a fraction of the memory traffic). Algebra: out = zp + alpha * (s * g - zp) with
    s = min(r, tau) / r."""
    phi, tau, alpha = float(phi), float(tau), float(alpha)
    g = torch.lerp(z_neg, z_pos, 1.0 + phi) if z_pos.dtype == z_neg.dtype else \
        z_pos + phi * (z_pos - z_neg)                                   # zp + phi (zp - zn)
    eps = torch.finfo(torch.float32).eps
    n_g = torch.linalg.vector_norm(g, ord=1, dim=-1, keepdim=True, dtype=torch.float32)
    n_p = torch.linalg.vector_norm(z_pos, ord=1, dim=-1, keepdim=True, dtype=torch.float32)
    r = n_g.clamp_min(eps) / n_p.clamp_min(eps)
    s = (r.clamp_max(tau) / r).to(z_pos.dtype)
    return torch.lerp(z_pos, g * s, alpha)


class _TFVmulProcessor:
    """Text-fusion refiner attention with NegPiP value scaling (token axis)."""

    def __init__(self, orig, rt):
        self.orig, self.rt = orig, rt
        self._attention_backend = getattr(orig, "_attention_backend", None)
        self._parallel_config = getattr(orig, "_parallel_config", None)

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        vm = self.rt._tf_vmul
        if vm is None or vm.numel() != hidden_states.shape[1]:
            return self.orig(attn, hidden_states, attention_mask, image_rotary_emb)
        q = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        k = attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        v = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        gate = attn.to_gate(hidden_states)
        q, k = attn.norm_q(q), attn.norm_k(k)
        v = v * vm.to(v.device, v.dtype).view(1, -1, 1, 1)
        if attention_mask is not None:
            idx = attention_mask.reshape(attention_mask.shape[0], -1)[0].bool().nonzero().squeeze(1)
            k, v = k.index_select(1, idx), v.index_select(1, idx)
        o, _ = SA.attend_lse(q, k, v)
        o = o.flatten(2, 3) * torch.sigmoid(gate)
        return attn.to_out[0](o)


class EditGuidanceRuntime:
    """rt = EditGuidanceRuntime(pipe, edit=..., guide=...); rt.install(); rt.set_stage(n);
    rt.remove().  `edit` / `guide` are the prepared dicts built by Ultra (see
    nodes/krea2_edit.py, nodes/krea2_guidance.py)."""

    def __init__(self, pipe, edit=None, guide=None):
        self.pipe = pipe
        self.tr = pipe.transformer
        self.edit = edit
        self.guide = guide or {}
        self.stage = 1
        self.orig_forward = None
        self.orig_tf = {}
        self._tf_vmul = None
        self.n_calls = 0
        self._said = set()

    # ---- lifecycle ----
    def install(self):
        self.orig_forward = self.tr.forward
        orig = self.orig_forward
        rt = self

        def wrapped(hidden_states, encoder_hidden_states, timestep, position_ids,
                    encoder_attention_mask=None, attention_kwargs=None, return_dict=True, **extra):
            return rt._forward(orig, hidden_states, encoder_hidden_states, timestep, position_ids,
                               encoder_attention_mask, attention_kwargs, return_dict, extra)
        self.tr.forward = wrapped
        g = self.guide
        if g.get("negpip") and g.get("negpip_text_fusion"):
            for i, blk in enumerate(self.tr.text_fusion.refiner_blocks):
                self.orig_tf[i] = blk.attn.processor
                blk.attn.processor = _TFVmulProcessor(blk.attn.processor, self)
        parts = []
        if self.edit:
            e = self.edit
            parts.append(f"edit {len(e['refs'])} source(s) fit={e['fit_mode']} ref_boost="
                         f"{e['ref_boost']:g}/{e['ref_boost_a']:g} on S1 grid {e['target_grid']}")
        if g.get("nag"):
            parts.append(f"NAG phi {g['phi']:g} tau {g['tau']:g} alpha {g['alpha']:g} stages "
                         f"{sorted(g['nag_stages'])} sigma [{g['sigma_end']:g}, {g['sigma_start']:g}]")
        if g.get("negpip"):
            parts.append(f"NegPiP blocks {g['block_start']}-{g['block_end']}"
                         + (" + text fusion" if g.get("negpip_text_fusion") else ""))
        print(f"{_LOG} installed: " + "; ".join(parts))

    def remove(self):
        if self.orig_forward is not None:
            # restore without leaving a bound-method cycle (tr -> forward -> tr) when the
            # original was the class method; keeps the model freeable by refcount alone
            if getattr(self.orig_forward, "__func__", None) is getattr(type(self.tr), "forward", None) \
                    and getattr(self.orig_forward, "__self__", None) is self.tr:
                self.tr.__dict__.pop("forward", None)
            else:
                self.tr.forward = self.orig_forward
            self.orig_forward = None
        for i, p in self.orig_tf.items():
            self.tr.text_fusion.refiner_blocks[i].attn.processor = p
        self.orig_tf = {}
        try:
            del self.tr._eric_cfg_pass
        except AttributeError:
            pass
        print(f"{_LOG} removed after {self.n_calls} model calls")

    def set_stage(self, n):
        self.stage = int(n)

    def _once(self, key, msg):
        if key not in self._said:
            self._said.add(key)
            print(f"{_LOG} {msg}")

    # ---- per-call decisions ----
    def _nag_on(self, sigma):
        g = self.guide
        if not g.get("nag") or self.stage not in g["nag_stages"]:
            return False
        return float(g["sigma_end"]) <= sigma <= float(g["sigma_start"])

    def _forward(self, orig, hs, ehs, timestep, pos, mask, attention_kwargs, return_dict, extra):
        self.n_calls += 1
        cfg_pass = getattr(self.tr, "_eric_cfg_pass", "pos")
        L = hs.shape[1]
        img_pos = pos[-L:]
        grid = (int(img_pos[:, 1].max().item()) + 1, int(img_pos[:, 2].max().item()) + 1)
        sigma = float(timestep.reshape(-1)[0].float())
        e, g = self.edit, self.guide

        # positive conditioning per stage (edit: S2/S3 may refine with a different prompt)
        if cfg_pass == "pos" and e and self.stage >= 2 and e.get("refine_cond") is not None:
            rc = e["refine_cond"]
            ehs, mask = rc["embeds"].to(hs.device), rc["mask"].to(hs.device)
        refs_on = bool(e) and self.stage == 1 and tuple(grid) == tuple(e["target_grid"])
        if bool(e) and self.stage == 1 and not refs_on:
            self._once("gridmiss", f"stage grid {grid} != edit grid {e['target_grid']} - sources "
                                   "not attached for this call")
        if cfg_pass == "neg" and refs_on and e.get("neg_cond") is not None:
            ehs, mask = e["neg_cond"]["embeds"].to(hs.device), e["neg_cond"]["mask"].to(hs.device)
        nag_on = cfg_pass == "pos" and self._nag_on(sigma)
        vmul = None
        if cfg_pass == "pos" and g.get("negpip"):
            vm = g.get("vmul_by_len", {}).get(int(ehs.shape[1]))
            if vm is not None:
                vmul = vm
            else:
                self._once(f"vm{ehs.shape[1]}", f"NegPiP: no weights for a {ehs.shape[1]}-token "
                                                 "conditioning - inactive for those calls")
        if not (refs_on or nag_on or vmul is not None):
            if pos.shape[0] != ehs.shape[1] + L:          # conditioning was swapped
                pos = self._repos(pos, ehs.shape[1], L)
            return orig(hidden_states=hs, encoder_hidden_states=ehs, timestep=timestep,
                        position_ids=pos, encoder_attention_mask=mask,
                        attention_kwargs=attention_kwargs, return_dict=return_dict, **extra)
        out = self._custom(hs, ehs, mask, timestep, img_pos, refs_on, nag_on, vmul)
        if not return_dict:
            return (out,)
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        return Transformer2DModelOutput(sample=out)

    @staticmethod
    def _repos(pos, txt_len, L):
        img = pos[-L:]
        return torch.cat([torch.zeros(txt_len, 3, device=pos.device, dtype=pos.dtype), img], 0)

    # ---- the custom forward ----
    @torch.no_grad()
    def _custom(self, hs, ehs, mask, timestep, img_pos, refs_on, nag_on, vmul):
        tr, e, g = self.tr, self.edit, self.guide
        from diffusers.models.embeddings import apply_rotary_emb
        device, dtype = hs.device, hs.dtype
        B, L, _ = hs.shape
        txt_len = ehs.shape[1]

        temb = tr.time_embed(timestep, dtype=dtype)
        mod = tr.time_mod_proj(F.gelu(temb, approximate="tanh"))

        def text_stream(emb, msk, vm=None):
            m4 = msk[:, None, None, :] if msk is not None else None
            self._tf_vmul = vm if (vm is not None and g.get("negpip_text_fusion")) else None
            try:
                c = tr.text_fusion(emb.to(device), attention_mask=m4)
            finally:
                self._tf_vmul = None
            return tr.txt_in(c)

        ctx = text_stream(ehs, mask, vmul)
        valid = (mask[0].bool() if mask is not None
                 else torch.ones(txt_len, dtype=torch.bool, device=device))
        t_idx = valid.nonzero().squeeze(1).to(device)

        # sources
        ref_tok, ref_rows, ref_lens = [], [], []
        if refs_on:
            for r in e["refs"]:
                ref_tok.append(r["packed"].to(device=device, dtype=dtype).expand(B, -1, -1))
                ref_rows.append(r["pos"].to(device))
                ref_lens.append(int(r["packed"].shape[1]))
        ref_len = sum(ref_lens)
        # target embedded alone (Control-LoRA's img_in hook expands exactly the target grid);
        # the source span separately, flagged so control hooks leave it untouched
        img = tr.img_in(hs)
        if ref_tok:
            tr._eric_ctrl_skip = True
            try:
                img = torch.cat([tr.img_in(torch.cat(ref_tok, dim=1)), img], dim=1)
            finally:
                tr._eric_ctrl_skip = False
        h = torch.cat([ctx, img], dim=1)
        rows = [torch.zeros(txt_len, 3, device=device)] + ref_rows + [img_pos.to(device).float()]
        rot = tr.rotary_emb(torch.cat(rows, 0))
        t0 = txt_len + ref_len                                  # first target token
        Lq = h.shape[1]

        # ref_boost log-weights per ref group (target rows only) + optional masked split
        boosts = []
        if refs_on:
            nref = len(ref_lens)
            for i in range(nref):
                b = float(e["ref_boost"]) if i == nref - 1 else float(e["ref_boost_a"])
                boosts.append(b)
        def boost_lw(b):
            if b == 1.0:
                return None
            lw = torch.zeros(1, Lq, 1, device=device)
            lw[:, t0:] = math.log(max(b, 1e-4))
            return lw

        # NAG negative stream
        hn = n_idx = None
        if nag_on:
            ng = g["neg_cond"]
            hn = text_stream(ng["embeds"], ng["mask"])
            n_idx = ng["mask"][0].bool().nonzero().squeeze(1).to(device)

        bs, be = int(g.get("block_start", 0)), int(g.get("block_end", 10_000))
        vm_dev = vmul.to(device=device, dtype=dtype)[t_idx] if vmul is not None else None
        H = tr.transformer_blocks[0].attn.num_heads

        for bi, blk in enumerate(tr.transformer_blocks):
            m = (mod.unflatten(-1, (6, -1)) + blk.scale_shift_table).unbind(-2)
            a = blk.attn
            x = (1.0 + m[0]) * blk.norm1(h) + m[1]
            q = a.to_q(x).unflatten(-1, (a.num_heads, a.head_dim))
            k = a.to_k(x).unflatten(-1, (a.num_kv_heads, a.head_dim))
            v = a.to_v(x).unflatten(-1, (a.num_kv_heads, a.head_dim))
            gate = a.to_gate(x)
            q, k = a.norm_q(q), a.norm_k(k)
            q = apply_rotary_emb(q, rot, sequence_dim=1)
            k = apply_rotary_emb(k, rot, sequence_dim=1)

            kt, vt = k[:, :txt_len].index_select(1, t_idx), v[:, :txt_len].index_select(1, t_idx)
            if vm_dev is not None and bs <= bi <= be:
                vt = vt * vm_dev.view(1, -1, 1, 1)
            # key segments (k, v, boost); adjacent segments with the same boost are attended
            # in ONE kernel call - splits only where the math needs them (boost / NAG text)
            segs = [] if hn is not None else [(kt, vt, 1.0)]
            off = txt_len
            for ri, rl in enumerate(ref_lens):
                kr, vr = k[:, off:off + rl], v[:, off:off + rl]
                bmask = e["refs"][ri].get("boost_mask") if refs_on else None
                if bmask is not None and boosts[ri] != 1.0:
                    bm = bmask.to(device)
                    i_in, i_out = bm.nonzero().squeeze(1), (~bm).nonzero().squeeze(1)
                    segs.append((kr.index_select(1, i_in), vr.index_select(1, i_in), boosts[ri]))
                    segs.append((kr.index_select(1, i_out), vr.index_select(1, i_out), 1.0))
                else:
                    segs.append((kr, vr, boosts[ri] if refs_on else 1.0))
                off += rl
            segs.append((k[:, t0:], v[:, t0:], 1.0))
            groups = []
            for kk, vv, b in segs:
                if groups and groups[-1][2] == b:
                    groups[-1] = (groups[-1][0] + [kk], groups[-1][1] + [vv], b)
                else:
                    groups.append(([kk], [vv], b))
            rest_parts = []
            for ks, vs, b in groups:
                kk = ks[0] if len(ks) == 1 else torch.cat(ks, 1)
                vv = vs[0] if len(vs) == 1 else torch.cat(vs, 1)
                p_ = SA.attend_lse(q, kk, vv)
                rest_parts.append((p_[0], p_[1], boost_lw(b)))
            if hn is not None:
                p_txt = SA.attend_lse(q, kt, vt)
                o = SA.merge([(p_txt[0], p_txt[1], None)] + rest_parts, out_dtype=dtype)
            else:
                o = SA.merge(rest_parts, out_dtype=dtype)                       # [B, Lq, H, D]
            ref_parts = rest_parts          # (sources + image groups; text excluded when NAG)

            if hn is not None:
                xn = (1.0 + m[0]) * blk.norm1(hn) + m[1]
                qn = a.norm_q(a.to_q(xn).unflatten(-1, (a.num_heads, a.head_dim)))
                kn = a.norm_k(a.to_k(xn).unflatten(-1, (a.num_kv_heads, a.head_dim)))
                vn = a.to_v(xn).unflatten(-1, (a.num_kv_heads, a.head_dim))
                gaten = a.to_gate(xn)
                kn_v, vn_v = kn.index_select(1, n_idx), vn.index_select(1, n_idx)
                # target queries vs the NEGATIVE text; sources/image groups shared
                pn = SA.attend_lse(q[:, t0:], kn_v, vn_v)
                z_neg = SA.merge([(pn[0], pn[1], None)]
                                 + [SA.slice_rows(p, t0, Lq) for p in ref_parts], out_dtype=dtype)
                z_pos = o[:, t0:]
                zg = nag_combine(z_pos.flatten(2, 3), z_neg.flatten(2, 3),
                                 g["phi"], g["tau"], g["alpha"]).unflatten(-1, (H, -1))
                o = torch.cat([o[:, :t0], zg], dim=1)
                # negative text stream's own attention: [neg text | sources | image]
                pnn = SA.attend_lse(qn, kn_v, vn_v)
                pni = SA.attend_lse(qn, k[:, txt_len:], v[:, txt_len:])        # sources + image
                on = SA.merge([(pnn[0], pnn[1], None), (pni[0], pni[1], None)], out_dtype=dtype)
                attn_n = a.to_out[0](on.flatten(2, 3) * torch.sigmoid(gaten))
                hn = hn + m[2] * attn_n
                hn = hn + m[5] * blk.ff((1.0 + m[3]) * blk.norm2(hn) + m[4])

            attn_out = a.to_out[0](o.flatten(2, 3) * torch.sigmoid(gate))
            h = h + m[2] * attn_out
            h = h + m[5] * blk.ff((1.0 + m[3]) * blk.norm2(h) + m[4])

        return tr.final_layer(h[:, t0:t0 + L], temb)
