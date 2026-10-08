# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 split-key attention (log-sum-exp merge)
==================================================
Attention over a key set that is the union of GROUPS (text | neg-text | ref1 | ref2 |
image ...) computed group by group and merged exactly:

    attn(Q, K, V) = sum_g  exp(lse_g - M) * o_g  /  sum_g exp(lse_g - M)

where (o_g, lse_g) is ordinary attention of Q over group g alone and M = max_g lse_g.
Adding a per-(query-row, group) log-weight  log w  to lse_g before the merge is the same
as adding log w to every attention logit of that group (= multiplying those keys'
post-softmax weight by w and renormalising) - that is krea2edit's ref_boost, exactly,
without materialising a dense L x L bias (which forces SDPA off flash and OOMs at 3 MP).

It also makes NAG cheap: the positive and negative attentions of the image queries share
the expensive image (and reference) groups; only the small text groups differ.

Kernel: flash-attn `flash_attn_func(..., return_attn_probs=True)` (returns softmax_lse,
supports GQA natively). Fallback: chunked fp32 math (CPU / no flash) - exact, slower.

Author: Eric Hiss (GitHub: EricRollei)
"""

from __future__ import annotations

import math

import torch

_STATE = {"flash": None, "warned": False}


def _flash_fn():
    if _STATE["flash"] is None:
        try:
            from flash_attn import flash_attn_func
            _STATE["flash"] = flash_attn_func
        except Exception:
            _STATE["flash"] = False
    return _STATE["flash"] or None


def _math_attn(q, k, v, chunk=2048):
    """q [B, Lq, H, D], k/v [B, Lk, Hkv, D] -> (o [B, Lq, H, D] in q.dtype, lse [B, Lq, H] fp32)."""
    B, Lq, H, D = q.shape
    rep = H // k.shape[2]
    kt = k.float().repeat_interleave(rep, dim=2).permute(0, 2, 3, 1)   # B H D Lk
    vt = v.float().repeat_interleave(rep, dim=2).permute(0, 2, 1, 3)   # B H Lk D
    scale = 1.0 / math.sqrt(D)
    outs, lses = [], []
    for s in range(0, Lq, chunk):
        qq = q[:, s:s + chunk].float().permute(0, 2, 1, 3)              # B H l D
        logits = torch.matmul(qq, kt) * scale                          # B H l Lk
        lse = torch.logsumexp(logits, dim=-1)                          # B H l
        p = torch.exp(logits - lse[..., None])
        outs.append(torch.matmul(p, vt).permute(0, 2, 1, 3))           # B l H D
        lses.append(lse.permute(0, 2, 1))                              # B l H
    return torch.cat(outs, 1).to(q.dtype), torch.cat(lses, 1)


def attend_lse(q, k, v):
    """One group: (o [B, Lq, H, D], lse [B, Lq, H] fp32). Empty key group -> (None, None)."""
    if k is None or k.shape[1] == 0:
        return None, None
    fn = _flash_fn() if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) else None
    if fn is not None:
        try:
            o, lse, _ = fn(q.contiguous(), k.contiguous(), v.contiguous(), return_attn_probs=True)
            return o, lse.transpose(1, 2).float()          # flash lse is [B, H, Lq]
        except Exception as e:
            if not _STATE["warned"]:
                _STATE["warned"] = True
                print(f"[EricKrea2-Attn] flash-attn LSE path failed ({type(e).__name__}: "
                      f"{str(e)[:120]}); using the exact fp32 math path (slower)")
    return _math_attn(q, k, v)


def merge(parts, out_dtype=None):
    """parts: list of (o [B,L,H,D], lse [B,L,H], logw) with logw None | float | tensor
    broadcastable to [B, L, H] (e.g. [1, L, 1]). Returns merged o [B, L, H, D]."""
    parts = [p for p in parts if p[0] is not None]
    if len(parts) == 1 and parts[0][2] is None:
        return parts[0][0] if out_dtype is None else parts[0][0].to(out_dtype)
    if len(parts) == 2:
        # closed form: o = o1 + (o2 - o1) * sigmoid(a2 - a1); weights in fp32, the blend in
        # the activation dtype (one fused lerp instead of fp32 copies of both outputs)
        (o1, l1, w1), (o2, l2, w2) = parts
        a1 = l1 if w1 is None else l1 + (w1 if torch.is_tensor(w1) else float(w1))
        a2 = l2 if w2 is None else l2 + (w2 if torch.is_tensor(w2) else float(w2))
        wt = torch.sigmoid(a2 - a1)[..., None]
        dt = out_dtype or o1.dtype
        if o1.dtype == torch.float32:
            return torch.lerp(o1, o2.float(), wt).to(dt)
        return torch.lerp(o1, o2.to(o1.dtype), wt.to(o1.dtype)).to(dt)
    adj = []
    for o, lse, lw in parts:
        a = lse if lw is None else lse + (lw if torch.is_tensor(lw) else float(lw))
        adj.append(a)
    m = torch.stack(adj, 0).amax(0)
    num = None
    den = None
    for (o, _, _), a in zip(parts, adj):
        w = torch.exp(a - m)                                            # [B, L, H]
        t = o.float() * w[..., None]
        num = t if num is None else num + t
        den = w if den is None else den + w
    out = num / den[..., None]
    return out.to(out_dtype or parts[0][0].dtype)


def slice_rows(part, a, b):
    o, lse, lw = part
    if o is None:
        return part
    if torch.is_tensor(lw) and lw.dim() >= 2 and lw.shape[1] > 1:
        lw = lw[:, a:b]
    return (o[:, a:b], lse[:, a:b], lw)
