"""Numerical check of _edit_guidance's custom forward (split-LSE attention) against
independent dense references built from the stock diffusers Krea2 modules:
  T1 all-off custom path == stock forward
  T2 edit sources + ref_boost (incl. masked boost) == dense additive-bias attention
  T3 NegPiP value scaling == dense reference with scaled text V rows
  T4 NAG == naive two-sequence NAG (iljung1106 Krea2-NAG structure, MIT)
Tiny random model, fp32, CPU math path. Usage: python test_edit_guidance_math.py PACK_DIR"""
import importlib
import math
import sys
import types

import torch
import torch.nn.functional as F
from diffusers import Krea2Transformer2DModel
from diffusers.models.embeddings import apply_rotary_emb

PACK = sys.argv[1]
pkg = types.ModuleType("k2t"); pkg.__path__ = [PACK]; sys.modules["k2t"] = pkg
EG = importlib.import_module("k2t._edit_guidance")
torch.manual_seed(0)

tr = Krea2Transformer2DModel(in_channels=16, num_layers=3, attention_head_dim=32,
                             num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
                             timestep_embed_dim=32, text_hidden_dim=40, num_text_layers=3,
                             text_num_attention_heads=4, text_num_key_value_heads=4,
                             text_intermediate_size=32, axes_dims_rope=(8, 12, 12)).eval()
for p in tr.parameters():                       # zero-init tables -> randomize for a real test
    p.data.normal_(0, 0.05) if p.dim() >= 1 else None

gh, gw = 6, 8
L = gh * gw
T = 20
ehs = torch.randn(1, T, 3, 40)
mask = torch.ones(1, T, dtype=torch.bool); mask[:, 14:18] = False        # pad in the middle
hs = torch.randn(1, L, 16)
ts = torch.tensor([0.63])
img_pos = torch.zeros(gh, gw, 3); img_pos[..., 1] = torch.arange(gh)[:, None]
img_pos[..., 2] = torch.arange(gw)[None]; img_pos = img_pos.reshape(-1, 3)
pos = torch.cat([torch.zeros(T, 3), img_pos])


class Pipe:  # minimal stand-in
    transformer = tr


def make_rt(edit=None, guide=None):
    rt = EG.EditGuidanceRuntime(Pipe(), edit=edit, guide=guide)
    return rt


def stock():
    return tr(hidden_states=hs, encoder_hidden_states=ehs, timestep=ts, position_ids=pos,
              encoder_attention_mask=mask, return_dict=False)[0]


def dense_forward(ehs_, mask_, refs=None, bias_fn=None, vscale=None, vscale_blocks=None):
    """Reference: stock blocks, dense attention with an additive float bias."""
    temb = tr.time_embed(ts, dtype=hs.dtype)
    mod = tr.time_mod_proj(F.gelu(temb, approximate="tanh"))
    ctx = tr.txt_in(tr.text_fusion(ehs_, attention_mask=mask_[:, None, None, :]))
    toks = [r["packed"] for r in (refs or [])] + [hs]
    img = tr.img_in(torch.cat(toks, 1))
    h = torch.cat([ctx, img], 1)
    rows = [torch.zeros(ehs_.shape[1], 3)] + [r["pos"] for r in (refs or [])] + [img_pos]
    rot = tr.rotary_emb(torch.cat(rows))
    Lq = h.shape[1]
    keymask = torch.cat([mask_[0], torch.ones(Lq - ehs_.shape[1], dtype=torch.bool)])
    bias = torch.zeros(Lq, Lq)
    bias[:, ~keymask] = float("-inf")
    if bias_fn is not None:
        bias = bias + bias_fn(Lq)
    for bi, blk in enumerate(tr.transformer_blocks):
        m = (mod.unflatten(-1, (6, -1)) + blk.scale_shift_table).unbind(-2)
        a = blk.attn
        x = (1 + m[0]) * blk.norm1(h) + m[1]
        q = a.norm_q(a.to_q(x).unflatten(-1, (a.num_heads, a.head_dim)))
        k = a.norm_k(a.to_k(x).unflatten(-1, (a.num_kv_heads, a.head_dim)))
        v = a.to_v(x).unflatten(-1, (a.num_kv_heads, a.head_dim))
        if vscale is not None and bi in vscale_blocks:
            v = v.clone(); v[:, :ehs_.shape[1]] *= vscale.view(1, -1, 1, 1)
        q, k = apply_rotary_emb(q, rot, sequence_dim=1), apply_rotary_emb(k, rot, sequence_dim=1)
        rep = a.num_heads // a.num_kv_heads
        k, v = k.repeat_interleave(rep, 2), v.repeat_interleave(rep, 2)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           attn_mask=bias).transpose(1, 2)
        h = h + m[2] * a.to_out[0](o.flatten(2, 3) * torch.sigmoid(a.to_gate(x)))
        h = h + m[5] * blk.ff((1 + m[3]) * blk.norm2(h) + m[4])
    t0 = Lq - L
    return tr.final_layer(h[:, t0:], temb)


def err(a, b):
    return float((a - b).abs().max() / b.abs().max())


ok = True
with torch.no_grad():
    # T1
    rt = make_rt(guide={})
    y = rt._custom(hs, ehs, mask, ts, img_pos, False, False, None)
    e1 = err(y, stock()); print(f"T1 all-off vs stock: rel max err {e1:.2e}"); ok &= e1 < 1e-4

    # T2 sources + ref_boost (2 refs, last ref partly masked)
    rg = [(4, 8, 1.0, 0.0), (6, 6, 0.0, 1.0)]  # (gh, gw, off_h, off_w)
    refs = []
    for i, (rh, rw, oh, ow) in enumerate(rg):
        p = torch.zeros(rh, rw, 3); p[..., 0] = i + 1
        p[..., 1] = (torch.arange(rh) + oh)[:, None]; p[..., 2] = (torch.arange(rw) + ow)[None]
        refs.append({"packed": torch.randn(1, rh * rw, 16), "pos": p.reshape(-1, 3)})
    bm = torch.zeros(36, dtype=torch.bool); bm[10:20] = True
    refs[1]["boost_mask"] = bm
    edit = {"refs": refs, "ref_boost": 3.0, "ref_boost_a": 0.5, "fit_mode": "fit",
            "target_grid": (gh, gw)}
    rt = make_rt(edit=edit, guide={})
    y = rt._custom(hs, ehs, mask, ts, img_pos, True, False, None)

    def bias_fn(Lq):
        b = torch.zeros(Lq, Lq); t0 = Lq - L
        o1, o2 = T, T + 32
        b[t0:, o1:o1 + 32] = math.log(0.5)
        cols = o2 + bm.nonzero().squeeze(1)
        b[t0:, cols] = math.log(3.0)
        return b
    e2 = err(y, dense_forward(ehs, mask, refs, bias_fn)); print(f"T2 sources+ref_boost(mask) vs dense bias: {e2:.2e}")
    ok &= e2 < 1e-4

    # T3 NegPiP on blocks 1..2
    vm = torch.ones(T); vm[3:6] = -1.5
    rt = make_rt(guide={"negpip": True, "block_start": 1, "block_end": 2})
    y = rt._custom(hs, ehs, mask, ts, img_pos, False, False, vm)
    e3 = err(y, dense_forward(ehs, mask, vscale=vm, vscale_blocks={1, 2}))
    print(f"T3 NegPiP vs dense V-scaled: {e3:.2e}"); ok &= e3 < 1e-4

    # T4 NAG vs naive two-sequence reference
    Tn = 12
    neh = torch.randn(1, Tn, 3, 40); nmask = torch.ones(1, Tn, dtype=torch.bool); nmask[:, 9:] = False
    G = {"nag": True, "phi": 4.0, "tau": 2.5, "alpha": 0.25, "nag_stages": {1},
         "sigma_start": 1.0, "sigma_end": 0.0, "neg_cond": {"embeds": neh, "mask": nmask}}
    rt = make_rt(guide=G)
    y = rt._custom(hs, ehs, mask, ts, img_pos, False, True, None)

    def naive():
        temb = tr.time_embed(ts, dtype=hs.dtype)
        mod = tr.time_mod_proj(F.gelu(temb, approximate="tanh"))
        pt = tr.txt_in(tr.text_fusion(ehs, attention_mask=mask[:, None, None, :]))
        nt = tr.txt_in(tr.text_fusion(neh, attention_mask=nmask[:, None, None, :]))
        img = tr.img_in(hs)
        rot_p = tr.rotary_emb(torch.cat([torch.zeros(T, 3), img_pos]))
        rot_n = tr.rotary_emb(torch.cat([torch.zeros(Tn, 3), img_pos]))

        def raw(a, x, rot, km):
            q = a.norm_q(a.to_q(x).unflatten(-1, (a.num_heads, a.head_dim)))
            k = a.norm_k(a.to_k(x).unflatten(-1, (a.num_kv_heads, a.head_dim)))
            v = a.to_v(x).unflatten(-1, (a.num_kv_heads, a.head_dim))
            q, k = apply_rotary_emb(q, rot, sequence_dim=1), apply_rotary_emb(k, rot, sequence_dim=1)
            rep = a.num_heads // a.num_kv_heads
            k, v = k.repeat_interleave(rep, 2), v.repeat_interleave(rep, 2)
            bias = torch.zeros(x.shape[1], x.shape[1]); bias[:, ~km] = float("-inf")
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                               attn_mask=bias).transpose(1, 2)
            return o.flatten(2, 3), a.to_gate(x)
        kmp = torch.cat([mask[0], torch.ones(L, dtype=torch.bool)])
        kmn = torch.cat([nmask[0], torch.ones(L, dtype=torch.bool)])
        for blk in tr.transformer_blocks:
            m = (mod.unflatten(-1, (6, -1)) + blk.scale_shift_table).unbind(-2)
            P = torch.cat([pt, img], 1); N = torch.cat([nt, img], 1)
            xp = (1 + m[0]) * blk.norm1(P) + m[1]; xn = (1 + m[0]) * blk.norm1(N) + m[1]
            rp, gp = raw(blk.attn, xp, rot_p, kmp); rn, gn = raw(blk.attn, xn, rot_n, kmn)
            gi = EG.nag_combine(rp[:, T:], rn[:, Tn:], 4.0, 2.5, 0.25)
            rp = torch.cat([rp[:, :T], gi], 1)
            P = P + m[2] * blk.attn.to_out[0](rp * torch.sigmoid(gp))
            ntx = nt + m[2] * blk.attn.to_out[0](rn[:, :Tn] * torch.sigmoid(gn[:, :Tn]))
            P = P + m[5] * blk.ff((1 + m[3]) * blk.norm2(P) + m[4])
            nt = ntx + m[5] * blk.ff((1 + m[3]) * blk.norm2(ntx) + m[4])
            pt, img = P[:, :T], P[:, T:]
        return tr.final_layer(img, temb)
    e4 = err(y, naive()); print(f"T4 NAG vs naive two-sequence: {e4:.2e}"); ok &= e4 < 1e-4

    # T5 the wrapper routes: all-off call goes to the stock forward, bit-exact
    rt = make_rt(guide={}); rt.install()
    y5 = tr(hidden_states=hs, encoder_hidden_states=ehs, timestep=ts, position_ids=pos,
            encoder_attention_mask=mask, return_dict=False)[0]
    rt.remove()
    e5 = float((y5 - stock()).abs().max()); print(f"T5 installed, all off -> stock: abs {e5:.1e}"); ok &= e5 == 0
print("PASS" if ok else "FAIL")
