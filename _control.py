# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Eric Krea2 Control-LoRA runtime (depth / pose / canny ... "ControlNet-LoRA")
===========================================================================
Diffusers-side implementation of the Krea-2 Control-LoRA mechanism, mirrored from
the authors' reference pipeline (Tanmaypatil123/Krea-2-controlnet, pipeline.py;
ComfyUI port: facok/comfyui-krea2-controlnet):

  * the DiT input projection is EXPANDED 64 -> 128 input channels:
        img_in'( [noisy patches ; control patches] )      (trained weights + bias,
    both halves, from the checkpoint's ``first.weight [6144,128]`` / ``first.bias``)
  * rank-64 LoRA (alpha = rank, so scale 1) on every block's q/k/v/o/gate and
    mlp gate/up/down, stored as ``blocks.N.<leaf>.A [r,in]`` / ``.B [out,r]``
  * the control image (e.g. a depth map, near = white) is VAE-encoded with Krea2's
    latent normalization and packed exactly like the image latents - our packing
    is (c, p, q) per token, identical to the reference ``(c p q)`` rearrange.
  * the control latent enters CLEAN at every step (never noised).

Reference semantics of strength (``--lora-scale``): scales the block LoRA only; the
expanded input layer is always applied in full. We keep that, per stage:
strength 0 = control OFF for that stage (stock img_in, no LoRA delta).

Nothing is merged into the base weights: everything is forward hooks installed for
one generation and removed in ``finally`` (cancel-safe), so it composes with the
LoRA stack (PEFT-wrapped modules are hooked at the wrapper) and never leaks.

Grid discovery: a forward-pre-hook on the transformer reads the image rows of
``position_ids`` -> exact (h, w) token grid of THIS call, so every stage (including
VAE-hop stages whose size is only known after the hop) gets a control latent at
its own resolution, encoded lazily once per grid and cached for the run.

Author: Eric Hiss (GitHub: EricRollei). Mechanism credit: Tanmay Patil
(Krea-2-controlnet, weights Patil/Krea-2-depth-controlnet) and facok.
"""

import os

import torch
import torch.nn.functional as F

_LOG = "[EricKrea2-Control]"
_LEAVES = {  # checkpoint (comfy/original) leaf -> diffusers module path suffix
    "attn.wq": "attn.to_q", "attn.wk": "attn.to_k", "attn.wv": "attn.to_v",
    "attn.wo": "attn.to_out.0", "attn.gate": "attn.to_gate",
    "mlp.gate": "ff.gate", "mlp.up": "ff.up", "mlp.down": "ff.down",
}
_WEIGHT_CACHE = {}   # path -> (sig, parsed dict on CPU)


# -- checkpoint -----------------------------------------------------------------

def load_control_checkpoint(path):
    """Parse a Krea2 Control-LoRA file -> {'first_w', 'first_b', 'lora': {diffusers_path: (A, B)},
    'rank', 'meta'} (CPU fp32). Cached by file size+mtime."""
    from safetensors import safe_open
    st = os.stat(path)
    sig = (st.st_size, st.st_mtime)
    hit = _WEIGHT_CACHE.get(path)
    if hit and hit[0] == sig:
        return hit[1]
    sd = {}
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            sd[k] = f.get_tensor(k)

    def strip(k):
        for pre in ("model.diffusion_model.", "diffusion_model.", "transformer.", "model."):
            if k.startswith(pre):
                return k[len(pre):]
        return k
    sd = {strip(k): v for k, v in sd.items()}
    fw = sd.get("first.weight", sd.get("img_in.weight"))
    fb = sd.get("first.bias", sd.get("img_in.bias"))
    if fw is None or fw.ndim != 2:
        raise ValueError(f"{os.path.basename(path)}: no expanded input layer (first.weight) - "
                         f"not a Krea2 Control-LoRA checkpoint")
    lora, rank = {}, None
    for k, a in sd.items():
        if not k.endswith(".A") or not k.startswith("blocks."):
            continue
        b = sd.get(k[:-2] + ".B")
        if b is None:
            continue
        parts = k[:-2].split(".")            # blocks, N, attn|mlp, leaf
        leaf = ".".join(parts[2:])
        if leaf not in _LEAVES:
            continue
        lora[f"transformer_blocks.{parts[1]}.{_LEAVES[leaf]}"] = (a.float(), b.float())
        rank = int(a.shape[0])
    parsed = {"first_w": fw.float(), "first_b": None if fb is None else fb.float(),
              "lora": lora, "rank": rank, "meta": meta}
    _WEIGHT_CACHE.clear()                     # keep at most one ~0.9 GB file resident
    _WEIGHT_CACHE[path] = (sig, parsed)
    print(f"{_LOG} loaded {os.path.basename(path)}: input layer {tuple(fw.shape)}, "
          f"{len(lora)} LoRA modules (rank {rank}), meta {meta}")
    return parsed


# -- control image preprocessing ---------------------------------------------------

def preprocess_control_image(image, channel_mode="grayscale", normalize="per_image_minmax",
                             invert=False):
    """ComfyUI IMAGE [B,H,W,C] 0..1 -> [1,H,W,3] 0..1 (first frame), per the
    facok / reference conventions (depth: grayscale + per-image min-max, near=white)."""
    x = image[:1].float().clone()
    if x.shape[-1] == 4:
        x = x[..., :3]
    if channel_mode == "grayscale":
        g = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).unsqueeze(-1)
        x = g.repeat(1, 1, 1, 3)
    if normalize == "per_image_minmax":
        lo, hi = x.amin(), x.amax()
        x = (x - lo) / (hi - lo).clamp_min(1e-6)
    if invert:
        x = 1.0 - x
    return x.clamp(0.0, 1.0)


def _cover_crop(image, th, tw):
    """[1,H,W,3] -> [1,th,tw,3]: scale to cover, center crop (reference resize_center_crop)."""
    x = image.permute(0, 3, 1, 2)
    h, w = x.shape[-2:]
    s = max(th / h, tw / w)
    nh, nw = max(th, round(h * s)), max(tw, round(w * s))
    x = F.interpolate(x, size=(nh, nw), mode="bicubic", align_corners=False, antialias=True)
    t, l = (nh - th) // 2, (nw - tw) // 2
    return x[..., t:t + th, l:l + tw].clamp(0, 1).permute(0, 2, 3, 1).contiguous()


# -- runtime -------------------------------------------------------------------

class ControlRuntime:
    """Installs the hooks for one generation. Use as:
        rt = ControlRuntime(pipe, bundle); rt.install(); ...; rt.set_stage(n); ...; rt.remove()"""

    def __init__(self, pipe, bundle):
        self.pipe = pipe
        self.tr = pipe.transformer
        self.bundle = bundle
        self.strengths = {1: float(bundle.get("s1", 1.0)), 2: float(bundle.get("s2", 0.0)),
                          3: float(bundle.get("s3", 0.0))}
        self.stage = 1
        self.scale = 0.0
        self.grid = None          # (h, w) tokens of the current transformer call
        self.tokens = {}          # (h, w) -> packed control tokens on device
        self.handles = []
        self.w = None
        self.b = None
        self.lora_dev = {}
        self.n_ctrl_calls = 0

    # ---- install / remove ----
    def install(self):
        ck = load_control_checkpoint(self.bundle["path"])
        dev = next(self.tr.parameters()).device
        dt = self.tr.img_in.weight.dtype
        in_f = self.tr.img_in.in_features
        if ck["first_w"].shape != (self.tr.img_in.out_features, 2 * in_f):
            raise ValueError(f"{_LOG} input layer {tuple(ck['first_w'].shape)} does not match this "
                             f"transformer's img_in ({self.tr.img_in.out_features}, {in_f}) x2")
        self.w = ck["first_w"].to(dev, dt)
        self.b = None if ck["first_b"] is None else ck["first_b"].to(dev, dt)
        mods = dict(self.tr.named_modules())
        missing = 0
        for name, (a, b) in ck["lora"].items():
            m = mods.get(name)
            if m is None:
                missing += 1
                continue
            self.lora_dev[name] = (a.to(dev, dt), b.to(dev, dt))
            self.handles.append(m.register_forward_hook(self._make_lora_hook(name)))
        self.handles.append(self.tr.register_forward_pre_hook(self._grid_hook, with_kwargs=True))
        self.handles.append(self.tr.img_in.register_forward_hook(self._img_in_hook))
        print(f"{_LOG} installed: {len(self.lora_dev)} LoRA hooks"
              f"{f' ({missing} modules not found!)' if missing else ''}, expanded input layer; "
              f"strength S1/S2/S3 = {self.strengths[1]}/{self.strengths[2]}/{self.strengths[3]}")
        self.set_stage(1)

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()
        self.lora_dev.clear()
        self.tokens.clear()
        self.w = self.b = None
        if self.n_ctrl_calls:
            print(f"{_LOG} removed (control applied on {self.n_ctrl_calls} transformer calls)")

    def set_stage(self, idx):
        self.stage = int(idx)
        self.scale = self.strengths.get(self.stage, 0.0)
        state = f"ON, LoRA scale {self.scale}" if self.scale > 0 else "OFF (stock model)"
        print(f"{_LOG} stage {self.stage}: control {state}")

    # ---- hooks ----
    def _grid_hook(self, module, args, kwargs):
        try:
            hs = kwargs.get("hidden_states", args[0] if args else None)
            pos = kwargs.get("position_ids", args[3] if len(args) > 3 else None)
            if hs is None or pos is None:
                self.grid = None
                return None
            L = hs.shape[1]
            img = pos[-L:]
            self.grid = (int(img[:, 1].max().item()) + 1, int(img[:, 2].max().item()) + 1)
            if self.grid[0] * self.grid[1] != L:
                self.grid = None
        except Exception:
            self.grid = None
        return None

    def _control_tokens(self, gh, gw, device, dtype):
        key = (gh, gw)
        tok = self.tokens.get(key)
        if tok is None:
            from ._latent_utils import standard_encode
            px = _cover_crop(self.bundle["image"], gh * 16, gw * 16)
            packed, h, w = standard_encode(self.pipe, px)
            if packed.shape[1] != gh * gw:
                raise RuntimeError(f"{_LOG} control encode gave {packed.shape[1]} tokens, "
                                   f"expected {gh}x{gw}")
            tok = packed.to(device, dtype)
            self.tokens[key] = tok
            print(f"{_LOG} control image encoded at {w}x{h} px ({gh}x{gw} tokens)")
        return tok

    def _img_in_hook(self, module, args, output):
        if (self.scale <= 0 or self.grid is None or getattr(self.tr, "_eric_style_inversion", False)
                or getattr(self.tr, "_eric_paint_precompute", False)
                or getattr(self.tr, "_eric_ctrl_skip", False)):   # reference spans (edit / ref_latents)
            return None
        x = args[0]
        tb = getattr(self.tr, "_eric_style_target_b", None)   # style batch: [target, reference]
        if tb and x.shape[0] == 2 * tb:
            gh, gw = self.grid
            if x.shape[1] != gh * gw:
                return None
            ctrl = self._control_tokens(gh, gw, x.device, x.dtype)[:1].expand(tb, -1, -1)
            self.n_ctrl_calls += 1
            out_t = F.linear(torch.cat([x[:tb], ctrl], dim=-1), self.w.to(x.dtype),
                             None if self.b is None else self.b.to(x.dtype))
            return torch.cat([out_t, output[tb:]], dim=0)
        gh, gw = self.grid
        if x.ndim != 3 or x.shape[1] != gh * gw:
            return None          # e.g. a reference-latent span embedded separately
        ctrl = self._control_tokens(gh, gw, x.device, x.dtype)
        if ctrl.shape[0] != x.shape[0]:
            ctrl = ctrl[:1].expand(x.shape[0], -1, -1)
        self.n_ctrl_calls += 1
        return F.linear(torch.cat([x, ctrl], dim=-1), self.w.to(x.dtype),
                        None if self.b is None else self.b.to(x.dtype))

    def _make_lora_hook(self, name):
        def hook(module, args, output):
            if (self.scale <= 0 or getattr(self.tr, "_eric_style_inversion", False)
                    or getattr(self.tr, "_eric_paint_precompute", False)):
                return None
            a, b = self.lora_dev[name]
            x = args[0]
            tb = getattr(self.tr, "_eric_style_target_b", None)
            if tb and x.shape[0] == 2 * tb:           # style batch: control the target rows only
                d = F.linear(F.linear(x[:tb], a.to(x.dtype)), b.to(x.dtype)) * self.scale
                return torch.cat([output[:tb] + d, output[tb:]], dim=0)
            return output + (F.linear(F.linear(x, a.to(x.dtype)), b.to(x.dtype)) * self.scale)
        return hook
