# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.md.
"""
Krea2 transformer-block speed benchmark: bf16 vs native quantized matmuls.

Times ONE Krea2 block's work at real shapes (hidden 6144, 48 q / 12 kv heads x
128, MLP 16384) for a given token count:
  * the 8 Linears (q, k, v, o, gate, mlp gate/up/down) in each compute mode,
    INCLUDING the per-call activation quantization a quantized forward pays;
  * attention (SDPA, GQA) in bf16 - identical in every mode.
Then reports the per-block and whole-step speedup each mode would give.

Modes: bf16 | fp8 (per-tensor scaled, torch._scaled_mm) | mxfp8 | nvfp4
(comfy_kitchen kernels). Unsupported modes are reported, not fatal.

Usage: python bench_quant_matmul.py [--device cuda:0] [--tokens 13000 33300]
Author: Eric Hiss (GitHub: EricRollei)
"""

import argparse

import torch
import torch.nn.functional as F

HID, FF, HQ, HKV, HD = 6144, 16384, 48, 12, 128
LINEARS = {  # name: (out, in)
    "q": (HID, HID), "k": (HKV * HD, HID), "v": (HKV * HD, HID), "o": (HID, HID),
    "gate": (HID, HID), "mlp_gate": (FF, HID), "mlp_up": (FF, HID), "mlp_down": (HID, FF),
}


def timeit(fn, dev, iters=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(dev)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize(dev)
    return s.elapsed_time(e) / iters


def make_modes(dev):
    import comfy_kitchen as ck
    modes = {}

    def bf16(w):
        wb = w.to(torch.bfloat16)
        return lambda x: F.linear(x, wb)
    modes["bf16"] = bf16

    def fp8(w):
        ws = (w.abs().amax().float() / 448.0).clamp_min(1e-12)
        wq = (w.float() / ws).to(torch.float8_e4m3fn)
        wt = wq.t()

        def run(x):
            xs = (x.abs().amax().float() / 448.0).clamp_min(1e-12)
            xq = (x.float() / xs).to(torch.float8_e4m3fn)
            return torch._scaled_mm(xq, wt, scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16)
        return run
    modes["fp8"] = fp8

    def mxfp8(w):
        wq, wsc = ck.quantize_mxfp8(w.to(torch.bfloat16))

        def run(x):
            xq, xsc = ck.quantize_mxfp8(x)
            return ck.scaled_mm_mxfp8(xq, wq, xsc, wsc, out_dtype=torch.bfloat16)
        return run
    modes["mxfp8"] = mxfp8

    def nvfp4(w):
        wb = w.to(torch.bfloat16)
        wts = (wb.abs().amax().float() / (448.0 * 6.0))
        wq, wbs = ck.quantize_nvfp4(wb, wts)

        def run(x):
            xts = (x.abs().amax().float() / (448.0 * 6.0))
            xq, xbs = ck.quantize_nvfp4(x, xts)
            return ck.scaled_mm_nvfp4(xq, wq, xts, wts, xbs, wbs, out_dtype=torch.bfloat16)
        return run
    modes["nvfp4"] = nvfp4
    return modes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tokens", type=int, nargs="+", default=[13056, 33280])
    a = ap.parse_args()
    a.tokens = [(t + 127) // 128 * 128 for t in a.tokens]   # MX/NV kernels need /32 /16 rows
    dev = torch.device(a.device)
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(dev)} | torch {torch.__version__}")
    # Sanity: raw bf16 throughput. A busy GPU (e.g. ComfyUI generating) makes every
    # number below meaningless - refuse to pretend.
    sa = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    import time as _time
    _t0 = _time.perf_counter()   # ~2 s spin so the GPU leaves idle P8 clocks first
    while _time.perf_counter() - _t0 < 2.0:
        sa @ sa
        torch.cuda.synchronize(dev)
    tf = 2 * 8192 ** 3 / (timeit(lambda: sa @ sa, dev) / 1000) / 1e12
    print(f"sanity: bf16 8192^3 matmul = {tf:.0f} TFLOPS")
    if tf < 50:
        print("note: if this is far below your usual figure the GPU may be busy (ComfyUI "
              "running?) - compare ratios, not absolutes.")
    del sa
    weights = {n: (torch.randn(o, i, device=dev) * 0.02) for n, (o, i) in LINEARS.items()}
    modes = make_modes(dev)
    prepared = {}
    for m, mk in modes.items():
        try:
            prepared[m] = {n: mk(w) for n, w in weights.items()}
        except Exception as e:
            print(f"  mode {m}: weight prep unsupported ({type(e).__name__}: {e})")

    for T in a.tokens:
        x = torch.randn(T, HID, device=dev, dtype=torch.bfloat16)
        xf = torch.randn(T, FF, device=dev, dtype=torch.bfloat16)
        q = torch.randn(1, HQ, T, HD, device=dev, dtype=torch.bfloat16)
        # GQA kv expanded to full heads (what the model's attention sees) so the
        # flash / cuDNN kernels are eligible - enable_gqa fell back to the math path.
        kv = torch.randn(1, HKV, T, HD, device=dev, dtype=torch.bfloat16).repeat_interleave(HQ // HKV, dim=1)
        from torch.nn.attention import sdpa_kernel, SDPBackend

        def _att():
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                              SDPBackend.EFFICIENT_ATTENTION]):
                return F.scaled_dot_product_attention(q, kv, kv)
        att = timeit(_att, dev)
        print(f"\n=== {T} tokens (~{T * 256 / 1e6:.1f} MP) | attention {att:.2f} ms/block ===")
        base = None
        for m, fns in prepared.items():
            try:
                ms = 0.0
                for n, fn in fns.items():
                    inp = xf if n == "mlp_down" else x
                    ms += timeit(lambda fn=fn, inp=inp: fn(inp), dev)
            except Exception as e:
                print(f"  {m:6s} unsupported at runtime ({type(e).__name__}: {str(e)[:120]})")
                continue
            blk = ms + att
            if base is None:
                base = (ms, blk)
            print(f"  {m:6s} linears {ms:7.2f} ms  block {blk:7.2f} ms  "
                  f"linear x{base[0] / ms:4.2f}  block x{base[1] / blk:4.2f}  "
                  f"(28 blocks ~{blk * 28 / 1000:5.2f} s/step)")
        del x, xf, q, kv
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
