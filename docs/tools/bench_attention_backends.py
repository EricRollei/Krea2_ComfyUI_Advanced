"""Attention-kernel bench at Krea2 shapes (B=1, 48 q heads / 12 kv heads, D=128,
L = 512 text + image tokens): flash-attn 2 (current), torch SDPA (flash / cuDNN / efficient),
SageAttention 2 (auto + explicit int8/fp8 kernels), xformers. Speed (ms per call) and accuracy vs an
fp32 reference on a query subsample (rel L2 err + cosine). Usage: python bench_attention_backends.py"""
import math, time
import torch
import torch.nn.functional as F

dev = "cuda"
torch.manual_seed(0)
H, HK, D = 48, 12, 128
SIZES = {"1.5MP": 512 + 6084, "3MP": 512 + 12288, "8.4MP": 512 + 33024}


def mk(L):
    # realistic-ish scale: q/k RMS-normed (Krea2 uses q/k RMSNorm) -> unit-variance rows
    q = torch.randn(1, L, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, L, HK, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, L, HK, D, device=dev, dtype=torch.bfloat16)
    return q, k, v


def ref_rows(q, k, v, idx):
    rep = H // HK
    qq = q[:, idx].float().permute(0, 2, 1, 3)
    kk = k.float().repeat_interleave(rep, 2).permute(0, 2, 3, 1)
    vv = v.float().repeat_interleave(rep, 2).permute(0, 2, 1, 3)
    p = torch.softmax(qq @ kk / math.sqrt(D), -1)
    return (p @ vv).permute(0, 2, 1, 3)


def backends():
    out = {}
    try:
        from flash_attn import flash_attn_func
        out["flash_attn2 (current)"] = lambda q, k, v: flash_attn_func(q, k, v)
    except Exception as e:
        print("flash_attn missing", e)
    rep = H // HK

    def sdpa(kind):
        from torch.nn.attention import sdpa_kernel, SDPBackend
        b = {"flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION,
             "efficient": SDPBackend.EFFICIENT_ATTENTION}[kind]

        def f(q, k, v):
            with sdpa_kernel(b):
                return F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.repeat_interleave(rep, 2).transpose(1, 2),
                    v.repeat_interleave(rep, 2).transpose(1, 2)).transpose(1, 2)
        return f
    for kind in ("flash", "cudnn", "efficient"):
        out[f"torch sdpa {kind}"] = sdpa(kind)
    try:
        import sageattention as sa
        out["sage2 auto"] = lambda q, k, v: sa.sageattn(q, k, v, tensor_layout="NHD")
        out["sage2 int8qk fp16pv (triton)"] = lambda q, k, v: sa.sageattn_qk_int8_pv_fp16_triton(q, k, v, tensor_layout="NHD")
        out["sage2 int8qk fp8pv (cuda)"] = lambda q, k, v: sa.sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout="NHD")
        out["sage2 int8qk fp16pv (cuda)"] = lambda q, k, v: sa.sageattn_qk_int8_pv_fp16_cuda(q, k, v, tensor_layout="NHD")
    except Exception as e:
        print("sageattention import failed", e)
    try:
        import xformers.ops as xo
        out["xformers"] = lambda q, k, v: xo.memory_efficient_attention(
            q, k.repeat_interleave(rep, 2), v.repeat_interleave(rep, 2))
    except Exception as e:
        print("xformers missing", e)
    return out


def tm(f, q, k, v, n=6):
    f(q, k, v); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        f(q, k, v)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


B = backends()
for name, L in SIZES.items():
    q, k, v = mk(L)
    idx = torch.randperm(L, device=dev)[:512]
    r = ref_rows(q, k, v, idx)
    print(f"\n== {name}: L={L} tokens ==")
    base = None
    for bn, f in B.items():
        try:
            o = f(q, k, v)
            o = o[0] if isinstance(o, tuple) else o
            d = o[:, idx].float()
            err = float((d - r).norm() / r.norm())
            cos = float(F.cosine_similarity(d.flatten(), r.flatten(), dim=0))
            ms = tm(f, q, k, v)
            base = base or ms
            print(f"  {bn:32s} {ms:8.2f} ms  x{base / ms:4.2f}   rel err {err:.2e}  cos {cos:.6f}")
        except Exception as e:
            print(f"  {bn:32s} FAILED {type(e).__name__}: {str(e)[:120]}")
    del q, k, v
    torch.cuda.empty_cache()
print("DONE")
