"""GPU check of _split_attn: flash-attn LSE path vs exact fp32 math, Krea2 shapes (GQA 48/12,
D=128), a 3-group split + log-weight merge vs one dense attention with an additive bias, and
timing of split vs single flash call. Usage: python test_split_attn_gpu.py PACK_DIR"""
import importlib.util, math, os, sys, time
import torch
import torch.nn.functional as F
spec = importlib.util.spec_from_file_location("sa", os.path.join(sys.argv[1], "_split_attn.py"))
SA = importlib.util.module_from_spec(spec); spec.loader.exec_module(SA)
torch.manual_seed(0)
dev = "cuda"
B, H, Hk, D = 1, 48, 12, 128
Lt, Lr, Li = 480, 4096, 4096
q = torch.randn(B, Lt + Lr + Li, H, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(B, Lt + Lr + Li, Hk, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(B, Lt + Lr + Li, Hk, D, device=dev, dtype=torch.bfloat16)
assert SA._flash_fn() is not None, "flash_attn not importable"
o_f, l_f = SA.attend_lse(q, k, v)
o_m, l_m = SA._math_attn(q, k, v)
print("flash vs math  out max err", float((o_f.float() - o_m.float()).abs().max()),
      " lse max err", float((l_f - l_m).abs().max()))
# split + boost vs dense bias
b = 4.0
t0 = Lt + Lr
lw = torch.zeros(1, q.shape[1], 1, device=dev); lw[:, t0:] = math.log(b)
parts = [SA.attend_lse(q, k[:, :Lt], v[:, :Lt]), SA.attend_lse(q, k[:, Lt:t0], v[:, Lt:t0]),
         SA.attend_lse(q, k[:, t0:], v[:, t0:])]
o_s = SA.merge([(parts[0][0], parts[0][1], None), (parts[1][0], parts[1][1], lw),
                (parts[2][0], parts[2][1], None)], out_dtype=torch.float32)
bias = torch.zeros(q.shape[1], q.shape[1], device=dev); bias[t0:, Lt:t0] = math.log(b)
rep = H // Hk
ref = F.scaled_dot_product_attention(q.float().transpose(1, 2), k.float().repeat_interleave(rep, 2).transpose(1, 2),
                                     v.float().repeat_interleave(rep, 2).transpose(1, 2), attn_mask=bias).transpose(1, 2)
print("split+boost vs dense-bias fp32  max err", float((o_s - ref).abs().max()), " ref max", float(ref.abs().max()))
# timing
def tm(f, n=10):
    f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1000
from flash_attn import flash_attn_func
t_one = tm(lambda: flash_attn_func(q, k, v))
t_split = tm(lambda: SA.merge([SA.attend_lse(q, k[:, :Lt], v[:, :Lt]) + (None,),
                               SA.attend_lse(q, k[:, Lt:t0], v[:, Lt:t0]) + (lw,),
                               SA.attend_lse(q, k[:, t0:], v[:, t0:]) + (None,)]))
print(f"timing ({q.shape[1]} tokens): single flash {t_one:.2f} ms, 3-group split+merge {t_split:.2f} ms")
print("DONE")
