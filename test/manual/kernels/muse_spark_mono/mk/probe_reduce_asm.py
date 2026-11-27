import torch, triton, triton.language as tl
import utlx_plugin as tlx
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import ROW1
@triton.jit
def k(X, Y):
    i = tlx.require_layout(tl.arange(0, 4096), ROW1)
    x = tlx.require_layout(tl.load(X + i), ROW1, pin=True)
    tl.store(Y, tl.sum(x * x, 0))
x = torch.zeros(4096, device="cuda")
c = k.warmup(x, x, grid=(1,), num_warps=4)
lines = [l.strip() for l in c.asm["amdgcn"].splitlines()]
body = [l for l in lines if l.startswith(("v_", "ds_", "s_barrier", "v_permlane", "v_mov_b32_dpp"))]
adds = [l for l in body if "add_f32" in l or "fma" in l or "mul_f32" in l or "dpp" in l or "permlane" in l or "ds_" in l or "barrier" in l]
print(len(adds))
print("\n".join(adds[-60:]))
