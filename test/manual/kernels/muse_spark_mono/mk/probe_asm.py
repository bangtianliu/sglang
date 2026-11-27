import os, torch
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_fused34
x = torch.zeros(8, 8192, device="cuda", dtype=torch.bfloat16)
w = torch.zeros(4608, 8192, device="cuda", dtype=torch.bfloat16)
p = torch.zeros(8, 4608, device="cuda", dtype=torch.int32); s = torch.zeros(1024, device="cuda", dtype=torch.int32)
k = tlx_fused34.project4.warmup(x, w, p, s, s, s, NB=8, num_warps=4, enable_fp_fusion=False, grid=(288,))
lines = k.asm["amdgcn"].splitlines()
idx = [i for i, l in enumerate(lines) if "s_barrier" in l]
print("barriers", len(idx))
for i in idx[:3]:
    print("----")
    print("\n".join(l.rstrip() for l in lines[max(0, i - 12): i + 3]))
