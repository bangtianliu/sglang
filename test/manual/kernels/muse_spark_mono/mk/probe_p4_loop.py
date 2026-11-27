import os, sys, json
import torch
assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_fused34, tlx_projection
torch.manual_seed(0)
x = torch.randn(8, 8192, device="cuda").to(torch.bfloat16)
ws = [(torch.randn(4608, 8192, device="cuda") * 0.02).to(torch.bfloat16) for _ in range(3)]
fail = torch.zeros(2, device="cuda", dtype=torch.int32)
refs = []
for w in ws:
    pr = torch.zeros(8, 4608, device="cuda", dtype=torch.int32); sr = torch.zeros(288, device="cuda", dtype=torch.int32)
    tlx_projection.project[(288,)](x, w, pr, sr, Failure=fail, N=4608, K=8192, SK=8, STAGE=4, ROUND_OUTPUT=False, num_warps=2, enable_fp_fusion=False)
    refs.append(pr)
pn = torch.zeros(8, 4608, device="cuda", dtype=torch.int32); sn = torch.zeros(288, device="cuda", dtype=torch.int32)
bad = []
ep = torch.zeros(288, device='cuda', dtype=torch.int32)
for it in range(int(sys.argv[2])):
    j = it % 3
    pn.zero_()
    if os.environ.get("WAITPATH"):
        cnt = torch.full((1024,), 8 * (it + 1), device="cuda", dtype=torch.int32)
        k = tlx_fused34.project4_wait[(288,)](x, ws[j], pn, sn, cnt, ep, NB=int(sys.argv[1]), num_warps=4, enable_fp_fusion=False)
    else:
        k = tlx_fused34.project4[(288,)](x, ws[j], pn, sn, sn, sn, NB=int(sys.argv[1]), num_warps=4, enable_fp_fusion=False)
    torch.cuda.synchronize()
    m = (pn != refs[j])
    if m.any():
        bad.append((it, int(m.sum()), sorted(set((m.nonzero()[:, 1] // 16).tolist()))[:6]))
print(json.dumps(dict(iters=int(sys.argv[2]), bad=len(bad), first=bad[:5], vgpr=k.n_regs, lds=k.metadata.shared)), flush=True)
