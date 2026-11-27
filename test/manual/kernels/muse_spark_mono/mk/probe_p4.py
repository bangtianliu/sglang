import os, sys, json
import torch
assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_fused34, tlx_projection
res = []
for seed in range(3):
    torch.manual_seed(seed)
    x = torch.randn(8, 8192, device="cuda").to(torch.bfloat16)
    w = (torch.randn(4608, 8192, device="cuda") * 0.02).to(torch.bfloat16)
    pr = torch.zeros(8, 4608, device="cuda", dtype=torch.int32); pn = torch.zeros_like(pr)
    sr = torch.zeros(288, device="cuda", dtype=torch.int32); sn = torch.zeros_like(sr)
    fail = torch.zeros(2, device="cuda", dtype=torch.int32)
    tlx_projection.project[(288,)](x, w, pr, sr, Failure=fail, N=4608, K=8192, SK=8, STAGE=4,
        ROUND_OUTPUT=False, num_warps=2, enable_fp_fusion=False)
    k = tlx_fused34.project4[(288,)](x, w, pn, sn, sn, sn, NB=int(sys.argv[1]), num_warps=4, enable_fp_fusion=False)
    torch.cuda.synchronize()
    res.append(dict(seed=seed, exact=bool(torch.equal(pr, pn)), mism=int((pr != pn).sum()),
                    vgpr=k.n_regs, spills=k.n_spills, lds=k.metadata.shared))
print(json.dumps(res), flush=True)
