"""Run tlx_fused34.project_wait alone (counters pre-satisfied) vs tlx_projection.project."""
import os, sys, json
import torch
assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_fused34, tlx_projection
torch.manual_seed(0)
x = torch.randn(8, 8192, device="cuda").to(torch.bfloat16)
w = (torch.randn(4608, 8192, device="cuda") * 0.02).to(torch.bfloat16)
p_ref = torch.zeros(8, 4608, device="cuda", dtype=torch.int32)
p_new = torch.zeros_like(p_ref)
st_ref = torch.zeros(288, device="cuda", dtype=torch.int32)
st_new = torch.zeros_like(st_ref)
fail = torch.zeros(2, device="cuda", dtype=torch.int32)
tlx_projection.project[(288,)](x, w, p_ref, st_ref, Failure=fail, N=4608, K=8192, SK=8, STAGE=4,
    ROUND_OUTPUT=False, num_warps=2, enable_fp_fusion=False)
torch.cuda.synchronize(); print("ref ok", flush=True)
counters = torch.full((32 * 32,), 8, device="cuda", dtype=torch.int32)
epoch = torch.zeros(288, device="cuda", dtype=torch.int32)
bstat = torch.ones(512, device="cuda", dtype=torch.int32)
nb = int(sys.argv[1]) if len(sys.argv) > 1 else 8
tlx_fused34.project_wait[(288,)](x, w, p_new, st_new, counters, epoch, bstat, Failure=fail, NB=nb,
    BOUND=2000, num_warps=2, enable_fp_fusion=False)
torch.cuda.synchronize()
print(json.dumps(dict(nb=nb, exact=bool(torch.equal(p_ref, p_new)), status=int(st_new.sum()),
      mismatches=int((p_ref != p_new).sum()))), flush=True)
