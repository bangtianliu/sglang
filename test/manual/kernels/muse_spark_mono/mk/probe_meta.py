import os, json, torch
assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_fused34, tlx_projection
x = torch.zeros(8, 8192, device="cuda", dtype=torch.bfloat16)
w = torch.zeros(4608, 8192, device="cuda", dtype=torch.bfloat16)
p = torch.zeros(8, 4608, device="cuda", dtype=torch.int32); s = torch.zeros(1024, device="cuda", dtype=torch.int32)
f = torch.zeros(2, device="cuda", dtype=torch.int32)
def meta(name, k):
    md = k.metadata
    print(json.dumps(dict(k=name, global_scratch=getattr(md, "global_scratch_size", None),
        global_scratch_align=getattr(md, "global_scratch_align", None),
        profile_scratch=getattr(md, "profile_scratch_size", None), lds=md.shared)), flush=True)
meta("prod_project", tlx_projection.project.warmup(x, w, p, s, Failure=f, N=4608, K=8192, SK=8, STAGE=4, ROUND_OUTPUT=False, num_warps=2, enable_fp_fusion=False, grid=(288,)))
meta("project4", tlx_fused34.project4.warmup(x, w, p, s, s, s, NB=8, num_warps=4, enable_fp_fusion=False, grid=(288,)))
meta("project_wait", tlx_fused34.project_wait.warmup(x, w, p, s, s, s, s, Failure=f, NB=8, num_warps=2, enable_fp_fusion=False, grid=(288,)))
