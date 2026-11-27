import os, json, torch
assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_fused34, tlx_projection
x = torch.randn(8, 8192, device="cuda").to(torch.bfloat16)
ws = [(torch.randn(4608, 8192, device="cuda") * 0.02).to(torch.bfloat16) for _ in range(4)]
p = torch.zeros(8, 4608, device="cuda", dtype=torch.int32); s = torch.zeros(288, device="cuda", dtype=torch.int32)
f = torch.zeros(2, device="cuda", dtype=torch.int32)
def t(fn):
    for w in ws: fn(w)
    torch.cuda.synchronize()
    out = []
    for r in range(40):
        w = ws[r % 4]
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(w); b.record(); torch.cuda.synchronize(); out.append(a.elapsed_time(b) * 1000)
    out.sort(); return out[len(out) // 2]
r = dict(
  project2=t(lambda w: tlx_projection.project[(288,)](x, w, p, s, Failure=f, N=4608, K=8192, SK=8, STAGE=4, ROUND_OUTPUT=False, num_warps=2, enable_fp_fusion=False)),
  project4=t(lambda w: tlx_fused34.project4[(288,)](x, w, p, s, s, s, NB=8, num_warps=4, enable_fp_fusion=False)),
  project4_nb4=t(lambda w: tlx_fused34.project4[(288,)](x, w, p, s, s, s, NB=4, num_warps=4, enable_fp_fusion=False)),
)
print(json.dumps(r))
