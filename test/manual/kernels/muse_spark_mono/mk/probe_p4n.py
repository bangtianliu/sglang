import os, json, torch
assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_fused34, tlx_projection
torch.manual_seed(0)
x = torch.randn(8, 8192, device="cuda").to(torch.bfloat16)
ws = [(torch.randn(4608, 8192, device="cuda") * 0.02).to(torch.bfloat16) for _ in range(4)]
s = torch.zeros(288, device="cuda", dtype=torch.int32); f = torch.zeros(2, device="cuda", dtype=torch.int32)
def p2(w, p): tlx_projection.project[(288,)](x, w, p, s, Failure=f, N=4608, K=8192, SK=8, STAGE=4, ROUND_OUTPUT=False, num_warps=2, enable_fp_fusion=False)
res = {}
for nb in (2, 3, 4):
    def p4n(w, p, nb=nb): return tlx_fused34.project4n[(72,)](x, w, p, s, NB=nb, num_warps=4, enable_fp_fusion=False)
    exact = True
    for w in ws:
        a = torch.zeros(8, 4608, device="cuda", dtype=torch.int32); b = torch.zeros_like(a)
        p2(w, a); k = p4n(w, b); torch.cuda.synchronize(); exact &= bool(torch.equal(a, b))
    def t(fn):
        out = []
        for r in range(40):
            w = ws[r % 4]; pp = torch.empty(8, 4608, device="cuda", dtype=torch.int32)
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); fn(w, pp); e1.record(); torch.cuda.synchronize(); out.append(e0.elapsed_time(e1) * 1000)
        out.sort(); return out[len(out) // 2]
    res[f"nb{nb}"] = dict(exact=exact, us=t(p4n), vgpr=k.n_regs, spills=k.n_spills, lds=k.metadata.shared)
res["project2_us"] = t(p2)
print(json.dumps(res))
