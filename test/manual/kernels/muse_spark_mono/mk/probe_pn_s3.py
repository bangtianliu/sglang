import os, json, torch
assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_fused34, tlx_projection
torch.manual_seed(0)
x = torch.randn(8, 2048, device="cuda").to(torch.bfloat16)
ws = [(torch.randn(8192, 2048, device="cuda") * 0.03).to(torch.bfloat16) for _ in range(4)]
f = torch.zeros(2, device="cuda", dtype=torch.int32)
def ref(w, p, pk, s):
    tlx_projection.project[(512,)](x, w, p, s, Failure=f, N=8192, K=2048, STAGE=3, Packed=pk, num_warps=2, enable_fp_fusion=False)
def new(w, p, pk, s, nb):
    return tlx_fused34.project_n[(128,)](x, w, p, s, pk, N=8192, K=2048, NB=nb, SK=4, ROUND_OUTPUT=True, num_warps=4, enable_fp_fusion=False)
res = {}
for nb in (2, 4):
    exact = True
    for w in ws:
        a = torch.zeros(8, 8192, device="cuda", dtype=torch.int32); b = torch.zeros_like(a)
        pa = torch.zeros(8, 8192, device="cuda", dtype=torch.bfloat16); pb = torch.zeros_like(pa)
        sa = torch.zeros(512, device="cuda", dtype=torch.int32); sb = torch.zeros_like(sa)
        ref(w, a, pa, sa); k = new(w, b, pb, sb, nb); torch.cuda.synchronize()
        exact &= bool(torch.equal(a, b)) and bool(torch.equal(pa.view(torch.int16), pb.view(torch.int16))) and int(sb.sum()) == 512
    def t(fn):
        out = []
        for r in range(40):
            w = ws[r % 4]; p = torch.empty(8, 8192, device="cuda", dtype=torch.int32); pk = torch.empty(8, 8192, device="cuda", dtype=torch.bfloat16); s = torch.empty(512, device="cuda", dtype=torch.int32)
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); fn(w, p, pk, s); e1.record(); torch.cuda.synchronize(); out.append(e0.elapsed_time(e1) * 1000)
        out.sort(); return out[len(out) // 2]
    res[f"nb{nb}"] = dict(exact=exact, us=t(lambda w, p, pk, s: new(w, p, pk, s, nb)), vgpr=k.n_regs, lds=k.metadata.shared)
res["ref_us"] = t(ref)
print(json.dumps(res))
