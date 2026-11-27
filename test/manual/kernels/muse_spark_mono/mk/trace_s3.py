import sys, json, functools, types
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import torch
from bench_chained47 import setup
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_chain, tlx_attn_tail, tlx_projection, tlx_s3split
base, native, *rest = setup(8192, -1)
x, stream, w, m, _ = base["chained"][1].args
ws = tlx_chain.make_workspace("cuda", context=8192, left=-1)
prep = functools.partial(tlx_chain.prepare_peers, ws, *native.partials, rank=0)
sums = torch.zeros(256, device="cuda"); rowcnt = torch.zeros(256, device="cuda", dtype=torch.int32); trace = torch.zeros(64 * 8, device="cuda", dtype=torch.int64)
def attention_tail(xx, gate, o_proj, stream_, pre, alpha, beta, *, rank, workspace, store_reduced, split_norm, output, failure, **_):
    W = workspace; out, updated = output
    fn = tlx_attn_tail.normalize_rows if split_norm else tlx_attn_tail.normalize
    fn[(256 if split_norm else 16,)](xx, gate, W.normalized, failure, num_warps=4, enable_fp_fusion=False)
    tlx_projection.project[(512,)](W.normalized, o_proj, W.projection, W.state.status, Failure=failure, N=8192, K=2048, STAGE=3, Packed=W.partial[rank], num_warps=2, enable_fp_fusion=False)
    tlx_s3split.boundary_one(W.partial, stream_, pre, alpha, beta, out, updated, W, rank=rank, failure=failure, sums=sums, rowcnt=rowcnt, trace=trace)
    return out, updated
shim = types.SimpleNamespace(attention_tail=attention_tail)
res = []
for it in range(6):
    prep(); torch.cuda.synchronize()
    old = tlx_chain.tlx_attn_tail; tlx_chain.tlx_attn_tail = shim
    try: tlx_chain.launch(x, stream, w, m, ws, rank=0)
    finally: tlx_chain.tlx_attn_tail = old
    torch.cuda.synchronize()
    t = trace.view(64, 8).double()
    t0 = t[:, 0].min()
    rel = (t - t0) / 100.0  # 100 MHz ticks -> us
    names = ["start", "status", "push+drain", "flag poll", "4-src load+sum", "sums computed", "row wait done", "end"]
    if it >= 4:
        res.append({n: [round(float(rel[:, i].min()), 2), round(float(rel[:, i].median()), 2), round(float(rel[:, i].max()), 2)] for i, n in enumerate(names)})
print(json.dumps(res, indent=0))
