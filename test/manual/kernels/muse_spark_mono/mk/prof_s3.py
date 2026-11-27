import sys, json, functools
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import torch
from torch.profiler import profile, ProfilerActivity
from bench_chained47 import setup
from bench_s3n import split_tail
import types
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_s3fused
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_chain
base, native, *rest = setup(8192, -1)
x, stream, w, m, _ = base["chained"][1].args
out = {}
for name in ("tlx", "s3n"):
    ws = tlx_chain.make_workspace("cuda", context=8192, left=-1)
    prep = functools.partial(tlx_chain.prepare_peers, ws, *native.partials, rank=0)
    sums = torch.zeros(256, device="cuda"); rowcnt = torch.zeros(256, device="cuda", dtype=torch.int32)
    st4 = tlx_s3fused.State("cuda")
    def fn():
        if name == "tlx":
            return tlx_chain.launch(x, stream, w, m, ws, rank=0)
        st4 = tlx_s3fused.State("cuda")
        def at(x_, gate, o_proj, stream_, pre, alpha, beta, *, rank, workspace, store_reduced, split_norm, output, failure, **_):
            return tlx_s3fused.tail(x_, gate, o_proj, stream_, pre, alpha, beta, rank=rank, ws=workspace, st=st4, output=output, failure=failure, split_norm=split_norm, store_reduced=store_reduced)
        old = tlx_chain.tlx_attn_tail; tlx_chain.tlx_attn_tail = types.SimpleNamespace(attention_tail=at)
        try: return tlx_chain.launch(x, stream, w, m, ws, rank=0)
        finally: tlx_chain.tlx_attn_tail = old
    for _ in range(3):
        prep(); fn(); torch.cuda.synchronize()
    agg = {}
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(5):
            prep(); torch.cuda.synchronize(); fn(); torch.cuda.synchronize()
    for e in pr.events():
        if e.device_type.name == "CUDA" and e.name and "prepare" not in e.name:
            agg.setdefault(e.name[:40], []).append(e.device_time)
    out[name] = {k: round(sorted(v)[len(v)//2], 2) for k, v in agg.items()}
print(json.dumps(out, indent=0))
