"""Per-CTA start/end stamps of the W13 and W2 expert phases (GPU0, simulated peers,
8k full by default): how many CTAs do work, how they are spread over time (rounds)."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
from bench_chained47 import setup  # noqa: E402
from layer_fixture import require_gpu0  # noqa: E402

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_chain, tlx_experts  # noqa: E402

require_gpu0()
context, left = int(os.environ.get("CTX_LEN", "8192")), int(os.environ.get("LEFT", "-1"))
base, native, *rest = setup(context, left)
x, stream, w, m, _ = base["chained"][1].args
ws = tlx_chain.make_workspace("cuda", context=context, left=left)
tr = (torch.zeros(1728 * 2, device="cuda", dtype=torch.int64),
      torch.zeros(1728 * 2, device="cuda", dtype=torch.int64))
tlx_experts.PHASE_TRACE = tr
for kv in filter(None, os.environ.get("CHAIN_FLAGS", "").split(",")):
    k, v = kv.split("=")
    setattr(tlx_experts, k.split(".")[1], int(v))
prep = lambda: tlx_chain.prepare_peers(ws, *native.partials, rank=0)  # noqa: E731
prep(); tlx_chain.launch(x, stream, w, m, ws, rank=0); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
prep(); torch.cuda.synchronize()
with torch.cuda.graph(g):
    tlx_chain.launch(x, stream, w, m, ws, rank=0)
tasks = int(ws.s4.num_post[0]) // 16 * 64
for it in range(5):
    prep(); torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
for name, t in zip(("w13", "w2"), tr):
    v = t.view(1728, 2).double().cpu()
    s0 = v[:, 0].min()
    st, en = (v[:, 0] - s0) / 100, (v[:, 1] - s0) / 100
    act = st[:tasks], en[:tasks]
    dur = act[1] - act[0]
    hist = torch.histc(act[0], bins=8, min=0, max=float(en.max()))
    print(json.dumps(dict(phase=name, tasks=tasks, kernel_us=round(float(en.max()), 2),
                          active_end_max=round(float(act[1].max()), 2),
                          active_start_quantiles=[round(float(q), 2) for q in torch.quantile(act[0], torch.tensor([0., .25, .5, .75, .9, 1.], dtype=torch.float64))],
                          task_us_quantiles=[round(float(q), 2) for q in torch.quantile(dur, torch.tensor([0., .5, .9, 1.], dtype=torch.float64))],
                          idle_end_max=round(float(en[tasks:].max()), 2) if tasks < 1728 else None,
                          start_hist=[int(h) for h in hist])), flush=True)
