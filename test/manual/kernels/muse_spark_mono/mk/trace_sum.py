"""sum_reduce phase stamps (GPU0, simulated peers): start, failure read, slots done,
partial stored+drained, AR done, end. Median over replays of the whole chain (graph)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
from bench_chained47 import setup  # noqa: E402
from layer_fixture import require_gpu0  # noqa: E402

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_chain, tlx_experts  # noqa: E402

require_gpu0()
context, left = 8192, -1
base, native, *rest = setup(context, left)
x, stream, w, m, _ = base["chained"][1].args
ws = tlx_chain.make_workspace("cuda", context=context, left=left)
trace = torch.zeros(64, device="cuda", dtype=torch.int64)
tlx_experts.SUM_TRACE = trace
import os; tlx_experts.EARLY_SUM = os.environ.get("CHAIN_FLAGS") != "late"
prep = lambda: tlx_chain.prepare_peers(ws, *native.partials, rank=0)  # noqa: E731
prep(); tlx_chain.launch(x, stream, w, m, ws, rank=0); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
prep(); torch.cuda.synchronize()
with torch.cuda.graph(g):
    tlx_chain.launch(x, stream, w, m, ws, rank=0)
rows = []
for _ in range(30):
    prep(); torch.cuda.synchronize()
    g.replay(); torch.cuda.synchronize()
    assert int(ws.failure[0]) == 0
    t = trace.view(8, 8)[:, :6].double().cpu()
    rows.append((t - t[:, :1].min()) / 100)  # 100 MHz -> µs from first CTA start
r = torch.stack(rows).median(0).values
names = ["start", "failure_read", "slots_done", "partial_drained", "ar_done", "end"]
print(json.dumps({n: [round(float(v), 2) for v in r[:, k]] for k, n in enumerate(names)}))
print(json.dumps({"max_per_phase": {n: round(float(r[:, k].max()), 2) for k, n in enumerate(names)}}))
