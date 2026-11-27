"""MK.41 probe: per-CTA stamps of the fused W13->W2 kernel (GPU0, 8k full by default)
vs the unfused W13 / W2 phases. Env: GRID, DEPTH2, CTX_LEN, LEFT."""
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
G = int(os.environ.get("GRID", "576"))
base, native, *rest = setup(context, left)
x, stream, w, m, _ = base["chained"][1].args
keep = []
q = lambda t: [round(float(v), 2) for v in torch.quantile(t, torch.tensor([0., .5, .9, 1.], dtype=torch.float64))]  # noqa: E731
for fused in (False, True):
    tlx_experts.FUSE13_2 = fused
    tlx_experts.FUSED_GRID = G
    tlx_experts.FUSED_DEPTH2 = int(os.environ.get("DEPTH2", "2"))
    tlx_experts.FUSED_PREWAIT = bool(int(os.environ.get("PREWAIT", "0")))
    tr = torch.zeros(G * 4, device="cuda", dtype=torch.int64)
    pt = (torch.zeros(1728 * 2, device="cuda", dtype=torch.int64),
          torch.zeros(1728 * 2, device="cuda", dtype=torch.int64))
    tlx_experts.FUSED_TRACE = tr if fused else None
    tlx_experts.PHASE_TRACE = None if fused else pt
    ws = tlx_chain.make_workspace("cuda", context=context, left=left)
    prep = lambda: tlx_chain.prepare_peers(ws, *native.partials, rank=0)  # noqa: E731
    prep(); tlx_chain.launch(x, stream, w, m, ws, rank=0); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    prep(); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        tlx_chain.launch(x, stream, w, m, ws, rank=0)
    keep.append((ws, g, tr, pt))
    tasks = int(ws.s4.num_post[0]) // 16 * 64
    for it in range(5):
        prep(); torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
    if fused:
        v = tr.view(G, 4).double().cpu()
        s0 = v[:, 0].min()
        print(json.dumps(dict(fused=True, grid=G, tasks=tasks,
                              kernel_us=round(float((v[:, 3] - s0).max()) / 100, 2),
                              w13_done_us=q((v[:, 1] - s0) / 100),
                              w2_wait_us=q(v[:, 2] / 100),
                              end_us=q((v[:, 3] - s0) / 100))), flush=True)
    else:
        a = pt[0].view(1728, 2).double().cpu(); b = pt[1].view(1728, 2).double().cpu()
        s0 = a[:, 0].min()
        print(json.dumps(dict(fused=False, tasks=tasks,
                              w13_end_us=round(float((a[:tasks, 1] - s0).max()) / 100, 2),
                              w2_start_us=round(float((b[:, 0] - s0).min()) / 100, 2),
                              w2_end_us=round(float((b[:, 1] - s0).max()) / 100, 2))), flush=True)
