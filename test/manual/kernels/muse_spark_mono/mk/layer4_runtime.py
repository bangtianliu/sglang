"""MK.46: the plugin's ChainRuntime on 4 real GPUs (host-memory exchange), without a model.

Layers 1..N run back to back, each layer's outputs feeding the next (full layers at i % 4 == 1,
sliding otherwise), with the test fixtures' weights and KV metadata for the two layer kinds.
Checks: no failures; a CUDA graph replay equals the eager run; and 2 workspaces per kind
(alternating, as in serving) give the same bits as one fresh workspace per layer.
"""
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

assert os.environ.get("NCCL_P2P_DISABLE") == "1", "fabric must stay disabled"
RANK = int(os.environ["LOCAL_RANK"])
os.environ["HIP_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"] = str(RANK)
os.environ["MUSE_PEER_HOST"] = "1"

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
N = int(os.environ.get("LAYERS", "8"))


def main():
    torch.cuda.set_device(0)
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))

    def barrier():
        dist.monitored_barrier(timeout=timedelta(seconds=60))

    from layer_case import make

    from sglang.srt.kernel_plugins.amd.muse_spark import runtime

    context = int(os.environ.get("CTX_LEN", "8192"))
    fix = {"full": make(7, context, -1, RANK), "sliding": make(7, context, 2047, RANK)}
    kind = lambda i: "full" if i % 4 == 1 else "sliding"  # noqa: E731
    runtime.LAST_LAYER = N
    x0, s0 = fix["sliding"][0].x.clone(), fix["sliding"][0].stream.clone()

    def build(copies, tag):
        return runtime.ChainRuntime(
            None, rank=RANK, barrier=barrier, context=context,
            name=f"muse_rt_{os.getppid()}_{tag}", device="cuda", copies=copies,
            layer_weights=lambda i: fix[kind(i)][1],
            step_metadata=lambda i, positions, fb: fix[kind(i)][2],
        )

    def step(rt):
        rt.begin_step()
        x, stream = x0, s0
        for i in range(1, N + 1):
            r = rt.run_layer(i, None, stream, None, x)
            assert r is not None, i
            stream, x, _ = r
        return x, stream

    row = dict(rank=RANK, layers=N, context=context)
    results = {}
    for tag, copies in (("two", 2), ("fresh", N)):
        rt = build(copies, tag)
        # compile on every rank inside a capture (nothing runs): ranks compile at different
        # speeds, and an exchange started meanwhile would time out
        warm = torch.cuda.CUDAGraph()
        with torch.cuda.graph(warm):
            step(rt)
        torch.cuda.synchronize()
        barrier()
        rt.reset_failures()
        barrier()
        x, stream = step(rt)
        torch.cuda.synchronize()
        eager = (x.clone(), stream.clone())
        fails = rt.failures()
        barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gx, gs = step(rt)
        barrier()
        times = []
        for _ in range(6):
            barrier()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize()
            times.append(a.elapsed_time(b) * 1000 / N)
        graph = (gx.clone(), gs.clone())
        fails += rt.failures()
        results[tag] = (eager, graph, rt)
        row[tag] = dict(failures=fails, layer_us=round(sorted(times)[len(times) // 2], 1),
                        graph_vs_eager_bytes=sum(int((a_.view(-1).view(torch.uint8)
                                                      != b_.view(-1).view(torch.uint8)).sum())
                                                 for a_, b_ in zip(eager, graph)),
                        finite=bool(torch.isfinite(eager[0].float()).all()
                                    and torch.isfinite(eager[1]).all()))
    row["two_vs_fresh_bytes"] = sum(int((a_.view(-1).view(torch.uint8)
                                         != b_.view(-1).view(torch.uint8)).sum())
                                    for a_, b_ in zip(results["two"][0], results["fresh"][0]))
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
