"""MK.110: the standalone host-memory all-reduce on 4 GPUs: exact against an fp32 reference,
and time per call from a CUDA graph of 100 calls, next to torch.distributed.all_reduce."""
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))


def rows(rank, step, width):
    g = torch.Generator().manual_seed(1000 * step + rank)
    return (torch.randn(8, width, generator=g) * 3).to(torch.bfloat16)


def main():
    assert os.environ.get("MUSE_PEER_HOST") == "1" and os.environ.get("NCCL_P2P_DISABLE") == "1"
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    cpu = dist.new_group(backend="gloo")
    barrier = lambda: dist.monitored_barrier(cpu, timeout=timedelta(seconds=120))  # noqa: E731
    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_host_collective as hc

    out = {}
    for width in (8192, 4096):
        ar = hc.HostAllReduce("cuda", width=width, rank=rank, barrier=barrier,
                              name=f"muse_ar4_{os.getppid()}")
        bad = 0
        for step in range(40):
            x = rows(rank, step, width).cuda()
            barrier()
            ar(x)
            torch.cuda.synchronize()
            ref = sum(rows(r, step, width).float() for r in range(4)).to(torch.bfloat16)
            bad += int((x.cpu().view(torch.int16) != ref.view(torch.int16)).sum())
        x = rows(rank, 0, width).cuda()
        y = rows(rank, 0, width).cuda()
        res = {}
        for name, fn in (("host", lambda t: ar(t)), ("rccl", lambda t: dist.all_reduce(t))):
            try:
                g = torch.cuda.CUDAGraph()
                barrier()
                with torch.cuda.graph(g):
                    for _ in range(100):
                        fn(x if name == "host" else y)
                best = 1e9
                for _ in range(8):
                    torch.cuda.synchronize()
                    barrier()
                    t = time.perf_counter()
                    g.replay()
                    torch.cuda.synchronize()
                    best = min(best, (time.perf_counter() - t) / 100 * 1e6)
                res[name] = round(best, 2)
            except Exception as e:  # noqa: BLE001
                res[name] = f"{type(e).__name__}: {str(e)[:80]}"
        out[width] = dict(mismatched_values=bad, us_per_call=res)
    print(json.dumps(dict(rank=rank, **{str(k): v for k, v in out.items()})), flush=True)


if __name__ == "__main__":
    main()
