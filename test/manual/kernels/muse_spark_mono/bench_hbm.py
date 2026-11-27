"""GPU0 read and copy bandwidth; each working set is at least four times L2."""

import json
from pathlib import Path

import numpy as np
import torch
import triton
from layer_fixture import require_gpu0
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def stream_read(X, Y, N: gl.constexpr, B: gl.constexpr):
    i = gl.program_id(0) * B + gl.arange(
        0, B, layout=gl.BlockedLayout([1], [32], [4], [0])
    )
    x = gl.load(X + i, i < N, other=0)
    gl.store(Y + gl.program_id(0), gl.sum(x, 0))


def graph_times(fn, samples=100):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(0)
    graph = torch.cuda.CUDAGraph()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    with torch.cuda.graph(graph):
        fn()
    for start, end in zip(starts, ends):
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize(0)
    values = [s.elapsed_time(e) * 1000 for s, e in zip(starts, ends)]
    return {
        "median_us": float(np.median(values)),
        "p99_us": float(np.quantile(values, 0.99)),
    }


def main():
    require_gpu0()
    records = []
    for mib in (16, 64, 256, 1024):
        n = mib * 1024 * 1024 // 4
        x = torch.rand((n,), device="cuda:0", dtype=torch.float32)
        y = torch.empty((triton.cdiv(n, 4096),), device="cuda:0", dtype=torch.float32)
        z = torch.empty_like(x)
        read = lambda: stream_read[(triton.cdiv(n, 4096),)](x, y, n, 4096)
        read()
        torch.testing.assert_close(y, x.reshape(-1, 4096).sum(dim=1))
        for name, fn in (("gluon_read", read), ("torch_copy", lambda: z.copy_(x))):
            time = graph_times(fn)
            item = dict(
                mib=mib,
                operation=name,
                **time,
                read_GB_s=n * 4 / (time["median_us"] * 1e3),
            )
            if name == "torch_copy":
                torch.testing.assert_close(z, x)
                item["total_read_write_GB_s"] = 2 * item["read_GB_s"]
            records.append(item)
            print(json.dumps(item), flush=True)
    result = {
        "L2_bytes": torch.cuda.get_device_properties(0).L2_cache_size,
        "records": records,
    }
    Path("/home/bangtliu/coord/logs/phase3.0-read-and-copy-bandwidth.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
