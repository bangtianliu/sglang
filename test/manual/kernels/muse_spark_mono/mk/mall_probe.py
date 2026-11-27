"""Memory-side cache probe (GPU0): time reading an N MiB buffer cold (after a 2 GiB flush)
vs right after another kernel read it (warm), and vs warm after an interleaved small kernel."""
import json
import torch
import triton
import triton.language as tl


@triton.jit
def stream_read(X, Out, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), tl.int32)
    for i in range(pid * BLOCK, N, tl.num_programs(0) * BLOCK):
        acc += tl.load(X + i + tl.arange(0, BLOCK))
    tl.store(Out + pid * BLOCK + tl.arange(0, BLOCK), acc)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from layer_fixture import require_gpu0  # noqa: E402


def t(fn, n=1):
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record(); fn(); e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000


def main():
    require_gpu0()
    flush = torch.empty(2 << 30, dtype=torch.uint8, device="cuda")
    out = {}
    for mib in (8, 16, 32, 64, 96, 128, 192, 256, 384):
        x = torch.empty(mib << 20, dtype=torch.uint8, device="cuda").view(torch.int32)
        x.fill_(1)
        acc = torch.empty((), dtype=torch.int64, device="cuda")
        o = torch.empty(768 * 4096, dtype=torch.int32, device="cuda")
        rd = lambda: stream_read[(768,)](x, o, x.numel(), BLOCK=4096, num_warps=8)
        rd(); torch.cuda.synchronize()
        cold, warm = [], []
        for _ in range(15):
            flush.add_(1); torch.cuda.synchronize()
            cold.append(t(rd))
            warm.append(t(rd))
        med = lambda v: round(sorted(v)[len(v) // 2], 1)
        out[mib] = dict(cold_us=med(cold), warm_us=med(warm),
                        cold_tbs=round((mib << 20) / med(cold) / 1e6, 2),
                        warm_tbs=round((mib << 20) / med(warm) / 1e6, 2))
        print(json.dumps({mib: out[mib]}), flush=True)


if __name__ == "__main__":
    main()
