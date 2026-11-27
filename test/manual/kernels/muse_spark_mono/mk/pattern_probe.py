"""MK.10b (GPU0): is the expert weight access pattern the bandwidth limit?
10 expert blocks x 64 tasks x 256 KiB (= W13 bytes of 10 experts, 160 MiB). Each CTA reads one
task's contiguous 256 KiB either (lin) linearly, 32 KiB per step, or (kstep) like the expert
kernel: per step 128 rows x 256 B at a 2 KiB row stride, 8 steps. Rotates 3 copies."""
import json
import os

import torch
import triton
import triton.language as tl

assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"


@triton.jit
def reader(W, Sink, KSTEP: tl.constexpr, ROWB: tl.constexpr):
    pid = tl.program_id(0)
    base = W + pid.to(tl.int64) * 65536  # one task = 256 KiB = 65536 words
    acc = tl.zeros([2048], tl.int32)
    i = tl.arange(0, 2048)  # 2048 x 16 B = 32 KiB per step
    per: tl.constexpr = ROWB // 16           # 16 B units per row piece
    rows: tl.constexpr = 32768 // ROWB        # rows touched per step
    chunks: tl.constexpr = 2048 // ROWB       # pieces per 2 KiB row
    for step in range(8):
        if KSTEP:
            rg = step // chunks
            ch = step % chunks
            r = i // per
            c = i % per
            off = (rg * rows + r) * 128 + ch * per + c
        else:
            off = step * 2048 + i
        v = tl.load(base + off * 4)
        acc ^= v
    tl.store(Sink + pid * 2048 + i, acc)


def main():
    torch.manual_seed(0)
    n_cta = 640
    words = n_cta * 256 * 1024 // 4
    copies = [torch.randint(0, 1 << 30, (words,), device="cuda", dtype=torch.int32) for _ in range(3)]
    sink = torch.zeros(n_cta * 2048, device="cuda", dtype=torch.int32)
    for name, kstep, rowb in (("lin", False, 256), ("kstep256", True, 256), ("kstep512", True, 512), ("kstep1024", True, 1024)):
        gs = []
        for c in copies:
            reader[(n_cta,)](c, sink, KSTEP=kstep, ROWB=rowb, num_warps=4)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                reader[(n_cta,)](c, sink, KSTEP=kstep, ROWB=rowb, num_warps=4)
            gs.append(g)
        ts = []
        for _ in range(20):
            for g in gs:
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record(); g.replay(); e.record(); torch.cuda.synchronize()
                ts.append(s.elapsed_time(e) * 1000)
        ts.sort()
        med = ts[len(ts) // 2]
        print(json.dumps(dict(pattern=name, us=round(med, 1), TBps=round(160 * 2**20 / med / 1e6, 2))), flush=True)


if __name__ == "__main__":
    main()
