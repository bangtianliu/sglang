"""MK.8a (GPU0): does reading the S4 router weight ahead of time make the S4
projection faster? If yes, the S3 all-reduce wait (idle on 4 GPUs) can hide it.

Each sample: evict caches (stream 1 GiB), optionally prefetch a fraction of W4
(4608 x 8192 BF16 = 72 MiB), then time the production S4 projection alone.
"""

import json
import os

import torch
import triton
import triton.language as tl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_projection

assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"


@triton.jit
def touch(W, Sink, N_WORDS, BLOCK: tl.constexpr, CG: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    acc = tl.zeros([BLOCK], tl.int32)
    i = tl.arange(0, BLOCK)
    for start in range(pid * BLOCK, N_WORDS, nprog * BLOCK):
        if CG:
            acc += tl.load(W + start + i, mask=start + i < N_WORDS, other=0, cache_modifier=".cg")
        else:
            acc += tl.load(W + start + i, mask=start + i < N_WORDS, other=0)
    tl.store(Sink + pid, tl.sum(acc, 0))


def main():
    torch.manual_seed(0)
    W = (torch.randn(4608, 8192, device="cuda") * 0.02).bfloat16()
    X = torch.randn(8, 8192, device="cuda").bfloat16()
    P = torch.zeros(8, 4608, device="cuda", dtype=torch.float32)
    St = torch.zeros(288 * 4, device="cuda", dtype=torch.int32)
    fz = torch.zeros(2, device="cuda", dtype=torch.int32)
    big = torch.ones(256 * 1024 * 1024, device="cuda", dtype=torch.int32)
    sink = torch.zeros(4096, device="cuda", dtype=torch.int32)
    wwords = W.view(torch.int32).view(-1)

    def proj():
        tlx_projection.project[(288,)](X, W, P, St, Failure=fz, N=4608, K=8192, SK=8, STAGE=4,
                                       ROUND_OUTPUT=False, num_warps=2, enable_fp_fusion=False)

    def evict():
        touch[(1024,)](big, sink, big.numel(), BLOCK=1024, CG=True, num_warps=4)

    def pre(frac, ctas, cg):
        n = int(wwords.numel() * frac)
        touch[(ctas,)](wwords, sink, n, BLOCK=1024, CG=cg, num_warps=4)

    proj()
    torch.cuda.synchronize()
    ref = P.clone()
    res = {}
    for name, frac, ctas, cg in [("cold", 0, 0, False), ("pre_all_plain", 1.0, 1024, False),
                                 ("pre_all_cg", 1.0, 1024, True), ("pre_half", 0.5, 1024, False),
                                 ("pre_all_64cta", 1.0, 64, False), ("warm_repeat", -1, 0, False)]:
        ts, tp = [], []
        for _ in range(15):
            evict()
            if frac > 0:
                s0, e0 = torch.cuda.Event(True), torch.cuda.Event(True)
                s0.record()
                pre(frac, ctas, cg)
                e0.record()
            elif frac < 0:
                proj()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            proj()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1000)
            if frac > 0:
                tp.append(s0.elapsed_time(e0) * 1000)
            assert torch.equal(P, ref)
        ts.sort()
        tp.sort()
        res[name] = dict(proj_us=round(ts[len(ts) // 2], 2),
                         prefetch_us=round(tp[len(tp) // 2], 2) if tp else None)
        print(json.dumps({name: res[name]}), flush=True)


if __name__ == "__main__":
    main()
