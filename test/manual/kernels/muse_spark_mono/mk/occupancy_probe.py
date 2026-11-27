"""MK.9b (GPU0): how many CTAs of a 4-warp kernel are co-resident at a given LDS size?
Each CTA allocates LDS, spins ~SPIN us, and records start/end realtime; we count the
max number of overlapping CTAs."""
import json
import os
import sys

import torch
import triton
import triton.language as tl
import utlx_plugin as tlx

sys.path.insert(0, "python")
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_standin

assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"


@triton.jit
def occ(Out, Sink, BANKS: tl.constexpr, SPIN: tl.constexpr):
    pid = tl.program_id(0)
    t0 = tlx_standin.clock()
    sh: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([(512, 8)], [64, 512], [1, 0])
    sb = tlx.local_alloc((64, 512), tl.bfloat16, BANKS, layout=sh)
    x = tl.full((64, 512), 1.0, tl.bfloat16)
    acc = tl.zeros((64, 512), tl.float32)
    for j in tl.static_range(BANKS):
        tlx.local_store(tlx.local_view(sb, j), x)
    tlx.workgroup_barrier()
    for j in tl.static_range(BANKS):
        acc += tlx.local_load(tlx.local_view(sb, j)).to(tl.float32)
    t = tlx_standin.clock()
    while t - t0 < SPIN:
        t = tlx_standin.clock()
    tl.store(Sink + pid, tl.sum(tl.sum(acc, 1), 0))
    k = tl.arange(0, 2)
    tl.store(Out + pid * 2 + k, tl.where(k == 0, t0, t).to(tl.int64))


def main():
    res = {}
    for banks in (1, 2, 3, 4):
        n = 1024
        out = torch.zeros(n * 2, device="cuda", dtype=torch.int64)
        sink = torch.zeros(n, device="cuda", dtype=torch.float32)
        for _ in range(2):
            occ[(n,)](out, sink, BANKS=banks, SPIN=5000, num_warps=4)  # 50 us
        torch.cuda.synchronize()
        t = out.view(n, 2).cpu()
        ev = sorted([(int(a), 1) for a in t[:, 0]] + [(int(b), -1) for b in t[:, 1]], key=lambda e: (e[0], e[1]))
        cur = mx = 0
        for _, d in ev:
            cur += d
            mx = max(mx, cur)
        k = next(iter(occ.device_caches.values()))[0]
        lds = [kk.metadata.shared for kk in k.values()][-1]
        res[banks] = dict(lds_bytes=lds, max_coresident=mx, span_us=round((int(t[:, 1].max()) - int(t[:, 0].min())) / 100, 1))
        print(json.dumps({banks: res[banks]}), flush=True)


if __name__ == "__main__":
    main()
