"""gfx1250 workgroup clusters (GPU0): cost and visibility of an in-kernel exchange.

Each round: every CTA of a cluster writes 128 words (round-dependent), drains its
stores, cluster_barrier, reads all CLUSTER CTAs' words and counts stale ones,
cluster_barrier again. Reports cycles (realtime 100 MHz) per round and stale counts
for plain / .cg / SCOPE_DEV loads. Clusters are spread over the whole GPU.
"""
import json
import sys

import torch
import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_memory, tlx_standin
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import ROW1


@triton.jit
def probe(Buf, Stale, Cycles, ROUNDS: tl.constexpr, CLUSTER: tl.constexpr, MODE: tl.constexpr,
          BARRIER: tl.constexpr):
    pid = tl.program_id(0)
    rank = tlx.cluster_cta_rank()
    base = pid - rank  # first CTA of this cluster
    i = tlx.require_layout(tl.arange(0, 128), ROW1)
    stale = tl.zeros((128,), tl.int32)
    t0 = tlx_standin.clock()
    for it in range(ROUNDS):
        val = it * 1000 + rank * 128 + i + 1
        tl.store(Buf + pid * 128 + i, val)
        tlx_memory.drain_stores()
        if BARRIER:
            tlx.cluster_barrier()
        for r in tl.static_range(CLUSTER):
            ptr = Buf + (base + r) * 128 + i
            if MODE == 0:
                got = tl.load(ptr)
            elif MODE == 1:
                got = tl.load(ptr, cache_modifier=".cg")
            else:
                got = tlx_memory.load_words(ptr)
            stale += (got != it * 1000 + r * 128 + i + 1).to(tl.int32)
        if BARRIER:
            tlx.cluster_barrier()
        else:
            tlx.workgroup_barrier()
    t1 = tlx_standin.clock()
    tl.store(Stale + pid * 128 + i, stale)
    tl.store(Cycles + pid, (t1 - t0).to(tl.int64))


def main():
    assert __import__("os").environ.get("HIP_VISIBLE_DEVICES") == "0"
    rounds = 200
    out = []
    for cluster in (2, 4, 8, 16):
        for mode in (0, 1, 2):
            clusters = 192 // cluster
            n = clusters * cluster
            buf = torch.zeros(n * 128, device="cuda", dtype=torch.int32)
            stale = torch.zeros(n * 128, device="cuda", dtype=torch.int32)
            cyc = torch.zeros(n, device="cuda", dtype=torch.int64)
            try:
                probe[(clusters,)](buf, stale, cyc, ROUNDS=rounds, CLUSTER=cluster, MODE=mode,
                                   BARRIER=True, num_warps=4, num_ctas=cluster)
                torch.cuda.synchronize()
                out.append(dict(cluster=cluster, mode=["plain", "cg", "scope_dev"][mode],
                                us_per_round=round(float(cyc.double().median()) / 100 / rounds, 3),
                                stale=int(stale.sum()), reads=n * 128 * cluster * rounds))
            except Exception as e:  # noqa: BLE001
                out.append(dict(cluster=cluster, mode=mode, error=str(e)[:300]))
            print(json.dumps(out[-1]), flush=True)


if __name__ == "__main__":
    main()
