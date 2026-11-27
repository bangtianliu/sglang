"""gfx1250 hardware clusters through framework/tlx_cluster.py (GPU0). env CLUSTER=N (one per run).

Per round each CTA stores 128 words at cluster scope, cluster barrier, reads all N CTAs' words at
cluster scope (counting stale ones), cluster barrier. Reports µs per round and stale counts, and
a control with the workgroup barrier only (expected stale > 0 if the cluster barrier matters).
"""
import json
import os

import torch
import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_cluster, tlx_standin
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import ROW1

N = int(os.environ.get("CLUSTER", "4"))


@triton.jit
def probe(Buf, Stale, Cycles, ROUNDS: tl.constexpr, CLUSTER: tl.constexpr, USE: tl.constexpr):
    pid = tl.program_id(0)
    rank = pid % CLUSTER
    base = pid - rank
    i = tlx.require_layout(tl.arange(0, 128), ROW1)
    stale = tlx.require_layout(tl.zeros((128,), tl.int32), ROW1)
    t0 = tlx_standin.clock()
    for it in range(ROUNDS):
        val = it * 1000 + rank * 128 + i + 1
        z = tlx.require_layout(tlx_cluster.store_words(
            tlx.release_layout(Buf + pid * 128 + i), tlx.release_layout(val)), ROW1)
        if USE:
            z = tlx.require_layout(tlx_cluster.barrier(tlx.release_layout(z), tlx.release_layout(i)), ROW1)
        else:
            tlx.workgroup_barrier()
        for r in tl.static_range(CLUSTER):
            got = tlx.require_layout(tlx_cluster.load_words(
                tlx.release_layout(Buf + (base + r) * 128 + i + z)), ROW1)
            stale += (got != it * 1000 + r * 128 + i + 1).to(tl.int32)
        if USE:
            z = tlx.require_layout(tlx_cluster.barrier(tlx.release_layout(stale * 0), tlx.release_layout(i)), ROW1)
        else:
            tlx.workgroup_barrier()
    t1 = tlx_standin.clock()
    tl.store(Stale + pid * 128 + i, stale)
    tl.store(Cycles + pid, (t1 - t0).to(tl.int64))


def main():
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
    tlx_cluster.clustered(probe, N)
    rounds = 200
    clusters = 192 // N
    n = clusters * N
    for use in (1, 0):
        buf = torch.zeros(n * 128, device="cuda", dtype=torch.int32)
        stale = torch.zeros(n * 128, device="cuda", dtype=torch.int32)
        cyc = torch.zeros(n, device="cuda", dtype=torch.int64)
        try:
            probe[(clusters,)](buf, stale, cyc, ROUNDS=rounds, CLUSTER=N, USE=use, num_warps=4)
            torch.cuda.synchronize()
            ran = int((cyc != 0).sum())
            row = dict(cluster=N, cluster_barrier=bool(use), ctas_ran=ran, ctas=n,
                       us_per_round=round(float(cyc[cyc != 0].double().median()) / 100 / rounds, 3)
                       if ran else None,
                       stale=int(stale.sum()), reads=n * 128 * N * rounds)
        except Exception as e:  # noqa: BLE001
            row = dict(cluster=N, cluster_barrier=bool(use), error=str(e)[-300:])
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
