"""gfx1250 hardware clusters (GPU0), second attempt (MK.64): cluster-scope exchange as ATOM does it.

Per round every CTA of a cluster stores 128 words at SCOPE_SE (cluster scope: the shader engine's
L2), waits for its stores, passes a cluster barrier, reads all CLUSTER CTAs' words at SCOPE_SE and
counts stale ones, then a second cluster barrier. Rank in the cluster = pid % CLUSTER (uTLX has no
AMD lowering for cluster_cta_rank). BAR: "tlx" = tlx.cluster_barrier(), "asm" = s_barrier -3.
Reports µs per round (two barriers + CLUSTER x 128 words read) and stale counts.
"""
import json

import torch
import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_standin
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import ROW1


@triton.jit
def _st_se(ptr, v):
    return tlx.require_layout(tl.inline_asm_elementwise(
        "global_store_b32 $1, $2, off scope:SCOPE_SE\ns_wait_xcnt 0x0\nv_mov_b32 $0, 0",
        "=&v,v,v", [tlx.release_layout(ptr), tlx.release_layout(v)], dtype=tl.int32,
        is_pure=False, pack=1), ROW1)


@triton.jit
def _ld_se(ptr):
    return tlx.require_layout(tl.inline_asm_elementwise(
        "global_load_b32 $0, $1, off scope:SCOPE_SE\ns_wait_loadcnt 0x0\ns_wait_xcnt 0x0",
        "=v,v", [tlx.release_layout(ptr)], dtype=tl.int32, is_pure=False, pack=1), ROW1)


@triton.jit
def _cluster_barrier(dep, BAR: tl.constexpr):
    if BAR == "tlx":
        d = tlx.require_layout(tl.inline_asm_elementwise(
            "s_wait_storecnt 0x0\nv_mov_b32 $0, $1", "=v,v", [tlx.release_layout(dep)],
            dtype=tl.int32, is_pure=False, pack=1), ROW1)
        tlx.cluster_barrier()
        return d
    return tlx.require_layout(tl.inline_asm_elementwise(
        "s_wait_storecnt 0x0\ns_barrier_signal -3\ns_barrier_wait -3\nv_mov_b32 $0, $1",
        "=v,v", [tlx.release_layout(dep)], dtype=tl.int32, is_pure=False, pack=1), ROW1)


@triton.jit
def probe(Buf, Stale, Cycles, ROUNDS: tl.constexpr, CLUSTER: tl.constexpr, BAR: tl.constexpr):
    pid = tl.program_id(0)
    rank = pid % CLUSTER
    base = pid - rank
    i = tlx.require_layout(tl.arange(0, 128), ROW1)
    stale = tlx.require_layout(tl.zeros((128,), tl.int32), ROW1)
    t0 = tlx_standin.clock()
    for it in range(ROUNDS):
        val = it * 1000 + rank * 128 + i + 1
        z = _st_se(Buf + pid * 128 + i, val)
        z = _cluster_barrier(z, BAR)
        for r in tl.static_range(CLUSTER):
            got = _ld_se(Buf + (base + r) * 128 + i + z)
            stale += (got != it * 1000 + r * 128 + i + 1).to(tl.int32)
        z = _cluster_barrier(stale * 0, BAR)
    t1 = tlx_standin.clock()
    tl.store(Stale + pid * 128 + i, stale)
    tl.store(Cycles + pid, (t1 - t0).to(tl.int64))


def main():
    assert __import__("os").environ.get("HIP_VISIBLE_DEVICES") == "0"
    rounds = 200
    for bar in ("tlx", "asm"):
        for cluster in (2, 4, 8, 16):
            clusters = 192 // cluster
            n = clusters * cluster
            buf = torch.zeros(n * 128, device="cuda", dtype=torch.int32)
            stale = torch.zeros(n * 128, device="cuda", dtype=torch.int32)
            cyc = torch.zeros(n, device="cuda", dtype=torch.int64)
            try:
                probe[(n,)](buf, stale, cyc, ROUNDS=rounds, CLUSTER=cluster, BAR=bar,
                            num_warps=4, ctas_per_cga=(cluster, 1, 1))
                torch.cuda.synchronize()
                row = dict(barrier=bar, cluster=cluster, ctas_ran=int((cyc != 0).sum()), ctas=n,
                           us_per_round=round(float(cyc[cyc != 0].double().median()) / 100 / rounds, 3),
                           stale=int(stale.sum()), reads=n * 128 * cluster * rounds)
            except Exception as e:  # noqa: BLE001
                row = dict(barrier=bar, cluster=cluster, error=str(e)[-260:])
            print(json.dumps(row), flush=True)
            if "error" in row:
                break


if __name__ == "__main__":
    main()
