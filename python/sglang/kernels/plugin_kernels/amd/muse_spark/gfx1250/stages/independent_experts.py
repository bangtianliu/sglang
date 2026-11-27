"""Independent S5 phases: nine-CTA/CU expert roots, unchanged norm/sum/AR."""

import msgspec
import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    chain_guard,
    standin,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.packed_memory import (
    store_bf16_vector,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import common_experts
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.boundary import (
    delayed_point,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.experts_async import (
    expert_tile,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.experts_complete import (
    _norm,
    _sum,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.qkvg import (
    _advance_tag,
)

_make_common_workspace = common_experts.make_workspace


@g.jit
def expert_phase(
    X,
    Sorted,
    Experts,
    NumPost,
    W,
    Scale,
    Out,
    Status,
    Failure,
    GRID: gl.constexpr,
    FIRST: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
    WEIGHT_CACHE: gl.constexpr = "",
):
    pid = gl.program_id(0)
    allowed = chain_guard.enabled(Failure, 5) if FIRST else gl.load(Failure) == 0
    ok = allowed & (pid != SKIP)
    delayed_point(Seed, pid, 0)
    if ok:
        tasks = gl.load(NumPost) // 16 * 64
        for task in range(pid, tasks, GRID):
            block = task // 64
            expert = gl.load(Experts + block)
            # Keep the accepted 4.0c constant async.cg A-read policy in W2.
            expert_tile(
                X,
                Sorted,
                W,
                Scale,
                Out,
                block,
                expert,
                task % 64,
                FIRST,
                not FIRST,
                READ_MODE=2,
                SCOPED_OUTPUT=False,
                WEIGHT_CACHE=WEIGHT_CACHE,
            )
    delayed_point(Seed, pid, 1)
    gl.store(Status + pid, ok)
    if allowed:
        chain_guard.finish_value(Failure, ok, pid, 5)


@g.jit
def norm_phase(
    Y, Gamma, Ids, Scaled, Inv, Status, Failure, RANK: gl.constexpr, Seed=None
):
    hit = gl.program_id(0)
    ok = gl.load(Failure) == 0
    if ok:
        expert = gl.load(Ids + hit)
        if (expert >= RANK * 64) & (expert < (RANK + 1) * 64):
            _norm(Y, Gamma, Scaled, Inv, RANK * 64 + hit, 32)
    delayed_point(Seed, hit, 2)
    if not ok:
        gl.store(Status + hit, 0)


@g.jit
def sum_reduce(
    Ids,
    Scores,
    Scaled,
    Inv,
    Weighted,
    Partial,
    Scratch,
    Peers,
    Reduced,
    Epochs,
    Status,
    Failure,
    Out,
    GRID: gl.constexpr,
    RANK: gl.constexpr,
    PAYLOAD: gl.constexpr,
    FLAGS: gl.constexpr,
    Seed=None,
):
    row = gl.program_id(0)
    tag = _advance_tag(Epochs, row)
    ok = gl.load(Failure) == 0
    if ok:
        _sum(Ids, Scores, Scaled, Inv, Weighted, Partial, RANK, row, 32, False)
        delayed_point(Seed, row, 3)
        value, ok = standin.reduce_row(
            Partial, Scratch, Peers, Reduced, tag, row, 4096, GRID, RANK, PAYLOAD, FLAGS
        )
        if ok:
            store_bf16_vector(Out + row * 4096, value, 4096)
    gl.store(Status + row, ok)
    chain_guard.finish_value(Failure, ok, row, 5)


class Workspace(common_experts.Workspace):
    local_failure: object = None
    layer_out: object = None
    kernels: dict = msgspec.field(default_factory=dict)


def make_workspace(device):
    b = _make_common_workspace(device)
    b.state.status = torch.empty((1728,), device=device, dtype=torch.int32)
    b.state.grid = 1728
    b.state.epochs = torch.zeros((1728, 32), device=device, dtype=torch.int32)
    return Workspace(
        b.inter,
        b.output,
        b.scaled,
        b.inv,
        b.weighted,
        b.partial,
        b.state,
        b.k_partial,
        b.peers,
        torch.zeros((2,), device=device, dtype=torch.int32),
        torch.empty((8, 4096), device=device, dtype=torch.bfloat16),
    )


def s5(
    x,
    ids,
    scores,
    table,
    weights,
    gamma,
    ws,
    *,
    rank,
    skip=-1,
    delay_seed=None,
    compile_only=False,
    trace=None,
    read_mode=2,
    w2_sk=1,
    task_trace=None,
    failure=None,
    guard_mode=0,
    output=None,
    weight_cache="",
):
    assert (
        trace is None
        and task_trace is None
        and read_mode == 2
        and w2_sk == 1
        and guard_mode == 0
    )
    assert weight_cache in ("", ".cs")
    s = ws.state
    p = ws.peers
    failure = ws.local_failure if failure is None else failure
    output = ws.layer_out if output is None else output
    sorted_ids, experts, num_post = table
    common = dict(GRID=1728, num_warps=4, waves_per_eu=9, enable_fp_fusion=False)
    a1 = (
        x,
        sorted_ids,
        experts,
        num_post,
        weights["w13"],
        weights["w13_scale"],
        ws.inter[rank],
        s.status,
        failure,
    )
    a2 = (
        ws.inter[rank],
        sorted_ids,
        experts,
        num_post,
        weights["w2"],
        weights["w2_scale"],
        ws.output[rank],
        s.status,
        failure,
    )
    k1 = dict(
        **common, FIRST=True, SKIP=skip, Seed=delay_seed, WEIGHT_CACHE=weight_cache
    )
    k2 = dict(**common, FIRST=False, Seed=delay_seed, WEIGHT_CACHE=weight_cache)
    an = (ws.output, gamma, ids, ws.scaled, ws.inv, s.status, failure)
    kn = dict(RANK=rank, Seed=delay_seed, num_warps=4, enable_fp_fusion=False)
    ar = (
        ids,
        scores,
        ws.scaled,
        ws.inv,
        ws.weighted,
        ws.partial,
        p.scratch,
        p.peers,
        p.reduced,
        s.epochs,
        s.status,
        failure,
        output,
    )
    kr = dict(
        GRID=p.ctas,
        RANK=rank,
        PAYLOAD=p.payload,
        FLAGS=p.flags,
        Seed=delay_seed,
        num_warps=4,
        enable_fp_fusion=False,
    )
    key = (rank, skip, delay_seed is not None, gamma.dtype, weight_cache)
    phases = [
        ("w13", expert_phase, a1, k1, 1728),
        ("w2", expert_phase, a2, k2, 1728),
        ("norm", norm_phase, an, kn, 64),
        ("sum_AR", sum_reduce, ar, kr, 8),
    ]
    if key not in s.admitted:
        records = []
        kernels = []
        for name, fn, aa, kk, grid in phases:
            k = fn.warmup(*aa, **kk, grid=(grid,))
            r = kernel_occupancy("independent-S5-" + name, k)
            assert r.active_ctas_per_cu >= (9 if name in ("w13", "w2") else 1), (
                name,
                r,
            )
            records.append(
                dict(
                    phase=name,
                    vgprs=k.n_regs,
                    spills=k.n_spills,
                    lds=k.metadata.shared,
                    occupancy=r.active_ctas_per_cu,
                    waves=k.metadata.num_warps,
                    grid=grid,
                )
            )
            kernels.append(k)
        s.admitted[key] = records
        ws.kernels[key] = kernels
    if not compile_only:
        for name, fn, aa, kk, grid in phases:
            fn[(grid,)](*aa, **kk)
    return ws.kernels[key][1]
