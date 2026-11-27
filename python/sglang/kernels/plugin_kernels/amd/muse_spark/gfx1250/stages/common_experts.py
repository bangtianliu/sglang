"""One real EP rank: common-shape S5 with reference stand-in peer partials."""

import msgspec
import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    chain_guard,
    scoped,
    standin,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.common_state import (
    State,
    make_state,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.packed_memory import (
    load_tile_words,
    store_bf16_tile,
    store_bf16_vector,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.timeline import (
    stamp,
)
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


@g.jit
def stage(
    task_index,
    X,
    Ids,
    Scores,
    Sorted,
    Experts,
    NumPost,
    W13,
    S13,
    W2,
    S2,
    Gamma,
    Inter,
    Y,
    Scaled,
    Inv,
    Weighted,
    Partial,
    Reduced,
    Epochs,
    Barriers,
    Status,
    Scratch,
    Peers,
    Ready,
    GRID: gl.constexpr,
    BAR_STRIDE: gl.constexpr,
    PAYLOAD: gl.constexpr,
    FLAGS: gl.constexpr,
    RANK: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
    Trace=None,
    READ_MODE: gl.constexpr = 0,
    W2_SK: gl.constexpr = 1,
    KPartial=None,
    TaskTrace=None,
    Tag=None,
    INTERNAL: gl.constexpr = False,
    LayerOut=None,
    LayerCounter=None,
    LayerFlags=None,
    LayerPublish=True,
    PREFETCH_BEFORE_POLL: gl.constexpr = False,
):
    rank: gl.constexpr = RANK
    tag = _advance_tag(Epochs, task_index) if Tag is None else Tag
    stamp(Trace, task_index, 0)
    tasks = gl.load(NumPost, cache_modifier=".cg" if INTERNAL else "") // 16 * 64
    ok = task_index != SKIP
    delayed_point(Seed, task_index, 0)
    if task_index != SKIP:
        for task in range(task_index, tasks, GRID):
            stamp(TaskTrace, task, 0)
            block = task // 64
            expert = gl.load(Experts + block, cache_modifier=".cg" if INTERNAL else "")
            expert_tile(
                X,
                Sorted,
                W13,
                S13,
                Inter + rank * 64 * 4096,
                block,
                expert,
                task % 64,
                True,
                True,
                SCOPED_X=INTERNAL,
            )
            stamp(TaskTrace, task, 1)
            scoped.drain_stores()
            last = scoped.arrive_epoch(Barriers + block * 32, tag, 64)
            if last:
                scoped.publish_flags(
                    Barriers + 2048 + block * 128 * 32, tag, 64 * W2_SK
                )
            stamp(TaskTrace, task, 2)
    stamp(Trace, task_index, 1)
    delayed_point(Seed, task_index, 1)
    if task_index != SKIP:
        for task in range(task_index, tasks * W2_SK, GRID):
            block = task // (64 * W2_SK)
            tile = task % 64
            stamp(TaskTrace, task, 3)
            flag = Barriers + 2048 + (block * 128 + task % (64 * W2_SK)) * 32
            if PREFETCH_BEFORE_POLL:
                gl.static_assert(READ_MODE == 2 and W2_SK == 1 and TaskTrace is None)
                seen = tag
                polls = 0
            else:
                seen, polls = scoped.wait_flag(
                    Barriers + 2048 + (block * 128 + task % (64 * W2_SK)) * 32,
                    tag,
                    10_000_000,
                    MAX_SPINS=1_048_576,
                )
            stamp(TaskTrace, task, 4)
            if TaskTrace is not None:
                gl.store(TaskTrace + task * 32 + 12, polls)
            ok = ok & (seen == tag)
            stamp(Trace, task_index, 2)
            if ok:
                expert = gl.load(
                    Experts + block, cache_modifier=".cg" if INTERNAL else ""
                )
                ready = expert_tile(
                    Inter + rank * 64 * 4096,
                    Sorted,
                    W2,
                    S2,
                    (KPartial + (task // 64 % W2_SK) * 4 * 64 * 4096 + rank * 64 * 4096)
                    if W2_SK > 1
                    else Y + rank * 64 * 4096,
                    block,
                    expert,
                    tile,
                    False,
                    True,
                    READ_MODE,
                    W2_SK,
                    task // 64 % W2_SK,
                    INTERNAL,
                    WAIT_FLAG=flag if PREFETCH_BEFORE_POLL else None,
                    WAIT_TAG=tag,
                )
                if PREFETCH_BEFORE_POLL:
                    ok = ok & ready
                stamp(TaskTrace, task, 5)
                if ok:
                    scoped.drain_stores()
                    completed = gl.full(
                        (), True, gl.int1, gl.BlockedLayout([1], [32], [4], [0])
                    )
                    if W2_SK > 1:
                        completed = scoped.arrive_epoch(
                            Barriers + 299520 + (block * 64 + tile) * 32, tag, W2_SK
                        )
                        if completed:
                            l: gl.constexpr = gl.BlockedLayout(
                                [1, 4], [1, 32], [4, 1], [1, 0]
                            )
                            r = gl.arange(0, 16, layout=gl.SliceLayout(1, l))
                            c = (
                                gl.arange(0, 64, layout=gl.SliceLayout(0, l))
                                + tile * 64
                            )
                            hit = gl.load(Sorted + block * 16 + r)
                            ptr = rank * 64 * 4096 + hit[:, None] * 4096 + c[None, :]
                            v0 = load_tile_words(
                                KPartial + ptr, hit[:, None] < 64, 16 * 64
                            ).to(gl.float32, bitcast=True)
                            v1 = load_tile_words(
                                KPartial + 4 * 64 * 4096 + ptr,
                                hit[:, None] < 64,
                                16 * 64,
                            ).to(gl.float32, bitcast=True)
                            store_bf16_tile(Y + ptr, v0 + v1, hit[:, None] < 64, 16, 64)
                            scoped.drain_stores()
                    if completed:
                        last = scoped.arrive_epoch(
                            Barriers + 264192 + block * 32, tag, 64
                        )
                        if last:
                            scoped.publish_flags(
                                Barriers + 266240 + block * 16 * 32, tag, 16
                            )
                    stamp(TaskTrace, task, 6)
                stamp(Trace, task_index, 3)
                if tile < 16 and task // 64 % W2_SK == 0:
                    hit = gl.load(
                        Sorted + block * 16 + tile,
                        cache_modifier=".cg" if INTERNAL else "",
                    )
                    if hit < 64:
                        stamp(TaskTrace, task, 7)
                        seen, polls = scoped.wait_flag(
                            Barriers + 266240 + (block * 16 + tile) * 32,
                            tag,
                            10_000_000,
                            MAX_SPINS=1_048_576,
                        )
                        stamp(TaskTrace, task, 8)
                        if TaskTrace is not None:
                            gl.store(TaskTrace + task * 32 + 13, polls)
                        ok = ok & (seen == tag)
                        stamp(Trace, task_index, 4)
                        if ok:
                            _norm(Y, Gamma, Scaled, Inv, rank * 64 + hit, 32)
                            stamp(TaskTrace, task, 9)
                            scoped.drain_stores()
                            row = hit // 8
                            slots = gl.arange(
                                0, 8, layout=gl.BlockedLayout([1], [32], [4], [0])
                            )
                            ids = gl.load(
                                Ids + row * 8 + slots,
                                cache_modifier=".cg" if INTERNAL else "",
                            )
                            hits = gl.sum(
                                ((ids >= rank * 64) & (ids < (rank + 1) * 64)).to(
                                    gl.int32
                                ),
                                0,
                            )
                            last = scoped.arrive_epoch(
                                Barriers + 299008 + row * 32, tag, hits
                            )
                            if last:
                                scoped.publish(Barriers + 299264 + row * 32, tag)
                        stamp(TaskTrace, task, 10)
                stamp(Trace, task_index, 5)
    delayed_point(Seed, task_index, 2)
    if ok and task_index < 8:
        row = task_index
        slots = gl.arange(0, 8, layout=gl.BlockedLayout([1], [32], [4], [0]))
        ids = gl.load(Ids + row * 8 + slots, cache_modifier=".cg" if INTERNAL else "")
        hits = gl.sum(((ids >= rank * 64) & (ids < (rank + 1) * 64)).to(gl.int32), 0)
        token_polls = gl.full((), 0, gl.int32, gl.BlockedLayout([1], [32], [4], [0]))
        if hits > 0:
            seen, token_polls = scoped.wait_flag(
                Barriers + 299264 + row * 32, tag, 10_000_000, MAX_SPINS=1_048_576
            )
            ok = seen == tag
        if Trace is not None:
            gl.store(Trace + task_index * 32 + 24, token_polls)
        stamp(Trace, task_index, 6)
        if ok:
            _sum(Ids, Scores, Scaled, Inv, Weighted, Partial, rank, row, 32, INTERNAL)
            stamp(Trace, task_index, 7)
            delayed_point(Seed, task_index, 3)
            value, ok = standin.reduce_row(
                Partial,
                Scratch,
                Peers,
                Reduced,
                tag,
                row,
                4096,
                GRID,
                RANK,
                PAYLOAD,
                FLAGS,
                Trace,
            )
            if ok and LayerOut is not None:
                store_bf16_vector(LayerOut + row * 4096, value, 4096)
                scoped.drain_stores()
                if LayerPublish:
                    ticket = gl.atomic_add(LayerCounter, 1, sem="relaxed", scope="gpu")
                    if ticket + 1 == tag * 8:
                        scoped.publish_flags(LayerFlags, tag, 512)
    stamp(Trace, task_index, 31)
    gl.store(Status + task_index, ok)

    return ok


@g.jit
def _launch(
    X,
    Ids,
    Scores,
    Sorted,
    Experts,
    NumPost,
    W13,
    S13,
    W2,
    S2,
    Gamma,
    Inter,
    Y,
    Scaled,
    Inv,
    Weighted,
    Partial,
    Reduced,
    Epochs,
    Barriers,
    Status,
    Scratch,
    Peers,
    Ready,
    GRID: gl.constexpr,
    BAR_STRIDE: gl.constexpr,
    PAYLOAD: gl.constexpr,
    FLAGS: gl.constexpr,
    RANK: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
    Trace=None,
    READ_MODE: gl.constexpr = 0,
    W2_SK: gl.constexpr = 1,
    KPartial=None,
    TaskTrace=None,
    Failure=None,
    Out=None,
    GUARD_MODE: gl.constexpr = 0,
    PREFETCH_BEFORE_POLL: gl.constexpr = False,
):
    pid = gl.program_id(0)
    if GUARD_MODE == -1:
        mode = gl.load(Failure + 2)
    else:
        mode = gl.full((), GUARD_MODE, gl.int32, gl.BlockedLayout([1], [32], [4], [0]))
    if chain_guard.enabled_runtime(Failure, 5, mode):
        success = stage(
            gl.program_id(0),
            X,
            Ids,
            Scores,
            Sorted,
            Experts,
            NumPost,
            W13,
            S13,
            W2,
            S2,
            Gamma,
            Inter,
            Y,
            Scaled,
            Inv,
            Weighted,
            Partial,
            Reduced,
            Epochs,
            Barriers,
            Status,
            Scratch,
            Peers,
            Ready,
            GRID,
            BAR_STRIDE,
            PAYLOAD,
            FLAGS,
            RANK,
            SKIP,
            Seed,
            Trace,
            READ_MODE,
            W2_SK,
            KPartial,
            TaskTrace,
            LayerOut=Out,
            LayerPublish=False,
            PREFETCH_BEFORE_POLL=PREFETCH_BEFORE_POLL,
        )
        if mode == 0:
            chain_guard.finish(Failure, Status, pid, 5)
        elif mode == 1:
            chain_guard.finish_value(Failure, success, pid, 5)
    else:
        gl.store(Status + pid, 0)


class Workspace(msgspec.Struct):
    inter: torch.Tensor
    output: torch.Tensor
    scaled: torch.Tensor
    inv: torch.Tensor
    weighted: torch.Tensor
    partial: torch.Tensor
    state: State
    k_partial: torch.Tensor
    peers: standin.Standins


def make_workspace(device, *, shape=None):
    s = make_state(device, phases=17, shape=shape)

    def empty(shape, dtype=torch.float32):
        return torch.empty(shape, device=device, dtype=dtype)

    return Workspace(
        empty((4, 64, 4096), torch.bfloat16),
        empty((4, 64, 4096), torch.bfloat16),
        empty((4, 64, 4096)),
        empty((4, 64, 32)),
        empty((4, 8, 4096)),
        empty((4, 8, 4096), torch.bfloat16),
        s,
        empty((2, 4, 64, 4096), torch.int32),
        standin.make_standins(device, width=4096, ctas=s.grid),
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
    prefetch_before_poll=False,
):
    assert read_mode in (0, 1, 2) and w2_sk in (1, 2)
    if output is not None:
        assert (
            output.shape == (8, 4096)
            and output.dtype == torch.bfloat16
            and output.device == x.device
            and output.is_contiguous()
        )
    s = ws.state
    p = ws.peers
    args = (
        x,
        ids,
        scores,
        *table,
        weights["w13"],
        weights["w13_scale"],
        weights["w2"],
        weights["w2_scale"],
        gamma,
        ws.inter,
        ws.output,
        ws.scaled,
        ws.inv,
        ws.weighted,
        ws.partial,
        p.reduced,
        s.epochs,
        s.barriers,
        s.status,
        p.scratch,
        p.peers,
        p.ready,
    )
    opts = dict(
        GRID=s.grid,
        BAR_STRIDE=s.stride,
        PAYLOAD=p.payload,
        FLAGS=p.flags,
        RANK=rank,
        SKIP=skip,
        Seed=delay_seed,
        Trace=trace,
        READ_MODE=read_mode,
        W2_SK=w2_sk,
        KPartial=ws.k_partial,
        TaskTrace=task_trace,
        **s.shape.options(),
        Failure=failure,
        GUARD_MODE=guard_mode,
        Out=output,
        PREFETCH_BEFORE_POLL=prefetch_before_poll,
    )
    if compile_only:
        return _launch.warmup(*args, **opts, grid=(s.grid,))
    key = (
        prefetch_before_poll,
        task_trace is not None,
        w2_sk,
        read_mode,
        trace is not None,
        rank,
        gamma.dtype,
        skip,
        delay_seed is not None,
        failure is not None,
        guard_mode,
        output is not None,
    )
    if key not in s.admitted:
        kernel = _launch.warmup(*args, **opts, grid=(s.grid,))
        r = s.shape.admit("common-S5", kernel, grid=s.grid)
        s.admitted[key] = dict(
            vgprs=kernel.n_regs,
            lds=kernel.metadata.shared,
            spills=kernel.n_spills,
            occupancy=r.active_ctas_per_cu,
        )
    return _launch[s.grid,](*args, **opts)
