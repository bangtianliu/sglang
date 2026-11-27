"""One rank's S1..S6 in one launch, fixed task lists and tagged dependencies."""

import msgspec
import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    layer_flow as flow,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    scoped,
    standin,
    vector_memory,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.common_shape import (
    CommonShape,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.packed_memory import (
    store_bf16_tile,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.timeline import (
    stamp,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_attention,
    common_attn_tail,
    common_experts,
    common_qkvg,
    common_route,
    common_tail,
    layer_accepted,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    parallel_rows as rows,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    producer_rows as producers,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.attention import (
    _merge,
    _store128,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.boundary import (
    delayed_point,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.qkvg import (
    _advance_tag,
    head_epilogue,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.route import (
    _finish_row,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.slot_table import table
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.streamed_gemm import (
    partial_tile,
    reduce_tile,
)

WEIGHT_KEYS = (
    "qkvg",
    "o_proj",
    "joint_router_pooler",
    "w13",
    "w13_scale",
    "w2",
    "w2_scale",
    "unpooler",
    "pre3",
    "alpha3",
    "beta3",
    "bias",
    "route_norm",
    "gamma5",
    "post6",
    "pre6",
    "alpha6",
    "beta6",
)


@g.jit
def _launch(
    X,
    Stream,
    W,
    M,
    D,
    Epochs,
    Counters,
    Flags,
    Status,
    Diag,
    Metrics,
    Row3,
    Row4,
    Row6,
    GRID: gl.constexpr,
    RANK: gl.constexpr,
    LEFT: gl.constexpr,
    SPLITS: gl.constexpr,
    MAX_PAGES: gl.constexpr,
    PAY3: gl.constexpr,
    FL3: gl.constexpr,
    PAY5: gl.constexpr,
    FL5: gl.constexpr,
    PREFETCH: gl.constexpr = True,
    SKIP_STAGE: gl.constexpr = -1,
    SKIP_TASK: gl.constexpr = 0,
    Seed=None,
    Trace=None,
    WaitTrace=None,
    ExpertTrace=None,
    TaskTrace=None,
    TAIL_MODE: gl.constexpr = 3,
    TASK_TIMES: gl.constexpr = True,
    RowTrace=None,
    ProfileTasks=None,
):
    (
        Wq,
        Wo,
        Wr,
        W13,
        S13,
        W2,
        S2,
        Wu,
        Pre3,
        Alpha3,
        Beta3,
        Bias,
        RouteNorm,
        Gamma5,
        Post6,
        Pre6,
        Alpha6,
        Beta6,
    ) = W
    POS, COS, LOC, Map, KC, VC, KS, VS, Pages, Length = M
    (
        P1,
        F1,
        Q,
        K,
        V,
        G,
        AP,
        AL,
        AOut,
        P3,
        F3,
        Part3,
        Scratch3,
        Peers3,
        Reduced3,
        X4,
        MidStream,
        P4,
        F4,
        Ids,
        Scores,
        Pool,
        Sorted,
        Experts,
        NumPost,
        Inter,
        Y,
        Scaled,
        Inv,
        Weighted,
        Part5,
        Scratch5,
        Peers5,
        Reduced5,
        X6,
        P6,
        F6,
        Out,
        Updated,
        Bar5,
        Status5,
        Ready5,
    ) = D
    pid = gl.program_id(0)
    tag = _advance_tag(Epochs, pid)
    z = gl.arange(0, 32, layout=gl.BlockedLayout([1], [32], [4], [0]))
    gl.store(Metrics + pid * 32 + z, gl.full_like(z, 0))
    gl.store(Diag + pid * 16 + gl.minimum(z, 15), gl.full_like(z, 0), z < 16)
    ok = gl.full((), True, gl.int1, gl.BlockedLayout([1], [32], [4], [0]))
    PT1 = ProfileTasks + 0 * 4096 * 32 if ProfileTasks is not None else None
    PT2 = ProfileTasks + 1 * 4096 * 32 if ProfileTasks is not None else None
    PT3 = ProfileTasks + 2 * 4096 * 32 if ProfileTasks is not None else None
    PT4 = ProfileTasks + 3 * 4096 * 32 if ProfileTasks is not None else None
    PT6 = ProfileTasks + 5 * 4096 * 32 if ProfileTasks is not None else None
    stamp(Trace, pid, 0)
    delayed_point(Seed, pid, 0)
    # Each Q/K/V/gate head contributes once to its KV group's18-head dependency.
    for task in range(pid, 576, GRID):
        if SKIP_STAGE == 1 and task == SKIP_TASK:
            ok = gl.full((), False, gl.int1, gl.BlockedLayout([1], [32], [4], [0]))
            flow.note(Diag, pid, 1, 1, task, tag, -1, 0)
        else:
            head = task // 8
            if TASK_TIMES:
                stamp(PT1, task, 0)
            partial_tile(X, Wq, P1, task, 4608, 8192, 8, 64)
            if TASK_TIMES:
                stamp(PT1, task, 1)
            if flow.arrive(F1 + head * 32, tag, 8):
                value = reduce_tile(P1, F1 + 2048 * 32, head, 4608, 8, True)
                scoped.publish_flags(Flags + (flow.S1_HEAD + head * 4) * 32, tag, 4)
            if TASK_TIMES:
                stamp(PT1, task, 2)
    if pid >= 576:
        for task in range(pid - 576, 288, GRID - 576):
            if TASK_TIMES:
                stamp(PT1, 576 + task, 0)
            ready = rows.wait_line(
                Flags + (flow.S1_HEAD + task) * 32,
                tag,
                pid,
                task,
                Diag,
                1,
                PT1 + 576 * 32 if PT1 is not None else None,
            )
            if TASK_TIMES:
                stamp(PT1, 576 + task, 1)
            ok = ok & ready
            if ready:
                head = task // 4
                head_epilogue(
                    F1 + 2048 * 32,
                    head,
                    POS,
                    COS,
                    LOC,
                    Map,
                    KC,
                    VC,
                    Q,
                    K,
                    V,
                    G,
                    KS,
                    VS,
                    1e-5,
                    5.4730064863875,
                    LEFT >= 0,
                    LEFT >= 0,
                    SCOPED_IO=True,
                    RowStart=(task % 4) * 2,
                    PAD_ROWS=2,
                )
                group = gl.where(
                    head < 32,
                    head // 8,
                    gl.where(
                        head < 36,
                        head - 32,
                        gl.where(head < 40, head - 36, (head - 40) // 8),
                    ),
                )
                if flow.arrive(Counters + group * 32, tag, 72):
                    scoped.publish_flags(
                        Flags + (flow.S2_INPUT + group * 128) * 32, tag, SPLITS
                    )
            if TASK_TIMES:
                stamp(PT1, 576 + task, 2)
    gl.store(Metrics + pid * 32 + 16, 1)
    gl.store(Status + pid, ok)
    stamp(Trace, pid, 1)
    stamp(Trace, pid, 2)
    if ok and pid < SPLITS * 4:
        task = pid
        group = task % 4
        split = task // 4
        ok = flow.wait(
            Flags + (flow.S2_INPUT + group * 128 + split) * 32,
            tag,
            pid,
            task,
            Diag,
            Metrics,
            Wo,
            Wr,
            8192 * 2048 // 2,
            4608 * 8192 // 2,
            2,
            1,
            GRID,
            PREFETCH,
            WaitTrace,
        )
        if ok:
            if SKIP_STAGE == 2 and task == SKIP_TASK:
                ok = gl.full((), False, gl.int1, gl.BlockedLayout([1], [32], [4], [0]))
                flow.note(Diag, pid, 2, 2, task, tag, -1, 0)
            else:
                if TASK_TIMES:
                    stamp(PT2, task, 0)
                value, lse = common_attention.partial(
                    Q,
                    KC,
                    VC,
                    Pages,
                    Map,
                    Length,
                    KS,
                    VS,
                    group,
                    split,
                    0.125,
                    LEFT,
                    SPLITS,
                    LEFT >= 0,
                    False,
                    MAX_PAGES,
                    True,
                )
                if TASK_TIMES:
                    stamp(PT2, task, 1)
                ml: gl.constexpr = value.type.layout
                row = gl.arange(0, 64, layout=gl.SliceLayout(1, ml))
                a, b, c, d = vector_memory.split4(
                    value.reshape((4096,)).to(gl.int32, bitcast=True), 1024
                )
                ids = gl.arange(0, 1024, layout=gl.BlockedLayout([1], [32], [4], [0]))
                _store128(
                    AP
                    + ((group * 64 + ids // 16) * SPLITS + split) * 64
                    + (ids % 16) * 4,
                    a,
                    b,
                    c,
                    d,
                    1024,
                )
                scoped.store_words(
                    AL + (group * 64 + row) * SPLITS + split,
                    lse.to(gl.int32, bitcast=True),
                    row >= 0,
                )
                if flow.arrive(Counters + (4 + group) * 32, tag, SPLITS):
                    scoped.publish_flags(
                        Flags + (flow.S2_MERGE + group * 64) * 32, tag, 64
                    )
                if TASK_TIMES:
                    stamp(PT2, task, 2)
    if ok and pid < 256:
        group = pid // 64
        ok = flow.wait(
            Flags + (flow.S2_MERGE + pid) * 32,
            tag,
            pid,
            pid,
            Diag,
            Metrics,
            Wo,
            Wr,
            8192 * 2048 // 2,
            4608 * 8192 // 2,
            2,
            2,
            GRID,
            PREFETCH,
            WaitTrace,
        )
        if ok:
            if TASK_TIMES:
                stamp(PT2, 512 + pid, 0)
            _merge(
                AP,
                AL,
                Flags + flow.S2_MERGE * 32,
                Status + GRID * 6,
                AOut,
                pid,
                tag,
                SPLITS,
                None,
                pid,
                True,
                True,
            )
            if flow.arrive(Counters + (8 + group) * 32, tag, 64):
                scoped.publish_flags(
                    Flags + (flow.S3_INPUT + group * 128) * 32, tag, 128
                )
            if TASK_TIMES:
                stamp(PT2, 512 + pid, 2)
    gl.store(Status + GRID + pid, ok)
    stamp(Trace, pid, 3)
    stamp(Trace, pid, 4)
    delayed_point(Seed, pid, 1)
    if ok:
        for task in range(pid, 512, GRID):
            tile = task // 4
            group = task % 4
            ok = flow.wait(
                Flags + (flow.S3_INPUT + group * 128 + tile) * 32,
                tag,
                pid,
                task,
                Diag,
                Metrics,
                Wo,
                Wr,
                8192 * 2048 // 2,
                4608 * 8192 // 2,
                3,
                2,
                GRID,
                PREFETCH,
                WaitTrace,
            )
            if ok:
                if SKIP_STAGE == 3 and task == SKIP_TASK:
                    ok = gl.full(
                        (), False, gl.int1, gl.BlockedLayout([1], [32], [4], [0])
                    )
                    flow.note(Diag, pid, 3, 3, task, tag, -1, 0)
                else:
                    if TASK_TIMES:
                        stamp(PT3, task, 0)
                    partial_tile(AOut, Wo, P3, task, 8192, 2048, 4, 128, True, Gate=G)
                    if TASK_TIMES:
                        stamp(PT3, task, 1)
                    if flow.arrive(F3 + tile * 32, tag, 4):
                        value = reduce_tile(P3, F3 + 2048 * 32, tile, 8192, 4, True)
                        l: gl.constexpr = value.type.layout
                        r = gl.arange(0, 8, layout=gl.SliceLayout(1, l))
                        c = gl.arange(0, 64, layout=gl.SliceLayout(0, l)) + tile * 64
                        store_bf16_tile(
                            Part3 + RANK * 8 * 8192 + r[:, None] * 8192 + c[None, :],
                            value,
                            r[:, None] >= 0,
                            8,
                            64,
                        )
                        if TAIL_MODE == 3:
                            rows.put_tile(value, tile, Peers3, tag, RANK, PAY3, 8192)
                            if flow.arrive(F3 + 128 * 32, tag, 128):
                                rows.publish_sources(Peers3, tag, RANK, FL3, 1024)
                        else:
                            summed, ready = producers.collective(
                                value, tile, Peers3, Scratch3, tag, RANK, PAY3, FL3, PT3
                            )
                            ok = ok & ready
                            if ready:
                                producers.produce(
                                    summed,
                                    tile,
                                    Row3,
                                    tag,
                                    Stream,
                                    None,
                                    Pre3,
                                    Alpha3,
                                    Beta3,
                                    X4,
                                    MidStream,
                                    8192,
                                    TAIL_MODE,
                                    False,
                                    (RowTrace + 0 * GRID * 32)
                                    if RowTrace is not None
                                    else None,
                                )
                            else:
                                flow.note(Diag, pid, 3, 3, tile, tag, -1, 0)
                    if TASK_TIMES:
                        stamp(PT3, task, 2)
    if ok:
        if TAIL_MODE == 3:
            ok = rows.boundary(
                pid,
                tag,
                Row3,
                None,
                Stream,
                None,
                Pre3,
                Alpha3,
                Beta3,
                X4,
                MidStream,
                Counters + 12 * 32,
                Flags + flow.S4_INPUT * 32,
                Diag,
                Metrics,
                GRID,
                3,
                True,
                Scratch3,
                RANK,
                PAY3,
                FL3,
                Trace=(RowTrace + 0 * GRID * 32) if RowTrace is not None else None,
                Record=PT3 + 1024 * 32 if PT3 is not None else None,
            )
        else:
            ok = producers.apply(
                pid,
                tag,
                Row3,
                Stream,
                Pre3,
                Alpha3,
                Beta3,
                X4,
                MidStream,
                Diag,
                Metrics,
                8192,
                TAIL_MODE,
                False,
                3,
                Counters + 12 * 32,
                Flags + flow.S4_INPUT * 32,
                576,
                Trace=(RowTrace + 0 * GRID * 32) if RowTrace is not None else None,
                Record=PT3 + 1024 * 32 if PT3 is not None else None,
            )
    gl.store(Status + GRID * 2 + pid, ok)
    stamp(Trace, pid, 5)
    stamp(Trace, pid, 6)
    if ok:
        for task in range(pid, 576, GRID):
            ok = flow.wait(
                Flags + (flow.S4_INPUT + task) * 32,
                tag,
                pid,
                task,
                Diag,
                Metrics,
                Wo,
                Wr,
                8192 * 2048 // 2,
                4608 * 8192 // 2,
                4,
                3,
                GRID,
                PREFETCH,
                WaitTrace,
            )
            if ok:
                if SKIP_STAGE == 4 and task == SKIP_TASK:
                    ok = gl.full(
                        (), False, gl.int1, gl.BlockedLayout([1], [32], [4], [0])
                    )
                    flow.note(Diag, pid, 4, 4, task, tag, -1, 0)
                else:
                    tile = task // 8
                    if TASK_TIMES:
                        stamp(PT4, task, 0)
                    partial_tile(X4, Wr, P4, task, 4608, 8192, 8, 64, True)
                    if TASK_TIMES:
                        stamp(PT4, task, 1)
                    if flow.arrive(F4 + tile * 32, tag, 8):
                        value = reduce_tile(P4, F4 + 2048 * 32, tile, 4608, 8, False)
                        if TAIL_MODE != 3 and tile < 64:
                            producers.produce(
                                value,
                                tile,
                                Row4,
                                tag,
                                None,
                                None,
                                RouteNorm,
                                None,
                                None,
                                Pool,
                                None,
                                4096,
                                TAIL_MODE,
                                True,
                                (RowTrace + GRID * 32)
                                if RowTrace is not None
                                else None,
                            )
                        if flow.arrive(F4 + 72 * 32, tag, 72):
                            scoped.publish_flags(Flags + flow.S4_ROWS * 32, tag, 8)
                            if TAIL_MODE == 3:
                                rows.publish_ready(Row4, tag, 4096)
                    if TASK_TIMES:
                        stamp(PT4, task, 2)
    if ok and pid < 8:
        ok = flow.wait(
            Flags + (flow.S4_ROWS + pid) * 32,
            tag,
            pid,
            pid,
            Diag,
            Metrics,
            Wo,
            Wr,
            8192 * 2048 // 2,
            4608 * 8192 // 2,
            4,
            4,
            GRID,
            PREFETCH,
            WaitTrace,
        )
        if ok:
            if TASK_TIMES:
                stamp(PT4, 768 + pid, 0)
            _finish_row(
                pid,
                F4 + 2048 * 32,
                Bias,
                RouteNorm,
                Scores,
                Ids,
                Pool,
                SCOPED_IO=True,
                POOL=False,
            )
            if TASK_TIMES:
                stamp(PT4, 768 + pid, 2)
            if flow.arrive(Counters + 13 * 32, tag, 8):
                scoped.publish(Flags + flow.SORT * 32, tag)
    if ok:
        if TAIL_MODE == 3:
            ok = rows.pool(
                pid,
                tag,
                Row4,
                F4 + 2048 * 32,
                RouteNorm,
                Pool,
                Counters + 15 * 32,
                Flags + flow.S5_INPUT * 32,
                Diag,
                Metrics,
                GRID,
                Trace=(RowTrace + 1 * GRID * 32) if RowTrace is not None else None,
                Record=PT4 + 1024 * 32 if PT4 is not None else None,
            )
        else:
            ok = producers.apply(
                pid,
                tag,
                Row4,
                None,
                RouteNorm,
                None,
                None,
                Pool,
                None,
                Diag,
                Metrics,
                4096,
                TAIL_MODE,
                True,
                4,
                Row4 + producers.plan(4096)[11],
                Flags + flow.S5_INPUT * 32,
                GRID,
                Trace=(RowTrace + 1 * GRID * 32) if RowTrace is not None else None,
                Record=PT4 + 1024 * 32 if PT4 is not None else None,
                Join=Counters + 15 * 32,
            )
    if ok and pid == 0:
        ok = flow.wait(
            Flags + flow.SORT * 32,
            tag,
            pid,
            0,
            Diag,
            Metrics,
            Wo,
            Wr,
            8192 * 2048 // 2,
            4608 * 8192 // 2,
            4,
            4,
            GRID,
            PREFETCH,
            WaitTrace,
        )
        if ok:
            i = gl.arange(0, 1024, layout=gl.BlockedLayout([1], [32], [4], [0]))
            scoped.store_words(Sorted + i, gl.full_like(i, 64), i >= 0)
            scoped.store_words(Experts + gl.minimum(i, 63), gl.full_like(i, -1), i < 64)
            table(Ids, Sorted, Experts, NumPost, RANK * 64)
            if flow.arrive(Counters + 15 * 32, tag, 2):
                scoped.publish_flags(Flags + flow.S5_INPUT * 32, tag, GRID)
    gl.store(Status + GRID * 3 + pid, ok)
    stamp(Trace, pid, 7)
    stamp(Trace, pid, 8)
    delayed_point(Seed, pid, 2)
    if ok:
        ok = flow.wait(
            Flags + (flow.S5_INPUT + pid) * 32,
            tag,
            pid,
            pid,
            Diag,
            Metrics,
            Wo,
            Wr,
            8192 * 2048 // 2,
            4608 * 8192 // 2,
            5,
            4,
            GRID,
            PREFETCH,
            WaitTrace,
        )
        if ok:
            ok = common_experts.stage(
                pid,
                Pool,
                Ids,
                Scores,
                Sorted,
                Experts,
                NumPost,
                W13,
                S13,
                W2,
                S2,
                Gamma5,
                Inter,
                Y,
                Scaled,
                Inv,
                Weighted,
                Part5,
                Reduced5,
                Epochs,
                Bar5,
                Status5,
                Scratch5,
                Peers5,
                Ready5,
                GRID,
                0,
                PAY5,
                FL5,
                RANK,
                SKIP_TASK if SKIP_STAGE == 5 else -1,
                Seed,
                ExpertTrace,
                2,
                1,
                None,
                TaskTrace,
                tag,
                True,
                X6,
                Counters + 14 * 32,
                Flags + flow.S6_INPUT * 32,
            )
            if not ok:
                flow.note(Diag, pid, 5, 5, pid, tag, -1, 0)
    gl.store(Status + GRID * 4 + pid, ok)
    stamp(Trace, pid, 9)
    stamp(Trace, pid, 10)
    if ok:
        for task in range(pid, 512, GRID):
            ok = flow.wait(
                Flags + (flow.S6_INPUT + task) * 32,
                tag,
                pid,
                task,
                Diag,
                Metrics,
                Wu,
                None,
                8192 * 4096 // 2,
                0,
                6,
                5,
                GRID,
                PREFETCH,
                WaitTrace,
            )
            if ok:
                if SKIP_STAGE == 6 and task == SKIP_TASK:
                    ok = gl.full(
                        (), False, gl.int1, gl.BlockedLayout([1], [32], [4], [0])
                    )
                    flow.note(Diag, pid, 6, 6, task, tag, -1, 0)
                else:
                    tile = task // 4
                    if TASK_TIMES:
                        stamp(PT6, task, 0)
                    partial_tile(X6, Wu, P6, task, 8192, 4096, 4, 128, True)
                    if TASK_TIMES:
                        stamp(PT6, task, 1)
                    if flow.arrive(F6 + tile * 32, tag, 4):
                        value = reduce_tile(P6, F6 + 2048 * 32, tile, 8192, 4, True)
                        if TAIL_MODE != 3:
                            producers.produce(
                                value,
                                tile,
                                Row6,
                                tag,
                                MidStream,
                                Post6,
                                Pre6,
                                Alpha6,
                                Beta6,
                                Out,
                                Updated,
                                8192,
                                TAIL_MODE,
                                False,
                                (RowTrace + 2 * GRID * 32)
                                if RowTrace is not None
                                else None,
                            )
                        if TAIL_MODE == 3 and flow.arrive(F6 + 128 * 32, tag, 128):
                            rows.publish_ready(Row6, tag, 8192)
                    if TASK_TIMES:
                        stamp(PT6, task, 2)
    if ok:
        if TAIL_MODE == 3:
            ok = rows.boundary(
                pid,
                tag,
                Row6,
                F6 + 2048 * 32,
                MidStream,
                Post6,
                Pre6,
                Alpha6,
                Beta6,
                Out,
                Updated,
                None,
                None,
                Diag,
                Metrics,
                GRID,
                6,
                Trace=(RowTrace + 2 * GRID * 32) if RowTrace is not None else None,
                Record=PT6 + 1024 * 32 if PT6 is not None else None,
            )
        else:
            ok = producers.apply(
                pid,
                tag,
                Row6,
                MidStream,
                Pre6,
                Alpha6,
                Beta6,
                Out,
                Updated,
                Diag,
                Metrics,
                8192,
                TAIL_MODE,
                False,
                6,
                None,
                None,
                0,
                Trace=(RowTrace + 2 * GRID * 32) if RowTrace is not None else None,
                Record=PT6 + 1024 * 32 if PT6 is not None else None,
            )
    if PREFETCH and ok and pid >= 512:
        # No unpool task assigned: complete one finite stripe of upcoming weights.
        count = gl.full((), 0, gl.int32, gl.BlockedLayout([1], [32], [4], [0]))
        for step in range(gl.cdiv(8192 * 4096 // 2, GRID * 128)):
            count += flow.warm(Wu, 8192 * 4096 // 2, pid, step, GRID)
        gl.store(Metrics + pid * 32 + 6, gl.load(Metrics + pid * 32 + 6) + count)
    gl.store(Status + GRID * 5 + pid, ok)
    stamp(Trace, pid, 11)


class Workspace(msgspec.Struct):
    s1: object
    s2: object
    s3: object
    s4: object
    s5: object
    s6: object
    buffers: dict
    epochs: torch.Tensor
    counters: torch.Tensor
    flags: torch.Tensor
    status: torch.Tensor
    diagnostics: torch.Tensor
    metrics: torch.Tensor
    row3: torch.Tensor
    row4: torch.Tensor
    row6: torch.Tensor
    shape: CommonShape
    context: int
    left: int
    variant: int
    admitted: dict = msgspec.field(default_factory=dict)


def make_workspace(device, *, context, left, shape=None, variant=0):
    assert variant in (0, 1, 2, 3)
    shape = CommonShape() if shape is None else shape
    grid = shape.grid(device)
    assert grid >= 768 and grid % 12 == 0

    def empty(size, dtype=torch.bfloat16):
        return torch.empty(size, device=device, dtype=dtype)

    def zero(size, dtype=torch.int32):
        return torch.zeros(size, device=device, dtype=dtype)

    s1 = common_qkvg.make_workspace(device, shape=shape)
    s2 = common_attention.make_workspace(
        device, context=context, window_left=left, shape=shape
    )
    s3 = common_attn_tail.make_workspace(device, shape=shape)
    s3.peers = standin.make_standins(
        device, width=8192, ctas=1024 if variant == 3 else (128 if variant else grid)
    )
    s3.peers.ready = torch.empty((1,), device=device, dtype=torch.int32)
    s4 = common_route.make_workspace(device, shape=shape)
    s5 = common_experts.make_workspace(device, shape=shape)
    s6 = common_tail.make_workspace(device, shape=shape)
    b = {
        n: empty(size, dtype)
        for n, size, dtype in [
            ("q", (8, 2048), torch.bfloat16),
            ("k", (8, 256), torch.bfloat16),
            ("v", (8, 256), torch.bfloat16),
            ("gate", (8, 2048), torch.bfloat16),
            ("attn", (8, 2048), torch.bfloat16),
            ("x4", (8, 8192), torch.bfloat16),
            ("mid_stream", (8, 8192), torch.float32),
            ("x6", (8, 4096), torch.bfloat16),
            ("out", (8, 8192), torch.bfloat16),
            ("stream", (8, 8192), torch.float32),
        ]
    }
    for name, n in (("f1", 4608), ("f3", 8192), ("f4", 4608), ("f6", 8192)):
        b[name] = zero(2048 * 32 + 8 * n)
    return Workspace(
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        b,
        zero((grid, 32)),
        zero((16, 32)),
        zero((flow.FLAG_LINES, 32)),
        empty((7, grid), torch.int32),
        zero((grid, 16), torch.int64),
        zero((grid, 32), torch.int64),
        (producers.make_state if variant in (1, 2) else rows.make_state)(device, 8192),
        (producers.make_state if variant in (1, 2) else rows.make_state)(device, 4096),
        (producers.make_state if variant in (1, 2) else rows.make_state)(device, 8192),
        shape,
        context,
        left,
        variant,
    )


def launch(
    x,
    stream,
    weights,
    metadata,
    ws,
    *,
    rank,
    prefetch=True,
    skip_stage=-1,
    skip_task=0,
    seed=None,
    trace=None,
    wait_trace=None,
    expert_trace=None,
    task_trace=None,
    row_trace=None,
    profile_tasks=None,
    task_times=True,
    code_warm=None,
    compile_only=False,
):
    b = ws.buffers
    s1, s2, s3, s4, s5, s6 = ws.s1, ws.s2, ws.s3, ws.s4, ws.s5, ws.s6
    # F1/F3/F4/F6 contain counters followed by full FP32 projections.
    D = (
        s1.partial,
        ws.buffers["f1"],
        b["q"],
        b["k"],
        b["v"],
        b["gate"],
        s2.partial,
        s2.lse,
        b["attn"],
        s3.k_partial,
        b["f3"],
        s3.partial,
        s3.peers.scratch,
        s3.peers.peers,
        s3.peers.reduced,
        b["x4"],
        b["mid_stream"],
        s4.partial,
        b["f4"],
        s4.ids,
        s4.scores,
        s4.pooled,
        s4.sorted_ids,
        s4.expert_ids,
        s4.num_post,
        s5.inter,
        s5.output,
        s5.scaled,
        s5.inv,
        s5.weighted,
        s5.partial,
        s5.peers.scratch,
        s5.peers.peers,
        s5.peers.reduced,
        b["x6"],
        s6.partial,
        b["f6"],
        b["out"],
        b["stream"],
        s5.state.barriers,
        s5.state.status,
        s5.peers.ready,
    )
    W = tuple(weights[k] for k in WEIGHT_KEYS)
    args = (
        x,
        stream,
        W,
        metadata,
        D,
        ws.epochs,
        ws.counters,
        ws.flags,
        ws.status,
        ws.diagnostics,
        ws.metrics,
        ws.row3,
        ws.row4,
        ws.row6,
    )
    opts = dict(
        GRID=ws.shape.grid(x.device),
        RANK=rank,
        LEFT=ws.left,
        SPLITS=s2.splits,
        MAX_PAGES=s2.pages_per_chunk,
        PAY3=s3.peers.payload,
        FL3=s3.peers.flags,
        PAY5=s5.peers.payload,
        FL5=s5.peers.flags,
        PREFETCH=prefetch,
        SKIP_STAGE=skip_stage,
        SKIP_TASK=skip_task,
        Seed=seed,
        Trace=trace,
        WaitTrace=wait_trace,
        ExpertTrace=expert_trace,
        TaskTrace=task_trace,
        RowTrace=row_trace,
        ProfileTasks=profile_tasks,
        **ws.shape.options(),
    )
    entry = _launch
    if ws.variant == 0:
        entry = layer_accepted._launch
        opts["CodeWarm"] = code_warm
        args = args[:-3]
        opts.pop("RowTrace")
        opts.pop("ProfileTasks")
    else:
        opts["TAIL_MODE"] = ws.variant
        opts["TASK_TIMES"] = task_times
    if compile_only:
        return entry.warmup(*args, **opts, grid=(opts["GRID"],))
    key = (
        rank,
        prefetch,
        skip_stage,
        skip_task,
        seed is not None,
        trace is not None,
        wait_trace is not None,
        expert_trace is not None,
        task_trace is not None,
        row_trace is not None,
        profile_tasks is not None,
        task_times,
        code_warm is not None,
        weights["gamma5"].dtype,
    )
    if key not in ws.admitted:
        k = entry.warmup(*args, **opts, grid=(opts["GRID"],))
        r = ws.shape.admit("layer", k, grid=opts["GRID"])
        ws.admitted[key] = dict(
            vgprs=k.n_regs,
            lds=k.metadata.shared,
            spills=k.n_spills,
            occupancy=r.active_ctas_per_cu,
        )
    entry[(opts["GRID"],)](*args, **opts)
    return b["out"], b["stream"]


def prepare_peers(ws, partial3, partial5, *, rank):
    standin.prepare(ws.s3.peers, partial3, ws.epochs, rank=rank)
    standin.prepare(ws.s5.peers, partial5, ws.epochs, rank=rank)


def diagnostic_records(ws):
    """Decode bounded failure records after the launch has completed."""
    records = []
    for consumer, row in enumerate(ws.diagnostics.cpu().tolist()):
        if row[0]:
            stage, producer, task, tag, seen, polls, clock = row[:7]
            records.append(
                dict(
                    consumer=consumer,
                    stage=flow.STAGES[stage - 1],
                    producer=flow.STAGES[producer - 1],
                    task=task,
                    tag=tag,
                    seen=seen,
                    polls=polls,
                    clock=clock,
                )
            )
    return records
