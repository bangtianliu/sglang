"""Accepted phase4.2 device kernel, retained as the single-rank default."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    code_probe,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    layer_flow as flow,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    scoped,
    standin,
    vector_memory,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.packed_memory import (
    store_bf16_tile,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.timeline import (
    stamp,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_attention,
    common_experts,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.attention import (
    _merge,
    _store128,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.boundary import (
    delayed_point,
    norm_residual,
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
    CodeWarm=None,
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
    if CodeWarm is not None:
        if (pid >= 576) & (pid < 588):
            code_probe.warm(CodeWarm, pid - 576)
    stamp(Trace, pid, 0)
    delayed_point(Seed, pid, 0)
    # Each Q/K/V/gate head contributes once to its KV group's18-head dependency.
    for task in range(pid, 576, GRID):
        if SKIP_STAGE == 1 and task == SKIP_TASK:
            ok = gl.full((), False, gl.int1, gl.BlockedLayout([1], [32], [4], [0]))
            flow.note(Diag, pid, 1, 1, task, tag, -1, 0)
        else:
            head = task // 8
            partial_tile(X, Wq, P1, task, 4608, 8192, 8, 64)
            if flow.arrive(F1 + head * 32, tag, 8):
                value = reduce_tile(P1, F1 + 2048 * 32, head, 4608, 8, True)
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
                    value,
                    True,
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
                if flow.arrive(Counters + group * 32, tag, 18):
                    scoped.publish_flags(
                        Flags + (flow.S2_INPUT + group * 128) * 32, tag, SPLITS
                    )
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
                    partial_tile(AOut, Wo, P3, task, 8192, 2048, 4, 128, True, Gate=G)
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
                        if flow.arrive(F3 + 128 * 32, tag, 128):
                            scoped.publish_flags(Flags + flow.S3_ROWS * 32, tag, 8)
    if ok and pid < 8:
        ok = flow.wait(
            Flags + (flow.S3_ROWS + pid) * 32,
            tag,
            pid,
            pid,
            Diag,
            Metrics,
            Wo,
            Wr,
            8192 * 2048 // 2,
            4608 * 8192 // 2,
            3,
            3,
            GRID,
            PREFETCH,
            WaitTrace,
        )
        if ok:
            value, ok = standin.reduce_row(
                Part3, Scratch3, Peers3, Reduced3, tag, pid, 8192, GRID, RANK, PAY3, FL3
            )
            if ok:
                norm_residual(
                    pid,
                    Reduced3 + RANK * 8 * 8192,
                    Stream,
                    None,
                    Pre3,
                    Alpha3,
                    Beta3,
                    X4,
                    MidStream,
                    False,
                    False,
                    True,
                    Value=value,
                    SCOPED_IO=True,
                )
                if flow.arrive(Counters + 12 * 32, tag, 8):
                    scoped.publish_flags(Flags + flow.S4_INPUT * 32, tag, 576)
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
                    partial_tile(X4, Wr, P4, task, 4608, 8192, 8, 64, True)
                    if flow.arrive(F4 + tile * 32, tag, 8):
                        reduce_tile(P4, F4 + 2048 * 32, tile, 4608, 8, False)
                        if flow.arrive(F4 + 72 * 32, tag, 72):
                            scoped.publish_flags(Flags + flow.S4_ROWS * 32, tag, 8)
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
            _finish_row(
                pid, F4 + 2048 * 32, Bias, RouteNorm, Scores, Ids, Pool, SCOPED_IO=True
            )
            if flow.arrive(Counters + 13 * 32, tag, 8):
                scoped.publish(Flags + flow.SORT * 32, tag)
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
                    partial_tile(X6, Wu, P6, task, 8192, 4096, 4, 128, True)
                    if flow.arrive(F6 + tile * 32, tag, 4):
                        reduce_tile(P6, F6 + 2048 * 32, tile, 8192, 4, True)
                        if flow.arrive(F6 + 128 * 32, tag, 128):
                            scoped.publish_flags(Flags + flow.S6_ROWS * 32, tag, 8)
    if ok and pid < 8:
        ok = flow.wait(
            Flags + (flow.S6_ROWS + pid) * 32,
            tag,
            pid,
            pid,
            Diag,
            Metrics,
            Wu,
            None,
            8192 * 4096 // 2,
            0,
            6,
            6,
            GRID,
            PREFETCH,
            WaitTrace,
        )
        if ok:
            norm_residual(
                pid,
                F6 + 2048 * 32,
                MidStream,
                Post6,
                Pre6,
                Alpha6,
                Beta6,
                Out,
                Updated,
                True,
                False,
                True,
                SCOPED_IO=True,
            )
    if PREFETCH and ok and pid >= 512:
        # No unpool task assigned: complete one finite stripe of upcoming weights.
        count = gl.full((), 0, gl.int32, gl.BlockedLayout([1], [32], [4], [0]))
        for step in range(gl.cdiv(8192 * 4096 // 2, GRID * 128)):
            count += flow.warm(Wu, 8192 * 4096 // 2, pid, step, GRID)
        gl.store(Metrics + pid * 32 + 6, gl.load(Metrics + pid * 32 + 6) + count)
    gl.store(Status + GRID * 5 + pid, ok)
    stamp(Trace, pid, 11)
