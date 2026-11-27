"""Measurement-only adjacent common-body fusions, preserving row layouts."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    code_probe,
    scoped,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    boundary as borig,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_attention as a2,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_attn_tail as a3,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_experts as a5,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_qkvg as a1,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_route as a4,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_tail as a6,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import qkvg as qorig
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    streamed_gemm as gemm,
)


@g.jit
def agent_store(Base, offset, Value, Mask):
    valid = gl.full_like(offset, True).to(gl.int1) & Mask
    gl.amd.gfx1250.buffer_store(
        Value.to(Base.dtype.element_ty), Base, offset.to(gl.int32), valid, cache=".cg"
    )


@g.jit
def decision_enabled(Decision, STAGE: gl.constexpr):
    failed = scoped.load_flag(Decision + gl.program_id(0) * 32)
    return (failed == 0) | (failed == STAGE)


@g.jit
def boundary_barrier(Storage, tag, Failure, Decision, GRID: gl.constexpr):
    pid = gl.program_id(0)
    scoped.drain_stores()
    ticket = gl.atomic_add(Storage + pid // 32 * 32, 1, sem="relaxed", scope="gpu")
    if ticket + 1 == tag * 32:
        top = gl.atomic_add(Storage + GRID // 32 * 32, 1, sem="relaxed", scope="gpu")
        if top + 1 == tag * (GRID // 32):
            failed = scoped.load_flag(Failure)
            i = gl.arange(0, 1024, layout=gl.BlockedLayout([1], [32], [4], [0]))
            scoped.store_words(Decision + i * 32, gl.full_like(i, failed), i < GRID)
            scoped.publish_flags(Storage + (GRID // 32 + 1) * 32, tag, GRID)
    seen, polls = scoped.wait_flag(
        Storage + (GRID // 32 + 1 + pid) * 32,
        tag,
        100000000,
        MAX_SPINS=1048576,
        SHADER_CLOCK=True,
    )
    return (seen == tag, polls)


@g.jit
def head_agent(
    P,
    head,
    POS,
    COS,
    LOC,
    MAP,
    KC,
    VC,
    Q,
    K,
    V,
    G,
    KS,
    VS,
    EPS: gl.constexpr,
    SCALE: gl.constexpr,
    ROPE: gl.constexpr,
    HAS_MAP: gl.constexpr,
    Value=None,
    SCOPED_IO: gl.constexpr = False,
    RowStart=0,
    PAD_ROWS: gl.constexpr = 16,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 2], [1, 32], [gl.num_warps(), 1], [1, 0]
    )
    row = gl.arange(0, PAD_ROWS, layout=gl.SliceLayout(1, layout)) + RowStart
    d = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
    if Value is None:
        value = qorig.scoped.load_words_masked(
            P + row[:, None] * 4608 + head * 64 + d[None, :], row[:, None] < 8
        ).to(gl.float32, bitcast=True)
    else:
        small = gl.convert_layout(Value, layout)
        index = gl.minimum(row[:, None], 7) + gl.full(
            (PAD_ROWS, 64), 0, gl.int32, layout
        )
        value = gl.gather(small, index, 0)
    if head < 32:
        q_value = qorig._norm_rope(
            value, POS, COS, EPS, SCALE, ROPE, True, RowStart, PAD_ROWS
        )
        if SCOPED_IO:
            qorig.store_bf16_tile(
                Q + row[:, None] * 2048 + head * 64 + d[None, :],
                q_value,
                row[:, None] < 8,
                PAD_ROWS,
                64,
            )
        else:
            agent_store(
                Q,
                0 + row[:, None] * 2048 + head * 64 + d[None, :],
                q_value,
                row[:, None] < 8,
            )
    elif head < 40:
        if head < 36:
            kh = head - 32
            kv_value = qorig._norm_rope(
                value, POS, COS, EPS, 1.0, ROPE, False, RowStart, PAD_ROWS
            )
            if SCOPED_IO:
                qorig.store_bf16_tile(
                    K + row[:, None] * 256 + kh * 64 + d[None, :],
                    kv_value,
                    row[:, None] < 8,
                    PAD_ROWS,
                    64,
                )
            else:
                agent_store(
                    K,
                    0 + row[:, None] * 256 + kh * 64 + d[None, :],
                    kv_value,
                    row[:, None] < 8,
                )
            cache = KC
            scale = gl.load(KS)
        else:
            kh = head - 36
            kv_value = value.to(gl.bfloat16)
            if SCOPED_IO:
                qorig.store_bf16_tile(
                    V + row[:, None] * 256 + kh * 64 + d[None, :],
                    kv_value,
                    row[:, None] < 8,
                    PAD_ROWS,
                    64,
                )
            else:
                agent_store(
                    V,
                    0 + row[:, None] * 256 + kh * 64 + d[None, :],
                    kv_value,
                    row[:, None] < 8,
                )
            cache = VC
            scale = gl.load(VS)
        slot = gl.load(LOC + row, row < 8, other=-1).to(gl.int64)
        if HAS_MAP:
            slot = gl.load(MAP + slot, (row < 8) & (slot >= 0), other=-1).to(gl.int64)
        if SCOPED_IO:
            qorig.store_fp8_tile(
                cache + slot[:, None] * 256 + kh * 64 + d[None, :],
                (kv_value.to(gl.float32) / scale).to(cache.dtype.element_ty),
                (row[:, None] < 8) & (slot[:, None] >= 0),
                PAD_ROWS * 64,
            )
        else:
            agent_store(
                cache,
                0 + slot[:, None] * 256 + kh * 64 + d[None, :],
                kv_value.to(gl.float32) / scale,
                (row[:, None] < 8) & (slot[:, None] >= 0),
            )
    elif SCOPED_IO:
        qorig.store_bf16_tile(
            G + row[:, None] * 2048 + (head - 40) * 64 + d[None, :],
            value.to(gl.bfloat16),
            row[:, None] < 8,
            PAD_ROWS,
            64,
        )
    else:
        agent_store(
            G,
            0 + row[:, None] * 2048 + (head - 40) * 64 + d[None, :],
            value.to(gl.bfloat16),
            row[:, None] < 8,
        )


@g.jit
def boundary_agent(
    row,
    Branch,
    Stream,
    Post,
    Pre,
    Alpha,
    Beta,
    Out,
    NewStream,
    POST_SCALE: gl.constexpr,
    PRE_BF16: gl.constexpr,
    WORDS: gl.constexpr,
    Trace=None,
    Value=None,
    SCOPED_IO: gl.constexpr = False,
    PACKED_ROW: gl.constexpr = False,
):
    """Exact legacy default; packed four-wave experiment requires explicit opt-in."""
    if PACKED_ROW and SCOPED_IO and (gl.num_warps() == 4):
        borig._norm_residual_scoped(
            row,
            Branch,
            Stream,
            Post,
            Pre,
            Alpha,
            Beta,
            Out,
            NewStream,
            POST_SCALE,
            PRE_BF16,
            WORDS,
            Trace,
            Value,
        )
    else:
        i = gl.arange(
            0, 8192, layout=gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
        )
        if Value is not None:
            branch = gl.convert_layout(Value, i.type.layout)
        elif WORDS:
            branch = borig.load_tile_words(Branch + row * 8192 + i, i >= 0, 8192).to(
                gl.float32, bitcast=True
            )
        else:
            branch = borig.scoped.load_words_masked(
                (Branch + row * 8192 + i).to(gl.pointer_type(gl.int32)), i >= 0
            ).to(gl.float32, bitcast=True)
        branch = branch.to(gl.bfloat16).to(gl.float32)
        if Trace is not None:
            gl.store(Trace + 3, borig.sync.clock())
        if POST_SCALE:
            branch = (
                (branch * gl.load(Post + i).to(gl.float32))
                .to(gl.bfloat16)
                .to(gl.float32)
            )
        if Trace is not None:
            gl.store(Trace + 4, borig.sync.clock())
        branch *= gl.rsqrt(gl.sum(branch * branch, 0) / 8192 + 1e-08)
        branch = branch.to(gl.bfloat16).to(gl.float32)
        if Trace is not None:
            gl.store(Trace + 5, borig.sync.clock())
        stream = gl.load(
            Stream + row * 8192 + i, cache_modifier=".cg" if SCOPED_IO else ""
        )
        updated = gl.load(Alpha + i) * stream + gl.load(Beta + i) * branch
        if SCOPED_IO:
            borig.store_tile_words(
                NewStream.to(gl.pointer_type(gl.int32)) + row * 8192 + i,
                updated.to(gl.int32, bitcast=True),
                i >= 0,
                8192,
            )
        else:
            agent_store(NewStream, 0 + row * 8192 + i, updated, True)
        if Trace is not None:
            gl.store(Trace + 6, borig.sync.clock())
        pre = updated.to(gl.bfloat16).to(gl.float32) if PRE_BF16 else updated
        pre *= gl.rsqrt(gl.sum(pre * pre, 0) / 8192 + 1e-05)
        if Pre is not None:
            pre *= gl.load(Pre + i).to(gl.float32)
        if Trace is not None:
            gl.store(Trace + 7, borig.sync.clock())
        if SCOPED_IO:
            borig.store_bf16_vector(Out + row * 8192, pre, 8192)
        else:
            agent_store(Out, 0 + row * 8192 + i, pre.to(gl.bfloat16), True)
        if Trace is not None:
            gl.store(Trace + 8, borig.sync.clock())


@g.jit
def project_agent(
    pid,
    X,
    W,
    Partials,
    Projection,
    Counters,
    Flags,
    tag,
    N: gl.constexpr,
    K: gl.constexpr,
    SK: gl.constexpr,
    BK: gl.constexpr,
    GRID: gl.constexpr,
    ROUND: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Packed=None,
    Trace=None,
    Gate=None,
):
    """Last split merges in fixed order; last N tile releases eight private row owners."""
    gemm.stamp(Trace, pid, 8)
    if pid != SKIP:
        for task in range(pid, N // 64 * SK, GRID):
            tile = task // SK
            gemm.partial_tile(X, W, Partials, task, N, K, SK, BK, True, Trace, Gate)
            gemm.stamp(Trace, pid, 9)
            gemm.scoped.drain_stores()
            ticket = gl.atomic_add(Counters + tile * 32, 1, sem="relaxed", scope="gpu")
            gemm.stamp(Trace, pid, 10)
            if ticket + 1 == tag * SK:
                value = gemm.reduce_tile(Partials, Projection, tile, N, SK, ROUND)
                if Packed is not None:
                    l: gl.constexpr = value.type.layout
                    r = gl.arange(0, 8, layout=gl.SliceLayout(1, l))
                    c = gl.arange(0, 64, layout=gl.SliceLayout(0, l)) + tile * 64
                    gemm.store_bf16_tile(
                        Packed + r[:, None] * N + c[None, :],
                        value,
                        r[:, None] >= 0,
                        8,
                        64,
                    )
                gemm.stamp(Trace, pid, 11)
                gemm.scoped.drain_stores()
                merged = gl.atomic_add(
                    Counters + N // 64 * 32, 1, sem="relaxed", scope="gpu"
                )
                if merged + 1 == tag * (N // 64):
                    gemm.scoped.publish_flags(Flags, tag, 8)
    ok = pid != SKIP
    if pid < 8:
        seen, _ = gemm.scoped.wait_flag(
            Flags + pid * 32, tag, 10000000, MAX_SPINS=1048576
        )
        ok = ok & (seen == tag)
    gemm.stamp(Trace, pid, 12)
    return ok


@g.jit
def stage1(
    pid,
    X,
    W,
    P,
    State,
    Partial,
    Barriers,
    Status,
    POS,
    COS,
    LOC,
    MAP,
    KC,
    VC,
    Q,
    K,
    V,
    G,
    KS,
    VS,
    GRID: gl.constexpr,
    STRIDE: gl.constexpr,
    EPS: gl.constexpr,
    SCALE: gl.constexpr,
    ROPE: gl.constexpr,
    HAS_MAP: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
    Trace=None,
):
    a1.stamp(Trace, pid, 0)
    tag = a1._advance_tag(State, pid)
    a1.delayed_point(Seed, pid, 0)
    ok = pid != SKIP
    if pid != SKIP:
        for task in range(pid, 72 * 8, GRID):
            tile = task // 8
            a1.partial_tile(X, W, Partial, task, 4608, 8192, 8, 64, Trace=Trace)
            a1.stamp(Trace, pid, 1)
            a1.delayed_point(Seed, pid, 1)
            a1.scoped.drain_stores()
            ticket = gl.atomic_add(Barriers + tile * 32, 1, sem="relaxed", scope="gpu")
            a1.stamp(Trace, pid, 2)
            if ticket + 1 == tag * 8:
                value = a1.reduce_tile(Partial, P, tile, 4608, 8, True)
                a1.stamp(Trace, pid, 3)
                a1.delayed_point(Seed, pid, 2)
                head_agent(
                    P,
                    tile,
                    POS,
                    COS,
                    LOC,
                    MAP,
                    KC,
                    VC,
                    Q,
                    K,
                    V,
                    G,
                    KS,
                    VS,
                    EPS,
                    SCALE,
                    ROPE,
                    HAS_MAP,
                    value,
                )
                gl.store(State + GRID * 32 + tile, tag)
    a1.stamp(Trace, pid, 31)
    gl.store(Status + pid, ok)


@g.jit
def stage2(
    pid,
    Q,
    K,
    V,
    Pages,
    Map,
    Length,
    KS,
    VS,
    P,
    LSE,
    Epochs,
    Barriers,
    Status,
    CtaStatus,
    Out,
    GRID: gl.constexpr,
    SPLITS: gl.constexpr,
    MAX_PAGES: gl.constexpr,
    SCALE: gl.constexpr,
    LEFT: gl.constexpr,
    HAS_MAP: gl.constexpr,
    MAP_PAGE: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
):
    tag = a2._advance_tag(Epochs, pid)
    if pid < 256:
        gl.store(Status + pid, 0)
    a2.delayed_point(Seed, pid, 0)
    if pid != SKIP:
        gl.static_assert(GRID >= SPLITS * 4)
        if pid < SPLITS * 4:
            task = pid
            split = task // 4
            head = task % 4
            value, lse = a2.partial(
                Q,
                K,
                V,
                Pages,
                Map,
                Length,
                KS,
                VS,
                head,
                split,
                SCALE,
                LEFT,
                SPLITS,
                HAS_MAP,
                MAP_PAGE,
                MAX_PAGES,
                SCOPED_IO=True,
            )
            ml: gl.constexpr = value.type.layout
            row = gl.arange(0, 64, layout=gl.SliceLayout(1, ml))
            col = gl.arange(0, 64, layout=gl.SliceLayout(0, ml))
            a, b, c, d = a2.vector_memory.split4(
                value.reshape((4096,)).to(gl.int32, bitcast=True), 1024
            )
            ids = gl.arange(0, 1024, layout=gl.BlockedLayout([1], [32], [4], [0]))
            a2._store128(
                P + ((head * 64 + ids // 16) * SPLITS + split) * 64 + ids % 16 * 4,
                a,
                b,
                c,
                d,
                1024,
            )
            a2.scoped.store_words(
                LSE + (head * 64 + row) * SPLITS + split,
                lse.to(gl.int32, bitcast=True),
                row >= 0,
            )
    a2.delayed_point(Seed, pid, 1)
    ok = gl.full((), False, gl.int1, gl.BlockedLayout([1], [32], [4], [0]))
    if pid != SKIP:
        ok, _, _ = a2.scoped.counter_barrier(
            Barriers, tag, 10000000, GRID, 12, MAX_SPINS=1048576
        )
    a2.delayed_point(Seed, pid, 2)
    if ok and pid < 256:
        a2._merge(
            P,
            LSE,
            Barriers + (GRID // 12 + 1) * 32,
            Status,
            Out,
            pid,
            tag,
            SPLITS,
            None,
            pid,
        )
    gl.store(CtaStatus + pid, ok)


@g.jit
def stage3(
    pid,
    X,
    Gate,
    W,
    Stream,
    Pre,
    Alpha,
    Beta,
    Normalized,
    KPart,
    P,
    Partial,
    Epochs,
    Barriers,
    Status,
    Scratch,
    Peers,
    Ready,
    Reduced,
    Out,
    Updated,
    GRID: gl.constexpr,
    STRIDE: gl.constexpr,
    RANK: gl.constexpr,
    PAYLOAD: gl.constexpr,
    FLAGS: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
    Trace=None,
):
    a3.stamp(Trace, pid, 0)
    tag = a3._advance_tag(Epochs, pid)
    a3.delayed_point(Seed, pid, 0)
    ok = a3.project_last(
        pid,
        X,
        W,
        KPart,
        P,
        Barriers,
        Barriers + STRIDE,
        tag,
        8192,
        2048,
        4,
        128,
        GRID,
        True,
        SKIP,
        Partial + RANK * 8 * 8192,
        Trace,
        Gate,
    )
    a3.stamp(Trace, pid, 3)
    a3.delayed_point(Seed, pid, 1)
    if ok and pid < 8:
        value, ok = a3.standin.reduce_row(
            Partial,
            Scratch,
            Peers,
            Reduced,
            tag,
            pid,
            8192,
            GRID,
            RANK,
            PAYLOAD,
            FLAGS,
            Trace,
        )
        a3.stamp(Trace, pid, 4)
        a3.delayed_point(Seed, pid, 2)
        if ok:
            boundary_agent(
                pid,
                Reduced + RANK * 8 * 8192,
                Stream,
                None,
                Pre,
                Alpha,
                Beta,
                Out,
                Updated,
                False,
                False,
                True,
                Value=value,
            )
    a3.stamp(Trace, pid, 31)
    gl.store(Status + pid, ok)


@g.jit
def stage4(
    pid,
    X,
    W,
    Bias,
    Norm,
    KPart,
    P,
    Epochs,
    Barriers,
    Status,
    Scores,
    Ids,
    Y,
    Sorted,
    Experts,
    NumPost,
    GRID: gl.constexpr,
    STRIDE: gl.constexpr,
    LOCAL_START: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
):
    tag = a4._advance_tag(Epochs, pid)
    a4.delayed_point(Seed, pid, 0)
    if pid == 0:
        i = gl.arange(0, 1024, layout=gl.BlockedLayout([1], [32], [4], [0]))
        a4.scoped.store_words(Sorted + i, gl.full_like(i, 64), i >= 0)
        a4.scoped.store_words(Experts + gl.minimum(i, 63), gl.full_like(i, -1), i < 64)
    ok = project_agent(
        pid,
        X,
        W,
        KPart,
        P,
        Barriers,
        Barriers + STRIDE,
        tag,
        4608,
        8192,
        8,
        64,
        GRID,
        False,
        SKIP,
    )
    a4.delayed_point(Seed, pid, 1)
    if ok and pid < 8:
        a4._finish_row(pid, P, Bias, Norm, Scores, Ids, Y)
        a4.scoped.drain_stores()
        ticket = gl.atomic_add(Barriers + 2 * STRIDE, 1, sem="relaxed", scope="gpu")
        if ticket + 1 == tag * 8:
            a4.scoped.publish(Barriers + 2 * STRIDE + 32, tag)
    if ok and pid == 0:
        seen, _ = a4.scoped.wait_flag(
            Barriers + 2 * STRIDE + 32, tag, 10000000, MAX_SPINS=1048576
        )
        ok = seen == tag
    a4.delayed_point(Seed, pid, 2)
    if ok and pid == 0:
        a4.table(Ids, Sorted, Experts, NumPost, LOCAL_START)
    gl.store(Status + pid, ok)


@g.jit
def stage6(
    pid,
    X,
    W,
    Stream,
    Post,
    Pre,
    Alpha,
    Beta,
    KPart,
    P,
    Epochs,
    Barriers,
    Status,
    Out,
    Updated,
    GRID: gl.constexpr,
    STRIDE: gl.constexpr,
    PRE_BF16: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
    Trace=None,
):
    a6.stamp(Trace, pid, 0)
    tag = a6._advance_tag(Epochs, pid)
    a6.delayed_point(Seed, pid, 0)
    ok = project_agent(
        pid,
        X,
        W,
        KPart,
        P,
        Barriers,
        Barriers + STRIDE,
        tag,
        8192,
        4096,
        4,
        128,
        GRID,
        True,
        SKIP,
        Trace=Trace,
    )
    a6.stamp(Trace, pid, 1)
    a6.delayed_point(Seed, pid, 1)
    if ok and pid < 8:
        a6.norm_residual(
            pid, P, Stream, Post, Pre, Alpha, Beta, Out, Updated, True, PRE_BF16, True
        )
    a6.stamp(Trace, pid, 31)
    gl.store(Status + pid, ok)


@g.jit
def root1(
    X,
    W,
    P,
    State,
    Partial,
    Barriers,
    Status,
    POS,
    COS,
    LOC,
    MAP,
    KC,
    VC,
    Q,
    K,
    V,
    G,
    KS,
    VS,
    GRID: gl.constexpr,
    STRIDE: gl.constexpr,
    EPS: gl.constexpr,
    SCALE: gl.constexpr,
    ROPE: gl.constexpr,
    HAS_MAP: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
    Trace=None,
    Failure=None,
):
    pid = gl.program_id(0)
    if a1.chain_guard.enabled(Failure, 1):
        stage1(
            gl.program_id(0),
            X,
            W,
            P,
            State,
            Partial,
            Barriers,
            Status,
            POS,
            COS,
            LOC,
            MAP,
            KC,
            VC,
            Q,
            K,
            V,
            G,
            KS,
            VS,
            GRID,
            STRIDE,
            EPS,
            SCALE,
            ROPE,
            HAS_MAP,
            SKIP,
            Seed,
            Trace,
        )
        a1.chain_guard.finish(Failure, Status, pid, 1)
    else:
        gl.store(Status + pid, 0)


@g.jit
def root2(
    Q,
    K,
    V,
    Pages,
    Map,
    Length,
    KS,
    VS,
    P,
    LSE,
    Epochs,
    Barriers,
    Status,
    CtaStatus,
    Out,
    GRID: gl.constexpr,
    SPLITS: gl.constexpr,
    MAX_PAGES: gl.constexpr,
    SCALE: gl.constexpr,
    LEFT: gl.constexpr,
    HAS_MAP: gl.constexpr,
    MAP_PAGE: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
    Failure=None,
    Decision=None,
):
    pid = gl.program_id(0)
    if decision_enabled(Decision, 2):
        stage2(
            gl.program_id(0),
            Q,
            K,
            V,
            Pages,
            Map,
            Length,
            KS,
            VS,
            P,
            LSE,
            Epochs,
            Barriers,
            Status,
            CtaStatus,
            Out,
            GRID,
            SPLITS,
            MAX_PAGES,
            SCALE,
            LEFT,
            HAS_MAP,
            MAP_PAGE,
            SKIP,
            Seed,
        )
        a2.chain_guard.finish(Failure, CtaStatus, pid, 2)
        if pid < 256:
            a2.chain_guard.finish(Failure, Status, pid, 2)
    else:
        gl.store(CtaStatus + pid, 0)
        if pid < 256:
            gl.store(Status + pid, 0)


@g.jit
def root3(
    X,
    Gate,
    W,
    Stream,
    Pre,
    Alpha,
    Beta,
    Normalized,
    KPart,
    P,
    Partial,
    Epochs,
    Barriers,
    Status,
    Scratch,
    Peers,
    Ready,
    Reduced,
    Out,
    Updated,
    GRID: gl.constexpr,
    STRIDE: gl.constexpr,
    RANK: gl.constexpr,
    PAYLOAD: gl.constexpr,
    FLAGS: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
    Trace=None,
    Failure=None,
):
    pid = gl.program_id(0)
    if a3.chain_guard.enabled(Failure, 3):
        stage3(
            gl.program_id(0),
            X,
            Gate,
            W,
            Stream,
            Pre,
            Alpha,
            Beta,
            Normalized,
            KPart,
            P,
            Partial,
            Epochs,
            Barriers,
            Status,
            Scratch,
            Peers,
            Ready,
            Reduced,
            Out,
            Updated,
            GRID,
            STRIDE,
            RANK,
            PAYLOAD,
            FLAGS,
            SKIP,
            Seed,
            Trace,
        )
        a3.chain_guard.finish(Failure, Status, pid, 3)
    else:
        gl.store(Status + pid, 0)


@g.jit
def root4(
    X,
    W,
    Bias,
    Norm,
    KPart,
    P,
    Epochs,
    Barriers,
    Status,
    Scores,
    Ids,
    Y,
    Sorted,
    Experts,
    NumPost,
    GRID: gl.constexpr,
    STRIDE: gl.constexpr,
    LOCAL_START: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
    Failure=None,
    Decision=None,
):
    pid = gl.program_id(0)
    if decision_enabled(Decision, 4):
        stage4(
            gl.program_id(0),
            X,
            W,
            Bias,
            Norm,
            KPart,
            P,
            Epochs,
            Barriers,
            Status,
            Scores,
            Ids,
            Y,
            Sorted,
            Experts,
            NumPost,
            GRID,
            STRIDE,
            LOCAL_START,
            SKIP,
            Seed,
        )
        a4.chain_guard.finish(Failure, Status, pid, 4)
    else:
        gl.store(Status + pid, 0)


@g.jit
def root5(
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
):
    pid = gl.program_id(0)
    if a5.chain_guard.enabled(Failure, 5):
        a5.stage(
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
        )
        a5.chain_guard.finish(Failure, Status, pid, 5)
    else:
        gl.store(Status + pid, 0)


@g.jit
def root6(
    X,
    W,
    Stream,
    Post,
    Pre,
    Alpha,
    Beta,
    KPart,
    P,
    Epochs,
    Barriers,
    Status,
    Out,
    Updated,
    GRID: gl.constexpr,
    STRIDE: gl.constexpr,
    PRE_BF16: gl.constexpr,
    SKIP: gl.constexpr = -1,
    Seed=None,
    Trace=None,
    Failure=None,
    Decision=None,
):
    pid = gl.program_id(0)
    if decision_enabled(Decision, 6):
        stage6(
            gl.program_id(0),
            X,
            W,
            Stream,
            Post,
            Pre,
            Alpha,
            Beta,
            KPart,
            P,
            Epochs,
            Barriers,
            Status,
            Out,
            Updated,
            GRID,
            STRIDE,
            PRE_BF16,
            SKIP,
            Seed,
            Trace,
        )
        a6.chain_guard.finish(Failure, Status, pid, 6)
    else:
        gl.store(Status + pid, 0)


@g.jit
def pair12(
    A,
    CA: gl.constexpr,
    RA,
    B,
    CB: gl.constexpr,
    RB,
    Epochs,
    Sync,
    Status,
    Failure,
    Decision,
    GRID: gl.constexpr,
    CodeWarm=None,
):
    pid = gl.program_id(0)
    if CodeWarm is not None:
        if (pid >= 740) & (pid < 752):
            code_probe.warm(CodeWarm, pid - 740)
    tag = qorig._advance_tag(Epochs, pid)
    root1(
        *A,
        GRID=CA[0],
        STRIDE=CA[1],
        EPS=CA[2],
        SCALE=CA[3],
        ROPE=CA[4],
        HAS_MAP=CA[5],
        SKIP=CA[6],
        Seed=RA[0],
        Trace=RA[1],
        Failure=RA[2],
    )
    ready, polls = boundary_barrier(Sync, tag, Failure, Decision, GRID)
    gl.store(Status + pid, ready)
    if not ready:
        scoped.publish(Decision + pid * 32, 1)
        old = gl.atomic_cas(Failure, 0, 1, sem="relaxed", scope="gpu")
        if old == 0:
            gl.store(Failure + 1, pid)
    root2(
        *B,
        GRID=CB[0],
        SPLITS=CB[1],
        MAX_PAGES=CB[2],
        SCALE=CB[3],
        LEFT=CB[4],
        HAS_MAP=CB[5],
        MAP_PAGE=CB[6],
        SKIP=CB[7],
        Seed=RB[0],
        Failure=RB[1],
        Decision=Decision,
    )


@g.jit
def pair34(
    A,
    CA: gl.constexpr,
    RA,
    B,
    CB: gl.constexpr,
    RB,
    Epochs,
    Sync,
    Status,
    Failure,
    Decision,
    GRID: gl.constexpr,
    CodeWarm=None,
):
    pid = gl.program_id(0)
    if CodeWarm is not None:
        if (pid >= 740) & (pid < 752):
            code_probe.warm(CodeWarm, pid - 740)
    tag = qorig._advance_tag(Epochs, pid)
    root3(
        *A,
        GRID=CA[0],
        STRIDE=CA[1],
        RANK=CA[2],
        PAYLOAD=CA[3],
        FLAGS=CA[4],
        SKIP=CA[5],
        Seed=RA[0],
        Trace=RA[1],
        Failure=RA[2],
    )
    ready, polls = boundary_barrier(Sync, tag, Failure, Decision, GRID)
    gl.store(Status + pid, ready)
    if not ready:
        scoped.publish(Decision + pid * 32, 3)
        old = gl.atomic_cas(Failure, 0, 3, sem="relaxed", scope="gpu")
        if old == 0:
            gl.store(Failure + 1, pid)
    root4(
        *B,
        GRID=CB[0],
        STRIDE=CB[1],
        LOCAL_START=CB[2],
        SKIP=CB[3],
        Seed=RB[0],
        Failure=RB[1],
        Decision=Decision,
    )


@g.jit
def pair56(
    A,
    CA: gl.constexpr,
    RA,
    B,
    CB: gl.constexpr,
    RB,
    Epochs,
    Sync,
    Status,
    Failure,
    Decision,
    GRID: gl.constexpr,
    CodeWarm=None,
):
    pid = gl.program_id(0)
    if CodeWarm is not None:
        if (pid >= 740) & (pid < 752):
            code_probe.warm(CodeWarm, pid - 740)
    tag = qorig._advance_tag(Epochs, pid)
    root5(
        *A,
        GRID=CA[0],
        BAR_STRIDE=CA[1],
        PAYLOAD=CA[2],
        FLAGS=CA[3],
        RANK=CA[4],
        SKIP=CA[5],
        READ_MODE=CA[6],
        W2_SK=CA[7],
        Seed=RA[0],
        Trace=RA[1],
        KPartial=RA[2],
        TaskTrace=RA[3],
        Failure=RA[4],
        Out=RA[5],
    )
    ready, polls = boundary_barrier(Sync, tag, Failure, Decision, GRID)
    gl.store(Status + pid, ready)
    if not ready:
        scoped.publish(Decision + pid * 32, 5)
        old = gl.atomic_cas(Failure, 0, 5, sem="relaxed", scope="gpu")
        if old == 0:
            gl.store(Failure + 1, pid)
    root6(
        *B,
        GRID=CB[0],
        STRIDE=CB[1],
        PRE_BF16=CB[2],
        SKIP=CB[3],
        Seed=RB[0],
        Trace=RB[1],
        Failure=RB[2],
        Decision=Decision,
    )
