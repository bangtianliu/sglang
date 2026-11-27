"""S2 trial: next chunk's QK precedes current PV; fixed split/reduction math."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout, gfx1250

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.common_shape import (
    CommonShape,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_attention as original,
)


@g.jit
def copy_chunk(
    K,
    V,
    Pages,
    Map,
    offset,
    start,
    end,
    head,
    MAX_PAGES: gl.constexpr,
    HAS_MAP: gl.constexpr,
    MAP_PAGE: gl.constexpr,
    SCOPED_IO: gl.constexpr,
    SharedK,
    SharedV,
    bank,
):
    load: gl.constexpr = gl.BlockedLayout([1, 16], [8, 4], [4, 1], [1, 0])
    token_id = gl.arange(0, 64, layout=gl.SliceLayout(1, load))
    d = gl.arange(0, 64, layout=gl.SliceLayout(0, load))
    token = offset + token_id
    page = gl.gather(
        Pages, gl.minimum(gl.maximum((token - start) // 16, 0), MAX_PAGES - 1), 0
    )
    slot = page + token % 16 if MAP_PAGE else page * 16 + token % 16
    if HAS_MAP and not MAP_PAGE:
        slot = gl.load(Map + slot)
    valid = (token < end) & (slot >= 0)
    pointer = gl.maximum(slot, 0)[:, None] * 256 + head * 64 + d[None, :]
    gfx1250.async_copy.global_to_shared(
        SharedK.index(bank),
        K + pointer,
        valid[:, None],
        0.0,
        cache_modifier=".cg" if SCOPED_IO else "",
    )
    gfx1250.async_copy.global_to_shared(
        SharedV.index(bank),
        V + pointer,
        valid[:, None],
        0.0,
        cache_modifier=".cg" if SCOPED_IO else "",
    )
    gfx1250.async_copy.commit_group()
    return valid


@g.jit
def partial(
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
    SCALE: gl.constexpr,
    LEFT: gl.constexpr,
    SPLITS: gl.constexpr,
    HAS_MAP: gl.constexpr,
    MAP_PAGE: gl.constexpr,
    MAX_PAGES: gl.constexpr,
    SCOPED_IO: gl.constexpr = False,
):
    mma: gl.constexpr = AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[1, 0], [2, 0]],
        instr_shape=[16, 16, 32],
    )
    load: gl.constexpr = gl.BlockedLayout([1, 16], [8, 4], [4, 1], [1, 0])
    r = gl.arange(0, 64, layout=gl.SliceLayout(1, load))
    d = gl.arange(0, 64, layout=gl.SliceLayout(0, load))
    q = gl.load(
        Q + (r[:, None] // 8) * 2048 + (head * 8 + r[:, None] % 8) * 64 + d[None, :],
        cache_modifier=".cg" if SCOPED_IO else "",
    )
    q = gl.convert_layout(q, gl.DotOperandLayout(0, mma, 8))
    length = gl.load(Length)
    first = gl.maximum(0, length - 8 - LEFT) // 64 * 64 if LEFT >= 0 else 0
    tiles = gl.cdiv(length - first, 64)
    start = first + split * tiles // SPLITS * 64
    end = gl.minimum(length, first + (split + 1) * tiles // SPLITS * 64)
    pages = original._cached_pages(Pages, Map, start, end, MAX_PAGES, MAP_PAGE)
    pages = gl.convert_layout(pages, gl.SliceLayout(1, load))
    row = gl.arange(0, 64, layout=gl.SliceLayout(1, mma))
    col = gl.arange(0, 64, layout=gl.SliceLayout(0, mma))
    acc = gl.full((64, 64), 0, gl.float32, mma)
    maximum = gl.full((64,), float("-inf"), gl.float32, gl.SliceLayout(1, mma))
    denom = gl.full((64,), 0, gl.float32, gl.SliceLayout(1, mma))
    scale = SCALE * 1.4426950408889634 * gl.load(KS)
    sh: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 4, [1, 0])
    sk = gl.allocate_shared_memory(K.dtype.element_ty, [2, 64, 64], sh)
    sv = gl.allocate_shared_memory(V.dtype.element_ty, [2, 64, 64], sh)
    current_valid = gl.full((64,), False, gl.int1, gl.SliceLayout(1, load))
    score = gl.full((64, 64), float("-inf"), gl.float32, mma)
    if start < end:
        current_valid = copy_chunk(
            K,
            V,
            pages,
            Map,
            start,
            start,
            end,
            head,
            MAX_PAGES,
            HAS_MAP,
            MAP_PAGE,
            SCOPED_IO,
            sk,
            sv,
            0,
        )
        gfx1250.async_copy.wait_group(0)
        gl.barrier()
        kk = gl.convert_layout(
            sk.index(0).load(load).to(gl.bfloat16).permute((1, 0)),
            gl.DotOperandLayout(1, mma, 8),
        )
        score = gfx1250.wmma(q, kk, gl.full((64, 64), 0, gl.float32, mma)) * scale
    for offset in range(start, end, 64):
        bank = (offset - start) // 64 % 2
        next_valid = current_valid
        if offset + 64 < end:
            next_valid = copy_chunk(
                K,
                V,
                pages,
                Map,
                offset + 64,
                start,
                end,
                head,
                MAX_PAGES,
                HAS_MAP,
                MAP_PAGE,
                SCOPED_IO,
                sk,
                sv,
                1 - bank,
            )
        absolute = length - 8 + row // 8
        valid = gl.convert_layout(current_valid, gl.SliceLayout(0, mma))
        mask = (offset + col[None, :] <= absolute[:, None]) & valid[None, :]
        if LEFT >= 0:
            mask &= offset + col[None, :] >= absolute[:, None] - LEFT
        score = gl.where(mask, score, float("-inf"))
        new_max = gl.maximum(maximum, gl.max(score, 1))
        safe = gl.where(new_max == float("-inf"), 0.0, new_max)
        prob = gl.exp2(score - safe[:, None])
        alpha = gl.exp2(maximum - safe)
        denom = denom * alpha + gl.sum(prob, 1)
        acc *= alpha[:, None]
        next_score = score
        if offset + 64 < end:
            gfx1250.async_copy.wait_group(0)
            gl.barrier()
            kk = gl.convert_layout(
                sk.index(1 - bank).load(load).to(gl.bfloat16).permute((1, 0)),
                gl.DotOperandLayout(1, mma, 8),
            )
            next_score = (
                gfx1250.wmma(q, kk, gl.full((64, 64), 0, gl.float32, mma)) * scale
            )
        vv = gl.convert_layout(
            sv.index(bank).load(load).to(gl.bfloat16), gl.DotOperandLayout(1, mma, 8)
        )
        acc = gfx1250.wmma(
            gl.convert_layout(prob.to(gl.bfloat16), gl.DotOperandLayout(0, mma, 8)),
            vv,
            acc,
        )
        maximum = new_max
        score = next_score
        current_valid = next_valid
        gl.barrier()
    value = acc * (gl.load(VS) / gl.where(denom > 0, denom, 1.0))[:, None]
    lse = gl.where(denom > 0, maximum + gl.log2(denom), float("-inf"))
    return value, lse


def make_workspace(device, *, context, window_left=-1, **kwargs):
    shape = CommonShape(k=3 if window_left < 0 else 2, vgpr_cap=320)
    return original.make_workspace(
        device, context=context, window_left=window_left, shape=shape
    )


@g.jit
def stage(
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
    tag = original._advance_tag(Epochs, pid)
    if pid < 256:
        gl.store(Status + pid, 0)
    original.delayed_point(Seed, pid, 0)
    if pid != SKIP:
        gl.static_assert(GRID >= SPLITS * 4)
        if pid < SPLITS * 4:
            task = pid
            split = task // 4
            head = task % 4
            value, lse = partial(
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
            )
            ml: gl.constexpr = value.type.layout
            row = gl.arange(0, 64, layout=gl.SliceLayout(1, ml))
            col = gl.arange(0, 64, layout=gl.SliceLayout(0, ml))
            a, b, c, d = original.vector_memory.split4(
                value.reshape((4096,)).to(gl.int32, bitcast=True), 1024
            )
            ids = gl.arange(0, 1024, layout=gl.BlockedLayout([1], [32], [4], [0]))
            original._store128(
                P + ((head * 64 + ids // 16) * SPLITS + split) * 64 + ids % 16 * 4,
                a,
                b,
                c,
                d,
                1024,
            )
            original.scoped.store_words(
                LSE + (head * 64 + row) * SPLITS + split,
                lse.to(gl.int32, bitcast=True),
                row >= 0,
            )
    original.delayed_point(Seed, pid, 1)
    ok = gl.full((), False, gl.int1, gl.BlockedLayout([1], [32], [4], [0]))
    if pid != SKIP:
        ok, _, _ = original.scoped.counter_barrier(
            Barriers, tag, 10000000, GRID, 12, MAX_SPINS=1048576
        )
    original.delayed_point(Seed, pid, 2)
    merge_ok = gl.full((), True, gl.int1, gl.BlockedLayout([1], [32], [4], [0]))
    if ok and pid < 256:
        merge_ok = original._merge(
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
    return ok & merge_ok


@g.jit
def _launch(
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
    GUARD_MODE: gl.constexpr = 0,
):
    pid = gl.program_id(0)
    if GUARD_MODE == -1:
        mode = gl.load(Failure + 2)
    else:
        mode = gl.full((), GUARD_MODE, gl.int32, gl.BlockedLayout([1], [32], [4], [0]))
    if original.chain_guard.enabled_runtime(Failure, 2, mode):
        success = stage(
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
        if mode == 0:
            original.chain_guard.finish(Failure, CtaStatus, pid, 2)
            if pid < 256:
                original.chain_guard.finish(Failure, Status, pid, 2)
        elif mode == 1:
            original.chain_guard.finish_value(Failure, success, pid, 2)
    else:
        gl.store(CtaStatus + pid, 0)
        if pid < 256:
            gl.store(Status + pid, 0)


def attention(
    q,
    k,
    v,
    page_table,
    length,
    k_descale,
    v_descale,
    *,
    context,
    window_left=-1,
    mapping=None,
    mapping_page_aligned=False,
    workspace=None,
    ctas=None,
    block_n=64,
    waves=4,
    softmax_scale=0.125,
    skip=-1,
    delay_seed=None,
    shape=None,
    output=None,
    failure=None,
    guard_mode=0,
):
    ws = (
        make_workspace(
            q.device, context=context, window_left=window_left, ctas=ctas, shape=shape
        )
        if workspace is None
        else workspace
    )
    assert block_n == 64 and waves == ws.shape.waves
    out = original.torch.empty_like(q) if output is None else output
    assert (
        out.shape == q.shape
        and out.dtype == q.dtype
        and (out.device == q.device)
        and out.is_contiguous()
    )
    args = (
        q,
        k,
        v,
        page_table,
        mapping,
        length,
        k_descale,
        v_descale,
        ws.partial,
        ws.lse,
        ws.epochs,
        ws.counters,
        ws.status,
        ws.cta_status,
        out,
    )
    opts = dict(
        GRID=ws.ctas,
        SPLITS=ws.splits,
        MAX_PAGES=ws.pages_per_chunk,
        SCALE=softmax_scale,
        LEFT=window_left,
        HAS_MAP=mapping is not None,
        MAP_PAGE=mapping_page_aligned,
        SKIP=skip,
        Seed=delay_seed,
        **ws.shape.options(),
        Failure=failure,
        GUARD_MODE=guard_mode,
    )
    key = (
        window_left,
        mapping is not None,
        mapping_page_aligned,
        skip,
        delay_seed is not None,
        softmax_scale,
        failure is not None,
        guard_mode,
    )
    if key not in ws.admitted:
        kernel = _launch.warmup(*args, **opts, grid=(ws.ctas,))
        resource = ws.shape.admit("common-S2", kernel, grid=ws.ctas)
        ws.admitted[key] = dict(
            vgprs=kernel.n_regs,
            lds=kernel.metadata.shared,
            spills=kernel.n_spills,
            occupancy=resource.active_ctas_per_cu,
        )
    _launch[ws.ctas,](*args, **opts)
    return out
