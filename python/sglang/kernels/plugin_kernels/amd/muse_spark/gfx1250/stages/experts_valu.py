"""Low-hit A16W4 VALU GEMV: packed conversion, BF16 pairs and FP32 FMA.

Plain BF16 dot2 is rejected on this gfx1250 toolchain; VOPD dot2 assembled but
faulted in the bounded smoke probe. The supported fallback promotes each BF16
pair exactly into two FP32 values and uses FP32 FMA, with independent lane-local
accumulators. No activation quantization, WMMA or dot-operand layout is used.

The converter runs at unit scale because variable E8M0 payloads in this plain
VALU lane layout did not match the native dequantization. Each lane applies its
power-of-two scale afterward in FP32; the validated fixture scales118..122 keep
these dyadic FP4 values exactly representable in BF16.

A positive H_VALU gives a static row capacity; only actual hit rows compute and
store. This experimental path missed its latency targets. The caller retains
WMMA as its default and for experts above the threshold.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.vector_memory import (
    _load_asm,
    join4,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.attention import (
    _load128,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.qkvg import (
    _advance_tag,
)


@g.jit
def _cache_pairs(X, Sorted, block, SECOND: gl.constexpr, H: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 4, 8], [1, 2, 8], [2, 1, 0])
    if SECOND:
        q = gl.arange(0, H * 1024, layout=gl.BlockedLayout([1], [32], [16], [0]))
        slots = gl.load(Sorted + block * 16 + q // 1024)
        if H == 1:
            a, b, c, d = _load128(X + slots * 4096 + (q % 1024) * 4, H * 2)
        else:
            asm: gl.constexpr = _load_asm(1, H * 2)
            a, b, c, d, saved = gl.inline_asm_elementwise(
                asm[0],
                asm[1],
                [X + slots * 4096 + (q % 1024) * 4, (slots < 64).to(gl.int32)],
                dtype=(gl.int32,) * 5,
                is_pure=False,
                pack=H * 2,
            )
        value = join4(a, b, c, d, H * 1024).to(gl.float32, bitcast=True).to(gl.bfloat16)
        low, high = gl.split(value.reshape((H * 2048, 2)))
        pairs = low.to(gl.uint16, bitcast=True).to(gl.uint32) | (
            high.to(gl.uint16, bitcast=True).to(gl.uint32) << 16
        )
        pairs = pairs.reshape((H, 8, 256))
        pairs = gl.convert_layout(pairs, layout)
    else:
        hh = gl.arange(0, H, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
        ss = gl.arange(0, 8, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout)))
        kk = gl.arange(0, 256, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
        slots = gl.load(Sorted + block * 16 + hh)
        pairs = gl.load(
            X.to(gl.pointer_type(gl.uint32))
            + (slots[:, None, None] // 8) * 2048
            + ss[None, :, None] * 256
            + kk[None, None, :],
            slots[:, None, None] < 64,
            other=0,
        )
    pairs = gl.convert_layout(
        pairs.reshape((H * 8, 256)), gl.BlockedLayout([1, 4], [4, 8], [2, 8], [1, 0])
    )
    shared = gl.allocate_shared_memory(
        gl.uint32, [H * 8, 256], gl.SwizzledSharedLayout(4, 1, 8, [1, 0])
    )
    shared.store(pairs)
    return shared


@g.jit
def _input_pairs(cached, hit, step):
    layout: gl.constexpr = gl.BlockedLayout([1, 16], [8, 4], [1, 16], [0, 1])
    j = gl.arange(0, 1024, layout=gl.SliceLayout(0, layout))
    index = gl.full((8, 1024), 0, gl.int32, layout) + step * 16 + j[None, :] % 16
    value = (
        cached.slice(hit * 8, 8, dim=0)
        .gather(index, 1)
        .reshape((8, 64, 16))
        .permute((1, 0, 2))
    )
    return gl.convert_layout(
        value, gl.BlockedLayout([1, 1, 16], [4, 8, 1], [16, 1, 1], [2, 1, 0])
    )


@g.jit
def _weight_packet(W, Scale, expert, tile, step, ROWS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 4], [4, 8, 1], [16, 1, 1], [2, 1, 0])
    nn = gl.arange(0, 64, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    ss = gl.arange(0, 8, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    word = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    row = tile * 64 + nn
    values = gl.load(
        W.to(gl.pointer_type(gl.uint32))
        + expert * ROWS * 512
        + row[:, None, None] * 512
        + ss[None, :, None] * 64
        + step * 4
        + word[None, None, :],
        cache_modifier=".cs",
    )
    even, odd = gl.split(values.reshape((64, 8, 2, 2)))
    p0, p2 = gl.split(even)
    p1, p3 = gl.split(odd)
    base: gl.constexpr = gl.BlockedLayout([1, 1], [4, 8], [16, 1], [1, 0])
    r = gl.arange(0, 64, layout=gl.SliceLayout(1, base))
    k = gl.arange(0, 8, layout=gl.SliceLayout(0, base))
    scale = gl.load(
        Scale
        + expert * ROWS * 128
        + (tile * 64 + r[:, None]) * 128
        + k[None, :] * 16
        + step
    )
    return (
        gl.convert_layout(p0, base),
        gl.convert_layout(p1, base),
        gl.convert_layout(p2, base),
        gl.convert_layout(p3, base),
        scale,
    )


@g.constexpr_function
def _converter_asm():
    asm = [
        f"v_cvt_scale_pk8_bf16_fp4 v[{24 + 4 * i}:{27 + 4 * i}], ${16 + i}, 0x7f7f7f7f"
        for i in range(4)
    ]
    asm += ["s_wait_alu depctr_va_vdst(0)"]
    constraints = [f"=&{{v{24 + i}}}" for i in range(16)] + ["v"] * 4
    return "\n".join(asm), ",".join(constraints)


@g.jit
def _convert_packet(p0, p1, p2, p3):
    """Convert at uniform unit scale; the lane E8M0 factor is applied in FP32."""
    asm: gl.constexpr = _converter_asm()
    b = gl.inline_asm_elementwise(
        asm[0], asm[1], [p0, p1, p2, p3], dtype=(gl.uint32,) * 16, is_pure=True, pack=1
    )
    q0 = gl.join(gl.join(b[0], b[1]), gl.join(b[2], b[3]))
    q1 = gl.join(gl.join(b[4], b[5]), gl.join(b[6], b[7]))
    q2 = gl.join(gl.join(b[8], b[9]), gl.join(b[10], b[11]))
    q3 = gl.join(gl.join(b[12], b[13]), gl.join(b[14], b[15]))
    pairs = (
        gl.join(gl.join(q0, q1), gl.join(q2, q3))
        .permute((0, 1, 5, 4, 3, 2))
        .reshape((64, 8, 16))
    )
    return gl.convert_layout(
        pairs, gl.BlockedLayout([1, 1, 16], [4, 8, 1], [16, 1, 1], [2, 1, 0])
    )


@g.jit
def _piece(values, CHUNK: gl.constexpr, WIDTH: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 16], [4, 8, 1], [16, 1, 1], [2, 1, 0]
    )
    component = gl.arange(0, WIDTH, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    index = (
        gl.full((64, 8, WIDTH), CHUNK * WIDTH, gl.int32, layout)
        + component[None, None, :]
    )
    return gl.gather(values, index, 2)


@g.jit
def _gemv(W, Scale, cached, expert, tile, hits, ROWS: gl.constexpr, H: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 16], [4, 8, 1], [16, 1, 1], [2, 1, 0]
    )
    # Keep independent product chains, limiting state for larger static capacities.
    ACC: gl.constexpr = 16 if H == 1 else (4 if H <= 4 else 2)
    accum = ()
    for hit in gl.static_range(H):
        accum += (gl.full((64, 8, ACC), 0, gl.float32, layout),)
    current = _weight_packet(W, Scale, expert, tile, 0, ROWS)
    following = _weight_packet(W, Scale, expert, tile, 1, ROWS)
    future = _weight_packet(W, Scale, expert, tile, 2, ROWS)
    for step in range(16):
        pairs = _convert_packet(current[0], current[1], current[2], current[3])
        factor_bits = gl.where(
            current[4] == 0, 0x00400000, current[4].to(gl.uint32) << 23
        ).to(gl.uint32)
        factor = gl.convert_layout(
            factor_bits.to(gl.float32, bitcast=True), gl.SliceLayout(2, layout)
        )
        low = (pairs << 16).to(gl.float32, bitcast=True) * factor[:, :, None]
        high = (pairs & 0xFFFF0000).to(gl.float32, bitcast=True) * factor[:, :, None]
        for hit in gl.static_range(H):
            if hit < hits:
                x = _input_pairs(cached, hit, step)
                value = accum[hit]
                for chunk in gl.static_range(16 // ACC):
                    xp = _piece(x, chunk, ACC)
                    xlow = (xp << 16).to(gl.float32, bitcast=True)
                    xhigh = (xp & 0xFFFF0000).to(gl.float32, bitcast=True)
                    value = gl.fma(_piece(low, chunk, ACC), xlow, value)
                    value = gl.fma(_piece(high, chunk, ACC), xhigh, value)
                accum = accum[:hit] + (value,) + accum[hit + 1 :]
        current = following
        following = future
        if step + 3 < 16:
            future = _weight_packet(W, Scale, expert, tile, step + 3, ROWS)
    out = ()
    # Fixed pair-position tree followed by the eight K-split lanes.
    for hit in gl.static_range(H):
        out += (gl.sum(gl.sum(accum[hit], 2), 1),)
    return out


@g.jit
def _valu_work(
    X,
    Sorted,
    W,
    Scale,
    Inter,
    Out,
    Arrivals,
    Flags,
    block,
    expert,
    tile,
    tag,
    hits,
    SECOND: gl.constexpr,
    H: gl.constexpr,
):
    ready = gl.full((), True, gl.int1, gl.BlockedLayout([1], [32], [16], [0]))
    if SECOND:
        seen, polls = scoped.wait_flag(
            Flags + (block * 64 + tile) * 32, tag, 100_000_000, SHADER_CLOCK=True
        )
        ready = seen == tag
    if ready:
        cached = _cache_pairs(X, Sorted, block, SECOND, H)
        first = _gemv(W, Scale, cached, expert, tile, hits, 4096 if SECOND else 8192, H)
        if not SECOND:
            second = _gemv(W, Scale, cached, expert, tile + 64, hits, 8192, H)
        cols = gl.arange(
            0,
            64,
            layout=gl.SliceLayout(1, gl.BlockedLayout([1, 1], [4, 8], [16, 1], [1, 0])),
        )
        for hit in gl.static_range(H):
            if hit < hits:
                value = first[hit]
                if not SECOND:
                    value = value / (1.0 + gl.exp(-value)) * second[hit]
                value = (
                    gl.convert_layout(value, cols.type.layout)
                    .to(gl.bfloat16)
                    .to(gl.float32)
                )
                slot = gl.load(Sorted + block * 16 + hit)
                scoped.store_words(
                    Out + slot * 4096 + tile * 64 + cols,
                    value.to(gl.int32, bitcast=True),
                    (cols >= 0) & (slot < 64) & (hit < hits),
                )
        if not SECOND:
            scoped.drain_stores()
            ticket = _advance_tag(Arrivals, block)
            if ticket % 64 == 0:
                scoped.publish_flags(Flags + block * 64 * 32, tag, 64)
    return ready


@g.jit
def valu_task(
    X,
    Sorted,
    W,
    Scale,
    Inter,
    Out,
    Arrivals,
    Flags,
    block,
    expert,
    tile,
    tag,
    hits,
    SECOND: gl.constexpr,
    LIMIT: gl.constexpr,
):
    if LIMIT > 0:
        return _valu_work(
            X,
            Sorted,
            W,
            Scale,
            Inter,
            Out,
            Arrivals,
            Flags,
            block,
            expert,
            tile,
            tag,
            hits,
            SECOND,
            LIMIT,
        )
    return gl.full((), False, gl.int1, gl.BlockedLayout([1], [32], [16], [0]))
