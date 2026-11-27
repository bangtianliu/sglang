"""MK.4: S3 o_proj + all-reduce + boundary in ONE launch (MI355 structure).

128 CTAs x 4 warps. Every CTA computes an exact N-split o_proj tile (64 columns,
tlx_fused34.ntile_dot), stores the usual projection/partial outputs, then pushes
its 64 columns of the 8 rows straight into the 4 ranks' all-reduce slots
(producer-side push), drains, and takes an arrival number. The last 8 arrivals
become the row workers: they wait for the remaining arrivals (short), raise the
row flags, gather the 4 ranks' row in rank order and run the unchanged
norm_residual. Row math and sum order are those of s3_boundary: bit-exact.
"""

import torch
import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_code_warm,
    tlx_guard,
    tlx_standin,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_memory as mem,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_asm import (
    _load_asm,
    _store_asm,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    ROW1,
    BlockedEncoding,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_route import (
    finish_row_body,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_boundary import (
    norm_residual,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_fused34 import (
    ntile_dot,
    ntile_dot_half,
    ntile_dot_quarter,
    ntile_dot_wait,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_s3split import (
    _drain_zero,
)

TILE = tl.constexpr(BlockedEncoding([1, 2], [4, 8], [4, 1], [1, 0]))
GROUP8 = tl.constexpr(BlockedEncoding([1, 8], [16, 2], [1, 4], [1, 0]))
GROUP4 = tl.constexpr(BlockedEncoding([1, 4], [8, 4], [2, 2], [1, 0]))
TILES = tl.constexpr(128)
_DBG = tl.constexpr(int(__import__("os").environ.get("MK_DBG", "0")))
_DBGM = tl.constexpr(int(__import__("os").environ.get("MK_DBGM", "31")))


@triton.jit
def _dbg(Dbg, slot, value):
    lane = tl.arange(0, 128)
    tlx_standin._store_sys_masked(Dbg + slot + lane * 0, lane * 0 + value, lane == 0)


@triton.jit(do_not_specialize=["code_off", "code_quads"])
def s3_fused(
    X, W, P, Packed, Status, Cnt, Ep,
    Stream, Pre, Alpha, Beta, Out, Updated, Epochs, Scratch, Peers, Reduced, Failure,
    GRID: tl.constexpr, RANK: tl.constexpr, PAYLOAD: tl.constexpr, FLAGS: tl.constexpr,
    STORE_REDUCED: tl.constexpr = True, NB: tl.constexpr = 2, BOUND: tl.constexpr = 20000,
    Trace=None,
    W4=None, P4=None, Status4=None, X4Cnt=None, S4: tl.constexpr = False,
    NB4: tl.constexpr = 2, SLEEP4: tl.constexpr = 8, PRE4: tl.constexpr = True,
    S4TILES: tl.constexpr = 72, X4WT: tl.constexpr = True,
    CodeSink=None, code_off=0, code_quads=0, WARM: tl.constexpr = False,
    S4X: tl.constexpr = False, Xchg=None, PairCnt=None, Ep4=None,
    S4ARR: tl.constexpr = False,
    ROWS4: tl.constexpr = False, S4Arr=None, Bias4=None, Norm4=None, Scores4=None, Ids4=None,
    Y4=None, Rows4=None, HEAD_SLEEP: tl.constexpr = 0, WARM4: tl.constexpr = False,
    X4Ok=None, Dbg=None, FAST4: tl.constexpr = 0,
    S4Q: tl.constexpr = False, Xq=None, QOk=None,
):
    WIDTH: tl.constexpr = 8192
    blk = tl.program_id(0)
    t0 = tlx_standin.clock()
    if S4Q:
        if blk >= 128:
            # MK.9: S4 projection as 288 exact quarter-tiles. Each CTA holds its whole
            # 256 KiB weight slice in LDS before x4 is ready (hides behind the S3
            # all-reduce), then the last of a tile's 4 quarters combines them.
            if _DBG == 1:
                return
            q = blk - 128
            tile = q // 4
            part = q % 4
            e4 = tl.load(Ep4 + q) + 1
            tlx.workgroup_barrier()  # every wave loads before any wave stores
            tl.store(Ep4 + q, e4)
            allowed4 = tlx_guard.enabled(Failure, 4)
            if (Dbg is not None) & ((_DBGM & 1) != 0):
                _dbg(Dbg, q * 16 + 0, e4)
            sq, rdy, tqw, tql = ntile_dot_quarter(Out, W4, tile, part, X4Cnt + (q % 32) * 32, 8 * e4,
                                        BOUND, SLEEP4, FAST4, _DBG == 2)
            if _DBG == 3:
                tl.store(Xq + tile * 2048 + part * 512 + tl.arange(0, 16)[:, None] * 64
                         + tl.arange(0, 64)[None, :], sq.to(tl.int32, bitcast=True),
                         tl.arange(0, 16)[:, None] < 8)
                return
            lv = tlx.require_layout(tl.arange(0, 128), ROW1)
            xv = mem.load_words_masked(X4Ok + (lv % 8) * 32, lv < 8)
            x4ok = tl.sum(((xv == e4) & (lv < 8)).to(tl.int32), 0) == 8
            okq = rdy & x4ok
            tqd = tlx_standin.clock()
            if (Dbg is not None) & ((_DBGM & 2) != 0):
                _dbg(Dbg, q * 16 + 1, 1 + rdy.to(tl.int32) * 10 + x4ok.to(tl.int32) * 100)
            if _DBG == 6:
                tl.store(QOk + tile * 4 + part, tl.where(okq, e4, 0))
                return
            # b128 device-scope stores from a blocked layout (per-element masked asm
            # stores on the WMMA layout fault: wild addresses)
            wq = tlx.require_layout(sq.to(tl.int32, bitcast=True), GROUP4)
            qe, qo = tl.split(tl.reshape(wq, (16, 16, 2, 2)))
            qa, qc = tl.split(qe)
            qb, qd = tl.split(qo)
            gr = tl.arange(0, 16)[:, None]
            gq = tl.arange(0, 16)[None, :]
            qptr = Xq + tile * 2048 + part * 512 + gr * 64 + gq * 4
            qm = tl.broadcast_to(gr < 8, (16, 16)).to(tl.int32)
            specs: tl.constexpr = _store_asm(1, 1, True)
            zz = tl.inline_asm_elementwise(specs[0], specs[1], [qa, qb, qc, qd, qptr, qm],
                                           dtype=tl.int32, is_pure=False, pack=1)
            if _DBG == 7:
                z7 = _drain_zero(zz)
                tl.store(QOk + tile * 4 + part, tl.max(tl.max(z7, 1), 0))
                return
            zo = mem.store_words(QOk + tile * 4 + part + lv * 0, lv * 0 + tl.where(okq, e4, 0),
                                 lv == 0)
            z2 = _drain_zero(zz)
            z3 = _drain_zero(zo)
            tlx.workgroup_barrier()
            if _DBG == 8:
                tl.store(QOk + tile * 4 + part, tl.max(tl.max(z2, 1), 0) + tl.max(z3, 0))
                return
            old = tl.atomic_add(PairCnt + tile * 32 + tl.max(tl.max(z2, 1), 0) + tl.max(z3, 0), 1,
                                sem="relaxed", scope="gpu")
            if (Dbg is not None) & ((_DBGM & 4) != 0):
                _dbg(Dbg, q * 16 + 2, old + 1)
            if _DBG == 4:
                return
            if _DBG == 9:
                tl.store(Xq + 72 * 2048 - 1 - q, old)
                return
            if Trace is not None:
                kq = tl.arange(0, 8)
                tl.store(Trace + (272 + q) * 8 + kq,
                         tl.where(kq == 0, t0, tl.where(kq == 1, tqw, tl.where(kq == 2, tqd, tl.where(kq == 3, tlx_standin.clock(), tql)))).to(tl.int64), kq < 5)
            if old == 4 * e4 - 1:
                # last quarter: the other three are published (stored, drained, counted)
                base = Xq + tile * 2048 + lv * 4
                one = lv * 0 + 1
                specq: tl.constexpr = _load_asm(4, 1)
                raw = tl.inline_asm_elementwise(
                    specq[0], specq[1],
                    [tlx.release_layout(base), tlx.release_layout(base + 512),
                     tlx.release_layout(base + 1024), tlx.release_layout(base + 1536),
                     tlx.release_layout(one)],
                    dtype=(tl.int32,) * 17, is_pure=False, pack=1,
                )
                if _DBG == 5:
                    return
                ov = mem.load_words_masked(QOk + tile * 4 + lv, lv < 4)
                allok = tl.sum(((ov == e4) & (lv < 4)).to(tl.int32), 0) == 4
                ok4 = allowed4 & allok
                prow = lv // 16
                pcol = tile * 64 + (lv % 16) * 4
                for k in tl.static_range(4):
                    s0 = tlx.require_layout(raw[k], ROW1).to(tl.float32, bitcast=True)
                    s1 = tlx.require_layout(raw[4 + k], ROW1).to(tl.float32, bitcast=True)
                    s2 = tlx.require_layout(raw[8 + k], ROW1).to(tl.float32, bitcast=True)
                    s3 = tlx.require_layout(raw[12 + k], ROW1).to(tl.float32, bitcast=True)
                    v4 = (s0 + s2) + (s1 + s3)
                    if ok4:
                        tl.store(P4 + prow * 4608 + pcol + k, v4.to(tl.int32, bitcast=True))
                tl.store(Status4 + tile * 4 + tl.arange(0, 4), ok4.to(tl.int32))
                if allowed4:
                    tlx_guard.finish(Failure, ok4, tile, 4)
                if (Dbg is not None) & ((_DBGM & 8) != 0):
                    _dbg(Dbg, q * 16 + 3, 1 + ok4.to(tl.int32))
                if Trace is not None:
                    tl.store(Trace + (128 + tile) * 8 + tl.arange(0, 2),
                             tl.where(tl.arange(0, 2) == 0, t0, tlx_standin.clock()).to(tl.int64))
            return
    if S4X:
        if blk >= 128:
            if _DBG == 1:
                return
            h = blk - 128
            tile = h // 2
            e4 = tl.load(Ep4 + h) + 1
            tlx.workgroup_barrier()  # every wave loads before any wave stores
            tl.store(Ep4 + h, e4)
            allowed4 = tlx_guard.enabled(Failure, 4)
            if WARM4:  # S4 CTAs are idle at the start: warm the whole .text per XCD
                tlx_code_warm.code_warm(h, 144, CodeSink, code_off, code_quads)
            if HEAD_SLEEP > 0:
                for _hs in tl.static_range(HEAD_SLEEP):
                    tl.inline_asm_elementwise("s_sleep 127\nv_mov_b32 $0, 0", "=v", [], dtype=tl.int32,
                                              is_pure=False, pack=1)
            if Dbg is not None:
                _dbg(Dbg, h * 16 + 0, 1)
                _dbg(Dbg, h * 16 + 1, e4)
            if h % 2 == 1:
                vh, rdy = ntile_dot_half(Out, W4, tile, 1, X4Cnt + (h % 32) * 32, 8 * e4,
                                         NB4, BOUND, SLEEP4, FAST4)
                if Dbg is not None:
                    _dbg(Dbg, h * 16 + 2, 2 + rdy.to(tl.int32) * 100)
                rr = tl.arange(0, 16)[:, None]
                cc = tl.arange(0, 64)[None, :]
                zz = mem.store_words(Xchg + tile * 1024 + rr * 64 + cc,
                                     vh.to(tl.int32, bitcast=True), rr < 8)
                z2 = _drain_zero(zz)
                tlx.workgroup_barrier()
                tl.atomic_add(PairCnt + tile * 32 + tl.max(tl.max(z2, 1), 0), 1,
                              sem="relaxed", scope="gpu")
            else:
                vl, rdy = ntile_dot_half(Out, W4, tile, 0, X4Cnt + (h % 32) * 32, 8 * e4,
                                         NB4, BOUND, SLEEP4, FAST4)
                pc = mem.load_flag(PairCnt + tile * 32)
                n2 = 0
                while (pc < e4) & (n2 < BOUND):
                    if FAST4 > 0:
                        if n2 >= FAST4:
                            tl.inline_asm_elementwise("s_sleep 127\nv_mov_b32 $0, 0", "=v", [],
                                                      dtype=tl.int32, is_pure=False, pack=1)
                    pc = mem.load_flag(PairCnt + tile * 32)
                    n2 += 1
                d2 = tl.where(pc >= e4, tl.minimum(pc - e4, 0), 0)
                if Dbg is not None:
                    _dbg(Dbg, h * 16 + 2, 3 + rdy.to(tl.int32) * 100)
                    _dbg(Dbg, h * 16 + 3, pc)
                    _dbg(Dbg, h * 16 + 4, d2)
                rr = tl.arange(0, 16)[:, None]
                cc = tl.arange(0, 64)[None, :]
                vrw = tl.load(Xchg + d2 + tile * 1024 + rr * 64 + cc, mask=rr < 8, other=0)
                v4 = vl + vrw.to(tl.float32, bitcast=True)
                lv = tlx.require_layout(tl.arange(0, 128), ROW1)
                xv = mem.load_words_masked(X4Ok + (lv % 8) * 32, lv < 8)
                x4ok = tl.sum(((xv == e4) & (lv < 8)).to(tl.int32), 0) == 8
                ok4 = allowed4 & rdy & (pc >= e4) & x4ok
                if Dbg is not None:
                    _dbg(Dbg, h * 16 + 5, 4 + ok4.to(tl.int32) * 100)
                n4 = tile * 64 + tl.arange(0, 64)
                ta = tlx_standin.clock()
                if ROWS4:
                    # device-scope asm stores: the row workers read them in this launch
                    # b128 device-scope stores: 4 consecutive FP32 columns per thread
                    w32 = tlx.require_layout(v4.to(tl.int32, bitcast=True), GROUP4)
                    pe, po = tl.split(tl.reshape(w32, (16, 16, 2, 2)))
                    pa, pc2 = tl.split(pe)
                    pb, pd = tl.split(po)
                    gr = tl.arange(0, 16)[:, None]
                    gq = tl.arange(0, 16)[None, :]
                    pptr = P4 + gr * 4608 + tile * 64 + gq * 4
                    pm = tl.broadcast_to((gr < 8) & ok4, (16, 16)).to(tl.int32)
                    spec1: tl.constexpr = _store_asm(1, 1, True)
                    zp = tl.inline_asm_elementwise(spec1[0], spec1[1], [pa, pb, pc2, pd, pptr, pm],
                                                   dtype=tl.int32, is_pure=False, pack=1)
                    l4 = tl.arange(0, 4)
                    zs = mem.store_words(Status4 + tile * 4 + l4, l4 * 0 + ok4.to(tl.int32), l4 < 4)
                    z3 = _drain_zero(zp)
                    z4 = _drain_zero(zs)
                    tlx.workgroup_barrier()
                    old4 = tl.atomic_add(S4Arr + tl.max(tl.max(z3, 1), 0) + tl.max(z4, 0), 1,
                                         sem="relaxed", scope="gpu")
                    if allowed4:
                        tlx_guard.finish(Failure, ok4, tile, 4)
                    i4 = old4 - 72 * (e4 - 1)
                    tb = tlx_standin.clock()
                    if Trace is not None:
                        k2 = tl.arange(0, 4)
                        tl.store(Trace + blk * 8 + k2, tl.where(k2 == 0, t0, tl.where(k2 == 1, ta, tl.where(k2 == 2, tb, i4.to(tl.uint64)))).to(tl.int64))
                    if i4 >= 64:
                        prow = i4 - 64
                        t4 = 72 * e4
                        ca = mem.load_flag(S4Arr)
                        n3 = 0
                        while (ca < t4) & (n3 < BOUND):
                            ca = mem.load_flag(S4Arr)
                            n3 += 1
                        dep4 = tl.minimum(ca - t4, 0)
                        tc = tlx_standin.clock()
                        finish_row_body(prow, P4, Bias4, Norm4, Scores4, Ids4, Y4, Status4,
                                        Rows4, Failure, dep4, ca >= t4)
                        if Trace is not None:
                            td = tlx_standin.clock()
                            kk2 = tl.arange(0, 2)
                            tl.store(Trace + blk * 8 + 4 + kk2, tl.where(kk2 == 0, tc, td).to(tl.int64))
                else:
                    if ok4:
                        tl.store(P4 + rr * 4608 + n4[None, :], v4.to(tl.int32, bitcast=True), rr < 8)
                    tl.store(Status4 + tile * 4 + tl.arange(0, 4), ok4.to(tl.int32))
                    if allowed4:
                        tlx_guard.finish(Failure, ok4, tile, 4)
            return
    allowed = tlx_guard.enabled(Failure, 3)
    # ---- o_proj tile (exact) ----
    value = ntile_dot(X, W, blk, 2048, NB, 4)
    value = value.to(tl.bfloat16).to(tl.float32)
    t1 = tlx_standin.clock()
    # ---- producer-side push of this tile's 64 columns (32 BF16-pair words) ----
    bits = (value.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 65535)
    bits = tlx.require_layout(bits, GROUP8)  # 8 consecutive columns per thread
    lo, hi = tl.split(tl.reshape(bits, (16, 8, 4, 2)))  # m = 2k + j
    word4 = lo | (hi << 16)  # [16, 8, 4], word k of group g
    ev, od = tl.split(tl.reshape(word4, (16, 8, 2, 2)))  # k = 2i + j
    wa, wc = tl.split(ev)
    wb, wd = tl.split(od)
    rr = tl.arange(0, 16)[:, None]
    gg = tl.arange(0, 8)[None, :]
    pmask = tl.broadcast_to(rr < 8, (16, 8)).to(tl.int32)
    tag0 = tl.load(Epochs) + 1  # this launch's tag (rows advance only after all pushes)
    if tlx_standin.HOST:
        # real ranks: stage the tile in device scratch (fast DEV stores); the row
        # workers push whole rows to host memory (many small SYS writes are slow)
        hptr = Scratch + RANK * 4 * WIDTH + rr * (WIDTH // 2) + blk * 32 + gg * 4
        spech: tl.constexpr = _store_asm(1, 1, True)
        z = tl.inline_asm_elementwise(
            spech[0], spech[1], [wa, wb, wc, wd, hptr, pmask],
            dtype=tl.int32, is_pure=False, pack=1,
        )
    else:
        ptrs = ()
        for dst in tl.static_range(4):
            peer = tl.load(Peers + dst).to(tl.pointer_type(tl.int32))
            ptrs += (
                peer + PAYLOAD[0] + dst * PAYLOAD[1] + (tag0 % 2) * PAYLOAD[2]
                + RANK * 4 * WIDTH + rr * (WIDTH // 2) + blk * 32 + gg * 4,
            )
        spec: tl.constexpr = _store_asm(4, 1, True)
        z = tl.inline_asm_elementwise(
            spec[0], spec[1],
            [wa, wb, wc, wd, ptrs[0], ptrs[1], ptrs[2], ptrs[3], pmask],
            dtype=tl.int32, is_pure=False, pack=1,
        )
    zd = _drain_zero(z * 0)
    tlx.workgroup_barrier()
    e = tl.load(Ep + blk) + 1
    tlx.workgroup_barrier()  # every wave loads before any wave stores
    tl.store(Ep + blk, e)
    old = tl.atomic_add(Cnt + tl.max(tl.max(zd, 1), 0), 1, sem="relaxed", scope="gpu")
    t2 = tlx_standin.clock()
    # outputs no CTA of this launch reads: after the arrival, off the critical path
    r = tl.arange(0, 16)
    nn = blk * 64 + tl.arange(0, 64)
    tl.store(P + r[:, None] * WIDTH + nn[None, :], value.to(tl.int32, bitcast=True), r[:, None] < 8)
    tl.store(Packed + r[:, None] * WIDTH + nn[None, :], value.to(tl.bfloat16), r[:, None] < 8)
    tl.store(Status + blk * 4 + tl.arange(0, 4), allowed.to(tl.int32))
    idx = old - TILES * (e - 1)
    if WARM & (not WARM4):  # after the arrival: off the tile's critical path
        tlx_code_warm.code_warm(blk, 128, CodeSink, code_off, code_quads)
    if S4ARR:
        if idx < S4TILES:
            allowed4 = tlx_guard.enabled(Failure, 4)
            v4, ready4 = ntile_dot_wait(Out, W4, idx, X4Cnt + (idx % 32) * 32, 8 * e, 8192,
                                        NB4, 8, BOUND, SLEEP4, PRE4, FAST4)
            ok4 = allowed4 & ready4
            if Trace is not None:
                t4d = tlx_standin.clock()
                tl.store(Trace + blk * 8 + 7, t4d.to(tl.int64))
            r4 = tl.arange(0, 16)
            n4 = idx * 64 + tl.arange(0, 64)
            if ok4:
                tl.store(P4 + r4[:, None] * 4608 + n4[None, :], v4.to(tl.int32, bitcast=True), r4[:, None] < 8)
            tl.store(Status4 + idx * 4 + tl.arange(0, 4), ok4.to(tl.int32))
            if allowed4:
                tlx_guard.finish(Failure, ok4, idx, 4)
    if Trace is not None:
        k4 = tl.arange(0, 4)
        tl.store(Trace + blk * 8 + k4, tl.where(k4 == 0, t0, tl.where(k4 == 1, t1, tl.where(k4 == 2, t2, idx.to(tl.uint64)))).to(tl.int64))
    if idx >= TILES - 8:
        row = idx - (TILES - 8)
        target = TILES * e
        c = mem.load_flag(Cnt)
        n = 0
        while (c < target) & (n < BOUND):
            c = mem.load_flag(Cnt)
            n += 1
        ready = c >= target
        if (Dbg is not None) & ((_DBGM & 16) != 0):
            _dbg(Dbg, (288 + row) * 16 + 0, c + 1)
        t3 = tlx_standin.clock()
        tag = tlx_standin.advance(Epochs, row)
        # ---- raise row flags, poll the sources, gather in rank order ----
        i = tlx.require_layout(tl.arange(0, 128), ROW1)
        dstmask = i < 4
        Q: tl.constexpr = WIDTH // 8
        q = tlx.require_layout(tl.arange(0, Q), ROW1)
        at = row * (WIDTH // 2) + q * 4
        mask = tlx.require_layout(tl.full((Q,), 1, tl.int32), ROW1)
        if tlx_standin.HOST:
            spec_l: tl.constexpr = _load_asm(1, 4)
            own = tl.inline_asm_elementwise(
                spec_l[0], spec_l[1],
                [tlx.release_layout(Scratch + RANK * 4 * WIDTH + at), tlx.release_layout(mask)],
                dtype=(tl.int32,) * 5, is_pure=False, pack=4,
            )
            rv, got = tlx_standin._host_exchange(
                tlx.require_layout(own[0], ROW1), tlx.require_layout(own[1], ROW1),
                tlx.require_layout(own[2], ROW1), tlx.require_layout(own[3], ROW1),
                Peers, Reduced, tag, row, at, GRID, RANK, WIDTH, False)
        else:
            peerv = tl.load(Peers + i, dstmask, 0).to(tl.pointer_type(tl.int32))
            offset = FLAGS[0] + i * FLAGS[1] + (tag % 2) * FLAGS[2] + (RANK * GRID + row) * 32
            mem.store_words(peerv + offset, tl.full((128,), tag, tl.int32), dstmask)
            ptr = Scratch + FLAGS[0] + RANK * FLAGS[1] + (tag % 2) * FLAGS[2] + (i * GRID + row) * 32
            seen = mem.load_words_masked(ptr, dstmask)
            got = tl.sum(((seen == tag) & dstmask).to(tl.int32), 0) == 4
            spins = 1
            while (~got) & (spins < 4096):
                seen = mem.load_words_masked(ptr, dstmask)
                got = tl.sum(((seen == tag) & dstmask).to(tl.int32), 0) == 4
                spins += 1
            mem.drain_stores()
            base = Scratch + PAYLOAD[0] + RANK * PAYLOAD[1] + (tag % 2) * PAYLOAD[2] + at
            spec2: tl.constexpr = _load_asm(4, 4)
            raw = tl.inline_asm_elementwise(
                spec2[0], spec2[1],
                [tlx.release_layout(base), tlx.release_layout(base + 4 * WIDTH),
                 tlx.release_layout(base + 8 * WIDTH), tlx.release_layout(base + 12 * WIDTH),
                 tlx.release_layout(mask)],
                dtype=(tl.int32,) * 17, is_pure=False, pack=4,
            )
            zf = tlx.require_layout(tl.full((Q,), 0, tl.float32), ROW1)
            a, b, c2, d, e2, f, g, h = zf, zf, zf, zf, zf, zf, zf, zf
            for src in tl.static_range(4):
                w0 = tlx.require_layout(raw[src * 4], ROW1)
                w1 = tlx.require_layout(raw[src * 4 + 1], ROW1)
                w2 = tlx.require_layout(raw[src * 4 + 2], ROW1)
                w3 = tlx.require_layout(raw[src * 4 + 3], ROW1)
                a, b = tlx_standin.pair(a, b, w0)
                c2, d = tlx_standin.pair(c2, d, w1)
                e2, f = tlx_standin.pair(e2, f, w2)
                g, h = tlx_standin.pair(g, h, w3)
            rv = mem.join8(a, b, c2, d, e2, f, g, h, Q)
        t4 = tlx_standin.clock()
        if STORE_REDUCED:
            j = tl.arange(0, WIDTH)
            tl.store(Reduced + RANK * 8 * WIDTH + row * WIDTH + j, rv)
        ok = allowed & ready & got
        if ok:
            norm_residual(
                row, Reduced + RANK * 8 * WIDTH, Stream, None, Pre, Alpha, Beta, Out,
                Updated, False, False, Value=rv, PIN_LAYOUT=True, CARRIER_RULE=False,
                OUTPUT_WT=S4 & X4WT,
            )
        tl.store(Status + row, ok)
        if (Dbg is not None) & ((_DBGM & 16) != 0):
            _dbg(Dbg, (288 + row) * 16 + 1, 1 + ok.to(tl.int32))
        t5 = tlx_standin.clock()
        if S4:
            lane = tlx.require_layout(tl.arange(0, 128), ROW1)
            # row validity for this launch: S4 consumers must not project a stale x4
            mem.store_words(X4Ok + row * 32 + lane * 0, lane * 0 + tl.where(ok, e, 0), lane == 0)
            zx = _drain_zero(lane)
            tlx.workgroup_barrier()
            tl.atomic_add(X4Cnt + lane * 32 + tl.max(zx, 0), 1, mask=lane < 32,
                          sem="relaxed", scope="gpu")
        if Trace is not None:
            t6 = tlx_standin.clock()
            k3 = tl.arange(0, 4)
            tl.store(Trace + blk * 8 + 4 + k3, tl.where(k3 == 0, t3, tl.where(k3 == 1, t4, tl.where(k3 == 2, t5, t6))).to(tl.int64))
        tlx_guard.finish(Failure, ok, row, 3)


class State:
    def __init__(self, device):
        self.cnt = torch.zeros(32, device=device, dtype=torch.int32)
        self.ep = torch.zeros(128, device=device, dtype=torch.int32)
        self.x4cnt = torch.zeros(32 * 32, device=device, dtype=torch.int32)
        self.sink = torch.zeros(16, device=device, dtype=torch.int32)
        self.xchg = torch.zeros(72 * 1024, device=device, dtype=torch.int32)
        self.paircnt = torch.zeros(72 * 32, device=device, dtype=torch.int32)
        self.ep4 = torch.zeros(144, device=device, dtype=torch.int32)
        self.s4arr = torch.zeros(32, device=device, dtype=torch.int32)
        self.x4ok = torch.zeros(8 * 32, device=device, dtype=torch.int32)
        self.ep4q = torch.zeros(288, device=device, dtype=torch.int32)
        self.xq = torch.zeros(72 * 2048, device=device, dtype=torch.int32)
        self.qok = torch.zeros(72 * 4, device=device, dtype=torch.int32)
        self.code = (0, 0)  # filled from the compiled kernel after the first launch


def tail(x, gate, o_proj, stream, pre, alpha, beta, *, rank, ws, st, output, failure,
         split_norm, store_reduced, nb=2, trace=None, s4=None, sleep4=8, pre4=True,
         s4tiles=72, x4wt=True, warm=False, nb4=2, s4x=False, rows4=None, head_sleep=0,
         warm4=False, dbg=None, s4q=False):
    """tlx_attn_tail.attention_tail with projection + AR + boundary in one launch."""
    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_attn_tail

    out, updated = output
    p, s = ws.peers, ws.state
    fn = tlx_attn_tail.normalize_rows if split_norm else tlx_attn_tail.normalize
    fn[(256 if split_norm else 16,)](x, gate, ws.normalized, failure, num_warps=4,
                                     enable_fp_fusion=False)
    s3_fused[(128 + (144 if s4x else 0) + (288 if s4q else 0),)](
        ws.normalized, o_proj, ws.projection, ws.partial[rank], s.status, st.cnt, st.ep,
        stream, pre, alpha, beta, out, updated, s.epochs, p.scratch, p.peers, p.reduced,
        failure, GRID=p.ctas, RANK=rank, PAYLOAD=p.payload, FLAGS=p.flags,
        STORE_REDUCED=store_reduced, NB=nb, Trace=trace,
        W4=None if s4 is None else s4[0], P4=None if s4 is None else s4[1],
        Status4=None if s4 is None else s4[2], X4Cnt=st.x4cnt, S4=s4 is not None,
        SLEEP4=sleep4, PRE4=pre4, S4TILES=s4tiles, X4WT=x4wt, NB4=nb4,
        CodeSink=st.sink, code_off=st.code[0], code_quads=st.code[1], WARM=warm,
        S4X=s4x, Xchg=st.xchg, PairCnt=st.paircnt, Ep4=st.ep4q if s4q else st.ep4,
        S4ARR=(s4 is not None) and not s4x and not s4q, S4Q=s4q, Xq=st.xq, QOk=st.qok,
        ROWS4=rows4 is not None, S4Arr=st.s4arr, HEAD_SLEEP=head_sleep, WARM4=warm4 and s4x,
        Bias4=None if rows4 is None else rows4[0], Norm4=None if rows4 is None else rows4[1],
        Scores4=None if rows4 is None else rows4[2], Ids4=None if rows4 is None else rows4[3],
        Y4=None if rows4 is None else rows4[4], Rows4=None if rows4 is None else rows4[5],
        X4Ok=st.x4ok, Dbg=dbg, FAST4=256 if tlx_standin.HOST.value else 0, num_warps=4, enable_fp_fusion=False,
    )
    return out, updated


def refresh_code_range(st):
    """After a first (unwarmed or warmed) launch, point st.code at the kernel's .text."""
    kernels = [k for dc in s3_fused.device_caches.values() for k in dc[0].values()]
    if len(kernels) != 1:
        raise RuntimeError(f"code warm needs exactly one compiled s3_fused, found {len(kernels)}")
    st.code = tlx_code_warm.text_range(kernels[0])
