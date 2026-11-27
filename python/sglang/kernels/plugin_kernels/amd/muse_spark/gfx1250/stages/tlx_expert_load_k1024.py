"""T10.2 private tile-packed load search; never selected by the chain."""

import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_guard
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    BlockedEncoding,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_attention import (
    fixed,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tlx_expert_core import (
    A_LAYOUT,
    AL,
    B_EXPANDED,
    B_PACKED,
    BL,
    MMA,
    SCALE_LAYOUT,
    SL,
    WORD_LOAD,
    async_agent,
    fp4_upcast,
)

AW = tl.constexpr(
    tlx.padded_shared_layout_encoding.with_identity_for([(64, 4)], [16, 64], [1, 0])
)
BW = tl.constexpr(
    tlx.padded_shared_layout_encoding.with_identity_for([(64, 4)], [16, 64], [0, 1])
)
SW = tl.constexpr(
    tlx.shared_linear_layout_encoding(
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], alignment=16
    )
)
CRC4 = tl.constexpr(BlockedEncoding([4], [32], [4], [0]))
FLAT = tl.constexpr(
    tlx.padded_shared_layout_encoding.with_identity_for([(64, 4)], [8192], [0])
)
SCALE_FLAT = tl.constexpr(
    tlx.shared_linear_layout_encoding(
        [[1], [2], [4], [8], [16], [32], [64], [128], [256]], alignment=16
    )
)


@triton.jit
def checksum_words(buffer):
    view = tlx.local_reinterpret(buffer, tl.uint32, [8192], layout=FLAT)
    words = tlx.local_load(view, layout=CRC4)
    groups = words.reshape((16, 128, 4))
    return tl.sum(tl.sum(groups, 2), 0)


@triton.jit
def checksum_scales(buffer):
    view = tlx.local_reinterpret(buffer, tl.uint32, [512], layout=SCALE_FLAT)
    words = tlx.local_load(view, layout=CRC4)
    return tl.sum(words, 0)


@triton.jit
def copy_scale(ptr, dst, SCALE_ASYNC: tl.constexpr = False, Word=None):
    if SCALE_ASYNC:
        word_layout: tl.constexpr = tlx.shared_linear_layout_encoding(
            [[1, 0], [2, 0], [4, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]],
            alignment=16,
        )
        word_dst = tlx.local_reinterpret(dst, tl.uint32, [8, 64], layout=word_layout)
        words = tlx.require_layout(
            Word.reshape((64,))[None, :] + tl.arange(0, 8)[:, None], WORD_LOAD
        )
        tlx.async_load(words, word_dst)
    else:
        ptr = tlx.require_layout(ptr, SL)
        tlx.local_store(dst, tl.load(ptr))


@triton.jit
def prefetch(
    X,
    W,
    Scale,
    ap,
    bp,
    sp,
    spw,
    step,
    bank,
    sa,
    sb,
    ss,
    mask,
    FIRST: tl.constexpr,
    WEIGHT_CACHE: tl.constexpr,
    SCALE_ASYNC: tl.constexpr,
    CROSS_LAUNCH_A: tl.constexpr,
    PACK_W: tl.constexpr,
    PACK_S: tl.constexpr,
):
    if FIRST:
        tlx.async_load(
            X + ap + step * 1024,
            tlx.local_view(sa, bank),
            mask,
            tlx.require_layout(tl.full((16, 1024), 0.0, tl.bfloat16), AL),
        )
    elif CROSS_LAUNCH_A:
        # An ordered W13 dispatch produced X; match native's default-cache DMA.
        # Direct/in-kernel core users retain device scope via the default below.
        tlx.async_load(
            X + ap + step * 1024,
            tlx.local_view(sa, bank),
            mask,
            tlx.require_layout(tl.full((16, 1024), 0.0, tl.bfloat16), AL),
        )
    else:
        async_agent(
            X + ap + step * 1024,
            tlx.local_view(sa, bank),
            mask,
            tlx.require_layout(tl.full((16, 1024), 0.0, tl.bfloat16), AL),
        )
    if WEIGHT_CACHE == "":
        tlx.async_load(
            W + bp + step * (32768 if PACK_W else 512), tlx.local_view(sb, bank)
        )
    else:
        tlx.local_store(
            tlx.local_view(sb, bank),
            tl.load(
                W + bp + step * (32768 if PACK_W else 512), cache_modifier=WEIGHT_CACHE
            ),
        )
    copy_scale(
        Scale + sp + step * (2048 if PACK_S else 32),
        tlx.local_view(ss, bank),
        SCALE_ASYNC,
        Scale.to(tl.pointer_type(tl.uint32)) + spw + step * (512 if PACK_S else 8),
    )


@triton.jit
def weight_sub(sb, ss, bank, sub: tl.constexpr):
    packed = tlx.local_load(
        tlx.local_slice(tlx.local_view(sb, bank), [sub * 64, 0], [64, 64]),
        layout=B_PACKED,
    )
    scales = tlx.local_load(
        tlx.local_slice(tlx.local_view(ss, bank), [sub * 4, 0], [4, 64]),
        layout=SCALE_LAYOUT,
    )
    expanded = tl.broadcast_to(scales[:, None, :], (4, 32, 64)).reshape((128, 64))
    expanded = tlx.require_layout(expanded, B_EXPANDED)
    return fp4_upcast(packed, expanded)


@triton.jit
def expert_tile(
    X,
    Sorted,
    W,
    Scale,
    Out,
    block,
    expert,
    tile,
    FIRST: tl.constexpr,
    WEIGHT_CACHE: tl.constexpr = "",
    SCALE_ASYNC: tl.constexpr = False,
    CROSS_LAUNCH_A: tl.constexpr = False,
    OUTPUT_AGENT: tl.constexpr = False,
    PACK_W: tl.constexpr = False,
    PACK_S: tl.constexpr = False,
    DEPTH: tl.constexpr = 2,
    PREFETCH_FIRST: tl.constexpr = False,
    MODE: tl.constexpr = "load",
    Checksum=None,
    TASK=0,
    HITS: tl.constexpr = (),
):
    row = tl.arange(0, 16)
    ak = tl.arange(0, 1024)
    if MODE == "compute":
        count = tl.full((), 0, tl.int32)
        for j in tl.static_range(len(HITS)):
            count = tl.where(block == j, HITS[j], count)
        slots = tl.where(row < count, row, 64)
    else:
        slots = tl.load(Sorted + block * 16 + row)
    input_rows = slots // 8 if FIRST else slots
    ap = tlx.require_layout(input_rows[:, None] * 4096 + ak[None, :], AL)
    mask = tlx.require_layout(tl.broadcast_to(slots[:, None] < 64, (16, 1024)), AL)
    bk = tl.arange(0, 512)
    bn = tl.arange(0, 64) + tile * 64
    sk = tl.arange(0, 32)
    sn = tl.arange(0, 64) + tile * 64
    N: tl.constexpr = 8192 if FIRST else 4096
    bp = tlx.require_layout(
        expert.to(tl.int64) * N * 2048
        + (tile * 131072 + (bn[None, :] % 64) * 512 if PACK_W else bn[None, :] * 2048)
        + bk[:, None],
        BL,
        WORD_LOAD,
    )
    bp = tlx.require_layout(bp, BL, pin=True)
    bp = tl.max_contiguous(tl.multiple_of(bp, (16, 1)), (16, 1))
    sp = tlx.require_layout(
        expert.to(tl.int64) * N * 128
        + (tile * 8192 + (sn[None, :] % 64) * 32 if PACK_S else sn[None, :] * 128)
        + sk[:, None],
        SL,
    )
    spw = expert.to(tl.int64) * N * 32 + (
        tile * 2048 + (sn[None, :] % 64) * 8 if PACK_S else sn[None, :] * 32
    )
    ash: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(128, 8)], [16, 1024], [1, 0]
    )
    bsh: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(256, 16)], [512, 64], [0, 1]
    )
    ssh: tl.constexpr = tlx.shared_linear_layout_encoding(
        [
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
        ],
        alignment=16,
    )
    sa = tlx.local_alloc((16, 1024), tl.bfloat16, DEPTH, layout=ash)
    sb = tlx.local_alloc((512, 64), tl.uint8, DEPTH, layout=bsh)
    ss = tlx.local_alloc((32, 64), tl.uint8, DEPTH, layout=ssh)
    if FIRST:
        su = tlx.local_alloc((512, 64), tl.uint8, DEPTH, layout=bsh)
        st = tlx.local_alloc((32, 64), tl.uint8, DEPTH, layout=ssh)
    if MODE == "compute":
        word_k = tl.arange(0, 16)
        word_n = tl.arange(0, 64)
        arow = tl.arange(0, 16)
        acol = tl.arange(0, 64)
        lo = (0x3C00 | ((arow[:, None] * 3 + acol[None, :] + TASK) & 127)).to(tl.uint32)
        hi = (0xBC00 | ((arow[:, None] * 5 + acol[None, :] * 3 + TASK) & 127)).to(
            tl.uint32
        )
        av = tl.where(arow[:, None] < count, lo | (hi << 16), 0)
        pp = (
            word_k[:, None].to(tl.uint32) * 0x1234567
            + word_n[None, :].to(tl.uint32) * 0x9E3779B9
            + TASK.to(tl.uint32) * 0x7654321
        )
        scales = (
            tl.full((1, 64), 0x7E7E7E7E, tl.uint32)
            + (word_n[None, :] & 1).to(tl.uint32) * 0x01010101
        )
        for init_bank in tl.static_range(2):
            tlx.local_store(
                tlx.local_reinterpret(
                    tlx.local_view(sa, init_bank), tl.uint32, [16, 64], layout=AW
                ),
                av,
            )
            tlx.local_store(
                tlx.local_reinterpret(
                    tlx.local_view(sb, init_bank), tl.uint32, [16, 64], layout=BW
                ),
                pp,
            )
            tlx.local_store(
                tlx.local_reinterpret(
                    tlx.local_view(ss, init_bank), tl.uint32, [1, 64], layout=SW
                ),
                scales,
            )
            if FIRST:
                tlx.local_store(
                    tlx.local_reinterpret(
                        tlx.local_view(su, init_bank), tl.uint32, [16, 64], layout=BW
                    ),
                    pp ^ 0x35353535,
                )
                tlx.local_store(
                    tlx.local_reinterpret(
                        tlx.local_view(st, init_bank), tl.uint32, [1, 64], layout=SW
                    ),
                    scales,
                )
        tlx.workgroup_barrier()
    else:
        for init_step in tl.static_range(DEPTH - 1):
            prefetch(
                X,
                W,
                Scale,
                ap,
                bp,
                sp,
                spw,
                init_step,
                init_step,
                sa,
                sb,
                ss,
                mask,
                FIRST,
                WEIGHT_CACHE,
                SCALE_ASYNC,
                CROSS_LAUNCH_A,
                PACK_W,
                PACK_S,
            )
            if FIRST:
                if WEIGHT_CACHE == "":
                    tlx.async_load(
                        W + bp + init_step * (32768 if PACK_W else 512) + 4096 * 2048,
                        tlx.local_view(su, init_step),
                    )
                else:
                    tlx.local_store(
                        tlx.local_view(su, init_step),
                        tl.load(
                            W
                            + bp
                            + init_step * (32768 if PACK_W else 512)
                            + 4096 * 2048,
                            cache_modifier=WEIGHT_CACHE,
                        ),
                    )
                copy_scale(
                    Scale + sp + init_step * (2048 if PACK_S else 32) + 4096 * 128,
                    tlx.local_view(st, init_step),
                    SCALE_ASYNC,
                    Scale.to(tl.pointer_type(tl.uint32))
                    + spw
                    + init_step * (512 if PACK_S else 8)
                    + 4096 * 32,
                )
            tlx.async_load_commit_group()
    if MODE == "load":
        crc = tl.zeros((128,), tl.uint32)
        scale_crc = tl.full((), 0, tl.uint32)
    acc = fixed(tl.zeros((16, 64), tl.float32), MMA)
    if FIRST:
        up = fixed(tl.zeros((16, 64), tl.float32), MMA)
    for step in range(4):
        bank = step % DEPTH
        if not PREFETCH_FIRST:
            tlx.async_load_wait_group(0)
            tlx.workgroup_barrier()
        if MODE != "compute" and step + DEPTH - 1 < 4:
            prefetch(
                X,
                W,
                Scale,
                ap,
                bp,
                sp,
                spw,
                step + DEPTH - 1,
                (step + DEPTH - 1) % DEPTH,
                sa,
                sb,
                ss,
                mask,
                FIRST,
                WEIGHT_CACHE,
                SCALE_ASYNC,
                CROSS_LAUNCH_A,
                PACK_W,
                PACK_S,
            )
            if FIRST:
                if WEIGHT_CACHE == "":
                    tlx.async_load(
                        W
                        + bp
                        + (step + DEPTH - 1) * (32768 if PACK_W else 512)
                        + 4096 * 2048,
                        tlx.local_view(su, (step + DEPTH - 1) % DEPTH),
                    )
                else:
                    tlx.local_store(
                        tlx.local_view(su, (step + DEPTH - 1) % DEPTH),
                        tl.load(
                            W
                            + bp
                            + (step + DEPTH - 1) * (32768 if PACK_W else 512)
                            + 4096 * 2048,
                            cache_modifier=WEIGHT_CACHE,
                        ),
                    )
                copy_scale(
                    Scale
                    + sp
                    + (step + DEPTH - 1) * (2048 if PACK_S else 32)
                    + 4096 * 128,
                    tlx.local_view(st, (step + DEPTH - 1) % DEPTH),
                    SCALE_ASYNC,
                    Scale.to(tl.pointer_type(tl.uint32))
                    + spw
                    + (step + DEPTH - 1) * (512 if PACK_S else 8)
                    + 4096 * 32,
                )
            tlx.async_load_commit_group()
        if PREFETCH_FIRST:
            if step + DEPTH - 1 < 4:
                tlx.async_load_wait_group(DEPTH - 1)
            else:
                tlx.async_load_wait_group(0)
            tlx.workgroup_barrier()
        if MODE == "load":
            crc += checksum_words(tlx.local_view(sa, bank)) + checksum_words(
                tlx.local_view(sb, bank)
            )
            scale_crc += checksum_scales(tlx.local_view(ss, bank))
            if FIRST:
                crc += checksum_words(tlx.local_view(su, bank))
                scale_crc += checksum_scales(tlx.local_view(st, bank))
        else:
            for sub in tl.static_range(8):
                av = tlx.local_slice(
                    tlx.local_view(sa, bank), [0, sub * 128], [16, 128]
                )
                a = tlx.local_load(av, layout=A_LAYOUT)
                b = weight_sub(sb, ss, bank, sub)
                acc = tl.dot(a, b, acc)
                if FIRST:
                    up = tl.dot(a, weight_sub(su, st, bank, sub), up)
        tlx.workgroup_barrier()
    tlx.async_load_wait_group(0)
    if MODE == "load":
        tl.store(Checksum + TASK * 128 + tl.arange(0, 128), crc + scale_crc)
    else:
        if FIRST:
            acc = acc / (1.0 + tl.exp(-acc)) * up
        rows = tl.arange(0, 16)
        col = tl.arange(0, 64) + tile * 64
        if MODE == "compute":
            output_slots = block * 16 + rows
            output_mask = rows < count
        else:
            output_slots = tl.load(Sorted + block * 16 + rows)
            output_mask = output_slots < 64
        tl.store(
            Out + output_slots[:, None] * 4096 + col[None, :],
            acc.to(tl.bfloat16),
            output_mask[:, None],
            cache_modifier=".cg" if OUTPUT_AGENT else "",
        )


@triton.jit
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
    GRID: tl.constexpr,
    FIRST: tl.constexpr,
    SKIP: tl.constexpr = -1,
    Seed=None,
    WEIGHT_CACHE: tl.constexpr = "",
    SCALE_ASYNC: tl.constexpr = False,
    CROSS_LAUNCH_A: tl.constexpr = False,
    PACK_W: tl.constexpr = False,
    PACK_S: tl.constexpr = False,
    DEPTH: tl.constexpr = 2,
    PREFETCH_FIRST: tl.constexpr = False,
    MODE: tl.constexpr = "load",
    Checksum=None,
    TASKS: tl.constexpr = 0,
    HITS: tl.constexpr = (),
):
    pid = tl.program_id(0)
    if MODE == "compute":
        allowed = tl.full((), True, tl.int1)
    else:
        allowed = tlx_guard.enabled(Failure, 5) if FIRST else tl.load(Failure) == 0
    ok = allowed & (pid != SKIP)
    tlx_guard.delay(Seed, pid, 0)
    if ok:
        if MODE == "compute":
            tasks = TASKS
        else:
            tasks = tl.load(NumPost) // 16 * 64
        for task in range(pid, tasks, GRID):
            block = task // 64
            if MODE == "compute":
                expert = block
            else:
                expert = tl.load(Experts + block)
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
                WEIGHT_CACHE,
                SCALE_ASYNC,
                CROSS_LAUNCH_A,
                DEPTH=DEPTH,
                PREFETCH_FIRST=PREFETCH_FIRST,
                PACK_W=PACK_W,
                PACK_S=PACK_S,
                MODE=MODE,
                Checksum=Checksum,
                TASK=task,
                HITS=HITS,
            )
    tlx_guard.delay(Seed, pid, 1)
    STATUS_WIDTH: tl.constexpr = (
        1
        if GRID >= 1728
        else 2
        if GRID >= 864
        else 4
        if GRID >= 432
        else 8
        if GRID >= 216
        else 16
    )
    status_ids = pid + GRID * tl.arange(0, STATUS_WIDTH)
    tl.store(Status + status_ids, ok, status_ids < 1728)
    if allowed:
        tlx_guard.finish(Failure, ok, pid, 5)
