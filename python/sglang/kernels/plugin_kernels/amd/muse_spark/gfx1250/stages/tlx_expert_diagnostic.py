"""T10 same-body expert decomposition; never selected by the chain."""

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
    BL,
    MMA,
    SL,
    copy_scale,
    prefetch,
    weight,
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
    tlx.padded_shared_layout_encoding.with_identity_for([(64, 4)], [1024], [0])
)
SCALE_FLAT = tl.constexpr(
    tlx.shared_linear_layout_encoding([[1], [2], [4], [8], [16], [32]], alignment=16)
)


@triton.jit
def checksum_words(buffer):
    view = tlx.local_reinterpret(buffer, tl.uint32, [1024], layout=FLAT)
    words = tlx.local_load(view, layout=CRC4)
    groups = words.reshape((2, 128, 4))
    return tl.sum(tl.sum(groups, 2), 0)


@triton.jit
def checksum_scales(buffer):
    view = tlx.local_reinterpret(buffer, tl.uint32, [64], layout=SCALE_FLAT)
    words = tlx.local_load(view, layout=CRC4)
    return tl.sum(words, 0)


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
    MODE: tl.constexpr = "full",
    Checksum=None,
    TASK=0,
    HITS: tl.constexpr = (),
):
    row = tl.arange(0, 16)
    ak = tl.arange(0, 128)
    if MODE == "compute":
        count = tl.full((), 0, tl.int32)
        for j in tl.static_range(len(HITS)):
            count = tl.where(block == j, HITS[j], count)
        slots = tl.where(row < count, row, 64)
    else:
        slots = tl.load(Sorted + block * 16 + row)
    input_rows = slots // 8 if FIRST else slots
    ap = tlx.require_layout(input_rows[:, None] * 4096 + ak[None, :], AL)
    mask = tlx.require_layout(tl.broadcast_to(slots[:, None] < 64, (16, 128)), AL)
    bk = tl.arange(0, 64)
    bn = tl.arange(0, 64) + tile * 64
    sk = tl.arange(0, 4)
    sn = tl.arange(0, 64) + tile * 64
    N: tl.constexpr = 8192 if FIRST else 4096
    bp = tlx.require_layout(
        expert.to(tl.int64) * N * 2048 + bn[None, :] * 2048 + bk[:, None], BL
    )
    sp = tlx.require_layout(
        expert.to(tl.int64) * N * 128 + sn[None, :] * 128 + sk[:, None], SL
    )
    spw = expert.to(tl.int64) * N * 32 + sn[None, :] * 32
    ash: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(128, 8)], [16, 128], [1, 0]
    )
    bsh: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(256, 16)], [64, 64], [0, 1]
    )
    ssh: tl.constexpr = tlx.shared_linear_layout_encoding(
        [[1, 0], [2, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], alignment=16
    )
    sa = tlx.local_alloc((16, 128), tl.bfloat16, 2, layout=ash)
    sb = tlx.local_alloc((64, 64), tl.uint8, 2, layout=bsh)
    ss = tlx.local_alloc((4, 64), tl.uint8, 2, layout=ssh)
    if FIRST:
        su = tlx.local_alloc((64, 64), tl.uint8, 2, layout=bsh)
        st = tlx.local_alloc((4, 64), tl.uint8, 2, layout=ssh)
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
        prefetch(
            X,
            W,
            Scale,
            ap,
            bp,
            sp,
            spw,
            0,
            0,
            sa,
            sb,
            ss,
            mask,
            FIRST,
            WEIGHT_CACHE,
            SCALE_ASYNC,
            CROSS_LAUNCH_A,
        )
        if FIRST:
            if WEIGHT_CACHE == "":
                tlx.async_load(W + bp + 4096 * 2048, tlx.local_view(su, 0))
            else:
                tlx.local_store(
                    tlx.local_view(su, 0),
                    tl.load(W + bp + 4096 * 2048, cache_modifier=WEIGHT_CACHE),
                )
            copy_scale(
                Scale + sp + 4096 * 128,
                tlx.local_view(st, 0),
                SCALE_ASYNC,
                Scale.to(tl.pointer_type(tl.uint32)) + spw + 4096 * 32,
            )
        tlx.async_load_commit_group()
    if MODE == "load":
        crc = tl.zeros((128,), tl.uint32)
        scale_crc = tl.full((), 0, tl.uint32)
    acc = fixed(tl.zeros((16, 64), tl.float32), MMA)
    if FIRST:
        up = fixed(tl.zeros((16, 64), tl.float32), MMA)
    for step in range(32):
        bank = step % 2
        tlx.async_load_wait_group(0)
        tlx.workgroup_barrier()
        if MODE != "compute" and step + 1 < 32:
            prefetch(
                X,
                W,
                Scale,
                ap,
                bp,
                sp,
                spw,
                step + 1,
                1 - bank,
                sa,
                sb,
                ss,
                mask,
                FIRST,
                WEIGHT_CACHE,
                SCALE_ASYNC,
                CROSS_LAUNCH_A,
            )
            if FIRST:
                if WEIGHT_CACHE == "":
                    tlx.async_load(
                        W + bp + (step + 1) * 64 + 4096 * 2048,
                        tlx.local_view(su, 1 - bank),
                    )
                else:
                    tlx.local_store(
                        tlx.local_view(su, 1 - bank),
                        tl.load(
                            W + bp + (step + 1) * 64 + 4096 * 2048,
                            cache_modifier=WEIGHT_CACHE,
                        ),
                    )
                copy_scale(
                    Scale + sp + (step + 1) * 4 + 4096 * 128,
                    tlx.local_view(st, 1 - bank),
                    SCALE_ASYNC,
                    Scale.to(tl.pointer_type(tl.uint32)) + spw + (step + 1) + 4096 * 32,
                )
            tlx.async_load_commit_group()
        if MODE == "load":
            crc += checksum_words(tlx.local_view(sa, bank)) + checksum_words(
                tlx.local_view(sb, bank)
            )
            scale_crc += checksum_scales(tlx.local_view(ss, bank))
            if FIRST:
                crc += checksum_words(tlx.local_view(su, bank))
                scale_crc += checksum_scales(tlx.local_view(st, bank))
        else:
            a = tlx.local_load(tlx.local_view(sa, bank), layout=A_LAYOUT)
            b = weight(sb, ss, bank)
            acc = tl.dot(a, b, acc)
            if FIRST:
                up = tl.dot(a, weight(su, st, bank), up)
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
    MODE: tl.constexpr = "full",
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
                MODE=MODE,
                Checksum=Checksum,
                TASK=task,
                HITS=HITS,
            )
    tlx_guard.delay(Seed, pid, 1)
    tl.store(Status + pid, ok)
    if allowed:
        tlx_guard.finish(Failure, ok, pid, 5)
