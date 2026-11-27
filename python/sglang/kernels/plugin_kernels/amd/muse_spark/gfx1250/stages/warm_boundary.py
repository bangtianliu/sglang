"""Unselected S6 row-boundary experiment with private next-layer warm consumers.

Warm traffic starts only after all eight rows consumed their critical inputs.
The 72-CTA grid must fit concurrently; every consumer owns a 128-byte flag.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import gfx1250

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    chain_guard,
    scoped,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.packed_memory import (
    load_tile_words,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.qkvg import (
    _advance_tag,
)


@g.jit
def finish(
    P,
    Stream,
    Post,
    Pre,
    Alpha,
    Beta,
    Out,
    Updated,
    Status,
    Failure,
    Next,
    Epochs,
    Counter,
    Flags,
    WarmStatus,
    PRE_BF16: gl.constexpr,
    PREFIX: gl.constexpr = 32,
):
    pid = gl.program_id(0)
    tag = _advance_tag(Epochs, pid)
    if pid < 8:
        row = pid
        ids = gl.arange(0, 512, layout=gl.BlockedLayout([1], [32], [4], [0]))
        ok = gl.min(gl.load(Status + ids), 0) == 1
        if ok:
            i = gl.arange(0, 8192, layout=gl.BlockedLayout([1], [32], [4], [0]))
            branch = load_tile_words(P + row * 8192 + i, i >= 0, 8192).to(
                gl.float32, bitcast=True
            )
            branch = branch.to(gl.bfloat16).to(gl.float32)
            branch = (
                (branch * gl.load(Post + i).to(gl.float32))
                .to(gl.bfloat16)
                .to(gl.float32)
            )
            branch *= gl.rsqrt(gl.sum(branch * branch, 0) / 8192 + 1e-8)
            branch = branch.to(gl.bfloat16).to(gl.float32)
            stream = gl.load(Stream + row * 8192 + i)
            updated = gl.load(Alpha + i) * stream + gl.load(Beta + i) * branch
            gl.store(Updated + row * 8192 + i, updated)
            gain = (
                gl.load(Pre + i).to(gl.float32)
                if Pre is not None
                else gl.full((8192,), 1, gl.float32, i.type.layout)
            )
            # A compiler memory clobber and LOADcnt drain precede permission.
            gl.inline_asm_elementwise(
                "s_wait_loadcnt 0\nv_mov_b32 $0,0",
                "=v,~{memory}",
                [],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            )
            last = scoped.arrive_epoch(Counter, tag, 8)
            if last:
                scoped.publish_flags(Flags, tag, 64)
            pre = updated.to(gl.bfloat16).to(gl.float32) if PRE_BF16 else updated
            pre *= gl.rsqrt(gl.sum(pre * pre, 0) / 8192 + 1e-5)
            pre *= gain
            gl.store(Out + row * 8192 + i, pre.to(gl.bfloat16))
        else:
            gl.store(Status + row, 0)
            last = scoped.arrive_epoch(Counter, tag, 8)
            if last:
                scoped.publish_flags(Flags, tag, 64)
        chain_guard.finish_value(Failure, ok, row, 6)
    else:
        gl.inline_asm_elementwise(
            "s_sleep 8\nv_mov_b32 $0,0", "=v", [], dtype=gl.int32, is_pure=False, pack=1
        )
        seen, polls = scoped.wait_flag(
            Flags + (pid - 8) * 32,
            tag,
            10_000_000,
            MAX_SPINS=1_048_576,
            SHADER_CLOCK=True,
        )
        ok = seen == tag
        if ok:
            layout: gl.constexpr = gl.BlockedLayout([4], [32], [4], [0])
            sh: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
            sink = gl.allocate_shared_memory(gl.int32, [2, 4096], sh)
            j = gl.arange(0, 4096, layout=layout)
            for step in range(gl.cdiv(4608 * 8 * PREFIX, 64 * 4096)):
                linear = ((pid - 8) + step * 64) * 4096 + j
                row = linear // (8 * PREFIX)
                split = (linear // PREFIX) % 8
                off = row * 4096 + split * 512 + linear % PREFIX
                gfx1250.async_copy.global_to_shared(
                    sink.index(step % 2),
                    Next.to(gl.pointer_type(gl.int32)) + off,
                    linear < 4608 * 8 * PREFIX,
                    0,
                    cache_modifier="",
                )
                gfx1250.async_copy.commit_group()
                gfx1250.async_copy.wait_group(1)
            gfx1250.async_copy.wait_group(0)
        gl.store(WarmStatus + pid - 8, ok)
