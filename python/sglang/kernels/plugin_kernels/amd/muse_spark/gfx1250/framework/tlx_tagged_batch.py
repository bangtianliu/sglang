# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""T13 bounded full-chunk poll:32pairs/thread before one wait, no padding."""

import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    BlockedEncoding,
)

SMALL = tl.constexpr(BlockedEncoding([1, 1], [2, 16], [4, 1], [1, 0]))


@triton.constexpr_function
def batch_asm(limit):
    p = 4
    groups = 8
    count = p * groups
    # pairs32, temporary4, retry4, scalar scratch4; ptr32,tag4,mask4.
    constraints = (
        ["=&v"] * (count + 8)
        + ["=&s"] * 4
        + ["v"] * count
        + ["s"] * 4
        + ["v"] * 4
        + ["~{scc}"]
    )
    lines = ["s_wait_xcnt 0x0", "s_mov_b32 $42, exec_lo", "s_mov_b32 $43, 0"]
    for j in range(count):
        lines += [
            f"v_mov_b64 ${j}, ${76 + j % 4}",
            f"v_lshlrev_b64 ${j}, 32, ${j}",
            f"v_cmp_ne_u32_e64 $41, 0, ${80 + j % 4}",
            "s_and_b32 exec_lo, $42, $41",
            f"global_load_b64 ${j}, ${44 + j}, off scope:SCOPE_DEV",
            "s_mov_b32 exec_lo, $42",
        ]
    lines += ["1:", "s_wait_loadcnt 0", "s_wait_xcnt 0x0", "s_mov_b32 $40, 0"]
    for j in range(count):
        lines += [
            f"v_lshrrev_b64 $32, 32, ${j}",
            f"v_cmp_ne_u64_e64 $41, ${76 + j % 4}, $32",
            "s_or_b32 $40, $40, $41",
        ]
    lines += [
        "s_and_b32 $40, $40, $42",
        "s_cmp_eq_u32 $40, 0",
        "s_cbranch_scc1 2f",
        f"s_cmp_ge_u32 $43, {limit}",
        "s_cbranch_scc1 2f",
        "s_add_u32 $43, $43, 1",
        "s_sleep 1",
    ]
    for j in range(count):
        lines += [
            "s_wait_alu depctr_va_vdst(0)",
            f"v_lshrrev_b64 $32, 32, ${j}",
            f"v_cmp_ne_u64_e64 $41, ${76 + j % 4}, $32",
            "s_and_b32 exec_lo, $42, $41",
            f"global_load_b64 ${j}, ${44 + j}, off scope:SCOPE_DEV",
            "s_mov_b32 exec_lo, $42",
        ]
    lines += ["s_branch 1b", "2:", "s_mov_b32 exec_lo, $42"]
    lines += [f"v_mov_b32 ${36 + j}, $43" for j in range(4)]
    return "\n".join(lines), ",".join(constraints)


@triton.jit
def cat_columns(x, y, N: tl.constexpr):
    return tl.permute(tl.join(x, y), (0, 2, 1)).reshape((16, N * 2))


@triton.jit
def poll_chunk(Pairs, Rows, step, tag, Failure, Record, LIMIT: tl.constexpr):
    row = tl.arange(0, 16)
    col = tl.arange(0, 32)
    mask = tlx.require_layout(tl.broadcast_to((Rows < 64)[:, None], (16, 32)), SMALL)
    indices = ()
    pointers = ()
    for group in tl.static_range(8):
        index = tlx.require_layout(
            Rows[:, None] * 2048 + step * 256 + group * 32 + col[None, :], SMALL
        )
        indices += (index,)
        pointers += (tlx.release_layout(Pairs + index),)
    spec: tl.constexpr = batch_asm(LIMIT)
    raw = tl.inline_asm_elementwise(
        spec[0],
        spec[1],
        [
            pointers[0],
            pointers[1],
            pointers[2],
            pointers[3],
            pointers[4],
            pointers[5],
            pointers[6],
            pointers[7],
            tag.to(tl.uint32).to(tl.uint64),
            tlx.release_layout(mask.to(tl.int32)),
        ],
        dtype=(tl.uint64,) * 9 + (tl.int32, tl.uint32),
        is_pure=False,
        pack=4,
    )
    rec = (
        Record
        + (tl.program_id(0) * 4 + (row // 2) % 4)[:, None] * 8
        + tl.zeros((16, 32), tl.int32)
    )
    values = ()
    for group in tl.static_range(8):
        pair = raw[group]
        late = tlx.release_layout(
            mask & ((pair >> 32).to(tl.uint32) != tag.to(tl.uint32))
        )
        tl.atomic_xchg(
            Failure + tl.zeros((16, 32), tl.int32),
            5,
            mask=late,
            sem="relaxed",
            scope="gpu",
        )
        previous = tl.atomic_xchg(
            rec, tag.to(tl.int32), mask=late, sem="relaxed", scope="gpu"
        )
        owner = late & (previous != tag.to(tl.int32))
        tl.store(rec + 1, tlx.release_layout((pair >> 32).to(tl.int32)), mask=owner)
        tl.store(rec + 2, tlx.release_layout(indices[group]), mask=owner)
        tl.store(rec + 3, tlx.release_layout(raw[9]), mask=owner)
        values += (pair.to(tl.uint32),)
    a = cat_columns(values[0], values[1], 32)
    b = cat_columns(values[2], values[3], 32)
    c = cat_columns(values[4], values[5], 32)
    d = cat_columns(values[6], values[7], 32)
    return cat_columns(cat_columns(a, b, 64), cat_columns(c, d, 64), 128), raw[9]
