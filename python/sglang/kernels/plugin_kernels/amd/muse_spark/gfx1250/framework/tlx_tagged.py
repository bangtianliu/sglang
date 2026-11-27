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

"""GPU0 gfx1250 adaptation of TokenSpeed gfx950 tagged-pair mailbox.

One relaxed gpu-scope atomic 64-bit store couples tag and payload. Four pair
loads issue before one wait; a wave retries only stale lanes, sleeping between
attempts. No flag or release/acquire fence. Bounded failure records replace
TokenSpeed's hardware trap: [expected, observed, pair_index, retries] in each
wave's eight-word slot, plus Failure=5. The output is discarded on failure.
"""

import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    ROW1,
)


@triton.jit
def put(ptr, word, tag, mask):
    pair = (tag.to(tl.uint32).to(tl.uint64) << 32) | word.to(tl.uint32).to(tl.uint64)
    # MK.42: on gfx1250 a relaxed gpu-scope atomic *store* is a SCOPE_DEV global_store, visible
    # to another CTA only after ~10-18 us (MK.39). A real RMW atomic is visible in ~1 us.
    tl.atomic_xchg(
        tlx.release_layout(ptr.to(tl.pointer_type(tl.int64))),
        tlx.release_layout(pair.to(tl.int64)),
        mask=tlx.release_layout(mask),
        sem="relaxed",
        scope="gpu",
    )


@triton.constexpr_function
def poll_asm(limit):
    # Pairs and a temporary use LLVM-managed 64-bit operands. No literal
    # VGPR numbers: the enclosing kernel owns register allocation/bank state.
    constraints = ["=&v"] * 12 + ["=&s"] * 4
    constraints += ["v"] * 4 + ["s"] * 4 + ["v"] * 4 + ["~{scc}"]
    lines = ["s_wait_xcnt 0x0", "s_mov_b32 $14, exec_lo", "s_mov_b32 $15, 0", "v_mov_b64 $5, 0"]
    for j in range(4):
        lines += [
            f"v_mov_b64 ${j}, ${20 + j}",
            f"v_lshlrev_b64 ${j}, 32, ${j}",
            f"v_cmp_ne_u32_e64 $13, 0, ${24 + j}",
            "s_and_b32 exec_lo, $14, $13",
            f"global_atomic_or_b64 ${j}, ${16 + j}, $5, off th:TH_ATOMIC_RETURN scope:SCOPE_DEV",
            "s_mov_b32 exec_lo, $14",
        ]
    lines += ["1:", "s_wait_loadcnt 0", "s_wait_xcnt 0x0", "s_mov_b32 $12, 0"]
    for j in range(4):
        lines += [
            f"v_lshrrev_b64 $4, 32, ${j}",
            f"v_cmp_ne_u64_e64 $13, ${20 + j}, $4",
            "s_or_b32 $12, $12, $13",
        ]
    lines += [
        "s_and_b32 $12, $12, $14",
        "s_cmp_eq_u32 $12, 0",
        "s_cbranch_scc1 2f",
        f"s_cmp_ge_u32 $15, {limit}",
        "s_cbranch_scc1 2f",
        "s_add_u32 $15, $15, 1",
        "s_sleep 1",
    ]
    for j in range(4):
        lines += [
            "s_wait_alu depctr_va_vdst(0)",
            f"v_lshrrev_b64 $4, 32, ${j}",
            f"v_cmp_ne_u64_e64 $13, ${20 + j}, $4",
            "s_and_b32 exec_lo, $14, $13",
            f"global_atomic_or_b64 ${j}, ${16 + j}, $5, off th:TH_ATOMIC_RETURN scope:SCOPE_DEV",
            "s_mov_b32 exec_lo, $14",
        ]
    lines += ["s_branch 1b", "2:", "s_mov_b32 exec_lo, $14"]
    lines += [f"v_mov_b32 ${8 + j}, $15" for j in range(4)]
    return "\n".join(lines), ",".join(constraints)


@triton.jit
def poll(ptr, tag, mask, Failure, Record, index, LIMIT: tl.constexpr = 4096):
    tl.static_assert(
        ptr.numel >= 512 and ptr.numel % 512 == 0,
        "pack4 poll requires >=4 defined elements per thread (4 waves)",
    )
    spec: tl.constexpr = poll_asm(LIMIT)
    pairs, temporary, retries, scratch = tl.inline_asm_elementwise(
        spec[0],
        spec[1],
        [
            tlx.release_layout(ptr),
            tag.to(tl.uint32).to(tl.uint64),
            tlx.release_layout(mask.to(tl.int32)),
        ],
        dtype=(tl.uint64, tl.uint64, tl.int32, tl.uint32),
        is_pure=False,
        pack=4,
    )
    lo = pairs.to(tl.uint32)
    hi = (pairs >> 32).to(tl.uint32)
    late = tlx.release_layout(mask & (hi != tag.to(tl.uint32)))
    record_ptr = tlx.release_layout(Record)
    tl.atomic_xchg(
        Failure + tl.zeros(index.shape, tl.int32),
        5,
        mask=late,
        sem="relaxed",
        scope="gpu",
    )
    previous = tl.atomic_xchg(
        record_ptr, tag.to(tl.int32), mask=late, sem="relaxed", scope="gpu"
    )
    owner = late & (previous != tag.to(tl.int32))
    tl.store(record_ptr + 1, tlx.release_layout(hi.to(tl.int32)), mask=owner)
    tl.store(record_ptr + 2, tlx.release_layout(index), mask=owner)
    tl.store(record_ptr + 3, tlx.release_layout(retries), mask=owner)
    return lo, retries


@triton.jit
def pre_poll(Pairs, Sorted, block, step, tag, Failure, Record, LIMIT: tl.constexpr):
    lane = tlx.require_layout(tl.arange(0, 512), ROW1)
    first = tl.load(Sorted + block * 16)
    # Eight producer hints for this K512 chunk, no extra flag. Other waves
    # issue no memory operation; the actual payload poll still checks all tags.
    index = first * 2048 + step * 256 + (lane % 8) * 32 + 31
    rec = Record + (tl.program_id(0) * 4 + (lane // 32) % 4) * 8
    _, retries = poll(Pairs + index, tag, lane < 8, Failure, rec, index, LIMIT)
    tlx.workgroup_barrier()
    return tl.max(retries, 0)
