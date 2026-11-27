"""GPU0 TLX scoped handoff plus live-register XCNT regression; bounded polls.

Never executes an unsafe no-XCNT variant, trap, or multi-GPU operation.
"""

import json
from pathlib import Path

import torch
import triton
import triton.language as tl
import utlx_plugin as tlx
from layer_fixture import require_gpu0
from test_scoped_xcnt import pressure as gluon_pressure
from tlx_ir import dump
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped

LOG = Path("/home/bangtliu/coord/logs")


@triton.jit
def store_words(ptr, value, mask):
    return tl.inline_asm_elementwise(
        "v_cmp_ne_u32_e64 $0, 0, $3\ns_and_saveexec_b32 $0, $0\n"
        "global_store_b32 $1, $2, off scope:SCOPE_DEV\ns_mov_b32 exec_lo, $0\ns_wait_xcnt 0x0",
        constraints="=&s,v,v,v,~{scc}",
        args=[ptr, value, mask.to(tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def load_words(ptr):
    return tl.inline_asm_elementwise(
        "global_load_b32 $0, $1, off scope:SCOPE_DEV\ns_wait_loadcnt 0\ns_wait_xcnt 0x0",
        constraints="=v,v",
        args=[ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def flag_value(flag):
    lanes = tl.arange(0, 128)
    saved, value = tl.inline_asm_elementwise(
        "v_mov_b32 $1, 0\nv_cmp_ne_u32_e64 $0, 0, $3\ns_and_saveexec_b32 $0, $0\n"
        "global_load_b32 $1, $2, off scope:SCOPE_DEV\ns_wait_loadcnt 0\ns_mov_b32 exec_lo, $0\ns_wait_xcnt 0x0",
        constraints="=&s,=&v,v,v,~{scc}",
        args=[flag + tl.full((128,), 0, tl.int32), (lanes == 0).to(tl.int32)],
        dtype=(tl.int32, tl.int32),
        is_pure=False,
        pack=1,
    )
    return tl.sum(value, 0)


@triton.jit
def shader_clock():
    return tl.inline_asm_elementwise(
        "s_get_shader_cycles_u64 $0",
        constraints="=s",
        args=[],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def handoff(Payload, Flag, Results, Polls, Epoch, SKIP: tl.constexpr = False):
    tag = tl.load(Epoch)
    i = tl.arange(0, 1024)
    if tl.program_id(0) == 0:
        if not SKIP:
            store_words(Payload + i, i + tag, tl.full((1024,), True, tl.int1))
            tl.inline_asm_elementwise(
                "s_wait_storecnt 0\nv_mov_b32 $0,0",
                constraints="=v",
                args=[],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            tlx.workgroup_barrier()
            lanes = tl.arange(0, 128)
            store_words(
                Flag + tl.full((128,), 0, tl.int32),
                tl.full((128,), tag, tl.int32),
                lanes == 0,
            )
    else:
        start = shader_clock()
        elapsed = tl.full((), 0, tl.uint64)
        seen = flag_value(Flag)
        spins = 1
        while (seen != tag) & (spins < 4096) & (elapsed < 1_000_000):
            seen = flag_value(Flag)
            spins += 1
            if spins % 64 == 0:
                elapsed = shader_clock() - start
        tlx.workgroup_barrier()
        values = load_words(Payload + i)
        good = tl.sum((values == i + tag).to(tl.int32), 0) == 1024
        tl.store(Results + tag - 1, good & (seen == tag))
        tl.store(Polls + tag - 1, spins)


@g.jit
def reference_handoff(Payload, Flag, Results, Polls, Epoch):
    tag = gl.load(Epoch)
    i = gl.arange(0, 1024, layout=gl.BlockedLayout([1], [32], [4], [0]))
    if gl.program_id(0) == 0:
        scoped.store_words(Payload + i, i + tag, gl.full_like(i, True, dtype=gl.int1))
        scoped.publish(Flag, tag)
    else:
        seen, spins = scoped.wait_flag(
            Flag, tag, 1_000_000, MAX_SPINS=4096, SHADER_CLOCK=True
        )
        values = scoped.load_words(Payload + i)
        gl.store(
            Results + tag - 1,
            (gl.sum((values == i + tag).to(gl.int32), 0) == 1024) & (seen == tag),
        )
        gl.store(Polls + tag - 1, spins)


@triton.jit
def pressure(X, Scratch, Out, ROUNDS: tl.constexpr):
    row = tl.arange(0, 16)
    col = tl.arange(0, 2048)
    index = tl.program_id(0) * 32768 + row[:, None] * 2048 + col[None, :]
    value = tl.load(X + index)
    mask = (col[None, :] % 17 != 0) & (row[:, None] >= 0)
    for step in range(ROUNDS):
        store_words(Scratch + step * 8 * 32768 + index, value, mask)
        value = value * 1664525 + 1013904223
    tl.inline_asm_elementwise(
        "s_wait_storecnt 0\nv_mov_b32 $0,0",
        constraints="=v",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    tlx.workgroup_barrier()
    for step in range(ROUNDS):
        tl.store(
            Out + step * 8 * 32768 + index,
            load_words(Scratch + step * 8 * 32768 + index),
        )


def main():
    require_gpu0()
    results = {}
    for label, fn in [("gluon", reference_handoff), ("tlx", handoff)]:
        payload = torch.empty((1024,), device="cuda", dtype=torch.int32)
        flag = torch.zeros((32,), device="cuda", dtype=torch.int32)
        out = torch.full((1001,), -1, device="cuda", dtype=torch.int32)
        polls = torch.empty_like(out)
        epoch = torch.ones((1,), device="cuda", dtype=torch.int32)
        k = fn.warmup(
            payload,
            flag,
            out,
            polls,
            epoch,
            grid=(2,),
            num_warps=4,
            enable_fp_fusion=False,
        )
        r = dump(k, "handoff", label)
        assert r["occupancy"] >= 1 and 2 <= 192 * r["occupancy"]
        flag.zero_()
        out.fill_(-1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for tag in range(1, 1001):
                epoch.fill_(tag)
                fn[(2,)](
                    payload,
                    flag,
                    out,
                    polls,
                    epoch,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        graph.replay()
        torch.cuda.synchronize()
        assert bool(out[:1000].eq(1).all()), dict(
            tensor="handoff",
            side=label,
            rank=0,
            location=out[:1000].ne(1).nonzero()[:1].tolist(),
        )
        record = dict(
            resources=r, launches=1000, mismatches=0, max_polls=int(polls[:1000].max())
        )
        if label == "tlx":
            epoch.fill_(1001)
            fn[(2,)](
                payload,
                flag,
                out,
                polls,
                epoch,
                SKIP=True,
                num_warps=4,
                enable_fp_fusion=False,
            )
            torch.cuda.synchronize()
            assert (
                int(out[1000]) == 0
                and 1 <= int(polls[1000]) <= 4096
                and int(flag[0]) == 1000
            )
            record["missing_producer"] = dict(
                rejected=True, stale_tag=1000, polls=int(polls[1000])
            )
        results[label] = record
    rounds = 16
    x = torch.arange(8 * 32768, device="cuda", dtype=torch.int32)
    expected = []
    value = x.reshape(8, 16, 2048)
    mask = torch.arange(2048, device="cuda") % 17 != 0
    for _ in range(rounds):
        expected.append(torch.where(mask, value, -1))
        value = value * 1664525 + 1013904223
    expected = torch.stack(expected)
    for label, fn in [("gluon", gluon_pressure), ("tlx", pressure)]:
        scratch = torch.full(
            (rounds, 8, 16, 2048), -1, device="cuda", dtype=torch.int32
        )
        out = torch.empty_like(scratch)
        k = fn.warmup(
            x, scratch, out, rounds, grid=(8,), num_warps=16, enable_fp_fusion=False
        )
        r = dump(k, "xcnt", label)
        assert r["occupancy"] >= 1
        fn[(8,)](x, scratch, out, rounds, num_warps=16, enable_fp_fusion=False)
        torch.cuda.synchronize()
        wrong = int(out.ne(expected).sum())
        assert wrong == 0, dict(
            tensor="XCNT payload",
            side=label,
            rank=0,
            location=out.ne(expected).nonzero()[:1].tolist(),
            mismatches=wrong,
        )
        assert "s_wait_xcnt" in k.asm["amdgcn"] and "scope:SCOPE_DEV" in k.asm["amdgcn"]
        results[label]["xcnt"] = dict(
            words=out.numel(), rounds=rounds, mismatches=wrong, resources=r
        )
    (LOG / "phaseT0-scoped.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
