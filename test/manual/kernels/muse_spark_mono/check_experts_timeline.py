"""GPU0-only S5 timeline and independent W13/W2 bounds; never a production path.

Detailed marks consume/reduce the previous result and wait its scoreboard before
reading realtime. They serialize and perturb the pipeline. Use unmarked bounds
for latency conclusions; detailed durations include reduction/timer overhead.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from bench_hbm import graph_times
from check_experts import cases, load
from layer_fixture import require_gpu0
from profile_s2 import calibrate
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import gfx1250

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped, sync
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    experts_routed as stage,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.experts import (
    _fp4_upcast,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.qkvg import (
    _advance_tag,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.skinny_gemm import _mma

ROOT = Path("/home/bangtliu/coord/logs")


@g.jit
def _mark(value):
    flat = value.to(gl.int32).reshape((value.numel,))
    dependency = gl.sum(flat, 0)
    gl.barrier()
    return gl.inline_asm_elementwise(
        "// dependency $1\ns_wait_loadcnt 0\ns_wait_dscnt 0\ns_wait_alu 0\n"
        "s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)\ns_wait_kmcnt 0",
        "=s,v",
        [dependency],
        dtype=gl.uint64,
        is_pure=False,
        pack=1,
    )


@g.jit
def _gemm(
    Cached,
    W,
    Scale,
    block,
    expert,
    tile,
    ROWS: gl.constexpr,
    Trace,
    base,
    DETAIL: gl.constexpr,
):
    if DETAIL:
        mma: gl.constexpr = _mma(16, 64)
        acc = gl.full((4, 16, 64), 0, gl.float32, mma)
        begin = sync.clock()
        p0, s0 = stage._weights(W, Scale, expert, tile, 0, ROWS, 64, 64, 4)
        p1, s1 = stage._weights(W, Scale, expert, tile, 1, ROWS, 64, 64, 4)
        p2, s2 = stage._weights(W, Scale, expert, tile, 2, ROWS, 64, 64, 4)
        load_time = gl.full((), 0, gl.uint64, gl.BlockedLayout([1], [32], [16], [0]))
        input_time = load_time
        convert_time = load_time
        mma_time = load_time
        first = begin
        for step in range(16):
            t0 = sync.clock()
            t1 = _mark(p0)
            if step == 0:
                first = t1
            a = stage._hit_tile(Cached, step, 64, 64)
            t2 = _mark(a)
            b = _fp4_upcast(p0, s0, 64, 64, 4, "direct")
            t3 = _mark(b)
            acc = gfx1250.wmma(a, b, acc)
            t4 = _mark(acc)
            load_time += t1 - t0
            input_time += t2 - t1
            convert_time += t3 - t2
            mma_time += t4 - t3
            p0 = p1
            s0 = s1
            p1 = p2
            s1 = s2
            if step + 3 < 16:
                p2, s2 = stage._weights(
                    W, Scale, expert, tile, step + 3, ROWS, 64, 64, 4
                )
        gl.store(Trace + base + 0, first)
        gl.store(Trace + base + 1, load_time)
        gl.store(Trace + base + 2, input_time)
        gl.store(Trace + base + 3, convert_time)
        gl.store(Trace + base + 4, mma_time)
        return gl.sum(acc, 0)
    else:
        return stage._routed_gemm(
            Cached, W, W, Scale, block, expert, tile, ROWS, False, 64, 64
        )


@g.jit
def _run(
    X,
    Sorted,
    Experts,
    NumPost,
    W13,
    S13,
    W2,
    S2,
    Inter,
    Out,
    Epochs,
    Arrivals,
    Flags,
    Status,
    Trace,
    MODE: gl.constexpr,
    LEVEL: gl.constexpr,
):
    cta = gl.program_id(0)
    tasks = gl.load(NumPost) // 16 * 64
    tag = _advance_tag(Epochs, cta)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [16], [0])
    ok = gl.full((), 1, gl.int32, layout)
    for j in range(cta, tasks * (2 if MODE == 0 else 1), 192):
        task = j + (tasks if MODE == 2 else 0)
        block = (task % tasks) // 64
        tile = task % 64
        expert = gl.load(Experts + block)
        base = task * 32
        if LEVEL:
            start = sync.clock()
            gl.store(Trace + base, start)
            gl.store(Trace + base + 1, cta)
        ready = gl.full((), True, gl.int1, layout)
        if task >= tasks and MODE == 0:
            seen, polls = scoped.wait_flag(
                Flags + (block * 64 + tile) * 32, tag, 100_000_000, SHADER_CLOCK=True
            )
            ready = seen == tag
        if LEVEL:
            gl.store(Trace + base + 2, sync.clock())
        if ready:
            if task < tasks:
                cached = stage._cache_hits(X, Sorted, block, False)
                if LEVEL:
                    gl.store(Trace + base + 3, sync.clock())
                gate = _gemm(
                    cached,
                    W13,
                    S13,
                    block,
                    expert,
                    tile,
                    8192,
                    Trace,
                    base + 8,
                    LEVEL == 2,
                )
                up = _gemm(
                    cached,
                    W13,
                    S13,
                    block,
                    expert,
                    tile + 64,
                    8192,
                    Trace,
                    base + 16,
                    LEVEL == 2,
                )
                value = gate / (1.0 + gl.exp(-gate)) * up
                if LEVEL:
                    gl.store(
                        Trace + base + 4, _mark(value) if LEVEL == 2 else sync.clock()
                    )
                stage._store_slots(Inter, Sorted, block, tile, value, 64)
                scoped.drain_stores()
                if LEVEL:
                    gl.store(Trace + base + 5, sync.clock())
                if MODE == 0:
                    ticket = _advance_tag(Arrivals, block)
                    if ticket % 64 == 0:
                        scoped.publish_flags(Flags + block * 64 * 32, tag, 64)
            else:
                cached = stage._cache_hits(Inter, Sorted, block, True)
                if LEVEL:
                    gl.store(Trace + base + 3, sync.clock())
                value = _gemm(
                    cached,
                    W2,
                    S2,
                    block,
                    expert,
                    tile,
                    4096,
                    Trace,
                    base + 8,
                    LEVEL == 2,
                )
                if LEVEL:
                    gl.store(
                        Trace + base + 4, _mark(value) if LEVEL == 2 else sync.clock()
                    )
                stage._store_slots(Out, Sorted, block, tile, value, 64)
                scoped.drain_stores()
                if LEVEL:
                    gl.store(Trace + base + 5, sync.clock())
        ok = ok & ready
        if LEVEL:
            gl.store(Trace + base + 6, sync.clock())
    gl.store(Status + cta, ok)


def main():
    require_gpu0()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rate = calibrate()
    (ROOT / f"phase3e1d-clock-{args.seed}.json").write_text(
        json.dumps(dict(ticks_per_us=rate))
    )
    x, w = load(args.seed)
    ids = cases(args.seed, x.device)["typical"]
    table = stage_table(ids)
    n = table[2].item() // 16 * 64
    valid = ids.flatten() < 64
    ws = stage.make_workspace(x.device)
    stage.experts(x, *table, w["w13"], w["w13_scale"], w["w2"], w["w2_scale"], ws)
    torch.cuda.synchronize()
    assert ws.status.eq(1).all()
    expected_inter = ws.inter.clone()
    expected_output = ws.output.clone()
    trace = torch.zeros((2 * n, 32), device=x.device, dtype=torch.uint64)
    results = []
    for mode in (0, 1, 2):
        if mode == 2:
            # Synthetic intermediate: independent of W13, same shape/dtype/amplitude.
            gen = torch.Generator(device=x.device).manual_seed(args.seed + 921)
            ws.inter[:64].copy_(
                torch.randn((64, 4096), device=x.device, generator=gen)
                .bfloat16()
                .float()
            )
            # Compare isolated W2 to the original core, with the identical synthetic input.
            synthetic = ws.inter.clone()
        for level in (0, 1, 2):
            if mode != 2:
                ws.inter.copy_(expected_inter)
            call = lambda: _run[(192,)](
                x,
                *table,
                w["w13"],
                w["w13_scale"],
                w["w2"],
                w["w2_scale"],
                ws.inter,
                ws.output,
                ws.epochs,
                ws.arrivals,
                ws.flags,
                ws.status,
                trace,
                MODE=mode,
                LEVEL=level,
                num_warps=16,
                enable_fp_fusion=False,
            )
            kernel = call()
            torch.cuda.synchronize()
            assert ws.status.eq(1).all()
            if mode < 2:
                assert torch.equal(ws.inter[:64][valid], expected_inter[:64][valid])
            if mode == 0:
                assert torch.equal(ws.output[valid], expected_output[valid])
            if mode == 2 and level == 0:
                synthetic_result = ws.output.clone()
            if mode == 2:
                assert torch.equal(ws.inter, synthetic)
                assert torch.equal(ws.output[valid], synthetic_result[valid])
            timing = graph_times(call, samples=30)
            assert ws.status.eq(1).all()
            filename = f"phase3e1d-timeline-{args.seed}-m{mode}-l{level}"
            if level:
                np.save(ROOT / (filename + ".npy"), trace.cpu().numpy())
            (ROOT / (filename + ".amdgcn")).write_text(kernel.asm["amdgcn"])
            record = dict(
                ticks_per_us=rate,
                seed=args.seed,
                mode=mode,
                level=level,
                tasks=n * (2 if mode == 0 else 1),
                time=timing,
                registers=kernel.n_regs,
                spills=kernel.n_spills,
                shared=kernel.metadata.shared,
            )
            print(json.dumps(record), flush=True)
            results.append(record)
    (ROOT / f"phase3e1d-timeline-{args.seed}.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


def stage_table(ids):
    from sglang.srt.rocm_muse_spark.muse_moe_triton import moe_align_block_size_small

    return moe_align_block_size_small(ids, 64, 0, 16)


if __name__ == "__main__":
    main()
