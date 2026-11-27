"""GPU0 A16W4 conversion throughput: RAW chains and full WMMA-operand layouts.

Fits timing slopes over instruction/iteration counts, removing fixed timer and
sink costs. RAW independent streams are independent dependency chains; they
retain one true recurrence per stream. No activation quantization or GEMM math.
"""

import json
from pathlib import Path

import numpy as np
import torch
from bench_hbm import graph_times
from layer_fixture import require_gpu0
from profile_s2 import calibrate
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna5

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import sync
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.experts import (
    _fp4_upcast,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.skinny_gemm import _mma

ROOT = Path("/home/bangtliu/coord/logs")
VALUES19 = 19 * (8192 * 4096 + 4096 * 4096)


@g.constexpr_function
def _raw_asm(streams, repeats):
    n = 4 * streams
    constraints = [f"=&{{v{4 + i}}}" for i in range(n)] + ["=&s", "v", "~{scc}"]
    asm = [f"v_mov_b32 v{4 + 4 * s}, ${n + 1}" for s in range(streams)]
    if repeats < 16:
        asm += [
            line
            for _ in range(repeats if repeats < 16 else 1)
            for st in range(streams)
            for line in (
                f"s_wait_alu depctr_va_vdst({streams - 1})",
                f"v_cvt_scale_pk8_bf16_fp4 v[{4 + 4 * st}:{7 + 4 * st}], v{4 + 4 * st}, 0x7f7f7f7f",
            )
        ]
        asm += [f"s_mov_b32 ${n}, 0"]
    else:
        asm += [f"s_mov_b32 ${n}, {repeats}", "1:", ".rept 16"]
        asm += [
            line
            for _ in range(repeats if repeats < 16 else 1)
            for st in range(streams)
            for line in (
                f"s_wait_alu depctr_va_vdst({streams - 1})",
                f"v_cvt_scale_pk8_bf16_fp4 v[{4 + 4 * st}:{7 + 4 * st}], v{4 + 4 * st}, 0x7f7f7f7f",
            )
        ]
        asm += [
            ".endr",
            f"s_sub_u32 ${n}, ${n}, 16",
            f"s_cmp_gt_u32 ${n}, 0",
            "s_cbranch_scc1 1b",
        ]
    asm += ["s_wait_alu depctr_va_vdst(0)"]
    return "\n".join(asm), ",".join(constraints)


@g.jit
def raw(
    Out,
    Trace,
    STREAMS: gl.constexpr,
    REPEATS: gl.constexpr,
    CHECK: gl.constexpr = False,
):
    gl.static_assert(REPEATS > 0 and (REPEATS < 16 or REPEATS % 16 == 0))
    lanes = gl.arange(0, 128, layout=gl.BlockedLayout([1], [32], [4], [0]))
    packed = gl.full((128,), 0x76543210, gl.uint32, lanes.type.layout)
    if not CHECK:
        packed = packed ^ lanes.to(gl.uint32)
    asm: gl.constexpr = _raw_asm(STREAMS, REPEATS)
    rt0 = sync.clock()
    cy0 = sync.shader_cycles()
    result = gl.inline_asm_elementwise(
        asm[0],
        asm[1],
        [packed],
        dtype=(gl.uint32,) * (4 * STREAMS + 1),
        is_pure=False,
        pack=1,
    )
    cy1 = sync.shader_cycles()
    rt1 = sync.clock()
    for component in gl.static_range(4 * STREAMS):
        gl.store(
            Out + gl.program_id(0) * (4 * STREAMS * 128) + component * 128 + lanes,
            result[component],
        )
    gl.store(Trace + gl.program_id(0) * 4, cy0)
    gl.store(Trace + gl.program_id(0) * 4 + 1, cy1)
    gl.store(Trace + gl.program_id(0) * 4 + 2, rt0)
    gl.store(Trace + gl.program_id(0) * 4 + 3, rt1)


@g.jit
def layout_probe(P, Scale, Sink, Trace, ITER: gl.constexpr, PRELAYOUT: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 8, 1], [1, 4, 8], [4, 1, 4], [1, 2, 0])
    ss = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    kk = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    nn = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    index = ss[:, None, None] * 2048 + kk[None, :, None] * 64 + nn[None, None, :]
    p = gl.load(P + index)
    scale = gl.load(
        Scale
        + ss[:, None, None] * 128
        + (kk[None, :, None] // 16) * 64
        + nn[None, None, :]
    )
    mma: gl.constexpr = _mma(16, 64)
    operand: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    if PRELAYOUT:
        p = gl.convert_layout(p, gl.DotOperandLayout(1, mma, 4))
        expanded = gl.join(scale, scale).permute((0, 1, 3, 2)).reshape((4, 64, 64))
        expanded = gl.convert_layout(expanded, operand)
    accum = gl.full((4, 64, 64), 0, gl.uint16, operand)
    rt0 = sync.clock()
    cy0 = sync.shader_cycles()
    # This layout proxy includes source mutation and an observable XOR sink.
    # Treat its extrapolation as a current-layout estimate, not a pure ISA floor.
    for step in range(ITER):
        packed = (p ^ step).to(gl.uint8)
        if PRELAYOUT:
            y = cdna5.scaled_upcast(
                packed, (expanded ^ (step & 1)).to(gl.uint8), gl.bfloat16, axis=1
            )
            y = gl.convert_layout(y, operand)
        else:
            y = _fp4_upcast(
                packed, (scale ^ (step & 1)).to(gl.uint8), 64, 64, 4, "direct"
            )
        accum = accum ^ y.to(gl.uint16, bitcast=True)
    gl.store(
        Sink + gl.program_id(0), gl.sum(gl.sum(gl.sum(accum.to(gl.uint32), 0), 0), 0)
    )
    cy1 = sync.shader_cycles()
    rt1 = sync.clock()
    gl.store(Trace + gl.program_id(0) * 4, cy0)
    gl.store(Trace + gl.program_id(0) * 4 + 1, cy1)
    gl.store(Trace + gl.program_id(0) * 4 + 2, rt0)
    gl.store(Trace + gl.program_id(0) * 4 + 3, rt1)


def measure(call, trace):
    compiled = call()
    torch.cuda.synchronize()
    values = trace.cpu().to(torch.int64)
    return compiled, dict(
        cycles=float((values[:, 1] - values[:, 0]).double().median()),
        realtime=float((values[:, 3] - values[:, 2]).double().median()),
        **graph_times(call, samples=30),
    )


def main():
    require_gpu0()
    rate = calibrate()
    rows = []
    fits = []
    props = torch.cuda.get_device_properties(0)
    assert props.multi_processor_count == 192 and props.warp_size == 32
    topology = Path("/sys/class/kfd/kfd/topology/nodes/2/properties").read_text()
    assert "simd_count 768\n" in topology and "simd_per_cu 4\n" in topology
    (ROOT / "phase3e1b-gpu0-topology.txt").write_text(topology)
    trace = torch.empty((192, 4), device="cuda", dtype=torch.uint64)
    output = torch.empty(192 * 128 * 32, device="cuda", dtype=torch.uint32)
    lut = (
        torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=torch.bfloat16,
        )
        .view(torch.uint16)
        .long()
        .tolist()
    )
    for streams in (1, 2, 4, 8):
        for repeats in (1, 2, 4):
            value = 0x76543210
            for _ in range(repeats):
                expected = [
                    lut[(value >> (8 * j)) & 15]
                    | (lut[(value >> (8 * j + 4)) & 15] << 16)
                    for j in range(4)
                ]
                value = expected[0]
            raw[(1,)](output, trace, streams, repeats, True, num_warps=4)
            torch.cuda.synchronize()
            got = output[: 4 * streams * 128].reshape(4 * streams, 128).cpu().long()
            want = (
                torch.tensor(expected * streams).reshape(4 * streams, 1).expand_as(got)
            )
            assert torch.equal(got, want), (
                streams,
                repeats,
                got[:, 0].tolist(),
                want[:, 0].tolist(),
            )
    print(
        "raw conversion functional PASS:12 dependent-chain/independent-stream configurations",
        flush=True,
    )
    for streams in (1, 2, 4, 8):
        group = []
        for repeats in (64, 256, 1024, 4096):
            call = lambda: raw[(192,)](output, trace, streams, repeats, num_warps=4)
            k, r = measure(call, trace)
            r.update(
                kind="raw",
                streams=streams,
                repeats=repeats,
                registers=k.n_regs,
                spills=k.n_spills,
            )
            rows.append(r)
            group.append(r)
            print(json.dumps(r), flush=True)
            if repeats == 4096:
                (ROOT / f"phase3e1b-ceiling-raw-{streams}.amdgcn").write_text(
                    k.asm["amdgcn"]
                )
        x = np.array([r["repeats"] for r in group])
        cy = np.array([r["cycles"] for r in group])
        rt = np.array([r["realtime"] for r in group])
        us = np.array([r["median_us"] for r in group])
        cyc_slope = float(np.polyfit(x, cy, 1)[0])
        rt_slope = float(np.polyfit(x, rt, 1)[0])
        event_slope = float(np.polyfit(x, us, 1)[0])
        assert cyc_slope > 0 and rt_slope > 0 and event_slope > 0
        clock = cyc_slope / rt_slope * rate
        wave_ops = streams / cyc_slope
        aggregate = 192 * 4 * 32 * 8 * streams / event_slope
        f = dict(
            kind="raw",
            streams=streams,
            cycles_per_repeat=cyc_slope,
            shader_cycles_per_us=clock,
            wave_instructions_per_cycle=wave_ops,
            per_SIMD_assuming_one_wave_per_SIMD=wave_ops,
            aggregate_values_per_us=aggregate,
            conversion_19_experts_us=VALUES19 / aggregate,
        )
        fits.append(f)
        print(json.dumps(f), flush=True)
    p = torch.randint(0, 256, (8192,), device="cuda", dtype=torch.uint8)
    sc = torch.randint(118, 123, (8192,), device="cuda", dtype=torch.uint8)
    sink = torch.empty(192, device="cuda", dtype=torch.int32)
    for pre in (True, False):
        group = []
        for iterations in (16, 64, 256, 1024):
            call = lambda: layout_probe[(192,)](
                p,
                sc,
                sink,
                trace,
                iterations,
                pre,
                num_warps=16,
                enable_fp_fusion=False,
            )
            k, r = measure(call, trace)
            r.update(
                kind="layout",
                prelayout=pre,
                iterations=iterations,
                registers=k.n_regs,
                spills=k.n_spills,
                shared=k.metadata.shared,
            )
            rows.append(r)
            group.append(r)
            print(json.dumps(r), flush=True)
            if iterations == 1024:
                (ROOT / f"phase3e1b-ceiling-layout-{pre}.amdgcn").write_text(
                    k.asm["amdgcn"]
                )
        event_slope = float(
            np.polyfit(
                [r["iterations"] for r in group], [r["median_us"] for r in group], 1
            )[0]
        )
        cy_slope = float(
            np.polyfit(
                [r["iterations"] for r in group], [r["cycles"] for r in group], 1
            )[0]
        )
        throughput = 192 * 16384 / event_slope
        f = dict(
            kind="layout",
            prelayout=pre,
            cycles_per_iteration=cy_slope,
            event_us_per_iteration=event_slope,
            aggregate_values_per_us=throughput,
            conversion_19_experts_us=VALUES19 / throughput,
        )
        fits.append(f)
        print(json.dumps(f), flush=True)
    (ROOT / "phase3e1b-ceiling.json").write_text(
        json.dumps(
            dict(
                realtime_ticks_per_us=rate,
                values_19_experts=VALUES19,
                properties=str(props),
                topology=topology,
                records=rows,
                fits=fits,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
