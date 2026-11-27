"""Bounded H_VALU sweep; FP64/native checks and exact hybrid selection fixtures."""

import argparse
import json
from pathlib import Path

import torch
import triton
from bench_hbm import graph_times
from check_expert_native import error
from check_experts import SEEDS, cases, load
from check_experts_routed import oracle, single_eight
from layer_fixture import require_gpu0
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    experts_routed as stage,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.experts_valu import (
    _cache_pairs,
    _convert_packet,
    _gemv,
    _weight_packet,
)
from sglang.srt.rocm_muse_spark import muse_moe_triton as native

ROOT = Path("/home/bangtliu/coord/logs")


@g.jit
def _unit_packet_probe(W, Scale, Out):
    a, b, c, d, scale = _weight_packet(W, Scale, 0, 0, 0, 8192)
    pairs = _convert_packet(a, b, c, d)
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 16], [4, 8, 1], [16, 1, 1], [2, 1, 0]
    )
    row = gl.arange(0, 64, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    split = gl.arange(0, 8, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    pair = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    gl.store(
        Out
        + row[:, None, None] * 128
        + split[None, :, None] * 16
        + pair[None, None, :],
        pairs,
    )


@g.jit
def _isolated(X, Sorted, W, Scale, Out):
    cached = _cache_pairs(X, Sorted, 0, False, 1)
    tile = gl.program_id(0)
    gate = _gemv(W, Scale, cached, 0, tile, 1, 8192, 1)
    up = _gemv(W, Scale, cached, 0, tile + 64, 1, 8192, 1)
    value = gate[0] / (1.0 + gl.exp(-gate[0])) * up[0]
    n = gl.arange(0, 64, layout=value.type.layout)
    gl.store(Out + tile * 64 + n, value.to(gl.bfloat16))


def isolate():
    x, w = load(7)
    packet = torch.empty((64, 8, 16), device=x.device, dtype=torch.int32)
    _unit_packet_probe[(1,)](w["w13"], w["w13_scale"], packet, num_warps=16)
    torch.cuda.synchronize()
    dense_unit = native.dequant_mxfp4(
        w["w13"][0, :64], torch.full_like(w["w13_scale"][0, :64], 127)
    ).bfloat16()
    indices = (
        torch.arange(8, device=x.device)[:, None] * 512
        + torch.arange(32, device=x.device)[None, :]
    )
    got = packet.view(torch.bfloat16).reshape(64, 8, 32)
    expected = dense_unit[:, indices]
    assert torch.equal(got, expected)
    print(
        json.dumps(
            dict(
                event="unit-converter",
                values=got.numel(),
                mismatches=0,
                bit_mismatches=int(
                    got.view(torch.int16)
                    .ne(expected.contiguous().view(torch.int16))
                    .sum()
                ),
            )
        ),
        flush=True,
    )
    table = torch.full((1024,), 64, device=x.device, dtype=torch.int32)
    table[0] = 0
    output = torch.empty(4096, device=x.device, dtype=torch.bfloat16)
    k = _isolated[(64,)](
        x, table, w["w13"], w["w13_scale"], output, num_warps=16, enable_fp_fusion=False
    )
    torch.cuda.synchronize()
    dense = native.dequant_mxfp4(w["w13"][0], w["w13_scale"][0]).double()
    gate, up = (x[0].double() @ dense.T).split(4096, -1)
    ref = (gate.sigmoid() * gate * up).bfloat16()
    err = error(output, ref)
    assert err["relative_l2"] < 0.01
    asm = k.asm["amdgcn"]
    assert "v_wmma" not in asm and "dot2" not in asm
    assert "global_load_b128" in asm and "TH_LOAD_NT" in asm
    (ROOT / "phase3e1c-valu-only.amdgcn").write_text(asm)
    record = dict(
        event="isolate",
        error=err,
        registers=k.n_regs,
        spills=k.n_spills,
        shared=k.metadata.shared,
        no_wmma=True,
        b128_nontemporal=True,
    )
    print(json.dumps(record), flush=True)
    (ROOT / "phase3e1c-valu-only.json").write_text(json.dumps(record, indent=2) + "\n")


def mixed(device):
    values = []
    for e, count in enumerate((1, 2, 3, 4, 5, 6, 7, 8, 17)):
        values.extend([e] * count)
    values.extend([255] * (64 - len(values)))
    return torch.tensor(values, device=device, dtype=torch.int32).reshape(8, 8)


def native_reference():
    assert triton.__version__.split(".")[:2] == ["3", "9"]
    for seed in SEEDS:
        x, w = load(seed)
        ids = mixed(x.device)
        table = native.moe_align_block_size_small(ids, 64, 0, 16)
        inter = torch.full(
            (64, 4096), float("nan"), device=x.device, dtype=torch.bfloat16
        )
        y = torch.full_like(inter, float("nan"))
        cfg = native._pick_config(8, 8, 64, True)

        def call():
            native._launch_gemm(
                x,
                w["w13"],
                w["w13_scale"],
                inter,
                *table,
                4096,
                4096,
                64,
                8,
                True,
                True,
                True,
                cfg,
            )
            native._launch_gemm(
                inter,
                w["w2"],
                w["w2_scale"],
                y,
                *table,
                4096,
                4096,
                64,
                8,
                False,
                False,
                True,
                cfg,
            )

        call()
        torch.cuda.synchronize()
        result = dict(inter=inter.cpu(), y=y.cpu(), time=graph_times(call))
        torch.save(result, ROOT / f"phase3e1c-native-mixed-{seed}.pt")
        print(
            json.dumps(dict(seed=seed, case="mixed", native39=result["time"])),
            flush=True,
        )


def validate(threshold, aligned=False, tracked=False, core_k=0):
    prefix = "phase3e1d-aligned" if aligned else f"phase3e1c-H{threshold}"
    if tracked:
        prefix = f"phase3e1e-tracked-k{core_k}" if core_k else "phase3e1e-tracked"
    records = []
    for seed in SEEDS:
        x, w = load(seed)
        refs = torch.load(ROOT / f"phase3e1-native-{seed}.pt", weights_only=True)
        refs["one8"] = torch.load(
            ROOT / f"phase3e1b-native-one8-{seed}.pt", weights_only=True
        )
        refs["mixed"] = torch.load(
            ROOT / f"phase3e1c-native-mixed-{seed}.pt", weights_only=True
        )
        patterns = cases(seed, x.device)
        patterns.update(one8=single_eight(x.device), mixed=mixed(x.device))
        ws = stage.make_workspace(x.device)
        for name, ids in patterns.items():
            table = native.moe_align_block_size_small(ids, 64, 0, 16)
            ws.inter[:64].fill_(float("nan"))
            ws.output.fill_(float("nan"))
            call = lambda: stage.experts(
                x,
                *table,
                w["w13"],
                w["w13_scale"],
                w["w2"],
                w["w2_scale"],
                ws,
                h_valu=threshold,
                aligned=aligned,
                tracked=tracked,
                core_k=core_k,
            )
            compiled = call()
            torch.cuda.synchronize()
            if tracked:
                (ROOT / f"{prefix}-compile-{seed}.amdgcn").write_text(
                    compiled.asm["amdgcn"]
                )
            assert ws.status.eq(1).all()
            valid = (ids.flatten() >= 0) & (ids.flatten() < 64)
            nr = refs[name]
            comparison = None
            if valid.any():
                comparison = dict(
                    inter=error(ws.inter[:64][valid], nr["inter"].cuda()[valid]),
                    output=error(ws.output[valid], nr["y"].cuda()[valid]),
                )
                if (
                    comparison["inter"]["relative_l2"] >= 0.01
                    or comparison["output"]["relative_l2"] >= 0.01
                ):
                    print(
                        json.dumps(
                            dict(
                                event="mismatch",
                                seed=seed,
                                case=name,
                                aligned=aligned,
                                tracked=tracked,
                                core_k=core_k,
                                error=comparison,
                            )
                        ),
                        flush=True,
                    )
                assert (
                    comparison["inter"]["relative_l2"] < 0.01
                    and comparison["output"]["relative_l2"] < 0.01
                )
            assert (
                ws.inter[:64][~valid].isnan().all() and ws.output[~valid].isnan().all()
            )
            fp64 = oracle(x, ids, w, ws)
            timing = graph_times(call)
            assert ws.status.eq(1).all()
            counts = torch.bincount(ids.flatten()[valid].long(), minlength=64)
            chosen = (
                ((counts > 0) & (counts <= threshold)).sum().item() if threshold else 0
            )
            if name == "mixed":
                assert chosen == threshold
            record = dict(
                seed=seed,
                case=name,
                H_VALU=threshold,
                triton_version=triton.__version__,
                aligned=aligned,
                tracked=tracked,
                core_k=core_k,
                hits=int(valid.sum()),
                m_blocks=table[2].item() // 16,
                valu_experts=chosen,
                wmma_experts=int((counts > threshold).sum())
                if threshold
                else int((counts > 0).sum()),
                stage=timing,
                native39=nr["time"],
                native_error=comparison,
                oracle=fp64,
                registers=compiled.n_regs,
                spills=compiled.n_spills,
                shared=compiled.metadata.shared,
            )
            print(json.dumps(record), flush=True)
            records.append(record)
            if name == "typical":
                (ROOT / f"{prefix}-{seed}.amdgcn").write_text(compiled.asm["amdgcn"])
        # Missing VALU producer and stale flags, after all normal timing/correctness.
        ids = patterns["typical"]
        table = native.moe_align_block_size_small(ids, 64, 0, 16)
        call = lambda: stage.experts(
            x,
            *table,
            w["w13"],
            w["w13_scale"],
            w["w2"],
            w["w2_scale"],
            ws,
            h_valu=threshold,
            aligned=aligned,
            tracked=tracked,
            core_k=core_k,
        )
        call()
        torch.cuda.synchronize()
        old = ws.flags[0].clone()
        ws.output.fill_(float("nan"))
        stage.experts(
            x,
            *table,
            w["w13"],
            w["w13_scale"],
            w["w2"],
            w["w2_scale"],
            ws,
            h_valu=threshold,
            aligned=aligned,
            tracked=tracked,
            core_k=core_k,
            skip=0,
        )
        torch.cuda.synchronize()
        failed = int(ws.status.eq(0).sum())
        assert failed == 64
        slots = table[0][:16]
        slots = slots[slots < 64].long()
        assert ws.output[slots].isnan().all()
        assert torch.equal(ws.flags[0], old)
        print(
            json.dumps(
                dict(
                    event="missing",
                    H_VALU=threshold,
                    seed=seed,
                    failed=failed,
                    stale_rejected=True,
                )
            ),
            flush=True,
        )
    (ROOT / f"{prefix}.json").write_text(json.dumps(records, indent=2) + "\n")


def main():
    require_gpu0()
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-reference", action="store_true")
    parser.add_argument("--aligned", action="store_true")
    parser.add_argument("--tracked", action="store_true")
    parser.add_argument("--core-k", type=int, choices=(0, 64, 128), default=0)
    parser.add_argument("--isolate", action="store_true")
    parser.add_argument("--threshold", type=int, choices=(0, 1, 2, 4, 8), default=1)
    args = parser.parse_args()
    if args.isolate:
        isolate()
    elif args.native_reference:
        native_reference()
    else:
        validate(
            0 if args.aligned or args.tracked else args.threshold,
            args.aligned,
            args.tracked,
            args.core_k,
        )


if __name__ == "__main__":
    main()
