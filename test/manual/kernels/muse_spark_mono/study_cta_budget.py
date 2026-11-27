"""3f GPU0 common four-wave budget study; each invocation bounded externally300s."""

import argparse
import json
from pathlib import Path

import torch
from bench_hbm import graph_times
from budget_kernels import attention, expert, padding_only, skinny
from check_experts import cases
from check_experts_routed import single_eight
from layer_fixture import build_layer, require_gpu0
from reference_archive import checked_load

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.srt.rocm_muse_spark import muse_moe_triton as native
from sglang.test.kernels.tlx_attention import paged_reference

ROOT = Path("/home/bangtliu/coord/logs")


def compile_budget(fn, args, kwargs, k, label):
    cu = torch.cuda.get_device_properties(0).multi_processor_count
    options = dict(num_warps=4, waves_per_eu=k, enable_fp_fusion=False)
    base = fn.warmup(*args, **kwargs, PAD_WORDS=0, **options, grid=(cu * k,))
    target = {4: 73728, 6: 49152, 9: 32768}[k]
    words = max(0, (target - base.metadata.shared) // 256 * 64)
    history = []
    for attempt in range(16):
        compiled = fn.warmup(
            *args, **kwargs, PAD_WORDS=words, **options, grid=(cu * k,)
        )
        measured = kernel_occupancy(label, compiled)
        row = dict(
            label=label,
            k=k,
            pad_bytes=words * 4,
            base_lds=base.metadata.shared,
            vgprs=compiled.n_regs,
            lds=compiled.metadata.shared,
            spills=compiled.n_spills,
            occupancy=measured.active_ctas_per_cu,
            waves_per_eu=k,
        )
        history.append(row)
        print(json.dumps(dict(event="compile-budget", **row)), flush=True)
        if measured.active_ctas_per_cu == k:
            break
        if measured.active_ctas_per_cu < k:
            if words == 0:
                break
            words = max(0, words - 64)
        else:
            words += 64
    else:
        raise RuntimeError(("LDS calibration did not converge", history))
    assert measured.active_ctas_per_cu >= k, ("budget infeasible", history)
    assert measured.active_ctas_per_cu == k, (
        "exact occupancy cap not established",
        history,
    )
    options.update(PAD_WORDS=words)

    def call():
        return fn[(cu * k,)](*args, **kwargs, **options)

    (ROOT / f"phase3f-{label}-k{k}.amdgcn").write_text(compiled.asm["amdgcn"])
    return call, row, history


def validate_padding(sink, pin, pad_bytes):
    words = pad_bytes // 4
    expected = sum(int(pin[: 1 << b].sum()) for b in range(6, 16) if words & (1 << b))
    assert sink.eq(expected).all(), ("padding checksum", pad_bytes, expected)


def expert_case(k, pin, sink):
    saved = checked_load("phase3e1-inputs-7.pt")
    x = saved["x"]
    w = saved["weights"]
    refs = checked_load("phase3e1-native-7.pt")
    refs["one8"] = checked_load("phase3e1b-native-one8-7.pt")
    patterns = dict(typical=cases(7, "cuda")["typical"], one8=single_eight("cuda"))
    rows = []
    for name, ids in patterns.items():
        table = native.moe_align_block_size_small(ids, 64, 0, 16)
        valid = ids.flatten() < 64
        inter = torch.empty((64, 4096), device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(inter)
        for phase, first, source, weight, scale, dest in (
            ("w13", True, x, w["w13"], w["w13_scale"], inter),
            ("w2", False, refs[name]["inter"], w["w2"], w["w2_scale"], out),
        ):
            call, resource, history = compile_budget(
                expert,
                (source, *table, weight, scale, dest, pin, sink),
                dict(GRID=sink.numel(), FIRST=first),
                k,
                phase,
            )
            call()
            torch.cuda.synchronize()
            validate_padding(sink, pin, resource["pad_bytes"])
            ref = refs[name]["inter" if first else "y"]
            diff = (dest[valid].float() - ref[valid].float()).double()
            err = float(diff.norm() / ref[valid].double().norm())
            assert err < 0.01, (phase, name, err)
            timing = graph_times(call, samples=50)
            rows.append(
                dict(
                    kind="expert",
                    case=name,
                    phase=phase,
                    resources=resource,
                    calibration=history,
                    time=timing,
                    relative_l2=err,
                )
            )
    return rows


def skinny_case(k, pin, sink):
    gen = torch.Generator(device="cuda").manual_seed(7)
    rows = []
    for name, n, kk in (("qkvg", 4608, 8192), ("o_proj", 8192, 2048)):
        x = torch.randn((8, kk), device="cuda", dtype=torch.bfloat16, generator=gen)
        w = (
            torch.randn((n, kk), device="cuda", dtype=torch.bfloat16, generator=gen)
            * 0.02
        )
        out = torch.empty((16, 8, n), device="cuda")
        ref = x.double() @ w.double().T
        call, resource, history = compile_budget(
            skinny, (x, w, out, pin, sink), dict(N=n, K=kk, GRID=sink.numel()), k, name
        )
        call()
        torch.cuda.synchronize()
        validate_padding(sink, pin, resource["pad_bytes"])
        got = out.double().sum(0)
        diff = got - ref
        err = float(diff.norm() / ref.norm())
        assert err < 0.005, (name, err)
        rows.append(
            dict(
                kind="skinny",
                case=name,
                split_k=16,
                active_tasks=n // 64 * 16,
                resources=resource,
                calibration=history,
                time=graph_times(call, samples=50),
                relative_l2=err,
            )
        )
    return rows


def attention_case(k, pin, sink, query_rows=16):
    context = 131072
    f = build_layer(seed=7, context=context, components=())
    gen = torch.Generator(device="cuda").manual_seed(8)
    q = (
        torch.randn((8, 32, 64), device="cuda", dtype=torch.bfloat16, generator=gen) * 2
    ).contiguous()
    rows = []
    for name, left, splits in (("full128k", -1, 128), ("sliding", 2047, 4)):
        out = torch.empty((splits * 256 // query_rows, query_rows, 64), device="cuda")
        lse = torch.empty((splits * 256 // query_rows, query_rows), device="cuda")
        args = dict(
            q=q,
            k=f.cache_k,
            v=f.cache_v,
            out=torch.empty_like(q),
            cu_seqlens_q=f.query_start,
            seqused_k=f.seq_lens,
            max_seqlen_q=8,
            max_seqlen_k=context,
            softmax_scale=0.125,
            causal=True,
            window_size=(left, 0),
            block_table=f.page_table,
            k_descale=f.k_scale,
            v_descale=f.v_scale,
        )
        ref = paged_reference(args, [8], [context])
        call, resource, history = compile_budget(
            attention,
            (
                q,
                f.cache_k,
                f.cache_v,
                f.page_table,
                f.seq_lens,
                f.k_scale,
                f.v_scale,
                out,
                lse,
                pin,
                sink,
            ),
            dict(GRID=sink.numel(), SPLITS=splits, LEFT=left, QUERY_ROWS=query_rows),
            k,
            name if query_rows == 16 else name + "-m64",
        )
        call()
        torch.cuda.synchronize()
        validate_padding(sink, pin, resource["pad_bytes"])
        logs = lse.reshape(splits, 4, 64).double()
        weights = torch.exp2(logs - logs.max(0, keepdim=True).values)
        merged = (out.reshape(splits, 4, 64, 64).double() * weights[:, :, :, None]).sum(
            0
        ) / weights.sum(0)[:, :, None]
        got = merged.reshape(4, 8, 8, 64).permute(1, 0, 2, 3).reshape(8, 32, 64)
        diff = (got - ref).abs()
        err = float(diff.max() / ref.abs().max())
        assert err < 0.008, (name, err)
        rows.append(
            dict(
                kind="attention",
                case=name,
                context=context,
                window_left=left,
                splits=splits,
                active_tasks=splits * 256 // query_rows,
                query_rows=query_rows,
                resources=resource,
                calibration=history,
                time=graph_times(call, samples=50),
                rel_linf=err,
                scope="partial loop plus padding; host merge excluded",
            )
        )
    return rows


def main():
    require_gpu0()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind", choices=("expert", "skinny", "attention"), required=True
    )
    parser.add_argument("--k", type=int, choices=(4, 6, 9), required=True)
    parser.add_argument("--query-rows", type=int, choices=(16, 64), default=16)
    args = parser.parse_args()
    grid = args.k * torch.cuda.get_device_properties(0).multi_processor_count
    pin = torch.arange(32768, device="cuda", dtype=torch.int32) % 1024
    sink = torch.empty(grid, device="cuda", dtype=torch.int32)
    if args.kind == "attention":
        rows = attention_case(args.k, pin, sink, args.query_rows)
    else:
        rows = {"expert": expert_case, "skinny": skinny_case}[args.kind](
            args.k, pin, sink
        )
    for row in rows:
        words = row["resources"]["pad_bytes"] // 4
        control = lambda: padding_only[(grid,)](
            pin, sink, PAD_WORDS=words, num_warps=4, waves_per_eu=args.k
        )
        row["padding_control"] = graph_times(control, samples=30)
        print(json.dumps(row), flush=True)
    suffix = "-m64" if args.kind == "attention" and args.query_rows == 64 else ""
    (ROOT / f"phase3f-{args.kind}-k{args.k}{suffix}.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
