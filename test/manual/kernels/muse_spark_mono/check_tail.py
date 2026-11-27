"""S3/S6 native unfused stage calls plus an independent FP64 oracle, GPU0 only."""

import argparse
import json
from pathlib import Path

import torch
from bench_hbm import graph_times, stream_read
from layer_fixture import build_layer, require_gpu0

LOG = Path("/home/bangtliu/coord/logs")


def native_boundary(branch, stream, pre, alpha, beta, post=None, last=False):
    from sglang.srt.rocm_muse_spark.fused_norm_residual import (
        _fused_norm_residual_triton_kernel,
    )

    out = torch.empty_like(branch)
    updated = torch.empty_like(stream)
    _fused_norm_residual_triton_kernel[(8,)](
        branch,
        stream,
        out,
        updated,
        None,
        None,
        None,
        post,
        pre,
        alpha,
        beta,
        1e-8,
        1e-5,
        H=8192,
        BLOCK_H=8192,
        POST_NORM_IS_INPUT_SCALE=post is not None,
        USE_RESIDUAL_GATE=True,
        RESIDUAL_GATE_CHANNEL_WISE=True,
        POST_NORM_HAS_WEIGHT=post is not None,
        PRE_NORM_HAS_WEIGHT=pre is not None,
        HAS_ROUTER=False,
        PRE_NORM_BF16=last,
        M_BUCKET=0,
        HAS_FP8=False,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out, updated


def oracle_boundary(branch, stream, pre, alpha, beta, post=None, last=False):
    b = branch.double()
    if post is not None:
        b = (b * post.double()).bfloat16().double()
    b = (b * (b.square().mean(-1, keepdim=True) + 1e-8).rsqrt()).bfloat16().double()
    updated = alpha.double() * stream.double() + beta.double() * b
    p = updated.bfloat16().double() if last else updated
    p = p * (p.square().mean(-1, keepdim=True) + 1e-5).rsqrt()
    if pre is not None:
        p = p * pre.double()
    return p.bfloat16(), updated.float()


def native_norm_gate(x, gate):
    from sglang.srt.rocm_muse_spark.fused_norm_gate import _rms_norm_gate_fwd_kernel

    out = torch.empty_like(x)
    _rms_norm_gate_fwd_kernel[(32,)](
        x,
        gate,
        out,
        None,
        1e-5,
        256,
        32,
        2048,
        2048,
        HEADS_PER_GROUP=32,
        D=64,
        BD=64,
        BLOCK_M=8,
        HAS_WEIGHT=False,
        HAS_NORM=True,
        HAS_GATE=True,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


def error(actual, expected):
    delta = (actual.double() - expected.double()).abs()
    assert torch.isfinite(actual).all()
    return dict(
        max_abs=delta.max().item(),
        rel_linf=(delta.max() / expected.double().abs().max().clamp_min(1e-20)).item(),
    )


def check(actual, expected, limit=0.015):
    result = error(actual, expected)
    assert result["rel_linf"] < limit, result
    return result


def fixture(seed):
    f = build_layer(seed=seed, components=("norms",))
    w = f.weights
    return f, w["pre_ffn_effective"].bfloat16(), w["attn_alpha"], w["attn_beta"]


def s3(seed, replay=False):
    from aiter.tuned_gemm import tgemm

    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.attn_tail import (
        attention_tail,
        make_workspace,
    )

    f, pre, alpha, beta = fixture(seed)
    torch.manual_seed(seed)
    x = torch.randn((4, 8, 2048), device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    weight = torch.randn((4, 8192, 2048), device="cuda", dtype=torch.bfloat16) * (
        2048**-0.5
    )
    ws = make_workspace(x.device)
    call = lambda: attention_tail(
        x, gate, weight, f.stream, pre, alpha, beta, workspace=ws
    )
    actual, stream = call()
    torch.cuda.synchronize()
    if not ws.state.status.eq(1).all():
        print(
            json.dumps(
                dict(
                    event="readiness",
                    status=ws.state.status.tolist(),
                    ready=ws.ready[0].tolist(),
                    arrivals=ws.state.arrivals[:, :, 0].tolist(),
                    epochs=ws.state.epochs[:, 0].tolist(),
                )
            ),
            flush=True,
        )
    assert ws.state.status.eq(1).all(), ws.state.status.tolist()
    normed = [native_norm_gate(x[r], gate[r]) for r in range(4)]
    native_partial = torch.stack(
        [tgemm.mm(normed[r], weight[r], None, otype=torch.bfloat16) for r in range(4)]
    )
    projected = []
    for r in range(4):
        raw = x[r].double().reshape(8, 32, 64)
        ng = (
            (
                raw
                * (raw.square().mean(-1, keepdim=True) + 1e-5).rsqrt()
                * gate[r].double().reshape_as(raw).sigmoid()
            )
            .reshape(8, 2048)
            .bfloat16()
        )
        projected.append((ng.double() @ weight[r].double().T).bfloat16())
    oracle_partial = torch.stack(projected)
    native_sum = torch.zeros_like(f.stream)
    oracle_sum = torch.zeros_like(f.stream)
    exact_sum = torch.zeros_like(f.stream)
    for r in range(4):
        native_sum = native_sum + native_partial[r].float()
        oracle_sum = oracle_sum + oracle_partial[r].float()
        exact_sum = exact_sum + ws.partial[r].float()
    if not torch.equal(ws.reduced, exact_sum.expand_as(ws.reduced)):
        payload = ws.payload
        parity = ws.state.epochs[0, 0].item() % 2
        words = ws.partial.view(torch.int32).reshape(4, -1)
        details = []
        for dst in range(4):
            for src in range(4):
                start = (
                    payload[0] + dst * payload[1] + parity * payload[2] + src * 32768
                )
                received = ws.scratch[start : start + 32768]
                details.append(
                    dict(
                        src=src,
                        dst=dst,
                        mismatches=(received != words[src]).sum().item(),
                    )
                )
        print(
            json.dumps(
                dict(
                    event="exchange_mismatch",
                    payload=details,
                    bad_per_row=(ws.reduced != exact_sum).sum(-1).tolist(),
                    first_bad=(ws.reduced[0].flatten() != exact_sum.flatten())
                    .nonzero()
                    .flatten()[:32]
                    .tolist(),
                    max_diff=(ws.reduced - exact_sum).abs().max().item(),
                    got=ws.reduced[0, 0, :32].tolist(),
                    expected=exact_sum[0, :32].tolist(),
                )
            ),
            flush=True,
        )
    assert torch.equal(ws.reduced, exact_sum.expand_as(ws.reduced)), (
        "rank order / exchange"
    )
    native = native_boundary(native_sum.bfloat16(), f.stream, pre, alpha, beta)
    oracle = oracle_boundary(oracle_sum.bfloat16(), f.stream, pre, alpha, beta)
    matched = native_boundary(exact_sum.bfloat16(), f.stream, pre, alpha, beta)
    # The stage's logical task index must be independent of the physical CTA.
    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.attn_tail import (
        _standalone,
    )

    remapped_out, remapped_stream = torch.empty_like(actual), torch.empty_like(stream)
    state = ws.state
    _standalone[(128,)](
        x,
        gate,
        weight,
        f.stream,
        pre,
        alpha,
        beta,
        ws.partial,
        ws.reduced,
        state.epochs,
        state.arrivals,
        state.flags,
        state.status,
        ws.scratch,
        ws.peers,
        ws.ready,
        remapped_out,
        remapped_stream,
        ws.payload,
        ws.flags,
        ROTATE=17,
        num_warps=16,
        enable_fp_fusion=False,
    )
    torch.cuda.synchronize()
    assert state.status.eq(1).all()
    assert torch.equal(remapped_out, actual) and torch.equal(remapped_stream, stream)
    result = dict(
        stage="S3",
        logical_cta_permutation_exact=True,
        seed=seed,
        partial_vs_native=check(ws.partial, native_partial),
        partial_vs_fp64=check(ws.partial, oracle_partial),
        out_vs_native=check(actual, native[0].expand_as(actual)),
        stream_vs_native=check(stream, native[1].expand_as(stream)),
        out_vs_fp64=check(actual, oracle[0].expand_as(actual)),
        stream_vs_fp64=check(stream, oracle[1].expand_as(stream)),
        matched_boundary_out=check(actual, matched[0].expand_as(actual), 0.004),
        matched_boundary_stream=check(stream, matched[1].expand_as(stream), 1e-5),
        timing=graph_times(call),
    )

    def native_call():
        parts = [
            tgemm.mm(
                native_norm_gate(x[r], gate[r]), weight[r], None, otype=torch.bfloat16
            )
            for r in range(4)
        ]
        total = torch.zeros_like(f.stream)
        for p in parts:
            total = total + p.float()
        boundaries = [
            native_boundary(total.bfloat16(), f.stream, pre, alpha, beta)
            for _ in range(4)
        ]
        return boundaries[0]

    result["native"] = graph_times(native_call)
    # Exact same four-shard weight byte count; streaming set greatly exceeds L2.
    source = torch.rand((weight.numel() // 2,), device="cuda", dtype=torch.float32)
    sink = torch.empty((source.numel() // 4096,), device="cuda")
    result["read"] = graph_times(
        lambda: stream_read[(sink.numel(),)](source, sink, source.numel(), 4096)
    )
    result["weight_bytes"] = weight.numel() * 2
    if replay:
        from stage_replay import exercise

        delay = torch.tensor([seed], device="cuda", dtype=torch.int32)
        run_delayed = lambda: attention_tail(
            x, gate, weight, f.stream, pre, alpha, beta, workspace=ws, delay_seed=delay
        )
        result["replay"] = exercise(
            run_delayed,
            x,
            ws.state,
            delay,
            lambda: tuple(t.expand(4, -1, -1) for t in native_call()),
            "S3",
        )
        # Missing producer is last; discard workspace after this bounded failure.
        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.attn_tail import (
            _standalone,
        )

        ws.state.status.fill_(-1)
        out = torch.full_like(actual, float("nan"))
        updated = torch.full_like(stream, float("nan"))
        state = ws.state
        _standalone[(128,)](
            x,
            gate,
            weight,
            f.stream,
            pre,
            alpha,
            beta,
            ws.partial,
            ws.reduced,
            state.epochs,
            state.arrivals,
            state.flags,
            state.status,
            ws.scratch,
            ws.peers,
            ws.ready,
            out,
            updated,
            ws.payload,
            ws.flags,
            SKIP=127,
            num_warps=16,
            enable_fp_fusion=False,
        )
        torch.cuda.synchronize()
        assert state.status[:127].eq(0).all() and state.status[127].item() == -1
        assert out.isnan().all() and updated.isnan().all()
        result["missing_arrival"] = dict(
            skipped=127, failed=127, absent=1, output_unwritten=True
        )
    print(json.dumps(result), flush=True)
    return result


def s6_prepare():
    for seed in (7, 20261117):
        f = build_layer(seed=seed, components=("unpooler", "norms"))
        values = dict(
            x=f.x[:, :4096].contiguous() * 0.5,
            weight=f.weights["unpooler"],
            stream=f.stream,
            pre=f.weights["pre_attn_effective"].bfloat16(),
            post=f.weights["post_ffn_effective"].bfloat16(),
            alpha=f.weights["ffn_alpha"],
            beta=f.weights["ffn_beta"],
        )
        torch.save(
            {k: v.cpu() for k, v in values.items()},
            LOG / f"phase3d-S6-inputs-{seed}.pt",
        )


def s6_native_reference():
    import triton

    from sglang.kernels.ops.gemm.unpool_gemm_norm_residual import (
        unpool_gemm_norm_residual,
    )

    for seed in (7, 20261117):
        d = {
            k: v.cuda()
            for k, v in torch.load(
                LOG / f"phase3d-S6-inputs-{seed}.pt", weights_only=True
            ).items()
        }
        projection = (d["x"].double() @ d["weight"].double().T).bfloat16()
        results = {}
        for kind in ("normal", "last_weighted", "last_unweighted"):
            pre = torch.ones_like(d["pre"]) if kind == "last_unweighted" else d["pre"]
            last = kind != "normal"
            call = lambda: unpool_gemm_norm_residual(
                d["x"],
                d["weight"],
                d["stream"],
                d["post"],
                pre,
                d["alpha"],
                d["beta"],
                1e-8,
                1e-5,
                pre_bf16=last,
            )
            actual = call()
            expected = oracle_boundary(
                projection,
                d["stream"],
                pre,
                d["alpha"],
                d["beta"],
                post=d["post"],
                last=last,
            )
            errors = [check(a, b) for a, b in zip(actual, expected)]
            results[kind] = dict(
                output=tuple(t.cpu() for t in actual),
                timing=graph_times(call),
                errors=errors,
                torch=str(torch.__version__),
                triton=str(triton.__version__),
            )
            print(
                json.dumps(
                    dict(
                        event="S6_native_reference",
                        seed=seed,
                        kind=kind,
                        timing=results[kind]["timing"],
                        errors=errors,
                    )
                ),
                flush=True,
            )
        torch.save(results, LOG / f"phase3d-S6-native-{seed}.pt")


def check_boundary_rounding(reference=None, *, waves=16):
    from triton.experimental import gluon as g
    from triton.experimental.gluon import language as gl

    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.boundary import (
        norm_residual,
    )

    @g.jit
    def direct(
        Branch, Stream, Post, Pre, Alpha, Beta, Out, Updated, LAST: gl.constexpr
    ):
        norm_residual(
            gl.program_id(0),
            Branch,
            Stream,
            Post,
            Pre,
            Alpha,
            Beta,
            Out,
            Updated,
            True,
            LAST,
            True,
        )

    # Repeated two-value patterns make the power-of-two sums exact, isolating
    # BF16 rounding points from GEMM error and reduction-order noise.
    branch = torch.tensor([0.75, 1.125], device="cuda", dtype=torch.bfloat16).repeat(
        8, 4096
    )
    stream = torch.tensor([0.1234567, -0.479213], device="cuda").repeat(8, 4096)
    post = torch.tensor(
        [1.0078125, 0.9921875], device="cuda", dtype=torch.bfloat16
    ).repeat(4096)
    weight = torch.tensor(
        [1.015625, 0.9765625], device="cuda", dtype=torch.bfloat16
    ).repeat(4096)
    alpha = torch.full((8192,), 0.87, device="cuda")
    beta = torch.full((8192,), 0.42, device="cuda")
    partial = branch.float().view(torch.int32)
    for pre in (weight, None):
        outputs = []
        for last in (False, True):
            out = torch.empty_like(branch)
            updated = torch.empty_like(stream)
            direct[(8,)](
                partial,
                stream,
                post,
                pre,
                alpha,
                beta,
                out,
                updated,
                last,
                num_warps=waves,
                enable_fp_fusion=False,
            )
            expected = (native_boundary if reference is None else reference)(
                branch, stream, pre, alpha, beta, post=post, last=last
            )
            assert torch.equal(out, expected[0]) and torch.equal(
                updated, expected[1]
            ), "boundary rounding pattern"
            outputs.append(out)
        assert not torch.equal(outputs[0], outputs[1]), (
            "pattern must distinguish pre-norm input dtype"
        )
    print(
        json.dumps(
            dict(
                event="S6_boundary_rounding",
                cases=4,
                bitwise=True,
                pre_norm_dtype_distinguished=True,
            )
        ),
        flush=True,
    )


def s6(seed, replay=False):
    from aiter.tuned_gemm import tgemm

    if seed == 7:
        check_boundary_rounding()
    from stage_replay import exercise

    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.tail import (
        _standalone,
        make_workspace,
        unpool_boundary,
    )

    d = {
        k: v.cuda()
        for k, v in torch.load(
            LOG / f"phase3d-S6-inputs-{seed}.pt", weights_only=True
        ).items()
    }
    x, weight, stream, post, alpha, beta = (
        d[k] for k in ("x", "weight", "stream", "post", "alpha", "beta")
    )
    saved = torch.load(LOG / f"phase3d-S6-native-{seed}.pt", weights_only=True)
    native_projection = tgemm.mm(x, weight, None, otype=torch.bfloat16)
    oracle_projection = (x.double() @ weight.double().T).bfloat16()
    source = torch.rand((weight.numel() // 2,), device="cuda")
    sink = torch.empty((source.numel() // 4096,), device="cuda")
    read = graph_times(
        lambda: stream_read[(sink.numel(),)](source, sink, source.numel(), 4096)
    )
    rows = []
    replay_args = []
    for kind in ("normal", "last_weighted", "last_unweighted"):
        pre = None if kind == "last_unweighted" else d["pre"]
        last = kind != "normal"
        ws = make_workspace(x.device)
        call = lambda: unpool_boundary(
            x, weight, stream, post, pre, alpha, beta, pre_bf16=last, workspace=ws
        )
        actual = call()
        torch.cuda.synchronize()
        assert ws.state.status.eq(1).all(), ws.state.status.tolist()
        own = native_boundary(
            native_projection, stream, pre, alpha, beta, post=post, last=last
        )
        ideal = oracle_boundary(
            oracle_projection, stream, pre, alpha, beta, post=post, last=last
        )
        matched = native_boundary(
            ws.projection.view(torch.float32).bfloat16(),
            stream,
            pre,
            alpha,
            beta,
            post=post,
            last=last,
        )
        fused = tuple(t.cuda() for t in saved[kind]["output"])

        def native_call():
            projected = tgemm.mm(x, weight, None, otype=torch.bfloat16)
            return native_boundary(
                projected, stream, pre, alpha, beta, post=post, last=last
            )

        row = dict(
            stage="S6",
            seed=seed,
            kind=kind,
            projection_vs_native=check(
                ws.projection.view(torch.float32), native_projection
            ),
            projection_vs_fp64=check(
                ws.projection.view(torch.float32), oracle_projection
            ),
            out_vs_native=check(actual[0], own[0]),
            stream_vs_native=check(actual[1], own[1]),
            out_vs_fp64=check(actual[0], ideal[0]),
            stream_vs_fp64=check(actual[1], ideal[1]),
            matched_out=check(actual[0], matched[0], 0.008),
            matched_stream=check(actual[1], matched[1], 0.004),
            out_vs_native_fused39=check(actual[0], fused[0]),
            stream_vs_native_fused39=check(actual[1], fused[1]),
            timing=graph_times(call),
            native=graph_times(native_call),
            read=read,
            weight_bytes=weight.numel() * 2,
            native_fused39=saved[kind]["timing"],
            native_fused39_stack=dict(
                torch=saved[kind]["torch"], triton=saved[kind]["triton"]
            ),
        )
        if kind == "normal":
            normal_out, normal_stream = actual[0].clone(), actual[1].clone()
        else:
            assert torch.equal(actual[1], normal_stream), (
                "pre-norm settings must not change the stream"
            )
            assert not torch.equal(actual[0], normal_out), (
                "last/pre-weight paths must be exercised"
            )
        print(json.dumps(row), flush=True)
        rows.append(row)
        if replay:
            # Fresh input per mode; the helper ends with x negated.
            replay_args.append((kind, x.clone(), pre, last))
    # Correctness/timing first, then fresh-workspace replay; fault is last.
    for kind, changing, pre, last in replay_args:
        ws = make_workspace(x.device)
        delay = torch.tensor([seed], device="cuda", dtype=torch.int32)
        call = lambda: unpool_boundary(
            changing,
            weight,
            stream,
            post,
            pre,
            alpha,
            beta,
            pre_bf16=last,
            workspace=ws,
            delay_seed=delay,
        )
        reference = lambda: native_boundary(
            tgemm.mm(changing, weight, None, otype=torch.bfloat16),
            stream,
            pre,
            alpha,
            beta,
            post=post,
            last=last,
        )
        exercise(call, changing, ws.state, delay, reference, f"S6-{seed}-{kind}")
    if replay:
        ws = make_workspace(x.device)
        s = ws.state
        # Prime a stale completion tag before suppressing the next producer.
        unpool_boundary(x, weight, stream, post, d["pre"], alpha, beta, workspace=ws)
        old = s.flags[0, 0, 0].item()
        s.status.fill_(-1)
        out = torch.full((8, 8192), float("nan"), device="cuda", dtype=torch.bfloat16)
        updated = torch.full((8, 8192), float("nan"), device="cuda")
        _standalone[(192,)](
            x,
            weight,
            stream,
            post,
            d["pre"],
            alpha,
            beta,
            ws.projection,
            s.epochs,
            s.arrivals,
            s.flags,
            s.status,
            out,
            updated,
            False,
            SKIP=127,
            num_warps=16,
            enable_fp_fusion=False,
        )
        torch.cuda.synchronize()
        expected_status = torch.zeros_like(s.status)
        expected_status[127] = -1
        assert (
            torch.equal(s.status, expected_status)
            and out.isnan().all()
            and updated.isnan().all()
        )
        assert s.flags[0, 0, 0].item() == old
        print(
            json.dumps(
                dict(
                    event="S6_missing_arrival",
                    seed=seed,
                    skipped=127,
                    failed=191,
                    absent=1,
                    stale_tag=old,
                )
            ),
            flush=True,
        )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("S3", "S6"), default="S3")
    p.add_argument("--replay", action="store_true")
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--native-reference", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    require_gpu0()
    if args.prepare:
        assert args.stage == "S6"
        s6_prepare()
        return
    if args.native_reference:
        assert args.stage == "S6"
        s6_native_reference()
        return
    result = (
        s3(args.seed, replay=args.replay)
        if args.stage == "S3"
        else s6(args.seed, replay=args.replay)
    )
    (LOG / f"phase3d-{args.stage}-seed{args.seed}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
