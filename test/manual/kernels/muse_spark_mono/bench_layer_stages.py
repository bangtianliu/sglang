"""GPU0 same-fixture stage sums, with local reference peers and native math."""

import argparse
import json
from pathlib import Path

import torch
from bench_hbm import graph_times
from check_experts_complete import table_for
from check_route import native as native_route
from check_tail import check, native_boundary, native_norm_gate
from layer_case import make
from layer_fixture import require_gpu0
from layer_reference import build
from reference_archive import checked_load

from sglang.kernels.ops.attention.qk_norm_rope_cache import qk_norm_rope_cache
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import standin
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_attention as s2,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_attn_tail as s3,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_experts as s5,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_qkvg as s1,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_route as s4,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_tail as s6,
)
from sglang.srt.rocm_muse_spark import muse_moe_triton as moe


def timed(call, prepare=None):
    if prepare is None:
        return graph_times(call, samples=100)
    prepare()
    call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepare()
        start.record()
        call()
        end.record()
    values = []
    for i in range(110):
        graph.replay()
        torch.cuda.synchronize()
        if i >= 10:
            values.append(start.elapsed_time(end) * 1000)
    values.sort()
    return dict(median_us=(values[49] + values[50]) / 2, p99_us=values[98])


def benchmark(f, w, m, ref, context, left, rank):
    own = {}
    native = {}

    def run(name, call, prep=None):
        if prep:
            prep()
        out = call()
        torch.cuda.synchronize()
        own[name] = timed(call, prep)
        return out

    ws = s1.make_workspace("cuda")
    run(
        "S1",
        lambda: s1.qkvg(
            f.x,
            w["qkvg"],
            f.positions,
            m[1],
            f.cache_k,
            f.cache_v,
            f.locations,
            m[3],
            1e-5,
            5.4730064863875,
            f.k_scale,
            f.v_scale,
            workspace=ws,
        ),
    )
    assert (
        ws.status.eq(1).all()
        and ws.state[ws.ctas * 32 : ws.ctas * 32 + 72].eq(ws.state[0]).all()
    )
    qa = ref["q"].contiguous()
    aa = ref["attn"].bfloat16().contiguous()
    ws = s2.make_workspace("cuda", context=context, window_left=left)
    out = run(
        "S2",
        lambda: s2.attention(
            qa,
            f.cache_k,
            f.cache_v,
            f.page_table,
            f.seq_lens,
            f.k_scale,
            f.v_scale,
            context=context,
            window_left=left,
            mapping=m[3],
            workspace=ws,
        ),
    )
    check(out, ref["attn"], 0.008)
    assert ws.status.eq(1).all()
    ws = s3.make_workspace("cuda")
    prep = lambda: standin.prepare(
        ws.peers, ref["partial3"], ws.state.epochs, rank=rank
    )
    run(
        "S3",
        lambda: s3.attention_tail(
            aa,
            ref["gate"],
            w["o_proj"],
            f.stream,
            w["pre3"],
            w["alpha3"],
            w["beta3"],
            rank=rank,
            workspace=ws,
        ),
        prep,
    )
    assert ws.state.status.eq(1).all()
    ws = s4.make_workspace("cuda")
    run(
        "S4",
        lambda: s4.routing(
            ref["x4"],
            w["joint_router_pooler"],
            w["bias"],
            w["route_norm"],
            rank=rank,
            workspace=ws,
        ),
    )
    assert ws.state.status.eq(1).all()
    tables = table_for(ref["ids"])
    table = tuple(t.reshape(4, -1)[rank].contiguous() for t in tables)
    ws = s5.make_workspace("cuda")
    prep = lambda: standin.prepare(
        ws.peers, ref["partial5"], ws.state.epochs, rank=rank
    )
    run(
        "S5",
        lambda: s5.s5(
            ref["pool"], ref["ids"], ref["scores"], table, w, w["gamma5"], ws, rank=rank
        ),
        prep,
    )
    assert ws.state.status.eq(1).all()
    ws = s6.make_workspace("cuda")
    run(
        "S6",
        lambda: s6.unpool_boundary(
            ref["x6"],
            w["unpooler"],
            ref["mid_stream"],
            w["post6"],
            w["pre6"],
            w["alpha6"],
            w["beta6"],
            workspace=ws,
        ),
    )
    assert ws.state.status.eq(1).all()

    # Native unfused BF16 linear, followed by the native epilogue. Never _dense_partial.
    def ns1():
        p = torch.nn.functional.linear(f.x, w["qkvg"])
        q, k, v, g = p.split((2048, 256, 256, 2048), -1)
        qk_norm_rope_cache(
            q,
            k,
            v,
            f.positions,
            m[1],
            f.cache_k,
            f.cache_v,
            f.locations,
            m[3],
            1e-5,
            5.4730064863875,
            f.k_scale,
            f.v_scale,
        )

    native["S1"] = timed(ns1)
    # Native AITER consumes original physical pages rather than compact SWA slots.
    k = f.cache_k
    v = f.cache_v
    if left >= 0:
        valid = f.mapping >= 0
        k = torch.zeros(
            (context // 16, 16, 4, 64), device="cuda", dtype=torch.float8_e4m3fn
        )
        v = torch.zeros_like(k)
        k.view(torch.uint8).reshape(context, 4, 64)[valid] = f.cache_k.view(
            torch.uint8
        ).reshape(-1, 4, 64)[f.mapping[valid]]
        v.view(torch.uint8).reshape(context, 4, 64)[valid] = f.cache_v.view(
            torch.uint8
        ).reshape(-1, 4, 64)[f.mapping[valid]]
    from aiter.ops.triton.attention.unified_attention import unified_attention

    ao = torch.empty((8, 32, 64), device="cuda", dtype=torch.bfloat16)

    def ns2():
        unified_attention(
            q=qa.reshape(8, 32, 64),
            k=k,
            v=v,
            out=ao,
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
            softcap=0.0,
            q_descale=None,
        )

    ns2()
    check(ao.reshape(8, 2048), ref["attn"], 0.008)
    native["S2"] = timed(ns2)

    def ordered(local, parts):
        value = torch.zeros_like(local, dtype=torch.float32)
        for src in range(4):
            value = value + (local if src == rank else parts[src]).float()
        return value

    def ns3():
        part = torch.nn.functional.linear(
            native_norm_gate(aa, ref["gate"]), w["o_proj"]
        )
        return native_boundary(
            ordered(part, ref["partial3"]).bfloat16(),
            f.stream,
            w["pre3"],
            w["alpha3"],
            w["beta3"],
        )

    native["S3"] = timed(ns3)
    native["S4"] = timed(
        lambda: native_route(
            ref["x4"], w["joint_router_pooler"], w["bias"], w["route_norm"], rank
        )
    )
    cfg = moe._pick_config(8, 8, 64, True)
    inter = torch.empty((64, 4096), device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(inter)
    part = torch.empty((8, 4096), device="cuda", dtype=torch.bfloat16)

    def ns5():
        moe._launch_gemm(
            ref["pool"],
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
        moe._launch_gemm(
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
        moe.finalize_norm_weighted_sum_kernel[(8,)](
            y,
            w["gamma5"],
            ref["scores"],
            ref["ids"],
            part,
            4096,
            rank * 64,
            64,
            1e-8,
            1.0,
            0,
            TOP_K=8,
            HAS_NORM=True,
            HAS_SCALE=True,
            BLOCK_H=4096,
            SPLIT_K=1,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return ordered(part, ref["partial5"])

    ns5()
    check(part, ref["partial5"][rank], 0.02)
    native["S5"] = timed(ns5)
    native["S6"] = timed(
        lambda: native_boundary(
            torch.nn.functional.linear(ref["x6"], w["unpooler"]),
            ref["mid_stream"],
            w["pre6"],
            w["alpha6"],
            w["beta6"],
            post=w["post6"],
        )
    )
    return dict(
        standalone=own,
        native=native,
        standalone_sum_us=sum(t["median_us"] for t in own.values()),
        native_sum_us=sum(t["median_us"] for t in native.values()),
        native_collective="GPU-local ordered FP32 tensor additions; not a real GPU-peer collective",
    )


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    for name, default in (("context", 8192), ("left", -1)):
        p.add_argument("--" + name, type=int, default=default)
    a = p.parse_args()
    f, w, m = make(7, a.context, a.left, 0)
    ref = build(f, w, 7, a.context, a.left, expert_fp64=False)
    weights = {n: w[n] for n in ("w13", "w13_scale", "w2", "w2_scale")}
    wo = checked_load("S3-inputs-7.pt")["weight"]
    rows = []
    for rank in range(4):
        w["o_proj"] = wo[rank]
        for n, t in weights.items():
            w[n] = t.roll(rank * 11, 0).contiguous()
        row = dict(
            context=a.context,
            left=a.left,
            rank=rank,
            **benchmark(f, w, m, ref, a.context, a.left, rank),
        )
        print(json.dumps(row), flush=True)
        rows.append(row)
        Path(
            f"/home/bangtliu/coord/logs/phase43-stage-sums-{a.context}-{a.left}.json"
        ).write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
