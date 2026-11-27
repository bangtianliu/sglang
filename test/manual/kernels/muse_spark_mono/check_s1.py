"""GPU0 S1 parity against SGLang and an independent FP64 mathematical oracle."""

import argparse
import json
from pathlib import Path

import torch
from bench_hbm import graph_times
from layer_fixture import EPS, Q_SCALE
from layer_fixture import build_layer as original_build_layer
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.qkvg import (
    fused_s1,
    head_epilogue,
    make_workspace,
    qkvg,
)

LOG = Path("/home/bangtliu/coord/logs")


ARCHIVE = False
FIXTURE_KEYS = {}


def build_layer(**kwargs):
    if not ARCHIVE:
        return original_build_layer(**kwargs)
    from reference_archive import fixture

    seed = kwargs.get("seed", 20261117)
    context = kwargs.get("context", 8192)
    f = fixture(seed, context)
    FIXTURE_KEYS[id(f)] = (seed, context)
    return f


def project(f):
    if not ARCHIVE:
        from bench_s1_reference import project as native_project

        return native_project(f)
    from reference_archive import checked_load

    seed, context = FIXTURE_KEYS[id(f)]
    return checked_load(f"S1-projection-{seed}-{context}.pt")


def reference(f, *, rope, mapping, cache_k=None, cache_v=None, projected=None):
    if not ARCHIVE:
        from bench_s1_reference import reference as native_reference

        return native_reference(
            f,
            rope=rope,
            mapping=mapping,
            cache_k=cache_k,
            cache_v=cache_v,
            projected=projected,
        )
    from reference_archive import checked_load, s1_key

    seed, context = FIXTURE_KEYS[id(f)]
    saved = checked_load(s1_key(seed, context, rope, mapping, bool(f.locations[0] < 0)))
    if projected is not None:
        assert torch.equal(projected, project(f))
    kc = f.cache_k if cache_k is None else cache_k
    vc = f.cache_v if cache_v is None else cache_v
    kc.view(torch.uint8).reshape(-1, 4, 64)[saved["positions"]] = saved["cache_k"]
    vc.view(torch.uint8).reshape(-1, 4, 64)[saved["positions"]] = saved["cache_v"]
    return saved["output"]


def candidate(
    f, *, rope, mapping, cache_k=None, cache_v=None, ctas=192, workspace=None
):
    return qkvg(
        f.x,
        f.weights["qkvg"],
        f.positions,
        f.cos_sin if rope else None,
        f.cache_k if cache_k is None else cache_k,
        f.cache_v if cache_v is None else cache_v,
        f.locations,
        f.mapping if mapping else None,
        EPS,
        Q_SCALE,
        f.k_scale,
        f.v_scale,
        ctas=ctas,
        workspace=workspace,
    )


def fp64(f, *, rope, mapping, round_stages):
    projected = f.x.double() @ f.weights["qkvg"].double().T

    def rounded(x):
        return x.to(torch.bfloat16).double() if round_stages else x

    q, k, v, gate = rounded(projected).split((2048, 256, 256, 2048), dim=1)

    def norm(x, scale):
        x = x.reshape(8, -1, 64)
        x = rounded(x * (x.square().mean(dim=-1, keepdim=True) + EPS).rsqrt())
        if scale != 1:
            x = rounded(x * scale)
        if rope:
            cs = f.cos_sin[f.positions].double()
            if round_stages:
                cs = cs.to(torch.bfloat16).double()
            cosine = cs[:, :32].repeat_interleave(2, dim=-1)[:, None, :]
            sine = cs[:, 32:].repeat_interleave(2, dim=-1)[:, None, :]
            rotated = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
            x = rounded(rounded(x * cosine) + rounded(rotated * sine))
        return x.reshape(8, -1)

    q, k = norm(q, Q_SCALE), norm(k, 1)
    kc, vc = f.cache_k.clone(), f.cache_v.clone()
    locations = f.locations.clone()
    if mapping:
        valid = locations >= 0
        locations[valid] = f.mapping[locations[valid]]
    valid = locations >= 0
    # uint8 views support indexed assignment where Float8 indexing is unsupported.
    kbytes = (
        (k.reshape(8, 4, 64) / f.k_scale.double())
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    vbytes = (
        (v.reshape(8, 4, 64) / f.v_scale.double())
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    kc.view(torch.uint8).reshape(-1, 4, 64)[locations[valid]] = kbytes[valid]
    vc.view(torch.uint8).reshape(-1, 4, 64)[locations[valid]] = vbytes[valid]
    return (q, k, v, gate), (kc, vc)


def errors(actual, expected):
    a, b = actual.double(), expected.double()
    diff = (a - b).abs()
    return dict(
        max_abs=diff.max().item(),
        rel_linf=(diff.max() / b.abs().max().clamp_min(1e-30)).item(),
        max_element_rel=(diff / b.abs().clamp_min(1e-3)).max().item(),
    )


def check_cache(*, f, kc, vc, rc, rv, mapping):
    # Different GEMM reductions can cross BF16/FP8 rounding thresholds. Full
    # byte mismatches are reported, with numeric bounds and untouched-slot checks.
    for a, b in ((kc, rc), (vc, rv)):
        torch.testing.assert_close(a.float(), b.float(), atol=0.0625, rtol=0.125)
    locations = f.locations.clone()
    valid = locations >= 0
    if mapping:
        locations[valid] = f.mapping[locations[valid]]
    touched = torch.zeros(f.cache_k.numel() // 256, device=f.x.device, dtype=torch.bool)
    touched[locations[locations >= 0]] = True
    for a, b in ((kc, f.cache_k), (vc, f.cache_v)):
        assert torch.equal(
            a.view(torch.uint8).reshape(-1, 256)[~touched],
            b.view(torch.uint8).reshape(-1, 256)[~touched],
        )
    distances = []
    for a, b in ((kc, rc), (vc, rv)):
        ac, bc = a.view(torch.uint8).int(), b.view(torch.uint8).int()
        ac = torch.where(ac < 128, ac, -(ac - 128))
        bc = torch.where(bc < 128, bc, -(bc - 128))
        distances.append(int((ac - bc).abs().max().item()))
    assert max(distances) <= 1
    return dict(
        numeric=[errors(a.float(), b.float()) for a, b in ((kc, rc), (vc, rv))],
        max_code_distance=distances,
        untouched_bytes_equal=True,
    )


def one_case(
    *, context, seed, rope, mapping, ctas, skip_slots=False, head_tagged=False
):
    f = build_layer(context=context, seed=seed)
    if skip_slots:
        # Negative full slots and negative mapped slots must leave cache untouched.
        if mapping:
            f.mapping[f.locations[1]] = -1
        f.locations[0] = -1
    kc, vc = f.cache_k.clone(), f.cache_v.clone()
    rc, rv = f.cache_k.clone(), f.cache_v.clone()
    workspace = make_workspace(f.x.device, ctas=ctas)
    actual = candidate(
        f,
        rope=rope,
        mapping=mapping,
        cache_k=kc,
        cache_v=vc,
        ctas=ctas,
        workspace=workspace,
    )
    assert (
        (workspace.state[-72:] == workspace.state[0]).all().item()
        if head_tagged
        else workspace.state[-72:].all().item()
    ), "bounded pair handoff failed"
    expected = reference(f, rope=rope, mapping=mapping, cache_k=rc, cache_v=rv)
    torch.cuda.synchronize(0)
    record = dict(
        context=context,
        seed=seed,
        rope=rope,
        mapping=mapping,
        ctas=ctas,
        skip_slots=skip_slots,
    )
    record["vs_sglang"] = {
        name: errors(a, b)
        for name, a, b in zip(("q", "k", "v", "gate"), actual, expected)
    }
    record["cache_byte_mismatches"] = [
        int((a.view(torch.uint8) != b.view(torch.uint8)).sum().item())
        for a, b in ((kc, rc), (vc, rv))
    ]
    print(json.dumps(record), flush=True)
    for name, a, b in zip(("q", "k", "v", "gate"), actual, expected):
        torch.testing.assert_close(
            a,
            b,
            atol=0.125 if name == "q" else 0.02,
            rtol=0.02,
            msg=lambda message: name + ": " + message,
        )
    record["cache_vs_sglang"] = check_cache(
        f=f, kc=kc, vc=vc, rc=rc, rv=rv, mapping=mapping
    )
    rounded, rounded_caches = fp64(f, rope=rope, mapping=mapping, round_stages=True)
    ideal, _ = fp64(f, rope=rope, mapping=mapping, round_stages=False)
    record["vs_fp64_rounded"] = {
        name: errors(a, b)
        for name, a, b in zip(("q", "k", "v", "gate"), actual, rounded)
    }
    record["vs_fp64_ideal"] = {
        name: errors(a, b) for name, a, b in zip(("q", "k", "v", "gate"), actual, ideal)
    }
    record["cache_vs_fp64_rounded"] = [
        int((a.view(torch.uint8) != b.view(torch.uint8)).sum().item())
        for a, b in zip((kc, vc), rounded_caches)
    ]
    for a, b, mathematical in zip(actual, rounded, ideal):
        assert errors(a, b)["rel_linf"] < 0.005
        assert errors(a, mathematical)["rel_linf"] < 0.02
    assert record["cache_vs_fp64_rounded"] == [0, 0]
    record["timing"] = graph_times(
        lambda: candidate(
            f,
            rope=rope,
            mapping=mapping,
            cache_k=kc,
            cache_v=vc,
            ctas=ctas,
            workspace=workspace,
        )
    )
    assert (
        (workspace.state[-72:] == workspace.state[0]).all().item()
        if head_tagged
        else workspace.state[-72:].all().item()
    )
    record["sglang_timing"] = (
        None
        if ARCHIVE
        else graph_times(
            lambda: reference(f, rope=rope, mapping=mapping, cache_k=rc, cache_v=rv)
        )
    )
    bounds = json.loads((LOG / "phase3.0-read-and-copy-traffic.json").read_text())
    record["compulsory_read_bound_us"] = bounds["stages"]["S1 QKVG"]["read_bound_us"]
    record["bound_fraction"] = (
        record["compulsory_read_bound_us"] / record["timing"]["median_us"]
    )
    print(json.dumps(record), flush=True)
    return record


@g.jit
def epilogue_only(
    P,
    POS,
    COS,
    LOC,
    MAP,
    KC,
    VC,
    Q,
    K,
    V,
    G,
    KS,
    VS,
    EPS: gl.constexpr,
    SCALE: gl.constexpr,
    ROPE: gl.constexpr,
    HAS_MAP: gl.constexpr,
):
    head_epilogue(
        P,
        gl.program_id(0),
        POS,
        COS,
        LOC,
        MAP,
        KC,
        VC,
        Q,
        K,
        V,
        G,
        KS,
        VS,
        EPS,
        SCALE,
        ROPE,
        HAS_MAP,
    )


def check_epilogue(*, rope, mapping, waves=16):
    f = build_layer(context=32768, seed=7)
    projected = project(f)
    p = projected.float().view(torch.int32)
    kc, vc = f.cache_k.clone(), f.cache_v.clone()
    rc, rv = f.cache_k.clone(), f.cache_v.clone()
    q = torch.empty((8, 2048), device=f.x.device, dtype=torch.bfloat16)
    k = torch.empty((8, 256), device=f.x.device, dtype=torch.bfloat16)
    v, gate = torch.empty_like(k), torch.empty_like(q)
    compiled = epilogue_only[(72,)](
        p,
        f.positions,
        f.cos_sin if rope else None,
        f.locations,
        f.mapping if mapping else None,
        kc,
        vc,
        q,
        k,
        v,
        gate,
        f.k_scale,
        f.v_scale,
        EPS,
        Q_SCALE,
        rope,
        mapping,
        num_warps=waves,
    )
    expected = reference(
        f, rope=rope, mapping=mapping, cache_k=rc, cache_v=rv, projected=projected
    )
    for a, b in zip((q, k, v, gate), expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal(kc.view(torch.uint8), rc.view(torch.uint8))
    assert torch.equal(vc.view(torch.uint8), rv.view(torch.uint8))
    (LOG / f"phase3b.1-epilogue-rope{rope}-map{mapping}.amdgcn").write_text(
        compiled.asm["amdgcn"]
    )
    print(
        json.dumps(
            dict(
                event="matched_projection_epilogue",
                rope=rope,
                mapping=mapping,
                bitwise_outputs_and_cache=True,
            )
        ),
        flush=True,
    )


def profile_stages():
    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.skinny_gemm import (
        _standalone,
    )

    f = build_layer()
    workspace = make_workspace(f.x.device)
    out = torch.empty((8, 4608), device=f.x.device, dtype=torch.bfloat16)
    core = lambda: _standalone[(192,)](
        f.x,
        f.weights["qkvg"],
        out,
        4608,
        8192,
        32,
        64,
        8,
        2,
        ".cs",
        False,
        num_warps=16,
    )
    compiled = core()
    (LOG / "phase3b.1-core.amdgcn").write_text(compiled.asm["amdgcn"])
    q = torch.empty((8, 2048), device=f.x.device, dtype=torch.bfloat16)
    k = torch.empty((8, 256), device=f.x.device, dtype=torch.bfloat16)
    v, gate = torch.empty_like(k), torch.empty_like(q)
    fused = lambda: fused_s1[(192,)](
        f.x,
        f.weights["qkvg"],
        workspace.projection,
        workspace.state,
        f.positions,
        f.cos_sin,
        f.locations,
        f.mapping,
        f.cache_k,
        f.cache_v,
        q,
        k,
        v,
        gate,
        f.k_scale,
        f.v_scale,
        EPS,
        Q_SCALE,
        True,
        True,
        True,
        num_warps=16,
    )
    compiled = fused()
    asm = compiled.asm["amdgcn"]
    assert "SCOPE_SYS" not in asm and "global_wb" not in asm and "global_inv" not in asm
    (LOG / "phase3b.1-fused.amdgcn").write_text(asm)
    result = {"core": graph_times(core), "fused_s1": graph_times(fused)}
    result["overhead_us"] = (
        result["fused_s1"]["median_us"] - result["core"]["median_us"]
    )
    assert workspace.state[-72:].all().item()
    (LOG / "phase3b.1-breakdown.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(event="stage_breakdown", **result)), flush=True)


def main():
    global ARCHIVE
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--saved-references", action="store_true")
    args = parser.parse_args()
    ARCHIVE = args.saved_references
    rows = []
    contexts = (8192,) if args.smoke else (8192, 32768, 131072)
    seeds = (7,) if args.smoke else (7, 20261117)
    for context in contexts:
        for seed in seeds:
            for rope, mapping in ((False, False), (True, True)):
                rows.append(
                    one_case(
                        context=context, seed=seed, rope=rope, mapping=mapping, ctas=192
                    )
                )
    if not args.smoke:
        for rope, mapping in (
            (False, False),
            (True, True),
            (True, False),
            (False, True),
        ):
            check_epilogue(rope=rope, mapping=mapping)
        profile_stages()
        for ctas, rope, mapping in (
            (1, True, True),
            (37, True, False),
            (191, False, True),
        ):
            rows.append(
                one_case(
                    context=8192,
                    seed=17,
                    rope=rope,
                    mapping=mapping,
                    ctas=ctas,
                    skip_slots=True,
                )
            )
    (
        LOG / ("phase3b.1-smoke.json" if args.smoke else "phase3b.1-results.json")
    ).write_text(json.dumps(rows, indent=2) + "\n")
    print(f"PASS {len(rows)} S1 cases", flush=True)


if __name__ == "__main__":
    main()
