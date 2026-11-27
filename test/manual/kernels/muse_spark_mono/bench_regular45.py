"""Phase4.5.1 actual dependent native layer graph versus accepted monokernel."""

import argparse
import json
from pathlib import Path

import torch
from layer_case import make
from layer_fixture import require_gpu0
from layer_reference import build, named_check, routing_check, teacher_reference
from paired45 import measure
from reference_archive import cpu, gpu
from regular_layer45 import Regular
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import layer

LOG = Path("/home/bangtliu/coord/logs")


@g.jit
def phase45_profile_begin(X):
    gl.store(X, 0)


@g.jit
def phase45_profile_end(X):
    gl.store(X, 1)


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--left", type=int, default=-1)
    p.add_argument("--trace", action="store_true")
    a = p.parse_args()
    tag = f"{a.context}-{a.left}"
    artifact = LOG / f"phase45-native-{tag}.pt"
    f, w, m = make(7, a.context, a.left, 0)
    if a.trace:
        saved = gpu(torch.load(artifact, weights_only=True))
        parts = (saved["partial3"], saved["partial5"])
    else:
        ref = build(f, w, 7, a.context, a.left, expert_fp64=False)
        parts = (ref["partial3"], ref["partial5"])
    native = Regular(f, w, m, a.context, a.left, parts)
    native.prepare()
    native()
    torch.cuda.synchronize()
    assert all(bool(s.eq(1).all()) for s in native.status)
    if a.trace:
        for n, t in native.outputs.items():
            assert torch.equal(t, saved["outputs"][n]), ("profile native", n)
        marker = torch.empty((1,), device="cuda", dtype=torch.int32)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            native.prepare()
            phase45_profile_begin[(1,)](marker, num_warps=4)
            native()
            phase45_profile_end[(1,)](marker, num_warps=4)
        graph.replay()
        torch.cuda.synchronize()
        assert marker.item() == 1
        for n, t in native.outputs.items():
            assert torch.equal(t, saved["outputs"][n]), ("profile replay", n)
        print(
            json.dumps(dict(event="profile-pass", context=a.context, left=a.left)),
            flush=True,
        )
        return
    # Validate routing before values, using the already-approved near-tie policy.
    routes = routing_check(native, ref, w)
    target = ref
    if any(not r["set_equal"] for r in routes):
        native.context = f"phase45native{a.context}"  # diagnostic filename only
        try:
            target = teacher_reference(f, w, ref, native, 0, routes)
        finally:
            native.context = a.context
    errors = {
        n: named_check(
            t,
            target[n],
            0.008
            if n == "attn"
            else (
                0.02
                if target.get("teacher_forced") and n in ("x6", "out", "stream")
                else 0.03
            ),
            "native " + n,
            {"_rank": 0},
        )
        for n, t in native.outputs.items()
    }
    # Native physical cache and monokernel compact SWA cache have the same logical slots.
    untouched = torch.ones((a.context,), device="cuda", dtype=torch.bool)
    untouched[f.locations] = False
    for cache, initial in ((native.k, native.initial_k), (native.v, native.initial_v)):
        assert torch.equal(
            cache.view(torch.uint8).reshape(a.context, 4, 64)[untouched],
            initial.reshape(a.context, 4, 64)[untouched],
        ), "native stray cache write"
    baseline = {n: t.clone() for n, t in native.outputs.items()}
    native_ids = native.s4.ids.clone()
    ws = layer.make_workspace("cuda", context=a.context, left=a.left)

    def pm():
        layer.prepare_peers(ws, *parts, rank=0)

    def mono():
        layer.launch(f.x, f.stream, w, m, ws, rank=0)

    pm()
    mono()
    torch.cuda.synchronize()
    from layer_reference import compare

    routes_m = routing_check(ws, ref, w)
    target_m = (
        teacher_reference(f, w, ref, ws, 0, routes_m)
        if any(not r["set_equal"] for r in routes_m)
        else ref
    )
    compare(f, ws, target_m, w)
    mono_out = {n: t.clone() for n, t in ws.buffers.items() if not n.startswith("f")}
    mono_ids = ws.s4.ids.clone()

    def cn():
        assert all(bool(s.eq(1).all()) for s in native.status)
        assert torch.equal(native.s4.ids, native_ids), "native replay routing"
        for n, t in native.outputs.items():
            assert torch.equal(t, baseline[n]), ("native replay", n)

    def cm():
        assert bool(ws.status[:6].eq(1).all()), layer.diagnostic_records(ws)
        assert torch.equal(ws.s4.ids, mono_ids), "mono replay routing"
        for n, t in mono_out.items():
            assert torch.equal(ws.buffers[n], t), ("mono replay", n)

    torch.save(
        cpu(dict(partial3=parts[0], partial5=parts[1], outputs=baseline)), artifact
    )
    result, graphs = measure(
        dict(regular=(native.prepare, native, cn), mono=(pm, mono, cm))
    )
    result.update(
        context=a.context,
        left=a.left,
        rank=0,
        regular_errors=errors,
        routing=routes,
        regular_kernel_path="native SGLang/AITER/uTLX + same standin.reduce_row collective",
        mono_over_regular=result["mono"]["median_us"] / result["regular"]["median_us"],
    )
    (LOG / f"phase45-regular-{tag}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                n: {k: v for k, v in r.items() if k != "samples_by_order_us"}
                if isinstance(r, dict)
                else r
                for n, r in result.items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
