"""Cold distinct-weight dependent two-layer TLX graph vs true regular graph.

Both use fixed GPU0 virtual-peer stand-ins. Layer2 consumes layer1 output and
residual. This is not a physical distributed or serving throughput experiment.
"""

import argparse
import functools
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from bench_chained47 import setup
from diagnose_guard48 import balanced
from layer_fixture import require_gpu0
from tlx_launch_audit import audited_tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_chain as chain,
)

LOG = Path("/home/bangtliu/coord/logs")


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, required=True)
    p.add_argument("--left", type=int, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--t8-reference", type=Path)
    a = p.parse_args()
    reference = None
    if a.t8_reference is not None:
        manifest = json.loads((a.t8_reference / "manifest.json").read_text())

        def archived(name):
            path = a.t8_reference / (name + ".py")
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest()
                == manifest["files"][name]["sha256"]
            )
            spec = importlib.util.spec_from_file_location(
                "tlx_two_reference_" + name, path
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod

        reference = archived("chain")
        reference.tlx_attn_tail = archived("s3")
        reference.tlx_route = archived("s4")
    rows = {n: [] for n in ("regular", "tlx")}
    if reference is not None:
        rows["t8"] = []
    reference_sets = []
    retained = []
    epochs = {}
    spaces = []
    native_sets = []
    snapshots = {}
    allocations = []
    for i in range(3):
        entries = []
        for layer in range(2):
            base, native, _, _, _ = setup(a.context, a.left)
            retained.append((base, native))
            x, stream, w, m, _ = base["chained"][1].args
            if layer == 1:
                for name in ("qkvg", "o_proj", "joint_router_pooler", "unpooler"):
                    w[name] = -w[name]
                for name in ("w13", "w13_scale", "w2", "w2_scale"):
                    w[name] = w[name].roll(1, 0).contiguous()
                f = native.f
                native.f = SimpleNamespace(
                    **{
                        k: getattr(f, k)
                        for k in (
                            "x",
                            "stream",
                            "positions",
                            "locations",
                            "k_scale",
                            "v_scale",
                            "query_start",
                            "seq_lens",
                            "page_table",
                        )
                    }
                )
            ws = chain.make_workspace("cuda", context=a.context, left=a.left)
            entries.append((native, x, stream, w, m, ws))
        if reference is not None:
            reference_spaces = [
                reference.make_workspace("cuda", context=a.context, left=a.left)
                for _ in entries
            ]
            reference_sets.append(reference_spaces)
        native_sets.append([e[0] for e in entries])
        spaces.append([e[5] for e in entries])
        allocations.append([e[3]["qkvg"].data_ptr() for e in entries])
        assert allocations[-1][0] != allocations[-1][1]

        def regular_prepare(entries=entries):
            for n, *_ in entries:
                n.prepare()

        def regular_call(entries=entries):
            first = entries[0][0]
            second = entries[1][0]
            out, stream = first()
            second.f.x = out
            second.f.stream = stream
            second()

        def regular_check(entries=entries):
            for n, *_ in entries:
                assert all(
                    torch.isfinite(t).all()
                    for t in n.outputs.values()
                    if t.is_floating_point()
                ), "regular two-layer nonfinite"

        def tlx_prepare(entries=entries):
            for n, x, s, w, m, ws in entries:
                chain.prepare_peers(ws, *n.partials, rank=0)

        def tlx_call(entries=entries):
            n, x, s, w, m, ws = entries[0]
            out, stream = chain.launch(x, s, w, m, ws, rank=0)
            n, x, s, w, m, ws = entries[1]
            chain.launch(out, stream, w, m, ws, rank=0)

        def tlx_check(entries=entries, i=i):
            for layer, (_, _, _, _, _, ws) in enumerate(entries):
                assert ws.failure.eq(0).all(), (
                    "TLX failure",
                    i,
                    layer,
                    ws.failure.tolist(),
                )
                for name in ("out", "stream", "q", "gate", "x4", "x6", "attn"):
                    value = ws.buffers[name]
                    assert torch.isfinite(value).all(), (
                        "TLX nonfinite",
                        i,
                        layer,
                        name,
                    )
                    key = (i, layer, name)
                    if key not in snapshots:
                        snapshots[key] = value.clone()
                    else:
                        assert torch.equal(
                            value.view(torch.uint8), snapshots[key].view(torch.uint8)
                        ), ("TLX two-layer replay", key)

        if reference is not None:

            def reference_prepare(entries=entries, rr=reference_spaces):
                for entry, ws in zip(entries, rr):
                    reference.prepare_peers(ws, *entry[0].partials, rank=0)

            def reference_call(entries=entries, rr=reference_spaces):
                n, x, stream, w, m, _ = entries[0]
                out, updated = reference.launch(x, stream, w, m, rr[0], rank=0)
                n, x, stream, w, m, _ = entries[1]
                reference.launch(out, updated, w, m, rr[1], rank=0)

            def reference_check(rr=reference_spaces):
                for ws in rr:
                    assert not bool(ws.failure.any()), "T8 two-layer Failure"
                    assert bool(torch.isfinite(ws.buffers["out"]).all()), (
                        "T8 two-layer output"
                    )

            rows["t8"].append((reference_prepare, reference_call, reference_check))
        rows["regular"].append((regular_prepare, regular_call, regular_check))
        rows["tlx"].append((tlx_prepare, tlx_call, tlx_check))
    for layer in (0, 1):
        assert len({p[layer] for p in allocations}) == 3
    cases = {}
    for label, rr in rows.items():

        def action(index, rr=rr):
            for row in rr:
                row[index]()

        cases[label] = tuple(functools.partial(action, j) for j in range(3))
    epoch_map = dict(
        regular=native_sets[0][0].epochs[0], tlx=spaces[0][0].s3.state.epochs
    )
    result = balanced({n: cases[n] for n in ("regular", "tlx")}, epoch_map)
    if reference is not None:
        previous = balanced(
            {n: cases[n] for n in ("t8", "tlx")},
            dict(t8=reference_sets[0][0].s3.state.epochs, tlx=epoch_map["tlx"]),
        )
        result["t8_pair"] = previous
    result["ratios"] = {
        o: result["timing"]["tlx"]["by_order"][o]["median_us"]
        / result["timing"]["regular"]["by_order"][o]["median_us"]
        for o in ("AB", "BA")
    }
    if reference is not None:
        result["ratios_t8"] = {
            o: previous["timing"]["tlx"]["by_order"][o]["median_us"]
            / previous["timing"]["t8"]["by_order"][o]["median_us"]
            for o in ("AB", "BA")
        }
        for i, (current, previous) in enumerate(zip(spaces, reference_sets)):
            for layer, (new, old) in enumerate(zip(current, previous)):
                for name in ("out", "stream", "q", "gate", "x4", "x6", "attn"):
                    assert torch.equal(
                        new.buffers[name].view(torch.uint8),
                        old.buffers[name].view(torch.uint8),
                    ), ("two-layer T8 exact", i, layer, name)
                assert torch.equal(new.s4.ids, old.s4.ids), (
                    "two-layer T8 routing",
                    i,
                    layer,
                )
        result["exact_t8_layers"] = 6
    result["per_two_layer_us"] = {
        n: {o: v["by_order"][o]["median_us"] / 3 for o in ("AB", "BA")}
        for n, v in result["timing"].items()
    }
    work = {n: [] for n in rows}
    if reference is not None:
        work["t8"] = [
            [
                int(
                    torch.unique(ws.s4.ids[(ws.s4.ids >= 0) & (ws.s4.ids < 64)]).numel()
                )
                for ws in rr
            ]
            for rr in reference_sets
        ]
    for nn, ww in zip(native_sets, spaces):
        work["regular"].append(
            [
                len(set(int(v) for v in n.s4.ids.cpu().flatten() if 0 <= int(v) < 64))
                for n in nn
            ]
        )
        work["tlx"].append(
            [
                len(set(int(v) for v in ws.s4.ids.cpu().flatten() if 0 <= int(v) < 64))
                for ws in ww
            ]
        )
    result.update(
        context=a.context,
        left=a.left,
        copies=3,
        t8_reference_manifest=manifest if reference is not None else None,
        layers=2,
        distinct_weight_pointers=allocations,
        work=work,
        protocol="Dependent two-layer graph, distinct actual layer weights (negated dense matrices and consistently permuted experts for layer2), three rotating copies, both orders/parities. Stand-in setup excluded. Times divided by3 are TWO-LAYER totals. Regular second input is its own first output; TLX second input its first output. Existing single-layer numerical matrix and exact TLX two-layer replay checks are separate from the native comparison; layer2 routing/work counts retained.",
    )
    (LOG / f"{a.label}-two-layer-{a.context}-{a.left}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("PASS two-layer", result["ratios"], "experts", work, flush=True)


if __name__ == "__main__":
    with audited_tlx("two-layer-" + "-".join(sys.argv[1:]).replace("/", "_")):
        main()
