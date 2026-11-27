"""Cold chain A/B: full-width async S5 vs accepted TLX and Gluon S5.

All other stages use the selected mixed policy. Three independent weight sets,
actual dependent graphs, stage and whole-chain events, balanced order/parity.
"""

import argparse
import functools
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from bench_chained47 import setup
from check_chained47 import archive_check
from diagnose_guard48 import balanced
from layer_fixture import require_gpu0
from paired45 import distribution
from reference_archive import gpu
from tlx_ir import dump
from triton.compiler import CompiledKernel

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_experts,
    independent_experts,
    tlx_chain,
    tlx_experts,
)

ROOT = Path("/home/bangtliu/coord/logs")


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, required=True)
    p.add_argument("--left", type=int, required=True)
    a = p.parse_args()
    groups = {n: [] for n in ("gluon", "normal", "async")}
    spaces = {n: [] for n in groups}
    retained = []
    owners = {}
    active = [None]
    seen = set()
    saved_metadata = CompiledKernel.launch_metadata
    for _ in range(3):
        base, native, oldws, _, _ = setup(a.context, a.left)
        retained.append((base, native, oldws))
        x, stream, w, m, _ = base["chained"][1].args
        old = gpu(
            torch.load(
                Path("/home/bangtliu/coord/phase44-baseline")
                / f"layer-7-{a.context}-{a.left}.pt",
                weights_only=True,
            )["rows"][0]
        )
        selection = tlx_chain.make_workspace(
            "cuda", context=a.context, left=a.left
        ).tlx_stages
        assert 5 in selection
        for label in groups:
            stages = (
                tuple(s for s in selection if s != 5) if label == "gluon" else selection
            )
            ws = tlx_chain.make_workspace(
                "cuda", context=a.context, left=a.left, tlx_stages=stages
            )
            if label != "gluon":
                ws.s5.scale_async = label == "async"
            owners[id(ws.s5)] = label
            spaces[label].append(ws)
            prepare = functools.partial(
                tlx_chain.prepare_peers, ws, *native.partials, rank=0
            )
            call = functools.partial(tlx_chain.launch, x, stream, w, m, ws, rank=0)
            check = functools.partial(archive_check, native.f, ws, old, 0, digest=False)
            groups[label].append((prepare, call, check))
    events = {
        n: [
            (
                torch.cuda.Event(enable_timing=True, external=True),
                torch.cuda.Event(enable_timing=True, external=True),
            )
            for _ in range(3)
        ]
        for n in groups
    }
    counts = dict.fromkeys(groups, 0)

    def marked(fn):
        def call(*aa, **kw):
            label = owners[id(aa[6])]
            i = counts[label] % 3
            counts[label] += 1
            start, end = events[label][i]
            start.record()
            active[0] = label
            try:
                return fn(*aa, **kw)
            finally:
                active[0] = None
                end.record()

        return call

    def metadata(kernel, *aa, **kw):
        out = saved_metadata(kernel, *aa, **kw)
        if active[0] is not None:
            fn = kernel.src.fn.fn
            mod = fn.__module__.rsplit(".", 1)[-1]
            phase = {
                "norm_phase": "norm",
                "sum_reduce": "sum_AR",
                "_launch": "common",
            }.get(fn.__name__)
            if fn.__name__ == "expert_phase":
                idx = kernel.src.fn.arg_names.index("FIRST")
                phase = (
                    "w13"
                    if kernel.src.constants.get(
                        (idx,), kernel.src.constants.get("FIRST")
                    )
                    else "w2"
                )
            key = (active[0], phase)
            if (
                mod in ("common_experts", "independent_experts", "tlx_experts")
                and phase
                and key not in seen
            ):
                dump(kernel, f"T4-TX-S5-{a.context}-{a.left}/" + phase, active[0])
                seen.add(key)
        return out

    cases = {}
    for label, rows in groups.items():

        def action(index, rows=rows):
            for row in rows:
                row[index]()

        cases[label] = tuple(functools.partial(action, i) for i in range(3))

    def observer(label):
        return dict(
            stage_us=sum(s.elapsed_time(e) * 1000 for s, e in events[label]) / 3
        )

    pairs = {}
    with ExitStack() as stack:
        stack.enter_context(patch.object(CompiledKernel, "launch_metadata", metadata))
        for module in (tlx_experts, common_experts, independent_experts):
            stack.enter_context(patch.object(module, "s5", marked(module.s5)))
        for ref in ("normal", "gluon"):
            r = balanced(
                {ref: cases[ref], "async": cases["async"]},
                {n: spaces[n][0].s3.state.epochs for n in (ref, "async")},
                observer,
            )
            r["stage_times"] = {
                n: {
                    o: distribution(
                        [
                            v["stage_us"]
                            for v in r["raw"]
                            if v["label"] == n and v["order"] == o
                        ]
                    )
                    for o in ("AB", "BA")
                }
                for n in (ref, "async")
            }
            r["stage_ratios"] = {
                o: r["stage_times"]["async"][o]["median_us"]
                / r["stage_times"][ref][o]["median_us"]
                for o in ("AB", "BA")
            }
            r["chain_ratios"] = {
                o: r["timing"]["async"]["by_order"][o]["median_us"]
                / r["timing"][ref]["by_order"][o]["median_us"]
                for o in ("AB", "BA")
            }
            pairs[ref] = r
            print(
                json.dumps(
                    dict(
                        reference=ref,
                        stage_ratios=r["stage_ratios"],
                        chain_ratios=r["chain_ratios"],
                    )
                ),
                flush=True,
            )
    keep = (
        all(v < 1 for v in pairs["normal"]["stage_ratios"].values())
        and all(v <= 1 for v in pairs["normal"]["chain_ratios"].values())
        and all(v <= 1.02 for v in pairs["gluon"]["stage_ratios"].values())
    )
    result = dict(
        context=a.context,
        left=a.left,
        copies=3,
        pairs=pairs,
        keep=keep,
        resources={
            n: [tlx_chain.resources(ws) for ws in copies]
            for n, copies in spaces.items()
        },
        protocol="Only S5 differs; other stages use current mixed selection. Three distinct weights, both orders/parities, stage and whole-chain times; keep if faster than accepted TLX in both stage and chain orders and <=2% vsGluon.",
    )
    (ROOT / f"phaseT4-S5-variant-{a.context}-{a.left}.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )
    print(json.dumps(dict(keep=keep)), flush=True)


if __name__ == "__main__":
    main()
