"""Cold rotating-weight TLX stage timing inside the dependent six-stage chain."""

import argparse
import functools
import json
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch
from bench_chained47 import setup
from check_chained47 import archive_check
from diagnose_guard48 import balanced
from layer_fixture import require_gpu0
from paired45 import distribution

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_attn_tail,
    common_experts,
    common_qkvg,
    common_route,
    independent_attn_tail,
    independent_experts,
    independent_qkvg,
    independent_tail,
    split_attention,
    tlx_attention,
    tlx_attn_tail,
    tlx_chain,
    tlx_experts,
    tlx_qkvg,
    tlx_route,
    tlx_tail,
)


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, choices=(1, 2, 3, 4, 5, 6), default=6)
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--left", type=int, default=-1)
    p.add_argument("--log-prefix", default="phaseT1")
    p.add_argument("--ir-prefix")
    p.add_argument("--unpinned", action="store_true")
    p.add_argument("--mixed-context", action="store_true")
    a = p.parse_args()
    groups = {"gluon": [], "tlx": []}
    spaces = {n: [] for n in groups}
    retained = []
    for _ in range(3):
        base, native, old_ws, _, _ = setup(a.context, a.left)
        retained.append((base, native, old_ws))
        if not a.mixed_context:
            groups["gluon"].append(base["chained"])
            spaces["gluon"].append(old_ws)
        f = base["chained"][1]
        x, stream, w, m, _ = f.args
        ws = tlx_chain.make_workspace(
            "cuda",
            context=a.context,
            left=a.left,
            tlx_stages=None if a.mixed_context else (a.stage,),
        )
        if a.mixed_context:
            assert a.stage in ws.tlx_stages
        old = torch.load(
            Path("/home/bangtliu/coord/phase44-baseline")
            / f"layer-7-{a.context}-{a.left}.pt",
            weights_only=True,
        )["rows"][0]
        from reference_archive import gpu

        old = gpu(old)
        if a.mixed_context:
            glws = tlx_chain.make_workspace(
                "cuda",
                context=a.context,
                left=a.left,
                tlx_stages=tuple(x for x in ws.tlx_stages if x != a.stage),
            )
            glprepare = functools.partial(
                tlx_chain.prepare_peers, glws, *native.partials, rank=0
            )
            glcall = functools.partial(tlx_chain.launch, x, stream, w, m, glws, rank=0)
            glcheck = functools.partial(
                archive_check, native.f, glws, old, 0, digest=False
            )
            groups["gluon"].append((glprepare, glcall, glcheck))
            spaces["gluon"].append(glws)
        prepare = functools.partial(
            tlx_chain.prepare_peers, ws, *native.partials, rank=0
        )
        call = functools.partial(tlx_chain.launch, x, stream, w, m, ws, rank=0)
        fixture = native.f
        check = functools.partial(archive_check, fixture, ws, old, 0, digest=False)
        groups["tlx"].append((prepare, call, check))
        spaces["tlx"].append(ws)
    cases = {}
    for label, rows in groups.items():

        def action(index, rows=rows):
            for row in rows:
                row[index]()

        cases[label] = tuple(functools.partial(action, i) for i in range(3))
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
    counters = dict.fromkeys(groups, 0)
    active = [None]
    seen_ir = set()

    def marked(fn, label):
        def call(*args, **kwargs):
            i = counters[label] % 3
            counters[label] += 1
            s, e = events[label][i]
            s.record()
            previous = active[0]
            active[0] = label
            try:
                if label == "tlx" and a.stage in (3, 6):
                    kwargs["pin_layout"] = not a.unpinned
                v = fn(*args, **kwargs)
            finally:
                active[0] = previous
            e.record()
            return v

        return call

    def observer(label):
        return dict(
            stage_us=sum(s.elapsed_time(e) * 1000 for s, e in events[label]) / 3
        )

    from tlx_ir import dump
    from triton.compiler import CompiledKernel

    original_metadata = CompiledKernel.launch_metadata

    def metadata(kernel, *args, **kwargs):
        result = original_metadata(kernel, *args, **kwargs)
        if a.ir_prefix and active[0] is not None:
            fn = getattr(getattr(kernel.src, "fn", None), "fn", None)
            module = getattr(fn, "__module__", "").rsplit(".", 1)[-1]
            expected = (
                {"common_experts", "independent_experts", "tlx_experts"}
                if a.stage == 5
                else {"common_route", "tlx_route", "tlx_projection"}
                if a.stage == 4
                else {"split_attention", "tlx_attention"}
                if a.stage == 2
                else {
                    "wide_projection",
                    "tlx_projection",
                    "independent_qkvg",
                    "common_qkvg",
                    "tlx_qkvg",
                }
                if a.stage == 1
                else {
                    "wide_projection",
                    "tlx_projection",
                    "independent_tail",
                    "tlx_tail",
                }
                if a.stage == 6
                else {
                    "wide_projection",
                    "tlx_projection",
                    "independent_attn_tail",
                    "tlx_attn_tail",
                    "common_attn_tail",
                }
            )
            phase = {
                "project": "projection",
                "qkvg_heads": "heads",
                "finish_rows": "rows",
                "slots": "slots",
                "partials": "partial",
                "merge": "merge",
                "finish": "boundary",
                "normalize": "norm_gate",
                "s3_boundary": "AR_boundary",
                "_launch": "common",
            }.get(getattr(fn, "__name__", ""))
            if a.stage == 5:
                name = getattr(fn, "__name__", "")
                phase = {
                    "norm_phase": "norm",
                    "sum_reduce": "sum_AR",
                    "_launch": "common",
                }.get(name)
                if name == "expert_phase":
                    idx = kernel.src.fn.arg_names.index("FIRST")
                    first = kernel.src.constants.get(
                        (idx,), kernel.src.constants.get("FIRST")
                    )
                    phase = "w13" if first else "w2"
            key = (active[0], phase)
            if module in expected and phase and key not in seen_ir:
                dump(kernel, a.ir_prefix + "/" + phase, active[0])
                seen_ir.add(key)
        return result

    with ExitStack() as stack:
        if a.ir_prefix:
            stack.enter_context(
                patch.object(CompiledKernel, "launch_metadata", metadata)
            )
        modules = (
            (
                (common_experts, "gluon"),
                (independent_experts, "gluon"),
                (tlx_experts, "tlx"),
            )
            if a.stage == 5
            else ((common_route, "gluon"), (tlx_route, "tlx"))
            if a.stage == 4
            else ((split_attention, "gluon"), (tlx_attention, "tlx"))
            if a.stage == 2
            else (
                (independent_qkvg, "gluon"),
                (common_qkvg, "gluon"),
                (tlx_qkvg, "tlx"),
            )
            if a.stage == 1
            else ((independent_tail, "gluon"), (tlx_tail, "tlx"))
            if a.stage == 6
            else (
                (independent_attn_tail, "gluon"),
                (common_attn_tail, "gluon"),
                (tlx_attn_tail, "tlx"),
            )
        )
        name = (
            "s5"
            if a.stage == 5
            else "routing"
            if a.stage == 4
            else "attention"
            if a.stage == 2
            else "qkvg"
            if a.stage == 1
            else "unpool_boundary"
            if a.stage == 6
            else "attention_tail"
        )
        for module, label in modules:
            stack.enter_context(
                patch.object(module, name, marked(getattr(module, name), label))
            )
        result = balanced(
            cases, {n: v[0].s3.state.epochs for n, v in spaces.items()}, observer
        )
    table = {
        n: {
            o: distribution(
                [
                    r["stage_us"]
                    for r in result["raw"]
                    if r["label"] == n and r["order"] == o
                ]
            )
            for o in ("AB", "BA")
        }
        for n in groups
    }
    ratios = {
        o: table["tlx"][o]["median_us"] / table["gluon"][o]["median_us"]
        for o in ("AB", "BA")
    }
    result.update(
        mixed_context=a.mixed_context,
        selected_stages={n: rows[0].tlx_stages for n, rows in spaces.items()}
        if a.mixed_context
        else None,
        chain_ratios={
            o: result["timing"]["tlx"]["by_order"][o]["median_us"]
            / result["timing"]["gluon"]["by_order"][o]["median_us"]
            for o in ("AB", "BA")
        },
        stage=a.stage,
        pin_layout=not a.unpinned,
        context=a.context,
        left=a.left,
        copies=3,
        stage_timing=table,
        resources={
            label: [
                list(
                    (
                        getattr(ws, f"s{a.stage}")
                        if a.stage in (1, 2)
                        else getattr(ws, f"s{a.stage}").state
                    ).admitted.values()
                )
                for ws in copies
            ]
            for label, copies in spaces.items()
        },
        ir_specializations="actual Failure-pointer chain calls"
        if a.ir_prefix
        else None,
        ratios=ratios,
        gate_pass=all(v <= 1.02 for v in ratios.values()),
        protocol="Three independently allocated weight sets; dependent full layer graphs; external events bracket the selected stage; both orders and epoch parities balanced. Outer graph time includes identical instrumentation; stage time per copy.",
    )
    path = (
        Path("/home/bangtliu/coord/logs")
        / f"{a.log_prefix}-S{a.stage}-chain-{a.context}-{a.left}.json"
    )
    # Preserve every run before updating the conventional latest-result path.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = path.with_name(path.stem + "-" + run_id + path.suffix)
    result["archive"] = str(archive)
    payload = json.dumps(result, indent=2, default=str) + "\n"
    archive.write_text(payload)
    path.write_text(payload)
    print(
        json.dumps(
            dict(
                stage_timing=table,
                ratios=ratios,
                chain_ratios=result["chain_ratios"],
                gate_pass=result["gate_pass"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
