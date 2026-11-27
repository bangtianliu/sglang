"""Cold dependent stage windows, event gaps and instrumentation bias.

Gaps are marker-to-marker proxies, not kernel activity timestamps. Both sides
use the same twelve external event records per layer; plain graphs are separate.
"""

import argparse
import ast
import functools
import inspect
import json
import textwrap
import types
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import regular_layer45
import torch
from bench_chained47 import setup
from check_chained47 import archive_check
from diagnose_guard48 import balanced
from layer_fixture import require_gpu0
from reference_archive import gpu

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_chain


def native_stamped(native):
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(regular_layer45.Regular.__call__))
    )
    fn = tree.body[0]
    body = []
    stage = 1
    start = True
    for node in fn.body:
        if stage <= 6 and start:
            body.extend(ast.parse(f"mark({stage}, 0)").body)
            start = False
        if isinstance(node, ast.If) and ast.unparse(node.test) == "mark is not None":
            body.extend(ast.parse(f"mark({stage}, 1)").body)
            stage += 1
            start = True
        else:
            body.append(node)
    assert stage == 7
    fn.body = body
    ast.fix_missing_locations(tree)
    ns = dict(vars(regular_layer45))
    exec(compile(tree, "<T7-native-markers>", "exec"), ns)
    return types.MethodType(ns["__call__"], native)


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--left", type=int, required=True)
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--label", default="phaseT7")
    a = p.parse_args()
    groups = {n: [] for n in ("regular", "tlx")}
    spaces = {n: [] for n in groups}
    retained = []
    enabled = [False]
    events = {
        n: [
            [
                (
                    torch.cuda.Event(enable_timing=True, external=True),
                    torch.cuda.Event(enable_timing=True, external=True),
                )
                for _ in range(6)
            ]
            for _ in range(3)
        ]
        for n in groups
    }
    for i in range(3):
        base, native, oldws, errors, routes = setup(a.context, a.left)
        retained.append((base, native, oldws, errors, routes))
        stamped = native_stamped(native)

        def native_call(native=native, stamped=stamped, i=i):
            if enabled[0]:
                return stamped(
                    mark=lambda s, pos: events["regular"][i][s - 1][pos].record()
                )
            return native()

        groups["regular"].append((native.prepare, native_call, base["regular"][2]))
        spaces["regular"].append(native)
        x, stream, w, m, _ = base["chained"][1].args
        ws = tlx_chain.make_workspace("cuda", context=a.context, left=a.left)
        spaces["tlx"].append(ws)
        old = gpu(
            torch.load(
                Path("/home/bangtliu/coord/phase44-baseline")
                / f"layer-7-{a.context}-{a.left}.pt",
                weights_only=True,
            )["rows"][0]
        )
        groups["tlx"].append(
            (
                functools.partial(
                    tlx_chain.prepare_peers, ws, *native.partials, rank=0
                ),
                functools.partial(tlx_chain.launch, x, stream, w, m, ws, rank=0),
                functools.partial(archive_check, native.f, ws, old, 0, digest=False),
            )
        )
    allocations = {}
    for n in (
        "qkvg",
        "o_proj",
        "joint_router_pooler",
        "w13",
        "w13_scale",
        "w2",
        "w2_scale",
        "unpooler",
    ):
        ptrs = [r[0]["chained"][1].args[2][n].data_ptr() for r in retained]
        assert len(set(ptrs)) == 3
        allocations[n] = ptrs
    cases = {}
    for n, rows in groups.items():

        def action(i, rows=rows):
            for row in rows:
                row[i]()

        cases[n] = tuple(functools.partial(action, i) for i in range(3))
    epochs = {
        "regular": spaces["regular"][0].epochs[0],
        "tlx": spaces["tlx"][0].s3.state.epochs,
    }
    plain = balanced(cases, epochs)
    counts = [0] * 6

    def wrap(fn, s):
        def call(*aa, **kw):
            if not enabled[0]:
                return fn(*aa, **kw)
            i = counts[s] % 3
            counts[s] += 1
            start, end = events["tlx"][i][s]
            start.record()
            try:
                return fn(*aa, **kw)
            finally:
                end.record()

        return call

    def observer(n):
        rows = events[n]
        stage = [
            sum(row[s][0].elapsed_time(row[s][1]) * 1000 for row in rows) / 3
            for s in range(6)
        ]
        gaps = [
            sum(row[s][1].elapsed_time(row[s + 1][0]) * 1000 for row in rows) / 3
            for s in range(5)
        ]
        return dict(
            stage_us=stage,
            gap_us=gaps,
            span_us=sum(row[0][0].elapsed_time(row[5][1]) * 1000 for row in rows) / 3,
        )

    with ExitStack() as stack:
        for s, (mod, api) in enumerate(
            [
                (tlx_chain.tlx_qkvg, "qkvg"),
                (tlx_chain.tlx_attention, "attention"),
                (tlx_chain.tlx_attn_tail, "attention_tail"),
                (tlx_chain.tlx_route, "routing"),
                (tlx_chain.tlx_experts, "s5"),
                (tlx_chain.tlx_tail, "unpool_boundary"),
            ]
        ):
            stack.enter_context(patch.object(mod, api, wrap(getattr(mod, api), s)))
        enabled[0] = True
        stamped = balanced(cases, epochs, observer)
    out = dict(
        context=a.context,
        resources=[tlx_chain.resources(ws) for ws in spaces["tlx"]],
        left=a.left,
        copies=3,
        allocations=allocations,
        plain=plain,
        stamped=stamped,
        limits="External stage-event windows include internal dispatch/flag costs. Inter-stage marker gaps include marker overhead; not raw kernel gaps. Plain/stamped runs quantify instrumentation bias; do not add independently reported medians to infer whole latency.",
    )
    Path(
        "/home/bangtliu/coord/logs", f"{a.label}-attribution-{a.context}-{a.left}.json"
    ).write_text(json.dumps(out, indent=2, default=str) + "\n")
    print("PASS attribution", a.left, flush=True)


if __name__ == "__main__":
    main()
