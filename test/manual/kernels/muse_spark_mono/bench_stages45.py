"""Stage event ranges in paired graphs: native dependent chain vs standalone stages.

Host AST instrumentation inserts only event records; device math is unchanged.
Broad stage event ranges include dispatch/event overhead, not hardware counters.
"""

import argparse
import ast
import inspect
import json
import textwrap
import types

import bench_layer_stages as oldbench
import regular_layer45
import torch
from layer_case import make
from layer_fixture import require_gpu0
from layer_reference import build
from mono_case45 import LOG
from paired45 import distribution, measure
from reference_archive import gpu
from regular_layer45 import Regular


def freeze(fn):
    if fn is None:
        return None
    # Snapshot cells because the existing factory reassigns its workspace variable.
    cells = tuple(
        (lambda v: lambda: v)(c.cell_contents).__closure__[0]
        for c in fn.__closure__ or ()
    )
    return types.FunctionType(
        fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, cells
    )


def own_stages(f, w, m, ref, context, left):
    captures = []

    def collect(call, prepare=None):
        captures.append((freeze(prepare), freeze(call)))
        return {"median_us": 0, "p99_us": 0}

    tree = ast.parse(inspect.getsource(oldbench.benchmark))
    fn = tree.body[0]
    index = next(
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.FunctionDef) and n.name == "ns1"
    )
    fn.body = fn.body[:index] + [ast.Return(ast.Constant(None))]
    ast.fix_missing_locations(tree)
    ns = dict(vars(oldbench))
    ns["timed"] = collect
    exec(compile(tree, "<phase45-existing-standalone-stage-factory>", "exec"), ns)
    ns["benchmark"](f, w, m, ref, context, left, 0)
    assert len(captures) == 6
    return captures


def native_stamped(native, events):
    tree = ast.parse(textwrap.dedent(inspect.getsource(Regular.__call__)))
    fn = tree.body[0]
    body = []
    count = 0
    # Stage boundaries follow exactly the native chain built in step4.5.1.
    for node in fn.body:
        body.append(node)
        if isinstance(node, ast.Assign):
            names = {x.id for x in ast.walk(node.targets[0]) if isinstance(x, ast.Name)}
            if names == {"projected"}:
                body.insert(
                    len(body) - 1, ast.parse("self.stage_events[0].record()").body[0]
                )
            if names in (
                {"qo", "ko"},
                {"x4", "mid"},
                {"scores", "ids", "pool", "table"},
                {"x6"},
                {"out", "stream"},
            ):
                # Attention is a call statement, inserted below after unified_attention.
                count += 1
                body.extend(ast.parse(f"self.stage_events[{count}].record()").body)
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "unified_attention"
        ):
            count += 1
            body.extend(ast.parse(f"self.stage_events[{count}].record()").body)
    assert count == 6, count
    fn.body = body
    ast.fix_missing_locations(tree)
    ns = dict(vars(regular_layer45))
    exec(compile(tree, "<phase45-native-stage-events>", "exec"), ns)
    native.stage_events = events
    return types.MethodType(ns["__call__"], native)


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--left", type=int, default=-1)
    a = p.parse_args()
    f, w, m = make(7, a.context, a.left, 0)
    ref = build(f, w, 7, a.context, a.left, expert_fp64=False)
    stages = own_stages(f, w, m, ref, a.context, a.left)
    native = Regular(f, w, m, a.context, a.left, (ref["partial3"], ref["partial5"]))
    saved = gpu(
        torch.load(LOG / f"phase45-native-{a.context}-{a.left}.pt", weights_only=True)
    )
    events = {
        name: [torch.cuda.Event(enable_timing=True, external=True) for _ in range(7)]
        for name in ("standalone", "native")
    }

    def prepare():
        for prep, _ in stages:
            if prep:
                prep()

    def own():
        events["standalone"][0].record()
        for i, (_, call) in enumerate(stages):
            call()
            events["standalone"][i + 1].record()

    def own_check():
        for i, (_, call) in enumerate(stages):
            scope = dict(
                zip(
                    call.__code__.co_freevars,
                    (c.cell_contents for c in call.__closure__),
                )
            )
            ws = scope["ws"]
            status = ws.status if i in (0, 1) else ws.state.status
            assert bool(status.eq(1).all()), ("stage status", i + 1)

    def native_check():
        assert all(bool(s.eq(1).all()) for s in native.status)
        for name, t in native.outputs.items():
            assert torch.equal(t, saved["outputs"][name]), ("native stage replay", name)

    native_call = native_stamped(native, events["native"])
    samples = {
        label: {f"S{s + 1}": {"AB": [], "BA": []} for s in range(6)} for label in events
    }

    def observe(label, order):
        for i in range(6):
            samples[label][f"S{i + 1}"][order].append(
                events[label][i].elapsed_time(events[label][i + 1]) * 1000
            )

    result, _ = measure(
        {
            "standalone": (prepare, own, own_check),
            "native": (native.prepare, native_call, native_check),
        },
        observer=observe,
    )
    active_ids = ref["ids"][((ref["ids"] >= 0) & (ref["ids"] < 64))]
    unique = int(active_ids.unique().numel())
    # Minimum weight/scale bytes and logical KV bytes, excluding workspaces/flags.
    byte_floor = [
        4608 * 8192 * 2,
        min(a.context, a.left + 8 if a.left >= 0 else a.context) * 4 * 64 * 2,
        8192 * 2048 * 2,
        4608 * 8192 * 2,
        unique * 3 * 4096 * 4096 * 17 // 32,
        8192 * 4096 * 2,
    ]
    details = {}
    for label, stages_data in samples.items():
        details[label] = {}
        for i, (stage, orders) in enumerate(stages_data.items()):
            d = distribution(orders["AB"] + orders["BA"])
            d.update(
                by_order={k: distribution(v) for k, v in orders.items()},
                samples_by_order_us=orders,
                minimum_bytes=byte_floor[i],
                effective_TB_s=byte_floor[i] / d["median_us"] / 1e6,
            )
            details[label][stage] = d
    output = dict(
        context=a.context,
        left=a.left,
        rank=0,
        active_local_experts=unique,
        total=result,
        stages=details,
        event_overhead="7 external event nodes per side, broad stage ranges; not hardware DRAM bandwidth",
        own_inputs="fixed FP64/native reference tensors per standalone stage",
        native_inputs="true dependent native chain",
    )
    (LOG / f"phase45-stage-bandwidth-{a.context}-{a.left}.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                label: {
                    s: dict(us=v["median_us"], effective_TB_s=v["effective_TB_s"])
                    for s, v in d.items()
                }
                for label, d in details.items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
