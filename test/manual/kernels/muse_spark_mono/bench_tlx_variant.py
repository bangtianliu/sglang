"""Cold paired current-vs-candidate TLX stage inside the all-TLX chain."""

import argparse
import functools
import importlib.util
import inspect
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import msgspec
import torch
from bench_chained47 import setup
from check_chained47 import archive_check
from diagnose_guard48 import balanced
from layer_fixture import require_gpu0
from msgspec.structs import replace
from paired45 import distribution
from reference_archive import gpu
from tlx_ir import dump
from triton.compiler import CompiledKernel

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_chain


def clone_workspace_headers(value):
    """Separate admission records while aliasing the exact same GPU buffers."""
    if isinstance(value, msgspec.Struct):
        return replace(
            value,
            **{
                n: clone_workspace_headers(getattr(value, n))
                for n in value.__struct_fields__
            },
        )
    if isinstance(value, dict):
        return {k: clone_workspace_headers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_workspace_headers(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_workspace_headers(v) for v in value)
    return value


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, required=True, choices=(1, 2, 3, 4, 5, 6))
    p.add_argument("--context", type=int, required=True)
    p.add_argument("--left", type=int, required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate")
    p.add_argument("--baseline-core")
    p.add_argument("--label", required=True)
    p.add_argument("--whole-only", action="store_true")
    p.add_argument("--reverse-allocation", action="store_true")
    p.add_argument("--shared-workspace", action="store_true")
    p.add_argument("--fuse-norm-sum", action="store_true")
    a = p.parse_args()
    attr, api = {
        1: ("tlx_qkvg", "qkvg"),
        2: ("tlx_attention", "attention"),
        3: ("tlx_attn_tail", "attention_tail"),
        4: ("tlx_route", "routing"),
        5: ("tlx_experts", "s5"),
        6: ("tlx_tail", "unpool_boundary"),
    }[a.stage]
    candidate = getattr(tlx_chain, attr)
    if a.candidate:
        candidate_spec = importlib.util.spec_from_file_location(
            "tlx_custom_candidate", a.candidate
        )
        candidate = importlib.util.module_from_spec(candidate_spec)
        sys.modules[candidate_spec.name] = candidate
        candidate_spec.loader.exec_module(candidate)
    spec = importlib.util.spec_from_file_location(
        "tlx_T6_baseline_" + str(a.stage), a.baseline
    )
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    if a.baseline_core:
        core_spec = importlib.util.spec_from_file_location(
            "tlx_reference_core", a.baseline_core
        )
        core = importlib.util.module_from_spec(core_spec)
        sys.modules[core_spec.name] = core
        core_spec.loader.exec_module(core)
        baseline.expert_tile = core.expert_tile
    if a.stage == 1:
        old_head = baseline.qkvg

        def compatible_head(*aa, **kw):
            if "head_rows" not in inspect.signature(old_head).parameters:
                kw.pop("head_rows", None)
            return old_head(*aa, **kw)

        baseline.qkvg = compatible_head
    if a.stage == 3:
        old_call = baseline.attention_tail

        def compatible(*aa, **kw):
            for key in ("store_reduced", "split_norm"):
                if key not in inspect.signature(old_call).parameters:
                    kw.pop(key, None)
            return old_call(*aa, **kw)

        baseline.attention_tail = compatible
    if a.stage == 6:
        old_tail = baseline.unpool_boundary

        def compatible_tail(*aa, **kw):
            for key in ("packed_handoff", "projection_depth"):
                if key not in inspect.signature(old_tail).parameters:
                    kw.pop(key, None)
            return old_tail(*aa, **kw)

        baseline.unpool_boundary = compatible_tail
    legacy_s5 = (
        a.stage == 5 and "fuse_norm_sum" not in baseline.Workspace.__struct_fields__
    )
    if legacy_s5:

        class CompatibleWorkspace(baseline.Workspace):
            fuse_norm_sum: bool = False

        baseline.Workspace = CompatibleWorkspace
    mods = {"baseline": baseline, "candidate": candidate}
    groups = {n: [] for n in mods}
    spaces = {n: [] for n in mods}
    retained = []
    for _ in range(3):
        base, native, oldws, errors, routes = setup(a.context, a.left)
        retained.append((base, native, oldws, errors, routes))
        x, stream, w, m, _ = base["chained"][1].args
        old = gpu(
            torch.load(
                Path("/home/bangtliu/coord/phase44-baseline")
                / f"layer-7-{a.context}-{a.left}.pt",
                weights_only=True,
            )["rows"][0]
        )
        allocation_order = tuple(mods)[::-1] if a.reverse_allocation else tuple(mods)
        for n in allocation_order:
            mod = mods[n]
            if a.shared_workspace and n != allocation_order[0]:
                ws = clone_workspace_headers(spaces[allocation_order[0]][-1])
                assert (
                    ws.s3.state.epochs.data_ptr()
                    == spaces[allocation_order[0]][-1].s3.state.epochs.data_ptr()
                )
            else:
                with patch.object(tlx_chain, attr, mod):
                    ws = tlx_chain.make_workspace(
                        "cuda", context=a.context, left=a.left
                    )
            if hasattr(mod, "bind_workspace"):
                mod.bind_workspace(ws)
            if legacy_s5 and n == "baseline":
                ws.s5.fuse_norm_sum = False
            if a.fuse_norm_sum and n == "candidate":
                assert a.stage == 5
                ws.s5.fuse_norm_sum = True
            spaces[n].append(ws)

            def call(mod=mod, ws=ws, x=x, stream=stream, w=w, m=m):
                with patch.object(tlx_chain, attr, mod):
                    tlx_chain.launch(x, stream, w, m, ws, rank=0)

            groups[n].append(
                (
                    functools.partial(
                        tlx_chain.prepare_peers, ws, *native.partials, rank=0
                    ),
                    call,
                    functools.partial(
                        archive_check, native.f, ws, old, 0, digest=False
                    ),
                )
            )
    allocations = {}
    for name in (
        "qkvg",
        "o_proj",
        "joint_router_pooler",
        "w13",
        "w13_scale",
        "w2",
        "w2_scale",
        "unpooler",
    ):
        ptrs = [r[0]["chained"][1].args[2][name].data_ptr() for r in retained]
        assert len(set(ptrs)) == 3, (name, ptrs)
        allocations[name] = ptrs
    cases = {}
    for n, rows in groups.items():

        def action(i, rows=rows):
            for row in rows:
                row[i]()

        cases[n] = tuple(functools.partial(action, i) for i in range(3))
    events = {
        n: [
            (
                torch.cuda.Event(enable_timing=True, external=True),
                torch.cuda.Event(enable_timing=True, external=True),
            )
            for _ in range(3)
        ]
        for n in mods
    }
    counters = dict.fromkeys(mods, 0)
    active = [None]
    seen = set()

    def marked(fn, n):
        def invoke(*aa, **kw):
            i = counters[n] % 3
            counters[n] += 1
            s, e = events[n][i]
            if not a.whole_only:
                s.record()
            active[0] = n
            try:
                return fn(*aa, **kw)
            finally:
                active[0] = None
                if not a.whole_only:
                    e.record()

        return invoke

    original = CompiledKernel.launch_metadata

    def metadata(kernel, *aa, **kw):
        result = original(kernel, *aa, **kw)
        if active[0] is not None:
            fn = kernel.src.fn.fn
            phase = fn.__name__
            if phase == "expert_phase":
                index = kernel.src.fn.arg_names.index("FIRST")
                first = kernel.src.constants.get(
                    (index,), kernel.src.constants.get("FIRST")
                )
                phase = "w13" if first else "w2"
            key = (active[0], phase)
            if key not in seen:
                dump(kernel, a.label + "/" + phase, active[0])
                seen.add(key)
        return result

    def observer(n):
        return dict(stage_us=sum(s.elapsed_time(e) * 1000 for s, e in events[n]) / 3)

    with ExitStack() as stack:
        stack.enter_context(patch.object(CompiledKernel, "launch_metadata", metadata))
        allocation_order = tuple(mods)[::-1] if a.reverse_allocation else tuple(mods)
        for n in allocation_order:
            mod = mods[n]
            stack.enter_context(patch.object(mod, api, marked(getattr(mod, api), n)))
        result = balanced(
            cases,
            {n: v[0].s3.state.epochs for n, v in spaces.items()},
            None if a.whole_only else observer,
        )
    if not a.whole_only:
        result["stages"] = {
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
            for n in mods
        }
        result["stage_ratios"] = {
            o: result["stages"]["candidate"][o]["median_us"]
            / result["stages"]["baseline"][o]["median_us"]
            for o in ("AB", "BA")
        }
    result.update(
        context=a.context,
        left=a.left,
        stage=a.stage,
        distinct_weight_pointers=allocations,
        baseline=a.baseline,
        reverse_allocation=a.reverse_allocation,
        shared_workspace=a.shared_workspace,
    )
    Path("/home/bangtliu/coord/logs", a.label + ".json").write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )
    print(json.dumps(result.get("stage_ratios", result["timing"])), flush=True)


if __name__ == "__main__":
    main()
