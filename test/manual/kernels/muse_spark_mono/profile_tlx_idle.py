"""TLX per-kernel timing proxies and grid/resources; no hardware idle counter."""

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
from reference_archive import gpu
from tlx_ir import dump
from triton.compiler import CompiledKernel

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_chain as chain,
)

LOG = Path("/home/bangtliu/coord/logs")


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, required=True)
    p.add_argument("--left", type=int, required=True)
    p.add_argument("--label", default="phaseT9")
    a = p.parse_args()
    modules = [
        (chain.tlx_qkvg, "qkvg_heads", "S1.heads"),
        (chain.tlx_attention, "partials", "S2.partial"),
        (chain.tlx_attention, "merge", "S2.merge"),
        (chain.tlx_attn_tail, "normalize", "S3.norm"),
        (chain.tlx_attn_tail, "normalize_rows", "S3.norm"),
        (chain.tlx_attn_tail, "s3_boundary", "S3.boundary"),
        (chain.tlx_route, "finish_rows", "S4.finish"),
        (chain.tlx_route, "slots", "S4.slots"),
        (chain.tlx_experts, "norm_phase", "S5.norm"),
        (chain.tlx_experts, "sum_reduce", "S5.ar"),
        (chain.tlx_tail, "finish", "S6.boundary"),
    ]
    proj = chain.tlx_qkvg.tlx_projection
    phases = [ph for _, _, ph in modules] + [
        "S1.project",
        "S3.project",
        "S4.project",
        "S5.w13",
        "S5.w2",
        "S6.project",
    ]
    events = {
        ph: [
            (
                torch.cuda.Event(enable_timing=True, external=True),
                torch.cuda.Event(enable_timing=True, external=True),
            )
            for _ in range(3)
        ]
        for ph in phases
    }
    counts = dict.fromkeys(phases, 0)
    active = [None]
    metadata = {}
    enabled = [False]
    retained = []
    cases = {}
    spaces = {}
    for label in ("plain", "marked"):
        rows = []
        spaces[label] = []
        for i in range(3):
            base, native, _, _, _ = setup(a.context, a.left)
            retained.append((base, native))
            x, stream, w, m, _ = base["chained"][1].args
            ws = chain.make_workspace("cuda", context=a.context, left=a.left)
            spaces[label].append(ws)
            old = gpu(
                torch.load(
                    Path("/home/bangtliu/coord/phase44-baseline")
                    / f"layer-7-{a.context}-{a.left}.pt",
                    weights_only=True,
                )["rows"][0]
            )
            rows.append(
                (
                    functools.partial(
                        chain.prepare_peers, ws, *native.partials, rank=0
                    ),
                    functools.partial(chain.launch, x, stream, w, m, ws, rank=0),
                    functools.partial(
                        archive_check, native.f, ws, old, 0, digest=False
                    ),
                )
            )

        def action(index, rows=rows, label=label):
            enabled[0] = label == "marked"
            for r in rows:
                r[index]()

        cases[label] = tuple(functools.partial(action, i) for i in range(3))

    class Proxy:
        def __init__(self, fn, phase):
            self.fn, self.phase = fn, phase

        def __getattr__(self, k):
            return getattr(self.fn, k)

        def __getitem__(self, grid):
            fn = self.fn[grid]

            def call(*aa, **kw):
                if not enabled[0]:
                    return fn(*aa, **kw)
                ph = self.phase(kw) if callable(self.phase) else self.phase
                i = counts[ph] % 3
                counts[ph] += 1
                s, e = events[ph][i]
                s.record()
                active[0] = (ph, grid)
                try:
                    return fn(*aa, **kw)
                finally:
                    active[0] = None
                    e.record()

            return call

    original = CompiledKernel.launch_metadata

    def meta(kernel, *aa, **kw):
        r = original(kernel, *aa, **kw)
        if active[0] and active[0][0] not in metadata:
            ph, g = active[0]
            metadata[ph] = dict(
                grid=list(g),
                resources=dump(
                    kernel, f"{a.label}-idle-{a.context}-{a.left}/" + ph, "tlx"
                ),
            )
        return r

    def observe(label):
        if label == "plain":
            return {}
        return dict(
            kernel_us={
                ph: sum(s.elapsed_time(e) * 1000 for s, e in events[ph]) / 3
                for ph in phases
                if ph in metadata
            }
        )

    with ExitStack() as stack:
        stack.enter_context(patch.object(CompiledKernel, "launch_metadata", meta))
        for mod, name, ph in modules:
            stack.enter_context(patch.object(mod, name, Proxy(getattr(mod, name), ph)))
        stack.enter_context(
            patch.object(
                proj,
                "project",
                Proxy(proj.project, lambda kw: "S" + str(kw["STAGE"]) + ".project"),
            )
        )
        stack.enter_context(
            patch.object(
                chain.tlx_experts,
                "expert_phase",
                Proxy(
                    chain.tlx_experts.expert_phase,
                    lambda kw: "S5.w13" if kw["FIRST"] else "S5.w2",
                ),
            )
        )
        result = balanced(
            cases, {n: w[0].s3.state.epochs for n, w in spaces.items()}, observe
        )
    (LOG / f"{a.label}-idle-{a.context}-{a.left}.json").write_text(
        json.dumps(
            dict(
                context=a.context,
                left=a.left,
                result=result,
                kernels=metadata,
                limit="Static resource/grid plus external event windows, not physical CU occupancy/activity counters. Marked/plain bias retained. Low-grid intervals are upper-bound proxies containing event/dispatch overhead.",
            ),
            indent=2,
            default=str,
        )
        + "\n"
    )
    print("PASS idle map", a.context, a.left, flush=True)


if __name__ == "__main__":
    main()
