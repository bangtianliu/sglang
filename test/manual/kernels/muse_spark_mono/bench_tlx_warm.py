"""Cold graph A/B for idle-CU warming and a dependent distinct-weight two-layer pair."""

import argparse
import functools
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from bench_chained47 import setup
from check_chained47 import archive_check
from diagnose_guard48 import balanced
from layer_fixture import require_gpu0
from reference_archive import gpu
from tlx_inline_warm import InlineWarm
from tlx_ir import dump
from tlx_row_candidate import RowPair

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_warm
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_chain as chain,
)

LOG = Path("/home/bangtliu/coord/logs")


class Warm:
    def __init__(self, weight, label):
        self.weight = weight
        self.stream = torch.cuda.Stream(device=weight.device)
        self.checks = torch.empty((384,), device=weight.device, dtype=torch.uint32)
        self.kw = dict(N=weight.numel(), GRID=384, HEAD_SLEEP=64, num_warps=4)
        kernel = tlx_warm.readonly.warmup(weight, self.checks, **self.kw, grid=(384,))
        dump(kernel, label + "/warm", "tlx")
        self.expected = weight.view(torch.uint16).to(torch.int64).sum().item() % (
            1 << 32
        )

    def verify(self):
        assert int(self.checks.to(torch.int64).sum()) % (1 << 32) == self.expected, (
            "warm read checksum"
        )

    @contextmanager
    def install(self, module, name):
        raw = getattr(module, name)
        owner = self

        class Proxy:
            def __getattr__(self, k):
                return getattr(raw, k)

            def __getitem__(self, grid):
                fn = raw[grid]

                def launch(*aa, **kw):
                    origin = torch.cuda.current_stream()
                    owner.stream.wait_stream(origin)
                    result = fn(*aa, **kw)
                    with torch.cuda.stream(owner.stream):
                        tlx_warm.readonly[(384,)](
                            owner.weight, owner.checks, **owner.kw
                        )
                    origin.wait_stream(owner.stream)
                    return result

                return launch

        with patch.object(module, name, Proxy()):
            yield


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--run-label", default="")
    p.add_argument("--context", type=int, required=True)
    p.add_argument("--left", type=int, required=True)
    p.add_argument(
        "--mode",
        choices=(
            "idle",
            "cross",
            "row-serial",
            "row-overlap",
            "inline-idle",
            "inline-cross",
        ),
        required=True,
    )
    a = p.parse_args()
    cases = {}
    spaces = {}
    retained = []
    warmers = []
    expected = {}
    allocations = []
    for label in ("baseline", "candidate"):
        rows = []
        spaces[label] = []
        for i in range(3):
            entries = []
            for layer in range(2 if a.mode in ("cross", "inline-cross") else 1):
                base, native, _, _, _ = setup(a.context, a.left)
                retained.append((base, native))
                x, stream, w, m, _ = base["chained"][1].args
                ws = chain.make_workspace("cuda", context=a.context, left=a.left)
                old = gpu(
                    torch.load(
                        Path("/home/bangtliu/coord/phase44-baseline")
                        / f"layer-7-{a.context}-{a.left}.pt",
                        weights_only=True,
                    )["rows"][0]
                )
                if layer == 1:
                    # Distinct layer values, preserving router high/low packing.
                    for name in ("qkvg", "o_proj", "joint_router_pooler", "unpooler"):
                        w[name] = -w[name]
                    for name in ("w13", "w13_scale", "w2", "w2_scale"):
                        w[name] = w[name].roll(1, 0).contiguous()
                entries.append((native, x, stream, w, m, ws, old))
                spaces[label].append(ws)
            weight = (
                entries[-1][3]["qkvg"]
                if a.mode in ("cross", "inline-cross")
                else entries[0][3]["joint_router_pooler"]
            )
            if a.mode.startswith("inline-"):
                warmer = (
                    InlineWarm(
                        weight,
                        f"phaseT9-{a.mode}-{a.context}-{a.left}"
                        + ("-" + a.run_label if a.run_label else ""),
                    )
                    if label == "candidate"
                    else None
                )
            elif a.mode.startswith("row-"):
                warmer = (
                    RowPair(
                        f"phaseT9-{a.mode}-{a.context}-{a.left}"
                        + ("-" + a.run_label if a.run_label else ""),
                        a.mode[4:],
                    )
                    if label == "candidate"
                    else None
                )
            else:
                warmer = (
                    Warm(weight, f"T9-{a.mode}-{a.context}-{a.left}")
                    if label == "candidate"
                    else None
                )
            if warmer:
                warmers.append(warmer)
            allocations.append(
                dict(
                    label=label,
                    copy=i,
                    pointers=[e[3]["qkvg"].data_ptr() for e in entries],
                )
            )

            def prepare(entries=entries):
                for native, x, stream, w, m, ws, old in entries:
                    chain.prepare_peers(ws, *native.partials, rank=0)

            def call(entries=entries, warmer=warmer):
                n, x, stream, w, m, ws, old = entries[0]
                module, name = (
                    (chain.tlx_attn_tail, "s3_boundary")
                    if a.mode in ("idle", "inline-idle")
                    else (chain.tlx_tail, "finish")
                )
                if warmer:
                    with warmer.install(module, name):
                        out, updated = chain.launch(x, stream, w, m, ws, rank=0)
                else:
                    out, updated = chain.launch(x, stream, w, m, ws, rank=0)
                if a.mode in ("cross", "inline-cross"):
                    n, x, stream, w, m, ws, old = entries[1]
                    chain.launch(out, updated, w, m, ws, rank=0)

            def check(entries=entries, label=label, i=i, warmer=warmer):
                for layer, (n, x, stream, w, m, ws, old) in enumerate(entries):
                    if layer == 0:
                        archive_check(n.f, ws, old, 0, digest=False)
                    for name in ("out", "stream", "q", "gate", "x4", "x6", "attn"):
                        tensor = ws.buffers[name]
                        key = (i, layer, name)
                        if label == "baseline":
                            expected[key] = tensor.clone()
                        else:
                            assert torch.equal(
                                tensor.view(torch.uint8),
                                expected[key].view(torch.uint8),
                            ), ("T9 cross-layer", i, layer, name)
                if warmer:
                    warmer.verify()

            rows.append((prepare, call, check))

        def action(index, rows=rows):
            for row in rows:
                row[index]()

        cases[label] = tuple(functools.partial(action, j) for j in range(3))
    result = balanced(
        cases, {label: ws[0].s3.state.epochs for label, ws in spaces.items()}
    )
    result["ratios"] = {
        o: result["timing"]["candidate"]["by_order"][o]["median_us"]
        / result["timing"]["baseline"]["by_order"][o]["median_us"]
        for o in ("AB", "BA")
    }
    result.update(
        context=a.context,
        left=a.left,
        mode=a.mode,
        copies=3,
        layers=2 if a.mode in ("cross", "inline-cross") else 1,
        allocations=allocations,
        note="Readonly warming overlaps row boundary; no activation handoff becomes a flag; all existing cache/scope paths retained. HEAD_SLEEP64. Per-case output and read checksum exact.",
    )
    (
        LOG
        / (
            f"phaseT9-warm-{a.mode}-{a.context}-{a.left}"
            + ("-" + a.run_label if a.run_label else "")
            + ".json"
        )
    ).write_text(json.dumps(result, indent=2) + "\n")
    print("PASS", a.mode, result["ratios"], flush=True)


if __name__ == "__main__":
    main()
