"""T1 S6 exact Gluon parity plus native/FP64/replay/missing suite."""

import argparse
import functools
import json
from pathlib import Path
from unittest.mock import patch

import torch
from layer_fixture import require_gpu0
from layer_reference import exact_check
from reference_archive import checked_load
from tlx_ir import dump

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_tail,
    independent_tail,
    tlx_tail,
    wide_projection,
)

LOG = Path("/home/bangtliu/coord/logs")


def main():
    global LOG
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--exact-only", action="store_true")
    p.add_argument("--suite-only", action="store_true")
    p.add_argument("--label", default="T1-S6")
    p.add_argument("--packed-handoff", action="store_true")
    a = p.parse_args()
    LOG = LOG / a.label
    LOG.mkdir(exist_ok=True)
    if a.packed_handoff:
        tlx_tail.unpool_boundary = functools.partial(
            tlx_tail.unpool_boundary, packed_handoff=True
        )
    d = checked_load(f"S6-inputs-{a.seed}.pt")
    records = []
    for kind in () if a.suite_only else ("normal", "last_weighted", "last_unweighted"):
        pre = None if kind == "last_unweighted" else d["pre"]
        for sign in (1, -1):
            spaces = []
            outputs = []
            for mod in (independent_tail, tlx_tail):
                ws = mod.make_workspace("cuda")
                spaces.append(ws)
                outputs.append(
                    mod.unpool_boundary(
                        d["x"] * sign,
                        d["weight"],
                        d["stream"],
                        d["post"],
                        pre,
                        d["alpha"],
                        d["beta"],
                        pre_bf16=kind != "normal",
                        workspace=ws,
                    )
                )
            torch.cuda.synchronize()
            for name, x, y in [
                ("projection", spaces[0].projection, spaces[1].projection),
                ("out", outputs[0][0], outputs[1][0]),
                ("stream", outputs[0][1], outputs[1][1]),
            ]:
                typ = torch.int16 if x.dtype == torch.bfloat16 else torch.int32
                exact_check(
                    x.view(typ),
                    y.view(typ),
                    f"S6 {kind} sign{sign} {name}",
                    {"_rank": 0},
                )
            if a.packed_handoff:
                exact_check(
                    spaces[1].packed.view(torch.int16),
                    spaces[0]
                    .projection.view(torch.float32)
                    .bfloat16()
                    .view(torch.int16),
                    "S6 BF16 handoff bits",
                    {"_rank": 0},
                )
            records.append(
                dict(
                    kind=kind,
                    sign=sign,
                    exact=True,
                    resources=list(spaces[1].state.admitted.values()),
                )
            )
            if kind == "normal" and sign == 1 and a.seed == 7:
                gs, ts = spaces
                k = wide_projection.project.warmup(
                    d["x"],
                    d["weight"],
                    gs.projection,
                    gs.state.status,
                    None,
                    N=8192,
                    K=4096,
                    STAGE=6,
                    grid=(512,),
                    num_warps=2,
                    enable_fp_fusion=False,
                )
                dump(k, f"{a.label}/projection", "gluon")
                fa = (
                    gs.projection,
                    d["stream"],
                    d["post"],
                    pre,
                    d["alpha"],
                    d["beta"],
                    *outputs[0],
                    gs.state.status,
                    None,
                )
                k = independent_tail.finish.warmup(
                    *fa, PRE_BF16=False, grid=(8,), num_warps=4, enable_fp_fusion=False
                )
                dump(k, f"{a.label}/boundary", "gluon")
                ks = next(iter(ts.kernels.values()))
                for phase, k in zip(("projection", "boundary"), ks):
                    dump(k, f"{a.label}/" + phase, "tlx")
            print(json.dumps(records[-1]), flush=True)
    if not a.suite_only:
        (LOG / f"phaseT1-S6-exact-{a.seed}.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
    if a.exact_only:
        return
    import check_common_s6 as suite

    original = suite.exercise
    original_rounding = suite.check_boundary_rounding

    def rounding(reference=None, *, waves=4):
        def callback(branch, stream, pre, alpha, beta, post=None, last=False):
            out = torch.empty_like(branch)
            updated = torch.empty_like(stream)
            status = torch.ones((512,), device="cuda", dtype=torch.int32)
            partial = branch if a.packed_handoff else branch.float().view(torch.int32)
            tlx_tail.finish[(8,)](
                partial,
                stream,
                post,
                pre,
                alpha,
                beta,
                out,
                updated,
                status,
                None,
                PRE_BF16=last,
                PACKED_INPUT=a.packed_handoff,
                num_warps=4,
                enable_fp_fusion=False,
            )
            torch.cuda.synchronize()
            if reference is not None:
                expected = reference(
                    branch, stream, pre, alpha, beta, post=post, last=last
                )
                for name, x, y in [
                    ("out", out, expected[0]),
                    ("stream", updated, expected[1]),
                ]:
                    typ = torch.int16 if x.dtype == torch.bfloat16 else torch.int32
                    exact_check(
                        x.view(typ),
                        y.view(typ),
                        "S6 rounding pattern " + name,
                        {"_rank": 0},
                    )
            return out, updated

        original_rounding(callback, waves=waves)

    def exercise(*args):
        return original(*args[:-1], a.label + "-" + args[-1])

    with (
        patch.object(common_tail, "make_workspace", tlx_tail.make_workspace),
        patch.object(common_tail, "unpool_boundary", tlx_tail.unpool_boundary),
        patch.object(suite, "exercise", exercise),
        patch.object(suite, "check_boundary_rounding", rounding),
        patch.object(
            suite,
            "graph_times",
            lambda call: {"skipped": "Use cold chain-context A/B for adoption"},
        ),
    ):
        rows = suite.s6(a.seed)
    (LOG / f"phaseT1-S6-suite-{a.seed}.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    print("PASS TLX S6 exact/native/FP64/replay/missing", flush=True)


if __name__ == "__main__":
    main()
