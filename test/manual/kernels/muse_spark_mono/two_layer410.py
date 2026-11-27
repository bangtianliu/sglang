"""Two distinct-weight, data-dependent layers; balanced graph A/B, GPU0 only.

The second layer uses fixed virtual-peer payloads in both controls. This is a
cache-policy experiment, not a distributed two-layer model accuracy claim.
"""

import argparse
import functools
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from check_chained47 import BASE, archive_check
from ci410 import block_verdict
from diagnose_guard48 import balanced
from layer_case import make
from layer_fixture import require_gpu0
from reference_archive import gpu

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    chained,
    independent_experts,
    independent_qkvg,
    independent_tail,
    split_attention,
    warm_boundary,
)

LOG = Path("/home/bangtliu/coord/logs")


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--left", type=int, default=-1)
    p.add_argument(
        "--candidate",
        choices=(
            "s5-nt",
            "s2-split",
            "s1",
            "s3-wide",
            "s5-k9",
            "warm",
            "warm-nt",
            "combined",
            "s5-prefetch",
        ),
        required=True,
    )
    p.add_argument("--attention-backend", choices=("common", "split"), default="split")
    p.add_argument(
        "--wide-s1-128",
        action="store_true",
        help="Reproduce the rejected S1+S3 sliding128k stack",
    )
    a = p.parse_args()
    inputs = [make(seed, a.context, a.left, 0) for seed in (7, 20261117)]
    old = [
        gpu(
            torch.load(
                BASE / f"layer-{seed}-{a.context}-{a.left}.pt", weights_only=True
            )["rows"][0]
        )
        for seed in (7, 20261117)
    ]
    for key in ("qkvg", "o_proj", "joint_router_pooler", "unpooler", "w13", "w2"):
        assert inputs[0][1][key].data_ptr() != inputs[1][1][key].data_ptr()
        assert not torch.equal(inputs[0][1][key], inputs[1][1][key]), key
    original_s1 = chained.s1.qkvg
    original_s2 = chained.s2.attention
    original_s5 = independent_experts.s5
    original_common_s5 = chained.s5.s5
    warm_state = {}

    class WarmFinish:
        def __init__(self):
            self.epochs = torch.zeros((72, 32), device="cuda", dtype=torch.int32)
            self.counter = torch.zeros((2,), device="cuda", dtype=torch.int32)
            self.flags = torch.zeros((64, 32), device="cuda", dtype=torch.int32)
            self.status = torch.zeros((64,), device="cuda", dtype=torch.int32)
            self.compiled = None

        def args(self, aa):
            return (
                *aa,
                inputs[1][1]["qkvg"],
                self.epochs,
                self.counter,
                self.flags,
                self.status,
            )

        def warmup(self, *aa, **kw):
            kw["grid"] = (72,)
            self.compiled = warm_boundary.finish.warmup(*self.args(aa), **kw)
            return self.compiled

        def __getitem__(self, grid):
            assert grid == (8,)
            return lambda *aa, **kw: warm_boundary.finish[(72,)](*self.args(aa), **kw)

    spaces = {}
    cases = {}
    epochs = {}
    expected = {}
    for label in ("baseline", "candidate"):
        pair = []
        for i in range(2):
            ws = chained.make_workspace(
                "cuda",
                context=a.context,
                left=a.left,
                attention_backend="common"
                if a.candidate == "s2-split"
                else a.attention_backend,
                projection_backend="common",
                stage_policy=(
                    "wide"
                    if (a.context, a.left) in ((8192, 2047), (131072, -1))
                    else "common",
                    "k9"
                    if (a.context, a.left) in ((8192, -1), (131072, -1), (131072, 2047))
                    else "common",
                    "wide",
                ),
            )
            if label == "candidate":
                if a.candidate == "s2-split":
                    ws.s2 = split_attention.make_workspace(
                        "cuda", context=a.context, window_left=a.left
                    )
                if a.candidate == "s1":
                    ws.s1 = independent_qkvg.make_workspace("cuda")
                if a.candidate in ("s5-nt", "s5-k9"):
                    ws.s5 = independent_experts.make_workspace("cuda")
                    ws.stage_policy = (ws.stage_policy[0], "k9", ws.stage_policy[2])
                if a.candidate == "s3-wide":
                    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
                        independent_attn_tail,
                    )

                    ws.s3 = independent_attn_tail.make_workspace("cuda")
                    ws.stage_policy = ("wide", *ws.stage_policy[1:])
                if a.candidate == "s5-prefetch":
                    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.common_shape import (
                        CommonShape,
                    )

                    ws.s5 = chained.s5.make_workspace(
                        "cuda", shape=CommonShape(vgpr_cap=320)
                    )
                    ws.stage_policy = (ws.stage_policy[0], "common", ws.stage_policy[2])
                if a.candidate == "combined":
                    from candidate_policy410 import configure

                    configure(ws, "cuda", a.context, a.left, wide_s1_128=a.wide_s1_128)
            pair.append(ws)
        spaces[label] = pair
        epochs[label] = pair[0].s3.state.epochs
        if label == "candidate" and a.candidate in ("warm", "warm-nt"):
            warm_state[label] = WarmFinish()

        def prepare(pair=pair):
            for i, ws in enumerate(pair):
                chained.prepare_peers(
                    ws, old[i]["partial3"], old[i]["partial5"], rank=0
                )

        def call(pair=pair, label=label):
            with ExitStack() as stack:
                if label == "candidate":
                    if a.candidate == "s5-prefetch":
                        stack.enter_context(
                            patch.object(
                                chained.s5,
                                "s5",
                                functools.partial(
                                    original_common_s5, prefetch_before_poll=True
                                ),
                            )
                        )
                    if a.candidate == "s1":
                        stack.enter_context(
                            patch.object(chained.s1, "qkvg", independent_qkvg.qkvg)
                        )
                    if a.candidate == "s2-split":
                        stack.enter_context(
                            patch.object(
                                chained.s2, "attention", split_attention.attention
                            )
                        )
                    if a.candidate in ("s5-nt", "warm-nt"):
                        stack.enter_context(
                            patch.object(
                                independent_experts,
                                "s5",
                                functools.partial(original_s5, weight_cache=".cs"),
                            )
                        )
                x, stream = inputs[0][0].x, inputs[0][0].stream
                for i, ws in enumerate(pair):
                    if (
                        label == "candidate"
                        and a.candidate in ("warm", "warm-nt")
                        and i == 0
                    ):
                        with patch.object(
                            independent_tail, "finish", warm_state[label]
                        ):
                            x, stream = chained.launch(
                                x, stream, inputs[i][1], inputs[i][2], ws, rank=0
                            )
                    else:
                        x, stream = chained.launch(
                            x, stream, inputs[i][1], inputs[i][2], ws, rank=0
                        )
                return x, stream

        def check(pair=pair, label=label):
            archive_check(inputs[0][0], pair[0], old[0], 0, digest=True)
            if label in warm_state:
                assert bool(warm_state[label].status.eq(1).all()), warm_state[
                    label
                ].status.tolist()
            for i, ws in enumerate(pair):
                assert int(ws.failure[0]) == 0, (label, i, ws.failure.tolist())
                assert all(bool(torch.all(s == 1)) for s in chained.statuses(ws)), (
                    label,
                    i,
                )
                for key, value in ws.buffers.items():
                    assert bool(torch.isfinite(value).all()), (label, i, key)
                    bits = value.view(torch.uint8)
                    if (i, key) not in expected:
                        expected[i, key] = bits.clone()
                    assert torch.equal(bits, expected[i, key]), dict(
                        tensor=key,
                        layer=i,
                        rank=0,
                        location=(bits != expected[i, key]).nonzero()[:1].tolist(),
                    )

        cases[label] = (prepare, call, check)
    d = balanced(cases, epochs)
    d.update(
        candidate=a.candidate,
        context=a.context,
        left=a.left,
        layers=2,
        seeds=[7, 20261117],
        distinct_weights=True,
        second_layer_input="first layer out/stream; fixed identical virtual peers in both controls",
        verdict=block_verdict(d),
        resources={
            n: [chained.resources(ws) for ws in pair] for n, pair in spaces.items()
        },
    )
    if a.candidate in ("warm", "warm-nt"):
        k = warm_state["candidate"].compiled
        (LOG / "phase410-warm.amdgcn").write_text(k.asm["amdgcn"])
        d["warm"] = dict(
            vgprs=k.n_regs,
            spills=k.n_spills,
            lds=k.metadata.shared,
            ctas=72,
            warm_ctas=64,
            bytes=4608 * 8 * 32 * 4,
            policy="default-cache; flags released after all 8 rows consumed critical inputs",
        )
    if a.candidate == "s5-nt":
        asm = {}
        for li, ws in enumerate(spaces["candidate"]):
            for key, ks in ws.s5.kernels.items():
                for name, k in zip(("w13", "w2", "norm", "sum"), ks):
                    if name in ("w13", "w2"):
                        path = LOG / f"phase410-nt-L{li}-{name}.amdgcn"
                        path.write_text(k.asm["amdgcn"])
                        lines = [
                            x
                            for x in k.asm["amdgcn"].splitlines()
                            if "global_load" in x
                        ]
                        nt = [
                            x
                            for x in lines
                            if "TH_LOAD_NT" in x or "TH_LOAD_RT_NT" in x
                        ]
                        assert nt, (name, "missing non-temporal lowering")
                        asm[f"L{li}-{name}"] = dict(
                            path=str(path), loads=len(lines), nt=len(nt), lines=lines
                        )
        d["asm"] = asm
    path = LOG / f"phase410-two-{a.candidate}-{a.context}-{a.left}.json"
    path.write_text(json.dumps(d, indent=2) + "\n")
    print(json.dumps(dict(timing=d["timing"], verdict=d["verdict"])), flush=True)


if __name__ == "__main__":
    main()
