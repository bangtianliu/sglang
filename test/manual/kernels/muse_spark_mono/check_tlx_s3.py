"""S3 exact old-stage parity, then unchanged native/FP64/replay/missing suite."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from layer_fixture import require_gpu0
from layer_reference import exact_check
from reference_archive import checked_load

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import standin
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    independent_attn_tail as old,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_attn_tail as new,
)

LOG = Path("/home/bangtliu/coord/logs/phaseT1-S3-suite")


def main():
    global LOG, LABEL
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--exact-only", action="store_true")
    p.add_argument("--suite-only", action="store_true")
    p.add_argument("--label", default="T1")
    p.add_argument("--no-debug-store", action="store_true")
    a = p.parse_args()
    assert not a.no_debug_store or a.exact_only
    LABEL = a.label
    LOG = Path("/home/bangtliu/coord/logs") / (LABEL + "-S3-suite")
    LOG.mkdir(exist_ok=True)
    d = checked_load(f"S3-inputs-{a.seed}.pt")
    native = checked_load(f"S3-native-{a.seed}.pt")
    records = []
    for rank in () if a.suite_only else range(4):
        for sign in (1, -1):
            spaces = []
            values = []
            for module in (old, new):
                ws = module.make_workspace("cuda")
                spaces.append(ws)
                standin.prepare(
                    ws.peers, native[sign]["partial"], ws.state.epochs, rank=rank
                )
                if module is new and a.no_debug_store:
                    ws.peers.reduced.fill_(float("nan"))
                values.append(
                    module.attention_tail(
                        d["x"][rank] * sign,
                        d["gate"][rank],
                        d["weight"][rank],
                        d["stream"],
                        d["pre"],
                        d["alpha"],
                        d["beta"],
                        rank=rank,
                        workspace=ws,
                        **(
                            {"store_reduced": False}
                            if module is new and a.no_debug_store
                            else {}
                        ),
                    )
                )
            torch.cuda.synchronize()
            for name, x, y in [
                ("projection", spaces[0].projection, spaces[1].projection),
                ("partial", spaces[0].partial[rank], spaces[1].partial[rank]),
                ("out", values[0][0], values[1][0]),
                ("stream", values[0][1], values[1][1]),
            ]:
                dtype = torch.int16 if x.dtype == torch.bfloat16 else torch.int32
                exact_check(
                    x.view(dtype),
                    y.view(dtype),
                    f"S3 sign{sign} {name} bits",
                    {"_rank": rank},
                )
            if a.no_debug_store:
                untouched = torch.isnan(spaces[1].peers.reduced)
                if not untouched.all():
                    location = torch.nonzero(~untouched)[0].tolist()
                    raise AssertionError(
                        f"S3 Reduced rank{rank} location{location}: disabled output written"
                    )
            records.append(
                dict(
                    rank=rank, sign=sign, exact=True, store_reduced=not a.no_debug_store
                )
            )
    (LOG / f"exact-{a.seed}.json").write_text(json.dumps(records, indent=2) + "\n")
    if a.exact_only:
        return
    import check_common_s3 as suite

    with (
        patch.object(suite.stage, "make_workspace", new.make_workspace),
        patch.object(suite.stage, "attention_tail", new.attention_tail),
        patch.object(suite, "ROOT", LOG),
        patch.object(
            sys,
            "argv",
            ["check_common_s3.py", "--seed", str(a.seed), "--replays", "200"],
        ),
    ):

        def untimed(prepare, launch):
            # The legacy checker expects timing to leave the positive case live.
            prepare()
            launch()
            torch.cuda.synchronize()
            return {"skipped": "Use cold chain A/B"}

        suite.rank_times = untimed
        suite.main()


if __name__ == "__main__":
    from tlx_ir import dump
    from triton.compiler import CompiledKernel

    seen = set()
    original_metadata = CompiledKernel.launch_metadata

    def metadata(kernel, *args, **kwargs):
        result = original_metadata(kernel, *args, **kwargs)
        module = kernel.src.fn.fn.__module__.rsplit(".", 1)[-1]
        function = kernel.src.fn.fn.__name__
        if "--suite-only" not in sys.argv and module in (
            "independent_attn_tail",
            "wide_projection",
            "tlx_attn_tail",
            "tlx_projection",
        ):
            side = "tlx" if module.startswith("tlx_") else "gluon"
            phase = {
                "normalize": "norm_gate",
                "project": "projection",
                "s3_boundary": "AR_boundary",
            }.get(function)
            key = (phase, side)
            if phase and key not in seen:
                dump(kernel, LABEL + "-S3/" + phase, side)
                seen.add(key)
        return result

    with patch.object(CompiledKernel, "launch_metadata", metadata):
        main()
