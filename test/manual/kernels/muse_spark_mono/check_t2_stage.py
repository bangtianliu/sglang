"""Run existing S6/S3 correctness packages with isolated T2 logs and IR."""

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import tlx_ir
from triton.compiler import CompiledKernel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", type=int, choices=(3, 6))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--label", default="")
    p.add_argument("--exact-only", action="store_true")
    p.add_argument("--suite-only", action="store_true")
    a = p.parse_args()
    suffix = "-" + a.label if a.label else ""
    root = Path("/home/bangtliu/coord/logs") / f"phaseT2-S{a.stage}{suffix}-suite"
    root.mkdir(exist_ok=True)
    if a.stage == 6:
        import check_common_s6 as suite
        import check_tlx_s6 as checker

        original_exercise = suite.exercise

        def exercise(*args):
            return original_exercise(*args[:-1], "T2-" + args[-1])

        suite.exercise = exercise
        suite.LOG = root
        checker.dump = lambda kernel, stage, side: tlx_ir.dump(
            kernel, stage.replace("T1-S6", "T2-S6" + suffix), side
        )
    else:
        import check_tlx_s3 as checker
    checker.LOG = root
    argv = (
        ["checker", "--seed", str(a.seed)]
        + (["--exact-only"] if a.exact_only else [])
        + (["--suite-only"] if a.suite_only else [])
    )
    seen = set()
    original = CompiledKernel.launch_metadata

    def metadata(kernel, *args, **kwargs):
        result = original(kernel, *args, **kwargs)
        if a.stage == 3 and not a.suite_only:
            module = kernel.src.fn.fn.__module__.rsplit(".", 1)[-1]
            function = kernel.src.fn.fn.__name__
            if module in (
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
                if phase and (phase, side) not in seen:
                    tlx_ir.dump(kernel, "T2-S3" + suffix + "/" + phase, side)
                    seen.add((phase, side))
        return result

    with (
        patch.object(sys, "argv", argv),
        patch.object(CompiledKernel, "launch_metadata", metadata),
    ):
        checker.main()


if __name__ == "__main__":
    main()
