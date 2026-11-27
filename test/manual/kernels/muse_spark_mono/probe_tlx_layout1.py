"""Bounded T1 S6 layout pinning pass using the accepted T0 proof harness."""

import argparse
from pathlib import Path

import proof_tlx_projection as proof
import tlx_ir

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_projection


class Pinned:
    def __init__(self, post):
        self.post = post

    def warmup(self, *aa, **kw):
        return tlx_projection.project.warmup(
            *aa, PINNED=True, POST_BARRIER=self.post, **kw
        )

    def __getitem__(self, grid):
        return lambda *aa, **kw: tlx_projection.project[grid](
            *aa, PINNED=True, POST_BARRIER=self.post, **kw
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--post", action="store_true")
    args = parser.parse_args()
    label = "post" if args.post else "pin"
    proof.project = Pinned(args.post)
    proof.LOG = Path("/home/bangtliu/coord/logs") / ("phaseT1-projection-" + label)
    proof.LOG.mkdir(exist_ok=True)
    tlx_ir.ROOT = Path("/home/bangtliu/coord/logs/phaseT-ir") / (
        "T1-projection-" + label
    )
    proof.main()


if __name__ == "__main__":
    main()
