"""Run archived exact/replay/missing chain checks with selected TLX stages."""

import argparse
import functools
import hashlib
import sys
from pathlib import Path
from unittest.mock import patch

import check_chained47 as suite
from tlx_launch_audit import audited_tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_chain


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--stages", default="selected")
    p.add_argument("--suite-label", default="T3c-mixed-chain")
    a, remaining = p.parse_known_args()
    stages = (
        None if a.stages == "selected" else tuple(int(x) for x in a.stages.split(","))
    )
    original = tlx_chain.make_workspace
    suite.chained = tlx_chain
    suite.LOG = Path("/home/bangtliu/coord/logs") / a.suite_label
    suite.LOG.mkdir(exist_ok=True)
    label = (
        a.suite_label
        + "-"
        + hashlib.sha256(" ".join(remaining).encode()).hexdigest()[:12]
    )
    with (
        audited_tlx(label),
        patch.object(
            tlx_chain,
            "make_workspace",
            original
            if stages is None
            else functools.partial(original, tlx_stages=stages),
        ),
        patch.object(sys, "argv", ["suite", *remaining]),
    ):
        suite.main()


if __name__ == "__main__":
    main()
