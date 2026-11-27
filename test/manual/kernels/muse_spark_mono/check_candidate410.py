"""Validate unselected S2 split dispatches with existing stage/layer checkers."""

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    chained,
    split_attention,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", action="store_true")
    p.add_argument("--combined", action="store_true")
    a, remaining = p.parse_known_args()
    if a.suite:
        import check_common_s2 as suite

        out = Path("/home/bangtliu/coord/logs/phase410-S2-suite")
        out.mkdir(exist_ok=True)
        suite.stage = split_attention
        suite.ROOT = out
        suite.suite.LOG = out
        sys.argv = [sys.argv[0], *remaining]
        suite.main()
    else:
        import check_chained47 as suite

        original = chained.make_workspace

        def factory(device, *, context, left, **kw):
            ws = original(device, context=context, left=left, **kw)
            ws.s2 = split_attention.make_workspace(
                device, context=context, window_left=left
            )
            if a.combined:
                from candidate_policy410 import configure

                configure(ws, device, context, left)
            return ws

        sys.argv = [sys.argv[0], *remaining]
        with (
            patch.object(chained, "make_workspace", factory),
            patch.object(chained.s2, "attention", split_attention.attention),
        ):
            suite.main()


if __name__ == "__main__":
    main()
