"""S1 wide projection: reuse native/FP64/cache and delayed-replay contracts."""

import shutil
from pathlib import Path

import check_common_s1 as suite

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import independent_qkvg


def main():
    suite.stage = independent_qkvg
    original = suite.suite.one_case

    def case(*args, **kw):
        kw["ctas"] = 288
        return original(*args, **kw)

    suite.suite.one_case = case
    out = Path("/home/bangtliu/coord/logs/phase410-S1-suite")
    out.mkdir(exist_ok=True)
    shutil.copyfile(
        out.parent / "phase3.0-read-and-copy-traffic.json",
        out / "phase3.0-read-and-copy-traffic.json",
    )
    suite.ROOT = out
    suite.suite.LOG = out
    suite.main()


if __name__ == "__main__":
    main()
