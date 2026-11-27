"""One-GPU gfx1250 primitives; every invocation needs timeout -s KILL 300."""

import os
import time
import unittest
from pathlib import Path

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import sync
from sglang.test.test_utils import CustomTestCase


@gluon.jit
def ordering(Payload, Flag, Out, Tag):
    i = gl.arange(0, 1024, layout=gl.BlockedLayout([1], [32], [8], [0]))
    if gl.program_id(0) == 0:
        gl.store(Payload + i, (i + Tag).to(gl.int32))
        sync.publish(Flag, Tag)
    else:
        seen = sync.wait_flag(Flag, Tag, 10_000_000_000)
        values = gl.load(Payload + i, volatile=True)
        good = gl.sum((values == i + Tag).to(gl.int32), 0)
        gl.store(Out, (seen == Tag) & (good == 1024))


@gluon.jit
def timestamp(Out, SLEEP: gl.constexpr):
    before = sync.clock()
    for _ in range(SLEEP):
        sync.backoff()
    after = sync.clock()
    gl.store(Out, before)
    gl.store(Out + 1, after)


def evidence(name, kernel):
    root = Path(os.environ.get("MONO_LOG_DIR", "/home/bangtliu/coord/logs"))
    root.mkdir(parents=True, exist_ok=True)
    (root / f"phase2.1-{name}.amdgcn").write_text(kernel.asm["amdgcn"])


class TestPrimitives(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.cuda.is_available() or not torch.cuda.get_device_properties(
            0
        ).gcnArchName.startswith("gfx1250"):
            raise unittest.SkipTest("requires gfx1250")
        assert torch.cuda.device_count() == 1, "expose GPU0 only"

    def test_flag_ordering(self):
        payload = torch.empty(1024, device="cuda", dtype=torch.int32)
        flag = torch.zeros(1, device="cuda", dtype=torch.int32)
        out = torch.zeros(1, device="cuda", dtype=torch.int32)
        kernel = ordering[(2,)](payload, flag, out, 1, num_warps=8)
        evidence("ordering", kernel)
        for tag in range(2, 1002):
            ordering[(2,)](payload, flag, out, tag, num_warps=8)
            assert out.item() == 1, tag
        print(
            "PASS 1001 launches: 4 KB stores + CTA barrier + release/acquire agent flag",
            flush=True,
        )

    def test_clock_and_backoff(self):
        out = torch.empty(2, device="cuda", dtype=torch.int64)
        kernel = timestamp[(1,)](out, SLEEP=32, num_warps=1)
        torch.cuda.synchronize()
        evidence("clock-backoff", kernel)
        first = out[1].item()
        start = time.monotonic()
        time.sleep(1.0)
        timestamp[(1,)](out, SLEEP=32, num_warps=1)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        values = out.tolist()
        hz = (values[1] - first) / elapsed
        assert values[1] > values[0]
        assert 1_000_000 < hz < 10_000_000_000, hz
        print(
            f"CLOCK host_seconds={elapsed:.6f} ticks_per_second={hz:.3f} backoff_ticks={values[1] - values[0]}",
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
