"""Regression: gfx1250 VMEM source VGPR reuse corrupted scoped-store payloads.

Sixteen waves keep sixteen2048-word tiles live, overwrite their source VGPRs
between stores, and drain only after all rounds. Distinct int32 words catch
corruption hidden by constant payloads. GPU0; run under timeout -s KILL300.
"""

import json
import re
import unittest
from pathlib import Path

import torch
from layer_fixture import require_gpu0
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped
from sglang.test.test_utils import CustomTestCase


@g.jit
def pressure(X, Scratch, Out, ROUNDS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [1, 16], [1, 0])
    tile = gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
    col = gl.arange(0, 2048, layout=gl.SliceLayout(0, layout))
    index = gl.program_id(0) * 32768 + tile[:, None] * 2048 + col[None, :]
    value = gl.load(X + index)
    mask = (col[None, :] % 17 != 0) & (tile[:, None] >= 0)
    for step in range(ROUNDS):
        scoped.store_words(Scratch + step * 8 * 32768 + index, value, mask)
        # The old asm hid the VMEM read of value/address VGPRs from LLVM.
        value = value * 1664525 + 1013904223
    scoped.drain_stores()
    for step in range(ROUNDS):
        words = scoped.load_words_masked(Scratch + step * 8 * 32768 + index, index >= 0)
        gl.store(Out + step * 8 * 32768 + index, words)


class TestScopedXcnt(CustomTestCase):
    def test_register_reuse(self):
        require_gpu0()
        rounds = 16
        x = torch.arange(8 * 32768, device="cuda", dtype=torch.int32)
        scratch = torch.full(
            (rounds, 8, 16, 2048), -1, device="cuda", dtype=torch.int32
        )
        out = torch.empty_like(scratch)
        compiled = pressure[(8,)](x, scratch, out, rounds, num_warps=16)
        torch.cuda.synchronize()
        expected = []
        value = x.reshape(8, 16, 2048)
        mask = torch.arange(2048, device="cuda") % 17 != 0
        for _ in range(rounds):
            expected.append(torch.where(mask, value, -1))
            value = value * 1664525 + 1013904223
        reference = torch.stack(expected)
        wrong = out.ne(reference).sum().item()
        print(
            json.dumps(
                dict(
                    event="xcnt_regression",
                    wrong=wrong,
                    words=out.numel(),
                    waves=16,
                    tiles=16,
                )
            ),
            flush=True,
        )
        self.assertEqual(wrong, 0)
        asm = compiled.asm["amdgcn"]
        self.assertIn("s_wait_xcnt", asm)
        path = Path("/home/bangtliu/coord/logs/phase3d-S3-fix-regression.amdgcn")
        path.write_text(asm)
        count = re.search(r"\.amdhsa_next_free_vgpr\s+(\d+)", asm)
        print(
            json.dumps(dict(next_free_vgpr=int(count[1]) if count else None)),
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
