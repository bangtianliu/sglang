"""Manual adapter; measure in-kernel idle roles without extra launches/events."""

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from tlx_ir import dump

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_idle_warm


class InlineWarm:
    def __init__(self, weight, label):
        self.weight = weight
        self.label = label
        self.admitted = False
        self.checks = torch.empty((184,), device=weight.device, dtype=torch.uint32)
        self.expected = weight[:, :1536].view(torch.uint16).to(
            torch.int64
        ).sum().item() % (1 << 32)

    def verify(self):
        assert int(self.checks.to(torch.int64).sum()) % (1 << 32) == self.expected, (
            "inline head checksum"
        )

    @contextmanager
    def install(self, module, name):
        raw = getattr(module, name)
        owner = self
        kernel = tlx_idle_warm.s3 if name == "s3_boundary" else tlx_idle_warm.s6

        class Proxy:
            def __getattr__(self, k):
                return getattr(raw, k)

            def __getitem__(self, grid):
                def call(*aa, **kw):
                    options = dict(
                        kw,
                        WarmW=owner.weight,
                        Checks=owner.checks,
                        WARM_N=owner.weight.shape[0],
                        WARM_K=owner.weight.shape[1],
                        WIDTH=1536,
                        WARM_GRID=184,
                        HEAD_SLEEP=32,
                    )
                    if not owner.admitted:
                        compiled = kernel.warmup(*aa, **options, grid=(192,))
                        resource = kernel_occupancy("T9-inline-warm", compiled)
                        assert (
                            resource.active_ctas_per_cu >= 1 and compiled.n_spills == 0
                        ), resource
                        r = dump(compiled, owner.label + "/boundary_warm", "tlx")
                        r["grid"] = 192
                        r["row_owners"] = 8
                        r["warm_owners"] = 184
                        r["read_bytes"] = owner.weight.shape[0] * 1536 * 2
                        Path(
                            "/home/bangtliu/coord/logs", owner.label + "-admission.json"
                        ).write_text(json.dumps(r, indent=2) + "\n")
                        owner.admitted = True
                    return kernel[(192,)](*aa, **options)

                return call

        with patch.object(module, name, Proxy()):
            yield
