"""Manual single-threaded capture of two public common-API calls into one kernel."""

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pairs47_device as device
import torch

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.code_probe import (
    text_info,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_attention as a2,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_attn_tail as a3,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_experts as a5,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_qkvg as a1,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_route as a4,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    common_tail as a6,
)

SCHEMA = json.loads(Path(__file__).with_name("pairs47_device.schema.json").read_text())
MODULES = {1: a1, 2: a2, 3: a3, 4: a4, 5: a5, 6: a6}


class Recorder:
    def __init__(self, owner, stage, original):
        self.owner, self.stage, self.original = owner, stage, original

    def warmup(self, *args, **kw):
        return self.original.warmup(*args, **kw)

    def __getitem__(self, grid):
        def capture(*args, **kw):
            self.owner.records[self.stage] = (args, kw)
            if self.stage == self.owner.pair[1]:
                self.owner.run_pair()

        return capture


class Pair:
    def __init__(self, ws, pair, code_warm=None):
        self.ws, self.pair, self.code_warm = ws, pair, code_warm
        self.records = {}
        self.admitted = {}
        self.last_kernel = None
        self.epochs = torch.zeros((768, 32), device="cuda", dtype=torch.int32)
        self.sync = torch.zeros(
            ((24 + 1 + 768) * 32,), device="cuda", dtype=torch.int32
        )
        self.status = torch.empty((768,), device="cuda", dtype=torch.int32)
        self.decision = torch.empty((768, 32), device="cuda", dtype=torch.int32)
        self.fn = getattr(device, f"pair{pair[0]}{pair[1]}")

    def context(self):
        stack = ExitStack()
        for stage in self.pair:
            mod = MODULES[stage]
            stack.enter_context(
                patch.object(mod, "_launch", Recorder(self, stage, mod._launch))
            )
        return stack

    def pack(self, stage):
        args, kw = self.records[stage]
        schema = SCHEMA[str(stage)]
        assert len(args) == len(schema["positional"]), (stage, len(args), schema)
        return (
            args,
            tuple(kw[k] for k in schema["constants"]),
            tuple(kw.get(k, schema["defaults"].get(k)) for k in schema["runtime"]),
        )

    def run_pair(self):
        aa, ca, ra = self.pack(self.pair[0])
        bb, cb, rb = self.pack(self.pair[1])
        shape = self.ws.s1.shape
        args = (
            aa,
            ca,
            ra,
            bb,
            cb,
            rb,
            self.epochs,
            self.sync,
            self.status,
            self.ws.failure,
            self.decision,
        )
        opts = dict(GRID=768, CodeWarm=self.code_warm, **shape.options())
        key = (ca, cb, tuple(None if x is None else x.dtype for x in ra + rb))
        if key not in self.admitted:
            k = self.fn.warmup(*args, **opts, grid=(768,))
            r = shape.admit("pair47", k, grid=768)
            info, body = text_info(k, marker=self.code_warm is not None)
            if self.code_warm is not None:
                assert info["words"] <= 1048576
                # All benchmark launches use a validated ELF .text range.
                self.code_warm[0] = info["pc_offset"]
            self.admitted[key] = dict(
                vgpr=k.n_regs,
                spills=k.n_spills,
                lds=k.metadata.shared,
                occupancy=r.active_ctas_per_cu,
                **info,
                sha256=hashlib.sha256(k.asm["hsaco"]).hexdigest(),
            )
            self.last_kernel = k
        self.fn[(768,)](*args, **opts)
