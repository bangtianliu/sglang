"""Manual T9.1d low-fan-out S4 pair, with explicit predecessor epoch binding."""

import json
from pathlib import Path
from unittest.mock import patch

import torch
from tlx_ir import dump

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_route as base,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_route_handoff as pair,
)

_original_routing = base.routing


class Proxy:
    def __init__(self, raw, action):
        self.raw, self.action = raw, action

    def __getattr__(self, k):
        return getattr(self.raw, k)

    def __getitem__(self, grid):
        return self.action


class Adapter:
    def __init__(self, mode, label):
        assert mode in ("serial", "overlap")
        self.mode = mode
        self.label = label
        self.bindings = {}
        self.spaces = {}
        self.admissions = []
        self.skip_row = -1

    def bind_workspace(self, ws):
        self.bindings[id(ws.s4)] = ws.s3.state.epochs

    def routing(self, *aa, **kw):
        ws = kw["workspace"]
        epoch = self.bindings[id(ws)]
        key = (id(ws), kw.get("delay_seed") is None, self.skip_row)
        if key not in self.spaces:
            self.spaces[key] = pair.make_workspace(epoch)
        extra = self.spaces[key]
        saved = []

        def deferred(*args, **opts):
            saved.append((args, opts))

        def launch(*args, **opts):
            first, fo = saved.pop()
            pa = (*first[:9], extra.epoch, extra.flags)
            po = dict(fo, SKIP_ROW=self.skip_row)
            ca = (*args[:7], extra.epoch, extra.flags, extra.record, extra.polls)
            co = dict(opts)
            if not hasattr(self, "admitted"):
                self.admitted = set()
            if key not in self.admitted:
                pk = pair.produce.warmup(*pa, **po, grid=(8,))
                ck = pair.consume.warmup(*ca, **co, grid=(64,))
                pr = kernel_occupancy("T9-row-producer", pk)
                cr = kernel_occupancy("T9-slot-consumer", ck)
                assert 8 + 64 < 192 * min(pr.active_ctas_per_cu, cr.active_ctas_per_cu)
                assert pk.n_spills == ck.n_spills == 0
                self.admissions.append(
                    dict(
                        producer=dump(pk, self.label + "/producer", "tlx"),
                        consumer=dump(ck, self.label + "/consumer", "tlx"),
                        producer_grid=8,
                        consumer_grid=64,
                        common_capacity=192
                        * min(pr.active_ctas_per_cu, cr.active_ctas_per_cu),
                        epoch="completed predecessor S3 epoch[0]",
                        extra_init_kernels=0,
                    )
                )
                Path(
                    "/home/bangtliu/coord/logs", self.label + "-admission.json"
                ).write_text(json.dumps(self.admissions, indent=2) + "\n")
                self.admitted.add(key)
            origin = torch.cuda.current_stream()
            if self.mode == "overlap":
                extra.stream.wait_stream(origin)
            pair.produce[(8,)](*pa, **po)
            if self.mode == "overlap":
                with torch.cuda.stream(extra.stream):
                    result = pair.consume[(64,)](*ca, **co)
                origin.wait_stream(extra.stream)
            else:
                result = pair.consume[(64,)](*ca, **co)
            return result

        with (
            patch.object(base, "finish_rows", Proxy(base.finish_rows, deferred)),
            patch.object(base, "slots", Proxy(base.slots, launch)),
        ):
            return _original_routing(*aa, **kw)


def configure(module, mode, label):
    adapter = Adapter(mode, label)
    module.adapter = adapter
    module.routing = adapter.routing
    module.bind_workspace = adapter.bind_workspace
    module.make_workspace = base.make_workspace
    module.Workspace = base.Workspace
