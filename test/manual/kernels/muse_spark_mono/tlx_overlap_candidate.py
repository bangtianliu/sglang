"""Manual cold-A/B adapter for the experimental S5 adjacent dataflow pair."""

import json
from pathlib import Path
from unittest.mock import patch

import torch
from tlx_ir import dump

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_expert_overlap as pair,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_experts as base,
)


class Adapter:
    def __init__(self, mode, label):
        assert mode in ("agent", "serial", "overlap", "prefetch")
        self.mode = mode
        self.label = label
        self.raw = base.expert_phase
        self.saved = None
        self.spaces = {}
        self.records = []

    def __getattr__(self, k):
        return getattr(self.raw, k)

    def __getitem__(self, grid):
        def call(*aa, **kw):
            if self.mode == "agent":
                kw = dict(kw)
                kw["CROSS_LAUNCH_A"] = False
                return self.raw[grid](*aa, **kw)
            if kw["FIRST"]:
                self.saved = (aa, kw)
                return None
            first, k1 = self.saved
            self.saved = None
            return self.launch(first, aa, k1, kw)

        return call

    def launch(self, a1, a2, k1, k2):
        key = (a1[7].data_ptr(), k1.get("SKIP", -1), k1.get("Seed") is None)
        if key not in self.spaces:
            ws = pair.make_workspace(a1[0].device)
            pg = 768
            cg = 768
            p_args = (*a1, ws.epoch, ws.counters, ws.flags)
            c_args = (*a2, ws.epoch, ws.flags, ws.record, ws.checks)
            po = dict(
                GRID=pg,
                SKIP=k1.get("SKIP", -1),
                Seed=k1.get("Seed"),
                num_warps=4,
                waves_per_eu=9,
                enable_fp_fusion=False,
            )
            co = dict(
                GRID=cg,
                WARM=self.mode == "prefetch",
                Seed=k2.get("Seed"),
                num_warps=4,
                waves_per_eu=9,
                enable_fp_fusion=False,
            )
            for _ in range(2):
                pk = pair.produce.warmup(*p_args, **po, grid=(pg,))
                ck = pair.consume.warmup(*c_args, **co, grid=(cg,))
                pr = kernel_occupancy("T9-producer", pk)
                cr = kernel_occupancy("T9-consumer", ck)
                capacity = min(pr.active_ctas_per_cu, cr.active_ctas_per_cu)
                if pg + cg <= 192 * (capacity - 1):
                    break
                cg = max(192, (192 * (capacity - 1) - pg) // 192 * 192)
                co["GRID"] = cg
            assert pg + cg <= 192 * (capacity - 1), (pg, cg, pr, cr)
            assert pk.n_spills == ck.n_spills == 0, (pk.n_spills, ck.n_spills)
            assert pr.waves == cr.waves == 4
            resource = dict(
                pg=pg,
                cg=cg,
                capacity_per_cu=capacity,
                one_CTA_per_CU_slack=True,
                producer=dump(pk, self.label + "/producer", "tlx"),
                consumer=dump(ck, self.label + "/consumer", "tlx"),
            )
            self.records.append(resource)
            Path(
                "/home/bangtliu/coord/logs", self.label + "-admission.json"
            ).write_text(json.dumps(self.records, indent=2) + "\n")
            self.spaces[key] = (ws, pg, cg, po, co)
        ws, pg, cg, po, co = self.spaces[key]
        pair.initialize[(1,)](ws.epoch, ws.counters, ws.record, a1[7], num_warps=4)
        origin = torch.cuda.current_stream()
        if self.mode != "serial":
            ws.stream.wait_stream(origin)
        pair.produce[(pg,)](*a1, ws.epoch, ws.counters, ws.flags, **po)
        if self.mode == "serial":
            result = pair.consume[(cg,)](
                *a2, ws.epoch, ws.flags, ws.record, ws.checks, **co
            )
        else:
            with torch.cuda.stream(ws.stream):
                result = pair.consume[(cg,)](
                    *a2, ws.epoch, ws.flags, ws.record, ws.checks, **co
                )
            origin.wait_stream(ws.stream)
        return result


def configure(module, mode, label):
    original = base.s5
    if mode == "roles":
        from tlx_role_candidate import RoleAdapter

        adapter = RoleAdapter(base.expert_phase, label)
    elif mode.startswith("model-") or mode == "geometry":
        from tlx_model_candidate import ModelAdapter

        adapter = ModelAdapter(base.expert_phase, label, mode)
    else:
        adapter = Adapter(mode, label)
    module.Workspace = base.Workspace
    module.make_workspace = base.make_workspace

    def s5(*aa, **kw):
        if mode.startswith("model-") or mode == "geometry":
            adapter.epoch = aa[6].state.epochs
        with patch.object(base, "expert_phase", adapter):
            return original(*aa, **kw)

    module.s5 = s5
    module.adapter = adapter
