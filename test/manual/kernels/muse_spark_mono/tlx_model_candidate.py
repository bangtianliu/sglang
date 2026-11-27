"""T9.1b controls: geometry alone, serial release/acquire, gated overlap."""

import json
from pathlib import Path

import torch
from tlx_ir import dump

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_model_handoff as model,
)


class ModelAdapter:
    def __init__(self, raw, label, mode):
        self.raw = raw
        self.label = label
        self.mode = mode
        self.epoch = None
        self.saved = None
        self.spaces = {}
        self.initialized = {}
        self.admissions = []

    def __getattr__(self, k):
        return getattr(self.raw, k)

    def __getitem__(self, grid):
        def call(*aa, **kw):
            pointer = aa[7].data_ptr()
            if pointer not in self.initialized:
                # Logical idle status slots stay true. This allocation-time
                # initialization is not a per-replay flag/counter reset.
                aa[7].fill_(1)
                self.initialized[pointer] = aa[7]
            if self.mode == "geometry":
                options = dict(kw, GRID=768)
                return self.raw[(768,)](*aa, **options)
            if kw["FIRST"]:
                self.saved = (aa, kw)
                return None
            a1, k1 = self.saved
            self.saved = None
            key = (pointer, k1.get("SKIP", -1), k1.get("Seed") is None)
            if key not in self.spaces:
                ws = model.make_workspace(self.epoch)
                pg = 768
                cg = 768
                pa = (*a1, ws.epoch, ws.counters, ws.flags)
                ca = (*aa, ws.epoch, ws.flags, ws.record, ws.polls)
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
                    Seed=kw.get("Seed"),
                    num_warps=4,
                    waves_per_eu=9,
                    enable_fp_fusion=False,
                )
                for _ in range(2):
                    pk = model.produce.warmup(*pa, **po, grid=(pg,))
                    ck = model.consume.warmup(*ca, **co, grid=(cg,))
                    pr = kernel_occupancy("T9-model-producer", pk)
                    cr = kernel_occupancy("T9-model-consumer", ck)
                    capacity = min(pr.active_ctas_per_cu, cr.active_ctas_per_cu)
                    if pg + cg < 192 * capacity:
                        break
                    cg = max(192, (192 * (capacity - 1) - pg) // 192 * 192)
                    co["GRID"] = cg
                assert pg + cg < 192 * capacity, (pr, cr, pg, cg)
                assert pk.n_spills == ck.n_spills == 0
                self.admissions.append(
                    dict(
                        producer=dump(pk, self.label + "/producer", "tlx"),
                        consumer=dump(ck, self.label + "/consumer", "tlx"),
                        pg=pg,
                        cg=cg,
                        common_resident_capacity=192 * capacity,
                        epoch_source="S5 sum/AR epoch[0]+1; advances only after pair completes",
                        per_call_init_kernels=0,
                    )
                )
                Path(
                    "/home/bangtliu/coord/logs", self.label + "-admission.json"
                ).write_text(json.dumps(self.admissions, indent=2) + "\n")
                self.spaces[key] = (ws, pg, cg, po, co)
            ws, pg, cg, po, co = self.spaces[key]
            origin = torch.cuda.current_stream()
            if self.mode == "model-overlap":
                ws.stream.wait_stream(origin)
            model.produce[(pg,)](*a1, ws.epoch, ws.counters, ws.flags, **po)
            if self.mode == "model-overlap":
                with torch.cuda.stream(ws.stream):
                    result = model.consume[(cg,)](
                        *aa, ws.epoch, ws.flags, ws.record, ws.polls, **co
                    )
                origin.wait_stream(ws.stream)
            else:
                result = model.consume[(cg,)](
                    *aa, ws.epoch, ws.flags, ws.record, ws.polls, **co
                )
            return result

        return call
