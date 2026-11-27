"""Compile/admit before overlapping an 8-CTA S3 boundary with S4 projection."""

import json
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from tlx_ir import dump

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_chain as chain,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_projection_overlap as consumer,
)


class RowPair:
    def __init__(self, label, mode, trace=False):
        self.trace = trace
        self.events = [
            torch.cuda.Event(enable_timing=True, external=True) for _ in range(4)
        ]
        self.label = label
        self.mode = mode
        self.stream = torch.cuda.Stream(device="cuda")
        self.epoch = torch.zeros((), device="cuda", dtype=torch.int32)
        self.counter = torch.zeros((), device="cuda", dtype=torch.int32)
        self.flags = torch.zeros((288 * 32,), device="cuda", dtype=torch.int32)
        self.record = torch.zeros((8,), device="cuda", dtype=torch.int32)
        self.saved = None
        self.admitted = False

    def verify(self):
        assert self.record.eq(0).all(), ("row overlap stuck", self.record.tolist())
        assert int(self.counter) == 8, ("row publication", int(self.counter))

    @contextmanager
    def install(self, *_):
        owner = self
        raw_p = chain.tlx_attn_tail.s3_boundary
        module = chain.tlx_route.tlx_projection
        raw_c = module.project

        class Producer:
            def __getattr__(self, k):
                return getattr(raw_p, k)

            def __getitem__(self, grid):
                def save(*aa, **kw):
                    owner.saved = (aa, kw)

                return save

        class Consumer:
            def __getattr__(self, k):
                return getattr(raw_c, k)

            def __getitem__(self, grid):
                fn = raw_c[grid]

                def call(*aa, **kw):
                    if kw["STAGE"] != 4:
                        return fn(*aa, **kw)
                    pa, po = owner.saved
                    owner.saved = None
                    po = dict(
                        po,
                        NEXT_FLAGS=owner.flags,
                        NEXT_COUNTER=owner.counter,
                        NEXT_EPOCH=owner.epoch,
                    )
                    co = dict(
                        kw, READY=owner.flags, EPOCH=owner.epoch, RECORD=owner.record
                    )
                    if not owner.admitted:
                        pk = raw_p.warmup(*pa, **po, grid=(8,))
                        ck = consumer.project.warmup(*aa, **co, grid=(288,))
                        pr = kernel_occupancy("T9-S3-producer", pk)
                        cr = kernel_occupancy("T9-S4-consumer", ck)
                        assert 8 + 288 < 192 * min(
                            pr.active_ctas_per_cu, cr.active_ctas_per_cu
                        ), (pr, cr)
                        assert pk.n_spills == ck.n_spills == 0
                        resource = dict(
                            producer=dump(pk, owner.label + "/producer", "tlx"),
                            consumer=dump(ck, owner.label + "/consumer", "tlx"),
                            grids=[8, 288],
                            common_resident_capacity=192
                            * min(pr.active_ctas_per_cu, cr.active_ctas_per_cu),
                            proof="Any common-occupancy number of either CTA fits each resource separately; their componentwise mixture fits too.296 CTAs <384slots. Producer has no dependency on consumer; virtual flags prefilled. All consumer waits bounded4096, failure record retained.",
                        )
                        Path(
                            "/home/bangtliu/coord/logs", owner.label + "-admission.json"
                        ).write_text(json.dumps(resource, indent=2) + "\n")
                        owner.admitted = True
                    consumer.reset[(1,)](
                        owner.epoch, owner.counter, owner.record, num_warps=2
                    )
                    origin = torch.cuda.current_stream()
                    if owner.mode == "overlap":
                        owner.stream.wait_stream(origin)
                    if owner.trace:
                        owner.events[0].record()
                    raw_p[(8,)](*pa, **po)
                    if owner.trace:
                        owner.events[1].record()
                    if owner.mode == "overlap":
                        with torch.cuda.stream(owner.stream):
                            if owner.trace:
                                owner.events[2].record()
                            result = consumer.project[(288,)](*aa, **co)
                            if owner.trace:
                                owner.events[3].record()
                        origin.wait_stream(owner.stream)
                    else:
                        if owner.trace:
                            owner.events[2].record()
                        result = consumer.project[(288,)](*aa, **co)
                        if owner.trace:
                            owner.events[3].record()
                    return result

                return call

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(chain.tlx_attn_tail, "s3_boundary", Producer())
            )
            stack.enter_context(patch.object(module, "project", Consumer()))
            yield
