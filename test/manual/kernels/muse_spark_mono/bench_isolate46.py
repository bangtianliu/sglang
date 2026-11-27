"""GPU0 runtime-mask fused-vs-extracted-stage isolation, exact archive bits."""

import argparse
import hashlib
import json
from unittest.mock import patch

import layer46
import torch
from layer_fixture import require_gpu0
from mono_case45 import LOG, case
from paired45 import measure

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    layer_flow as flow,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.code_probe import (
    text_info,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import layer

OUTPUTS = {
    1: ("q", "k", "v", "gate"),
    2: ("attn",),
    3: ("x4", "mid_stream"),
    4: (),
    5: ("x6",),
    6: ("out", "stream"),
}


def exact(a, b, name):
    if a.dtype == torch.bfloat16:
        a, b = a.view(torch.int16), b.view(torch.int16)
    elif a.dtype == torch.float32:
        a, b = a.view(torch.int32), b.view(torch.int32)
    if not torch.equal(a, b):
        at = tuple(int(v) for v in (a != b).nonzero()[0].tolist())
        raise AssertionError(
            dict(
                tensor=name,
                rank=0,
                location=at,
                actual=a[at].item(),
                expected=b[at].item(),
            )
        )


class Entry:
    def __init__(self, control, storage, only):
        self.control, self.storage, self.only = control, storage, only

    def options(self, kw):
        return dict(
            **kw, Control=self.control, GridStorage=self.storage, ONLY_STAGE=self.only
        )

    def warmup(self, *args, **kw):
        return layer46._launch.warmup(*args, **self.options(kw))

    def __getitem__(self, grid):
        return lambda *args, **kw: layer46._launch[grid](*args, **self.options(kw))


def table_values(ids):
    flat = ids.cpu().flatten().tolist()
    sorted_ids = torch.full((1024,), 64, dtype=torch.int32)
    experts = torch.full((64,), -1, dtype=torch.int32)
    at = 0
    for expert in range(64):
        hits = [i for i, e in enumerate(flat) if e == expert]
        if hits:
            assert len(hits) <= 8
            sorted_ids[at : at + len(hits)] = torch.tensor(hits, dtype=torch.int32)
            experts[at // 16] = expert
            at += 16
    return (
        sorted_ids.cuda(),
        experts.cuda(),
        torch.tensor([at], device="cuda", dtype=torch.int32),
    )


def factory(f, w, m, old, mask, *, only=0, barrier=0, accepted=False):
    ws = layer.make_workspace("cuda", context=8192, left=-1)
    ws.status = torch.empty((12, 768), device="cuda", dtype=torch.int32)
    control = torch.tensor([mask, barrier], device="cuda", dtype=torch.int32)
    storage = torch.zeros((5 * (24 + 1 + 768) * 32,), device="cuda", dtype=torch.int32)
    entry = Entry(control, storage, only)
    table = table_values(old["ids"])
    slots = f.locations

    def call(compile_only=False):
        if accepted:
            return layer.launch(
                f.x, f.stream, w, m, ws, rank=0, compile_only=compile_only
            )
        with patch.object(layer.layer_accepted, "_launch", entry):
            return layer.launch(
                f.x, f.stream, w, m, ws, rank=0, compile_only=compile_only
            )

    def prepare():
        ws.epochs.zero_()
        ws.counters.zero_()
        ws.flags.zero_()
        ws.s5.state.barriers.zero_()
        storage.zero_()
        ws.status.fill_(-3)
        for name in ("f1", "f3", "f4", "f6"):
            ws.buffers[name][: 2048 * 32].zero_()
        for name, t in old["buffers"].items():
            ws.buffers[name].copy_(t)
        ws.s4.ids.copy_(old["ids"])
        ws.s4.scores.copy_(old["scores"])
        ws.s4.pooled.copy_(old["pool"])
        for dst, src in zip(
            (ws.s4.sorted_ids, ws.s4.expert_ids, ws.s4.num_post), table
        ):
            dst.copy_(src)
        for name in ("cache_k", "cache_v"):
            getattr(f, name).view(torch.uint8).reshape(-1, 4, 64)[slots] = old["cache"][
                name
            ]
        for stage in range(1, 7):
            if mask & (1 << (stage - 1)):
                for name in OUTPUTS[stage]:
                    ws.buffers[name].fill_(float("nan"))
                if stage == 4:
                    ws.s4.ids.fill_(-1)
                    ws.s4.scores.fill_(float("nan"))
                    ws.s4.pooled.fill_(float("nan"))
        for stage, begin, end in (
            (2, flow.S2_INPUT, flow.S2_MERGE),
            (3, flow.S3_INPUT, flow.S3_ROWS),
            (4, flow.S4_INPUT, flow.S4_ROWS),
            (5, flow.S5_INPUT, flow.S6_INPUT),
            (6, flow.S6_INPUT, flow.S6_ROWS),
        ):
            if (mask & (1 << (stage - 1))) and not (mask & (1 << (stage - 2))):
                ws.flags[begin:end, 0].fill_(1)
        layer.prepare_peers(ws, old["partial3"], old["partial5"], rank=0)

    def check():
        exact(ws.status[:6], torch.ones_like(ws.status[:6]), "stage status")
        if barrier:
            exact(
                ws.status[7:12], torch.ones_like(ws.status[7:12]), "grid-barrier status"
            )
        exact(ws.s4.ids, old["ids"], "routing IDs")
        for name, t in old["buffers"].items():
            exact(ws.buffers[name], t, name)
        exact(ws.s4.scores, old["scores"], "scores")
        exact(ws.s4.pooled, old["pool"], "pool")
        for name in ("cache_k", "cache_v"):
            exact(
                getattr(f, name).view(torch.uint8).reshape(-1, 4, 64)[slots],
                old["cache"][name],
                name + " new slots",
            )

    k = call(compile_only=True)
    r = ws.shape.admit("stage46", k, grid=768)
    meta = dict(
        vgpr=k.n_regs,
        spills=k.n_spills,
        lds=k.metadata.shared,
        occupancy=r.active_ctas_per_cu,
        **text_info(k)[0],
        sha256=hashlib.sha256(k.asm["hsaco"]).hexdigest(),
        text_sha256=hashlib.sha256(text_info(k)[1]).hexdigest(),
    )
    return (prepare, call, check), meta, k


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode", choices=("overhead", "stages", "barrier"), default="overhead"
    )
    p.add_argument("--stage", type=int, choices=range(1, 7))
    a = p.parse_args()
    f, w, m, old = case()
    records = []
    fused_sha = None
    jobs = (
        [("accepted", 63, 0, 0, True), ("masked_all", 63, 0, 0, False)]
        if a.mode == "overhead"
        else (
            [("fine", 63, 0, 0, False), ("barrier", 63, 0, 1, False)]
            if a.mode == "barrier"
            else None
        )
    )
    pairs = (
        [("all", jobs)]
        if jobs
        else [
            (
                f"S{s}",
                [
                    (f"S{s}_standalone", 1 << (s - 1), s, 0, False),
                    (f"S{s}_fused", 1 << (s - 1), 0, 0, False),
                ],
            )
            for s in range(1, 7)
        ]
    )
    if a.stage is not None:
        assert a.mode == "stages"
        pairs = [row for row in pairs if row[0] == f"S{a.stage}"]
    for stage, jobs in pairs:
        callbacks = {}
        meta = {}
        for label, mask, only, barrier, accepted in jobs:
            callbacks[label], meta[label], k = factory(
                f, w, m, old, mask, only=only, barrier=barrier, accepted=accepted
            )
            for fmt in ("amdgcn", "llir", "hsaco"):
                path = LOG / f"phase46-{label}.{fmt}"
                data = k.asm[fmt]
                path.write_bytes(data) if isinstance(data, bytes) else path.write_text(
                    data
                )
            if not accepted and only == 0:
                if fused_sha is None:
                    fused_sha = meta[label]["sha256"]
                assert meta[label]["sha256"] == fused_sha, (
                    "runtime masks/barrier must share identical binary"
                )
            print(
                json.dumps(dict(event="compiled", label=label, **meta[label])),
                flush=True,
            )
        result, _ = measure(callbacks)
        row = dict(stage=stage, resources=meta, timing=result)
        records.append(row)
        (
            LOG / (f"phase46-{a.mode}" + (f"-S{a.stage}" if a.stage else "") + ".json")
        ).write_text(json.dumps(records, indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    event="paired",
                    stage=stage,
                    summary={
                        k: {n: v for n, v in x.items() if n != "samples_by_order_us"}
                        for k, x in result.items()
                    },
                )
            ),
            flush=True,
        )
        del callbacks


if __name__ == "__main__":
    main()
