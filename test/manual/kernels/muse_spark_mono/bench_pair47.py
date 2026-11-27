"""One adjacent-pair trial against the committed six-dispatch baseline."""

import argparse
import json
import random
import statistics
import struct

import torch
from check_chained47 import LOG, archive_check
from layer_fixture import require_gpu0
from mono_case45 import case
from pair_runtime47 import Pair
from paired45 import measure

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.code_probe import (
    text_info,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import chained


def verdict(timing, base, candidate):
    rng = random.Random(47)
    out = {}
    win = True
    for order in ("AB", "BA"):
        a = timing[base]["samples_by_order_us"][order]
        b = timing[candidate]["samples_by_order_us"][order]
        delta = [x - y for x, y in zip(a, b)]
        effects = sorted(
            statistics.median(rng.choices(delta, k=len(delta))) for _ in range(1000)
        )
        gain = 1 - statistics.median(b) / statistics.median(a)
        out[order] = dict(
            relative_median_reduction=gain,
            paired_median_delta_us=statistics.median(delta),
            paired_bootstrap_95_delta_us=[effects[25], effects[974]],
        )
        win = win and gain > 0.02 and effects[25] > 0
    return dict(
        win=bool(win),
        orders=out,
        rule="Both order median reductions >2%, and paired-bootstrap95% lower delta >0; predeclared, no sweeping.",
    )


def build(f, w, m, old, pair=None, warm=None):
    ws = chained.make_workspace("cuda", context=8192, left=-1)
    executor = Pair(ws, pair, code_warm=warm) if pair else None

    def prepare():
        chained.prepare_peers(ws, old["partial3"], old["partial5"], rank=0)

    def call():
        if executor:
            with executor.context():
                return chained.launch(f.x, f.stream, w, m, ws, rank=0)
        return chained.launch(f.x, f.stream, w, m, ws, rank=0)

    def check():
        archive_check(f, ws, old, 0, digest=False)
        if executor:
            assert bool(executor.status.eq(1).all()), dict(
                tensor="pair barrier",
                rank=0,
                location=executor.status.ne(1).nonzero()[0].tolist(),
            )

    return (prepare, call, check), executor


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--pair", choices=("12", "34", "56"), required=True)
    p.add_argument("--mode", choices=("baseline", "warm", "final"), default="baseline")
    a = p.parse_args()
    f, w, m, old = case()
    pair = tuple(map(int, a.pair))
    callbacks = {}
    executors = {}
    labels = ("cold", "warm") if a.mode == "warm" else ("baseline", "candidate")
    for label in labels:
        is_base = label == "baseline"
        control = None
        if a.mode in ("warm", "final") and not is_base:
            control = torch.zeros((18,), device="cuda", dtype=torch.int64)
        callbacks[label], executors[label] = build(
            f, w, m, old, None if is_base else pair, control
        )
        # Admit and determine runtime text controls before graph capture.
        prepare, call, check = callbacks[label]
        prepare()
        call()
        torch.cuda.synchronize()
        check()
        ex = executors[label]
        if ex is not None and control is not None:
            info, body = text_info(ex.last_kernel, marker=True)
            control[0] = info["pc_offset"]
            control[1] = info["words"] if label in ("warm", "candidate") else 0
            if label in ("warm", "candidate"):
                words = struct.unpack(
                    "<" + "I" * info["words"], body[: info["words"] * 4]
                )
                partial = [0, 0]
                for i, v in enumerate(words):
                    partial[i // 512 % 2] ^= v
                expected = [partial[t // 6] for t in range(12)] + [0] * 4
    timing, _ = measure(callbacks)
    resources = {
        label: list(ex.admitted.values()) if ex else None
        for label, ex in executors.items()
    }
    if a.mode in ("warm", "final"):
        hot = executors["warm" if a.mode == "warm" else "candidate"]
        observed = [int(v) & 0xFFFFFFFF for v in hot.code_warm[2:].tolist()]
        assert observed == expected, dict(
            tensor="code checksum", rank=0, location=observed, expected=expected
        )
    result = dict(
        pair=a.pair,
        mode=a.mode,
        timing=timing,
        resources=resources,
        verdict=verdict(timing, *labels),
        dispatches=dict(zip(labels, (6 if a.mode != "warm" else 5, 5))),
    )
    (LOG / f"phase47-pair{a.pair}-{a.mode}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    for label, ex in executors.items():
        if ex:
            for fmt in ("amdgcn", "llir", "hsaco"):
                data = ex.last_kernel.asm[fmt]
                path = LOG / f"phase47-pair{a.pair}-{a.mode}-{label}.{fmt}"
                path.write_bytes(data) if isinstance(data, bytes) else path.write_text(
                    data
                )
    print(
        json.dumps(
            dict(
                pair=a.pair,
                mode=a.mode,
                verdict=result["verdict"],
                resources=resources,
                summary={
                    k: {n: v for n, v in x.items() if n != "samples_by_order_us"}
                    for k, x in timing.items()
                },
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
