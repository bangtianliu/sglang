"""Same-binary self-code L2 warm A/B and mono-only profiler workload."""

import argparse
import functools
import json
import struct

import torch
from layer_fixture import require_gpu0
from mono_case45 import LOG, case, verify
from paired45 import measure

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.code_probe import (
    text_info,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import layer


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--counter", action="store_true")
    p.add_argument("--metadata", action="store_true")
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--left", type=int, default=-1)
    a = p.parse_args()
    f, w, m, old = case(a.context, a.left)
    plain = layer.make_workspace("cuda", context=a.context, left=a.left)
    k = layer.launch(f.x, f.stream, w, m, plain, rank=0, compile_only=True)
    plain.shape.admit("code45-plain", k, grid=768)
    size, _ = text_info(k)
    print(json.dumps(dict(event="plain-size", **size)), flush=True)
    (LOG / f"phase45-plain-{a.context}-{a.left}.hsaco").write_bytes(k.asm["hsaco"])
    if a.metadata:
        return
    if a.counter:
        layer.prepare_peers(plain, old["partial3"], old["partial5"], rank=0)
        layer.launch(f.x, f.stream, w, m, plain, rank=0)
        torch.cuda.synchronize()
        verify(f, plain, old)
        print("MONO_ONLY_COUNTER_PASS", flush=True)
        return
    cases = {}
    controls = {}
    resources = {}
    for label in ("no_read", "code_read"):
        ws = layer.make_workspace("cuda", context=a.context, left=a.left)
        control = torch.zeros((18,), device="cuda", dtype=torch.int64)
        kw = dict(rank=0, code_warm=control)
        compiled = layer.launch(f.x, f.stream, w, m, ws, compile_only=True, **kw)
        ws.shape.admit("code45-probe", compiled, grid=768)
        info, body = text_info(compiled, marker=True)
        assert 0 < info["words"] <= 1048576
        control[0] = info["pc_offset"]
        control[1] = info["words"] if label == "code_read" else 0
        prepare = functools.partial(
            layer.prepare_peers, ws, old["partial3"], old["partial5"], rank=0
        )
        call = functools.partial(layer.launch, f.x, f.stream, w, m, ws, **kw)
        check = functools.partial(verify, f, ws, old)
        cases[label] = (prepare, call, check)
        controls[label] = control
        resources[label] = dict(
            **info,
            vgpr=compiled.n_regs,
            spills=compiled.n_spills,
            lds=compiled.metadata.shared,
        )
        (LOG / f"phase45-{label}.hsaco").write_bytes(compiled.asm["hsaco"])
        if label == "code_read":
            words = struct.unpack("<" + "I" * info["words"], body[: info["words"] * 4])
            partials = [0, 0]
            for i, value in enumerate(words):
                partials[i // 512 % 2] ^= value
            expected = [partials[t // 6] for t in range(12)] + [0] * 4
    assert resources["no_read"] == resources["code_read"], (
        "A/B must compile exactly the same binary"
    )
    result, _ = measure(cases)
    observed = [int(v) & 0xFFFFFFFF for v in controls["code_read"][2:].tolist()]
    assert observed == expected, dict(observed=observed, expected=expected)
    result.update(
        plain=size,
        resources=resources,
        code_checksum=observed,
        code_read_over_control=result["code_read"]["median_us"]
        / result["no_read"]["median_us"],
    )
    (LOG / "phase45-code-warm-six-xcc.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("no_read", "code_read")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
