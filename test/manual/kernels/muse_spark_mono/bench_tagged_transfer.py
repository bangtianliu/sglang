"""GPU0 transfer controls: matched b64 relaxed stores and volatile pair loads.

No concurrent reader/writer, no spinning, no weight math. CTA-scope publication
is a diagnostic control only, never a candidate protocol. Three rotating buffers.
"""

import json
from pathlib import Path

import torch
import triton
import triton.language as tl
from layer_fixture import require_gpu0
from paired45 import distribution


@triton.jit
def store(Src, Dst, N: tl.constexpr, SCOPE: tl.constexpr, B: tl.constexpr = 512):
    i = tl.program_id(0) * B + tl.arange(0, B)
    word = tl.load(Src + i, i < N, 0).to(tl.uint64)
    pair = (1 << 32) | word
    tl.atomic_store(
        Dst + i, pair, mask=(i < N) & (i < 11 * 2048), sem="relaxed", scope=SCOPE
    )


@triton.jit
def load(Src, Out, N: tl.constexpr, DEV: tl.constexpr, B: tl.constexpr = 512):
    i = tl.program_id(0) * B + tl.arange(0, B)
    pair = tl.load(
        Src + i, i < N, 0, cache_modifier=".cg" if DEV else "", volatile=True
    )
    tl.store(Out + tl.program_id(0), tl.sum(pair, 0))


def main():
    require_gpu0()
    n = 131072
    sources = [
        torch.arange(n, device="cuda", dtype=torch.int64).to(torch.uint32)
        for _ in range(3)
    ]
    destinations = [torch.zeros(n, device="cuda", dtype=torch.uint64) for _ in range(3)]
    sums = [torch.zeros(n // 512, device="cuda", dtype=torch.uint64) for _ in range(3)]
    graphs = {}
    kernels = {}
    for name in ("store-cta", "store-gpu", "load-default", "load-dev"):

        def call(name=name):
            for x, y, z in zip(sources, destinations, sums):
                if name.startswith("store"):
                    k = store[(n // 512,)](x, y, n, name.split("-")[1], num_warps=4)
                else:
                    k = load[(n // 512,)](y, z, n, name.endswith("dev"), num_warps=4)
                kernels[name] = k

        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        a = torch.cuda.Event(enable_timing=True, external=True)
        b = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(graph):
            a.record()
            call()
            b.record()
        graphs[name] = (graph, a, b)
    raw = []
    for repeat in range(120):
        names = list(graphs)
        if repeat % 2:
            names.reverse()
        for name in names:
            graph, a, b = graphs[name]
            graph.replay()
            torch.cuda.synchronize()
            if repeat >= 20:
                raw.append(
                    dict(
                        name=name,
                        order="AB" if repeat % 2 == 0 else "BA",
                        us=a.elapsed_time(b) * 1000 / 3,
                    )
                )
    expected = torch.zeros(n, dtype=torch.int64)
    expected[: 11 * 2048] = (1 << 32) | torch.arange(11 * 2048, dtype=torch.int64)
    for y, z in zip(destinations, sums):
        torch.testing.assert_close(y.cpu().to(torch.int64), expected, rtol=0, atol=0)
        torch.testing.assert_close(
            z.cpu().to(torch.int64), expected.reshape(-1, 512).sum(1), rtol=0, atol=0
        )
    result = dict(
        raw=raw,
        timing={
            name: distribution([r["us"] for r in raw if r["name"] == name])
            for name in graphs
        },
        valid_store_bytes=11 * 2048 * 8,
        read_bytes=n * 8,
        weight_cold_claim=False,
    )
    root = Path("/home/bangtliu/coord/logs")
    for name, k in kernels.items():
        for ext in ("ttgir", "llir", "amdgcn"):
            (root / f"phaseT13-transfer-{name}.{ext}").write_text(k.asm[ext])
    (root / "phaseT13-transfer-controls.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result["timing"]), flush=True)


if __name__ == "__main__":
    main()
