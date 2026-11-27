"""GPU0 cache-behaviour probe; deliberate warm reuse vs three rotating arrays.

This characterizes cache reuse, not a cold-weight layer adoption comparison.
Reported HIP L2 size is not assumed to include MALL/Infinity Cache.
"""

import json
from pathlib import Path

import torch
from bench_hbm import stream_read
from ci410 import block_verdict
from diagnose_guard48 import balanced
from layer_fixture import require_gpu0


def main():
    require_gpu0()
    rows = []
    for mib in (2, 4, 8, 16, 32, 64, 128, 256):
        n = mib * 2**20 // 4
        x = [torch.ones((n,), device="cuda", dtype=torch.float32) for _ in range(3)]
        y = [
            torch.empty((n // 4096,), device="cuda", dtype=torch.float32)
            for _ in range(3)
        ]
        cases = {}
        epochs = {}
        for label in ("rotating", "reuse"):
            epoch = torch.zeros((1,), device="cuda", dtype=torch.int32)
            epochs[label] = epoch

            def call(label=label, epoch=epoch):
                for i in range(3):
                    stream_read[(n // 4096,)](
                        x[i if label == "rotating" else 0], y[i], n, 4096, num_warps=4
                    )
                epoch.add_(1)

            def check():
                assert all(bool(t.eq(4096).all()) for t in y)

            cases[label] = (lambda: None, call, check)
        d = balanced(cases, epochs)
        d.update(mib=mib, read_passes=3, verdict=block_verdict(d, "rotating", "reuse"))
        rows.append(d)
        print(
            mib,
            {k: round(v["median_us"] / 3, 3) for k, v in d["timing"].items()},
            flush=True,
        )
    d = dict(
        L2_reported_bytes=torch.cuda.get_device_properties(0).L2_cache_size,
        MALL_capacity="not exposed by HIP properties/sysfs on this node; no size inferred from one plateau",
        scope="three read dispatches plus equal device-counter bookkeeping; warm-vs-rotating is intentional",
        rows=rows,
    )
    Path("/home/bangtliu/coord/logs/phase410-cache-probe.json").write_text(
        json.dumps(d, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
