"""Exact GPU0-local stand-in setup vs frozen Gluon oracle, no peer memory."""

import json
from pathlib import Path
from unittest.mock import patch

import torch
from layer_fixture import require_gpu0
from tlx_ir import dump
from triton.compiler import CompiledKernel

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    standin as old,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_peers as new,
)

ROOT = Path("/home/bangtliu/coord/logs")
active = [None]
seen = set()
original = CompiledKernel.launch_metadata


def metadata(kernel, *aa, **kw):
    out = original(kernel, *aa, **kw)
    if active[0] and active[0] not in seen:
        case, side = active[0]
        dump(kernel, "T5-peer-prepare/" + case, side)
        seen.add(active[0])
    return out


def main():
    require_gpu0()
    rows = []
    for width in (4096, 8192):
        ref = (
            torch.arange(4 * 8 * width, device="cuda", dtype=torch.float32)
            .reshape(4, 8, width)
            .to(torch.bfloat16)
        )
        for ctas in (8, 768):
            for epoch in (0, 7):
                for rank in range(4):
                    peers = []
                    epochs = torch.full(
                        (ctas, 32), epoch, device="cuda", dtype=torch.int32
                    )
                    for module, side in ((old, "gluon"), (new, "tlx")):
                        p = module.make_standins("cuda", width=width, ctas=ctas)
                        peers.append(p)
                        active[0] = (f"{width}-{ctas}", side)
                        module.prepare(p, ref, epochs, rank=rank)
                    torch.cuda.synchronize()
                    active[0] = None
                    a, b = peers[0].scratch, peers[1].scratch
                    if not torch.equal(a, b):
                        at = int(torch.nonzero(a != b)[0])
                        torch.save(
                            dict(
                                actual=b.cpu(),
                                expected=a.cpu(),
                                width=width,
                                ctas=ctas,
                                epoch=epoch,
                                rank=rank,
                            ),
                            ROOT / "phaseT5-peer-prepare-failure.pt",
                        )
                        raise AssertionError(
                            dict(
                                tensor="peer scratch",
                                location=at,
                                rank=rank,
                                actual=int(b[at]),
                                expected=int(a[at]),
                            )
                        )
                    rows.append(
                        dict(width=width, ctas=ctas, epoch=epoch, rank=rank, exact=True)
                    )
    (ROOT / "phaseT5-peer-prepare.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("PASS TLX peer prepare", len(rows), "cases", flush=True)


if __name__ == "__main__":
    with patch.object(CompiledKernel, "launch_metadata", metadata):
        main()
