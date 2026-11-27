"""Three-copy operand-tiled projection trial, exact bits before measurement."""

import argparse
import json
from pathlib import Path

import torch
from ci410 import block_verdict
from diagnose_guard48 import balanced
from layer_fixture import require_gpu0

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    register_projection,
    wide_projection,
)


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("s1", "s3", "s6"), default="s6")
    a = p.parse_args()
    N, K, SK, stage = {
        "s1": (4608, 8192, 8, 1),
        "s3": (8192, 2048, 4, 3),
        "s6": (8192, 4096, 4, 6),
    }[a.stage]
    torch.manual_seed(410)
    x = torch.randn((8, K), device="cuda", dtype=torch.bfloat16) * 0.1
    weights = [
        torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(3)
    ]
    packed = [
        w.reshape(N // 16, 16, K // 16, 2, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(N, K)
        for w in weights
    ]
    cases = {}
    epochs = {}
    outputs = {}
    resources = {}
    expected = []
    for label, fn, ww in [
        ("baseline", wide_projection.project, weights),
        ("candidate", register_projection.project, packed),
    ]:
        out = [torch.empty((8, N), device="cuda", dtype=torch.int32) for _ in range(3)]
        statuses = [
            torch.zeros((N // 16,), device="cuda", dtype=torch.int32) for _ in range(3)
        ]
        counter = torch.zeros((1,), device="cuda", dtype=torch.int32)
        epochs[label] = counter
        outputs[label] = out
        opts = dict(N=N, K=K, STAGE=stage, SK=SK, num_warps=2, enable_fp_fusion=False)
        k = fn.warmup(x, ww[0], out[0], statuses[0], None, **opts, grid=(N // 16,))
        r = kernel_occupancy("register-projection-" + label, k)
        assert r.active_ctas_per_cu >= 1
        resources[label] = dict(
            vgprs=k.n_regs,
            spills=k.n_spills,
            lds=k.metadata.shared,
            occupancy=r.active_ctas_per_cu,
        )
        Path(
            f"/home/bangtliu/coord/logs/phase410-register-{a.stage}-{label}.amdgcn"
        ).write_text(k.asm["amdgcn"])

        def call(fn=fn, ww=ww, out=out, statuses=statuses, counter=counter):
            for w, o, s in zip(ww, out, statuses):
                fn[(N // 16,)](x, w, o, s, None, **opts)
            counter.add_(1)

        def check(out=out, statuses=statuses, label=label):
            assert all(bool(s.eq(1).all()) for s in statuses)
            if label == "baseline" and not expected:
                expected.extend([o.clone() for o in out])
            for i, (o, e) in enumerate(zip(out, expected)):
                assert torch.equal(o, e), dict(
                    tensor="projection bits",
                    stage=a.stage,
                    rank=0,
                    copy=i,
                    location=(o != e).nonzero()[:1].tolist(),
                    max_error=float(
                        (o.view(torch.float32) - e.view(torch.float32)).abs().max()
                    ),
                )

        cases[label] = (lambda: None, call, check)
    d = balanced(cases, epochs)
    d.update(
        stage=a.stage,
        N=N,
        K=K,
        SK=SK,
        weight_copies=3,
        resources=resources,
        verdict=block_verdict(d),
        startup_extra_bytes=N * K * 2,
        TP4_extra_bytes_per_layer=dict(
            qkvg=72 * 2**20, o_proj=32 * 2**20, pooler=64 * 2**20, post=64 * 2**20
        ),
        scope="3 projections plus identical device counter increment; no activation/flag policy change; no adoption",
    )
    Path(f"/home/bangtliu/coord/logs/phase410-register-{a.stage}.json").write_text(
        json.dumps(d, indent=2) + "\n"
    )
    print(
        json.dumps(dict(timing=d["timing"], verdict=d["verdict"], resources=resources)),
        flush=True,
    )


if __name__ == "__main__":
    main()
