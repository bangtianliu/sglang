"""T0 S6-shape tl/uTLX TDM-ring projection; no production stage is adopted."""

import json
from pathlib import Path

import torch
import triton
import triton.language as tl
import utlx_plugin as tlx
from ci410 import block_verdict
from diagnose_guard48 import balanced
from layer_fixture import require_gpu0
from tlx_ir import dump
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout, gfx1250

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import wide_projection

LOG = Path("/home/bangtliu/coord/logs")


@triton.jit
def project(X, W, P, Status):
    tile = tl.program_id(0)
    sh: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(512, 8)], [16, 512], [1, 0]
    )
    sa = tlx.local_alloc((16, 512), tl.bfloat16, 4, layout=sh)
    sb = tlx.local_alloc((16, 512), tl.bfloat16, 4, layout=sh)
    ad = tlx.make_tensor_descriptor(None, X, [8, 4096], [4096, 1], [16, 512])
    bd = tlx.make_tensor_descriptor(
        None, W + tile * 16 * 4096, [16, 4096], [4096, 1], [16, 512]
    )
    for step in tl.static_range(3):
        tlx.tdm_async_load(ad, tlx.local_view(sa, step), [0, step * 512])
        tlx.tdm_async_load(bd, tlx.local_view(sb, step), [0, step * 512])
    parts = ()
    acc = tl.zeros((16, 16), tl.float32)
    for step in tl.static_range(8):
        if step + 3 < 8:
            tlx.tdm_async_load(
                ad, tlx.local_view(sa, (step + 3) % 4), [0, (step + 3) * 512]
            )
            tlx.tdm_async_load(
                bd, tlx.local_view(sb, (step + 3) % 4), [0, (step + 3) * 512]
            )
        tlx.tdm_async_wait(min(3, 8 - step - 1) * 2)
        aa = tlx.local_load(tlx.local_view(sa, step % 4))
        bb = tlx.local_load(tlx.local_trans(tlx.local_view(sb, step % 4)))
        # All waves have consumed this bank before a later TDM overwrite.
        tlx.workgroup_barrier()
        if step % 2 == 0:
            acc = tl.zeros((16, 16), tl.float32)
        acc = tl.dot(aa, bb, acc)
        if step % 2 == 1:
            parts += (acc,)
    tlx.tdm_async_wait(0)
    value = (
        ((parts[0] + parts[2]) + (parts[1] + parts[3])).to(tl.bfloat16).to(tl.float32)
    )
    r = tl.arange(0, 16)
    n = tile * 16 + tl.arange(0, 16)
    tl.store(
        P + r[:, None] * 8192 + n[None, :],
        value.to(tl.int32, bitcast=True),
        r[:, None] < 8,
    )
    tl.store(Status + tile, 1)


@triton.jit
def small_dot(A, B, C):
    m = tl.arange(0, 16)
    n = tl.arange(0, 64)
    k = tl.arange(0, 128)
    a = tl.load(A + m[:, None] * 128 + k[None, :])
    b = tl.load(B + k[:, None] * 64 + n[None, :])
    tl.store(C + m[:, None] * 64 + n[None, :], tl.dot(a, b))


@g.jit
def reference_dot(A, B, C):
    mma: gl.constexpr = AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[0, 1], [0, 2]],
        instr_shape=[16, 16, 32],
    )
    al: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([16, 1], [4, 8], [1, 4], [0, 1])
    r = gl.arange(0, 16, layout=gl.SliceLayout(1, al))
    ka = gl.arange(0, 128, layout=gl.SliceLayout(0, al))
    kb = gl.arange(0, 128, layout=gl.SliceLayout(1, bl))
    n = gl.arange(0, 64, layout=gl.SliceLayout(0, bl))
    a = gl.convert_layout(
        gl.load(A + r[:, None] * 128 + ka[None, :]), gl.DotOperandLayout(0, mma, 8)
    )
    b = gl.convert_layout(
        gl.load(B + kb[:, None] * 64 + n[None, :]), gl.DotOperandLayout(1, mma, 8)
    )
    value = gfx1250.wmma(a, b, gl.full((16, 64), 0, gl.float32, mma))
    rr = gl.arange(0, 16, layout=gl.SliceLayout(1, mma))
    nn = gl.arange(0, 64, layout=gl.SliceLayout(0, mma))
    gl.store(C + rr[:, None] * 64 + nn[None, :], value)


@triton.jit
def compare(A, B, Errors, Row):
    i = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    a = tl.load(A + i)
    b = tl.load(B + i)
    tl.store(Errors + Row * 64 + tl.program_id(0), tl.sum((a != b).to(tl.int32), 0))


def main():
    require_gpu0()
    torch.manual_seed(20261118)
    xs = [
        torch.randn((8, 4096), device="cuda", dtype=torch.bfloat16) * 0.1
        for _ in range(3)
    ]
    ws = [
        torch.randn((8192, 4096), device="cuda", dtype=torch.bfloat16) * 0.1
        for _ in range(3)
    ]
    assert len(set(w.data_ptr() for w in ws)) == 3
    outputs = {
        n: [torch.empty((8, 8192), device="cuda", dtype=torch.int32) for _ in range(3)]
        for n in ("gluon", "tlx")
    }
    statuses = {
        n: [torch.empty((512,), device="cuda", dtype=torch.int32) for _ in range(3)]
        for n in outputs
    }
    originals = [x.clone() for x in xs]

    def run(label, i):
        args = (xs[i], ws[i], outputs[label][i], statuses[label][i])
        if label == "gluon":
            return wide_projection.project[(512,)](
                *args,
                None,
                N=8192,
                K=4096,
                STAGE=6,
                num_warps=2,
                enable_fp_fusion=False,
            )
        return project[(512,)](*args, num_warps=2, num_stages=1, enable_fp_fusion=False)

    kg = wide_projection.project.warmup(
        xs[0],
        ws[0],
        outputs["gluon"][0],
        statuses["gluon"][0],
        None,
        N=8192,
        K=4096,
        STAGE=6,
        grid=(512,),
        num_warps=2,
        enable_fp_fusion=False,
    )
    kt = project.warmup(
        xs[0],
        ws[0],
        outputs["tlx"][0],
        statuses["tlx"][0],
        grid=(512,),
        num_warps=2,
        num_stages=1,
        enable_fp_fusion=False,
    )
    resources = {
        "gluon": dump(kg, "projection", "gluon"),
        "tlx": dump(kt, "projection", "tlx"),
    }
    assert resources["tlx"]["occupancy"] >= 1
    accuracy = []
    for i in range(3):
        run("gluon", i)
        run("tlx", i)
        torch.cuda.synchronize()
        a = outputs["tlx"][i].view(torch.float32)
        b = outputs["gluon"][i].view(torch.float32)
        wrong = int(outputs["tlx"][i].ne(outputs["gluon"][i]).sum())
        oracle = xs[i].double() @ ws[i].double().T
        error = float((a.double() - oracle).abs().max() / oracle.abs().max())
        assert error < 0.01, dict(
            tensor="projection",
            rank=0,
            copy=i,
            relative_linf=error,
            location=(a.double() - oracle).abs().argmax().item(),
        )
        accuracy.append(
            dict(
                copy=i,
                bit_mismatches=wrong,
                fp64_relative_linf=error,
                max_abs_vs_gluon=float((a - b).abs().max()),
            )
        )
    exact = all(x["bit_mismatches"] == 0 for x in accuracy)
    expected_outputs = {
        label: [o.clone() for o in out] for label, out in outputs.items()
    }
    errors = torch.full((64, 64), -1, device="cuda", dtype=torch.int32)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(64):
            copy = i % 3
            xs[copy].copy_(originals[copy] if i % 2 == 0 else -originals[copy])
            run("gluon", copy)
            run("tlx", copy)
            compare[(64,)](
                outputs["gluon"][copy], outputs["tlx"][copy], errors, i, num_warps=4
            )
    graph.replay()
    torch.cuda.synchronize()
    replay_wrong = int(errors.sum())
    if exact:
        assert replay_wrong == 0, dict(
            tensor="projection replay bits",
            rank=0,
            location=errors.ne(0).nonzero()[:1].tolist(),
            mismatches=replay_wrong,
        )
    for x, old in zip(xs, originals):
        x.copy_(old)
    cases = {}
    epochs = {}
    for label in ("gluon", "tlx"):
        epoch = torch.zeros((1,), device="cuda", dtype=torch.int32)
        epochs[label] = epoch

        def call(label=label, epoch=epoch):
            for i in range(3):
                run(label, i)
            epoch.add_(1)

        def check(label=label):
            assert all(bool(s.eq(1).all()) for s in statuses[label])
            assert all(
                torch.equal(a, b)
                for a, b in zip(outputs[label], expected_outputs[label])
            ), dict(tensor="projection timing replay", side=label, rank=0)

        cases[label] = (lambda: None, call, check)
    d = balanced(cases, epochs)
    a = (torch.arange(16 * 128, device="cuda").reshape(16, 128) % 7 - 3).to(
        torch.bfloat16
    ) / 4
    b = (torch.arange(128 * 64, device="cuda").reshape(128, 64) % 5 - 2).to(
        torch.bfloat16
    ) / 8
    dot_records = {}
    for label, fn in [("gluon", reference_dot), ("tlx", small_dot)]:
        out = torch.empty((16, 64), device="cuda", dtype=torch.float32)
        k = fn.warmup(a, b, out, grid=(1,), num_warps=4, enable_fp_fusion=False)
        dot_records[label] = dump(k, "expert-dot", "%s" % label)
        fn[(1,)](a, b, out, num_warps=4, enable_fp_fusion=False)
        torch.cuda.synchronize()
        assert torch.equal(out.double(), a.double() @ b.double()), dict(
            tensor="M16N64K128 dot", side=label, rank=0
        )
        assert "v_wmma_" in k.asm["amdgcn"]
    d.update(
        resources=resources,
        accuracy=accuracy,
        exact=exact,
        replay_launches=64,
        replay_bit_mismatches=replay_wrong,
        weight_copies=3,
        scope="three projections plus identical counter increment; per-copy table divide by3",
        verdict=block_verdict(d, "gluon", "tlx"),
        expert_dot=dot_records,
    )
    (LOG / "phaseT0-projection.json").write_text(json.dumps(d, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                resources=resources,
                accuracy=accuracy,
                exact=exact,
                timing=d["timing"],
                verdict=d["verdict"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
