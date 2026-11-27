"""In-kernel S2 loop ablations; synthetic controls are diagnostics, not latency sums."""

import json
import re
from pathlib import Path

import numpy as np
import torch
from bench_s2 import partial_only
from layer_fixture import build_layer, require_gpu0
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout, gfx1250

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import sync
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.attention import (
    _cached_pages,
    _tile,
)


@g.jit
def touch(x):
    # An opaque side effect keeps every converted operand live without memory I/O.
    return gl.inline_asm_elementwise(
        "v_mov_b32 $0, $1",
        "=v,v",
        [x.to(gl.float32)],
        dtype=gl.float32,
        is_pure=False,
        pack=1,
    )


@g.jit
def ablate(
    Q,
    K,
    V,
    PAGES,
    LENGTH,
    KS,
    VS,
    OUT,
    LSE,
    TIMES,
    MODE: gl.constexpr,
    BN: gl.constexpr = 64,
    SPLITS: gl.constexpr = 192,
    MAX_PAGES: gl.constexpr = 64,
):
    split = gl.program_id(0)
    mma: gl.constexpr = AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[0, 1, 0], [1, 0, 0], [2, 0, 0]],
        instr_shape=[16, 16, 32],
        rank=3,
    )
    ql: gl.constexpr = gl.BlockedLayout([1, 1, 8], [1, 4, 8], [4, 2, 1], [2, 1, 0])
    h = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, ql)))
    r = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(2, ql)))
    d = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(1, ql)))
    q = gl.load(
        Q
        + (r[None, :, None] // 8) * 2048
        + (h[:, None, None] * 8 + r[None, :, None] % 8) * 64
        + d[None, None, :]
    )
    q = gl.convert_layout(q, gl.DotOperandLayout(0, mma, 8))
    length = gl.load(LENGTH)
    tiles = gl.cdiv(length, BN)
    start = split * tiles // SPLITS * BN
    end = gl.minimum(length, (split + 1) * tiles // SPLITS * BN)
    pages = _cached_pages(PAGES, None, start, end, MAX_PAGES, False)
    raw: gl.constexpr = gl.BlockedLayout([1, 16], [2, 16], [8, 1], [1, 0])
    tt = gl.arange(0, BN, layout=gl.SliceLayout(1, raw))
    dd = gl.arange(0, 256, layout=gl.SliceLayout(0, raw))
    seed = ((tt[:, None] * 3 + dd[None, :] * 5) % 32 + 48).to(gl.uint8)
    raw_k = seed.to(gl.float8e4nv, bitcast=True)
    raw_v = (seed ^ 1).to(gl.float8e4nv, bitcast=True)
    if MODE == "full" or MODE == "loads":
        k0, v0, valid0 = _tile(
            K, V, None, pages, start, start, end, BN, False, False, MAX_PAGES
        )
        k1, v1, valid1 = _tile(
            K, V, None, pages, start + BN, start, end, BN, False, False, MAX_PAGES
        )
    else:
        k0 = raw_k
        v0 = raw_v
        valid0 = tt >= 0
    if MODE == "math" or MODE == "touch":
        ks = gl.convert_layout(
            gl.reshape(raw_k.to(gl.bfloat16), (BN, 4, 64)).permute((1, 2, 0)),
            gl.DotOperandLayout(1, mma, 8),
        )
        vs = gl.convert_layout(
            gl.reshape(raw_v.to(gl.bfloat16), (BN, 4, 64)).permute((1, 0, 2)),
            gl.DotOperandLayout(1, mma, 8),
        )
        ps = gl.convert_layout(
            gl.full((4, 64, BN), 0.25, gl.float32, mma).to(gl.bfloat16),
            gl.DotOperandLayout(0, mma, 8),
        )
    acc = gl.full((4, 64, 64), 0, gl.float32, mma)
    red: gl.constexpr = gl.SliceLayout(2, mma)
    rr = gl.arange(0, 64, layout=gl.SliceLayout(0, red))
    cc = gl.arange(0, BN, layout=gl.SliceLayout(0, gl.SliceLayout(1, mma)))
    maximum = gl.full((4, 64), float("-inf"), gl.float32, red)
    denom = gl.full((4, 64), 0, gl.float32, red)
    scale = 0.125 * 1.4426950408889634 * gl.load(KS)
    fold = gl.full((BN, 256), 0, gl.uint8, raw)
    gl.barrier()
    rt0 = sync.clock()
    c0 = sync.shader_cycles()
    for offset in range(start, end, BN):
        if MODE == "full" or MODE == "loads":
            current_k = k0
            current_v = v0
            valid = valid0
            k0, v0, valid0 = k1, v1, valid1
            k1, v1, valid1 = _tile(
                K,
                V,
                None,
                pages,
                offset + 2 * BN,
                start,
                end,
                BN,
                False,
                False,
                MAX_PAGES,
            )
        else:
            # Toggle a mantissa bit so conversion and QK cannot be hoisted.
            bit = ((offset // BN) & 1).to(gl.uint8)
            current_k = (seed ^ bit).to(gl.float8e4nv, bitcast=True)
            current_v = (seed ^ (bit ^ 1)).to(gl.float8e4nv, bitcast=True)
            valid = tt >= 0
        if MODE == "loads":
            fold = (
                fold
                ^ current_k.to(gl.uint8, bitcast=True)
                ^ current_v.to(gl.uint8, bitcast=True)
            )
        elif MODE == "touch":
            touch(ks)
            touch(vs)
            touch(ps)
        elif MODE == "empty":
            gl.inline_asm_elementwise(
                "s_nop 0\nv_mov_b32 $0, $1",
                "=v,v",
                [offset],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            )
        else:
            if MODE == "math":
                kk = ks + (((offset // BN) & 1).to(gl.bfloat16) * 0.03125)
                vv = vs
            else:
                kk = gl.convert_layout(
                    gl.reshape(current_k.to(gl.bfloat16), (BN, 4, 64)).permute(
                        (1, 2, 0)
                    ),
                    gl.DotOperandLayout(1, mma, 8),
                )
                vv = gl.convert_layout(
                    gl.reshape(
                        gl.where(valid[:, None], current_v.to(gl.bfloat16), 0.0),
                        (BN, 4, 64),
                    ).permute((1, 0, 2)),
                    gl.DotOperandLayout(1, mma, 8),
                )
            if MODE == "conversion":
                pp = gl.convert_layout(
                    (
                        gl.full((4, 64, BN), 0.25, gl.float32, mma)
                        + ((offset // BN) & 1) * 0.03125
                    ).to(gl.bfloat16),
                    gl.DotOperandLayout(0, mma, 8),
                )
                touch(kk)
                touch(vv)
                touch(pp)
            else:
                score = (
                    gfx1250.wmma(q, kk, gl.full((4, 64, BN), 0, gl.float32, mma))
                    * scale
                )
                valid = gl.convert_layout(
                    valid, gl.SliceLayout(0, gl.SliceLayout(1, mma))
                )
                pos = offset + cc
                absolute = length - 8 + rr // 8
                score = gl.where(
                    (pos[None, None, :] <= absolute[None, :, None])
                    & valid[None, None, :],
                    score,
                    float("-inf"),
                )
                new_max = gl.maximum(maximum, gl.max(score, 2))
                safe = gl.where(new_max == float("-inf"), 0.0, new_max)
                prob = gl.exp2(score - safe[:, :, None])
                alpha = gl.exp2(maximum - safe)
                denom = denom * alpha + gl.sum(prob, 2)
                acc *= alpha[:, :, None]
                acc = gfx1250.wmma(
                    gl.convert_layout(
                        prob.to(gl.bfloat16), gl.DotOperandLayout(0, mma, 8)
                    ),
                    vv,
                    acc,
                )
                maximum = new_max
    gl.barrier()
    c1 = sync.shader_cycles()
    rt1 = sync.clock()
    gl.store(TIMES + split * 8, rt0)
    gl.store(TIMES + split * 8 + 1, rt1)
    gl.store(TIMES + split * 8 + 2, c0)
    gl.store(TIMES + split * 8 + 3, c1)
    gl.store(TIMES + split * 8 + 4, gl.cdiv(end - start, BN))
    if MODE == "loads":
        checksum = gl.sum(gl.sum(fold.to(gl.int32), 1), 0)
        gl.store(TIMES + split * 8 + 5, checksum)
    if MODE == "full":
        value = acc * (gl.load(VS) / gl.where(denom > 0, denom, 1.0))[:, :, None]
        lse = gl.where(denom > 0, maximum + gl.log2(denom), float("-inf"))
        om: gl.constexpr = mma
        hh = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, om)))
        row = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(2, om)))
        dim = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(1, om)))
        gl.store(
            OUT
            + split * 16384
            + hh[:, None, None] * 4096
            + row[None, :, None] * 64
            + dim[None, None, :],
            value,
        )
        gl.store(LSE + split * 256 + hh[:, None] * 64 + row[None, :], lse)
    elif MODE == "math":
        touch(acc)


def main():
    require_gpu0()
    log = Path("/home/bangtliu/coord/logs")
    f = build_layer(context=131072, components=())
    q = torch.randn((8, 32, 64), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((192, 4, 64, 64), device="cuda")
    lse = torch.empty((192, 4, 64), device="cuda")
    times = torch.zeros((192, 8), device="cuda", dtype=torch.uint64)
    records = []
    for mode in ("empty", "touch", "loads", "math", "conversion", "full"):

        def run():
            return ablate[(192,)](
                q,
                f.cache_k,
                f.cache_v,
                f.page_table,
                f.seq_lens,
                f.k_scale,
                f.v_scale,
                out,
                lse,
                times,
                mode,
                num_warps=8,
                enable_fp_fusion=False,
            )

        kernel = run()
        samples = []
        for _ in range(20):
            run()
            torch.cuda.synchronize()
            samples.append(times.cpu().numpy().copy())
        a = np.stack(samples)
        iters = a[:, :, 4].astype(np.float64)
        cycles = a[:, :, 3] - a[:, :, 2]
        rt = a[:, :, 1] - a[:, :, 0]
        assert (iters > 0).all() and (cycles > 0).all()
        asm = kernel.asm["amdgcn"]
        prefix = log / f"phase3c-S2-opt2-ablation-{mode}"
        prefix.with_suffix(".amdgcn").write_text(asm)
        prefix.with_suffix(".ttgir").write_text(kernel.asm.get("ttgir", ""))
        prefix.with_suffix(".raw.json").write_text(json.dumps(a.tolist()) + "\n")
        per = cycles / iters
        record = dict(
            mode=mode,
            cycles_per_iteration=dict(
                median=float(np.median(per)), p99=float(np.quantile(per, 0.99))
            ),
            realtime_ticks_per_iteration=float(np.median(rt / iters)),
            iterations=sorted(set(iters.flatten().tolist())),
            shared=kernel.metadata.shared,
            private_bytes=int(
                re.search(r"amdhsa_private_segment_fixed_size (\d+)", asm)[1]
            ),
            vgprs=int(re.search(r"amdhsa_next_free_vgpr (\d+)", asm)[1]),
            asm_counts={
                name: len(re.findall(r"\b" + name, asm))
                for name in (
                    "global_load_b128",
                    "ds_store",
                    "ds_load",
                    "s_wait_loadcnt",
                    "s_barrier",
                    "v_wmma",
                    "v_cvt_scale_pk8_bf16_fp8",
                )
            },
        )
        if mode == "full":
            expected = torch.empty_like(out)
            expected_lse = torch.empty_like(lse)
            partial_only[(192,)](
                q,
                f.cache_k,
                f.cache_v,
                f.page_table,
                f.seq_lens,
                f.k_scale,
                f.v_scale,
                expected,
                expected_lse,
                192,
                64,
                64,
                2,
                num_warps=8,
                enable_fp_fusion=False,
            )
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            torch.testing.assert_close(lse, expected_lse, rtol=0, atol=0)
            record["production_partial_bitwise"] = True
        records.append(record)
        print(json.dumps(record), flush=True)
        (log / "phase3c-S2-opt2-ablations.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
