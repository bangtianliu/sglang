"""Changing-input S1 replays with seeded CTA skew and device mismatch records."""

import argparse
import json
from pathlib import Path

import torch
from layer_fixture import EPS, Q_SCALE, build_layer, require_gpu0
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import sync
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.qkvg import (
    _advance_tag,
    make_workspace,
    qkvg,
    s1_stage,
)


@g.jit
def epoch_probe(State, Tags, BAD: gl.constexpr):
    lane = gl.arange(0, 512, layout=gl.BlockedLayout([1], [32], [16], [0]))
    gl.inline_asm_elementwise(
        "v_readfirstlane_b32 $0, $1\n1:\ns_sleep 1\ns_sub_u32 $0, $0, 1\ns_cbranch_scc1 1b",
        "=&s,v,~{scc},~{memory}",
        [(lane // 32) * 32 + 1],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    if BAD:
        tag = gl.load(State) + 1
        gl.store(State, tag)
    else:
        tag = _advance_tag(State, 0)
    gl.store(Tags + lane, gl.full_like(lane, tag))


def check_epoch_counter():
    state = torch.zeros(32, device="cuda:0", dtype=torch.int32)
    tags = torch.empty(512, device="cuda:0", dtype=torch.int32)
    epoch_probe[(1,)](state, tags, True, num_warps=16)
    old_tags = tags.unique().tolist()
    old_count = int(state[0])
    state.zero_()
    epoch_probe[(1,)](state, tags, False, num_warps=16)
    assert tags.eq(1).all() and int(state[0]) == 1
    report = dict(
        event="epoch_interleaving",
        old_tags=old_tags,
        old_counter=old_count,
        new_tags=tags.unique().tolist(),
        new_counter=int(state[0]),
    )
    print(json.dumps(report), flush=True)
    Path("/home/bangtliu/coord/logs/phase3b.1-epoch-interleaving.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


@g.jit
def delayed(
    X,
    W,
    P,
    State,
    POS,
    COS,
    LOC,
    MAP,
    KC,
    VC,
    Q,
    K,
    V,
    G,
    KS,
    VS,
    Seed,
    ROPE: gl.constexpr,
    HAS_MAP: gl.constexpr,
    SKIP: gl.constexpr = -1,
):
    pid = gl.program_id(0)
    if pid != SKIP:
        delay = ((pid * 1103515245 + Seed) ^ ((pid + Seed) >> 3)) & 31
        for _ in range(delay):
            sync.backoff()
        s1_stage(
            pid,
            gl.num_programs(0),
            X,
            W,
            P,
            State,
            POS,
            COS,
            LOC,
            MAP,
            KC,
            VC,
            Q,
            K,
            V,
            G,
            KS,
            VS,
            1e-5,
            5.4730064863875,
            ROPE,
            HAS_MAP,
            True,
        )


@g.jit
def verify(
    Q, K, V, G, KC, VC, BQ, BK, BV, BG, BKC, BVC, Slots, Record, Sign: gl.constexpr
):
    which = gl.program_id(0)
    i = gl.arange(0, 16384, layout=gl.BlockedLayout([1], [32], [8], [0]))
    valid = gl.full((16384,), True, gl.int1, gl.BlockedLayout([1], [32], [8], [0]))
    if which == 0:
        a = gl.load(Q + i).to(gl.float32)
        b = gl.load(BQ + i).to(gl.float32)
    elif which == 1:
        valid = i < 2048
        a = gl.load(K + i, valid, other=0).to(gl.float32)
        b = gl.load(BK + i, valid, other=0).to(gl.float32)
    elif which == 2:
        valid = i < 2048
        a = gl.load(V + i, valid, other=0).to(gl.float32)
        b = gl.load(BV + i, valid, other=0).to(gl.float32)
    elif which == 3:
        a = gl.load(G + i).to(gl.float32)
        b = gl.load(BG + i).to(gl.float32)
    else:
        valid = i < 2048
        slot = gl.load(Slots + i // 256, valid, other=0)
        if which == 4:
            a = gl.load(KC + slot * 256 + i % 256, valid, other=0.0).to(gl.float32)
            b = gl.load(BKC + i, valid, other=0.0).to(gl.float32)
        else:
            a = gl.load(VC + slot * 256 + i % 256, valid, other=0.0).to(gl.float32)
            b = gl.load(BVC + i, valid, other=0.0).to(gl.float32)
    bad = gl.sum((valid & (a != b * Sign)).to(gl.int32), 0)
    gl.store(Record + which, bad)


def run(rope):
    f = build_layer(seed=20261117)
    workspace = make_workspace(f.x.device)
    q, k, v, gate = qkvg(
        f.x,
        f.weights["qkvg"],
        f.positions,
        f.cos_sin if rope else None,
        f.cache_k,
        f.cache_v,
        f.locations,
        f.mapping if rope else None,
        EPS,
        Q_SCALE,
        f.k_scale,
        f.v_scale,
        workspace=workspace,
    )
    baseline = [t.clone() for t in (q, k, v, gate)]
    slots = (f.mapping[f.locations] if rope else f.locations).clone()
    caches = [
        t.view(torch.uint8).reshape(-1, 256)[slots].clone().view(torch.float8_e4m3fn)
        for t in (f.cache_k, f.cache_v)
    ]
    records = torch.zeros((1000, 32), device=f.x.device, dtype=torch.int32)
    args = (
        f.x,
        f.weights["qkvg"],
        workspace.projection,
        workspace.state,
        f.positions,
        f.cos_sin if rope else None,
        f.locations,
        f.mapping if rope else None,
        f.cache_k,
        f.cache_v,
        q,
        k,
        v,
        gate,
        f.k_scale,
        f.v_scale,
    )
    for seed in (16, 17):
        delayed[(192,)](*args, seed, ROPE=rope, HAS_MAP=rope, num_warps=16)
    for sign in (-1, 1):
        verify.warmup(
            q,
            k,
            v,
            gate,
            f.cache_k,
            f.cache_v,
            *baseline,
            *caches,
            slots,
            records[0],
            sign,
            grid=(6,),
            num_warps=8,
        )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(1000):
            f.x.neg_()
            delayed[(192,)](*args, 256 + i, ROPE=rope, HAS_MAP=rope, num_warps=16)
            verify[(6,)](
                q,
                k,
                v,
                gate,
                f.cache_k,
                f.cache_v,
                *baseline,
                *caches,
                slots,
                records[i],
                -1 if i % 2 == 0 else 1,
                num_warps=8,
            )
    graph.replay()
    torch.cuda.synchronize(0)
    mismatches = int(records.sum().item())
    epochs = workspace.state[: 72 * 32 : 32]
    report = dict(
        rope=rope,
        launches=1000,
        seed=20261117,
        mismatches=mismatches,
        epoch_min=int(epochs.min()),
        epoch_max=int(epochs.max()),
        all_ready=bool(workspace.state[-72:].all()),
    )
    print(json.dumps(report), flush=True)
    Path(f"/home/bangtliu/coord/logs/phase3b.1-replay-rope{rope}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if mismatches:
        print(
            "first mismatch records", torch.nonzero(records)[:16].tolist(), flush=True
        )
    assert mismatches == 0 and report["all_ready"]
    assert report["epoch_min"] == report["epoch_max"] == 1003
    delayed[(192,)](*args, 17, ROPE=rope, HAS_MAP=rope, SKIP=1, num_warps=16)
    ready = workspace.state[-72:].cpu()
    assert int(ready[0]) == 0 and bool(ready[1:].all())
    assert int(workspace.state[144 * 32]) == 1003
    print(
        json.dumps(
            dict(
                event="missing_producer",
                rope=rope,
                stale_flag=1003,
                requested_tag=int(workspace.state[0]),
                failed_head=0,
                bounded=True,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-references", action="store_true")
    if parser.parse_args().saved_references:
        from reference_archive import fixture as build_layer
    require_gpu0()
    check_epoch_counter()
    run(False)
    run(True)
