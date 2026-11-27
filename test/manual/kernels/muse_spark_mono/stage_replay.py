"""Device mismatch records for delayed, changing-input serial stage replays."""

import json
from pathlib import Path

import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def verify_pair(
    X,
    Y,
    A,
    B,
    Status,
    Errors,
    STEP: gl.constexpr,
    N: gl.constexpr,
    CTAS: gl.constexpr,
    CHUNKS: gl.constexpr,
):
    i = gl.program_id(0) * 256 + gl.arange(
        0, 256, layout=gl.BlockedLayout([1], [32], [4], [0])
    )
    mismatch = (gl.load(X + i, i < N, other=0) != gl.load(A + i, i < N, other=0)) | (
        gl.load(Y + i, i < N, other=0) != gl.load(B + i, i < N, other=0)
    )
    invalid = gl.load(Status + i % CTAS) != 1
    gl.store(
        Errors + STEP * CHUNKS + gl.program_id(0),
        gl.sum((mismatch | invalid).to(gl.int32), 0),
    )


def exercise(call, x, state, seed, reference, name):
    original = x.clone()
    baseline = []
    for sign in (1, -1):
        x.copy_(original * sign)
        out, stream = call()
        torch.cuda.synchronize()
        assert state.status.eq(1).all()
        expected = reference()
        for got, want in zip((out, stream), expected):
            delta = (got.double() - want.double()).abs().max()
            assert delta / want.double().abs().max() < 0.015
        baseline.append((out.clone(), stream.clone()))
    assert not torch.equal(baseline[0][0], baseline[1][0])
    chunks = (out.numel() + 255) // 256
    records = torch.full((200, chunks), -1, device=x.device, dtype=torch.int32)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for step in range(200):
            x.copy_(original if step % 2 == 0 else -original)
            seed.fill_(20261117 + step)
            out, stream = call()
            a, b = baseline[step % 2]
            verify_pair[(chunks,)](
                out,
                stream,
                a,
                b,
                state.status,
                records,
                step,
                out.numel(),
                state.status.numel(),
                chunks,
                num_warps=4,
            )
    graph.replay()
    torch.cuda.synchronize()
    assert records.eq(0).all(), records.sum().item()
    assert state.status.eq(1).all()
    Path(f"/home/bangtliu/coord/logs/phase3d-{name}-replay-records.json").write_text(
        json.dumps(records.cpu().tolist()) + "\n"
    )
    result = dict(
        event="replay", stage=name, launches=200, mismatches=0, all_ready=True
    )
    print(json.dumps(result), flush=True)
    return result
