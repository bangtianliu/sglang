"""Once-only authorized S3 packed boundary, guarded payload addresses and stage logs."""

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import bounds49_device as checked
import torch
from check_chained47 import archive_check
from layer_fixture import require_gpu0
from mono_case45 import case
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    standin,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    chained,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    independent_attn_tail as s3,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    wide_projection,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.qkvg import (
    _advance_tag,
)

LOG = Path("/home/bangtliu/coord/logs")
ATTEMPT = Path("/home/bangtliu/coord/phase49-packed-debug.attempted")


@g.jit
def debug_s3_packed_boundary49(
    Partial,
    Stream,
    Pre,
    Alpha,
    Beta,
    Out,
    Updated,
    Status,
    Epochs,
    Scratch,
    Peers,
    Reduced,
    Failure,
    Desc,
    Records,
    GRID: gl.constexpr,
    RANK: gl.constexpr,
    PAYLOAD: gl.constexpr,
    FLAGS: gl.constexpr,
):
    row = gl.program_id(0)
    gl.store(Records + (row * 16 + 15) * 8 + 7, 1)
    ep = checked.checked_ptr(Epochs + row * 32, 9, 4, Desc, Records, row, True)
    tag = _advance_tag(ep, 0)
    i = gl.arange(0, 512, layout=gl.BlockedLayout([1], [32], [4], [0]))
    sp = checked.checked_ptr(Status + i, 8, 4, Desc, Records, row, i >= 0)
    ok = gl.min(gl.load(sp), 0) == 1
    if ok:
        value, ok = standin.reduce_row(
            Partial, Scratch, Peers, Reduced, tag, row, 8192, GRID, RANK, PAYLOAD, FLAGS
        )
        gl.store(Records + (row * 16 + 15) * 8 + 7, 2)
        if ok:
            checked.packed_checked(
                row,
                Reduced + RANK * 8 * 8192,
                Stream,
                None,
                Pre,
                Alpha,
                Beta,
                Out,
                Updated,
                False,
                False,
                True,
                None,
                value,
                Desc,
                Records,
            )
        gl.store(Records + (row * 16 + 15) * 8 + 7, 3)
    dest = checked.checked_ptr(Status + row, 8, 4, Desc, Records, row, True)
    gl.store(dest, ok)
    # Failure publication uses checked addresses too; no spinning or traps.
    if not ok:
        fp = checked.checked_ptr(Failure, 10, 4, Desc, Records, row, True)
        old = gl.atomic_cas(fp, 0, 3, sem="relaxed", scope="gpu")
        if old == 0:
            pp = checked.checked_ptr(Failure + 1, 10, 4, Desc, Records, row, True)
            gl.store(pp, row)


class OffsetsFound(Exception):
    pass


def main():
    require_gpu0()
    if ATTEMPT.exists():
        raise RuntimeError(
            "the authorized packed-debug attempt was already used; do not repeat"
        )
    f, w, m, old = case()
    ws = chained.make_workspace("cuda", context=8192, left=-1)
    ws.s3 = s3.make_workspace("cuda")
    tensors = [
        ws.s3.peers.reduced[0],
        f.stream,
        w["alpha3"],
        w["beta3"],
        w["pre3"],
        ws.buffers["x4"],
        ws.buffers["mid_stream"],
        None,
        ws.s3.state.status,
        ws.s3.state.epochs,
        ws.failure,
    ]
    names = [
        "Branch",
        "Stream",
        "Alpha",
        "Beta",
        "Pre",
        "Out",
        "NewStream",
        "Post-disabled",
        "Status",
        "Epochs",
        "Failure",
    ]
    description = []
    ranges = []
    for name, tensor in zip(names, tensors):
        start = 0 if tensor is None else tensor.data_ptr()
        end = (
            start if tensor is None else start + tensor.numel() * tensor.element_size()
        )
        ranges.append([start, end])
        description.append(
            dict(
                pointer=name,
                base=start,
                end=end,
                shape=None if tensor is None else list(tensor.shape),
                dtype=None if tensor is None else str(tensor.dtype),
                element_bytes=0 if tensor is None else tensor.element_size(),
            )
        )
    ranges += [[0, 0]] * (16 - len(ranges))
    desc = torch.tensor(ranges, device="cuda", dtype=torch.int64)
    records = torch.zeros((8, 16, 8), device="cuda", dtype=torch.int64)
    meta = dict(
        allocations=description,
        diagnostic_address=records.data_ptr(),
        diagnostic_bytes=records.numel() * records.element_size(),
        shape=list(records.shape),
        protocol="one candidate chain call; payload guards before each scalar or b128 packet access, invalid pointers replaced by allocation base; no device assert/trap",
    )
    (LOG / "phase49-packed-debug-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    print(json.dumps(dict(event="metadata", **meta)), flush=True)

    def debug_attention(
        x,
        gate,
        weight,
        stream,
        pre,
        alpha,
        beta,
        *,
        rank,
        workspace,
        output,
        failure,
        skip=-1,
        delay_seed=None,
        guard_mode=0,
    ):
        assert rank == 0 and skip == -1 and delay_seed is None and guard_mode == 0
        s = workspace.state
        p = workspace.peers
        s3.normalize[(16,)](
            x, gate, workspace.normalized, failure, num_warps=4, enable_fp_fusion=False
        )
        wide_projection.project[(512,)](
            workspace.normalized,
            weight,
            workspace.projection,
            s.status,
            failure,
            N=8192,
            K=2048,
            STAGE=3,
            Packed=workspace.partial[rank],
            num_warps=2,
            enable_fp_fusion=False,
        )
        aa = (
            workspace.partial,
            stream,
            pre,
            alpha,
            beta,
            *output,
            s.status,
            s.epochs,
            p.scratch,
            p.peers,
            p.reduced,
            failure,
            desc,
            records,
        )
        kk = dict(
            GRID=p.ctas,
            RANK=rank,
            PAYLOAD=p.payload,
            FLAGS=p.flags,
            num_warps=4,
            enable_fp_fusion=False,
        )
        debug_s3_packed_boundary49[(8,)](*aa, **kk)
        torch.cuda.synchronize()
        data = records.cpu().tolist()
        (LOG / "phase49-packed-debug-records.json").write_text(json.dumps(data) + "\n")
        bad = [
            dict(
                row=r, pointer=names[i] if i < len(names) else str(i), record=data[r][i]
            )
            for r in range(8)
            for i in range(16)
            if data[r][i][0]
        ]
        print(
            json.dumps(
                dict(
                    event="checked-boundary",
                    bad=bad,
                    visits=[[int(data[r][i][7]) for i in range(16)] for r in range(8)],
                )
            ),
            flush=True,
        )
        if bad:
            (LOG / "phase49-packed-debug-result.json").write_text(
                json.dumps(dict(outcome="offsets-found", bad=bad), indent=2) + "\n"
            )
            raise OffsetsFound()
        return output

    # Compile before consuming the one execution marker. Compiler/API fixes
    # can be made without executing a packed diagnostic kernel.
    p3 = ws.s3.peers
    s3state = ws.s3.state
    compile_args = (
        ws.s3.partial,
        f.stream,
        w["pre3"],
        w["alpha3"],
        w["beta3"],
        ws.buffers["x4"],
        ws.buffers["mid_stream"],
        s3state.status,
        s3state.epochs,
        p3.scratch,
        p3.peers,
        p3.reduced,
        ws.failure,
        desc,
        records,
    )
    compile_opts = dict(
        GRID=p3.ctas,
        RANK=0,
        PAYLOAD=p3.payload,
        FLAGS=p3.flags,
        num_warps=4,
        enable_fp_fusion=False,
    )
    kernel = debug_s3_packed_boundary49.warmup(*compile_args, **compile_opts, grid=(8,))
    resource = kernel_occupancy("bounded-packed-debug", kernel)
    assert resource.active_ctas_per_cu >= 1
    (LOG / "phase49-packed-debug.amdgcn").write_text(kernel.asm["amdgcn"])
    (LOG / "phase49-packed-debug.llir").write_text(kernel.asm["llir"])
    print(
        json.dumps(
            dict(
                event="debug-admitted",
                vgpr=kernel.n_regs,
                spills=kernel.n_spills,
                lds=kernel.metadata.shared,
                occupancy=resource.active_ctas_per_cu,
            )
        ),
        flush=True,
    )
    chained.prepare_peers(ws, old["partial3"], old["partial5"], rank=0)
    # Mark the one allowed case before its first dispatch. Even a fault cannot
    # accidentally cause a retry by rerunning this script.
    with ATTEMPT.open("x") as marker:
        marker.write("one authorized case; see phase49-packed-debug-metadata.json\n")
    with ExitStack() as stack:
        stack.enter_context(patch.object(chained.s3, "attention_tail", debug_attention))
        modules = [
            (chained.s1, "qkvg"),
            (chained.s2, "attention"),
            (chained.s3, "attention_tail"),
            (chained.s4, "routing"),
            (chained.s5, "s5"),
            (chained.independent_experts, "s5"),
            (chained.independent_tail, "unpool_boundary"),
        ]
        for index, (module, name) in enumerate(modules):
            fn = getattr(module, name)

            def traced(*aa, _fn=fn, _name=module.__name__ + "." + name, **kk):
                print(json.dumps(dict(event="stage-start", name=_name)), flush=True)
                value = _fn(*aa, **kk)
                torch.cuda.synchronize()
                print(json.dumps(dict(event="stage-finished", name=_name)), flush=True)
                return value

            stack.enter_context(patch.object(module, name, traced))
        try:
            chained.launch(f.x, f.stream, w, m, ws, rank=0)
        except OffsetsFound:
            return
    try:
        archive_check(f, ws, old, 0, digest=False)
    except AssertionError as error:
        # Preserve the one execution's result even when correctness rejects it.
        (LOG / "phase49-packed-debug-result.json").write_text(
            json.dumps(
                dict(
                    outcome="no-bad-offsets-no-aperture-fault-but-numerical-mismatch",
                    attempts=1,
                    checker=error.args[0],
                    root_cause="unresolved; do not repeat",
                ),
                indent=2,
            )
            + "\n"
        )
        raise
    (LOG / "phase49-packed-debug-result.json").write_text(
        json.dumps(
            dict(outcome="no-bad-offsets-no-fault", archive_exact=True, attempts=1),
            indent=2,
        )
        + "\n"
    )
    print("PASS checked packed boundary, once only", flush=True)


if __name__ == "__main__":
    main()
