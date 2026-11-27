"""Minimal constexpr guard variants, balanced order/parity, dependent whole graph."""

import argparse
import functools
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from check_chained47 import BASE, archive_check
from diagnose_parity48 import capture
from layer_case import make
from layer_fixture import require_gpu0
from paired45 import distribution
from reference_archive import gpu

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import chained

LOG = Path("/home/bangtliu/coord/logs")


def balanced(cases, epochs, observer=None):
    runs = {}
    for label, (prepare, call, check) in cases.items():
        prepare()
        call()
        torch.cuda.synchronize()
        check()
        runs[label] = capture(prepare, call)
    rows = []
    labels = tuple(cases)
    for rep in range(208):
        order = "AB" if rep % 2 == 0 else "BA"
        first = (rep // 2) % 2
        for label in labels if rep % 4 < 2 else labels[::-1]:
            if (int(epochs[label].flatten()[0]) + 1) % 2 != first:
                runs[label][0].replay()
                torch.cuda.synchronize()
        for label in labels if order == "AB" else labels[::-1]:
            for position in range(2):
                graph, start, end = runs[label]
                graph.replay()
                torch.cuda.synchronize()
                parity = int(epochs[label].flatten()[0]) % 2
                assert parity == (first + position) % 2
                if rep >= 8:
                    row = dict(
                        label=label,
                        order=order,
                        parity=parity,
                        position=position,
                        us=start.elapsed_time(end) * 1000,
                    )
                    if observer is not None:
                        row.update(observer(label))
                    rows.append(row)
    result = {}
    for label in labels:
        cases[label][2]()
        select = lambda order, parity: [
            r["us"]
            for r in rows
            if r["label"] == label
            and r["order"] == order
            and (parity is None or r["parity"] == parity)
        ]
        result[label] = dict(
            **distribution([r["us"] for r in rows if r["label"] == label]),
            by_order={o: distribution(select(o, None)) for o in ("AB", "BA")},
            cells={
                f"{o}-p{p}": distribution(select(o, p))
                for o in ("AB", "BA")
                for p in (0, 1)
            },
        )
        assert all(x["n"] == 100 for x in result[label]["cells"].values())
    return dict(timing=result, raw=rows)


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--left", type=int, default=-1)
    p.add_argument("--candidate", type=int, choices=(1, 2), required=True)
    p.add_argument("--shared", action="store_true")
    p.add_argument("--runtime", action="store_true")
    a = p.parse_args()
    f, w, m = make(7, a.context, a.left, 0)
    old = gpu(
        torch.load(BASE / f"layer-7-{a.context}-{a.left}.pt", weights_only=True)[
            "rows"
        ][0]
    )
    cases = {}
    epochs = {}
    spaces = {}
    shared_ws = (
        chained.make_workspace(
            "cuda",
            context=a.context,
            left=a.left,
            stage_policy=("common", "common", "common"),
        )
        if a.shared
        else None
    )
    fixed = (
        (
            torch.empty((8, 2048), device="cuda", dtype=torch.bfloat16),
            torch.empty((8, 256), device="cuda", dtype=torch.bfloat16),
        )
        if a.shared
        else None
    )
    fixed = (
        (fixed[0], fixed[1], torch.empty_like(fixed[1]), torch.empty_like(fixed[0]))
        if fixed
        else None
    )
    if a.runtime:
        assert a.shared
        shared_ws.failure = torch.zeros((3,), device="cuda", dtype=torch.int32)
    original_qkvg = chained.s1.qkvg
    original_empty, original_like = torch.empty, torch.empty_like

    def fixed_qkvg(*args, **kwargs):
        # Restrict allocator substitution to this API's four known output
        # allocations. No other stage or CUDA graph setup is patched.
        outputs = iter(fixed)

        def allocate(*aa, **kk):
            value = next(outputs)
            expected = (
                tuple(aa[0]) if isinstance(aa[0], (tuple, list)) else tuple(aa[0].shape)
            )
            assert tuple(value.shape) == expected
            return value

        with (
            patch.object(torch, "empty", allocate),
            patch.object(torch, "empty_like", allocate),
        ):
            result = original_qkvg(*args, **kwargs)
        assert all(x is y for x, y in zip(result, fixed))
        return result

    for label, mode in [
        ("reload", 0),
        ("register" if a.candidate == 1 else "disabled", a.candidate),
    ]:
        ws = (
            shared_ws
            if a.shared
            else chained.make_workspace(
                "cuda",
                context=a.context,
                left=a.left,
                stage_policy=("common", "common", "common"),
            )
        )
        spaces[label] = ws
        prepare = functools.partial(
            chained.prepare_peers, ws, old["partial3"], old["partial5"], rank=0
        )
        if a.runtime:
            base_prepare = prepare

            def prepare(base=base_prepare, mode=mode, ws=ws):
                base()
                ws.failure[2:3].fill_(mode)

        call = functools.partial(
            chained.launch,
            f.x,
            f.stream,
            w,
            m,
            ws,
            rank=0,
            guard_mode=-1 if a.runtime else mode,
        )
        check = functools.partial(archive_check, f, ws, old, 0, digest=False)
        cases[label] = (prepare, call, check)
        epochs[label] = ws.s3.state.epochs
    compiled = {}
    from triton.compiler import CompiledKernel

    original_metadata = CompiledKernel.launch_metadata

    def metadata(kernel, *aa, **kk):
        result = original_metadata(kernel, *aa, **kk)
        module = kernel.src.fn.fn.__module__
        if (
            module.endswith(
                tuple(
                    "common_" + n
                    for n in (
                        "qkvg",
                        "attention",
                        "attn_tail",
                        "route",
                        "experts",
                        "tail",
                    )
                )
            )
            and kernel.src.fn.fn.__name__ == "_launch"
        ):
            sha = hashlib.sha256(kernel.asm["hsaco"]).hexdigest()
            compiled.setdefault(module, {})[sha] = dict(
                vgpr=kernel.n_regs,
                spills=kernel.n_spills,
                lds=kernel.metadata.shared,
                constants={str(k): str(v) for k, v in kernel.src.constants.items()},
            )
            if a.runtime and a.context == 8192 and a.left == -1 and a.candidate == 2:
                (
                    LOG / ("phase48-runtime-" + module.rsplit(".", 1)[-1] + ".amdgcn")
                ).write_text(kernel.asm["amdgcn"])
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(CompiledKernel, "launch_metadata", metadata))
        if a.shared:
            stack.enter_context(patch.object(chained.s1, "qkvg", fixed_qkvg))
        result = balanced(cases, epochs)
    result["compiled"] = compiled
    if a.runtime:
        assert len(compiled) == 6 and all(len(v) == 1 for v in compiled.values()), (
            compiled
        )
    result["same_workspace_and_outputs"] = a.shared
    result.update(
        context=a.context,
        left=a.left,
        candidate=a.candidate,
        resources={n: chained.resources(ws) for n, ws in spaces.items()},
        constexpr_pair=not a.runtime,
        identical_binary_runtime=a.runtime,
    )
    (
        LOG
        / f"phase48-guard-{a.candidate}-{a.context}-{a.left}{'-runtime' if a.runtime else ('-shared' if a.shared else '')}.json"
    ).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["timing"]), flush=True)


if __name__ == "__main__":
    main()
