"""Cold regular/TLX S5 kernel ranges and logical weight bandwidth.

Event ranges include marker overhead. Bandwidth is logical weight+scale bytes
per event time, not a hardware counter of HBM transactions.
"""

import argparse
import functools
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import regular_layer45 as regular
import torch
from bench_chained47 import setup
from check_chained47 import archive_check
from diagnose_guard48 import balanced
from layer_fixture import require_gpu0
from reference_archive import gpu
from tlx_ir import dump
from triton.compiler import CompiledKernel

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_chain


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--left", type=int, required=True)
    a = p.parse_args()
    groups = {n: [] for n in ("regular", "tlx")}
    spaces = {n: [] for n in groups}
    retained = []
    tables = []
    for _ in range(3):
        base, native, oldws, errors, routes = setup(8192, a.left)
        retained.append((base, native, oldws, errors, routes))
        groups["regular"].append(base["regular"])
        spaces["regular"].append(native)
        x, stream, w, m, _ = base["chained"][1].args
        ws = tlx_chain.make_workspace("cuda", context=8192, left=a.left)
        spaces["tlx"].append(ws)
        old = gpu(
            torch.load(
                Path("/home/bangtliu/coord/phase44-baseline")
                / f"layer-7-8192-{a.left}.pt",
                weights_only=True,
            )["rows"][0]
        )
        groups["tlx"].append(
            (
                functools.partial(
                    tlx_chain.prepare_peers, ws, *native.partials, rank=0
                ),
                functools.partial(tlx_chain.launch, x, stream, w, m, ws, rank=0),
                functools.partial(archive_check, native.f, ws, old, 0, digest=False),
            )
        )
    allocations = {}
    for name in ("w13", "w13_scale", "w2", "w2_scale"):
        ptrs = [r[0]["chained"][1].args[2][name].data_ptr() for r in retained]
        assert len(set(ptrs)) == 3
        allocations[name] = ptrs
    cases = {}
    for n, rows in groups.items():

        def action(i, rows=rows):
            for row in rows:
                row[i]()

        cases[n] = tuple(functools.partial(action, i) for i in range(3))
    epochs = {
        "regular": spaces["regular"][0].epochs[0],
        "tlx": spaces["tlx"][0].s3.state.epochs,
    }
    plain = balanced(cases, epochs)
    phases = {
        "regular": ["w13", "w2", "combine", "ar"],
        "tlx": ["w13", "w2", "norm", "combine_ar"],
    }
    events = {
        n: {
            ph: [
                (
                    torch.cuda.Event(enable_timing=True, external=True),
                    torch.cuda.Event(enable_timing=True, external=True),
                )
                for _ in range(3)
            ]
            for ph in pp
        }
        for n, pp in phases.items()
    }
    counts = {(n, ph): 0 for n, pp in phases.items() for ph in pp}
    active = [None]
    seen = set()
    kernels = {}

    def measured(n, ph, fn, *aa, **kw):
        i = counts[n, ph] % 3
        counts[n, ph] += 1
        s, e = events[n][ph][i]
        s.record()
        old = active[0]
        active[0] = (n, ph)
        try:
            return fn(*aa, **kw)
        finally:
            active[0] = old
            e.record()

    class Proxy:
        def __init__(self, fn, n, phase):
            self.fn, self.n, self.phase = fn, n, phase

        def __getattr__(self, k):
            return getattr(self.fn, k)

        def __getitem__(self, grid):
            fn = self.fn[grid]

            def launch(*aa, **kw):
                ph = self.phase(aa, kw) if callable(self.phase) else self.phase
                return measured(self.n, ph, fn, *aa, **kw)

            return launch

    old_launch = regular.moe._launch_gemm

    def native_launch(*aa, **kw):
        ph = "w13" if aa[12] else "w2"
        if ph == "w13":
            tables.append((aa[4], aa[5], aa[6], aa[1], aa[2]))
        return measured("regular", ph, old_launch, *aa, **kw)

    old_collective = regular.Regular.collective

    def collective(self, part, index):
        if index == 1:
            return measured("regular", "ar", old_collective, self, part, index)
        return old_collective(self, part, index)

    old_metadata = CompiledKernel.launch_metadata

    def metadata(kernel, *aa, **kw):
        result = old_metadata(kernel, *aa, **kw)
        if active[0] and active[0] not in seen:
            n, ph = active[0]
            fn = kernel.src.fn
            const = {}
            for key, v in kernel.src.constants.items():
                if isinstance(key, tuple) and len(key) == 1:
                    const[fn.arg_names[key[0]]] = v
                elif isinstance(key, str):
                    const[key] = v
            kernels[n + "-" + ph] = dict(
                owner=fn.fn.__module__,
                function=fn.fn.__name__,
                constants=const,
                grid=aa[0] if aa else None,
                resources=dump(kernel, f"T8-S5-8192-{a.left}/" + ph, n),
                hsaco=hashlib.sha256(kernel.asm["hsaco"]).hexdigest(),
            )
            seen.add(active[0])
        return result

    def observer(n):
        return dict(
            kernel_us={
                ph: sum(s.elapsed_time(e) * 1000 for s, e in pairs) / 3
                for ph, pairs in events[n].items()
                if counts[n, ph]
            }
        )

    with ExitStack() as stack:
        stack.enter_context(patch.object(CompiledKernel, "launch_metadata", metadata))
        stack.enter_context(patch.object(regular.moe, "_launch_gemm", native_launch))
        stack.enter_context(
            patch.object(
                regular.moe,
                "finalize_norm_weighted_sum_kernel",
                Proxy(
                    regular.moe.finalize_norm_weighted_sum_kernel, "regular", "combine"
                ),
            )
        )
        stack.enter_context(patch.object(regular.Regular, "collective", collective))
        mod = tlx_chain.tlx_experts
        for name, ph in [
            ("expert_phase", lambda aa, kw: "w13" if kw["FIRST"] else "w2"),
            ("norm_phase", "norm"),
            ("sum_reduce", "combine_ar"),
        ]:
            stack.enter_context(
                patch.object(mod, name, Proxy(getattr(mod, name), "tlx", ph))
            )
        stamped = balanced(cases, epochs, observer)
    work = {}
    for n in groups:
        copies = []
        for obj in spaces[n]:
            ids = obj.s4.ids if n == "regular" else obj.s4.ids
            vals = ids[(ids >= 0) & (ids < 64)]
            unique, cc = vals.unique(return_counts=True)
            blocks = int(((cc + 15) // 16).sum())
            copies.append(
                dict(
                    hits=vals.numel(),
                    experts=unique.cpu().tolist(),
                    blocks=blocks,
                    weight_scale_bytes={
                        "w13": blocks * (8192 * 4096 // 2 + 8192 * 4096 // 32),
                        "w2": blocks * (4096 * 4096 // 2 + 4096 * 4096 // 32),
                    },
                )
            )
        work[n] = copies
    result = dict(
        context=8192,
        left=a.left,
        copies=3,
        allocations=allocations,
        plain=plain,
        stamped=stamped,
        kernels=kernels,
        work=work,
        regular_config=spaces["regular"][0].cfg,
        fp4_mode=regular.moe.FP4_MODE,
        limits="Logical touched weight+scale bytes/time; does not measure physical HBM transactions; markers perturb dispatch. Plain control retained.",
    )
    Path(
        "/home/bangtliu/coord/logs", f"phaseT8-s5-attribution-8192-{a.left}.json"
    ).write_text(json.dumps(result, indent=2, default=str) + "\n")
    print("PASS S5 attribution", a.left, flush=True)


if __name__ == "__main__":
    main()
