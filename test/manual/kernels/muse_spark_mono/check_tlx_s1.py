"""TLX S1 exact Gluon bits/cache plus the existing native/FP64/replay suite."""

import argparse
import functools
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from layer_fixture import EPS, Q_SCALE, require_gpu0
from layer_reference import exact_check
from reference_archive import fixture
from tlx_ir import dump
from triton.compiler import CompiledKernel

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    independent_qkvg as old,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_qkvg as new

IR_TAG = "T3-S1"
ROOT = Path("/home/bangtliu/coord/logs/phaseT3-S1-suite")


def main():
    global ROOT, IR_TAG
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--head-rows", type=int, choices=(8, 16), default=16)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--suite-only", action="store_true")
    p.add_argument("--exact-only", action="store_true")
    p.add_argument("--label", default="")
    a = p.parse_args()
    new.qkvg = functools.partial(new.qkvg, head_rows=a.head_rows)
    if a.label:
        ROOT = ROOT.with_name("phaseT3-S1-" + a.label + "-suite")
        IR_TAG += "-" + a.label
    ROOT.mkdir(exist_ok=True)
    cases = [
        (c, s, r, r, False)
        for c in ((8192,) if a.smoke else (8192, 32768, 131072))
        for s in ((7,) if a.smoke else (7, 20261117))
        for r in (False, True)
    ]
    if not a.smoke:
        cases += [
            (8192, 17, r, m, True)
            for r, m in [(True, True), (True, False), (False, True)]
        ]
    records = []
    for context, seed, rope, mapping, skip in [] if a.suite_only else cases:
        f = fixture(seed, context)
        if skip:
            if mapping:
                f.mapping[f.locations[1]] = -1
            f.locations[0] = -1
        outputs = []
        spaces = []
        caches = []
        for side, module in [("gluon", old), ("tlx", new)]:
            ws = module.make_workspace("cuda")
            kc = f.cache_k.clone()
            vc = f.cache_v.clone()
            print(
                json.dumps(
                    dict(
                        begin=side,
                        context=context,
                        seed=seed,
                        rope=rope,
                        mapping=mapping,
                        skip=skip,
                    )
                ),
                flush=True,
            )
            outputs.append(
                module.qkvg(
                    f.x,
                    f.weights["qkvg"],
                    f.positions,
                    f.cos_sin if rope else None,
                    kc,
                    vc,
                    f.locations,
                    f.mapping if mapping else None,
                    EPS,
                    Q_SCALE,
                    f.k_scale,
                    f.v_scale,
                    workspace=ws,
                )
            )
            torch.cuda.synchronize()
            spaces.append(ws)
            caches.append((kc, vc))
        for name, x, y in (
            [("projection", spaces[0].projection, spaces[1].projection)]
            + [(n, x, y) for n, x, y in zip(["q", "k", "v", "gate"], *outputs)]
            + [(n, x, y) for n, x, y in zip(["cache_k", "cache_v"], *caches)]
        ):
            dtype = (
                torch.uint8
                if name.startswith("cache")
                else (torch.int16 if x.dtype == torch.bfloat16 else torch.int32)
            )
            exact_check(x.view(dtype), y.view(dtype), "S1 " + name, {"_rank": 0})
        records.append(
            dict(
                context=context,
                seed=seed,
                rope=rope,
                mapping=mapping,
                skip=skip,
                exact=True,
                resources=list(spaces[1].admitted.values()),
            )
        )
        (ROOT / "exact.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(records[-1]), flush=True)
    if a.exact_only or a.smoke:
        return
    import check_common_s1 as suite

    suite.stage = new
    original = suite.suite.one_case

    def case(*aa, **kw):
        kw["ctas"] = 288
        return original(*aa, **kw)

    suite.suite.one_case = case
    suite.ROOT = ROOT
    suite.suite.LOG = ROOT
    shutil.copyfile(
        ROOT.parent / "phase3.0-read-and-copy-traffic.json",
        ROOT / "phase3.0-read-and-copy-traffic.json",
    )

    class Epilogue:
        def __getitem__(self, grid):
            def launch(*aa, **kw):
                state = torch.zeros((288 * 32 + 72,), device="cuda", dtype=torch.int32)
                status = torch.ones((288,), device="cuda", dtype=torch.int32)
                return new.qkvg_heads[grid](
                    aa[0], state, status, *aa[1:13], None, *aa[13:17], **kw
                )

            return launch

    with (
        patch.object(sys, "argv", ["suite"]),
        patch.object(suite.suite, "epilogue_only", Epilogue()),
    ):
        suite.main()


if __name__ == "__main__":
    seen = set()
    orig = CompiledKernel.launch_metadata

    def metadata(kernel, *args, **kw):
        result = orig(kernel, *args, **kw)
        fn = kernel.src.fn.fn
        module = fn.__module__.rsplit(".", 1)[-1]
        phase = {"project": "projection", "qkvg_heads": "heads"}.get(fn.__name__)
        if phase and module in [
            "wide_projection",
            "independent_qkvg",
            "tlx_projection",
            "tlx_qkvg",
        ]:
            side = "tlx" if module.startswith("tlx") else "gluon"
            key = (side, phase)
            # Capture the first RoPE+MAP specialization, not later delayed suites.
            if key not in seen and (
                phase == "projection"
                or kernel.src.constants.get(
                    (fn.__code__.co_varnames.index("ROPE"),), False
                )
                or kernel.src.constants.get("ROPE", False)
            ):
                dump(kernel, IR_TAG + "/" + phase, side)
                seen.add(key)
        return result

    with patch.object(CompiledKernel, "launch_metadata", metadata):
        main()
