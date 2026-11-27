"""Monokernel step A/B on GPU0 (simulated peers): production T11 chain vs chain flag variants.

VARIANTS (env/argv): comma list of names from FLAGS below. Each variant runs the whole
layer with its tlx_chain module flags set, is archive-exact checked after every replay,
and is timed cold (3 rotating weight copies, graph replay, variants interleaved).
usage: bench_mono.py --context 8192 --left -1 --variants base,mergenorm
"""
import argparse
import functools
import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import torch  # noqa: E402
from bench_chained47 import setup  # noqa: E402
from check_chained47 import archive_check  # noqa: E402
from layer_fixture import require_gpu0  # noqa: E402
from reference_archive import gpu  # noqa: E402

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250 import stages  # noqa: E402
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_tagged  # noqa: E402,F401
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_tagged_experts  # noqa: E402
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_attention, tlx_chain, tlx_experts, tlx_projection, tlx_qkvg, tlx_route, tlx_tail  # noqa: E402,F401

FLAGS = {
    "base": {},
    "mergenorm": {"FUSE_MERGE_NORM": True},
    "s1heads": {"FUSE_S1_HEADS": True},
    "rowslots": {"FUSE_ROWS_SLOTS": True},
    "s1s2": {"FUSE_S1S2": True},
    "early": {"tlx_projection.EARLY_ISSUE": True},
    "early5": {"tlx_experts.EARLY_ISSUE": True},
    "early2": {"tlx_attention.EARLY_ISSUE": True},
    "d13": {"tlx_experts.DEPTH13": 3},
    "d2": {"tlx_experts.DEPTH2": 3},
    "d2x4": {"tlx_experts.DEPTH2": 4},
    "d34": {"tlx_experts.DEPTH13": 3, "tlx_experts.DEPTH2": 4},
    "d2x6": {"tlx_experts.DEPTH2": 6},
    "d13x4": {"tlx_experts.DEPTH13": 4},
    "d44": {"tlx_experts.DEPTH13": 4, "tlx_experts.DEPTH2": 4},
    "s3d2": {"tlx_projection.DEPTH_S3": 2},
    "s3d3": {"tlx_projection.DEPTH_S3": 3},
    "s1d3": {"tlx_projection.DEPTH_S1": 3},
    "s4d3": {"tlx_projection.DEPTH_S4": 3},
    "w13d3": {"tlx_experts.DEPTH13": 3},
    "w2d3": {"tlx_experts.DEPTH2": 3},
    "w2d5": {"tlx_experts.DEPTH2": 5},
    "s6d2": {"S6_DEPTH": 2},
    "s6d3": {"S6_DEPTH": 3},
    "s6d4": {"S6_DEPTH": 4},
    "s1d2": {"tlx_projection.DEPTH_S1": 2},
    "s4d2": {"tlx_projection.DEPTH_S4": 2},
    "hyb": {"tlx_projection.COPY_MODE": "hybrid"},
    "bn4": {"tlx_projection.BN_S4": 32},
    "bn6": {"tlx_projection.BN_S6": 32},
    "bn3": {"tlx_projection.BN_S3": 32},
    "bn46": {"tlx_projection.BN_S4": 32, "tlx_projection.BN_S6": 32},
    "bn346": {"tlx_projection.BN_S4": 32, "tlx_projection.BN_S6": 32, "tlx_projection.BN_S3": 32},
    "bn4d3": {"tlx_projection.BN_S4": 32, "tlx_projection.DEPTH_S4": 3},
    "bn4d6": {"tlx_projection.BN_S4": 32, "tlx_projection.DEPTH_S4": 6},
    "bn1": {"tlx_projection.BN_S1": 32},
    "bn1d3": {"tlx_projection.BN_S1": 32, "tlx_projection.DEPTH_S1": 3},
    "esum": {"tlx_experts.EARLY_SUM": True},
    "es3a": {"tlx_attn_tail.EARLY_S3": 1},
    "es3b": {"tlx_attn_tail.EARLY_S3": 2},
    "s3d4": {"tlx_projection.DEPTH_S3": 4},
    "bn6x64": {"tlx_projection.BN_S6": 64},
    "bn6x64d2": {"tlx_projection.BN_S6": 64, "S6_DEPTH": 2},
    "bn6x64d3": {"tlx_projection.BN_S6": 64, "S6_DEPTH": 3},
    "bn3x64": {"tlx_projection.BN_S3": 64},
    "bn3x64d3": {"tlx_projection.BN_S3": 64, "tlx_projection.DEPTH_S3": 3},
    "pre22": {"tlx_route.EARLY_RS": False, "tlx_projection.BN_S6": 32,
              "S6_DEPTHS": {(8192, -1): 2, (8192, 2047): 3, (131072, -1): 3, (131072, 2047): 2}},
    "fold2": {"tlx_experts.FOLD2": True, "_numeric": True},
    "mx13": {"tlx_experts.MX13": True, "tlx_experts.SPLIT13": False,
             "tlx_experts.FOLD13": False, "_numeric": True},
    "mxs": {"tlx_experts.MX13": True, "tlx_experts.SPLIT13": True, "_numeric": True},
    "fold": {"tlx_experts.MX13": True, "tlx_experts.SPLIT13": True, "tlx_experts.FOLD13": True,
             "_numeric": True},
    "mx2": {"tlx_experts.MX2": True, "_numeric": True},
    "mx": {"tlx_experts.MX13": True, "tlx_experts.MX2": True, "_numeric": True},
    "ers": {"tlx_route.EARLY_RS": True},
    "fn128": {"tlx_chain.FUSE_NORM_FULL128": True},
    "w13e1": {"tlx_experts.WPE13": 1},
    "w2e1": {"tlx_experts.WPE2": 1},
    "w13e3": {"tlx_experts.WPE13": 3},
    "w13e4": {"tlx_experts.WPE13": 4},
    "w2e3": {"tlx_experts.WPE2": 3},
    "e33": {"tlx_experts.WPE13": 3, "tlx_experts.WPE2": 3},
    "f132": {"tlx_experts.FUSE13_2": True},
    "f132g384": {"tlx_experts.FUSE13_2": True, "tlx_experts.FUSED_GRID": 384,
                 "tlx_experts.FUSED_DEPTH2": 4},
    "f132p": {"tlx_experts.FUSE13_2": True, "tlx_experts.FUSED_PREWAIT": True},
    "f132g384p": {"tlx_experts.FUSE13_2": True, "tlx_experts.FUSED_GRID": 384,
                  "tlx_experts.FUSED_DEPTH2": 4, "tlx_experts.FUSED_PREWAIT": True},
    "pub13": {"tlx_experts.FUSE13_2": False, "tlx_experts.PROBE_PUBLISH13": True},
    "unf": {"tlx_experts.FUSE13_2": False},
    # 256-CU parts (g13-14): fused at 3 CTAs per CU
    "f768p": {"tlx_experts.FUSE13_2": True, "tlx_experts.FUSED_GRID": 768,
              "tlx_experts.FUSED_PREWAIT": True},
    "unfd2": {"tlx_experts.FUSE13_2": False, "tlx_experts.DEPTH2": 2},
    "fglob": {"tlx_experts.FUSE13_2": True, "tlx_experts.FUSED_GLOBAL": True},
    "fnopre": {"tlx_experts.FUSE13_2": True, "tlx_experts.FUSED_PREWAIT": False},
    "f64": {"tlx_experts.PUBLISH64": True},
    # T13: the port of the gfx950 tagged-pair mailbox (TokenSpeed mailbox.py), W13 -> W2
    "t13": {"tlx_chain.tlx_experts": tlx_tagged_experts},
    "w13e1d3": {"tlx_experts.WPE13": 1, "tlx_experts.DEPTH13": 3},
}


def _owner(k):
    """FLAGS keys are tlx_chain attributes, or "module.ATTR" for another stage module
    (or a framework module)."""
    if "." in k:
        mod, attr = k.split(".")
        if hasattr(stages, mod):
            return getattr(stages, mod), attr
        return importlib.import_module(
            f"sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.{mod}"), attr
    return tlx_chain, k


def with_flags(flags, fn):
    flags = {k: v for k, v in flags.items() if not k.startswith("_")}
    saved = {k: getattr(*_owner(k)) for k in flags}
    for k, v in flags.items():
        setattr(*_owner(k), v)
    try:
        return fn()
    finally:
        for k, v in saved.items():
            setattr(*_owner(k), v)


NUMERIC = {}


def numeric_check(name, ws, ref):
    """Record (not assert) the deviation of a new-numerics variant from the exact one.

    The exact variant replays first in every round only when it is listed first;
    its buffers hold the output of the same inputs (same copy)."""
    d = NUMERIC.setdefault(name, {})
    for k in ("out", "stream", "x4", "attn"):
        a_, b_ = ws.buffers[k].float(), ref.buffers[k].float()
        err = (a_ - b_).abs().max().item()
        rel = err / max(b_.abs().max().item(), 1e-30)
        d.setdefault(k + "_max_abs", []).append(err)
        d.setdefault(k + "_max_rel", []).append(rel)
    d.setdefault("ids_equal", []).append(bool(torch.equal(ws.s4.ids, ref.s4.ids)))


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--left", type=int, default=-1)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--variants", default="base,mergenorm")
    p.add_argument("--copies", type=int, default=3)  # 1 = warm (same weights every replay)
    a = p.parse_args()
    names = a.variants.split(",")
    retained, copies = [], []
    for ci in range(a.copies):
        base, native, *rest = setup(a.context, a.left)
        x, stream, w, m, _ = base["chained"][1].args
        old = gpu(torch.load(Path("/home/bangtliu/coord/phase44-baseline")
                             / f"layer-7-{a.context}-{a.left}.pt", weights_only=True)["rows"][0])
        retained.append((base, native, rest, old))
        row = {}
        for name in names:
            ws = with_flags(FLAGS[name], functools.partial(
                tlx_chain.make_workspace, "cuda", context=a.context, left=a.left))
            prep = functools.partial(tlx_chain.prepare_peers, ws, *native.partials, rank=0)
            run = functools.partial(with_flags, FLAGS[name],
                                    functools.partial(tlx_chain.launch, x, stream, w, m, ws, rank=0))
            if FLAGS[name].get("_numeric"):
                # new numerics: compare with the first (exact) variant of this copy
                check = functools.partial(numeric_check, name, ws, row[names[0]][3])
            else:
                check = functools.partial(archive_check, native.f, ws, old, 0, digest=False)
            prep(); run(); torch.cuda.synchronize(); check()
            prep(); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                run()
            row[name] = (prep, g, check, ws)
            retained.append((ws, g))
        copies.append(row)
    times = {n: [] for n in names}
    for r in range(a.rounds):
        order = names if r % 2 == 0 else names[::-1]
        for c in copies:
            for name in order:
                prep, g, check, _ = c[name]
                prep(); torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
                times[name].append(e0.elapsed_time(e1) * 1000)
                check()
    med = {k: round(sorted(v)[len(v) // 2], 2) for k, v in times.items()}
    if NUMERIC:
        print(json.dumps(dict(numeric={n: {k: max(v) if k != "ids_equal" else min(v)
                                         for k, v in d.items()} for n, d in NUMERIC.items()})))
    print(json.dumps(dict(context=a.context, left=a.left, median_us=med,
                          exact=sorted(n for n in names if not FLAGS[n].get("_numeric")),
                          ratio={n: round(med[n] / med[names[0]], 4) for n in names[1:]},
                          n=len(times[names[0]]))), flush=True)


if __name__ == "__main__":
    main()
