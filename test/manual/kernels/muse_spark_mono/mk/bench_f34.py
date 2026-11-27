"""MK.1: T11 chain vs chain with S3-boundary || S4-projection (tlx_fused34).

GPU0 only. Three distinct weight copies (cold), alternating orders, archive-exact
checks after every timed replay of the fused variant.
"""

import argparse
import functools
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch  # noqa: E402
from bench_chained47 import setup  # noqa: E402
from check_chained47 import archive_check  # noqa: E402
from layer_fixture import require_gpu0  # noqa: E402

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (  # noqa: E402
    tlx_attn_tail,
    tlx_chain,
    tlx_fused34,
    tlx_route,
)


def fused_launch(st, nb, prefetch=True, one=False, sleep=0, wait=True):
    """tlx_chain.launch with S3 tail + S4 routing replaced by tlx_fused34."""
    stash = {}

    def attention_tail(x, gate, o_proj, stream, pre, alpha, beta, *, rank, workspace,
                       store_reduced, split_norm, output, failure, **_):
        stash.update(locals())
        return output

    def routing(x4, weight, bias, norm, *, rank, workspace, failure, **_):
        a = stash
        if one:
            return tlx_fused34.tail_route_one(
                a["x"], a["gate"], a["o_proj"], a["stream"], a["pre"], a["alpha"], a["beta"],
                weight, bias, norm, rank=rank, s3=a["workspace"], s4=workspace, st=st,
                output=a["output"], failure=failure, split_norm=a["split_norm"],
                store_reduced=a["store_reduced"], nb=nb, sleep=sleep, wait=wait,
            )
        return tlx_fused34.tail_route(
            a["x"], a["gate"], a["o_proj"], a["stream"], a["pre"], a["alpha"], a["beta"],
            weight, bias, norm, rank=rank, s3=a["workspace"], s4=workspace, st=st,
            output=a["output"], failure=failure, split_norm=a["split_norm"],
            store_reduced=a["store_reduced"], nb=nb, prefetch=prefetch,
        )

    shim3 = types.SimpleNamespace(attention_tail=attention_tail)
    shim4 = types.SimpleNamespace(routing=routing)

    def call(*args, **kw):
        old3, old4 = tlx_chain.tlx_attn_tail, tlx_chain.tlx_route
        tlx_chain.tlx_attn_tail, tlx_chain.tlx_route = shim3, shim4
        try:
            return tlx_chain.launch(*args, **kw)
        finally:
            tlx_chain.tlx_attn_tail, tlx_chain.tlx_route = old3, old4

    return call


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--left", type=int, default=-1)
    p.add_argument("--nb", type=int, default=8)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--no-prefetch", action="store_true")
    p.add_argument("--one", action="store_true", help="single fused launch")
    p.add_argument("--only", default="")
    p.add_argument("--sleep", type=int, default=0)
    p.add_argument("--nowait", action="store_true")
    p.add_argument("--p4n", action="store_true", help="chain with S4 projection = project4n only")
    a = p.parse_args()
    from reference_archive import gpu

    copies = []
    retained = []
    for _ in range(3):
        base, native, *rest = setup(a.context, a.left)
        retained.append((base, native, rest))
        x, stream, w, m, _ = base["chained"][1].args
        old = gpu(
            torch.load(
                Path("/home/bangtliu/coord/phase44-baseline")
                / f"layer-7-{a.context}-{a.left}.pt",
                weights_only=True,
            )["rows"][0]
        )
        row = {}
        for name in ([a.only] if a.only else ["tlx", "f34"]):
            ws = tlx_chain.make_workspace("cuda", context=a.context, left=a.left)
            prep = functools.partial(tlx_chain.prepare_peers, ws, *native.partials, rank=0)
            if name == "tlx":
                fn = functools.partial(tlx_chain.launch, x, stream, w, m, ws, rank=0)
            elif a.p4n:
                def fn(x=x, stream=stream, w=w, m=m, ws=ws):
                    old4 = tlx_chain.tlx_route
                    tlx_chain.tlx_route = types.SimpleNamespace(routing=tlx_fused34.routing_p4n)
                    try:
                        return tlx_chain.launch(x, stream, w, m, ws, rank=0)
                    finally:
                        tlx_chain.tlx_route = old4
            else:
                st = tlx_fused34.State("cuda")
                fn = functools.partial(fused_launch(st, a.nb, not a.no_prefetch, a.one, a.sleep, not a.nowait), x, stream, w, m, ws, rank=0)
            check = functools.partial(archive_check, native.f, ws, old, 0, digest=False)
            if False:
                rngs = {}
                def add(k, t):
                    if isinstance(t, torch.Tensor):
                        rngs[k] = (hex(t.data_ptr()), hex(t.data_ptr() + t.numel() * t.element_size()))
                for k, t in w.items():
                    add("w." + k, t)
                for k, t in ws.buffers.items():
                    add("b." + k, t)
                for sname in ("s3", "s4"):
                    sub = getattr(ws, sname)
                    for f in getattr(type(sub), "__struct_fields__", ()):
                        add(sname + "." + f, getattr(sub, f))
                for f in ("counters", "epoch"):
                    add("st." + f, getattr(st, f))
                add("ws.failure", ws.failure)
                pass
            prep()
            fn()
            torch.cuda.synchronize()
            print("EAGER", name, flush=True)
            if not (a.nowait and name == "f34"):
                check()  # eager exactness
            prep()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            row[name] = (prep, g, check)
            row["_ws_" + name] = ws
            row["_old"] = old
            row["_w"] = w
            if name == "f34" and not a.p4n:
                row["_st_" + name] = st
        copies.append(row)
    times = {k: [] for k in copies[0] if not k.startswith('_')}
    for r in range(a.rounds):
        names = [k for k in copies[0] if not k.startswith('_')]
        order = names if r % 2 == 0 else names[::-1]
        for c in copies:
            for name in order:
                prep, g, check = c[name]
                prep()
                torch.cuda.synchronize()
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                g.replay()
                e.record()
                torch.cuda.synchronize()
                times[name].append(s.elapsed_time(e) * 1000)
                try:
                    if not (a.nowait and name == "f34"):
                        check()
                except AssertionError:
                    st_ = c.get("_st_" + name)
                    ws_ = c.get("_ws_" + name)
                    print("FAILDBG", name, "round", r, "counters", st_.counters[::32][:4].tolist() if st_ else None,
                          "epoch", st_.epoch[:4].tolist() if st_ else None,
                          "proj_status_sum", int(ws_.s4.producer.sum()), "s3status", ws_.s3.state.status[:8].tolist(),
                          "failure", ws_.failure.tolist(), flush=True)
                    old_ = c["_old"]
                    for nm in ("x4", "mid_stream"):
                        if nm in old_["buffers"]:
                            a_ = ws_.buffers[nm].view(torch.int16); b_ = old_["buffers"][nm].view(torch.int16)
                            print("FAILDBG", nm, "mismatch", int((a_ != b_).sum()), "of", a_.numel(), flush=True)
                    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_projection
                    pr = torch.zeros_like(ws_.s4.projection); sr = torch.zeros(288, device="cuda", dtype=torch.int32)
                    fz = torch.zeros(2, device="cuda", dtype=torch.int32)
                    tlx_projection.project[(288,)](ws_.buffers["x4"], c["_w"]["joint_router_pooler"], pr, sr, Failure=fz,
                        N=4608, K=8192, SK=8, STAGE=4, ROUND_OUTPUT=False, num_warps=2, enable_fp_fusion=False)
                    torch.cuda.synchronize()
                    bad = (pr != ws_.s4.projection)
                    print("FAILDBG P-vs-recomputed-from-final-x4 mismatch", int(bad.sum()),
                          "tiles", sorted(set((bad.nonzero()[:, 1] // 16).tolist()))[:20], flush=True)
                    raise
    if not a.rounds:
        print("EAGER_OK", flush=True)
        return
    med = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
    print(json.dumps(dict(context=a.context, left=a.left, nb=a.nb, sleep=a.sleep, median_us=med,
                          ratio=(med["f34"] / med["tlx"]) if len(med) == 2 else None, n=len(times["tlx"]),
                          exact="all replays archive-exact")), flush=True)


if __name__ == "__main__":
    main()
