"""MK.3: T11 chain vs chain with the MI355-style split S3 boundary (tlx_s3split).

Production: archive bit-exact. New design: FP64 reference + regular-path tolerances
+ routing (layer_reference.compare) once, then every replay bit-identical to its
own first run (determinism). Cold: 3 weight copies, alternating order.
"""
import argparse, functools, json, os, sys, types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import torch  # noqa: E402
from bench_chained47 import setup  # noqa: E402
from check_chained47 import archive_check  # noqa: E402
from layer_fixture import require_gpu0  # noqa: E402
from layer_reference import build, compare  # noqa: E402
from reference_archive import gpu  # noqa: E402

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (  # noqa: E402
    tlx_attn_tail, tlx_chain, tlx_projection, tlx_route, tlx_s3fused, tlx_s3split,
)


def split_tail(sums, s, rowcnt=None):
    def attention_tail(x, gate, o_proj, stream, pre, alpha, beta, *, rank, workspace,
                       store_reduced, split_norm, output, failure, **_):
        ws = workspace
        out, updated = output
        fn = tlx_attn_tail.normalize_rows if split_norm else tlx_attn_tail.normalize
        fn[(256 if split_norm else 16,)](x, gate, ws.normalized, failure, num_warps=4,
                                         enable_fp_fusion=False)
        tlx_projection.project[(512,)](ws.normalized, o_proj, ws.projection, ws.state.status,
                                       Failure=failure, N=8192, K=2048, STAGE=3,
                                       Packed=ws.partial[rank], num_warps=2,
                                       enable_fp_fusion=False)
        if rowcnt is not None:
            tlx_s3split.boundary_one(ws.partial, stream, pre, alpha, beta, out, updated, ws,
                                     rank=rank, failure=failure, sums=sums, rowcnt=rowcnt, s=s)
        else:
            tlx_s3split.boundary(ws.partial, stream, pre, alpha, beta, out, updated, ws,
                                 rank=rank, failure=failure, sums=sums, s=s)
        return out, updated
    return types.SimpleNamespace(attention_tail=attention_tail)


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--left", type=int, default=-1)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--s", type=int, default=8)
    p.add_argument("--one", action="store_true")
    p.add_argument("--fused", action="store_true", help="MK.4 tlx_s3fused (archive-exact)")
    p.add_argument("--nb", type=int, default=2)
    p.add_argument("--s4", action="store_true", help="MK.5a: S4 projection inside the fused launch")
    p.add_argument("--warm", action="store_true")
    p.add_argument("--nb4", type=int, default=2)
    p.add_argument("--s4x", action="store_true")
    p.add_argument("--rows4", action="store_true")
    p.add_argument("--warm4", action="store_true")
    p.add_argument("--sleep4", type=int, default=8)
    p.add_argument("--s4q", action="store_true", help="MK.9: S4 as 288 LDS-resident quarter tiles")
    a = p.parse_args()
    copies, retained = [], []
    for ci in range(3):
        base, native, *rest = setup(a.context, a.left)
        retained.append((base, native, rest))
        x, stream, w, m, _ = base["chained"][1].args
        old = gpu(torch.load(Path("/home/bangtliu/coord/phase44-baseline")
                             / f"layer-7-{a.context}-{a.left}.pt", weights_only=True)["rows"][0])
        row = {}
        for name in ("tlx", "s3n"):
            ws = tlx_chain.make_workspace("cuda", context=a.context, left=a.left)
            prep = functools.partial(tlx_chain.prepare_peers, ws, *native.partials, rank=0)
            if name == "tlx":
                fn = functools.partial(tlx_chain.launch, x, stream, w, m, ws, rank=0)
                check = functools.partial(archive_check, native.f, ws, old, 0, digest=False)
            else:
                sums = torch.zeros(8 * a.s * 4, device="cuda", dtype=torch.float32)
                rowcnt = torch.zeros(8 * 32, device="cuda", dtype=torch.int32) if a.one else None
                shim = split_tail(sums, a.s, rowcnt)
                if a.fused:
                    st4 = tlx_s3fused.State("cuda")
                    retained.append(st4)
                    dbg = None
                    if os.environ.get("DBG_HOST") and ci == 0:
                        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                        from hostmem import HostShared
                        dh = HostShared("muse_dbg_gpu0", 512 * 16 * 4, 0, lambda: None, keep=True)
                        dh.zero()
                        dbg = dh.tensor(0, 512 * 16)
                        retained.append(dh)

                    s4args = (w["joint_router_pooler"], ws.s4.projection, ws.s4.producer) if a.s4 else None
                    rows4 = (w["bias"], w["route_norm"], ws.s4.scores, ws.s4.ids, ws.s4.pooled,
                             ws.s4.rows_ready) if a.rows4 else None

                    def at(x, gate, o_proj, stream, pre, alpha, beta, *, rank, workspace,
                           store_reduced, split_norm, output, failure, st4=st4, s4args=s4args,
                           rows4=rows4, dbg=dbg, **_):
                        return tlx_s3fused.tail(x, gate, o_proj, stream, pre, alpha, beta,
                                                rank=rank, ws=workspace, st=st4, output=output,
                                                failure=failure, split_norm=split_norm,
                                                store_reduced=store_reduced, nb=a.nb, s4=s4args,
                                                warm=a.warm, nb4=a.nb4, s4x=a.s4x, rows4=rows4,
                                                warm4=a.warm4, s4q=a.s4q, dbg=dbg,
                                                sleep4=a.sleep4)
                    shim = types.SimpleNamespace(attention_tail=at)
                retained.append((sums, rowcnt))

                def routing_rows_only(x4, weight, bias, norm, *, rank=0, workspace=None, failure=None, **_):
                    W = workspace
                    failure = W.local_failure if failure is None else failure
                    if not a.rows4:
                        tlx_route.finish_rows[(8,)](W.projection, bias, norm, W.scores, W.ids, W.pooled,
                                                    W.producer, W.rows_ready, failure, num_warps=4,
                                                    enable_fp_fusion=False)
                    tlx_route.slots[(64,)](W.ids, W.sorted_ids, W.expert_ids, W.num_post, W.rows_ready,
                                           W.state.status, failure, LOCAL_START=rank * 64,
                                           GRID=W.state.grid, num_warps=4, enable_fp_fusion=False)
                    return W.scores, W.ids, W.pooled, W.sorted_ids, W.expert_ids, W.num_post
                shim4 = types.SimpleNamespace(routing=routing_rows_only) if a.s4 else None

                def fn(x=x, stream=stream, w=w, m=m, ws=ws, shim=shim, shim4=shim4):
                    old3, old4 = tlx_chain.tlx_attn_tail, tlx_chain.tlx_route
                    tlx_chain.tlx_attn_tail = shim
                    if shim4 is not None:
                        tlx_chain.tlx_route = shim4
                    try:
                        return tlx_chain.launch(x, stream, w, m, ws, rank=0)
                    finally:
                        tlx_chain.tlx_attn_tail, tlx_chain.tlx_route = old3, old4
            if (os.environ.get("TIMEOUT_FIRST") or os.environ.get("STAGE_SYNC")) and name != "tlx":
                # no peer prep: the S3 exchange times out (as under cross-rank skew)
                import sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages as stg
                saved = {}
                for modn, fnn in (("tlx_qkvg", "qkvg"), ("tlx_attention", "attention"),
                                  ("tlx_experts", "s5"), ("tlx_tail", "unpool_boundary")):
                    mod = getattr(stg, modn)
                    orig = getattr(mod, fnn)
                    saved[(mod, fnn)] = orig
                    def wrapped(*aa, _o=orig, _n=fnn, **kk):
                        r = _o(*aa, **kk)
                        torch.cuda.synchronize()
                        print("STAGE done", _n, ws.failure.tolist(), flush=True)
                        return r
                    setattr(mod, fnn, wrapped)
                at0 = shim.attention_tail
                def at_w(*aa, **kk):
                    r = at0(*aa, **kk)
                    torch.cuda.synchronize()
                    print("STAGE done s3fused", ws.failure.tolist(), flush=True)
                    return r
                shim.attention_tail = at_w
                if shim4 is not None:
                    r40 = shim4.routing
                    def r4_w(*aa, **kk):
                        r = r40(*aa, **kk)
                        torch.cuda.synchronize()
                        print("STAGE done routing", ws.failure.tolist(), flush=True)
                        return r
                    shim4.routing = r4_w
                if os.environ.get("STAGE_SYNC"):
                    prep()
                    torch.cuda.synchronize()
                print("launching", flush=True)
                fn()
                torch.cuda.synchronize()
                print("TIMEOUT_FIRST survived", ws.failure.tolist(), flush=True)
                for (mod, fnn), orig in saved.items():
                    setattr(mod, fnn, orig)
                shim.attention_tail = at0
                if shim4 is not None:
                    shim4.routing = r40
                ws.failure.zero_()
            prep()
            fn()
            torch.cuda.synchronize()
            if name != "tlx" and a.fused and a.warm:
                tlx_s3fused.refresh_code_range(st4)
                prep(); fn(); torch.cuda.synchronize()
            if name == "tlx" or a.fused:
                check = functools.partial(archive_check, native.f, ws, old, 0, digest=False)
                check()
            else:
                if ci == 0:
                    ref = build(native.f, w, 7, a.context, a.left, expert_fp64=True)
                    ref["_rank"] = 0
                    res = compare(native.f, ws, ref, w)
                    print("NUMERIC", json.dumps({k: v for k, v in res.items() if k != "cache_boundaries"}, default=str)[:1500], flush=True)
                first = {k: t.clone() for k, t in ws.buffers.items()}
                ids0 = ws.s4.ids.clone()

                def check(ws=ws, first=first, ids0=ids0):
                    assert int(ws.failure[0]) == 0, ("failure", ws.failure.tolist())
                    assert torch.equal(ws.s4.ids, ids0), "routing changed across replays"
                    for k, t in first.items():
                        assert torch.equal(ws.buffers[k], t), ("replay differs", k)
            prep()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            row[name] = (prep, g, check)
        copies.append(row)
    times = {"tlx": [], "s3n": []}
    for r in range(a.rounds):
        order = ("tlx", "s3n") if r % 2 == 0 else ("s3n", "tlx")
        for c in copies:
            for name in order:
                prep, g, check = c[name]
                prep()
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record(); g.replay(); e1.record()
                torch.cuda.synchronize()
                times[name].append(e0.elapsed_time(e1) * 1000)
                check()
    med = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
    print(json.dumps(dict(context=a.context, left=a.left, s=a.s, one=a.one, fused=a.fused, s4=a.s4, warm=a.warm, nb4=a.nb4, s4x=a.s4x, rows4=a.rows4, warm4=a.warm4, median_us=med,
                          ratio=med["s3n"] / med["tlx"], n=len(times["tlx"]))), flush=True)


if __name__ == "__main__":
    main()
