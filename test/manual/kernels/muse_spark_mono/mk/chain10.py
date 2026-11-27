"""GPU0 (simulated peers): N layers back to back in ONE graph, per-kernel stamps per layer."""
import argparse
import functools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from bench_chained47 import setup  # noqa: E402
from bench_mono import FLAGS, with_flags  # noqa: E402
from check_chained47 import archive_check  # noqa: E402
from layer_fixture import require_gpu0  # noqa: E402
from reference_archive import gpu  # noqa: E402
from triton.runtime.jit import JITFunction  # noqa: E402

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_chain  # noqa: E402


@triton.jit
def stamp(Buf, slot):
    t = tl.inline_asm_elementwise(
        "s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)\ns_wait_kmcnt 0", "=s", [],
        dtype=tl.uint64, is_pure=False, pack=1)
    tl.store(Buf + slot, t.to(tl.int64))


def main():
    require_gpu0()
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--left", type=int, default=-1)
    p.add_argument("--layers", type=int, default=10)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--variants", default="unf,f132p")
    p.add_argument("--no-stamps", action="store_true")
    a = p.parse_args()
    names = a.variants.split(",")
    base, native, *rest = setup(a.context, a.left)
    x, stream, w, m, _ = base["chained"][1].args
    old = gpu(torch.load(Path("/home/bangtliu/coord/phase44-baseline")
                         / f"layer-7-{a.context}-{a.left}.pt", weights_only=True)["rows"][0])
    orig = JITFunction.run
    built, keep = {}, [base, native, rest, old]
    for name in names:
        ws = with_flags(FLAGS[name], functools.partial(
            tlx_chain.make_workspace, "cuda", context=a.context, left=a.left))
        prep = functools.partial(tlx_chain.prepare_peers, ws, *native.partials, rank=0)
        run = functools.partial(with_flags, FLAGS[name], functools.partial(
            tlx_chain.launch, x, stream, w, m, ws, rank=0))
        check = functools.partial(archive_check, native.f, ws, old, 0, digest=False)
        prep(); run(); torch.cuda.synchronize(); check()
        buf = torch.zeros(1 << 14, device="cuda", dtype=torch.int64)
        seq = []

        def traced(self, *args, grid, warmup, _seq=seq, _buf=buf, **kwargs):
            r = orig(self, *args, grid=grid, warmup=warmup, **kwargs)
            if not warmup and self is not stamp:
                _seq.append(self.fn.__name__)
                orig(stamp, _buf, len(_seq) - 1, grid=(1,), warmup=False, num_warps=1)
            return r

        g = torch.cuda.CUDAGraph()
        if not a.no_stamps:
            JITFunction.run = traced
        try:
            with torch.cuda.graph(g):
                for _ in range(a.layers):
                    prep()
                    run()
        finally:
            JITFunction.run = orig
        built[name] = (g, check, buf, seq, ws)
        keep.append((ws, g, buf))
    times = {n: [] for n in names}
    for r in range(a.rounds):
        for name in (names if r % 2 == 0 else names[::-1]):
            g, check, *_ = built[name]
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
            times[name].append(e0.elapsed_time(e1) * 1000 / a.layers)
            check()
    med = lambda v: round(sorted(v)[len(v) // 2], 2)  # noqa: E731
    print(json.dumps(dict(context=a.context, left=a.left, layers=a.layers,
                          layer_us={n: med(times[n]) for n in names})), flush=True)
    for name in names:
        _, _, buf, seq, _ = built[name]
        if not seq:
            continue
        per = len(seq) // a.layers
        v = buf.cpu().double()
        rows = []
        for k in range(per):
            d = [float(v[layer * per + k] - v[layer * per + k - 1]) / 100
                 for layer in range(a.layers) if layer * per + k > 0]
            first = None if k == 0 else round(d[0], 2)
            later = d[1:] if k > 0 else d
            rows.append([seq[k], first, med(later)])
        print(json.dumps(dict(variant=name, per_layer=per,
                              kernels_first_layer_vs_later=rows)), flush=True)


if __name__ == "__main__":
    main()
