"""Probe: realtime stamps inside a captured step (MK.74). One CTA per stamp.

``stamp[(1,)](Buf, Count, SLOT=k, N=n)`` writes the GPU realtime clock (100 ticks per µs) to
``Buf[(step % RING) * n + k]``; SLOT 0 starts a new step (increments ``Count``)."""

import triton
import triton.language as tl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_standin import clock

RING = 64


@triton.jit
def stamp(Buf, Count, SLOT: tl.constexpr, N: tl.constexpr, RING: tl.constexpr):
    c = tl.load(Count)
    if SLOT == 0:
        c = c + 1
        tl.store(Count, c)
    tl.store(Buf + (c % RING) * N + SLOT, clock().to(tl.int64))


class KernelStamps:
    """Probe: stamp after every Triton kernel launched inside ``with ks.layer():`` (one slot per
    kernel, in launch order; slot 0 = before the first). A side thread prints the median
    per-kernel time over the ring every 10 s as one JSON line."""

    N = 32

    def __init__(self, device, label):
        import threading

        import torch

        self.label = label
        self.buf = torch.zeros(RING * self.N, device=device, dtype=torch.int64)
        self.count = torch.zeros(1, device=device, dtype=torch.int64)
        self.names = []
        threading.Thread(target=self._report, daemon=True).start()

    def _mark(self, run, slot):
        run(stamp, self.buf, self.count, SLOT=slot, N=self.N, RING=RING, grid=(1,),
            warmup=False, num_warps=1)

    def layer(self):
        import contextlib

        from triton.runtime.jit import JITFunction

        ks = self

        @contextlib.contextmanager
        def scope():
            orig = JITFunction.run
            names = []

            def run(fn, *args, grid, warmup, **kwargs):
                r = orig(fn, *args, grid=grid, warmup=warmup, **kwargs)
                if not warmup and fn is not stamp and len(names) < ks.N - 1:
                    names.append(fn.fn.__name__)
                    ks._mark(orig, len(names))
                return r

            ks._mark(orig, 0)
            JITFunction.run = run
            try:
                yield
            finally:
                JITFunction.run = orig
                ks.names = names

        return scope()

    def _report(self):
        import json
        import time

        last = -1
        while True:
            time.sleep(10)
            n = int(self.count.item())
            k = len(self.names)
            if n == last or n < RING + 2 or not k:
                continue
            last = n
            v = self.buf.clone().cpu().reshape(RING, self.N).double()[:, : k + 1] / 100
            v = v[v[:, 0].argsort()][2:-2]
            d = (v[:, 1:] - v[:, :-1]).median(0).values
            print(json.dumps(dict(event="kernel_stamps", label=self.label, steps=n,
                                  layer_us=round(float((v[:, k] - v[:, 0]).median()), 1),
                                  kernels=[[a, round(float(b), 1)] for a, b in zip(self.names, d)])),
                  flush=True)
