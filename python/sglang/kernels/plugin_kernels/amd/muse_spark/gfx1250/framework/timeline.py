"""Optional private 256-byte per-CTA realtime trace, absent in normal binaries."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import sync


@g.jit
def stamp(Trace, pid, SLOT: gl.constexpr):
    if Trace is not None:
        gl.store(Trace + pid * 32 + SLOT, sync.clock())
