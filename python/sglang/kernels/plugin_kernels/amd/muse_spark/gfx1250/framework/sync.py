"""gfx1250 synchronization built from stock Triton Gluon operations.

Every consumer must poll its own 128-byte cache line. Producers fan out flags
or use a tree; many CTAs must never poll one shared line. Shared arrival counters
using atomics are allowed. This applies to mailboxes and virtual-peer collectives.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def clock():
    """Read the device realtime counter, shared across CUs and advancing while idle."""
    return gl.inline_asm_elementwise(
        "s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)\ns_wait_kmcnt 0",
        constraints="=s",
        args=[],
        dtype=gl.uint64,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def shader_cycles():
    """Read this wave's shader-cycle counter; compare deltas only within one CTA."""
    return gl.inline_asm_elementwise(
        "s_get_shader_cycles_u64 $0",
        constraints="=s",
        args=[],
        dtype=gl.uint64,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def backoff():
    """Throttle one polling wave with s_sleep; this is not a memory fence."""
    return gl.inline_asm_elementwise(
        "s_sleep 1\nv_mov_b32 $0, 0",
        constraints="=v",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def load_acquire(flag, SCOPE: gl.constexpr = "gpu"):
    """Acquire a 32-bit publication flag at agent scope, or explicit system scope."""
    return gl.atomic_add(flag, 0, sem="acquire", scope=SCOPE)


@gluon.jit
def publish(flag, tag, SCOPE: gl.constexpr = "gpu"):
    """Join the CTA's payload stores and release its tag at the selected scope."""
    gl.barrier()
    gl.atomic_xchg(flag, tag, sem="release", scope=SCOPE)


@gluon.jit
def wait_flag(
    flag,
    tag,
    timeout_ticks,
    MAX_SPINS: gl.constexpr = 1_000_000,
    SCOPE: gl.constexpr = "gpu",
):
    """Acquire exactly this tag with bounded spins/backoff, then join the CTA."""
    start = clock()
    elapsed = gl.full(
        (), 0, gl.uint64, gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    )
    seen = load_acquire(flag, SCOPE)
    spins = 0
    while (seen != tag) & (spins < MAX_SPINS) & (elapsed < timeout_ticks):
        backoff()
        seen = load_acquire(flag, SCOPE)
        spins += 1
        if spins % 64 == 0:
            elapsed = clock() - start
    gl.barrier()
    return seen


@gluon.jit
def trap():
    """Raise a GPU launch exception; callers must publish diagnostics first."""
    return gl.inline_asm_elementwise(
        "s_trap 2\nv_mov_b32 $0, 0",
        constraints="=v",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
