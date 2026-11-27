"""Bounded persistent-grid barrier; the caller owns the co-residency guarantee."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import sync


@gluon.constexpr_function
def _poll_width(grid):
    return 1 << (grid - 1).bit_length()


@gluon.jit
def arrive_and_wait(
    flags,
    tag,
    timeout_ticks,
    GRID: gl.constexpr,
    MAX_SPINS: gl.constexpr = 1_000_000,
    SLEEP_EVERY: gl.constexpr = 1,
    POLL_SEM: gl.constexpr = "acquire",
    RETURN_STATS: gl.constexpr = False,
):
    """Release this CTA's tag and acquire all grid arrivals, or return false."""
    sync.publish(flags + gl.program_id(0), tag)
    indices = gl.arange(
        0,
        _poll_width(GRID),
        layout=gl.BlockedLayout([1], [32], [gl.num_warps()], [0]),
    )
    seen = gl.atomic_add(
        flags + indices, 0, mask=indices < GRID, sem=POLL_SEM, scope="gpu"
    )
    ready = gl.sum(((seen == tag) | (indices >= GRID)).to(gl.int32), 0) == _poll_width(
        GRID
    )
    start = sync.clock()
    elapsed = start - start
    spins = 0
    sleeps = 0
    while (~ready) & (spins < MAX_SPINS) & (elapsed < timeout_ticks):
        if SLEEP_EVERY > 0:
            if spins % SLEEP_EVERY == SLEEP_EVERY - 1:
                sync.backoff()
                sleeps += 1
        seen = gl.atomic_add(
            flags + indices, 0, mask=indices < GRID, sem=POLL_SEM, scope="gpu"
        )
        ready = gl.sum(
            ((seen == tag) | (indices >= GRID)).to(gl.int32), 0
        ) == _poll_width(GRID)
        spins += 1
        if spins % 64 == 0:
            elapsed = sync.clock() - start
    if POLL_SEM != "acquire":
        gl.atomic_add(
            flags + indices, 0, mask=indices < GRID, sem="acquire", scope="gpu"
        )
    gl.barrier()
    if RETURN_STATS:
        return ready, spins + 1, sleeps
    else:
        return ready
