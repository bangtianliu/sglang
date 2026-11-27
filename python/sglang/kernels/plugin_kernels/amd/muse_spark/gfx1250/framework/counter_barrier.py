"""Agent-scope arrival counters with a release flag on a separate cache line."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import sync


@gluon.jit
def counter_barrier(
    storage,
    tag,
    timeout_ticks,
    GRID: gl.constexpr,
    GROUP: gl.constexpr,
    SLEEP_EVERY: gl.constexpr = 0,
    POLL_SEM: gl.constexpr = "relaxed",
    MAX_SPINS: gl.constexpr = 1_000_000,
):
    """Publish CTA arrivals through acq_rel counters and acquire the completion tag."""
    gl.static_assert(GRID % GROUP == 0)
    groups: gl.constexpr = GRID // GROUP
    pid = gl.program_id(0)
    group = pid // GROUP
    counter = storage + group * 32
    top = storage + groups * 32
    flag = storage + (groups + 1) * 32
    gl.barrier()
    ticket = gl.atomic_add(counter, 1, sem="acq_rel", scope="gpu")
    if ticket + 1 == tag * GROUP:
        if groups == 1:
            gl.atomic_xchg(flag, tag, sem="release", scope="gpu")
        else:
            leader_ticket = gl.atomic_add(top, 1, sem="acq_rel", scope="gpu")
            if leader_ticket + 1 == tag * groups:
                gl.atomic_xchg(flag, tag, sem="release", scope="gpu")
    start = sync.clock()
    elapsed = start - start
    seen = gl.atomic_add(flag, 0, sem=POLL_SEM, scope="gpu")
    polls = 1
    sleeps = 0
    while (seen != tag) & (polls < MAX_SPINS) & (elapsed < timeout_ticks):
        if SLEEP_EVERY > 0:
            if polls % SLEEP_EVERY == 0:
                sync.backoff()
                sleeps += 1
        seen = gl.atomic_add(flag, 0, sem=POLL_SEM, scope="gpu")
        polls += 1
        if polls % 64 == 0:
            elapsed = sync.clock() - start
    ready = seen == tag
    if POLL_SEM != "acquire":
        seen = sync.load_acquire(flag)
    gl.barrier()
    return ready, polls, sleeps
