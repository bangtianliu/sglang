"""gfx1250 agent-coherent word accesses without whole-cache maintenance.

Every cross-CTA payload access must use these scoped instructions. This is not
an acquire/release replacement for ordinary cached Gluon loads and stores.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import sync


@gluon.jit
def store_words(ptr, values, mask):
    """Protect store operands with XCNT; caller drains stores before publication."""
    return gl.inline_asm_elementwise(
        "v_cmp_ne_u32_e64 $0, 0, $3\n"
        "s_and_saveexec_b32 $0, $0\n"
        "global_store_b32 $1, $2, off scope:SCOPE_DEV\n"
        "s_mov_b32 exec_lo, $0\ns_wait_xcnt 0x0",
        constraints="=&s,v,v,v,~{scc}",
        args=[ptr, values, mask.to(gl.int32)],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def load_words(ptr):
    """Complete 32-bit agent-visible loads before returning, bypassing lower caches."""
    return gl.inline_asm_elementwise(
        "global_load_b32 $0, $1, off scope:SCOPE_DEV\ns_wait_loadcnt 0\ns_wait_xcnt 0x0",
        constraints="=v,v",
        args=[ptr],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def load_flag(flag):
    """Load one scoped word in lane zero and broadcast its value through CTA reduction."""
    lanes = gl.arange(
        0,
        gl.num_warps() * 32,
        layout=gl.BlockedLayout([1], [32], [gl.num_warps()], [0]),
    )
    saved_exec, value = gl.inline_asm_elementwise(
        "v_mov_b32 $1, 0\n"
        "v_cmp_ne_u32_e64 $0, 0, $3\n"
        "s_and_saveexec_b32 $0, $0\n"
        "global_load_b32 $1, $2, off scope:SCOPE_DEV\n"
        "s_wait_loadcnt 0\n"
        "s_mov_b32 exec_lo, $0\ns_wait_xcnt 0x0",
        constraints="=&s,=&v,v,v,~{scc}",
        args=[flag + gl.full_like(lanes, 0), (lanes == 0).to(gl.int32)],
        dtype=(gl.int32, gl.int32),
        is_pure=False,
        pack=1,
    )
    return gl.sum(value, 0)


@gluon.jit
def drain_stores():
    """Wait for this wave's scoped stores, then join every wave in the CTA."""
    gl.inline_asm_elementwise(
        "s_wait_storecnt 0\nv_mov_b32 $0, 0",
        "=v",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    gl.barrier()


@gluon.jit
def publish(flag, tag):
    """Drain all scoped payload stores in the CTA before issuing one scoped flag."""
    drain_stores()
    lanes = gl.arange(
        0,
        gl.num_warps() * 32,
        layout=gl.BlockedLayout([1], [32], [gl.num_warps()], [0]),
    )
    store_words(flag + gl.full_like(lanes, 0), gl.full_like(lanes, tag), lanes == 0)


@gluon.jit
def wait_flag(
    flag,
    tag,
    timeout_ticks,
    MAX_SPINS: gl.constexpr = 4096,
    CLOCK_EVERY: gl.constexpr = 64,
    SHADER_CLOCK: gl.constexpr = False,
):
    """Wait for an exact tag; bound every poll and check the selected clock at the given cadence."""
    gl.static_assert(CLOCK_EVERY > 0 and CLOCK_EVERY & (CLOCK_EVERY - 1) == 0)
    start = sync.shader_cycles() if SHADER_CLOCK else sync.clock()
    seen = load_flag(flag)
    spins = 1
    elapsed = start - start
    if CLOCK_EVERY == 1:
        elapsed = sync.clock() - start
    while (seen != tag) & (spins < MAX_SPINS) & (elapsed < timeout_ticks):
        seen = load_flag(flag)
        spins += 1
        if spins % CLOCK_EVERY == 0:
            now = sync.shader_cycles() if SHADER_CLOCK else sync.clock()
            elapsed = now - start
    gl.barrier()
    return seen, spins


@gluon.constexpr_function
def _width(grid):
    return 1 << (grid - 1).bit_length()


@gluon.jit
def arrive_and_wait(
    flags, tag, timeout_ticks, GRID: gl.constexpr, MAX_SPINS: gl.constexpr = 4096
):
    """Publish scoped arrivals and wait for all CTAs, with bounded polling."""
    publish(flags + gl.program_id(0), tag)
    ids = gl.arange(
        0, _width(GRID), layout=gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    )
    # Pad the allocation to _width(GRID); padded values are excluded from readiness.
    start = sync.clock()
    seen = load_words(flags + ids)
    ready = gl.sum(((seen == tag) | (ids >= GRID)).to(gl.int32), 0) == _width(GRID)
    spins = 1
    elapsed = sync.clock() - start
    while (~ready) & (spins < MAX_SPINS) & (elapsed < timeout_ticks):
        seen = load_words(flags + ids)
        ready = gl.sum(((seen == tag) | (ids >= GRID)).to(gl.int32), 0) == _width(GRID)
        spins += 1
        elapsed = sync.clock() - start
    gl.barrier()
    return ready, spins, 0


@gluon.jit
def counter_barrier(
    storage,
    tag,
    timeout_ticks,
    GRID: gl.constexpr,
    GROUP: gl.constexpr,
    MAX_SPINS: gl.constexpr = 4096,
    CLOCK_EVERY: gl.constexpr = 64,
    SPREAD_FLAGS: gl.constexpr = True,
):
    """Drain scoped payloads, count arrivals with relaxed atomics, then scope the flag."""
    gl.static_assert(GRID % GROUP == 0)
    groups: gl.constexpr = GRID // GROUP
    pid = gl.program_id(0)
    drain_stores()
    ticket = gl.atomic_add(storage + (pid // GROUP) * 32, 1, sem="relaxed", scope="gpu")
    if ticket + 1 == tag * GROUP:
        if groups == 1:
            if SPREAD_FLAGS:
                publish_flags(storage + (groups + 1) * 32, tag, GRID)
            else:
                publish(storage + (groups + 1) * 32, tag)
        else:
            top_ticket = gl.atomic_add(
                storage + groups * 32, 1, sem="relaxed", scope="gpu"
            )
            if top_ticket + 1 == tag * groups:
                if SPREAD_FLAGS:
                    publish_flags(storage + (groups + 1) * 32, tag, GRID)
                else:
                    publish(storage + (groups + 1) * 32, tag)
    offset = (groups + 1) * 32
    if SPREAD_FLAGS:
        offset += pid * 32
    seen, polls = wait_flag(
        storage + offset,
        tag,
        timeout_ticks,
        MAX_SPINS=MAX_SPINS,
        CLOCK_EVERY=CLOCK_EVERY,
    )
    return seen == tag, polls, 0


@gluon.jit
def publish_flags(flags, tag, GRID: gl.constexpr):
    """Publish one completion tag per CTA, on separate 128-byte lines."""
    drain_stores()
    ids = gl.arange(
        0, _width(GRID), layout=gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    )
    store_words(flags + ids * 32, gl.full_like(ids, tag), ids < GRID)


@gluon.jit
def load_words_masked(ptr, mask):
    """Complete masked agent-visible word loads, returning zero for inactive lanes."""
    saved_exec, value = gl.inline_asm_elementwise(
        "v_mov_b32 $1, 0\n"
        "v_cmp_ne_u32_e64 $0, 0, $3\n"
        "s_and_saveexec_b32 $0, $0\n"
        "global_load_b32 $1, $2, off scope:SCOPE_DEV\n"
        "s_wait_loadcnt 0\n"
        "s_mov_b32 exec_lo, $0\ns_wait_xcnt 0x0",
        constraints="=&s,=&v,v,v,~{scc}",
        args=[ptr, mask.to(gl.int32)],
        dtype=(gl.int32, gl.int32),
        is_pure=False,
        pack=1,
    )
    return value


@gluon.jit
def arrive_epoch(Counter, tag, expected):
    """Count variable producers using one tagged64-bit atomic word; no flag polling."""
    ptr = Counter.to(gl.pointer_type(gl.uint64))
    base = tag.to(gl.uint64) << 32
    gl.atomic_max(ptr, base, sem="relaxed", scope="gpu")
    ticket = gl.atomic_add(ptr, 1, sem="relaxed", scope="gpu")
    return (ticket & 0xFFFFFFFF) + 1 == expected
