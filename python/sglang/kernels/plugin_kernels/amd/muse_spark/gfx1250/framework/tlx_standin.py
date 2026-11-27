"""One-real-rank, four-virtual-rank scoped row exchange in tl/uTLX.

With MUSE_PEER_HOST=1 the exchange instead runs between real ranks through one
shared pinned host region (hipHostRegister, SCOPE_SYS, no GPU peer access):
each rank writes its partial once and reads all four. `Peers[0]` is the host base.
Host layout (int32 words): payload [parity][src][4*WIDTH], then flags [src][GRID][32].
"""

import os

import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    tlx_memory as mem,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_asm import (
    _load_asm,
    _store_asm,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.tlx_layouts import (
    ROW1,
    ROW4,
)


HOST = tl.constexpr(os.environ.get("MUSE_PEER_HOST") == "1")
# Waits of the HOST-MEMORY exchange (the 4-GPU path). Policy since MK.105 (user's decision
# 2026-11-26, after sgl-amd-mono #172): a deadline in seconds, and a wait that passes it stops the
# launch with a store to address 0 instead of letting the step go on without the peer's words.
# A pause of a few ms (a KFD queue eviction, MK.102) is then a slow step, not a wrong one; only a
# peer that never arrives ends the process. Ticks are 100 MHz. The server sets the deadline from
# the plugin's own (half of --watchdog-timeout, at most 60 s) through set_host_wait_seconds();
# MUSE_HOST_WAIT_TICKS overrides it, MUSE_HOST_WAIT_STOP=0 restores "give up and report failure".
# Read at compile time: set before the first launch.
HOST_WAIT_TICKS = tl.constexpr(int(os.environ.get("MUSE_HOST_WAIT_TICKS", "500000000")))
HOST_WAIT_SPINS = tl.constexpr(4096 if HOST_WAIT_TICKS.value <= 100_000 else 2_000_000_000)
HOST_WAIT_STOP = tl.constexpr(os.environ.get("MUSE_HOST_WAIT_STOP", "1") == "1")
# Waits of the single-GPU stand-in exchange (simulated peers; the missing-producer tests rely on
# this bound and on a reported failure): 1 ms, at most 4096 polls.
STANDIN_WAIT_TICKS = tl.constexpr(100_000)
STANDIN_WAIT_SPINS = tl.constexpr(4096)


def set_host_wait_seconds(seconds: float) -> None:
    """Set the host-exchange deadline unless MUSE_HOST_WAIT_TICKS fixed it. Before any launch."""
    global HOST_WAIT_TICKS, HOST_WAIT_SPINS
    if "MUSE_HOST_WAIT_TICKS" not in os.environ:
        HOST_WAIT_TICKS = tl.constexpr(int(seconds * 100_000_000))
        HOST_WAIT_SPINS = tl.constexpr(2_000_000_000)  # the deadline ends the wait first


def host_words(width, grid):
    return 32 * width + 4 * grid * 32


@triton.jit
def _load_sys_masked(ptr, mask):
    saved_exec, value = tl.inline_asm_elementwise(
        "v_mov_b32 $1, 0\nv_cmp_ne_u32_e64 $0, 0, $3\ns_and_saveexec_b32 $0, $0\nglobal_load_b32 $1, $2, off scope:SCOPE_SYS\ns_wait_loadcnt 0\ns_mov_b32 exec_lo, $0\ns_wait_xcnt 0x0",
        constraints="=&s,=&v,v,v,~{scc}",
        args=[tlx.release_layout(ptr), tlx.release_layout(mask.to(tl.int32))],
        dtype=(tl.int32, tl.int32),
        is_pure=False,
        pack=1,
    )
    return tlx.require_layout(value, ROW1)


@triton.jit
def _store_sys_masked(ptr, values, mask):
    return tl.inline_asm_elementwise(
        "v_cmp_ne_u32_e64 $0, 0, $3\ns_and_saveexec_b32 $0, $0\nglobal_store_b32 $1, $2, off scope:SCOPE_SYS\ns_mov_b32 exec_lo, $0\ns_wait_xcnt 0x0",
        constraints="=&s,v,v,v,~{scc}",
        args=[ptr, values, mask.to(tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def clock():
    return tl.inline_asm_elementwise(
        "s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)\ns_wait_kmcnt 0",
        "=s",
        [],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def advance(Epochs, row):
    i = tlx.require_layout(tl.arange(0, 128), ROW1)
    old = tl.atomic_add(
        Epochs + row * 32 + tl.full((128,), 0, tl.int32),
        1,
        mask=i == 0,
        sem="relaxed",
        scope="gpu",
    )
    return tl.sum(tl.where(i == 0, old + 1, 0), 0)


@triton.jit
def pair(a, b, word):
    lo = (word & 65535).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    hi = (word >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    return a + lo, b + hi


@triton.jit
def reduce_row(
    Partial,
    Scratch,
    Peers,
    Reduced,
    tag,
    row,
    GRID: tl.constexpr,
    RANK: tl.constexpr,
    PAYLOAD: tl.constexpr,
    FLAGS: tl.constexpr,
    WIDTH: tl.constexpr = 8192,
    KEEP_CARRIER: tl.constexpr = False,
    STORE_REDUCED: tl.constexpr = True,
    SPLIT: tl.constexpr = 0,
    XStamps=None,
):
    Q: tl.constexpr = WIDTH // 8
    q = tlx.require_layout(tl.arange(0, Q), ROW1)
    at = row * (WIDTH // 2) + q * 4
    v = mem.load_contiguous_words(
        Partial.to(tl.pointer_type(tl.int32)) + RANK * 4 * WIDTH + row * (WIDTH // 2),
        WIDTH // 2,
        ROW4,
    )
    a, b, c, d = mem.split4(v, Q)
    if HOST:
        return _host_exchange(a, b, c, d, Peers, Reduced, tag, row, at, GRID, RANK, WIDTH,
                              STORE_REDUCED, SPLIT, XStamps)
    pointers = ()
    for dst in tl.static_range(4):
        peer = tl.load(Peers + dst).to(tl.pointer_type(tl.int32))
        pointers += (
            peer
            + PAYLOAD[0]
            + dst * PAYLOAD[1]
            + (tag % 2) * PAYLOAD[2]
            + RANK * 4 * WIDTH
            + at,
        )
    spec: tl.constexpr = _store_asm(4, 4, True)
    mask = tlx.require_layout(tl.full((Q,), 1, tl.int32), ROW1)
    tl.inline_asm_elementwise(
        spec[0],
        spec[1],
        [a, b, c, d, pointers[0], pointers[1], pointers[2], pointers[3], mask],
        dtype=tl.int32,
        is_pure=False,
        pack=4,
    )
    mem.drain_stores()
    i = tlx.require_layout(tl.arange(0, 128), ROW1)
    dstmask = i < 4
    peer = tl.load(Peers + i, dstmask, 0).to(tl.pointer_type(tl.int32))
    offset = FLAGS[0] + i * FLAGS[1] + (tag % 2) * FLAGS[2] + (RANK * GRID + row) * 32
    mem.store_words(peer + offset, tl.full((128,), tag, tl.int32), dstmask)
    ptr = (
        Scratch
        + FLAGS[0]
        + RANK * FLAGS[1]
        + (tag % 2) * FLAGS[2]
        + (i * GRID + row) * 32
    )
    start = clock()
    seen = mem.load_words_masked(ptr, dstmask)
    ready = tl.sum(((seen == tag) & dstmask).to(tl.int32), 0) == 4
    spins = 1
    elapsed = start - start
    while (~ready) & (spins < STANDIN_WAIT_SPINS) & (elapsed < STANDIN_WAIT_TICKS):
        seen = mem.load_words_masked(ptr, dstmask)
        ready = tl.sum(((seen == tag) & dstmask).to(tl.int32), 0) == 4
        spins += 1
        if spins % 64 == 0:
            elapsed = clock() - start
    mem.drain_stores()
    base = Scratch + PAYLOAD[0] + RANK * PAYLOAD[1] + (tag % 2) * PAYLOAD[2] + at
    spec2: tl.constexpr = _load_asm(4, 4)
    if KEEP_CARRIER:
        raw = tl.inline_asm_elementwise(
            spec2[0],
            spec2[1],
            [base, base + 4 * WIDTH, base + 8 * WIDTH, base + 12 * WIDTH, mask],
            dtype=(tl.int32,) * 17,
            is_pure=False,
            pack=4,
        )
    else:
        raw = tl.inline_asm_elementwise(
            spec2[0],
            spec2[1],
            [
                tlx.release_layout(base),
                tlx.release_layout(base + 4 * WIDTH),
                tlx.release_layout(base + 8 * WIDTH),
                tlx.release_layout(base + 12 * WIDTH),
                tlx.release_layout(mask),
            ],
            dtype=(tl.int32,) * 17,
            is_pure=False,
            pack=4,
        )
    z = tlx.require_layout(tl.full((Q,), 0, tl.float32), ROW1)
    a, b, c, d, e, f, g, h = z, z, z, z, z, z, z, z
    for src in tl.static_range(4):
        w0 = tlx.require_layout(raw[src * 4], ROW1)
        w1 = tlx.require_layout(raw[src * 4 + 1], ROW1)
        w2 = tlx.require_layout(raw[src * 4 + 2], ROW1)
        w3 = tlx.require_layout(raw[src * 4 + 3], ROW1)
        a, b = pair(a, b, w0)
        c, d = pair(c, d, w1)
        e, f = pair(e, f, w2)
        g, h = pair(g, h, w3)
    value = mem.join8(a, b, c, d, e, f, g, h, Q)
    j = tl.arange(0, WIDTH)
    if STORE_REDUCED:
        tl.store(Reduced + RANK * 8 * WIDTH + row * WIDTH + j, value)
    return value, ready


@triton.jit
def publish_part(Partial, Peers, tag, row, part, PARTS: tl.constexpr, GRID: tl.constexpr,
                 RANK: tl.constexpr, WIDTH: tl.constexpr = 8192):
    """Sibling of a SPLIT host exchange: publish part `part` (1..PARTS-1) of this
    rank's row and raise flag word part * 32 // PARTS. Same words and addresses as
    the full publish."""
    Q: tl.constexpr = WIDTH // 8
    q = tlx.require_layout(tl.arange(0, Q), ROW1)
    at = row * (WIDTH // 2) + q * 4
    v = mem.load_contiguous_words(
        Partial.to(tl.pointer_type(tl.int32)) + RANK * 4 * WIDTH + row * (WIDTH // 2),
        WIDTH // 2,
        ROW4,
    )
    a, b, c, d = mem.split4(v, Q)
    host = tl.load(Peers).to(tl.pointer_type(tl.int32))
    pay = host + (tag % 2) * 16 * WIDTH
    flags = host + 32 * WIDTH
    mask = (q // (Q // PARTS) == part).to(tl.int32)
    spec: tl.constexpr = _store_asm(1, 4, True, "SCOPE_SYS")
    tl.inline_asm_elementwise(
        spec[0], spec[1], [a, b, c, d, pay + RANK * 4 * WIDTH + at, mask],
        dtype=tl.int32, is_pure=False, pack=4,
    )
    mem.drain_stores()
    i = tlx.require_layout(tl.arange(0, 128), ROW1)
    _store_sys_masked(flags + (RANK * GRID + row) * 32 + part * (32 // PARTS) + i * 0,
                      tl.full((128,), tag, tl.int32), i == 0)


# host all-reduce: take this rank's own partial from registers, not from host memory
HOST_SKIP_SELF = tl.constexpr(True)


@triton.jit
def _host_exchange(a, b, c, d, Peers, Reduced, tag, row, at, GRID: tl.constexpr,
                   RANK: tl.constexpr, WIDTH: tl.constexpr, STORE_REDUCED: tl.constexpr,
                   SPLIT: tl.constexpr = 0, XStamps=None):
    """SPLIT = parts per row (0: whole row): this CTA publishes part 0; sibling CTAs
    (publish_part) publish parts 1.. with flags in word part * 32 // SPLIT of the
    same flag slot. The reduce below is unchanged (own words from registers).
    XStamps (probe, int64 [8 rows, 8]): realtime at entry, after the publish and own flag,
    after every flag was seen, after the peers' payloads were read; then the poll count."""
    XT: tl.constexpr = XStamps is not None
    x0 = clock() if XT else 0
    Q: tl.constexpr = WIDTH // 8
    host = tl.load(Peers).to(tl.pointer_type(tl.int32))
    pay = host + (tag % 2) * 16 * WIDTH
    flags = host + 32 * WIDTH
    mask = tlx.require_layout(tl.full((Q,), 1, tl.int32), ROW1)  # also the read mask below
    if SPLIT:
        qh = tlx.require_layout(tl.arange(0, Q), ROW1)
        publish = (qh < Q // SPLIT).to(tl.int32)
    else:
        publish = mask
    spec: tl.constexpr = _store_asm(1, 4, True, "SCOPE_SYS")
    tl.inline_asm_elementwise(
        spec[0], spec[1], [a, b, c, d, pay + RANK * 4 * WIDTH + at, publish],
        dtype=tl.int32, is_pure=False, pack=4,
    )
    mem.drain_stores()
    i = tlx.require_layout(tl.arange(0, 128), ROW1)
    _store_sys_masked(flags + (RANK * GRID + row) * 32 + i * 0, tl.full((128,), tag, tl.int32), i == 0)
    if SPLIT:
        # every rank's part-0 flag + the 3 peers' other-part flags (own parts never read)
        ptr = flags + ((i % 4) * GRID + row) * 32 + (i // 4 % SPLIT) * (32 // SPLIT)
        dstmask = (i < 4) | ((i < 4 * SPLIT) & (i % 4 != RANK))
        need: tl.constexpr = 4 + 3 * (SPLIT - 1)
    else:
        ptr = flags + ((i % 4) * GRID + row) * 32
        dstmask = i < 4
        need: tl.constexpr = 4
    start = clock()
    seen = _load_sys_masked(ptr, dstmask)
    ready = tl.sum(((seen == tag) & dstmask).to(tl.int32), 0) == need
    spins = 1
    elapsed = start - start
    while (~ready) & (spins < HOST_WAIT_SPINS) & (elapsed < HOST_WAIT_TICKS):
        seen = _load_sys_masked(ptr, dstmask)
        ready = tl.sum(((seen == tag) & dstmask).to(tl.int32), 0) == need
        spins += 1
        if spins % 64 == 0:
            elapsed = clock() - start
    if HOST_WAIT_STOP:
        # past the deadline: stop this launch (and with it the process) with a store to address
        # 0, on the polling lanes only; nothing below may run on words that did not come
        late = dstmask & (~ready)
        nowhere = (i * 0).to(tl.int64).to(tl.pointer_type(tl.int32), bitcast=True)
        _store_sys_masked(nowhere, tl.full((128,), 0x0DEAD11E, tl.int32), late)
    x2 = clock() if XT else 0
    base = pay + at
    own0, own1, own2, own3 = a, b, c, d
    if HOST_SKIP_SELF:
        # this rank's own words are already in registers (the exact bits it wrote):
        # read only the 3 peers' payloads from host memory; same sum order below
        o0: tl.constexpr = 0 + (RANK <= 0)
        o1: tl.constexpr = 1 + (RANK <= 1)
        o2: tl.constexpr = 2 + (RANK <= 2)
        spec3: tl.constexpr = _load_asm(3, 4, "SCOPE_SYS")
        raw = tl.inline_asm_elementwise(
            spec3[0], spec3[1],
            [
                tlx.release_layout(base + o0 * 4 * WIDTH),
                tlx.release_layout(base + o1 * 4 * WIDTH),
                tlx.release_layout(base + o2 * 4 * WIDTH),
                tlx.release_layout(mask),
            ],
            dtype=(tl.int32,) * 13, is_pure=False, pack=4,
        )
    else:
        spec2: tl.constexpr = _load_asm(4, 4, "SCOPE_SYS")
        raw = tl.inline_asm_elementwise(
            spec2[0], spec2[1],
            [
                tlx.release_layout(base),
                tlx.release_layout(base + 4 * WIDTH),
                tlx.release_layout(base + 8 * WIDTH),
                tlx.release_layout(base + 12 * WIDTH),
                tlx.release_layout(mask),
            ],
            dtype=(tl.int32,) * 17, is_pure=False, pack=4,
        )
    if XT:
        x3 = clock()
        xk = tl.arange(0, 8)
        tl.store(XStamps + row * 8 + xk,
                 tl.where(xk == 0, x0, tl.where(xk == 1, start, tl.where(xk == 2, x2, tl.where(
                     xk == 3, x3, spins.to(tl.uint64))))).to(tl.int64), xk < 5)
    z = tlx.require_layout(tl.full((Q,), 0, tl.float32), ROW1)
    a, b, c, d, e, f, g, h = z, z, z, z, z, z, z, z
    for src in tl.static_range(4):
        if HOST_SKIP_SELF and src == RANK:
            w0 = tlx.require_layout(own0, ROW1)
            w1 = tlx.require_layout(own1, ROW1)
            w2 = tlx.require_layout(own2, ROW1)
            w3 = tlx.require_layout(own3, ROW1)
        else:
            # index of src among the loaded sources (own source skipped)
            w0 = tlx.require_layout(raw[(src - (HOST_SKIP_SELF and src > RANK)) * 4], ROW1)
            w1 = tlx.require_layout(raw[(src - (HOST_SKIP_SELF and src > RANK)) * 4 + 1], ROW1)
            w2 = tlx.require_layout(raw[(src - (HOST_SKIP_SELF and src > RANK)) * 4 + 2], ROW1)
            w3 = tlx.require_layout(raw[(src - (HOST_SKIP_SELF and src > RANK)) * 4 + 3], ROW1)
        a, b = pair(a, b, w0)
        c, d = pair(c, d, w1)
        e, f = pair(e, f, w2)
        g, h = pair(g, h, w3)
    value = mem.join8(a, b, c, d, e, f, g, h, Q)
    j = tl.arange(0, WIDTH)
    if STORE_REDUCED:
        tl.store(Reduced + RANK * 8 * WIDTH + row * WIDTH + j, value)
    return value, ready
