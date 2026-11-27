"""Host-readable bounded-poll diagnostics; trapping is explicitly opt-in."""

from collections.abc import Sequence

import msgspec
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped, sync

RECORD_WORDS = 8


class StuckRecord(msgspec.Struct, frozen=True):
    tag: int
    address: int
    observed: int
    clock: int
    rank: int
    program_id: int
    stage: int


def decode_record(words: Sequence[int]) -> StuckRecord:
    """Decode a published record after kernel completion or isolated process exit."""
    if len(words) != RECORD_WORDS or words[0] != 1:
        raise ValueError("stuck record is incomplete")
    return StuckRecord(*map(int, words[1:]))


@gluon.jit
def wait_or_record(
    flag,
    tag,
    timeout_ticks,
    record,
    rank,
    stage,
    TRAP: gl.constexpr = False,
    MAX_SPINS: gl.constexpr = 65536,
):
    """Publish a system-visible failure record before optionally trapping this CTA."""
    seen, polls = scoped.wait_flag(flag, tag, timeout_ticks, MAX_SPINS=MAX_SPINS)
    if seen != tag:
        gl.store(record + 1, gl.cast(tag, gl.int64), cache_modifier=".wt")
        gl.store(record + 2, flag.to(gl.int64), cache_modifier=".wt")
        gl.store(record + 3, seen.to(gl.int64), cache_modifier=".wt")
        gl.store(record + 4, sync.clock().to(gl.int64), cache_modifier=".wt")
        gl.store(record + 5, rank.to(gl.int64), cache_modifier=".wt")
        gl.store(record + 6, gl.program_id(0).to(gl.int64), cache_modifier=".wt")
        gl.store(record + 7, stage.to(gl.int64), cache_modifier=".wt")
        scoped.drain_stores()
        gl.store(record, 1, cache_modifier=".wt")
        scoped.drain_stores()
        if TRAP:
            sync.trap()
    return seen == tag
