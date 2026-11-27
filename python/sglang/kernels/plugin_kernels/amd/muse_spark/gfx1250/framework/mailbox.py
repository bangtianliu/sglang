"""Tagged agent-scope mailboxes; peer pointers are data, never mapped here."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped


@gluon.jit
def put_words(base, region_offset, indices, words, mask):
    """Issue scoped payload stores; publish with raise_flag after every producer wave."""
    scoped.store_words(base + region_offset + indices, words, mask)


@gluon.jit
def raise_flag(base, flag_offset, tag):
    """Drain the CTA's scoped payload before publishing this exact launch tag."""
    scoped.publish(base + flag_offset, tag)


@gluon.jit
def wait_flag(base, flag_offset, tag, timeout_ticks, MAX_SPINS: gl.constexpr = 4096):
    """Return the observed tag after bounded polling; callers must test equality."""
    seen, polls = scoped.wait_flag(
        base + flag_offset, tag, timeout_ticks, MAX_SPINS=MAX_SPINS
    )
    return seen


@gluon.jit
def put_words_peer(peer_base, region_offset, indices, words, mask):
    """Issue scoped payload stores to the supplied peer pointer plus word offset."""
    put_words(peer_base, region_offset, indices, words, mask)


@gluon.jit
def raise_flag_peer(peer_base, flag_offset, tag):
    """Publish to a peer pointer only after this CTA's scoped payload stores drain."""
    raise_flag(peer_base, flag_offset, tag)


@gluon.jit
def wait_flag_peer(
    peer_base, flag_offset, tag, timeout_ticks, MAX_SPINS: gl.constexpr = 4096
):
    """Boundedly poll a caller-supplied peer pointer; no peer-access API is invoked."""
    return wait_flag(peer_base, flag_offset, tag, timeout_ticks, MAX_SPINS=MAX_SPINS)
