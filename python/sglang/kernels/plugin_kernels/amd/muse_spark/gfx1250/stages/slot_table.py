"""One-wave routed-slot ordering, boundary ballots and padded-prefix allocation."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped


@g.jit
def ballot(value):
    return gl.inline_asm_elementwise(
        "v_cmp_ne_u32_e64 $0, $1, 0",
        "=s,v",
        [value.to(gl.int32)],
        dtype=gl.uint32,
        is_pure=True,
        pack=1,
    )


@g.jit
def table(Ids, Sorted, Experts, NumPost, LOCAL_START: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 2], [1, 32], [4, 1], [1, 0])
    wave = gl.arange(0, 4, layout=gl.SliceLayout(1, layout))
    j = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
    active = wave[:, None] == 0
    raw = (
        scoped.load_words_masked(Ids + j[None, :], active & (j[None, :] >= 0))
        - LOCAL_START
    )
    valid = (raw >= 0) & (raw < 64)
    keys = gl.where(valid, raw, 64) * 64 + j[None, :]
    # Stable integer keys keep the original slot order inside each expert.
    for level in gl.static_range(1, 7):
        for step in gl.static_range(level - 1, -1, -1):
            other = gl.gather(keys, (j[None, :] ^ (1 << step)).broadcast_to((4, 64)), 1)
            low = gl.minimum(keys, other)
            high = gl.maximum(keys, other)
            keys = gl.where(
                (((j & (1 << level)) == 0) == ((j & (1 << step)) == 0))[None, :],
                low,
                high,
            )
    expert = keys // 64
    previous = gl.gather(expert, gl.maximum(j[None, :] - 1, 0).broadcast_to((4, 64)), 1)
    boundary = (j[None, :] == 0) | (expert != previous)
    even, odd = gl.split(boundary.reshape((4, 32, 2)))
    even_mask = ballot(even)
    odd_mask = ballot(odd)
    em = gl.join(even_mask, even_mask).reshape((4, 64))
    om = gl.join(odd_mask, odd_mask).reshape((4, 64))
    lane = j[None, :] // 2
    through = gl.full_like(keys, 0xFFFFFFFF, dtype=gl.uint32) >> (31 - lane)
    before = (gl.full_like(keys, 1, dtype=gl.uint32) << lane) - 1
    em &= through
    om &= gl.where(j[None, :] % 2 == 0, before, through)
    epos = gl.where(
        em != 0,
        31
        - gl.inline_asm_elementwise(
            "v_ffbh_u32_e32 $0, $1", "=v,v", [em], dtype=gl.int32, is_pure=True, pack=1
        ),
        -1,
    )
    opos = gl.where(
        om != 0,
        31
        - gl.inline_asm_elementwise(
            "v_ffbh_u32_e32 $0, $1", "=v,v", [om], dtype=gl.int32, is_pure=True, pack=1
        ),
        -1,
    )
    start = gl.maximum(epos * 2, opos * 2 + 1)
    rank = j[None, :] - start
    next_e = gl.gather(expert, gl.minimum(j[None, :] + 1, 63).broadcast_to((4, 64)), 1)
    end = (j[None, :] == 63) | (expert != next_e)
    padded = gl.where(end & (expert < 64), gl.cdiv(rank + 1, 16) * 16, 0)
    prefix = gl.cumsum(padded, 1)
    base = prefix - padded
    scoped.store_words(Sorted + base + rank, keys % 64, active & (expert < 64))
    scoped.store_words(
        Experts + base // 16 + rank // 16,
        expert,
        active & (expert < 64) & (rank % 16 == 0),
    )
    total = gl.max(prefix, 1)
    scoped.store_words(
        NumPost + gl.full_like(keys, 0),
        total[:, None] + gl.full_like(keys, 0),
        active & (j[None, :] == 0),
    )
