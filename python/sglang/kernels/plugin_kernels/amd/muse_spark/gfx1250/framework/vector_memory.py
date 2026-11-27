"""Agent-scoped b128 batches with declared contiguous physical register operands."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.constexpr_function
def _load_asm(world, pack):
    # Four fixed output registers form each ISA b128 tuple. Early clobbers keep
    # input addresses disjoint; no unadvertised physical register is modified.
    constraints = []
    for word in range(4 * world):
        src, component = divmod(word, 4)
        for p in range(pack):
            constraints.append(f"=&{{v{4 + (p * world + src) * 4 + component}}}")
    constraints += ["=&s"] * pack
    outputs = (4 * world + 1) * pack
    constraints += ["v"] * ((world + 1) * pack)
    constraints += ["~{scc}"]
    asm = []
    for reg in range(4, 4 + 4 * world * pack):
        asm.append(f"v_mov_b32 v{reg}, 0")
    for p in range(pack):
        saved = 4 * world * pack + p
        mask = outputs + world * pack + p
        asm += [
            f"v_cmp_ne_u32_e64 ${saved}, 0, ${mask}",
            f"s_and_saveexec_b32 ${saved}, ${saved}",
        ]
        for src in range(world):
            reg = 4 + (p * world + src) * 4
            ptr = outputs + src * pack + p
            asm.append(
                f"global_load_b128 v[{reg}:{reg + 3}], ${ptr}, off scope:SCOPE_DEV"
            )
        asm.append(f"s_mov_b32 exec_lo, ${saved}")
    asm += ["s_wait_loadcnt 0", "s_wait_xcnt 0x0"]
    return "\n".join(asm), ",".join(constraints)


@gluon.constexpr_function
def _store_asm(world, pack, common):
    packets = 1 if common else world
    constraints = ["=&s"] * pack
    for packet in range(packets):
        for component in range(4):
            for p in range(pack):
                constraints.append(f"{{v{4 + (p * packets + packet) * 4 + component}}}")
    constraints += ["v"] * ((world + 1) * pack) + ["~{scc}"]
    asm = []
    for p in range(pack):
        mask = pack + 4 * packets * pack + world * pack + p
        asm += [f"v_cmp_ne_u32_e64 ${p}, 0, ${mask}", f"s_and_saveexec_b32 ${p}, ${p}"]
        for dst in range(world):
            ptr = pack + 4 * packets * pack + dst * pack + p
            reg = 4 + (p * packets + (0 if common else dst)) * 4
            asm.append(
                f"global_store_b128 ${ptr}, v[{reg}:{reg + 3}], off scope:SCOPE_DEV"
            )
        asm.append(f"s_mov_b32 exec_lo, ${p}")
    # Keep every fixed input register live until every store has consumed it.
    asm += ["s_wait_storecnt 0", "s_wait_xcnt 0x0"]
    return "\n".join(asm), ",".join(constraints)


@gluon.jit
def load_sources(
    base, mask, STRIDE: gl.constexpr, WORLD: gl.constexpr, PACK: gl.constexpr
):
    """Issue all source b128 loads for the block before one load-counter wait."""
    gl.static_assert(WORLD == 4 or WORLD == 8)
    gl.static_assert(4 + 4 * WORLD * PACK <= 128)
    asm: gl.constexpr = _load_asm(WORLD, PACK)
    if WORLD == 4:
        args = [
            base,
            base + STRIDE,
            base + 2 * STRIDE,
            base + 3 * STRIDE,
            mask.to(gl.int32),
        ]
    else:
        args = [
            base,
            base + STRIDE,
            base + 2 * STRIDE,
            base + 3 * STRIDE,
            base + 4 * STRIDE,
            base + 5 * STRIDE,
            base + 6 * STRIDE,
            base + 7 * STRIDE,
            mask.to(gl.int32),
        ]
    return gl.inline_asm_elementwise(
        asm[0],
        asm[1],
        args,
        dtype=(gl.int32,) * (4 * WORLD + 1),
        is_pure=False,
        pack=PACK,
    )


@gluon.jit
def store_destinations(
    pointers,
    values,
    mask,
    WORLD: gl.constexpr,
    PACK: gl.constexpr,
    COMMON: gl.constexpr = False,
):
    """Issue all destination stores before one drain, preserving packet register lifetimes."""
    gl.static_assert(WORLD == 4 or WORLD == 8)
    gl.static_assert(4 + 4 * (1 if COMMON else WORLD) * PACK <= 128)
    asm: gl.constexpr = _store_asm(WORLD, PACK, COMMON)
    return gl.inline_asm_elementwise(
        asm[0],
        asm[1],
        values + pointers + (mask.to(gl.int32),),
        dtype=gl.int32,
        is_pure=False,
        pack=PACK,
    )


@gluon.jit
def split4(values, Q: gl.constexpr):
    matrix = gl.permute(gl.reshape(values, (Q, 2, 2)), (0, 2, 1))
    left, right = gl.split(matrix)
    a, b = gl.split(left)
    c, d = gl.split(right)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    return (
        gl.convert_layout(a, layout),
        gl.convert_layout(b, layout),
        gl.convert_layout(c, layout),
        gl.convert_layout(d, layout),
    )


@gluon.jit
def join4(a, b, c, d, Q: gl.constexpr):
    matrix = gl.permute(gl.join(gl.join(a, b), gl.join(c, d)), (0, 2, 1))
    return gl.convert_layout(
        gl.reshape(matrix, (4 * Q,)), gl.BlockedLayout([4], [32], [gl.num_warps()], [0])
    )


@gluon.jit
def join8(a, b, c, d, e, f, g, h, Q: gl.constexpr):
    matrix = gl.join(
        gl.join(gl.join(a, b), gl.join(c, d)), gl.join(gl.join(e, f), gl.join(g, h))
    )
    return gl.convert_layout(
        gl.reshape(gl.permute(matrix, (0, 3, 2, 1)), (8 * Q,)),
        gl.BlockedLayout([8], [32], [gl.num_warps()], [0]),
    )


@gluon.jit
def split_world(values, N: gl.constexpr, WORLD: gl.constexpr):
    """Extract source columns held in registers from a single batched input load."""
    if WORLD == 4:
        even, odd = gl.split(gl.reshape(values, (N, 2, 2)))
        v0, v2 = gl.split(even)
        v1, v3 = gl.split(odd)
        return v0, v1, v2, v3
    else:
        even, odd = gl.split(gl.reshape(values, (N, 2, 2, 2)))
        v04, v26 = gl.split(even)
        v15, v37 = gl.split(odd)
        v0, v4 = gl.split(v04)
        v2, v6 = gl.split(v26)
        v1, v5 = gl.split(v15)
        v3, v7 = gl.split(v37)
        return v0, v1, v2, v3, v4, v5, v6, v7
