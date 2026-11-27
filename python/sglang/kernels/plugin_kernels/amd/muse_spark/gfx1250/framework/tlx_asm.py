"""Pure constexpr ASM emitters; preserved scope and XCNT strings."""

import triton


@triton.constexpr_function
def _load_asm(world, pack, scope="SCOPE_DEV"):
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
                f"global_load_b128 v[{reg}:{reg + 3}], ${ptr}, off scope:{scope}"
            )
        asm.append(f"s_mov_b32 exec_lo, ${saved}")
    asm += ["s_wait_loadcnt 0", "s_wait_xcnt 0x0"]
    return ("\n".join(asm), ",".join(constraints))


@triton.constexpr_function
def _store_asm(world, pack, common, scope="SCOPE_DEV"):
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
                f"global_store_b128 ${ptr}, v[{reg}:{reg + 3}], off scope:{scope}"
            )
        asm.append(f"s_mov_b32 exec_lo, ${p}")
    asm += ["s_wait_storecnt 0", "s_wait_xcnt 0x0"]
    return ("\n".join(asm), ",".join(constraints))
