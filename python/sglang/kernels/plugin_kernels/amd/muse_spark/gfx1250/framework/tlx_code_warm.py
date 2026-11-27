"""Warm a launch's own code into every XCD's L2 (TLX port of TokenSpeed's
gfx950 code_warm). Large fused kernels run latency-bound paths once per CTA;
without this every instruction fetch of those paths misses to memory.

The kernel reads its program counter (s_getpc_b64 next to a marker literal);
the host finds the marker in the compiled .text (text_range) and passes the
PC offset and the .text size as runtime arguments.
"""

import struct

import triton
import triton.language as tl

MARKER = 0x5EC0DE01
XCDS = tl.constexpr(8)
# int32 words per load. One load should cover a CTA's whole share: the loop below runs its
# iterations one after another (MK.94), so a small block means several memory round trips.
WARM_BLOCK = tl.constexpr(512)


@triton.jit
def code_warm(blk, NCTA: tl.constexpr, Sink, offset, quads):
    """CTA ``blk`` of ``NCTA`` reads its XCD-group share of the kernel .text.
    Call once per kernel (the marker must occur once)."""
    pc, _ = tl.inline_asm_elementwise(
        "s_getpc_b64 $0\ns_mov_b32 $1, 0x5ec0de01",
        "=s,=s",
        [],
        dtype=(tl.int64, tl.int32),
        is_pure=False,
        pack=1,
    )
    code = (pc - offset).to(tl.pointer_type(tl.int32), bitcast=True)
    BLOCK: tl.constexpr = WARM_BLOCK
    i = tl.arange(0, BLOCK)
    PARTS: tl.constexpr = NCTA // XCDS
    words = quads * 4
    share = (words + PARTS * BLOCK - 1) // (PARTS * BLOCK) * BLOCK
    first = (blk // XCDS) * share
    acc = tl.zeros([BLOCK], tl.int32)
    for k in range(0, share, BLOCK):
        o = first + k + i
        acc = acc ^ tl.load(code + o, mask=o < words, other=0)
    tl.store(Sink + i, acc, mask=(i < 16) & (quads < 0))


_RANGES = {}


def text_range(kernel):
    """(offset, quads) of a Triton CompiledKernel for code_warm."""
    got = _RANGES.get(id(kernel))
    if got is not None:
        return got[0]
    elf = kernel.asm["hsaco"]
    (shoff,) = struct.unpack_from("<Q", elf, 0x28)
    entsize, count, names_index = struct.unpack_from("<HHH", elf, 0x3A)
    sections = [struct.unpack_from("<IIQQQQIIQQ", elf, shoff + k * entsize) for k in range(count)]
    names = sections[names_index][4]

    def name(at):
        return elf[names + at: elf.index(b"\0", names + at)].decode()

    text = next(s for s in sections if name(s[0]) == ".text")
    body = elf[text[4]: text[4] + text[5]]
    marker = struct.pack("<I", MARKER)
    if body.count(marker) != 1:
        raise RuntimeError(f"code warm: marker occurs {body.count(marker)} times")
    got = (body.index(marker) - 4, text[5] // 16)
    _RANGES[id(kernel)] = (got, kernel)
    return got
