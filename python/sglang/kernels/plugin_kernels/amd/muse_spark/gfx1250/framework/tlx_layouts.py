"""Layout descriptions for the installed uTLX layout_encoding extension point.

No Gluon device operation or alternate compiler is used. The pinned stock
Triton builder already exposes these encoding factories, even though uTLX has
no public convenience WMMA/blocked constructor. Keep this ABI covered by IR
checks against the installed compiler revision.
"""

import triton.language as tl
import utlx_plugin as tlx


class WmmaEncoding(tlx.layout_encoding):
    """WMMA v3, warps along N: 2 warps (default) or 4 (warps=4)."""

    def __init__(self, operand=-1, warps=2):
        self.operand = operand
        self.warps = warps

    def __repr__(self):
        if self.warps == 2:
            return f"WMMA3_N2_operand{self.operand}"
        return f"WMMA3_N{self.warps}_operand{self.operand}"

    def __eq__(self, other):
        return (isinstance(other, WmmaEncoding) and self.operand == other.operand
                and self.warps == other.warps)

    def __hash__(self):
        return hash((WmmaEncoding, self.operand, self.warps))

    def to_ir(self, builder, *_):
        bases = [[0, 1]] if self.warps == 2 else [[0, 1], [0, 2]]
        parent = builder.get_amd_wmma_layout(3, True, bases, [], [], [16, 16, 32], 2)
        return (
            parent
            if self.operand < 0
            else builder.get_dot_operand_layout(self.operand, parent, 8)
        )


class BlockedEncoding(tlx.layout_encoding):
    def __init__(self, size, threads, warps, order):
        self.size = tuple(size)
        self.threads = tuple(threads)
        self.warps = tuple(warps)
        self.order = tuple(order)

    def __repr__(self):
        return f"Blocked{self.size}_{self.threads}_{self.warps}_{self.order}"

    def __eq__(self, other):
        return isinstance(other, BlockedEncoding) and (
            self.size,
            self.threads,
            self.warps,
            self.order,
        ) == (other.size, other.threads, other.warps, other.order)

    def __hash__(self):
        return hash((BlockedEncoding, self.size, self.threads, self.warps, self.order))

    def to_ir(self, builder, *_):
        return builder.get_blocked_layout(
            list(self.size), list(self.threads), list(self.warps), list(self.order), []
        )


WMMA = tl.constexpr(WmmaEncoding())
WMMA_A = tl.constexpr(WmmaEncoding(0))
WMMA_B = tl.constexpr(WmmaEncoding(1))
WMMA4 = tl.constexpr(WmmaEncoding(-1, 4))
WMMA4_A = tl.constexpr(WmmaEncoding(0, 4))
WMMA4_B = tl.constexpr(WmmaEncoding(1, 4))
ROW1 = tl.constexpr(BlockedEncoding([1], [32], [4], [0]))
ROW2 = tl.constexpr(BlockedEncoding([2], [32], [4], [0]))
ROW4 = tl.constexpr(BlockedEncoding([4], [32], [4], [0]))
ROW8 = tl.constexpr(BlockedEncoding([8], [32], [4], [0]))


def require_pin_support():
    """Reject the older plugin that silently ignores pin=True."""
    from triton._C.libtriton import passes

    if not hasattr(passes.plugin, "utlx_lower_pinned_layout"):
        raise RuntimeError(
            "Real layout pins require the patched uTLX image local/muse-mono-gfx1250:utlx-pin"
        )
