"""Static 32-bit scratch regions for local and virtual-peer monokernel stages."""

import msgspec


class RegionRequest(msgspec.Struct, frozen=True):
    name: str
    words: int
    cross_rank: bool = True


class Region(msgspec.Struct, frozen=True):
    name: str
    words: int
    base: int
    rank_stride: int
    parity_stride: int
    ranks: int
    copies: int

    def offset(self, rank: int, launch: int) -> int:
        """Return a word offset; parity isolates adjacent launches for peer regions."""
        if not 0 <= rank < self.ranks or launch < 0:
            raise ValueError("invalid scratch rank or launch")
        return (
            self.base
            + rank * self.rank_stride
            + (launch % self.copies) * self.parity_stride
        )


class StaticPlan(msgspec.Struct, frozen=True):
    ranks: int
    regions: tuple[Region, ...]
    total_words: int

    def region(self, name: str) -> Region:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(name)


def make_plan(ranks: int, requests: tuple[RegionRequest, ...]) -> StaticPlan:
    """Lay out every rank on 128-byte boundaries, with two copies of peer regions."""
    if ranks < 1:
        raise ValueError("ranks must be positive")
    regions = []
    names = set()
    cursor = 0
    for request in requests:
        if not request.name or request.name in names or request.words < 1:
            raise ValueError("scratch regions need unique names and positive sizes")
        names.add(request.name)
        stride = (request.words + 31) // 32 * 32
        copies = 2 if request.cross_rank else 1
        regions.append(
            Region(
                name=request.name,
                words=request.words,
                base=cursor,
                rank_stride=stride,
                parity_stride=stride * ranks,
                ranks=ranks,
                copies=copies,
            )
        )
        cursor += copies * ranks * stride
    return StaticPlan(ranks=ranks, regions=tuple(regions), total_words=cursor)
