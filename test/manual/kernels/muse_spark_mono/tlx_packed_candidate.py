"""Private T10 W13 packing adapter; preserve the current W2 and norm path."""

from unittest.mock import patch

import torch

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_expert_load_k256 as packed,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_experts as base,
)

make_workspace = base.make_workspace
Workspace = base.Workspace
_original = base.s5
_original_phase = base.expert_phase
_cache = {}


def pack(value, scale=False):
    key = (id(value), value._version, scale)
    if key not in _cache:
        chunk = 8 if scale else 128
        source = value.reshape(64, 128, 64, 16, chunk)
        result = source.permute(0, 1, 3, 2, 4).contiguous()
        assert torch.equal(result.permute(0, 1, 3, 2, 4).contiguous(), source), (
            "W13 startup permutation"
        )
        _cache[key] = (value, result.reshape(value.shape))
    return _cache[key][1]


class Phase:
    def prepare(self, args, options):
        options = dict(options)
        wide = options.pop("WIDE", None)
        if not options["FIRST"]:
            if wide is not None:
                options["WIDE"] = wide
            return _original_phase, args, options
        args = list(args)
        args[4] = pack(args[4])
        args[5] = pack(args[5], scale=True)
        options = dict(
            options,
            MODE="full",
            PACK_W=True,
            PACK_S=True,
            DEPTH=2,
            PREFETCH_FIRST=True,
            SCALE_ASYNC=True,
            waves_per_eu=3,
        )
        return packed.expert_phase, tuple(args), options

    def warmup(self, *args, **options):
        fn, args, options = self.prepare(args, options)
        kernel = fn.warmup(*args, **options)
        kernel_occupancy("T10-packed-W13", kernel)
        assert kernel.n_spills == 0, ("packed W13 spill", kernel.n_spills)
        return kernel

    def __getitem__(self, grid):
        def call(*args, **options):
            fn, args, options = self.prepare(args, options)
            return fn[grid](*args, **options)

        return call


_phase = Phase()


def s5(*args, **options):
    with patch.object(base, "expert_phase", _phase):
        return _original(*args, **options)
