"""T10 canonical expert K512 loads with the original K128 compute sequence."""

from unittest.mock import patch

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_expert_load_apad as apad_core,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_expert_load_k512 as core,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_experts as base,
)

make_workspace = base.make_workspace
Workspace = base.Workspace
_original = base.s5
_original_phase = base.expert_phase
TARGET = True
TDM = False
COMPUTE_K = 128
DEPTH = 2
APAD = False


class Phase:
    def prepare(self, args, options):
        options = dict(options)
        prior_wide = options.pop("WIDE", None)
        if options["FIRST"] != TARGET:
            if prior_wide is not None:
                options["WIDE"] = prior_wide
            return _original_phase, args, options
        options = dict(
            options,
            MODE="full",
            PACK_W=False,
            PACK_S=False,
            DEPTH=DEPTH,
            COMPUTE_K=COMPUTE_K,
            PREFETCH_FIRST=True,
            SCALE_ASYNC=True,
            TDM=TDM,
            waves_per_eu=2,
        )
        return (apad_core.expert_phase if APAD else core.expert_phase), args, options

    def warmup(self, *args, **options):
        fn, args, options = self.prepare(args, options)
        kernel = fn.warmup(*args, **options)
        kernel_occupancy("T10-canonical", kernel)
        assert kernel.n_spills == 0, ("canonical spill", kernel.n_spills)
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
