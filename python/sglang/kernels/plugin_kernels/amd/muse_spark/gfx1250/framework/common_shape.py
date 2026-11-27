"""Resource admission for the common four-wave per-rank stage launch."""

import msgspec
import torch

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)


class CommonShape(msgspec.Struct, frozen=True):
    waves: int = 4
    k: int = 4
    vgpr_cap: int = 224
    lds_cap: int = 72 * 1024

    def __post_init__(self):
        if self.waves != 4 or self.k < 1 or self.vgpr_cap < 1 or self.lds_cap < 1:
            raise ValueError("four waves and positive resource limits required")

    def grid(self, device):
        # Stages synchronize in groups of 12 CTAs (barrier counters at GRID // 12), so the
        # resident grid is rounded down to a multiple of 12: 768 on 192 CUs, 1020 on 256.
        cus = torch.cuda.get_device_properties(device).multi_processor_count
        return self.k * cus // 12 * 12

    def options(self):
        return dict(
            num_warps=self.waves,
            waves_per_eu=self.k,
            enable_fp_fusion=False,
            llvm_fn_attrs=f"amdgpu-num-vgpr={self.vgpr_cap}",
        )

    def admit(self, name, kernel, *, grid):
        assert 0 < grid <= self.grid(torch.cuda.current_device())
        resource = kernel_occupancy(name, kernel)
        assert resource.waves == self.waves
        assert resource.vgprs <= self.vgpr_cap, (name, resource)
        assert resource.lds_bytes <= self.lds_cap, (name, resource)
        assert resource.active_ctas_per_cu >= self.k, (name, resource)
        return resource
