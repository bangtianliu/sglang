"""WIP small-CTA geometry helpers; no production launcher adopts them yet.

A grid's guarantee applies only to its measured stage resource set. Validate
co-residency using the actual stage device functions before adopting a geometry;
the existing16-wave stages still require separate four-wave ports.
"""

import ctypes

import msgspec


class StageOccupancy(msgspec.Struct, frozen=True):
    name: str
    waves: int
    vgprs: int
    lds_bytes: int
    active_ctas_per_cu: int


class ResidentGrid(msgspec.Struct, frozen=True):
    cu_count: int
    waves: int
    ctas_per_cu: int
    stages: tuple[StageOccupancy, ...]

    @property
    def grid(self):
        return self.cu_count * self.ctas_per_cu

    def check_stage(self, stage):
        if stage.waves != self.waves or stage.active_ctas_per_cu < self.ctas_per_cu:
            raise ValueError(f"{stage.name} does not fit the validated resident grid")


def grid_from_occupancy(cu_count, waves, stages):
    stages = tuple(stages)
    if cu_count < 1 or waves not in (4, 8) or not stages:
        raise ValueError("positive CU count,4/8 waves and measured stages required")
    if any(s.waves != waves or s.active_ctas_per_cu < 1 for s in stages):
        raise ValueError("all stages must share the CTA size and be resident")
    return ResidentGrid(
        cu_count, waves, min(s.active_ctas_per_cu for s in stages), stages
    )


def kernel_occupancy(name, kernel):
    """Query HIP module occupancy for this compiled kernel's exact VGPR/LDS usage."""
    kernel._init_handles()
    from triton.backends.amd.driver import _get_path_to_hip_runtime_dylib

    hip = ctypes.CDLL(_get_path_to_hip_runtime_dylib())
    query = hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
    query.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    blocks = ctypes.c_int()
    waves = kernel.metadata.num_warps
    # Triton supplies this LDS allocation as dynamic shared memory at launch.
    result = query(
        ctypes.byref(blocks),
        ctypes.c_void_p(kernel.function),
        waves * 32,
        kernel.metadata.shared,
    )
    if result:
        raise RuntimeError(f"HIP module occupancy query failed: {result}")
    return StageOccupancy(
        name, waves, kernel.n_regs, kernel.metadata.shared, blocks.value
    )
