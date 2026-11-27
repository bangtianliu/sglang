"""Shared pinned host memory for the multi-GPU harnesses: now in the plugin framework."""
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.host_shared import (  # noqa: F401
    HostShared,
    _wrap,
)
