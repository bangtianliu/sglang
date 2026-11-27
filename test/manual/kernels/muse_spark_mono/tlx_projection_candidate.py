"""Private stage wrapper for T10 projection copy/buffer variants."""

from unittest.mock import patch

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.launch import (
    kernel_occupancy,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    tlx_attn_tail,
    tlx_projection_pipeline,
    tlx_qkvg,
    tlx_route,
    tlx_tail,
)


class Project:
    def __init__(self, depth, copy):
        self.depth, self.copy = depth, copy

    def options(self, kw):
        return dict(kw, DEPTH=self.depth, COPY=self.copy)

    def warmup(self, *args, **kw):
        kernel = tlx_projection_pipeline.project.warmup(*args, **self.options(kw))
        kernel_occupancy("T10-projection", kernel)
        assert kernel.n_spills == 0, ("T10 projection spills", kernel.n_spills)
        return kernel

    def __getitem__(self, grid):
        def call(*args, **kw):
            return tlx_projection_pipeline.project[grid](*args, **self.options(kw))

        return call


def configure(module, *, stage, depth, copy="tdm"):
    base, api = {
        1: (tlx_qkvg, "qkvg"),
        3: (tlx_attn_tail, "attention_tail"),
        4: (tlx_route, "routing"),
        6: (tlx_tail, "unpool_boundary"),
    }[stage]
    original = getattr(base, api)
    proxy = Project(depth, copy)

    def invoke(*args, **kw):
        with patch.object(base.tlx_projection, "project", proxy):
            return original(*args, **kw)

    module.make_workspace = base.make_workspace
    if hasattr(base, "Workspace"):
        module.Workspace = base.Workspace
    setattr(module, api, invoke)
