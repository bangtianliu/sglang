"""Phase4.10 choices validated before adoption; retain for explicit controls."""

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
    independent_attn_tail,
    independent_experts,
    independent_qkvg,
)


def configure(ws, device, context, left, *, wide_s1_128=False):
    if (context, left) == (8192, 2047) or (
        wide_s1_128 and (context, left) == (131072, 2047)
    ):
        ws.s1 = independent_qkvg.make_workspace(device)
        ws.projection_backend = "wide"
    if (context, left) == (131072, 2047):
        ws.s3 = independent_attn_tail.make_workspace(device)
        ws.stage_policy = ("wide", *ws.stage_policy[1:])
    if (context, left) == (8192, 2047):
        ws.s5 = independent_experts.make_workspace(device)
        ws.stage_policy = (ws.stage_policy[0], "k9", ws.stage_policy[2])
    return ws
