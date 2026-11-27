"""Per-layer operands of the gfx1250 Muse Spark kernel chain, read from the loaded model.

Every tensor is a view of what the native layer already holds (no copies). ``layer_weights``
checks each shape and dtype and reports all mismatches in one error, so a layer that cannot
use the chain is found at load, not at the first launch.
"""

import torch

HIDDEN, POOLED, INTER = 8192, 4096, 4096
LOCAL_EXPERTS, EXPERTS = 64, 256
QKVG_ROWS = 2048 + 256 + 256 + 2048  # q | k | v | output gate, this rank's slice

# name -> (shape, dtype); None dtype = checked separately
_EXPECT = {
    "qkvg": ((QKVG_ROWS, HIDDEN), torch.bfloat16),
    "o_proj": ((HIDDEN, 2048), torch.bfloat16),
    "pre3": ((HIDDEN,), torch.bfloat16),
    "alpha3": ((HIDDEN,), torch.float32),
    "beta3": ((HIDDEN,), torch.float32),
    "joint_router_pooler": ((POOLED + 512, HIDDEN), torch.bfloat16),
    "bias": ((EXPERTS,), torch.float32),
    "route_norm": ((POOLED,), None),
    "w13": ((LOCAL_EXPERTS, 2 * INTER, POOLED // 2), torch.uint8),
    "w13_scale": ((LOCAL_EXPERTS, 2 * INTER, POOLED // 32), torch.uint8),
    "w2": ((LOCAL_EXPERTS, POOLED, INTER // 2), torch.uint8),
    "w2_scale": ((LOCAL_EXPERTS, POOLED, INTER // 32), torch.uint8),
    "gamma5": ((POOLED,), None),
    "unpooler": ((HIDDEN, POOLED), torch.bfloat16),
    "post6": ((HIDDEN,), torch.bfloat16),
    "pre6": ((HIDDEN,), torch.bfloat16),
    "alpha6": ((HIDDEN,), torch.float32),
    "beta6": ((HIDDEN,), torch.float32),
}


def layer_weights(backbone, i: int) -> dict:
    """The chain's ``w`` of layer ``i`` (1 .. 60) on this rank.

    ``backbone`` is the loaded ``MuseSpark1xModel`` after ``post_load_weights`` (the fused
    boundaries hold their cached gains and gate coefficients).
    """
    if not 1 <= i <= 60:
        raise ValueError(f"layer {i}: the chain serves layers 1..60 (MXFP4 experts)")
    layer, boundary = backbone.layers[i], backbone._ffn_boundaries[i]
    attn, moe, mid = layer.self_attn, layer.mlp, layer._fused_attn_boundary
    mid._ensure_cached_weights(torch.bfloat16)
    boundary._ensure_cached_weights(torch.bfloat16)
    w = dict(
        qkvg=attn.qkv_proj.weight,
        o_proj=attn.o_proj.weight,
        pre3=mid._cc_pre_weight,
        alpha3=mid._cc_alpha,
        beta3=mid._cc_beta,
        joint_router_pooler=getattr(moe, "router_pooler_packed_weight", None),
        bias=moe.expert_bias,
        route_norm=moe.pre_expert_norm._get_effective_weight(),
        w13=moe.experts.w13_weight,
        w13_scale=moe.experts.w13_weight_scale,
        w2=moe.experts.w2_weight,
        w2_scale=moe.experts.w2_weight_scale,
        # the raw zero-centred gain: the kernel adds 1.0 itself
        gamma5=moe.post_expert_norm.weight,
        unpooler=moe.post_expert_proj.weight,
        post6=boundary._cc_post_weight,
        pre6=boundary._cc_pre_weight,
        alpha6=boundary._cc_alpha,
        beta6=boundary._cc_beta,
    )
    errors = []
    for name, (shape, dtype) in _EXPECT.items():
        t = w[name]
        if not isinstance(t, torch.Tensor):
            errors.append(f"{name}: missing")
            continue
        if tuple(t.shape) != shape:
            errors.append(f"{name}: shape {tuple(t.shape)}, expected {shape}")
        if dtype is not None and t.dtype != dtype:
            errors.append(f"{name}: dtype {t.dtype}, expected {dtype}")
        if not t.is_contiguous():
            errors.append(f"{name}: not contiguous")
    if errors:
        raise ValueError(f"layer {i} cannot use the gfx1250 chain: " + "; ".join(errors))
    return w


def step_metadata(layer, positions, forward_batch, length_out):
    """The chain's ``m`` for one target-verify step of one request (8 rows), or None when the
    native fused KV-write path would not apply either (the caller then runs the native layer).

    Reuses the native path's own resolution of the cache, scales, write slots, SWA mapping and
    RoPE table (``attention_kv_write._kv_write_target``), so both paths read the same tensors.
    Not handled here (see ~/coord/integration-plan.md): native marks unmapped SWA slots with 0
    (a dummy slot), the chain with a negative value; in-window tokens are always mapped.
    """
    from sglang.srt.rocm_muse_spark import attention_kv_write as kv

    attn = layer.self_attn
    if not kv._attention_shape_ok(attn, forward_batch):
        return None
    target = kv._kv_write_target(attn, forward_batch)
    if target is None:
        return None
    md = kv.get_attn_backend().forward_metadata
    if md.kv_indices is None or md.kv_indices.shape[0] != 1:
        return None
    # The KV length including this step's 8 rows. The aiter backend precomputes it only on its
    # gfx95 Muse paths; otherwise ``length_out`` holds it: a persistent int32 [1] buffer that
    # ChainRuntime.begin_step fills once per step.
    length = md.verify_seqlen_k
    if length is None:
        length = length_out
    return (
        positions,
        target.cos_sin,
        target.locations,
        target.swa_mapping,
        target.k_cache,
        target.v_cache,
        target.k_scale,
        target.v_scale,
        md.kv_indices,
        length,
    )


_EXPERT_KEYS = (
    ("w13", "w13_weight"),
    ("w13_scale", "w13_weight_scale"),
    ("w2", "w2_weight"),
    ("w2_scale", "w2_weight_scale"),
)
def replicate_experts(backbone, i: int, w: dict, masks, *, rank: int, name: str, barrier,
                      pad: int = 0) -> dict:
    """Experiment (MK.93-95): give layer ``i`` the second copies that ``masks`` (see
    ``tlx_route``) place on this rank. Every rank writes its 64 experts to a host file, all
    ranks meet, each reads the experts it hosts, all meet again, files are removed. Host
    memory only.

    The second copies follow this rank's own 64 experts in one tensor, compactly: the group
    with offset k = 0 (from rank - 1), then k = 1 (rank - 2), then k = 2 (rank - 3), each in
    ascending local id. The model's own parameters are re-pointed at the first 64 entries, so
    nothing is held twice. Must run on every rank at the same point."""
    import os

    import numpy as np

    from sglang.srt.kernel_plugins.amd.muse_spark.placement import bits

    experts = backbone.layers[i].mlp.experts
    paths = {}
    for key, _ in _EXPERT_KEYS:
        path = f"/dev/shm/{name}_xc_{rank}_{key}.bin"
        w[key].detach().cpu().numpy().tofile(path)
        paths[key] = path
    barrier()
    hosted = [((rank - 1 - k) % 4, bits(masks[(rank - 1 - k) % 4][k])) for k in range(3)]
    extra = sum(len(ids) for _, ids in hosted)
    for key, attr in _EXPERT_KEYS:
        own = w.pop(key)  # the caller's dict must not keep the old tensor alive
        # ``pad`` zeroed experts follow the copies. The caller pads every layer of a rank to
        # the same expert count: after a stage-5 (expert exchange) failure the guard lets the
        # W13 launches of the later layers of that step run on the slot table the failed
        # layer left in the shared workspace, and that table's local indices are only valid
        # for a tensor at least as long as the failed layer's (MK.101).
        big = torch.empty((LOCAL_EXPERTS + extra + pad,) + tuple(own.shape[1:]), dtype=own.dtype,
                          device=own.device)
        big[:LOCAL_EXPERTS].copy_(own)
        if pad:
            big[LOCAL_EXPERTS + extra:].zero_()
        at = LOCAL_EXPERTS
        for src, ids in hosted:
            if not ids:
                continue
            theirs = np.memmap(f"/dev/shm/{name}_xc_{src}_{key}.bin", dtype=np.uint8, mode="r",
                               shape=tuple(own.shape))
            big[at:at + len(ids)].copy_(torch.from_numpy(theirs[ids]).to(own.device))
            at += len(ids)
            del theirs
        param = getattr(experts, attr)
        assert param.dtype == torch.uint8 and param.data_ptr() == own.data_ptr(), (key, attr)
        param.data = big[:LOCAL_EXPERTS]
        del own
        # hand the old block back before the next allocation can be carved out of it: a
        # smaller tensor placed in a freed larger block pins the rest of that block (MK.95:
        # 20-28 GiB reserved but unused per GPU without this)
        torch.cuda.empty_cache()
        w[key] = big
    barrier()
    for path in paths.values():
        os.remove(path)
    return w
