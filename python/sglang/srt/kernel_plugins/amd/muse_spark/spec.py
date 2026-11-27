"""In-tree gfx1250 Muse Spark contract; stage 0 executes the native layers."""

import torch

from sglang.kernels.plugin_kernels.amd.spec import (
    MONOKERNEL_API_VERSION,
    DeviceLimits,
    Feature,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
)


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    from sglang.srt.configs.muse_spark import MuseSparkConfig

    if not isinstance(hf_config, MuseSparkConfig):
        return ["gfx1250 Muse Spark requires a nested MuseSparkConfig"]
    text = hf_config.text_config
    dimensions = (
        ("hidden_size", text.hidden_size, 8192),
        ("num_hidden_layers", text.num_hidden_layers, 62),
        ("num_attention_heads", text.num_attention_heads, 128),
        ("num_key_value_heads", text.num_key_value_heads, 16),
        ("head_dim", text.head_dim, 64),
        ("moe_hidden_size", text.moe_hidden_size, 4096),
        ("moe_intermediate_size", text.moe_intermediate_size, 4096),
        ("num_local_experts", text.num_local_experts, 256),
        ("num_experts_per_tok", text.num_experts_per_tok, 8),
        ("num_shared_experts", text.num_shared_experts, 0),
        ("sliding_window", text.sliding_window, 2048),
    )
    reasons = [
        f"text_config.{name}={got!r}; gfx1250 Muse Spark requires {want!r}"
        for name, got, want in dimensions
        if got != want
    ]
    kinds = ["full_attention" if i % 4 == 1 else "sliding_attention" for i in range(62)]
    if list(text.layer_types) != kinds:
        reasons.append(
            "layer_types must be full at indices 1,5,...,61 and sliding elsewhere"
        )
    theta = [0.0 if kind == "full_attention" else 500000.0 for kind in kinds]
    if list(text.layer_rope_theta) != theta:
        reasons.append(
            "layer_rope_theta must be zero on full layers and 500000 on sliding layers"
        )
    if list(text.mlp_layer_types) != ["sparse"] * 62:
        reasons.append("mlp_layer_types must be sparse in all 62 layers")
    return reasons


def _model_disable_reasons(model, hf_config, copies) -> list[str]:
    from sglang.srt.rocm_muse_spark.model import MuseSpark1xModel
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    sizes = (parallel.attn_tp_size, parallel.moe_tp_size, parallel.moe_ep_size)
    reasons = []
    if sizes != (4, 1, 4):
        reasons.append(f"attention TP / MoE TP / EP is {sizes}; requires (4, 1, 4)")
    backbone = model.model.language_model
    if not isinstance(backbone, MuseSpark1xModel):
        return reasons + ["gfx1250 Muse Spark requires the ROCm MuseSpark1xModel"]
    if len(backbone.layers) != 62:
        return reasons + [f"loaded {len(backbone.layers)} layers; requires 62"]
    if backbone._ffn_boundaries is None:
        reasons.append("gfx1250 Muse Spark stage 0 requires fused FFN boundaries")
    elif len(backbone._ffn_boundaries) != 62:
        reasons.append("gfx1250 Muse Spark stage 0 requires 62 fused FFN boundaries")
    for i in range(1, 61):
        moe = backbone.layers[i].mlp
        if not moe.mxfp4_checkpoint:
            reasons.append(f"layer {i} requires MXFP4 experts")
        elif moe.experts.w13_weight.dtype != torch.uint8:
            reasons.append(f"layer {i} requires native packed uint8 MXFP4 weights")
    return reasons


def _build(model, hf_config, ctx):
    import os

    from sglang.srt.kernel_plugins.amd.muse_spark.forward import Stage0Forward

    if os.environ.get("MUSE_HOST_TIMERS") == "1":
        # probe (MK.84 / MK.107): wall-clock wrappers on the scheduler / DFlash host path of this
        # rank; the module lives outside the tree (~/coord/no_peer_guard, on PYTHONPATH there)
        import muse_host_timers

        muse_host_timers.install()
    if os.environ.get("MUSE_COLLECTIVE_LOG") == "1":
        # probe (MK.110): which TP collectives a decode step issues (op, shape, dtype, whether
        # inside a graph capture, caller); printed once per distinct signature on rank 0
        import sys as _sys

        import torch as _torch

        from sglang.srt.runtime_context import get_parallel as _gp

        _grp, _seen = _gp().tp_group, {}

        def _wrap(name):
            fn = getattr(_grp, name)

            def w(x, *a, **k):
                t = x if isinstance(x, _torch.Tensor) else (a[0] if a and isinstance(a[0], _torch.Tensor) else None)
                f = _sys._getframe(1)
                key = (name, None if t is None else tuple(t.shape), None if t is None else str(t.dtype),
                       bool(_torch.cuda.is_current_stream_capturing()), f.f_code.co_name)
                _seen[key] = _seen.get(key, 0) + 1
                if _seen[key] == 1 and _grp.rank_in_group == 0:
                    print("muse_collective", key, f.f_code.co_filename.rsplit("/", 1)[-1], f.f_lineno, flush=True)
                return fn(x, *a, **k)

            setattr(_grp, name, w)

        for _n in ("all_reduce", "all_gather", "all_gather_into_tensor", "reduce_scatter", "broadcast"):
            if hasattr(_grp, _n):
                _wrap(_n)
    chain = None
    # The kernel chain exchanges between ranks through host memory; that mode is chosen by
    # the kernels' own switch, read when they are first imported. Without it: stage 0.
    from sglang.srt.kernel_plugins.amd.muse_spark import compare

    if os.environ.get("MUSE_PEER_HOST") == "1":
        from sglang.srt.kernel_plugins.amd.muse_spark.runtime import ChainRuntime
        from sglang.srt.runtime_context import get_parallel

        from datetime import timedelta

        import torch.distributed as dist

        group = get_parallel().tp_group

        def barrier():
            # a rank that diverges stops the server with an error instead of hanging it
            dist.monitored_barrier(group.cpu_group, timeout=timedelta(seconds=ctx.deadline_s))

        if os.environ.get("MUSE_CHAIN_DUMMY_SCALES") == "1":
            # DUMMY-loaded checkpoints only (random scale bytes, 0xFF is NaN): this
            # overwrites the expert scales, so it must never run on real weights
            compare.sanitize_dummy_scales(model.model.language_model, 1, 60)
        # exchange waits: the plugin's deadline (seconds) and a hard stop past it (MK.105)
        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_standin

        tlx_standin.set_host_wait_seconds(ctx.deadline_s)
        longest = getattr(hf_config.text_config, "max_position_embeddings", 131072)
        chain = ChainRuntime(
            model.model.language_model,
            rank=group.rank_in_group,
            barrier=barrier,
            context=8192 if longest <= 8192 else 131072,
            # the rank processes share a parent: one name per server
            name=f"muse_spark_{os.getppid()}",
        )
    if chain is not None and os.environ.get("MUSE_HOST_ALL_REDUCE") == "1":
        # experiment (MK.110): the [8, 8192] and [8, 4096] bf16 all-reduces of a decode step
        # outside the chain layers (layers 0 and 61, the input embedding) go through the
        # chain's host-memory exchange instead of the communication library. Other shapes
        # (prefill) are unchanged. Sums are rounded once, not pairwise.
        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
            tlx_host_collective,
        )

        _reducers = [
            tlx_host_collective.HostAllReduce(
                next(model.parameters()).device, width=w, rank=group.rank_in_group,
                barrier=barrier, name=f"muse_spark_{os.getppid()}",
            )
            for w in (8192, 4096)
        ]
        _library_all_reduce = group.all_reduce

        def _all_reduce(input_):
            for r in _reducers:
                if r.supported(input_):
                    return r(input_)
            return _library_all_reduce(input_)

        group.all_reduce = _all_reduce
        model._muse_host_all_reduce = _reducers  # must outlive every captured graph
    if chain is not None and compare.watch_only():
        return compare.WatchForward(model=model, chain=chain, rank=group.rank_in_group)
    if chain is not None and compare.enabled():
        return compare.CompareForward(
            model=model, chain=chain, rank=group.rank_in_group, barrier=barrier
        )
    return Stage0Forward(model=model, chain=chain)


SPEC = MonoSpec(
    name="muse_spark_gfx1250_stage0",
    devices={"gfx1250": DeviceLimits(widths=(8,), tp_sizes=(4,), resident_blocks=1)},
    features=frozenset({Feature.EXPERT_PARALLEL, Feature.SPECULATIVE_DECODING}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"tlx", "aiter"}),
        server_args={
            "ep_size": frozenset({4}),
            "speculative_algorithm": frozenset({"DFLASH"}),
            "speculative_num_draft_tokens": frozenset({8}),
            "kv_cache_dtype": frozenset({"fp8_e4m3"}),
            "page_size": frozenset({16}),
        },
        hf_config={"model_type": "muse_spark"},
    ),
    numerics=Numerics(kind=NumericsKind.BITWISE),
    config_disable_reasons=_config_disable_reasons,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=MONOKERNEL_API_VERSION,
)
