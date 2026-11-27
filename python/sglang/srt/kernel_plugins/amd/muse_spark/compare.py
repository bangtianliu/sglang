"""Validation mode for the gfx1250 Muse Spark kernel chain inside a real server (MUSE_CHAIN_COMPARE=1).

Every eligible layer of a target-verify step runs twice on the same inputs: first the chain, then
the native layer, whose results the model keeps (so the native trajectory, KV cache included,
is what later steps see: the native layer writes the 8 new KV rows last). Per layer and step it
logs one JSON line with the chain-vs-native error of the new residual stream and of the next
layer's attention input.

The chain matches its reference archive bit for bit, but the native layers sum in a different
order (aiter GEMMs), so the comparison is numeric: report max relative error and let the reader
apply the tolerances of bench_layer_native (q/k/v/gate 0.015, attention 0.008, boundaries 0.02).

Requires eager target-verify steps (no CUDA graph): each comparison synchronizes with the host.
Dummy-loaded MXFP4 checkpoints carry random E8M0 scale bytes (0xFF is NaN); ``sanitize_dummy_scales``
maps them into a finite range so both paths compute finite values.
"""

import json
import os

import torch

from sglang.srt.kernel_plugins.amd.muse_spark.forward import Stage0Forward

ENV = "MUSE_CHAIN_COMPARE"


def enabled() -> bool:
    return os.environ.get(ENV) in ("1", "2")


def watch_only() -> bool:
    """MUSE_CHAIN_COMPARE=2: the chain serves (no native rerun); log its failure word every step."""
    return os.environ.get(ENV) == "2"


def sanitize_dummy_scales(backbone, first: int, last: int) -> None:
    """Map random E8M0 scale bytes of layers first..last into 120..130 (2^-7 .. 2^3), in place."""
    for i in range(first, last + 1):
        experts = backbone.layers[i].mlp.experts
        for scale in (experts.w13_weight_scale, experts.w2_weight_scale):
            scale.data.copy_(120 + scale.data % 11)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


class CompareForward(Stage0Forward):
    """Stage0Forward whose chain layers are checked against the native layers."""

    def __init__(self, *, model, chain, rank, barrier):
        super().__init__(model=model, chain=chain)
        self._rank = rank
        self._barrier = barrier
        self._compiled = set()
        self._declined = set()
        self._step = 0

    def _decline_reason(self, i, positions, stream, forward_batch, attn_input):
        """Why ChainRuntime.run_layer returned None (mirrors its and step_metadata's checks)."""
        from sglang.srt.kernel_plugins.amd.muse_spark import runtime
        from sglang.srt.rocm_muse_spark import attention_kv_write as kv

        if not runtime.FIRST_LAYER <= i <= runtime.LAST_LAYER:
            return "layer outside the chain's range"
        if attn_input.shape != (runtime.ROWS, 8192) or stream.shape != (runtime.ROWS, 8192):
            return f"shapes attn {tuple(attn_input.shape)} stream {tuple(stream.shape)}"
        if attn_input.dtype != torch.bfloat16 or stream.dtype != torch.float32:
            return f"dtypes attn {attn_input.dtype} stream {stream.dtype}"
        attn = self._backbone.layers[i].self_attn
        mode = str(forward_batch.forward_mode)
        if not kv._attention_shape_ok(attn, forward_batch):
            return f"kv._attention_shape_ok False (mode {mode})"
        if kv._kv_write_target(attn, forward_batch) is None:
            return f"kv._kv_write_target None (mode {mode})"
        md = kv.get_attn_backend().forward_metadata
        seqlen = getattr(md, "verify_seqlen_k", "missing")
        kvi = getattr(md, "kv_indices", "missing")
        return (f"metadata: mode {mode}, verify_seqlen_k {None if seqlen is None else type(seqlen).__name__}, "
                f"kv_indices {None if kvi is None else getattr(kvi, 'shape', kvi)}, md {type(md).__name__}")

    def _forward_layers(self, *, positions, forward_batch, input_embeds):
        self._step += 1
        return super()._forward_layers(
            positions=positions, forward_batch=forward_batch, input_embeds=input_embeds
        )

    def _native(self, i, positions, stream, forward_batch, attn_input, attn_input_fp8):
        backbone = self._backbone
        return backbone.layers[i](
            positions,
            stream,
            forward_batch,
            attn_input=attn_input,
            ffn_boundary=backbone._ffn_boundaries[i],
            pre_norm_dtype=torch.bfloat16 if i == len(backbone.layers) - 1 else None,
            sp=None,
            attn_input_fp8=attn_input_fp8,
        )

    def _chain_once(self, i, positions, stream, forward_batch, attn_input):
        chain = self._chain
        kind = chain.variant(i)
        if kind not in self._compiled:
            # Compile on every rank inside a throwaway capture (nothing executes): ranks compile
            # at different speeds, and an exchange started meanwhile would time out (~1 ms).
            turn = dict(chain._turn)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                chain.run_layer(i, positions, stream.clone(), forward_batch, attn_input.clone())
            chain._turn = turn
            torch.cuda.synchronize()
            self._barrier()
            chain.reset_failures()
            self._compiled.add(kind)
        torch.cuda.synchronize()
        self._barrier()
        launched = chain.run_layer(i, positions, stream, forward_batch, attn_input)
        torch.cuda.synchronize()
        return launched

    def _run_layer(
        self, *, i, positions, stream, forward_batch, attn_input, attn_input_fp8
    ):
        stream0, attn0 = stream.clone(), attn_input.clone()
        launched = None
        if self._chain is not None:
            launched = self._chain_once(i, positions, stream0.clone(), forward_batch, attn0.clone())
        if launched is not None:
            chain_stream, chain_next = launched[0].clone(), launched[1].clone()
            failures = self._chain.failures()
            self._chain.reset_failures()
        if launched is None and (self._step, i) not in self._declined:
            self._declined.add((self._step, i))
            print(json.dumps(dict(event="chain_declined", rank=self._rank, step=self._step,
                                  layer=i, why=self._decline_reason(i, positions, stream,
                                                                    forward_batch, attn_input))),
                  flush=True)
        out = self._native(i, positions, stream, forward_batch, attn_input, attn_input_fp8)
        if launched is not None:
            torch.cuda.synchronize()
            row = dict(
                event="chain_compare",
                rank=self._rank,
                step=self._step,
                layer=i,
                kind=self._chain._kind(i),
                stream_rel=_rel(chain_stream, out[0]),
                next_attn_rel=_rel(chain_next, out[1]),
                chain_finite=bool(
                    torch.isfinite(chain_stream).all() and torch.isfinite(chain_next.float()).all()
                ),
                native_finite=bool(
                    torch.isfinite(out[0]).all() and torch.isfinite(out[1].float()).all()
                ),
                failures=failures,
            )
            print(json.dumps(row), flush=True)
        return out


class WatchForward(Stage0Forward):
    """The chain serves as in production; after each eager step, log the failure word (no reset)."""

    def __init__(self, *, model, chain, rank):
        super().__init__(model=model, chain=chain)
        self._rank = rank
        self._step = 0

    # MK.128: MUSE_CHAIN_FP=1 also logs a fingerprint of the residual stream after every layer
    # (float64 sum and abs-sum, as hex), to find the first (step, layer) where two runs differ.
    _FP = os.environ.get("MUSE_CHAIN_FP") == "1"

    def _run_layer(self, *, i, positions, stream, forward_batch, attn_input, attn_input_fp8):
        out = super()._run_layer(i=i, positions=positions, stream=stream, forward_batch=forward_batch,
                                 attn_input=attn_input, attn_input_fp8=attn_input_fp8)
        if self._FP:
            torch.cuda.synchronize()
            d = out[0].double()
            self._fp.append("%s/%s" % (float(d.sum()).hex(), float(d.abs().sum()).hex()))
            if self._step == 0 and i in (0, 1, 2):
                # per-stage fingerprints of the layer's intermediate buffers (byte sums), to name
                # the first stage whose result differs between two runs
                try:
                    ch = self._chain
                    kind = ch._kind(i)
                    ws = ch._workspaces[kind, (ch._turn[kind] - 1) % ch._copies]
                    f = lambda x: int(x.contiguous().view(-1).view(torch.uint8).to(torch.int64).sum())  # noqa: E731
                    st = {k: f(v) for k, v in ws.buffers.items() if torch.is_tensor(v)}
                    for name in ("ids", "scores", "pooled", "sorted_ids", "expert_ids", "num_post"):
                        v = getattr(ws.s4, name, None)
                        if torch.is_tensor(v):
                            st["s4." + name] = f(v)
                    st["s3.normalized"] = f(ws.s3.normalized) if torch.is_tensor(getattr(ws.s3, "normalized", None)) else None
                    # expert stage: its per-rank intermediates and the MXFP8 activation buffers
                    import msgspec

                    for name in msgspec.structs.fields(ws.s5):
                        v = getattr(ws.s5, name.name, None)
                        if torch.is_tensor(v) and v.numel() < (1 << 28):
                            st["s5." + name.name] = f(v)
                    try:
                        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_experts

                        for k2, v in tlx_experts.mx_buffers(out[0].device).items():
                            if torch.is_tensor(v):
                                st["mx." + k2] = f(v)
                    except Exception as e2:  # noqa: BLE001
                        st["mx"] = repr(e2)[:120]
                    print(json.dumps(dict(event="chain_fp_stage", rank=self._rank, layer=i, kind=str(kind), st=st)),
                          flush=True)
                    if i == 1 and os.environ.get("MUSE_CHAIN_FP_DUMP"):
                        # the expert stage's tensors of this layer and step, for an offline diff
                        dump = {"s5." + n.name: getattr(ws.s5, n.name).detach().cpu()
                                for n in msgspec.structs.fields(ws.s5)
                                if torch.is_tensor(getattr(ws.s5, n.name, None))}
                        for name in ("ids", "scores", "pooled", "sorted_ids", "expert_ids", "num_post"):
                            v = getattr(ws.s4, name, None)
                            if torch.is_tensor(v):
                                dump["s4." + name] = v.detach().cpu()
                        dump["x6"] = ws.buffers["x6"].detach().cpu()
                        torch.save(dump, "%s-r%d.pt" % (os.environ["MUSE_CHAIN_FP_DUMP"], self._rank))
                except Exception as e:  # noqa: BLE001
                    print(json.dumps(dict(event="chain_fp_stage", rank=self._rank, layer=i, error=repr(e)[:300])),
                          flush=True)
        return out

    def _forward_layers(self, *, positions, forward_batch, input_embeds):
        self._fp = []
        if self._FP:
            d = input_embeds.double()
            self._fp.append("%s/%s" % (float(d.sum()).hex(), float(d.abs().sum()).hex()))
            if self._step == 0:
                # fingerprint of the KV prefix that PREFILL wrote, for the first layers: tells a
                # difference that entered through the (native) prefill from one made by the chain
                try:
                    from sglang.srt.rocm_muse_spark import attention_kv_write as kv

                    out = {}
                    n = int(forward_batch.seq_lens[0])
                    pages = kv.get_attn_backend().forward_metadata.kv_indices[0].long()
                    slots = (pages[: (n + 15) // 16, None] * 16
                             + torch.arange(16, device=pages.device)[None, :]).flatten()[:n]
                    for i in (0, 1, 2, 3, 30):
                        t = kv._kv_write_target(self._backbone.layers[i].self_attn, forward_batch)
                        if t is None:
                            out[i] = "skipped"
                            continue
                        sl = slots
                        if t.swa_mapping is not None:  # sliding layer: mapped (in-window) slots only
                            sl = t.swa_mapping[slots].long()
                            sl = sl[sl > 0]
                        k = t.k_cache[sl]
                        v = t.v_cache[sl]
                        f = lambda x: int(x.contiguous().view(-1).view(torch.uint8).to(torch.int64).sum())  # noqa: E731
                        out[i] = [int(sl.numel()), f(k), f(v)]
                    print(json.dumps(dict(event="chain_fp_kv", rank=self._rank, prefix=n, kv=out)),
                          flush=True)
                except Exception as e:  # noqa: BLE001
                    print(json.dumps(dict(event="chain_fp_kv", rank=self._rank, error=repr(e)[:300])),
                          flush=True)
        out = super()._forward_layers(
            positions=positions, forward_batch=forward_batch, input_embeds=input_embeds
        )
        self._step += 1
        torch.cuda.synchronize()
        print(json.dumps(dict(event="chain_watch", rank=self._rank, step=self._step,
                              failures=self._chain.failures())), flush=True)
        if self._FP:
            print(json.dumps(dict(event="chain_fp", rank=self._rank, step=self._step,
                                  rows=int(positions.shape[0]), fp=self._fp)), flush=True)
        return out
