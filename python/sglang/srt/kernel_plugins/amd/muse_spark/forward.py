"""Stage-0 forward using Muse Spark's existing layers and DFlash capture points."""

import json
import os
import threading
import time

import torch

# probe (MK.74): MUSE_STEP_STAMPS=1 puts realtime stamps inside every 8-row decode step
# (start, before layer 0, after layers 0 / 60 / 61, after the logits processor) and prints the
# median split every 10 s from a side thread
STEP_STAMPS = os.environ.get("MUSE_STEP_STAMPS") == "1"
# MK.75: with the chain on, the LM head of 8-row verify steps runs on the chain's projection
# kernel (fp32 logits like the native aiter GEMM here; sums in another K order).
# MUSE_CHAIN_LM_HEAD=0: native.
CHAIN_LM_HEAD = os.environ.get("MUSE_CHAIN_LM_HEAD", "1") == "1"
_NAMES = ("embed", "layer0", "layers1-60", "layer61", "lm_head", "gather", "logits_tail")


class _Stamps:
    def __init__(self, device):
        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_stamp

        self.mod = tlx_stamp
        self.buf = torch.zeros(tlx_stamp.RING * 8, device=device, dtype=torch.int64)
        self.count = torch.zeros(1, device=device, dtype=torch.int64)
        threading.Thread(target=self._report, daemon=True).start()

    def mark(self, slot):
        self.mod.stamp[(1,)](self.buf, self.count, SLOT=slot, N=8, RING=self.mod.RING)

    def _report(self):
        last = -1
        while True:
            time.sleep(10)
            n = int(self.count.item())
            if n == last or n < self.mod.RING + 2:
                continue
            last = n
            v = self.buf.clone().cpu().reshape(self.mod.RING, 8).double() / 100
            v = v[v[:, 0].argsort()][2:-2]  # by start time; drop the row being written
            inside = (v[:, 1:] - v[:, :-1]).median(0).values
            period = (v[1:, 0] - v[:-1, 0]).median()
            row = {k: round(float(x), 1) for k, x in zip(_NAMES, inside)}
            row.update(event="step_stamps", steps=n, in_graph_us=round(float((v[:, 7] - v[:, 0]).median()), 1),
                       period_us=round(float(period), 1))
            print(json.dumps(row), flush=True)


class Stage0Forward:
    def __init__(self, *, model, chain=None):
        """``chain``: an optional ``runtime.ChainRuntime``; layers it declines run natively."""
        self._chain = chain
        self._model = model
        self._backbone = model.model.language_model
        self._original = model.forward
        self._stamps = None
        self._lm_ws = None

    @torch.no_grad()
    def forward(self, plan, input_ids, positions, forward_batch, input_embeds=None):
        if input_ids is None and input_embeds is None:
            raise ValueError("input_embeds is required for multimodal forwarding")
        if (
            input_ids is not None
            and not forward_batch.forward_mode.is_decode()
            and forward_batch.contains_mm_inputs()
        ):
            return self._original(
                input_ids, positions, forward_batch, input_embeds=input_embeds
            )
        st = None
        if STEP_STAMPS and positions.shape[0] == 8 and forward_batch.forward_mode.is_target_verify():
            if self._stamps is None:
                self._stamps = _Stamps(positions.device)
            st = self._stamps
            st.mark(0)
        self._st = st
        if input_embeds is None:
            input_embeds = self._model.get_input_embeddings()(input_ids)
        hidden, aux = self._forward_layers(
            positions=positions, forward_batch=forward_batch, input_embeds=input_embeds
        )
        lp = self._model.logits_processor
        if st is not None:
            # the softcap path gathers the vocabulary shards with the TP group's all-gather
            from sglang.srt.runtime_context import get_parallel

            group = get_parallel().tp_group
            names = ("all_gather", "all_gather_into_tensor")
            originals = {n: getattr(group, n) for n in names}

            def wrap(fn):
                def stamped(*a, **k):
                    st.mark(5)
                    y = fn(*a, **k)
                    st.mark(6)
                    return y
                return stamped

            for n in names:
                setattr(group, n, wrap(originals[n]))
            try:
                out = self._logits(lp, input_ids, hidden, forward_batch, aux)
            finally:
                for n in names:
                    delattr(group, n)
        else:
            out = self._logits(lp, input_ids, hidden, forward_batch, aux)
        if st is not None:
            st.mark(7)
        return out

    def _logits(self, lp, input_ids, hidden, forward_batch, aux):
        """The model's logits processor; in 8-row verify steps of a chain server its plain bf16
        LM-head GEMM is replaced by the chain's projection kernel."""
        lm_head = self._model.lm_head
        if not (CHAIN_LM_HEAD and self._chain is not None
                and forward_batch.forward_mode.is_target_verify()):
            if STEP_STAMPS and not getattr(self, "_lm_probe", False) and \
                    forward_batch.forward_mode.is_target_verify():
                # one-time probe: what the native LM head returns on this path
                self._lm_probe = True
                native = lp._compute_lm_head

                def probe(hidden_states, head, embedding_bias=None, *, out=None):
                    y = native(hidden_states, head, embedding_bias, out=out)
                    print(json.dumps(dict(
                        event="lm_head_probe", x=str(hidden_states.dtype),
                        x_shape=list(hidden_states.shape), out_arg=None if out is None else str(out.dtype),
                        y=str(y.dtype), y_shape=list(y.shape), w=str(head.weight.dtype),
                        fp32=[bool(lp.use_fp32_lm_head), bool(lp.use_fp32_lm_head_output)],
                        defer=bool(lp.defer_verify_gather), softcap=lp.final_logit_softcapping)),
                        flush=True)
                    return y

                lp._compute_lm_head = probe
                try:
                    return lp(input_ids, hidden, lm_head, forward_batch, aux)
                finally:
                    del lp._compute_lm_head
            return lp(input_ids, hidden, lm_head, forward_batch, aux)
        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_lm_head
        from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method

        original = lp._compute_lm_head

        def compute(hidden_states, head, embedding_bias=None, *, out=None):
            weight = getattr(head, "weight", None)
            plain = (
                head is lm_head and embedding_bias is None and out is None
                and weight is not None
                and not should_apply_lm_head_quant_method(
                    head, getattr(head, "quant_method", None))
                and not hasattr(head, "apply_lora")
                and not (lp.use_fp32_lm_head or lp.use_fp32_lm_head_output)
                and tlx_lm_head.supported(hidden_states, weight)
            )
            if not plain:
                return original(hidden_states, head, embedding_bias, out=out)
            if self._lm_ws is None:
                self._lm_ws = tlx_lm_head.Workspace(weight.device, weight.shape[0])
            return tlx_lm_head.launch(hidden_states, weight, self._lm_ws)

        lp._compute_lm_head = compute
        try:
            return lp(input_ids, hidden, lm_head, forward_batch, aux)
        finally:
            del lp._compute_lm_head

    def _forward_layers(self, *, positions, forward_batch, input_embeds):
        backbone = self._backbone
        stream = input_embeds.to(backbone.residual_dtype)
        attn_input = backbone.layers[0].self_attn_input_norm(stream)
        attn_input_fp8 = None
        aux = []
        if self._chain is not None:
            self._chain.begin_step(forward_batch, positions.shape[0])
        st = getattr(self, "_st", None)
        last = len(backbone.layers) - 1
        if st is not None:
            st.mark(1)
        for i in range(len(backbone.layers)):
            if st is not None and i in (1, last):
                st.mark(2 if i == 1 else 3)
            stream, attn_input, attn_input_fp8 = self._run_layer(
                i=i,
                positions=positions,
                stream=stream,
                forward_batch=forward_batch,
                attn_input=attn_input,
                attn_input_fp8=attn_input_fp8,
            )
            if i in backbone.layers_to_capture:
                aux.append(stream.to(torch.bfloat16))
        if st is not None:
            st.mark(4)
        return attn_input, aux if self._model.capture_aux_hidden_states else None

    def _run_layer(
        self, *, i, positions, stream, forward_batch, attn_input, attn_input_fp8
    ):
        backbone = self._backbone
        if self._chain is not None:
            launched = self._chain.run_layer(
                i, positions, stream, forward_batch, attn_input
            )
            if launched is not None:
                return launched
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
