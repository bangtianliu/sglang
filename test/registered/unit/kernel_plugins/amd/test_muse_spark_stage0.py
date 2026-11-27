"""CPU coverage of Muse Spark's stage-0 routing and native three-value handoff."""

import unittest
from unittest.mock import patch

import msgspec
import torch

from sglang.kernels.plugin_kernels.amd import spec as registry
from sglang.kernels.plugin_kernels.amd.spec import BuildContext, ShapeKey
from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.configs.muse_spark import MuseSparkConfig
from sglang.srt.kernel_plugins.amd import binding
from sglang.srt.kernel_plugins.amd.muse_spark.spec import SPEC
from sglang.srt.model_executor.cuda_graph_config import (
    Phase,
    default_cuda_graph_config,
    with_phase,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.rocm_muse_spark.model import (
    MuseSpark1xModel,
    MuseSparkTextForCausalLM,
)
from sglang.srt.runtime_context import get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class ModelConfig(msgspec.Struct):
    hf_config: MuseSparkConfig
    dtype: torch.dtype = torch.bfloat16


class Batch(msgspec.Struct):
    input_ids: torch.Tensor
    forward_mode: ForwardMode = ForwardMode.TARGET_VERIFY
    batch_size: int = 1
    media: bool = False

    def contains_mm_inputs(self):
        return self.media


class DecodeBackend(msgspec.Struct):
    decode_attention_backend_str: str = "tlx"


def config():
    kinds = ["full_attention" if i % 4 == 1 else "sliding_attention" for i in range(62)]
    return MuseSparkConfig(
        architectures=["MuseSparkForConditionalGeneration"],
        text_config={
            "layer_types": kinds,
            "layer_rope_theta": [
                0 if kind == "full_attention" else 500000 for kind in kinds
            ],
        },
    )


def server_args(**changes):
    values = dict(
        model_path="dummy",
        tp_size=4,
        ep_size=4,
        enable_amd_monokernel=True,
        speculative_algorithm="DFLASH",
        speculative_num_draft_tokens=8,
        kv_cache_dtype="fp8_e4m3",
        page_size=16,
        attention_backend="tlx",
    )
    return ServerArgs(**(values | changes))


class Scale(torch.nn.Module):
    def forward(self, x):
        return x * 0.5


class FakeExperts(torch.nn.Module):
    def __init__(self, *, packed):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.uint8 if packed else torch.bfloat16),
            requires_grad=False,
        )


class FakeMoE(torch.nn.Module):
    def __init__(self, *, packed):
        super().__init__()
        self.mxfp4_checkpoint = packed
        self.experts = FakeExperts(packed=packed)


class FakeLayer(torch.nn.Module):
    def __init__(self, *, index, calls):
        super().__init__()
        self.index = index
        self.calls = calls
        self.mlp = FakeMoE(packed=1 <= index <= 60)
        self.last_stream = None

    def self_attn_input_norm(self, stream):
        return (stream * 0.25).to(torch.bfloat16)

    def forward(
        self,
        positions,
        stream,
        forward_batch,
        *,
        attn_input,
        ffn_boundary,
        pre_norm_dtype,
        sp,
        attn_input_fp8,
    ):
        self.calls.append(
            (
                self.index,
                attn_input.clone(),
                attn_input_fp8,
                ffn_boundary,
                pre_norm_dtype,
                sp,
                positions.clone(),
            )
        )
        # The next result depends on the third handoff, so dropping it changes logits.
        previous = (
            0 if attn_input_fp8 is None else attn_input_fp8[0] * attn_input_fp8[1]
        )
        stream = stream + attn_input.float() * 0.001 + previous * 0.001
        stream = stream + (self.index + 1) * 0.005 + ffn_boundary * 0.0001
        self.last_stream = stream
        normalized = (stream * 0.5).to(torch.bfloat16)
        tag = (torch.tensor(float(self.index + 1)), torch.tensor(0.125))
        return stream, normalized, tag


class FakeBackbone(MuseSpark1xModel):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.calls = []
        self.residual_dtype = torch.float32
        self.layers_to_capture = [1, 16, 30, 44, 59]
        self.embed_tokens = torch.nn.Embedding(8, 4, dtype=torch.bfloat16)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(torch.arange(32).reshape(8, 4) / 16)
        self.embed_norm = Scale()
        self.layers = torch.nn.ModuleList(
            [FakeLayer(index=i, calls=self.calls) for i in range(62)]
        )
        self._ffn_boundaries = list(range(62))


class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = FakeBackbone()

    def forward(self, *args, **kwargs):
        return self.language_model(*args, **kwargs)


class FakeMuseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Wrapper()
        self.capture_aux_hidden_states = True
        self.media_calls = 0
        self.lm_head = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.arange(12).reshape(3, 4) / 8)

    def get_input_embeddings(self):
        backbone = self.model.language_model
        return torch.nn.Sequential(backbone.embed_tokens, backbone.embed_norm)

    def logits_processor(self, input_ids, hidden, head, forward_batch, aux):
        return head(hidden), aux

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        if input_ids is None and input_embeds is None:
            raise ValueError("input_embeds is required for multimodal forwarding")
        if (
            input_ids is not None
            and forward_batch.media
            and not forward_batch.forward_mode.is_decode()
        ):
            self.media_calls += 1
            embeds = self.get_input_embeddings()(input_ids) + 2
            return self.forward(None, positions, forward_batch, input_embeds=embeds)
        if input_embeds is None:
            input_embeds = self.get_input_embeddings()(input_ids)
        return MuseSparkTextForCausalLM.forward(
            self, input_ids, positions, forward_batch, input_embeds=input_embeds
        )


def build(model):
    return SPEC.build(
        model,
        config(),
        BuildContext(isa="gfx1250", widths=(8,), deadline_s=1, copies={}),
    )


class TestMuseSparkStage0(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(server_args(), role="test")

    def reasons(self, *, cfg=None, hf=None, isa="gfx1250"):
        return binding.monokernel_disable_reasons(
            cfg=resolving_view(server_args() if cfg is None else cfg),
            backend="amd",
            model_config=ModelConfig(config() if hf is None else hf),
            isa=isa,
        )[1]

    def test_configuration_refusals_and_attention_backends(self):
        for backend in ("tlx", "aiter"):
            self.assertEqual(
                self.reasons(cfg=server_args(attention_backend=backend)), []
            )
        for changes, reason in (
            ({"tp_size": 8}, "tp_size"),
            ({"ep_size": 1}, "ep_size"),
            ({"ep_size": 8}, "ep_size"),
            ({"speculative_num_draft_tokens": 4}, "speculative_num_draft_tokens"),
            ({"speculative_algorithm": None}, "speculative_algorithm"),
            ({"kv_cache_dtype": "bf16"}, "kv_cache_dtype"),
            ({"page_size": 32}, "page_size"),
            ({"attention_backend": "triton"}, "decode attention backend"),
        ):
            with self.subTest(changes=changes):
                self.assertTrue(
                    any(reason in r for r in self.reasons(cfg=server_args(**changes)))
                )
        with patch.dict(registry._REGISTRY, clear=True):
            self.assertTrue(any("no spec" in r for r in self.reasons(isa="gfx950")))

    def test_nested_dimensions_and_layer_pattern_are_checked(self):
        for field, value, reason in (
            ("hidden_size", 4096, "hidden_size"),
            ("num_key_value_heads", 8, "num_key_value_heads"),
            ("num_experts_per_tok", 4, "num_experts_per_tok"),
            ("sliding_window", 1024, "sliding_window"),
            ("layer_types", ["full_attention"] * 62, "layer_types"),
            ("layer_rope_theta", [500000] * 62, "layer_rope_theta"),
            ("mlp_layer_types", ["dense"] * 62, "mlp_layer_types"),
        ):
            with self.subTest(field=field):
                hf = config()
                setattr(hf.text_config, field, value)
                self.assertTrue(any(reason in r for r in self.reasons(hf=hf)))

    def test_loaded_topology_and_mxfp4_experts_are_checked(self):
        model = FakeMuseModel()
        self.assertEqual(SPEC.model_disable_reasons(model, config(), {}), [])
        for tp, attn_tp, moe_tp, moe_ep, attn_dp, moe_dp in (
            (8, 8, 1, 8, 1, 1),
            (4, 4, 4, 1, 1, 1),
            (4, 4, 1, 1, 1, 4),
        ):
            with get_parallel().override(
                tp_size=tp,
                moe_dp_size=moe_dp,
                attn_tp_size=attn_tp,
                attn_dp_size=attn_dp,
                moe_tp_size=moe_tp,
                moe_ep_size=moe_ep,
            ):
                reasons = SPEC.model_disable_reasons(model, config(), {})
                self.assertTrue(any("requires (4, 1, 4)" in r for r in reasons))
        model.model.language_model.layers[30].mlp.mxfp4_checkpoint = False
        self.assertIn(
            "layer 30 requires MXFP4 experts",
            SPEC.model_disable_reasons(model, config(), {}),
        )

    def test_builtin_and_external_specs_resolve_to_their_own_isa(self):
        from msgspec.structs import replace

        external = replace(
            SPEC, devices={"gfx950": registry.DeviceLimits((8,), (8,), 1)}
        )
        with (
            patch.dict(registry._REGISTRY, clear=True),
            patch.dict(globals(), EXTERNAL_SPEC=external),
        ):
            registry.register(
                backend="amd",
                architecture="MuseSparkForConditionalGeneration",
                path=f"{__name__}:EXTERNAL_SPEC",
            )
            for isa, expected in (("gfx950", external), ("gfx1250", SPEC)):
                selected, why = registry.find_spec(
                    backend="amd",
                    architectures=["MuseSparkForConditionalGeneration"],
                    isa=isa,
                )
                self.assertIs(selected, expected)
                self.assertEqual(why, "")

    def test_capture_bucket_includes_one_eight_row_request(self):
        args = server_args(
            cuda_graph_config=with_phase(
                default_cuda_graph_config(), Phase.DECODE, bs=[2, 4, 8]
            )
        )
        with (
            patch.object(binding, "device_isa", return_value="gfx1250"),
            patch.object(
                binding, "model_config_of", return_value=ModelConfig(config())
            ),
        ):
            binding.handle_monokernel(args)
        graph = resolving_view(args).cuda_graph_config
        self.assertEqual(list(graph.decode.bs), [1, 2, 4, 8])
        served = binding._served_keys(
            mode="target_verify", widths=SPEC.widths("gfx1250"), tokens=8
        )
        self.assertEqual(served, frozenset({ShapeKey("target_verify", 1, 8)}))

    def test_stage0_matches_the_native_forward_and_keeps_all_handoffs(self):
        for supplied_embeds, media in ((False, False), (True, False), (False, True)):
            with self.subTest(supplied_embeds=supplied_embeds, media=media):
                model = FakeMuseModel()
                ids, positions = torch.arange(8), torch.arange(8)
                batch = Batch(input_ids=ids, media=media)
                embeds = (
                    torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
                    if supplied_embeds
                    else None
                )
                input_ids = None if supplied_embeds else ids
                expected = model(input_ids, positions, batch, input_embeds=embeds)
                backbone = model.model.language_model
                expected_calls = list(backbone.calls)
                backbone.calls.clear()
                built = build(model)
                binding._install_forward(
                    model=model,
                    built=built,
                    spec=SPEC,
                    served=binding._served_keys(
                        mode="target_verify", widths=(8,), tokens=8
                    ),
                )
                with patch.object(
                    binding, "get_attn_backend", return_value=DecodeBackend()
                ):
                    actual = model(input_ids, positions, batch, input_embeds=embeds)
                self.assertTrue(torch.equal(actual[0], expected[0]))
                self.assertEqual([c[0] for c in backbone.calls], list(range(62)))
                for got, want in zip(backbone.calls, expected_calls):
                    self.assertTrue(torch.equal(got[1], want[1]))
                    self.assertEqual(got[3:6], want[3:6])
                    self.assertTrue(torch.equal(got[6], positions))
                    if want[2] is None:
                        self.assertIsNone(got[2])
                    else:
                        for a, b in zip(got[2], want[2]):
                            self.assertTrue(torch.equal(a, b))
                self.assertEqual(len(actual[1]), 5)
                for got, want in zip(actual[1], expected[1]):
                    self.assertEqual(got.dtype, torch.bfloat16)
                    self.assertTrue(torch.equal(got, want))
                saved = [x.clone() for x in actual[1]]
                for layer in backbone.layers:
                    layer.last_stream.add_(100)
                self.assertTrue(
                    all(torch.equal(a, b) for a, b in zip(actual[1], saved))
                )
                if media:
                    self.assertEqual(model.media_calls, 2)

    def test_capture_selection_can_change_after_the_adapter_is_built(self):
        model = FakeMuseModel()
        built = build(model)
        backbone = model.model.language_model
        backbone.layers_to_capture = [2, 7]
        ids = torch.arange(8)
        batch = Batch(input_ids=ids)
        expected = model(ids, ids, batch)
        actual = built.forward(None, ids, ids, batch)
        self.assertEqual(len(actual[1]), 2)
        for got, want in zip(actual[1], expected[1]):
            self.assertTrue(torch.equal(got, want))
        model.capture_aux_hidden_states = False
        backbone.layers_to_capture = []
        self.assertIsNone(built.forward(None, ids, ids, batch)[1])


if __name__ == "__main__":
    unittest.main()
