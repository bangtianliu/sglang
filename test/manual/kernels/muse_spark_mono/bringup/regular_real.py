"""Prepared rank-aware copy of regular_layer45; collective is real RCCL.

Gluon imports here belong to the separate native/reference graph only.
Not a production TLX module. Never executed during T12 preparation.
"""

from types import SimpleNamespace

import torch
from aiter.tuned_gemm import tgemm
from check_route import native as native_route
from check_tail import native_boundary
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

from sglang.kernels.ops.attention.qk_norm_rope_cache import qk_norm_rope_cache
from sglang.kernels.ops.attention.tlx.paged import unified_attention
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
    standin,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.common_shape import (
    CommonShape,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework.packed_memory import (
    store_bf16_vector,
)
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.qkvg import (
    _advance_tag,
)
from sglang.srt.rocm_muse_spark import muse_moe_triton as moe
from sglang.srt.rocm_muse_spark.fused_norm_gate import _rms_norm_gate_fwd_kernel


@g.jit
def local_collective(
    Partial,
    Scratch,
    Peers,
    Reduced,
    Epoch,
    Status,
    Out,
    WIDTH: gl.constexpr,
    PAY: gl.constexpr,
    FL: gl.constexpr,
):
    row = gl.program_id(0)
    tag = _advance_tag(Epoch, row)
    value, ok = standin.reduce_row(
        Partial, Scratch, Peers, Reduced, tag, row, WIDTH, 768, 0, PAY, FL
    )
    if ok:
        store_bf16_vector(Out + row * WIDTH, value, WIDTH)
    gl.store(Status + row, ok)


class Regular:
    def __init__(self, f, w, m, context, left, partials, rank, group):
        self.rank, self.group = rank, group
        self.f, self.w, self.m, self.context, self.left = f, w, m, context, left
        self.partials = partials
        self.k = f.cache_k.clone()
        self.v = f.cache_v.clone()
        if left >= 0:
            self.k = torch.zeros(
                (context // 16, 16, 4, 64), device="cuda", dtype=torch.float8_e4m3fn
            )
            self.v = torch.zeros_like(self.k)
            valid = f.mapping >= 0
            for dst, src in ((self.k, f.cache_k), (self.v, f.cache_v)):
                dst.view(torch.uint8).reshape(context, 4, 64)[valid] = src.view(
                    torch.uint8
                ).reshape(-1, 4, 64)[f.mapping[valid]]
        self.initial_k = self.k.view(torch.uint8).clone()
        self.initial_v = self.v.view(torch.uint8).clone()
        self.epochs = [
            torch.zeros((8, 32), device="cuda", dtype=torch.int32) for _ in range(2)
        ]
        self.status = [
            torch.empty((8,), device="cuda", dtype=torch.int32) for _ in range(2)
        ]
        self.peers = [
            standin.make_standins("cuda", width=width, ctas=768)
            for width in (8192, 4096)
        ]
        for p in self.peers:
            p.ready = torch.empty((1,), device="cuda", dtype=torch.int32)
        self.reduced = [
            torch.empty((8, width), device="cuda", dtype=torch.bfloat16)
            for width in (8192, 4096)
        ]
        self.inter = torch.empty((64, 4096), device="cuda", dtype=torch.bfloat16)
        self.y = torch.empty_like(self.inter)
        self.part5 = torch.empty((8, 4096), device="cuda", dtype=torch.bfloat16)
        self.attn = torch.empty((8, 32, 64), device="cuda", dtype=torch.bfloat16)
        self.cfg = moe._pick_config(8, 8, 64, True)
        self.outputs = {}
        shape = CommonShape()
        for width, e, p, st, out in zip(
            (8192, 4096), self.epochs, self.peers, self.status, self.reduced
        ):
            dummy = torch.empty((8, width), device="cuda", dtype=torch.bfloat16)
            k = local_collective.warmup(
                dummy,
                p.scratch,
                p.peers,
                p.reduced,
                e,
                st,
                out,
                width,
                p.payload,
                p.flags,
                grid=(8,),
                **shape.options(),
            )
            shape.admit("native-standin", k, grid=8)

    def prepare(self):
        for p, part, e in zip(self.peers, self.partials, self.epochs):
            standin.prepare(p, part, e, rank=0)

    def collective(self, part, index):
        import torch.distributed as dist

        value = part.float()
        dist.all_reduce(value, group=self.group)
        return value.bfloat16()

    def __call__(self, mark=None):
        f, w, m = self.f, self.w, self.m
        projected = tgemm.mm(f.x, w["qkvg"], None, otype=torch.bfloat16)
        q, k, v, gate = projected.split((2048, 256, 256, 2048), -1)
        qo, ko = qk_norm_rope_cache(
            q,
            k,
            v,
            f.positions,
            m[1],
            self.k,
            self.v,
            f.locations,
            None,
            1e-5,
            5.4730064863875,
            f.k_scale,
            f.v_scale,
        )
        if mark is not None:
            mark(1)
        unified_attention(
            q=qo.reshape(8, 32, 64),
            k=self.k,
            v=self.v,
            out=self.attn,
            cu_seqlens_q=f.query_start,
            seqused_k=f.seq_lens,
            max_seqlen_q=8,
            max_seqlen_k=self.context,
            softmax_scale=0.125,
            causal=True,
            window_size=(self.left, 0),
            block_table=f.page_table,
            k_descale=f.k_scale,
            v_descale=f.v_scale,
            softcap=0.0,
            q_descale=None,
        )
        if mark is not None:
            mark(2)
        ag = torch.empty((8, 2048), device="cuda", dtype=torch.bfloat16)
        _rms_norm_gate_fwd_kernel[(32,)](
            self.attn,
            gate,
            ag,
            None,
            1e-5,
            256,
            32,
            gate.stride(0),
            2048,
            HEADS_PER_GROUP=32,
            D=64,
            BD=64,
            BLOCK_M=8,
            HAS_WEIGHT=False,
            HAS_NORM=True,
            HAS_GATE=True,
            num_warps=4,
            enable_fp_fusion=False,
        )
        part3 = tgemm.mm(ag, w["o_proj"], None, otype=torch.bfloat16)
        total3 = self.collective(part3, 0)
        x4, mid = native_boundary(total3, f.stream, w["pre3"], w["alpha3"], w["beta3"])
        if mark is not None:
            mark(3)
        scores, ids, pool, *table = native_route(
            x4, w["joint_router_pooler"], w["bias"], w["route_norm"], self.rank
        )
        if mark is not None:
            mark(4)
        moe._launch_gemm(
            pool,
            w["w13"],
            w["w13_scale"],
            self.inter,
            *table,
            4096,
            4096,
            64,
            8,
            True,
            True,
            True,
            self.cfg,
        )
        moe._launch_gemm(
            self.inter,
            w["w2"],
            w["w2_scale"],
            self.y,
            *table,
            4096,
            4096,
            64,
            8,
            False,
            False,
            True,
            self.cfg,
        )
        moe.finalize_norm_weighted_sum_kernel[(8,)](
            self.y,
            w["gamma5"],
            scores,
            ids,
            self.part5,
            4096,
            self.rank * 64,
            64,
            1e-8,
            1.0,
            0,
            TOP_K=8,
            HAS_NORM=True,
            HAS_SCALE=True,
            BLOCK_H=4096,
            SPLIT_K=1,
            num_warps=4,
            enable_fp_fusion=False,
        )
        x6 = self.collective(self.part5, 1)
        if mark is not None:
            mark(5)
        projection = tgemm.mm(x6, w["unpooler"], None, otype=torch.bfloat16)
        out, stream = native_boundary(
            projection, mid, w["pre6"], w["alpha6"], w["beta6"], post=w["post6"]
        )
        if mark is not None:
            mark(6)
        self.outputs = dict(
            q=qo,
            k=ko,
            v=v,
            gate=gate,
            attn=self.attn.reshape(8, 2048),
            x4=x4,
            mid_stream=mid,
            x6=x6,
            out=out,
            stream=stream,
        )
        self.s4 = SimpleNamespace(ids=ids, scores=scores, pooled=pool)
        self.s5 = SimpleNamespace(
            inter=self.inter[None], output=self.y[None], partial=self.part5[None]
        )
        self.buffers = self.outputs
        return out, stream
