"""LM head for 8 verify rows with the chain's projection kernel (MK.75).

logits[8, V] = x[8, 8192] @ W[V, 8192]^T, bf16 in, fp32 sums out (not rounded to bf16). In
the server the native LM head on this path (aiter tuned GEMM) also returns fp32 [8, V], which
is what lets DFlash keep the logits as a per-rank vocabulary shard and defer the cross-rank
gather (LogitsProcessor._can_gather_softcap_fused needs fp32). The K order of the sums differs
from the native GEMM, so values agree to rounding, not bit for bit. On gfx1250 the native GEMM streams the 790 MiB vocabulary shard
at ~1.4 TB/s (574 µs); the TDM projection kernel reaches ~11 TB/s (75 µs).

The projection kernel takes whole BN-column tiles: V // 64 tiles of 64 columns, then the
remaining V % 64 columns (a multiple of 16) as 16-column tiles on offset views. No weight copy.
"""

import torch

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_projection

K = 8192
DEPTH = 3


class Workspace:
    def __init__(self, device, vocab: int):
        assert vocab % 16 == 0, vocab
        self.vocab = vocab
        self.words = torch.zeros(8 * vocab, device=device, dtype=torch.int32)  # fp32 bits
        self.status = torch.zeros(vocab // 16, device=device, dtype=torch.int32)
        self.out = self.words.view(torch.float32).view(8, vocab)


def supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.shape == (8, K) and x.dtype == torch.bfloat16 and x.is_contiguous()
        and weight.dim() == 2 and weight.shape[1] == K and weight.dtype == torch.bfloat16
        and weight.is_contiguous() and weight.shape[0] % 16 == 0
    )


def launch(x: torch.Tensor, weight: torch.Tensor, ws: Workspace) -> torch.Tensor:
    """Returns ws.out (fp32 [8, V]); valid until the next launch."""
    v = ws.vocab
    assert weight.shape[0] == v
    common = dict(N=v, K=K, STAGE=6, SK=4, COPY="tdm", EARLY=True, ROUND_OUTPUT=False,
                  num_stages=1, enable_fp_fusion=False)
    wide = v // 64
    if wide:
        tlx_projection.project[(wide,)](
            x, weight, ws.words, ws.status, DEPTH=DEPTH, BN=64, num_warps=4, **common)
    done = wide * 64
    if done < v:
        # the last (v - done) / 16 tiles: the same row stride v, every pointer moved by `done`
        tlx_projection.project[((v - done) // 16,)](
            x, weight[done:], ws.words[done:], ws.status[done // 16:],
            DEPTH=4, BN=16, num_warps=2, **common)
    return ws.out
