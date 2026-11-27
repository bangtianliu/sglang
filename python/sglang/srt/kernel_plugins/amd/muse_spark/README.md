# Muse Spark on gfx1250: stage 0

The built-in `amd` spec serves one DFlash request with eight verify rows at
attention TP4 / MoE TP1 / EP4 on a four-GPU SPX node, FP8 E4M3 KV, page size16, and the `tlx` or `aiter`
decode attention backend. The nested checkpoint config must match the released
62-layer Muse Spark dimensions and sliding/full pattern, with MXFP4 experts in
zero-based layers1–60.

Stage0 runs the loaded SGLang layer objects and declares bitwise numerics. It
preserves the FP32 residual stream, all three layer handoff values, selected
BF16 DFlash auxiliary states, the normalized text/media embedding path, and the
model's logits processor. The per-layer `_run_layer` call is the seam for future
kernel launches; boundary layers0 and61 continue to use their original layers.
It allocates no peer workspace and does not import the external `sgl_amd_mono`
plugin or require uTLX merely to import the spec.

The registry installs the spec lazily by path under the gfx1250 ISA key. Existing
external architecture-only gfx950 registrations remain valid fallbacks. Runtime
glue stays here; future kernels belong under
`sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250` and import nothing from srt.
On gfx1250, an ISA-specific architecture-only registration takes precedence
over a model-type-specific registration without an ISA key.

Phase1 validation is CPU-only. Real TP4 routing, graph capture, checkpoint math
and throughput remain unverified until an appropriate node and weights are
available. Peer-memory experiments require separate exact-script approval.
