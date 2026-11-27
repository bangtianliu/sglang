"""Per-rank state of the gfx1250 Muse Spark kernel chain inside SGLang.

One ``ChainRuntime`` per TP rank process, built once after the weights load and before any
CUDA graph capture. It owns the chain workspaces and the host-memory regions the ranks
exchange through (no GPU peer / P2P / IPC), and launches one layer per call.

Not yet run against a loaded model (see ~/coord/integration-plan.md).
"""

import os

import torch

from sglang.srt.kernel_plugins.amd.muse_spark import inputs

FIRST_LAYER, LAST_LAYER = 1, 60  # layers 0 and 61 keep the native layer
# probe (MK.72): serve only layers FIRST_LAYER..MUSE_CHAIN_LAST_LAYER with the chain, the rest
# natively, to split the per-step time into a per-layer cost and a fixed cost
LAST_LAYER = min(LAST_LAYER, int(os.environ.get("MUSE_CHAIN_LAST_LAYER", LAST_LAYER)))
# probe (MK.77): MUSE_LAYER_STAMPS="29,30": per-kernel realtime stamps inside those layers
LAYER_STAMPS = {int(v) for v in os.environ.get("MUSE_LAYER_STAMPS", "").split(",") if v}
_KERNEL_STAMPS = {}
# probe (MK.93): MUSE_ROUTE_STATS=1 copies every chain layer's 8 x 8 routed expert ids inside the
# step and prints, from a side thread on rank 0, the busiest rank's distinct experts under EP4,
# under the 2-copy levelled assignment and the ideal ceil(distinct / 4)
ROUTE_STATS = os.environ.get("MUSE_ROUTE_STATS") == "1"
# experiment (MK.93): MUSE_CHAIN_EXPERT_COPIES=2 holds every chain-layer expert on two ranks and
# assigns the routed experts per step (tlx_route.REPL); about twice the expert memory
EXPERT_COPIES = int(os.environ.get("MUSE_CHAIN_EXPERT_COPIES", "1"))
# ... on chain layers FIRST_LAYER .. MUSE_CHAIN_COPY_LAST_LAYER only (two full copies of all 60
# layers leave too little memory for the KV cache on a 288 GiB GPU)
COPY_LAST_LAYER = int(os.environ.get("MUSE_CHAIN_COPY_LAST_LAYER", "60"))
# ... placement: MUSE_CHAIN_COPY_FREQ = a MUSE_ROUTE_STATS_FILE of an earlier run (per-layer,
# per-expert routing counts) and MUSE_CHAIN_COPY_BUDGET = experts per layer that get a second
# copy (placement.place). Without a frequency file every expert is copied.
COPY_FREQ = os.environ.get("MUSE_CHAIN_COPY_FREQ")
COPY_BUDGET = int(os.environ.get("MUSE_CHAIN_COPY_BUDGET", "64"))
# MUSE_CHAIN_COPY_CHECK=1 (eager steps only, e.g. compare mode): after every chain layer read the
# slot tables back and check them against the layer's placement; prints "copy_check" events
COPY_CHECK = os.environ.get("MUSE_CHAIN_COPY_CHECK") == "1"
COPY_WARM = os.environ.get("MUSE_CHAIN_COPY_WARM", "1") == "1"  # code warming in rows_slots


def _route_loads(ids, passes=3):
    present = sorted(e for e in set(ids) if 0 <= e < 256)
    g = [[0] * 3 for _ in range(4)]
    for e in present:
        g[e >> 6][((e & 63) * 3) >> 6] += 1
    m = [[0] * 3 for _ in range(4)]
    ep4 = [sum(x) for x in g]
    load = list(ep4)
    for _ in range(passes):
        for a in range(4):
            for k in range(3):
                t = (a + 1 + k) & 3
                d = load[a] - load[t]
                h = d >> 1 if d >= 0 else -((-d) >> 1)
                mv = max(min(g[a][k] - m[a][k], h), -m[a][k])
                m[a][k] += mv
                load[a] -= mv
                load[t] += mv
    return ep4, load, len(present)
ROWS = 8
SLIDING_LEFT = 2047  # preceding keys of a sliding layer (span 2048)


def _require_host_mode():
    """The exchange mode is read once, when the kernel framework is first imported."""
    if os.environ.get("MUSE_PEER_HOST") != "1":
        raise RuntimeError(
            "the gfx1250 Muse Spark chain exchanges through host memory: set "
            "MUSE_PEER_HOST=1 before the kernel plugin is imported"
        )


class ChainRuntime:
    def __init__(self, backbone, *, rank: int, barrier, context: int, name: str,
                 device=None, layer_weights=None, step_metadata=None, copies: int = 2):
        """``barrier`` blocks until every TP rank has reached the same call (CPU group).
        ``context``: the attention geometry the workspaces are built for (8192 or 131072).
        ``name``: unique per server (the ranks derive the shared file names from it).
        ``layer_weights(i)`` / ``step_metadata(i, positions, forward_batch)`` replace the
        model-backed operands (harnesses without a loaded model); ``copies`` = workspaces per
        layer kind."""
        _require_host_mode()
        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
            host_shared,
            tlx_standin,
        )
        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_chain

        assert tlx_standin.HOST.value
        if context not in (8192, 131072):
            raise ValueError(f"context {context}: the chain is built for 8192 or 131072")
        self._chain = tlx_chain
        self._backbone = backbone
        self.rank = rank
        device = next(backbone.parameters()).device if device is None else device
        if layer_weights is None:
            layer_weights = lambda i: inputs.layer_weights(backbone, i)  # noqa: E731
        self._length = None
        if step_metadata is None:
            # KV length of the step when the attention backend does not precompute it
            length = torch.zeros(1, device=device, dtype=torch.int32)
            self._length = length  # must outlive every launch (graph replays read it)
            step_metadata = lambda i, positions, forward_batch: inputs.step_metadata(  # noqa: E731
                backbone.layers[i], positions, forward_batch, length
            )
        self._metadata = step_metadata
        self._copies = copies
        self._weights = {i: layer_weights(i) for i in range(FIRST_LAYER, LAST_LAYER + 1)}
        self._two_copies, self._route = {}, None  # layer -> placement masks
        if EXPERT_COPIES == 2:
            from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_route

            from sglang.srt.kernel_plugins.amd.muse_spark import placement

            self._route = tlx_route
            freq = None
            if COPY_FREQ:
                import json

                with open(COPY_FREQ) as f:
                    freq = json.load(f)["freq"]
            for i in range(FIRST_LAYER, min(LAST_LAYER, COPY_LAST_LAYER) + 1):
                self._two_copies[i] = (tlx_route.repl_masks_full() if freq is None
                                       else placement.place(freq[i], COPY_BUDGET))
            # every layer's expert tensors get the same length on this rank (see
            # inputs.replicate_experts): the most copies any layer hosts here, plus the
            # MUSE_CHAIN_COPY_PAD spare
            hosted = {i: tlx_route.repl_bases(m)[rank][3] for i, m in self._two_copies.items()}
            most = max(hosted.values()) + int(os.environ.get("MUSE_CHAIN_COPY_PAD", "0"))
            for i, masks in self._two_copies.items():
                self._weights[i] = inputs.replicate_experts(
                    backbone, i, self._weights[i], masks, rank=rank, name=name, barrier=barrier,
                    pad=most - hosted[i],
                )
            torch.cuda.empty_cache()
            if rank == 0:
                import json

                hosted = [sum(bin(m[(rank - 1 - k) % 4][k]).count("1") for k in range(3))
                          for m in self._two_copies.values()]
                print(json.dumps(dict(
                    event="expert_copies", layers=len(self._two_copies),
                    hosted_on_rank0=sum(hosted),
                    allocated_gib=round(torch.cuda.memory_allocated(device) / 2**30, 1),
                    reserved_gib=round(torch.cuda.memory_reserved(device) / 2**30, 1))), flush=True)
        # Two workspaces per layer kind, used alternately, so a layer never writes the
        # buffers its own inputs alias (three sliding layers run in a row).
        self._workspaces, self._hosts = {}, []
        for kind, left in (("full", -1), ("sliding", SLIDING_LEFT)):
            for copy in range(copies):
                ws = tlx_chain.make_workspace(device, context=context, left=left)
                for k, (stage, width) in enumerate(((ws.s3, 8192), (ws.s5, 4096))):
                    words = tlx_standin.host_words(width, stage.peers.ctas)
                    host = host_shared.HostShared(
                        f"{name}_{kind}{copy}_{k}", words * 4, rank, barrier
                    )
                    if rank == 0:
                        host.zero()
                    stage.peers.peers.fill_(host.dptr)
                    self._hosts.append(host)  # must outlive every launch
                self._workspaces[kind, copy] = ws
                if EXPERT_COPIES == 2:  # setup tensors must exist before any graph capture
                    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
                        tlx_route as _route,
                    )

                    _route.repl_mine(ws.s4.sorted_ids)
        barrier()
        self._barrier = barrier
        self._turn = {"full": 0, "sliding": 0}
        self._compiled = set()  # layer kinds whose kernels every rank has compiled
        # [0] failed workspace-steps since start, [1 + k] last failed stage of workspace k
        # (written by the step_reset kernels at the start of every step; MK.77)
        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_guard

        self._guard = tlx_guard
        self._fail_count = torch.zeros(1 + len(self._workspaces), device=device,
                                       dtype=torch.int32)
        if os.environ.get("MUSE_CHAIN_FAILURE_LOG", "1") == "1":
            import threading

            threading.Thread(target=self._log_failures, daemon=True).start()
        self._route_ids = None
        if ROUTE_STATS:
            import threading

            self._route_ids = torch.zeros((LAST_LAYER + 1, 64), device=device, dtype=torch.int32)
            # exact counts: [layer][expert] = routes (row, slot) that picked the expert, every step
            self._route_counts = torch.zeros((LAST_LAYER + 1, 256), device=device,
                                             dtype=torch.int64)
            self._route_ones = torch.ones(64, device=device, dtype=torch.int64)
            if rank == 0:
                threading.Thread(target=self._log_routes, daemon=True).start()

    def _log_routes(self):
        import json
        import time

        acc, n = [0.0] * 6, 0
        path = os.environ.get("MUSE_ROUTE_STATS_FILE")
        while True:
            time.sleep(0.5)
            v = self._route_ids.clone().cpu()[FIRST_LAYER:]
            if int(v.abs().sum()) == 0:
                continue
            for row in v.tolist():
                ep4, lev, distinct = _route_loads(row)
                acc = [acc[0] + max(ep4), acc[1] + max(lev), acc[2] + -(-distinct // 4),
                       acc[3] + distinct, acc[4] + min(ep4), acc[5] + 1]
            n += 1
            if n % 20 == 0:
                c = acc[5]
                print(json.dumps(dict(event="route_stats", samples=n, layers=int(c),
                                      distinct=round(acc[3] / c, 2), ep4_busiest=round(acc[0] / c, 2),
                                      ep4_idlest=round(acc[4] / c, 2),
                                      two_copy_busiest=round(acc[1] / c, 2),
                                      ideal=round(acc[2] / c, 2))), flush=True)
                if path:
                    with open(path + ".tmp", "w") as f:
                        json.dump(dict(samples=n, freq=self._route_counts.cpu().tolist()), f)
                    os.replace(path + ".tmp", path)

    def failure_count(self):
        """(failed workspace-steps since start, {workspace: last failed stage}); host sync."""
        v = self._fail_count.tolist()
        return v[0], {key: v[1 + k] for k, key in enumerate(self._workspaces) if v[1 + k]}

    def _log_failures(self):
        import json
        import time

        seen = 0
        while True:
            time.sleep(15)
            count, stages = self.failure_count()
            if count != seen:
                seen = count
                print(json.dumps(dict(event="chain_failures", rank=self.rank, count=count,
                                      last_stage={f"{k[0]}{k[1]}": v for k, v in stages.items()})),
                      flush=True)

    def _kind(self, i: int) -> str:
        return "full" if i % 4 == 1 else "sliding"

    def begin_step(self, forward_batch=None, rows: int = ROWS):
        """Call once per forward, before layer FIRST_LAYER (also during graph capture).

        Fills the step's KV length (sequence length + this step's rows) once, on device, when the
        attention backend does not precompute it: every layer of the step reads the same buffer.
        """
        self._turn = {"full": 0, "sliding": 0}
        # A failed exchange (a peer more than the wait bound late) must not outlive its step.
        for k, ws in enumerate(self._workspaces.values()):
            self._guard.step_reset[(1,)](ws.failure, self._fail_count, K=k, num_warps=1)
        if self._length is not None and forward_batch is not None:
            self._length.copy_(forward_batch.seq_lens[:1])  # casts on device, allocates nothing
            self._length.add_(rows)

    def _check_copies(self, i, ws):
        """Host check of one executed layer: every block's local expert index must be below
        64 + (copies hosted on this rank in this layer), and must be the index the Python rule
        gives for the routed ids. Logs the first violations and a running count."""
        import json

        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import tlx_route
        from sglang.srt.kernel_plugins.amd.muse_spark import placement

        torch.cuda.synchronize()
        masks = self._two_copies[i]
        bases = tlx_route.repl_bases(masks)
        hosted = bases[self.rank][3]
        nblk = int(ws.s4.num_post.item()) // 16
        experts = ws.s4.expert_ids[:max(nblk, 0)].tolist() if 0 <= nblk <= 64 else []
        ids = ws.s4.ids.view(-1).tolist()
        mine = tlx_route.repl_mine(ws.s4.sorted_ids).tolist()
        size = int(self._weights[i]["w13"].shape[0])
        want = placement.reference_locals(ids, masks, self.rank)
        st = self._copy_check = getattr(self, "_copy_check", dict(layers=0, bad=0, max_index=-1))
        st["layers"] += 1
        st["max_index"] = max([st["max_index"]] + experts)
        bad = (not 0 <= nblk <= 64 or any(not 0 <= e < 64 + hosted for e in experts)
               or sorted(experts) != sorted(want["locals"]) or mine != want["mine"])
        if bad:
            st["bad"] += 1
            if st["bad"] <= 5:
                print(json.dumps(dict(event="copy_check_bad", rank=self.rank, layer=i, nblk=nblk,
                                      hosted=hosted, tensor_experts=size, experts=experts,
                                      want=want["locals"], mine_ok=mine == want["mine"],
                                      ids=ids)), flush=True)
        if st["layers"] % 600 == 0:
            print(json.dumps(dict(event="copy_check", rank=self.rank, **st)), flush=True)

    def _set_copies(self, i):
        """The assignment kernels are selected by module flags read at launch (and capture)."""
        if self._route is not None:
            on = i in self._two_copies
            self._route.REPL = 1 if on else False
            self._route.REPL_TABLE = self._two_copies.get(i)
            self._route.CODE_WARM_RS = on and COPY_WARM

    def run_layer(self, i, positions, stream, forward_batch, attn_input):
        """Launch layer ``i``. Returns (stream, next attention input, None) like the native
        layer on this path, or None when this step or layer must use the native layer."""
        if not FIRST_LAYER <= i <= LAST_LAYER:
            return None
        if attn_input.shape != (ROWS, 8192) or stream.shape != (ROWS, 8192):
            return None
        if attn_input.dtype != torch.bfloat16 or stream.dtype != torch.float32:
            return None
        meta = self._metadata(i, positions, forward_batch)
        if meta is None:
            return None
        kind = self._kind(i)
        self._set_copies(i)
        ckey = (kind, i in self._two_copies)  # each kernel variant compiles on every rank first
        if ckey not in self._compiled and not torch.cuda.is_current_stream_capturing():
            self._compile(kind, i, positions, stream, forward_batch, attn_input, meta)
            self._compiled.add(ckey)
        ws = self._workspaces[kind, self._turn[kind]]
        self._turn[kind] = (self._turn[kind] + 1) % self._copies
        if i in LAYER_STAMPS:
            from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_stamp

            if i not in _KERNEL_STAMPS:
                _KERNEL_STAMPS[i] = tlx_stamp.KernelStamps(stream.device, f"layer{i}-{kind}")
            with _KERNEL_STAMPS[i].layer():
                out, new_stream = self._chain.launch(
                    attn_input, stream, self._weights[i], meta, ws, rank=self.rank
                )
            return new_stream, out, None
        out, new_stream = self._chain.launch(
            attn_input, stream, self._weights[i], meta, ws, rank=self.rank
        )
        if COPY_CHECK and i in self._two_copies and not torch.cuda.is_current_stream_capturing():
            self._check_copies(i, ws)
        if self._route_ids is not None:
            self._route_ids[i].copy_(ws.s4.ids.view(-1))
            self._route_counts[i].index_add_(0, ws.s4.ids.view(-1).long(), self._route_ones)
        return new_stream, out, None

    def _compile(self, kind, i, positions, stream, forward_batch, attn_input, meta):
        """Compile this kind's kernels on every rank before its first executed launch.

        Ranks compile at different speeds (seconds), and an exchange that one rank starts while
        another is still compiling gives up after its bound and sets the sticky failure word. So
        the first eager launch of each kind first runs inside a throwaway CUDA graph capture
        (compiles, executes nothing), then all ranks meet and clear their failure words. Every
        rank reaches the same layer of the same step, so the barrier is collective-safe.
        """
        graph = torch.cuda.CUDAGraph()
        copies = [self._workspaces[kind, c] for c in range(self._copies)]
        with torch.cuda.graph(graph):
            for ws in copies:  # every workspace copy, in case a copy compiles its own launch
                # the real operands: a compiled binary is keyed on each tensor's alignment
                self._chain.launch(attn_input, stream, self._weights[i], meta, ws,
                                   rank=self.rank)
        torch.cuda.synchronize()
        del graph
        self._barrier()
        self.reset_failures()
        torch.cuda.synchronize()
        self._barrier()

    def reset_failures(self):
        """A failed launch disables the later stages of its workspace until this is called
        (every rank, at the same point between steps)."""
        for ws in self._workspaces.values():
            ws.failure.zero_()

    def failures(self):
        """(workspace, stage, cta) of every workspace whose last launches failed (host sync;
        call outside graph capture, after the step)."""
        return [
            (key, *ws.failure.tolist())
            for key, ws in self._workspaces.items()
            if int(ws.failure[0]) != 0
        ]
