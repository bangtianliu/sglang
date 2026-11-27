"""MK.8: 4 real GPUs (host-memory AR), A/B of T11 chain vs fused S3 variants.

VARIANTS (env, comma list): t11 | mk4 (S3 proj+AR+boundary in one launch) |
mk5b (mk4 + S4 projection as 144 exact half-tiles in the same launch).
Each variant has its own workspace and host regions. Correctness: each fused
variant's outputs must equal the T11 outputs of the same rank bit for bit (the
T11 run itself is checked by layer4_oracle.py). Timing: 10 layers back-to-back
per graph, variants interleaved, medians.
"""

import json
import os
import sys
import types
from datetime import timedelta
from pathlib import Path

assert os.environ.get("NCCL_P2P_DISABLE") == "1", "fabric must stay disabled"
RANK = int(os.environ["LOCAL_RANK"])
os.environ["HIP_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"] = str(RANK)
os.environ["MUSE_PEER_HOST"] = "1"

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from hostmem import HostShared  # noqa: E402

CONTEXTS = [(8192, -1), (8192, 2047), (131072, -1), (131072, 2047)]


def main():
    out_dir = Path(sys.argv[1])
    only = os.environ.get("CTX")
    variants = os.environ.get("VARIANTS", "t11,mk4,mk5b").split(",")
    rounds = int(os.environ.get("ROUNDS", "8"))
    torch.cuda.set_device(0)
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))

    def barrier():
        dist.monitored_barrier(timeout=timedelta(seconds=60))

    from layer_case import make

    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_standin
    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250 import stages
    from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import (
        tlx_chain,
        tlx_experts,
        tlx_projection,
        tlx_route,
        tlx_s3fused,
    )

    assert tlx_standin.HOST.value
    retained = []
    records = []
    for ci, (context, left) in enumerate(CONTEXTS):
        if only and only != f"{context}:{left}":
            continue
        f, w, m = make(7, context, left, RANK)
        # COPIES > 1: the 10-layer graph rotates distinct weight copies (cold, like 60 real layers)
        ncopies = int(os.environ.get("COPIES", "1"))
        wcopies = [w] + [{k_: (t_.clone() if isinstance(t_, torch.Tensor) else t_)
                          for k_, t_ in w.items()} for _ in range(ncopies - 1)]
        # the KV cache too (m[4], m[5]) unless KV_ROTATE=0: the same cache every layer could be
        # served from the memory-side cache, which 60 distinct layers never get
        kv_rotate = os.environ.get("KV_ROTATE", "1") == "1"
        mcopies = [m] + [tuple(t_.clone() if kv_rotate and i_ in (4, 5) else t_
                               for i_, t_ in enumerate(m)) for _ in range(ncopies - 1)]
        built = {}
        for vi, name in enumerate(variants):
            base_name, warm = name.split("+")[0].split("@")[0], name.endswith("+w")
            # "t11@tlx_experts.DEPTH2=2@...": current chain with stage-module overrides
            extra = {}
            from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import (
                tlx_workspace as _tws,
            )
            _tws.S2_MAX_SPLITS = 128
            partial = None
            for kv in name.split("+")[0].split("@")[1:]:
                k, v = kv.split("=")
                if k == "S2SPLITS":  # workspace-level knob, applies at make_workspace
                    _tws.S2_MAX_SPLITS = int(v)
                    continue
                if k == "PARTIAL":  # copies of only v experts, placed from real routing counts
                    partial = int(v)
                    continue
                mod, attr = k.split(".")
                extra[(getattr(stages, mod), attr)] = int(v)
            if base_name == "t11old":
                old_ws = tlx_chain.FUSE_NORM_FULL128
                tlx_chain.FUSE_NORM_FULL128 = False
            ws = tlx_chain.make_workspace("cuda", context=context, left=left)
            # setup tensors of the replicated-expert path must exist before any capture
            stages.tlx_route.repl_mine(ws.s4.sorted_ids)
            if base_name == "t11old":
                tlx_chain.FUSE_NORM_FULL128 = old_ws
            hosts = []
            for k, (stage, width) in enumerate(((ws.s3, 8192), (ws.s5, 4096))):
                words = tlx_standin.host_words(width, stage.peers.ctas)
                h = HostShared(f"muse_mk8_{os.getppid()}_{ci}_{vi}_{k}", words * 4, RANK, barrier)
                if RANK == 0:
                    h.zero()
                stage.peers.peers.fill_(h.dptr)
                hosts.append(h)
            shim3 = shim4 = None
            st4 = None
            trace = None
            if base_name in ("mk4", "mk5b", "mk4s4", "mk9"):
                st4 = tlx_s3fused.State("cuda")
                trace = torch.zeros(560 * 8, device="cuda", dtype=torch.int64)
                dbg = None
                if os.environ.get("DBG_HOST") and base_name == "mk5b":
                    dh = HostShared(f"muse_dbg_rank{RANK}", 144 * 16 * 4, 0, lambda: None, keep=True)
                    dh.zero()
                    dbg = dh.tensor(0, 144 * 16)
                    hosts.append(dh)
                s4x = base_name == "mk5b"
                s4args = ((w["joint_router_pooler"], ws.s4.projection, ws.s4.producer)
                          if base_name in ("mk5b", "mk4s4", "mk9") else None)
                s4q = base_name == "mk9"

                def at(x, gate, o_proj, stream, pre, alpha, beta, *, rank, workspace,
                       store_reduced, split_norm, output, failure, st4=st4, s4args=s4args,
                       s4x=s4x, trace=trace, dbg=dbg, warm=warm, s4q=s4q, **_):
                    return tlx_s3fused.tail(x, gate, o_proj, stream, pre, alpha, beta,
                                            rank=rank, ws=workspace, st=st4, output=output,
                                            failure=failure, split_norm=split_norm,
                                            store_reduced=store_reduced, nb=2, s4=s4args, nb4=4,
                                            s4x=s4x, trace=trace, dbg=dbg, warm=warm,
                                            s4q=s4q)
                shim3 = types.SimpleNamespace(attention_tail=at)
                if s4args is not None:
                    def routing_rows_only(x4, weight, bias, norm, *, rank=0, workspace=None,
                                          failure=None, **_):
                        W = workspace
                        failure = W.local_failure if failure is None else failure
                        tlx_route.finish_rows[(8,)](W.projection, bias, norm, W.scores, W.ids,
                                                    W.pooled, W.producer, W.rows_ready, failure,
                                                    num_warps=4, enable_fp_fusion=False)
                        tlx_route.slots[(64,)](W.ids, W.sorted_ids, W.expert_ids, W.num_post,
                                               W.rows_ready, W.state.status, failure,
                                               LOCAL_START=rank * 64, GRID=W.state.grid,
                                               num_warps=4, enable_fp_fusion=False)
                        return W.scores, W.ids, W.pooled, W.sorted_ids, W.expert_ids, W.num_post
                    shim4 = types.SimpleNamespace(routing=routing_rows_only)

            # t11old: the chain before MK.11-MK.14 (all fusions / early issue off)
            old_flags = ({(tlx_chain, "FUSE_MERGE_NORM"): False, (tlx_chain, "FUSE_ROWS_SLOTS"): False,
                          (tlx_chain, "FUSE_S1S2"): False, (tlx_projection, "EARLY_ISSUE"): False,
                          (tlx_experts, "DEPTH2"): 2, (tlx_experts, "WPE13"): 2,
                          (tlx_chain, "FUSE_NORM_FULL128"): False,
                          (tlx_projection, "DEPTH_S3"): 4,
                          (tlx_projection, "BN_S1"): 16, (tlx_projection, "BN_S3"): 16,
                          (tlx_projection, "BN_S4"): 16, (tlx_projection, "BN_S6"): 16,
                          (tlx_chain, "S6_DEPTHS"): {(8192, -1): 2, (8192, 2047): 4,
                                                     (131072, -1): 2, (131072, 2047): 2}}
                         if base_name == "t11old" else {})
            old_flags.update(extra)
            vw = wcopies
            if extra.get((stages.tlx_route, "REPL")):
                # second copies: rank r holds, after its own 64 experts, the experts whose copy B
                # is placed on r, compactly (group k = 0, 1, 2 in bit order), exactly as the
                # server's inputs.replicate_experts builds them. Rank a's weights are this
                # rank's rolled by (a - r) * 11, as layer_case.make builds them.
                masks = stages.tlx_route.repl_masks_full()
                if partial is not None:
                    from eplb_place import place

                    counts = json.loads((HERE / "real_counts" / os.environ.get(
                        "PARTIAL_COUNTS", "h13-17_heldout_route_counts.json")).read_text())
                    masks = place(counts["freq"][int(os.environ.get("PARTIAL_LAYER", "20"))], partial)
                    old_flags[(stages.tlx_route, "REPL_TABLE")] = masks
                hosted_ = [((RANK - 1 - k3) % 4,
                            [b_ for b_ in range(64) if masks[(RANK - 1 - k3) % 4][k3] >> b_ & 1])
                           for k3 in range(3)]
                n_extra = sum(len(ids_) for _, ids_ in hosted_)
                print(json.dumps(dict(rank=RANK, variant=name, hosted=n_extra)), flush=True)

                def _repl(w_):
                    out = dict(w_)
                    for key_ in ("w13", "w13_scale", "w2", "w2_scale"):
                        own = w_[key_]
                        assert own.shape[0] == 64, (key_, own.shape)
                        big = torch.empty((64 + n_extra,) + tuple(own.shape[1:]), dtype=own.dtype,
                                          device=own.device)
                        big[:64] = own
                        at = 64
                        for a_, ids_ in hosted_:
                            if ids_:
                                big[at:at + len(ids_)] = own.roll((a_ - RANK) * 11, 0)[ids_]
                                at += len(ids_)
                        out[key_] = big.contiguous()
                    return out
                vw = [_repl(w_) for w_ in wcopies]
                retained.append(vw)

            def launch(ws=ws, shim3=shim3, shim4=shim4, flags=old_flags, wi=0, wcopies=vw,
                       mcopies=mcopies):
                w = wcopies[wi % len(wcopies)]
                m = mcopies[wi % len(mcopies)]
                old3, old4 = tlx_chain.tlx_attn_tail, tlx_chain.tlx_route
                saved = {k: getattr(*k) for k in flags}
                for k, v in flags.items():
                    setattr(*k, v)
                if shim3 is not None:
                    tlx_chain.tlx_attn_tail = shim3
                if shim4 is not None:
                    tlx_chain.tlx_route = shim4
                try:
                    tlx_chain.launch(f.x, f.stream, w, m, ws, rank=RANK)
                finally:
                    tlx_chain.tlx_attn_tail, tlx_chain.tlx_route = old3, old4
                    for k, v in saved.items():
                        setattr(*k, v)

            retained.append((ws, hosts, st4, shim3, shim4, trace))
            if os.environ.get("ADDRS"):
                rng = {}
                def add(k, t):
                    if isinstance(t, torch.Tensor) and t.numel():
                        rng[k] = (hex(t.data_ptr()), hex(t.data_ptr() + t.numel() * t.element_size()))
                for k_, t_ in w.items():
                    add("w." + k_, t_)
                for k_, t_ in ws.buffers.items():
                    add("b." + k_, t_)
                for sn in ("s3", "s4", "s5"):
                    sub = getattr(ws, sn)
                    for f_ in getattr(type(sub), "__struct_fields__", ()):
                        add(sn + "." + f_, getattr(sub, f_))
                    if hasattr(sub, "peers"):
                        for f_ in ("scratch", "peers", "reduced"):
                            add(sn + ".peers." + f_, getattr(sub.peers, f_))
                if st4 is not None:
                    for k_, t_ in vars(st4).items():
                        add("st4." + k_, t_)
                for k_, h_ in enumerate(hosts):
                    rng[f"host{k_}"] = (hex(h_.dptr), hex(h_.dptr + h_.nbytes))
                add("trace", trace)
                print(json.dumps(dict(rank=RANK, variant=name, ranges=rng)), flush=True)
            built[name] = (ws, launch, trace, st4, warm)

        def snapshot(ws):
            d = {n: t.clone().cpu() for n, t in ws.buffers.items()
                 if n in ("x4", "mid_stream", "x6", "out", "stream", "attn")}
            d.update(blocks=ws.s4.num_post.clone().cpu() // 16)
            d.update(ids=ws.s4.ids.clone().cpu(), scores=ws.s4.scores.clone().cpu(),
                     pool=ws.s4.pooled.clone().cpu(), failure=ws.failure.clone().cpu())
            return d

        def diff(a, b):
            return {k: int((a[k].contiguous().view(-1).view(torch.uint8)
                            != b[k].contiguous().view(-1).view(torch.uint8)).sum()) for k in a}

        # Compile inside graph capture (no execution), so no rank runs an exchange
        # while another is still compiling: skew would push the exchanges into
        # their timeout paths (MK.5b's timeout path hangs, see monokernel-log).
        graphs, singles = {}, {}

        def kernels():
            return {id(k): k for dc in tlx_s3fused.s3_fused.device_caches.values()
                    for k in dc[0].values()}

        from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import tlx_code_warm
        for name, (ws, launch, _t, st4, warm) in built.items():
            if not warm:
                continue
            # compile in a throwaway capture, then point code_warm at this kernel's .text
            before = kernels()
            gt = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gt):
                launch()
            retained.append(gt)
            new = [k for i, k in kernels().items() if i not in before]
            assert len(new) == 1, (name, len(new))
            st4.code = tlx_code_warm.text_range(new[0])
        for name, (ws, launch, *_r) in built.items():
            g1 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g1):
                launch()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for li in range(10):
                    launch(wi=li)
            singles[name], graphs[name] = g1, g
            print(json.dumps(dict(rank=RANK, progress=f"{name} captured")), flush=True)
        retained.append((graphs, singles))
        if RANK == 0:
            import re as _re2
            for dc in stages.tlx_route.rows_slots.device_caches.values():
                for k_ in dc[0].values():
                    asm = k_.asm.get("amdgcn", "")
                    info = {key: _re2.findall(r"\." + key + r":\s*(\d+)", asm)[:1]
                            for key in ("private_segment_fixed_size", "vgpr_count", "sgpr_count",
                                        "vgpr_spill_count", "group_segment_fixed_size")}
                    (out_dir / f"rows_slots-{asm.count(chr(10))}.s").write_text(asm)
                    print(json.dumps(dict(kernel="rows_slots", lines=asm.count("\n"),
                                          n_regs=getattr(k_, "n_regs", None),
                                          n_spills=getattr(k_, "n_spills", None), **info)), flush=True)
        if RANK == 0:
            for dc in tlx_s3fused.s3_fused.device_caches.values():
                for k_ in dc[0].values():
                    md = k_.metadata
                    print(json.dumps(dict(kernel="s3_fused", n_regs=getattr(k_, "n_regs", None),
                                          n_spills=getattr(k_, "n_spills", None),
                                          shared=getattr(md, "shared", None),
                                          s4x=("S4X" in str(getattr(k_, "src", "")) ))), flush=True)
                    asm = k_.asm.get("amdgcn", "")
                    import re as _re
                    for key in ("private_segment_fixed_size", "vgpr_count", "sgpr_count",
                                "vgpr_spill_count", "sgpr_spill_count", "group_segment_fixed_size"):
                        m_ = _re.findall(r"\." + key + r":\s*(\d+)", asm)
                        print(json.dumps(dict(key=key, vals=m_[:3])), flush=True)
        snaps = {}
        for name, (ws, launch, *_r) in built.items():
            for _ in range(2):
                ws.failure.zero_()
                torch.cuda.synchronize()
                barrier()
                singles[name].replay()
                torch.cuda.synchronize()
            snaps[name] = snapshot(ws)
        row = dict(rank=RANK, context=context, left=left)
        base = variants[0]
        for name in variants[1:]:
            d = diff(snaps[name], snaps[base])
            row[f"{name}_vs_{base}_bytes"] = sum(d.values())
            row[f"{name}_vs_{base}_ids_bytes"] = d["ids"]
            row[f"{name}_vs_{base}_max_abs"] = {
                k_: float((snaps[name][k_].float() - snaps[base][k_].float()).abs().max())
                for k_ in ("attn", "out") if snaps[name][k_].is_floating_point()}
        row["blocks"] = {n: int(snaps[n]["blocks"].view(-1)[0]) for n in variants}
        row["failures"] = {n: snaps[n]["failure"].tolist() for n in variants}
        times = {n: [] for n in variants}
        failed = {n: [] for n in variants}
        for r in range(rounds):
            order = variants if r % 2 == 0 else variants[::-1]
            for name in order:
                built[name][0].failure.zero_()
                torch.cuda.synchronize()
                barrier()
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                graphs[name].replay()
                e.record()
                torch.cuda.synchronize()
                times[name].append(s.elapsed_time(e) * 100)  # us per layer
                fl = built[name][0].failure.tolist()
                if fl[0]:
                    failed[name].append(fl)
        row["timed_failures"] = {n: failed[n][:4] for n in variants}
        row["timed_failed_rounds"] = {n: len(failed[n]) for n in variants}
        post = {n: snapshot(built[n][0]) for n in variants}
        row["replay_vs_eager_bytes"] = {n: sum(diff(post[n], snaps[n]).values()) for n in variants}
        med = {n: round(sorted(t)[len(t) // 2], 1) for n, t in times.items()}
        row["layer_us"] = med
        for n in variants:
            tr = built[n][2]
            if tr is None:
                continue
            t = tr.view(560, 8)[:128].double().cpu()
            t0 = t[:, 0].min()
            rows = t[(t[:, 3] >= 120)]  # idx >= TILES-8: the row workers
            row[f"{n}_trace_us"] = dict(
                proj_done_med=round(float((t[:, 1] - t0).median()) / 100, 2),
                proj_done_max=round(float((t[:, 1] - t0).max()) / 100, 2),
                pushed_med=round(float((t[:, 2] - t0).median()) / 100, 2),
                pushed_max=round(float((t[:, 2] - t0).max()) / 100, 2),
                rows_all_arrived=round(float((rows[:, 4] - t0).max()) / 100, 2),
                rows_gathered=round(float((rows[:, 5] - t0).max()) / 100, 2),
                rows_normed=round(float((rows[:, 6] - t0).max()) / 100, 2))
            tq = tr.view(560, 8)[128:200].double().cpu()
            qq = tr.view(560, 8)[272:560, :5].double().cpu()
            if bool((qq[:, 0] > 0).all()):
                x4t = float((rows[:, 6] - t0).max()) / 100
                rel = (qq - t0) / 100
                row[f"{n}_quarters_us"] = dict(
                    x4_ready=round(x4t, 1),
                    start_q10_q50_q90_max=[round(float(rel[:, 0].quantile(v)), 1) for v in (.1, .5, .9, 1.)],
                    waited_q50_max=[round(float(rel[:, 1].median()), 1), round(float(rel[:, 1].max()), 1)],
                    dot_done_q50_max=[round(float(rel[:, 2].median()), 1), round(float(rel[:, 2].max()), 1)],
                    end_q50_max=[round(float(rel[:, 3].median()), 1), round(float(rel[:, 3].max()), 1)],
                    started_after_x4=int((rel[:, 0] > x4t).sum()),
                    loaded_minus_start_q50_max=[round(float((rel[:, 4] - rel[:, 0]).median()), 1), round(float((rel[:, 4] - rel[:, 0]).max()), 1)],
                    first_wave_loaded_max=round(float(rel[:, 4][rel[:, 0] < x4t].max()), 1),
                    dot_after_wait_q50=round(float((rel[:, 2] - rel[:, 1]).median()), 1),
                    dot_after_wait_late=round(float((rel[:, 2] - rel[:, 1])[rel[:, 0] > x4t].median()), 1) if int((rel[:, 0] > x4t).sum()) else None)
            if bool((tq[:, 1] > 0).all()):
                row[f"{n}_trace_us"]["s4q_done_med"] = round(float((tq[:, 1] - t0).median()) / 100, 2)
                row[f"{n}_trace_us"]["s4q_done_max"] = round(float((tq[:, 1] - t0).max()) / 100, 2)
            s4 = t[(t[:, 3] < 72) & (t[:, 7] > 0)]
            if len(s4):
                row[f"{n}_trace_us"]["s4_done_med"] = round(float((s4[:, 7] - t0).median()) / 100, 2)
                row[f"{n}_trace_us"]["s4_done_max"] = round(float((s4[:, 7] - t0).max()) / 100, 2)
        row["ratio_vs_" + base] = {n: round(med[n] / med[base], 3) for n in variants[1:]}
        print(json.dumps(row), flush=True)
        records.append(row)
        barrier()
    (out_dir / f"rank-{RANK}.json").write_text(json.dumps(records, indent=2) + "\n")
    barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
