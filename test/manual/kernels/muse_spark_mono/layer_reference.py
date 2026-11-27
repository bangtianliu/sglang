"""Coherent native layer composition with independent FP64 checks, GPU0 only."""

import json
from types import SimpleNamespace

import torch
from check_experts_complete import reference as expert_reference
from check_experts_complete import table_for
from check_experts_routed import oracle as expert_oracle
from check_route import native as native_route
from check_route import oracle as route_oracle
from check_tail import check, native_boundary, native_norm_gate, oracle_boundary
from reference_archive import checked_load, s1_key


def attention(f, q, left, context):
    """FP64 logical-cache attention; no candidate kernel is used by this oracle."""
    logical = torch.arange(
        max(0, context - 8 - left) if left >= 0 else 0, context, device="cuda"
    )
    physical = f.page_table.flatten()[logical // 16].long() * 16 + logical % 16
    if left >= 0:
        physical = f.mapping[physical]
    keys = f.cache_k.reshape(-1, 4, 64).float()[physical].double() * f.k_scale.double()
    values = (
        f.cache_v.reshape(-1, 4, 64).float()[physical].double() * f.v_scale.double()
    )
    q = q.reshape(8, 32, 64).double()
    rows = []
    for group in range(4):
        scores = (
            torch.einsum("mhd,nd->mhn", q[:, group * 8 : group * 8 + 8], keys[:, group])
            * 0.125
        )
        mask = logical[None, :] <= f.positions[:, None]
        if left >= 0:
            mask &= logical[None, :] >= f.positions[:, None] - left
        scores = scores.masked_fill(~mask[:, None, :], -float("inf"))
        rows.append(torch.einsum("mhn,nd->mhd", scores.softmax(-1), values[:, group]))
    return torch.cat(rows, 1).reshape(8, 2048)


def build(f, w, seed, context, left, *, expert_fp64=True):
    s1 = checked_load(s1_key(seed, context, left >= 0, left >= 0))["output"]
    q, k, v, gate = (t.reshape(8, -1) for t in s1)
    loc = f.mapping[f.locations] if left >= 0 else f.locations
    expected_k = f.cache_k.view(torch.uint8).clone()
    expected_v = f.cache_v.view(torch.uint8).clone()
    for cache, value, scale in ((expected_k, k, f.k_scale), (expected_v, v, f.v_scale)):
        cache.reshape(-1, 4, 64)[loc] = (
            (value.reshape(8, 4, 64).float() / scale)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
    # Temporarily use native S1 cache writes for the full layer reference.
    saved_k = f.cache_k.view(torch.uint8).clone()
    saved_v = f.cache_v.view(torch.uint8).clone()
    f.cache_k.view(torch.uint8).copy_(expected_k)
    f.cache_v.view(torch.uint8).copy_(expected_v)
    attn64 = attention(f, q, left, context)
    f.cache_k.view(torch.uint8).copy_(saved_k)
    f.cache_v.view(torch.uint8).copy_(saved_v)
    attn = attn64.bfloat16()
    norm = native_norm_gate(attn, gate)
    d3 = checked_load(f"S3-inputs-{seed}.pt")
    partial3 = torch.stack(
        [(norm.double() @ wr.double().T).bfloat16() for wr in d3["weight"]]
    )
    total3 = torch.zeros((8, 8192), device="cuda")
    for r in range(4):
        total3 = total3 + partial3[r].float()
    x4, mid = native_boundary(
        total3.bfloat16(), f.stream, w["pre3"], w["alpha3"], w["beta3"]
    )
    for name, a, b in zip(
        ("x4", "mid_stream"),
        (x4, mid),
        oracle_boundary(
            total3.bfloat16(), f.stream, w["pre3"], w["alpha3"], w["beta3"]
        ),
    ):
        named_check(a, b, 0.015, f"reference {name}", {"_rank": 0})
    scores, ids, pool, *_ = native_route(
        x4, w["joint_router_pooler"], w["bias"], w["route_norm"], 0
    )
    os, oi, op = route_oracle(x4, w["joint_router_pooler"], w["bias"], w["route_norm"])
    exact_check(ids, oi, "reference native-FP64 routing IDs", {"_rank": 0})
    named_check(scores, os, 0.015, "reference routing scores", {"_rank": 0})
    named_check(pool, op, 0.015, "reference routing pool", {"_rank": 0})
    saved = checked_load(f"phase3e1-inputs-{seed}.pt")["weights"]
    weights = {
        n: torch.stack([t.roll(r * 11, 0) for r in range(4)]) for n, t in saved.items()
    }
    ref5 = expert_reference(pool, ids, scores, weights, w["gamma5"], table_for(ids))
    oracle5 = []
    if expert_fp64:
        for r in range(4):
            oracle5.extend(
                expert_oracle(
                    pool,
                    ids - r * 64,
                    {n: t[r] for n, t in weights.items()},
                    SimpleNamespace(inter=ref5["inter"][r], output=ref5["y"][r]),
                )
            )
    x6 = ref5["reduced"].bfloat16()
    proj6 = (x6.double() @ w["unpooler"].double().T).bfloat16()
    out, stream = native_boundary(
        proj6, mid, w["pre6"], w["alpha6"], w["beta6"], post=w["post6"]
    )
    for name, a, b in zip(
        ("out", "stream"),
        (out, stream),
        oracle_boundary(
            proj6, mid, w["pre6"], w["alpha6"], w["beta6"], post=w["post6"]
        ),
    ):
        named_check(a, b, 0.015, f"reference {name}", {"_rank": 0})
    return dict(
        q=q,
        k=k,
        v=v,
        gate=gate,
        cache_k=expected_k,
        cache_v=expected_v,
        initial_cache_k=saved_k,
        initial_cache_v=saved_v,
        attn=attn64,
        partial3=partial3,
        x4=x4,
        mid_stream=mid,
        ids=ids,
        scores=scores,
        pool=pool,
        partial5=ref5["partial"],
        x6=x6,
        out=out,
        stream=stream,
        expert_oracle=oracle5,
        seed=seed,
    )


def checker_failure(name, ref, location, **detail):
    record = dict(
        event="checker-failure",
        tensor=name,
        rank=ref["_rank"],
        location=list(location),
        **detail,
    )
    print(json.dumps(record), flush=True)
    raise AssertionError(record)


def require(condition, name, ref, location, **detail):
    if not bool(condition):
        checker_failure(name, ref, location, **detail)


def exact_check(actual, expected, name, ref):
    mismatch = actual != expected
    if bool(mismatch.any()):
        location = mismatch.nonzero()[0].tolist()
        index = tuple(location)
        value = (
            expected[index].item() if isinstance(expected, torch.Tensor) else expected
        )
        checker_failure(
            name,
            ref,
            location,
            actual=actual[index].item(),
            expected=value,
            location_kind="first mismatch",
        )


def named_check(actual, expected, limit, name, ref, *, row_offset=0):
    try:
        return check(actual, expected, limit)
    except AssertionError as error:
        a, b = actual.double(), expected.double()
        difference = (a - b).abs()
        flat = int(difference.argmax())
        cols = a.shape[-1]
        row = flat // cols + row_offset
        column = flat % cols
        detail = dict(
            event="value-failure",
            tensor=name,
            rank=ref["_rank"],
            row=row,
            column=column,
            location=[row, column],
            actual=float(a.flatten()[flat]),
            expected=float(b.flatten()[flat]),
            max_abs=float(difference.max()),
            per_row_max=difference.reshape(-1, cols).amax(1).tolist(),
            teacher_forced=ref.get("teacher_forced", False),
        )
        print(json.dumps(detail), flush=True)
        torch.save(
            dict(detail=detail, actual=actual.cpu(), expected=expected.cpu()),
            "/home/bangtliu/coord/logs/phase42-value-failure.pt",
        )
        raise AssertionError(detail) from error


def routing_check(ws, ref, w):
    # Stage ordering is exact on identical input. Across the composed layer,
    # upstream rounding may reorder near ties; bound cutoff changes by 2*delta.
    native_scores, native_ids, native_pool, *_ = native_route(
        ws.buffers["x4"], w["joint_router_pooler"], w["bias"], w["route_norm"], 0
    )
    exact_check(ws.s4.ids, native_ids, "identical-X4 routing IDs", ref)

    def raw(x):
        p = x.double() @ w["joint_router_pooler"].double().T
        return ((p[:, 4096:4352] + p[:, 4352:]) * 32**-0.5).sigmoid()

    rr = raw(ref["x4"])
    cc = raw(ws.buffers["x4"])
    rb = rr + w["bias"].double()
    cb = cc + w["bias"].double()
    delta = (cb - rb).abs().amax(1)
    threshold = 2 * delta
    ri = rb.argsort(descending=True, stable=True)
    sv = rb.gather(1, ri)
    routing = []
    for row in range(8):
        ci = ws.s4.ids[row].long()
        expected = ref.get("original_ids", ref["ids"])[row].long()
        require(
            ci.unique().numel() == 8,
            "routing IDs",
            ref,
            (row, 0),
            reason="duplicate expert",
            ids=ci.tolist(),
        )
        same = torch.equal(ci.sort().values, expected.sort().values)
        margin = sv[row, 7] - sv[row, 8]
        if not same:
            require(
                margin <= threshold[row],
                "routing cutoff scores",
                ref,
                (row, 7),
                reason="expert set outside measured tie bound",
                margin=float(margin),
                threshold=float(threshold[row]),
            )
            added = ci[~torch.isin(ci, expected)]
            removed = expected[~torch.isin(expected, ci)]
            require(
                (rb[row, removed].max() - rb[row, added].min()) <= threshold[row],
                "routing cutoff scores",
                ref,
                (row, 7),
                reason="nonlocal cutoff replacement",
                added=added.tolist(),
                removed=removed.tolist(),
            )

        routing.append(
            dict(
                row=row,
                set_equal=same,
                order_equal=torch.equal(ci, expected),
                perturbation=float(delta[row]),
                tie_threshold=float(threshold[row]),
                cutoff_margin=float(margin),
                cutoff_near_tie=bool(margin <= threshold[row]),
                candidate_ids=ci.tolist(),
                reference_ids=expected.tolist(),
            )
        )
    print(json.dumps(dict(event="routing-check", rows=routing)), flush=True)
    named_check(ws.s4.scores, native_scores, 0.001, "same-input routing scores", ref)
    named_check(ws.s4.pooled, native_pool, 0.015, "same-input routing pool", ref)
    for row in range(8):
        ci = ws.s4.ids[row].long()
        expected = ref.get("original_ids", ref["ids"])[row].long()
        named_check(
            ws.s4.scores[row],
            rr[row, ci] / rr[row, expected].sum(),
            0.01,
            "aligned routing scores",
            ref,
            row_offset=row,
        )
    return routing


def teacher_reference(f, w, ref, ws, rank, routes):
    """Native local S5/S6 for accepted cutoff rows, preserving actual peer payloads."""
    from pathlib import Path

    seed = ref["seed"]
    ids = ws.s4.ids.clone()
    scores = ws.s4.scores.clone()
    changed = torch.tensor(
        [not r["set_equal"] for r in routes], device=ids.device, dtype=torch.bool
    )
    snapshot = dict(
        rank=rank,
        routes=routes,
        ids=ids.cpu(),
        scores=scores.cpu(),
        pool=ws.s4.pooled.cpu(),
        inter=ws.s5.inter[rank].cpu(),
        expert_output=ws.s5.output[rank].cpu(),
        partial=ws.s5.partial.cpu(),
        actual={
            n: t.cpu()
            for n, t in ws.buffers.items()
            if n not in ("f1", "f3", "f4", "f6")
        },
        reference={
            n: ref[n].cpu()
            for n in (
                "partial5",
                "x6",
                "out",
                "stream",
                "ids",
                "scores",
                "pool",
                "mid_stream",
            )
        },
    )
    torch.save(
        snapshot,
        Path("/home/bangtliu/coord/logs")
        / f"phase42-cutoff-{seed}-{ws.context}-{ws.left}-r{rank}.pt",
    )
    saved = checked_load(f"phase3e1-inputs-{seed}.pt")["weights"]
    weights = {
        n: torch.stack([t.roll(r * 11, 0) for r in range(4)]) for n, t in saved.items()
    }
    native = expert_reference(
        ws.s4.pooled, ids, scores, weights, w["gamma5"], table_for(ids)
    )
    oracle = expert_oracle(
        ws.s4.pooled,
        ids - rank * 64,
        {n: t[rank] for n, t in weights.items()},
        SimpleNamespace(inter=native["inter"][rank], output=native["y"][rank]),
    )
    # Only the real rank's changed rows use candidate-conditioned local math.
    # Every stand-in peer partial is byte-for-byte the original launch payload.
    partial = ref["partial5"].clone()
    partial[rank, changed] = native["partial"][rank, changed]
    total = torch.zeros((8, 4096), device=ids.device, dtype=torch.float32)
    for src in range(4):
        total = total + partial[src].float()
    teacher_x6 = total.bfloat16()
    projection = (teacher_x6.double() @ w["unpooler"].double().T).bfloat16()
    teacher_out, teacher_stream = native_boundary(
        projection,
        ws.buffers["mid_stream"],
        w["pre6"],
        w["alpha6"],
        w["beta6"],
        post=w["post6"],
    )
    result = dict(ref)
    for name, value in (
        ("x6", teacher_x6),
        ("out", teacher_out),
        ("stream", teacher_stream),
    ):
        result[name] = ref[name].clone()
        result[name][changed] = value[changed]
    result.update(
        original_ids=ref["ids"],
        ids=ids,
        scores=scores,
        partial5=partial,
        teacher_forced=True,
        teacher_rows=changed.nonzero().flatten().tolist(),
        teacher_oracle_experts=len(oracle),
    )
    return result


def compare(f, ws, ref, w):
    result = {"routing": routing_check(ws, ref, w)}
    for name in (
        "q",
        "k",
        "v",
        "gate",
        "attn",
        "x4",
        "mid_stream",
        "x6",
        "out",
        "stream",
    ):
        limit = (
            0.008
            if name == "attn"
            else (
                0.02
                if ref.get("teacher_forced") and name in ("x6", "out", "stream")
                else 0.03
            )
        )
        result[name] = named_check(ws.buffers[name], ref[name], limit, name, ref)
    result["pool"] = named_check(ws.s4.pooled, ref["pool"], 0.03, "pool", ref)
    result["cache_boundaries"] = cache_check(f, ws, ref)
    return result


def cache_check(f, ws, ref):
    """Exact old bytes; adjacent new-slot FP8 flips only within measured midpoint bound."""
    slots = f.mapping[f.locations] if ws.left >= 0 else f.locations
    records = []
    for name, source, scale in (
        ("cache_k", "k", f.k_scale),
        ("cache_v", "v", f.v_scale),
    ):
        a = getattr(f, name).view(torch.uint8).reshape(-1, 4, 64)
        b = ref[name].reshape_as(a)
        old = ref["initial_" + name].reshape_as(a)
        untouched = torch.ones(a.shape[0], device=a.device, dtype=torch.bool)
        untouched[slots] = False
        stray = (a != old) & untouched[:, None, None]
        if bool(stray.any()):
            position = stray.nonzero()[0].tolist()
            index = tuple(position)
            checker_failure(
                name,
                ref,
                position,
                reason="stray cache write",
                actual=int(a[index]),
                expected=int(old[index]),
            )
        for slot, head, d in (a != b).nonzero().cpu().tolist():
            row = (slots == slot).nonzero().flatten()
            require(
                row.numel() == 1,
                name,
                ref,
                (slot, head, d),
                reason="cache mismatch outside new slots",
                mapped_rows=int(row.numel()),
            )
            row = int(row)
            ca, cb = int(a[slot, head, d]), int(b[slot, head, d])
            av = float(ws.buffers[source][row, head * 64 + d].float() / scale)
            bv = float(ref[source][row, head * 64 + d].float() / scale)
            decoded = (
                torch.tensor([ca, cb], dtype=torch.uint8)
                .view(torch.float8_e4m3fn)
                .float()
            )
            midpoint = float(decoded.mean())
            perturb = abs(av - bv)
            margin = abs(bv - midpoint)
            record = dict(
                rank=ref["_rank"],
                cache=name,
                slot=slot,
                query_row=row,
                head=head,
                dim=d,
                candidate_code=ca,
                reference_code=cb,
                code_distance=abs(ca - cb),
                candidate_value=av,
                reference_value=bv,
                midpoint=midpoint,
                reference_midpoint_margin=margin,
                prequant_perturbation=perturb,
                tie_threshold=2 * perturb,
            )
            records.append(record)
            print(json.dumps(dict(event="cache-boundary", **record)), flush=True)
            require(
                ca // 128 == cb // 128 and abs(ca - cb) == 1,
                name,
                ref,
                (slot, head, d),
                reason="non-adjacent FP8 mismatch",
                boundary=record,
            )
            quant = (
                torch.tensor([av, bv])
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
                .tolist()
            )
            require(
                quant == [ca, cb],
                name,
                ref,
                (slot, head, d),
                reason="prequant values do not explain cache codes",
                boundary=record,
                quantized=quant,
            )
            require(
                margin <= 2 * perturb,
                name,
                ref,
                (slot, head, d),
                reason="FP8 mismatch outside midpoint bound",
                boundary=record,
            )
    return records
