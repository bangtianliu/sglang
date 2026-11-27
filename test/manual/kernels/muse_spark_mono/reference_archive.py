"""Immutable, hashed native inputs/outputs for cross-stack GPU0 validation."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import msgspec
import torch
import triton
from bench_hbm import graph_times
from layer_fixture import LayerFixture, build_layer, require_gpu0

ROOT = Path("/home/bangtliu/coord/refs")
OLD = Path("/home/bangtliu/coord/logs")


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(cpu(v) for v in value)
    return value


def gpu(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, dict):
        return {k: gpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(gpu(v) for v in value)
    return value


def shapes(value):
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), dtype=str(value.dtype))
    if isinstance(value, dict):
        return {str(k): shapes(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [shapes(v) for v in value]
    return value


def record(path, stage, seed, context, producer, values):
    manifest = ROOT / "manifest.json"
    data = (
        json.loads(manifest.read_text())
        if manifest.exists()
        else dict(schema=1, artifacts={})
    )
    digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    data["artifacts"][path.name] = dict(
        stage=stage,
        seed=seed,
        context=context,
        sha256=digest,
        bytes=path.stat().st_size,
        producer=producer,
        tensors=shapes(values),
    )
    tmp = manifest.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(manifest)
    print(
        json.dumps(dict(artifact=path.name, stage=stage, seed=seed, sha256=digest)),
        flush=True,
    )


def save(name, value, stage, seed, context, method):
    path = ROOT / name
    if path.exists():
        raise FileExistsError(path)
    value = cpu(value)
    torch.save(value, path)
    record(
        path,
        stage,
        seed,
        context,
        dict(
            container="muse-mono-phase0",
            torch=str(torch.__version__),
            triton=str(triton.__version__),
            method=method,
        ),
        value,
    )


def checked_load(name, device=True):
    path = ROOT / name
    data = json.loads((ROOT / "manifest.json").read_text())["artifacts"][name]
    assert (
        hashlib.file_digest(path.open("rb"), "sha256").hexdigest() == data["sha256"]
    ), name
    value = torch.load(path, weights_only=True, map_location="cpu")
    return gpu(value) if device else value


def verify(stage):
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, entry in manifest["artifacts"].items():
        if entry["stage"] == stage:
            with (ROOT / name).open("rb") as f:
                assert (
                    hashlib.file_digest(f, "sha256").hexdigest() == entry["sha256"]
                ), name


def fixture(seed, context=8192):
    return LayerFixture(**checked_load(f"S1-inputs-{seed}-{context}.pt"))


def s1_key(seed, context, rope, mapping, skip=False):
    return f"S1-native-{seed}-{context}-r{int(rope)}-m{int(mapping)}-s{int(skip)}.pt"


def export_s1():
    from bench_s1_reference import project, reference

    configs = [
        (n, s, r, m, False)
        for n in (8192, 32768, 131072)
        for s in (7, 20261117)
        for r, m in ((False, False), (True, True))
    ]
    configs += [
        (8192, 17, r, m, True) for r, m in ((True, True), (True, False), (False, True))
    ]
    configs += [(32768, 7, r, m, False) for r, m in ((True, False), (False, True))]
    seen = set()
    for context, seed, rope, mapping, skip in configs:
        key = (seed, context)
        if key not in seen:
            f = build_layer(seed=seed, context=context)
            save(
                f"S1-inputs-{seed}-{context}.pt",
                msgspec.structs.asdict(f),
                "S1",
                seed,
                context,
                "seeded exact input fixture",
            )
            save(
                f"S1-projection-{seed}-{context}.pt",
                project(f),
                "S1",
                seed,
                context,
                "SGLang UnquantizedLinearMethod.apply, BF16 unfused linear (no _dense_partial)",
            )
            seen.add(key)
        f = fixture(seed, context)
        if skip:
            if mapping:
                f.mapping[f.locations[1]] = -1
            f.locations[0] = -1
        projected = checked_load(f"S1-projection-{seed}-{context}.pt")
        call = lambda: reference(f, rope=rope, mapping=mapping, projected=projected)
        output = call()
        torch.cuda.synchronize()
        loc = f.locations.clone()
        valid = loc >= 0
        if mapping:
            loc[valid] = f.mapping[loc[valid]]
        loc = loc[loc >= 0]
        value = dict(
            output=output,
            positions=loc,
            cache_k=f.cache_k.view(torch.uint8).reshape(-1, 4, 64)[loc],
            cache_v=f.cache_v.view(torch.uint8).reshape(-1, 4, 64)[loc],
            native_epilogue_timing=graph_times(call),
        )
        save(
            s1_key(seed, context, rope, mapping, skip),
            value,
            "S1",
            seed,
            context,
            "unfused BF16 linear projection + native qk_norm_rope_cache; cache rows only",
        )


def export_tail(stage):
    from aiter.tuned_gemm import tgemm
    from check_tail import fixture as tail_fixture
    from check_tail import native_boundary, native_norm_gate

    for seed in (7, 20261117):
        if stage == "S3":
            f, pre, alpha, beta = tail_fixture(seed)
            torch.manual_seed(seed)
            x = torch.randn((4, 8, 2048), device="cuda", dtype=torch.bfloat16)
            gate = torch.randn_like(x)
            weight = torch.randn(
                (4, 8192, 2048), device="cuda", dtype=torch.bfloat16
            ) * (2048**-0.5)
            d = dict(
                x=x,
                gate=gate,
                weight=weight,
                stream=f.stream,
                pre=pre,
                alpha=alpha,
                beta=beta,
            )
        else:
            d = gpu(
                torch.load(
                    OLD / f"phase3d-S6-inputs-{seed}.pt",
                    weights_only=True,
                    map_location="cpu",
                )
            )
        save(
            f"{stage}-inputs-{seed}.pt",
            d,
            stage,
            seed,
            None,
            "identical native-layout input tensors",
        )
        native = {}
        for sign in (1, -1):
            if stage == "S3":
                parts = torch.stack(
                    [
                        tgemm.mm(
                            native_norm_gate(d["x"][r] * sign, d["gate"][r]),
                            d["weight"][r],
                            None,
                            otype=torch.bfloat16,
                        )
                        for r in range(4)
                    ]
                )
                total = torch.zeros_like(d["stream"])
                for r in range(4):
                    total = total + parts[r].float()
                call = lambda: native_boundary(
                    total.bfloat16(), d["stream"], d["pre"], d["alpha"], d["beta"]
                )
                native[sign] = dict(
                    partial=parts, output=call(), timing=graph_times(call)
                )
            else:
                projection = tgemm.mm(
                    d["x"] * sign, d["weight"], None, otype=torch.bfloat16
                )
                native[sign] = dict(projection=projection, modes={})
                for kind in ("normal", "last_weighted", "last_unweighted"):
                    pre = None if kind == "last_unweighted" else d["pre"]
                    call = lambda: native_boundary(
                        projection,
                        d["stream"],
                        pre,
                        d["alpha"],
                        d["beta"],
                        post=d["post"],
                        last=kind != "normal",
                    )
                    native[sign]["modes"][kind] = dict(
                        output=call(), timing=graph_times(call)
                    )
        save(
            f"{stage}-native-{seed}.pt",
            native,
            stage,
            seed,
            None,
            "native norm/gate (S3), AITER BF16 tgemm.mm, ordered FP32 virtual-rank sum (S3), native fused_norm_residual boundary; no _dense_partial",
        )

    if stage == "S6":
        branch = torch.tensor(
            [0.75, 1.125], device="cuda", dtype=torch.bfloat16
        ).repeat(8, 4096)
        stream = torch.tensor([0.1234567, -0.479213], device="cuda").repeat(8, 4096)
        post = torch.tensor(
            [1.0078125, 0.9921875], device="cuda", dtype=torch.bfloat16
        ).repeat(4096)
        pre = torch.tensor(
            [1.015625, 0.9765625], device="cuda", dtype=torch.bfloat16
        ).repeat(4096)
        alpha = torch.full((8192,), 0.87, device="cuda")
        beta = torch.full((8192,), 0.42, device="cuda")
        outputs = {}
        for weighted in (False, True):
            for last in (False, True):
                outputs[f"{weighted}-{last}"] = native_boundary(
                    branch,
                    stream,
                    pre if weighted else None,
                    alpha,
                    beta,
                    post=post,
                    last=last,
                )
        save(
            "S6-boundary-patterns.pt",
            dict(
                branch=branch,
                stream=stream,
                post=post,
                pre=pre,
                alpha=alpha,
                beta=beta,
                outputs=outputs,
            ),
            "S6",
            None,
            None,
            "native fused_norm_residual, four exact two-value patterns",
        )


def archive_existing():
    names = []
    for seed in (7, 20261117):
        names += [("S4", seed, f"phase3d-S4-inputs-{seed}.pt")]
        names += [
            ("S4", seed, f"phase3d-S4-native-{seed}-ties{t}.pt") for t in (False, True)
        ]
        names += [
            ("S5", seed, name)
            for name in (
                f"phase3e1-inputs-{seed}.pt",
                f"phase3e1-native-{seed}.pt",
                f"phase3e1b-native-one8-{seed}.pt",
                f"phase3e1c-native-mixed-{seed}.pt",
            )
        ]
    for stage, seed, name in names:
        target = ROOT / name
        if target.exists():
            raise FileExistsError(target)
        shutil.copyfile(OLD / name, target)
        v = torch.load(target, weights_only=True, map_location="cpu")
        record(
            target,
            stage,
            seed,
            None,
            dict(
                source=str(OLD / name),
                native_container="muse-mono-gist-opt2",
                native_triton="3.9",
                method="previously validated native outputs on saved identical inputs"
                if "native" in name
                else "exact saved input tensors",
            ),
            v,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("S1", "S3", "S6", "existing"), required=True)
    a = p.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if a.stage == "existing":
        archive_existing()
        return
    require_gpu0()
    assert triton.__version__.split(".")[:2] == ["3", "8"]
    if a.stage == "S1":
        export_s1()
    else:
        export_tail(a.stage)


if __name__ == "__main__":
    main()
