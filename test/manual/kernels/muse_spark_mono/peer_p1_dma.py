"""P.1 proposal: run only after exact-script approval in coord/review.md.

Default invocation prints help without importing torch. The host driver records
kernel logs and launches the standard torch-copy worker in the dev container.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(
    "/home/bangtliu/sglang-mono/test/manual/kernels/muse_spark_mono/peer_p1_dma.py"
)
LOG_ROOT = Path("/home/bangtliu/coord/logs")
CONTAINER = "muse-mono-phase0"
BYTES = 64 * 1024 * 1024


def require_approval():
    digest = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    approval = Path("/home/bangtliu/coord/approvals") / f"{digest}.ok"
    if not approval.is_file():
        raise SystemExit(f"Multi-GPU execution refused: missing {approval}")


def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields)), flush=True)


def dmesg_snapshot(label, path):
    result = subprocess.run(
        ["dmesg", "--color=never"], capture_output=True, text=True, timeout=8
    )
    matches = [
        line
        for line in result.stdout.splitlines()
        if re.search(r"amdgpu|ualink", line, re.IGNORECASE)
    ]
    snapshot = dict(
        event="dmesg",
        label=label,
        returncode=result.returncode,
        stderr=result.stderr.strip(),
        lines=matches[-200:],
    )
    path.write_text(json.dumps(snapshot, indent=2) + "\n")
    emit(
        "dmesg_snapshot",
        label=label,
        path=str(path),
        **{k: v for k, v in snapshot.items() if k not in ("event", "label")},
    )
    if result.returncode:
        raise RuntimeError(f"dmesg snapshot failed: {label}")


def worker():
    require_approval()
    # torch is imported only in the explicitly requested, timeout-bounded worker.
    import torch

    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0,1"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1"
    assert torch.cuda.device_count() == 2, "expose GPUs0 and1 only"
    devices = []
    for index in (0, 1):
        props = torch.cuda.get_device_properties(index)
        assert props.gcnArchName.startswith("gfx1250")
        devices.append(
            dict(
                index=index,
                name=props.name,
                isa=props.gcnArchName,
                cus=props.multi_processor_count,
            )
        )
    emit(
        "devices",
        devices=devices,
        torch_version=torch.__version__,
        hip_version=torch.version.hip,
    )
    emit(
        "peer_capability",
        gpu0_to_gpu1=bool(torch.cuda.can_device_access_peer(0, 1)),
        gpu1_to_gpu0=bool(torch.cuda.can_device_access_peer(1, 0)),
    )
    generator = torch.Generator(device="cpu").manual_seed(20261117)
    expected = torch.randint(
        0, 256, (BYTES,), generator=generator, dtype=torch.uint8, device="cpu"
    )
    source = expected.to("cuda:0")
    torch.cuda.synchronize(0)
    emit(
        "source_ready",
        bytes=source.numel() * source.element_size(),
        contiguous=source.is_contiguous(),
    )
    start = time.monotonic()
    peer = source.to("cuda:1")
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    forward_seconds = time.monotonic() - start
    forward_cpu = peer.cpu()
    if not torch.equal(expected, forward_cpu):
        first = int(torch.nonzero(expected != forward_cpu)[0])
        emit(
            "mismatch",
            direction="0_to_1",
            index=first,
            expected=int(expected[first]),
            got=int(forward_cpu[first]),
        )
        raise AssertionError("P.1 GPU0->GPU1 data mismatch")
    emit(
        "forward_pass",
        bytes=BYTES,
        seconds=forward_seconds,
        cpu_sha256=hashlib.sha256(forward_cpu.numpy().tobytes()).hexdigest(),
    )
    start = time.monotonic()
    returned = peer.to("cuda:0")
    torch.cuda.synchronize(1)
    torch.cuda.synchronize(0)
    backward_seconds = time.monotonic() - start
    returned_cpu = returned.cpu()
    if not torch.equal(expected, returned_cpu):
        first = int(torch.nonzero(expected != returned_cpu)[0])
        emit(
            "mismatch",
            direction="1_to_0",
            index=first,
            expected=int(expected[first]),
            got=int(returned_cpu[first]),
        )
        raise AssertionError("P.1 GPU1->GPU0 data mismatch")
    emit(
        "roundtrip_pass",
        bytes=BYTES,
        seconds=backward_seconds,
        cpu_sha256=hashlib.sha256(returned_cpu.numpy().tobytes()).hexdigest(),
        compared_bytes=2 * BYTES,
    )


def driver():
    require_approval()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = LOG_ROOT / f"peer-P1-{stamp}"
    log_path = Path(str(prefix) + ".log")
    summary_path = Path(str(prefix) + ".json")
    print(f"P.1 log: {log_path}", flush=True)
    # Leave up to30seconds of the outer120second deadline for post-run telemetry.
    command = [
        "docker",
        "exec",
        "-e",
        "HIP_VISIBLE_DEVICES=0,1",
        "-e",
        "CUDA_VISIBLE_DEVICES=0,1",
        CONTAINER,
        "timeout",
        "-s",
        "KILL",
        "90",
        "python3",
        str(SCRIPT),
        "--worker",
    ]
    summary = dict(
        step="P.1",
        command=command,
        script_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        log_path=str(log_path),
        summary_path=str(summary_path),
        status="started",
    )
    with log_path.open("x", buffering=1) as log:
        saved_stdout = sys.stdout
        sys.stdout = log
        result = None
        start = time.monotonic()
        try:
            emit("start", **summary)
            dmesg_snapshot("before", Path(str(prefix) + "-dmesg-before.json"))
            log.flush()
            os.fsync(log.fileno())
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=100
            )
            if result.returncode:
                raise RuntimeError(
                    f"worker exited {result.returncode}; stop peer ladder"
                )
            summary["status"] = "done"
        except Exception as exc:
            summary.update(status="failed", error=repr(exc))
        finally:
            try:
                dmesg_snapshot("after", Path(str(prefix) + "-dmesg-after.json"))
            except Exception as exc:
                summary.update(status="failed", telemetry_error=repr(exc))
            summary.update(
                elapsed_seconds=time.monotonic() - start,
                worker_returncode=None if result is None else result.returncode,
            )
            emit("result", **summary)
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            sys.stdout = saved_stdout
    print(json.dumps(summary), flush=True)
    if summary["status"] != "done":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--run",
        action="store_true",
        help="host driver; requires exact-script Claude approval",
    )
    action.add_argument(
        "--worker",
        action="store_true",
        help="internal GPU worker, launched only by the approved driver",
    )
    args = parser.parse_args()
    if args.worker:
        worker()
    elif args.run:
        driver()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
