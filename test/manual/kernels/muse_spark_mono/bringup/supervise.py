"""Print one proposed run by default. --execute remains approval gated."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from guard import COORD, HERE, REPO, STEPS, manifest, require


def snapshot(out, suffix):
    ok = True
    for name, command in (
        ("dmesg", ["dmesg", "--ctime"]),
        ("amd-smi", ["amd-smi", "metric", "--gpu", "all", "--json"]),
    ):
        with (out / f"{name}-{suffix}.log").open("w") as log:
            try:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=5,
                    check=False,
                )
                if result.returncode:
                    raise RuntimeError(f"{name} snapshot failed: {result.returncode}")
            except Exception as error:
                ok = False
                log.write(str(error) + "\n")
                if suffix == "before":
                    raise
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("step", choices=STEPS)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    digest = manifest()
    world = STEPS[a.step]
    name = f"muse-bringup-{a.step}-{int(time.time())}"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + a.step
    out = COORD / "logs" / ("bringup-" + run_id)
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network=host",
        "--ipc=host",
        "--device=/dev/kfd",
        "--device=/dev/dri",
        "--group-add=video",
        "--security-opt=seccomp=unconfined",
        "-v",
        "/home/bangtliu:/home/bangtliu",
        "-w",
        str(REPO),
        "-e",
        "HIP_VISIBLE_DEVICES=" + ",".join(map(str, range(world))),
        "-e",
        "CUDA_VISIBLE_DEVICES=" + ",".join(map(str, range(world))),
        "-e",
        "PYTHONPATH=python:" + str(HERE),
        "-e",
        "MUSE_BRINGUP_APPROVAL_EXPIRES_UTC",
        "-e",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING=1",
        "-e",
        "NCCL_DEBUG=INFO",
        "-e",
        "SGLANG_HIP_VMM_HANDLE_TYPE=posix_fd",
        json.loads((COORD / "bringup" / "manifest.json").read_text())["image_id"],
        "timeout",
        "-s",
        "KILL",
        "60",
        "/opt/venv/bin/python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world}",
        "--max-restarts=0",
        "--log-dir",
        str(out / "ranks"),
        "--redirects=3",
        str(HERE / "worker.py"),
        a.step,
        "--output",
        str(out),
    ]
    proposal = dict(
        event="PROPOSED RUN",
        step=a.step,
        bundle_sha256=digest,
        world=world,
        command=command,
        log_directory=str(out),
        inner_timeout_seconds=60,
        watchdog_seconds=65,
        executed=False,
    )
    print(json.dumps(proposal, indent=2), flush=True)
    if not a.execute:
        return
    require(a.step)  # precedes ALL snapshots and docker/GPU interaction
    out.mkdir(parents=True, exist_ok=False)
    (out / "proposal.json").write_text(json.dumps(proposal, indent=2) + "\n")
    snapshot(out, "before")
    stamps = [(COORD / n).stat().st_mtime_ns for n in ("plan.md", "review.md")]
    timed_out = changed = False
    with (out / "launcher.log").open("w") as log:
        child = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        deadline = time.monotonic() + 65
        try:
            while child.poll() is None:
                changed = stamps != [
                    (COORD / n).stat().st_mtime_ns for n in ("plan.md", "review.md")
                ]
                timed_out = time.monotonic() >= deadline
                if changed or timed_out:
                    break
                time.sleep(0.25)
        finally:
            if child.poll() is None:
                # Kill only this uniquely named disposable container and launcher PG.
                # Killing docker exec alone would leave the GPU workers alive.
                try:
                    subprocess.run(
                        ["docker", "kill", "--signal=KILL", name],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=5,
                    )
                except Exception as error:
                    log.write(f"watchdog container kill failed: {error}\n")
                finally:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            child.wait(timeout=5)
    health_ok = snapshot(out, "after")
    before = (out / "dmesg-before.log").read_text()
    after = (out / "dmesg-after.log").read_text()
    if before and before.splitlines()[-1] in after.splitlines():
        new = "\n".join(
            after.splitlines()[after.splitlines().index(before.splitlines()[-1]) + 1 :]
        )
    else:
        new = after  # ring loss / inaccessible dmesg is not a clean health result
    (out / "dmesg-new.log").write_text(new + "\n")
    bad = any(
        token in new.lower()
        for token in (
            "amdgpu",
            "xgmi",
            "gpu reset",
            "page fault",
            "ring timeout",
            "ras",
            "a" + "er:",
        )
    )
    results = [out / f"rank-{r}-PASS.json" for r in range(world)]
    ok = (
        child.returncode == 0
        and not (timed_out or changed or bad)
        and health_ok
        and all(x.is_file() for x in results)
    )
    result = dict(
        step=a.step,
        done=ok,
        returncode=child.returncode,
        watchdog=timed_out,
        protocol_changed=changed,
        health_snapshots_ok=health_ok,
        new_driver_messages=bad,
        logs=str(out),
        next_step_authorized=False,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    with (COORD / "status.md").open("a") as log:
        log.write(
            f"{datetime.now(timezone.utc).isoformat()} bringup.{a.step} "
            f"{'done' if ok else 'failed'}: {json.dumps(result)}; STOP for review.\n"
        )
    print(json.dumps(result), flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
