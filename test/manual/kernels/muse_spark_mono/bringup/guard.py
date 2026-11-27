"""Fail-closed, CPU-only authorization for the supervised peer ladder."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

COORD = Path("/home/bangtliu/coord")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
STEPS = {"rccl2": 2, "rccl4": 4, "ping2": 2, "oneshot2": 2, "oneshot4": 4, "layer4": 4}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def manifest():
    path = COORD / "bringup" / "manifest.json"
    data = json.loads(path.read_text())
    for name, expected in data["files"].items():
        if sha(name) != expected:
            raise RuntimeError(
                f"UNAPPROVED SOURCE CHANGE: {name}; regenerate/review bundle"
            )
    return sha(path)


def require(step):
    """Only Claude writes approvals; this function never creates one."""
    digest = manifest()
    paths = [COORD / "bringup" / "manifest.json", *HERE.glob("*.py")]
    for path in paths:
        approval = COORD / "approvals" / (sha(path) + ".ok")
        if not approval.is_file():
            raise RuntimeError(f"MULTI-GPU HALTED: missing Claude approval {approval}")
    # The review entry is deliberately step-specific and bundle-specific.
    approval = f"APPROVED RUN: T12.{step} {digest}"
    if approval not in (COORD / "review.md").read_text().splitlines():
        raise RuntimeError(
            f"MULTI-GPU HALTED: review.md must contain exact line: {approval}"
        )
    expiry = os.environ.get("MUSE_BRINGUP_APPROVAL_EXPIRES_UTC", "")
    if not expiry or datetime.now(timezone.utc) >= datetime.fromisoformat(expiry):
        raise RuntimeError("missing/expired MUSE_BRINGUP_APPROVAL_EXPIRES_UTC")
    return digest
