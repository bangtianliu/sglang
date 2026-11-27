"""Run the unchanged T0 proofs with isolated T2 artifacts in the patched image."""

import argparse
from pathlib import Path

import tlx_ir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("proof", choices=("scoped", "projection"))
    a = p.parse_args()
    if a.proof == "scoped":
        import proof_tlx_scoped as proof
    else:
        import proof_tlx_projection as proof
    proof.LOG = Path("/home/bangtliu/coord/logs/phaseT2-T0-proofs")
    proof.LOG.mkdir(exist_ok=True)
    tlx_ir.ROOT = Path("/home/bangtliu/coord/logs/phaseT-ir/T2-T0-proofs")
    proof.main()


if __name__ == "__main__":
    main()
