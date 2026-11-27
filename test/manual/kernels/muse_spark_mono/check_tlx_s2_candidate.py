"""Run the complete S2 Gluon/native/FP64 package on an isolated TLX candidate."""

import argparse
import importlib.util
import sys

import check_tlx_s2 as check


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--candidate", required=True)
    args, remaining = parser.parse_known_args()
    spec = importlib.util.spec_from_file_location(
        "s2_candidate_checked", args.candidate
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    check.new = module
    sys.argv = [sys.argv[0]] + remaining
    check.main()


if __name__ == "__main__":
    main()
