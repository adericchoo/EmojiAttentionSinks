"""C2 command line: audit -> probes -> run -> figures.

    python scripts/c2.py audit
    python scripts/c2.py controls
    python scripts/c2.py probes   --model pythia
    python scripts/c2.py anatomy  --model pythia
    python scripts/c2.py handcheck
    python scripts/c2.py run      --model pythia --split dev
    python scripts/c2.py figures  --model pythia

Outputs go to outputs/c2/. See src/c2/ for the implementation.
"""
import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit", help="tokenisation audit for both models")
    a = ap.parse_args()

    if a.cmd == "audit":
        from c2 import audit_report
        audit_report.main()


if __name__ == "__main__":
    main()
