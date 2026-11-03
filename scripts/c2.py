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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # we fork git for the manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit", help="tokenisation audit for both models")
    sub.add_parser("controls", help="screen control fillers (w, s, x, p, r) for both models")
    sp = sub.add_parser("probes", help="generate matched probes + exclusion log for one model")
    sp.add_argument("--model", default="pythia", help="pythia | qwen")
    sp = sub.add_parser("anatomy", help="print one matched set with S / buffer / Q marked")
    sp.add_argument("--model", default="pythia")
    sp.add_argument("--set", default=None, help="matched_set_id (default: a dev set)")
    sub.add_parser("handcheck", help="print the synthetic + real hand-checked metric examples")
    sp = sub.add_parser("run", help="run probes through the model and write per-head metrics")
    sp.add_argument("--model", default="pythia")
    sp.add_argument("--split", default="dev", choices=["dev", "confirm", "all"])
    sp.add_argument("--probes", default=None, help="probe JSONL (default: outputs/c2/probes/probes_<model>.jsonl)")
    sp.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    sp.add_argument("--limit", type=int, default=None, help="only the first N matched sets (smoke test)")
    sp = sub.add_parser("figures", help="DEVELOPMENT PILOT figures from a dev run")
    sp.add_argument("--model", default="pythia")
    sp.add_argument("--run", default=None, help="run directory (default: latest dev run)")
    a = ap.parse_args()

    if a.cmd == "audit":
        from c2 import audit_report
        audit_report.main()
    elif a.cmd == "controls":
        from c2 import controls_report
        controls_report.main()
    elif a.cmd == "probes":
        from c2 import probes_report
        probes_report.main(a.model)
    elif a.cmd == "anatomy":
        from c2 import anatomy
        anatomy.main(a.model, a.set)
    elif a.cmd == "handcheck":
        from c2 import handcheck
        handcheck.main()
    elif a.cmd == "run":
        from c2 import runner
        runner.main(a.model, a.split, a.probes, a.device, a.limit)
    elif a.cmd == "figures":
        from c2 import figures
        figures.main(a.model, a.run)


if __name__ == "__main__":
    main()
