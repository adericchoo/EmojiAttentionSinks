"""Additional metric-validation checks (adds to the pipeline; changes nothing).

    python scripts/validate_metric.py first-token --model pythia
    python scripts/validate_metric.py near-block  --model pythia
    python scripts/validate_metric.py all                         # both models, both checks

Preconditions (same as the main runner): the probe files must already exist, e.g.
    python scripts/c2.py probes --model pythia
    python scripts/c2.py probes --model qwen

Outputs go to outputs/metric_validation/. Each run writes a per-head CSV, an npz,
a summary.txt and a manifest.json with pinned revisions, versions and hashes.
"""
import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # a git subprocess runs for the manifest


def _print_final_summary(first_results, near_results):
    print()
    print("=" * 72)
    print("METRIC VALIDATION SUMMARY")
    print("=" * 72)
    print("Check 1 — first-token (position 0) sink detection")
    print("  (share of heads where attention to the sink exceeds an ordinary word)")
    if not first_results:
        print("  (not run)")
    for res in first_results:
        flag = "PASS" if res["pass"] else "FAIL"
        print(f"  {res['tag']:14} share {res['share_heads_first_exceeds_ref']:.3f}  "
              f"{flag}   detects the first-token sink: {'yes' if res['pass'] else 'no'}")
    print()
    print("Check 2 — Delta(p,r) positive control, near buffer vs far block")
    print("  (share of heads where punctuation attracts more than an ordinary word)")
    if not near_results:
        print("  (not run)")
    for res in near_results:
        pr = res.get("contrasts", {}).get("d_pr")
        if pr is None:
            print(f"  {res['tag']:14} (no p/r sets)")
            continue
        far_flag = "PASS" if pr["pass_far"] else "FAIL"
        near_flag = "PASS" if pr["pass_near"] else "FAIL"
        print(f"  {res['tag']:14} far  {pr['share_far']:.3f} {far_flag}   "
              f"near {pr['share_near']:.3f} {near_flag}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("first-token", help="check 1: can the metric detect the position-0 sink?")
    p1.add_argument("--model", default="pythia", help="pythia | qwen")
    p1.add_argument("--split", default="dev", choices=["dev", "confirm", "all"])
    p1.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    p1.add_argument("--limit", type=int, default=None, help="only the first N reference probes")

    p2 = sub.add_parser("near-block", help="check 2: key contrasts on the near buffer vs far block")
    p2.add_argument("--model", default="pythia", help="pythia | qwen")
    p2.add_argument("--split", default="dev", choices=["dev", "confirm", "all"])
    p2.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    p2.add_argument("--limit", type=int, default=None, help="only the first N matched sets")

    p3 = sub.add_parser("all", help="both checks, both models; prints the combined summary")
    p3.add_argument("--split", default="dev", choices=["dev", "confirm", "all"])
    p3.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    p3.add_argument("--models", default="pythia,qwen", help="comma-separated: pythia,qwen")

    args = parser.parse_args()

    from metric_validation import first_token, near_block

    if args.cmd == "first-token":
        res = first_token.main(args.model, args.split, args.device, args.limit)
        _print_final_summary([res], [])
    elif args.cmd == "near-block":
        res = near_block.main(args.model, args.split, args.device, args.limit)
        _print_final_summary([], [res])
    elif args.cmd == "all":
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        first_results = []
        near_results = []
        for model in models:
            try:
                first_results.append(first_token.main(model, args.split, args.device))
            except Exception as exc:
                print(f"[first_token] {model}: skipped ({type(exc).__name__}: {exc})")
            try:
                near_results.append(near_block.main(model, args.split, args.device))
            except Exception as exc:
                print(f"[near_block] {model}: skipped ({type(exc).__name__}: {exc})")
        _print_final_summary(first_results, near_results)


if __name__ == "__main__":
    main()
