"""Step 3: the extraction-correctness suite.

    python scripts/02_run_validation.py
"""
from _bootstrap import load_config, select_model

from eas import load_model, capture_attention
from eas.validate import rows_sum_to_one, causal_mask_holds, deterministic


def check(name, passed, detail):
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return passed


def main():
    cfg = load_config()
    name, rev = select_model(cfg)
    model, tok = load_model(name, revision=rev)

    text = "I really loved the movie we watched last night."
    cap = capture_attention(model, tok, text)
    cap2 = capture_attention(model, tok, text)

    print("Validation suite:")
    ok = True
    ok &= check("rows sum to 1", *rows_sum_to_one(cap.attentions))
    ok &= check("causal mask holds", *causal_mask_holds(cap.attentions))
    ok &= check("deterministic", *deterministic(cap, cap2))

    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED — do not score until green")


if __name__ == "__main__":
    main()
