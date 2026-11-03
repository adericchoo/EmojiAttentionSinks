"""Step 1: prove attentions come out, and measure emoji fragmentation.

    python scripts/00_feasibility.py
"""
from _bootstrap import load_config, select_model

from eas import load_model, capture_attention

EMOJI_SAMPLES = ["\U0001F44D", "\U0001F44D\U0001F3FD", "\U0001F468‍\U0001F469‍\U0001F467",
                 "\U0001F600", "\U0001F680", "❤️"]  # 👍 👍🏽 👨‍👩‍👧 😀 🚀 ❤️


def main():
    cfg = load_config()
    name, rev = select_model(cfg)
    print(f"Loading {name} (revision={rev}) ...")
    model, tok = load_model(name, revision=rev)

    cap = capture_attention(model, tok, "The students opened their books today.")
    L, H, n, _ = cap.shape
    print(f"OK: attentions came out. shape = [L={L}, H={H}, n={n}, n={n}]")
    print(f"tokens: {cap.tokens}")

    print("\nEmoji fragmentation (tokens per emoji) under this tokenizer:")
    for e in EMOJI_SAMPLES:
        ids = tok(e, add_special_tokens=False)["input_ids"]
        pieces = tok.convert_ids_to_tokens(ids)
        print(f"  {e!r:12} -> {len(ids):2d} tokens  {pieces}")

    print("\nFeasibility OK. Next: scripts/01_hand_check.py")


if __name__ == "__main__":
    main()
