"""Step 2 (THE GATE): a hand-verifiable micro example.

Prints one head's attention matrix with token labels so you can confirm by eye
that rows are queries, columns are keys, and the causal structure is right.

    python scripts/01_hand_check.py
"""
from _bootstrap import load_config, select_model

from eas import load_model, capture_attention, build_stimulus
from eas.capture import locate
from eas.metric_reference import received_attention


def print_head(cap, layer=0, head=0):
    att = cap.attentions[layer, head]  # [n, n]
    toks = cap.tokens
    lengths = []
    for t in toks:
        lengths.append(len(t))
    w = max(lengths) + 1
    print(f"\nLayer {layer}, Head {head}  (row = query q, col = key k):")
    header = ""
    for t in toks:
        header += f"{t[:6]:>7}"
    print(" " * (w + 2) + header)
    for q, row in enumerate(att.tolist()):
        cells = ""
        for v in row:
            cells += f"{v:7.2f}"
        print(f"{toks[q]:>{w}}  {cells}")


def main():
    cfg = load_config()
    name, rev = select_model(cfg)
    model, tok = load_model(name, revision=rev)

    # tiny, fully inspectable input; span is the word "cat"
    stim = build_stimulus(prefix="A ", span_text="cat", suffix=" sat.")
    cap = capture_attention(model, tok, stim.text)
    span = locate(tok, cap, stim)
    print(f"text={stim.text!r}")
    span_token_strs = []
    for i in span.token_indices:
        span_token_strs.append(cap.tokens[i])
    print(f"span tokens {span.token_indices} -> {span_token_strs}")

    print_head(cap, layer=0, head=0)

    q = cap.later_queries(span)  # positions after the span
    M = received_attention(cap, span, q)  # [L, H]
    print(f"\nReference received attention M (queries {q} -> span {span.token_indices}):")
    print(f"  layer 0, head 0 : {M[0, 0].item():.4f}")
    argmax_pos = (M == M.max()).nonzero()[0]
    coords = []
    for x in argmax_pos:
        coords.append(x.item())
    print(f"  max over all (l,h): {M.max().item():.4f} at {tuple(coords)}")
    print("\nHand-check: verify the printed matrix by eye")


if __name__ == "__main__":
    main()
