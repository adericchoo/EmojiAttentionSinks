"""Meaning-matched control: does an emoji draw more attention than its meaning?

    python scripts/04_meaning_matched_control.py              # primary model
    python scripts/04_meaning_matched_control.py replication  # Qwen

Each emoji and its spelled-out meaning are placed in the SAME structured probe
slot (identical carrier, prefix and a FIXED 64-token far query block, so the
query count |Q| is the same across conditions and across spans of different
length). Attention is scored PER SPAN TOKEN, so a fragmented multi-token emoji
and a short spelled-out word are compared on an equal, per-token footing.

This is additive. It reuses, without modifying:
  - the structured probe builder with the fixed far block   (c2.probes)
  - attention capture and span location/validation          (eas)
  - the reference received-attention metric                 (eas.metric_reference)

The structured probe with the fixed far block is used deliberately: a simple
carrier whose query set grows with span length is not comparable across the
different-length spans an emoji and its spelling produce.
"""
import sys

from _bootstrap import load_config, select_model

from c2.probes import BUFFER, QLEN, Rejected, build_contexts, make_probe
from eas import capture_attention, load_model
from eas.metric_reference import received_attention
from eas.span import SpanNotFound, locate_span_by_offsets, validate_span
from eas.types import SpanLoc

# Make emoji printable on consoles whose default code page is not UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# Emoji paired with its spelled-out meaning. The spelled-out form is deliberately
# NOT token-count matched to the emoji; per-token scoring is what makes the two
# comparable. Written as escapes so the file is byte-stable regardless of editor.
MEANING_MATCHED_PAIRS = [
    ("\U0001F44D", "thumbs up"),      # thumbs-up sign
    ("\U0001F680", "rocket"),         # rocket
    ("\U0001F525", "fire"),           # fire
    ("❤️", "heart"),        # red heart
    ("\U0001F600", "smiling face"),   # grinning face
    ("⭐", "star"),               # star
    ("\U0001F319", "moon"),           # crescent moon
]


def load_pairs():
    """The (emoji, spelled_out) pairs to measure."""
    pairs = []
    for emoji, spelled in MEANING_MATCHED_PAIRS:
        pairs.append((emoji, spelled))
    return pairs


def per_token_metric(received, n_span_tokens):
    """Per-span-token attention: summed received attention M divided by |S|.

    `received` is the [L, H] tensor from received_attention (summed over the
    query block and the span). Dividing by the span-token count puts emoji of
    several tokens and a single word on the same per-token scale.
    """
    if n_span_tokens < 1:
        raise ValueError("span must have at least one token")
    return received / n_span_tokens


def capture_and_locate(model, tok, ctx, filler):
    """Build the structured slot, capture attention, and locate + validate the
    span. Returns (cap, span, queries) where queries is the FIXED 64-token far
    block, disjoint from the span.

    Raises Rejected (the slot could not be built cleanly) or SpanNotFound (the
    span could not be located and validated, or the far block runs off the end).
    """
    probe = make_probe(tok, ctx, filler)
    text = probe["text"]
    char_start, char_end = probe["char_span"]

    cap = capture_attention(model, tok, text)
    idx = locate_span_by_offsets(cap.offsets, char_start, char_end)
    validate_span(tok, cap.input_ids, idx, filler)

    span = SpanLoc(idx[0], idx[-1] + 1, idx)
    span_end = idx[-1] + 1
    q_start = span_end + BUFFER
    q_end = q_start + QLEN
    if q_end > cap.n:
        raise SpanNotFound(f"far query block runs past the sequence ({q_end} > {cap.n})")

    queries = []
    for q in range(q_start, q_end):
        queries.append(q)

    return cap, span, queries


def span_mean_metric(cap, span, queries):
    """Per-token attention averaged over the span's tokens: M / |S|, [L, H]."""
    received = received_attention(cap, span, queries)
    m_pt = per_token_metric(received, len(span.token_indices))
    result = {
        "n_span": len(span.token_indices),
        "n_q": len(queries),
        "m_pt_mean": float(m_pt.mean()),
        "m_pt_max": float(m_pt.max()),
    }
    return result


def measure_filler(model, tok, ctx, filler):
    """Place one filler in the structured slot and return its per-token metric."""
    cap, span, queries = capture_and_locate(model, tok, ctx, filler)
    return span_mean_metric(cap, span, queries)


def locate_pair(model, tok, contexts, emoji, spelled):
    """Locate both conditions in the SAME carrier. Try carriers in order until one
    lets both the emoji and its spelling locate cleanly. Returns a pair of
    (cap, span, queries) tuples, one for the emoji and one for the spelled form."""
    last_error = None
    for ctx in contexts:
        try:
            emoji_located = capture_and_locate(model, tok, ctx, emoji)
            spelled_located = capture_and_locate(model, tok, ctx, spelled)
        except (SpanNotFound, Rejected) as error:
            last_error = error
            continue
        return emoji_located, spelled_located
    if last_error is None:
        raise SpanNotFound("no carrier available")
    raise SpanNotFound(str(last_error))


def measure_pair(model, tok, contexts, emoji, spelled):
    """Span-mean metric for both conditions, located in the same carrier."""
    emoji_located, spelled_located = locate_pair(model, tok, contexts, emoji, spelled)
    emoji_result = span_mean_metric(emoji_located[0], emoji_located[1], emoji_located[2])
    spelled_result = span_mean_metric(spelled_located[0], spelled_located[1], spelled_located[2])
    return emoji_result, spelled_result


def _mean(values):
    if len(values) == 0:
        return float("nan")
    total = 0.0
    for v in values:
        total += v
    return total / len(values)


def run_for_model(name, revision):
    model, tok = load_model(name, revision=revision)
    contexts = build_contexts(tok)
    pairs = load_pairs()

    print(f"\nMEANING-MATCHED CONTROL  —  {name}")
    print("structured probe, fixed far query block; attention scored per span token (M / |S|)\n")
    header = (f"  {'pair':14} {'emoji':6} {'|S_e|':>5} {'|S_s|':>5} {'|Q|':>4} "
              f"{'Mpt_emoji':>11} {'Mpt_spelled':>12} {'delta':>11} "
              f"{'max_emoji':>11} {'max_spelled':>12}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    emoji_means = []
    spelled_means = []
    deltas = []
    for emoji, spelled in pairs:
        try:
            emoji_result, spelled_result = measure_pair(model, tok, contexts, emoji, spelled)
        except (SpanNotFound, Rejected) as error:
            print(f"  [skip] {spelled!r}: {error}")
            continue

        delta_mean = emoji_result["m_pt_mean"] - spelled_result["m_pt_mean"]
        emoji_means.append(emoji_result["m_pt_mean"])
        spelled_means.append(spelled_result["m_pt_mean"])
        deltas.append(delta_mean)

        print(f"  {spelled:14} {emoji:6} "
              f"{emoji_result['n_span']:>5} {spelled_result['n_span']:>5} "
              f"{emoji_result['n_q']:>4} "
              f"{emoji_result['m_pt_mean']:>11.6f} {spelled_result['m_pt_mean']:>12.6f} "
              f"{delta_mean:>11.6f} "
              f"{emoji_result['m_pt_max']:>11.6f} {spelled_result['m_pt_max']:>12.6f}")

    print()
    if len(deltas) == 0:
        print("  no pairs could be measured (all skipped).")
        return

    mean_emoji = _mean(emoji_means)
    mean_spelled = _mean(spelled_means)
    mean_delta = _mean(deltas)
    print(f"  aggregate ({len(deltas)} pairs): "
          f"mean Mpt(emoji) = {mean_emoji:.6f}, "
          f"mean Mpt(spelled) = {mean_spelled:.6f}, "
          f"mean per-pair delta = {mean_delta:.6f}")

    print("\n  interpretation key (per-token attention):")
    print("    Mpt(emoji) ~= Mpt(spelled): attention matches meaning (no sink)")
    print("    Mpt(emoji) >> Mpt(spelled): emoji draws attention beyond its meaning (sink signature)")
    print("    Mpt(emoji) << Mpt(spelled): fragmented emoji is ignored relative to its words")
    print()


def main():
    cfg = load_config()
    name, revision = select_model(cfg)
    run_for_model(name, revision)


if __name__ == "__main__":
    main()
