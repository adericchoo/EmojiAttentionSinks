"""Per-fragment breakdown of the meaning-matched control.

    python scripts/05_fragment_breakdown.py              # primary model
    python scripts/05_fragment_breakdown.py replication  # Qwen

The span-mean score (M / |S|) averages an emoji's attention over all its
fragments, which can hide a sink that lives in a SINGLE fragment. This re-scores
the SAME captured data at the fragment level: for each span token it reports the
attention that single token receives from the FIXED far query block, and surfaces
the hottest fragment.

No metric and no protected file is changed. The pair list and the probe/capture
helpers are imported from the span-mean script; only the per-fragment reduction is
new here. The per-fragment totals are literally received_attention's sum left
un-summed over the span tokens, which is checked once at runtime.

For a whole-token emoji (|S| = 1, as on Qwen) the per-fragment view degenerates to
the single token, and everything below runs unchanged.
"""
import importlib.util
import os
import sys

from _bootstrap import load_config, select_model

import torch
from c2.probes import Rejected
from eas.metric_reference import received_attention
from eas.span import SpanNotFound

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _load_span_mean_module():
    """Import the span-mean script (its name begins with a digit, so load by path)."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "04_meaning_matched_control.py")
    spec = importlib.util.spec_from_file_location("meaning_matched_control", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mmc = _load_span_mean_module()


def fragment_received(attentions, token_index, queries):
    """Attention a SINGLE key (token_index) receives from the query block.

    Mirrors received_attention's convention (sum over q in Q) but for one key:
    returns [L, H] where entry (l, h) = sum over q in Q of attentions[l, h, q, k].
    """
    n_layers = attentions.shape[0]
    n_heads = attentions.shape[1]
    frag = torch.zeros(n_layers, n_heads)
    for q in queries:
        frag = frag + attentions[:, :, q, token_index]
    return frag


def per_fragment_vector(cap, tok, span, queries):
    """One entry per span token: position, raw token, decoded token, mean and max."""
    frags = []
    position = 0
    for token_index in span.token_indices:
        frag = fragment_received(cap.attentions, token_index, queries)
        decoded = tok.decode([cap.input_ids[token_index]])
        entry = {
            "position": position,
            "token_index": token_index,
            "token": cap.tokens[token_index],
            "decoded": decoded,
            "frag_mean": float(frag.mean()),
            "frag_max": float(frag.max()),
        }
        frags.append(entry)
        position = position + 1
    return frags


def consistency_diff(cap, span, queries):
    """max | sum_k fragment_received(k)  -  received_attention(cap, span, Q) |."""
    n_layers = cap.attentions.shape[0]
    n_heads = cap.attentions.shape[1]
    total = torch.zeros(n_layers, n_heads)
    for token_index in span.token_indices:
        total = total + fragment_received(cap.attentions, token_index, queries)
    reference = received_attention(cap, span, queries)
    difference = (total - reference).abs().max()
    return float(difference)


def argmax_by(frags, field):
    best_index = 0
    best_value = frags[0][field]
    for i in range(1, len(frags)):
        if frags[i][field] > best_value:
            best_value = frags[i][field]
            best_index = i
    return best_index


def classify_position(position, n_fragments):
    if position == 0:
        return "first"
    if position == n_fragments - 1:
        return "last"
    return "middle"


def _mean(values):
    if len(values) == 0:
        return float("nan")
    total = 0.0
    for v in values:
        total += v
    return total / len(values)


def _print_fragment_table(label, glyph, frags):
    print(f"  {label} {glyph}  (|S| = {len(frags)}):")
    print(f"    {'pos':>3}  {'raw token':<16} {'decoded':<10} {'frag_mean':>10} {'frag_max':>10}")
    for frag in frags:
        raw = repr(frag["token"])
        if len(raw) > 16:
            raw = raw[:15] + "…"
        decoded = repr(frag["decoded"])
        if len(decoded) > 10:
            decoded = decoded[:9] + "…"
        print(f"    {frag['position']:>3}  {raw:<16} {decoded:<10} "
              f"{frag['frag_mean']:>10.6f} {frag['frag_max']:>10.6f}")


def run_for_model(name, revision):
    model, tok = mmc.load_model(name, revision=revision)
    contexts = mmc.build_contexts(tok)
    pairs = mmc.load_pairs()

    print(f"\nPER-FRAGMENT BREAKDOWN  —  {name}")
    print("same structured probe and fixed 64-token far block; attention per single span token\n")

    consistency_printed = False
    emoji_maxfrag_means = []
    emoji_spanmean_mpt = []
    delta_maxfrag_list = []
    winner_positions = {"first": 0, "middle": 0, "last": 0}
    winner_identities = {}

    for emoji, spelled in pairs:
        try:
            emoji_located, spelled_located = mmc.locate_pair(model, tok, contexts, emoji, spelled)
        except (SpanNotFound, Rejected) as error:
            print(f"  [skip] {spelled!r}: {error}")
            continue

        emoji_cap, emoji_span, emoji_queries = emoji_located
        spelled_cap, spelled_span, spelled_queries = spelled_located

        if not consistency_printed:
            difference = consistency_diff(emoji_cap, emoji_span, emoji_queries)
            status = "OK" if difference <= 1e-4 else "WARN — fragments do not re-sum to received_attention"
            print(f"  consistency check: max | sum_k frag - received | = {difference:.2e}   [{status}]\n")
            consistency_printed = True

        emoji_frags = per_fragment_vector(emoji_cap, tok, emoji_span, emoji_queries)
        spelled_frags = per_fragment_vector(spelled_cap, tok, spelled_span, spelled_queries)

        _print_fragment_table("emoji", emoji, emoji_frags)
        _print_fragment_table("spelled", f"'{spelled}'", spelled_frags)

        emoji_hot_mean_i = argmax_by(emoji_frags, "frag_mean")
        emoji_hot_max_i = argmax_by(emoji_frags, "frag_max")
        spelled_hot_mean_i = argmax_by(spelled_frags, "frag_mean")

        emoji_hot_mean = emoji_frags[emoji_hot_mean_i]
        emoji_hot_max = emoji_frags[emoji_hot_max_i]
        spelled_hot_mean = spelled_frags[spelled_hot_mean_i]

        delta_maxfrag = emoji_hot_mean["frag_mean"] - spelled_hot_mean["frag_mean"]
        span_mean_emoji = mmc.span_mean_metric(emoji_cap, emoji_span, emoji_queries)["m_pt_mean"]

        emoji_maxfrag_means.append(emoji_hot_mean["frag_mean"])
        emoji_spanmean_mpt.append(span_mean_emoji)
        delta_maxfrag_list.append(delta_maxfrag)

        position_class = classify_position(emoji_hot_mean["position"], len(emoji_frags))
        winner_positions[position_class] = winner_positions[position_class] + 1
        identity = emoji_hot_mean["decoded"]
        if identity in winner_identities:
            winner_identities[identity] = winner_identities[identity] + 1
        else:
            winner_identities[identity] = 1

        print(f"  => {spelled:12}  hottest emoji frag by mean: pos {emoji_hot_mean['position']} "
              f"{repr(emoji_hot_mean['decoded'])} mean={emoji_hot_mean['frag_mean']:.6f}"
              f" | by max: pos {emoji_hot_max['position']} max={emoji_hot_max['frag_max']:.6f}"
              f" | hottest spelled token: pos {spelled_hot_mean['position']} "
              f"mean={spelled_hot_mean['frag_mean']:.6f}"
              f" | Δ_maxfrag={delta_maxfrag:+.6f} | span-mean Mpt(emoji)={span_mean_emoji:.6f}\n")

    if len(delta_maxfrag_list) == 0:
        print("  no pairs could be measured (all skipped).")
        return

    mean_maxfrag = _mean(emoji_maxfrag_means)
    mean_spanmean = _mean(emoji_spanmean_mpt)
    mean_delta = _mean(delta_maxfrag_list)

    print("  AGGREGATE (per model)")
    print(f"    mean hottest-fragment frag_mean (emoji) = {mean_maxfrag:.6f}")
    print(f"    mean span-mean Mpt            (emoji) = {mean_spanmean:.6f}")
    if mean_spanmean != 0.0:
        print(f"    dilution factor (hottest / span-mean) = {mean_maxfrag / mean_spanmean:.2f}x")
    print(f"    mean Δ_maxfrag across pairs            = {mean_delta:+.6f}")

    print("    winning fragment position tally (by frag_mean):")
    for key in ("first", "middle", "last"):
        print(f"      {key:<6}: {winner_positions[key]}")

    print("    winning fragment decoded identities:")
    for identity in winner_identities:
        print(f"      {identity!r}: {winner_identities[identity]}")

    read = "YES" if mean_delta > 0 else "NO"
    print(f"\n  READ: hottest emoji fragment out-attracts the hottest token of its "
          f"meaning on average: {read} (mean Δ_maxfrag = {mean_delta:+.6f})\n")


def main():
    cfg = load_config()
    name, revision = select_model(cfg)
    run_for_model(name, revision)


if __name__ == "__main__":
    main()
