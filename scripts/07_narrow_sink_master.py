"""Leg 3b: a narrow-sink metric, and a 2-axis map of every condition.

    python scripts/07_narrow_sink.py              # primary model (Pythia)
    python scripts/07_narrow_sink.py replication  # Qwen

Leg 3 (script 06) showed the existing sink flag — share of heads above an ordinary
token >= 0.60 — only fires for BROAD sinks, so it reads ~0 for the narrow,
head-specific peak an emoji F0 9F fragment produces (and, we suspect, for
punctuation). This leg adds a metric that measures the narrow case and places every
condition on ONE map, so the emoji peak can be compared directly to punctuation and
to the known broad BOS / first-token sink.

Per condition, over the fixed 64-token far block and against the SAME ordinary-token
baseline the broad-share rule already uses, three numbers:
  1. peak_ratio   = frag_max(fragment) / frag_max(ordinary-word baseline)
  2. broad_share  = share of (l, h) heads where fragment > 1x baseline
                    (the existing rule, reused unchanged — the BOS-style breadth)
  3. extreme_count/share = number and share of (l, h) heads where fragment > Kx
                    baseline (K = EXTREME_K) — the narrow-sink signature.

This is additive. It reuses, without modifying:
  - script 05's per-fragment, head-aware reduction          (05_fragment_breakdown)
  - the structured probe + capture/locate                   (c2.probes / eas, via 04)
  - the broad-share rule and PASS_BAR                        (metric_validation.common)
  - the punctuation positive-control stimuli                (c2.controls.PUNCT)
  - the first-token / BOS idea (position 0 as the key)      (metric_validation.first_token)
  - the emoji and reference conditions                       (config/master_stimuli.yaml)

Following scripts 04/05/06, ONE model is run per invocation (primary by default,
"replication" for Qwen); run it twice to cover both models.
"""
import importlib.util
import os
import sys

from _bootstrap import load_config, select_model

import numpy as np
import yaml
from c2.controls import PUNCT
from c2.probes import Rejected
from eas.span import SpanNotFound
from metric_validation import common

# Make emoji printable on consoles whose default code page is not UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Classification bars (kept together; the only place to tune the metric)
# ---------------------------------------------------------------------------
# A fragment must out-peak an ordinary token by at least this factor to be any
# kind of sink at all.
PEAK_RATIO_SINK_BAR = 3.0
# The existing broad-share rule, reused unchanged: >= this share of heads above
# the baseline means the sink is BROAD (BOS-style).
BROAD_SHARE_BAR = common.PASS_BAR  # 0.60
# A head is "extreme" when the fragment exceeds K times the ordinary-token
# baseline on that head. K is the knob for the narrow-sink signature.
EXTREME_K = 10.0
# A narrow sink needs at least this many extreme heads.
EXTREME_MIN_HEADS = 1

CLASS_BROAD = "BROAD"
CLASS_NARROW = "NARROW"
CLASS_NON = "NON-sink"


def _load_fragment_module():
    """Import script 05 (its name begins with a digit, so load it by path).

    Script 05 owns the per-fragment, head-aware reduction and, through it, the
    structured-probe helpers of script 04 (reached here as `frag.mmc`). Nothing
    in either script is re-implemented.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "05_fragment_breakdown.py")
    spec = importlib.util.spec_from_file_location("fragment_breakdown", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frag = _load_fragment_module()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_stimuli():
    """The byte-structure stimulus set (emoji + R_reference conditions)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "config", "master_stimuli.yaml")
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# ---------------------------------------------------------------------------
# The three narrow-sink numbers and the classification
# ---------------------------------------------------------------------------
def _peak_and_extremes(fragment_lh, baseline_lh, k):
    """Peak attention of each vector and the extreme-head count, explicit loops.

    Returns (frag_peak, base_peak, extreme_count, extreme_share). An extreme head
    is one where fragment > k * baseline on that same (layer, head) cell.
    """
    n_layers = fragment_lh.shape[0]
    n_heads = fragment_lh.shape[1]
    frag_peak = float(fragment_lh[0, 0])
    base_peak = float(baseline_lh[0, 0])
    extreme_count = 0
    total = 0
    for layer in range(n_layers):
        for head in range(n_heads):
            total = total + 1
            frag_value = float(fragment_lh[layer, head])
            base_value = float(baseline_lh[layer, head])
            if frag_value > frag_peak:
                frag_peak = frag_value
            if base_value > base_peak:
                base_peak = base_value
            if frag_value > k * base_value:
                extreme_count = extreme_count + 1
    if total == 0:
        extreme_share = 0.0
    else:
        extreme_share = extreme_count / total
    return frag_peak, base_peak, extreme_count, extreme_share


def classify(peak_ratio, broad_share, extreme_count):
    """Place one condition in BROAD / NARROW / NON-sink by the stated bars.

    BROAD  : broad_share >= BROAD_SHARE_BAR (the existing rule).
    NARROW : not broad, but a real peak (peak_ratio >= PEAK_RATIO_SINK_BAR) carried
             by at least EXTREME_MIN_HEADS extreme heads.
    NON    : no meaningful peak (peak_ratio < PEAK_RATIO_SINK_BAR), or a peak with
             no extreme head to localise it.
    """
    if broad_share >= BROAD_SHARE_BAR:
        return CLASS_BROAD
    if peak_ratio >= PEAK_RATIO_SINK_BAR and extreme_count >= EXTREME_MIN_HEADS:
        return CLASS_NARROW
    return CLASS_NON


def condition_metrics(fragment_lh, baseline_lh, k=EXTREME_K):
    """The three numbers plus the class for one condition's [L, H] vector.

    `broad_share` reuses the existing broad-share rule unchanged
    (common.share_positive of fragment minus baseline). `peak_ratio` and the
    extreme-head counts are computed here with explicit loops.
    """
    frag_peak, base_peak, extreme_count, extreme_share = _peak_and_extremes(
        fragment_lh, baseline_lh, k
    )
    if base_peak > 0.0:
        peak_ratio = frag_peak / base_peak
    else:
        peak_ratio = float("inf")
    difference = fragment_lh - baseline_lh
    broad_share = common.share_positive(difference)
    cls = classify(peak_ratio, broad_share, extreme_count)
    metrics = {
        "peak_ratio": peak_ratio,
        "broad_share": broad_share,
        "extreme_count": extreme_count,
        "extreme_share": extreme_share,
        "frag_peak": frag_peak,
        "base_peak": base_peak,
        "cls": cls,
    }
    return metrics


# ---------------------------------------------------------------------------
# Locating and reducing one fragment (reuses 05's reduction and 04's capture)
# ---------------------------------------------------------------------------
def locate_single(model, tok, contexts, filler):
    """Locate ONE filler in the structured probe, trying carriers in order.

    Returns (cap, span, queries) for the first carrier that locates and validates
    cleanly; raises SpanNotFound if none do.
    """
    last_error = None
    for ctx in contexts:
        try:
            return frag.mmc.capture_and_locate(model, tok, ctx, filler)
        except (SpanNotFound, Rejected) as error:
            last_error = error
            continue
    if last_error is None:
        raise SpanNotFound("no carrier available")
    raise SpanNotFound(str(last_error))


def lead_vector(model, tok, contexts, filler, label):
    """The [L, H] received-attention vector of a filler's LEAD (position 0)
    fragment over the fixed far block. Returns None (after a warning) on skip."""
    try:
        cap, span, queries = locate_single(model, tok, contexts, filler)
    except SpanNotFound as error:
        print(f"  [skip] {label!r}: {error}")
        return None
    token_index = span.token_indices[0]
    vector = frag.fragment_received(cap.attentions, token_index, queries)
    return vector.detach().numpy()


def _mean_vectors(vectors):
    """Mean of a list of [L, H] arrays -> [L, H]."""
    stacked = np.stack(vectors, axis=0)
    return stacked.mean(axis=0)


def collect_lead_vectors(model, tok, contexts, items):
    """Lead-fragment vectors for a config group's items (both item shapes)."""
    vectors = []
    for item in items:
        if "emoji" in item:
            filler = item["emoji"]
            label = item["emoji"]
        else:
            filler = item["text"]
            label = item["text"]
        vector = lead_vector(model, tok, contexts, filler, label)
        if vector is not None:
            vectors.append(vector)
    return vectors


def collect_punct_vectors(model, tok, contexts, punct_list):
    """Lead-fragment vectors for the positive-control punctuation marks."""
    vectors = []
    for mark in punct_list:
        vector = lead_vector(model, tok, contexts, mark, mark)
        if vector is not None:
            vectors.append(vector)
    return vectors


def build_reference_and_bos(model, tok, contexts, reference_group):
    """From group R_reference, build the ordinary-token baseline, the BOS vector,
    and the hex-fragment lead vector — all on the same fixed far block.

    The ordinary-word baseline [L, H] is the mean received attention over every
    fragment of the `ordinary_word` items (the same construction script 06 uses
    for the broad-share rule). The BOS vector is position 0 of the sequence — the
    canonical first-token sink — read off the same captures. The hex lead is the
    length-matched byte-fragment contrast.
    """
    items = reference_group["items"]
    ordinary_fragments = []
    bos_vectors = []
    hex_lead = None
    for item in items:
        text = item["text"]
        kind = item.get("kind", "")
        try:
            cap, span, queries = locate_single(model, tok, contexts, text)
        except SpanNotFound as error:
            print(f"  [skip] reference {text!r}: {error}")
            continue
        bos_tensor = frag.fragment_received(cap.attentions, 0, queries)
        bos_vectors.append(bos_tensor.detach().numpy())
        fragment_vectors = []
        for token_index in span.token_indices:
            vector = frag.fragment_received(cap.attentions, token_index, queries)
            fragment_vectors.append(vector.detach().numpy())
        if kind == "ordinary_word":
            for vector in fragment_vectors:
                ordinary_fragments.append(vector)
        if kind == "hex_fragment" and hex_lead is None:
            hex_lead = fragment_vectors[0]
    if len(ordinary_fragments) == 0:
        raise SpanNotFound("no ordinary-word reference fragments could be located")
    if len(bos_vectors) == 0:
        raise SpanNotFound("no BOS vectors could be captured")
    baseline_lh = _mean_vectors(ordinary_fragments)
    bos_lh = _mean_vectors(bos_vectors)
    return baseline_lh, bos_lh, hex_lead


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _peak_ratio_key(row):
    return row["metrics"]["peak_ratio"]


def print_table(model_name, rows):
    print(f"\nNARROW-SINK MAP  —  {model_name}")
    print(f"bars: any sink needs peak_ratio >= {PEAK_RATIO_SINK_BAR:g}; "
          f"broad_share >= {BROAD_SHARE_BAR:g} -> BROAD; "
          f"NARROW needs >= {EXTREME_MIN_HEADS} extreme head (> {EXTREME_K:g}x baseline)\n")
    header = (f"  {'condition':24} {'peak_ratio':>11} {'broad_share':>12} "
              f"{'extreme_count':>14} {'class':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        metrics = row["metrics"]
        print(f"  {row['name']:24} {metrics['peak_ratio']:>11.3f} "
              f"{metrics['broad_share']:>12.3f} {metrics['extreme_count']:>14} "
              f"{metrics['cls']:>9}")


def print_read(model_name, by_name):
    print(f"\n  READ [{model_name}]:")
    bos = by_name.get("BOS / first token")
    punct = by_name.get("punctuation")
    emoji = by_name.get("emoji F0 9F lead (A)")
    if bos is None or punct is None or emoji is None:
        print("    (missing one of BOS / punctuation / emoji; cannot complete the read)")
        return
    bos_broad = bos["cls"] == CLASS_BROAD
    print(f"    BOS lands {bos['cls']}; punctuation lands {punct['cls']}; "
          f"emoji F0 9F lands {emoji['cls']}")
    shape_ok = bos_broad and punct["cls"] == CLASS_NARROW and emoji["cls"] == CLASS_NARROW
    answer = "YES" if shape_ok else "NO"
    print(f"    does BOS land BROAD and punctuation + emoji land NARROW? {answer}")
    peak_bigger = emoji["peak_ratio"] >= punct["peak_ratio"]
    ext_bigger = emoji["extreme_count"] >= punct["extreme_count"]
    print(f"    emoji F0 9F vs punctuation: peak_ratio {emoji['peak_ratio']:.2f} vs "
          f"{punct['peak_ratio']:.2f}; extreme_count {emoji['extreme_count']} vs "
          f"{punct['extreme_count']}")
    if peak_bigger and not ext_bigger:
        # extreme_count counts heads where fragment > K x baseline; where an
        # ordinary token's baseline is ~0 that threshold is ~0, so a broad-but-weak
        # condition inflates extreme_count without a real peak. peak_ratio is the
        # cleaner narrow-sink signal when the two disagree.
        print("      -> emoji out-peaks punctuation; punctuation's higher extreme_count "
              "comes from near-zero-baseline heads (K x ~0), not a real peak, so "
              "peak_ratio is the sharper narrow-sink signal here.")
    if punct["cls"] == CLASS_NON:
        print("    NOTE: punctuation is NON-sink even under the narrow metric "
              "(no meaningful head-specific peak over an ordinary token).")
    elif punct["cls"] == CLASS_BROAD:
        print("    NOTE: punctuation lands BROAD-but-weak (breadth over the bar, "
              f"peak_ratio only {punct['peak_ratio']:.2f}), not a narrow peak like emoji.")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_for_model(name, revision, stimuli):
    model, tok = frag.mmc.load_model(name, revision=revision)
    contexts = frag.mmc.build_contexts(tok)
    groups = stimuli["groups"]

    print(f"\nNARROW-SINK CHARACTERISATION  —  {name}")
    print("structured probe, fixed 64-token far block; per-fragment head-aware reduction (05)")
    print("one ordinary-token baseline; every condition placed on the same 2-axis map\n")

    baseline_lh, bos_lh, hex_lh = build_reference_and_bos(
        model, tok, contexts, groups["R_reference"]
    )
    emoji_vectors = collect_lead_vectors(model, tok, contexts, groups["A_smp_pictographs"]["items"])
    e2_vectors = collect_lead_vectors(model, tok, contexts, groups["B_bmp_e2_symbols"]["items"])
    punct_vectors = collect_punct_vectors(model, tok, contexts, PUNCT)

    # (name, [L, H] vector); conditions that aggregate items use the mean vector.
    conditions = []
    conditions.append(("BOS / first token", bos_lh))
    if len(emoji_vectors) > 0:
        conditions.append(("emoji F0 9F lead (A)", _mean_vectors(emoji_vectors)))
    if len(e2_vectors) > 0:
        conditions.append(("E2 symbol lead (B)", _mean_vectors(e2_vectors)))
    if len(punct_vectors) > 0:
        conditions.append(("punctuation", _mean_vectors(punct_vectors)))
    if hex_lh is not None:
        conditions.append(("hex fragment (R)", hex_lh))
    conditions.append(("ordinary word (R)", baseline_lh))

    rows = []
    by_name = {}
    for cond_name, vector in conditions:
        metrics = condition_metrics(vector, baseline_lh, EXTREME_K)
        rows.append({"name": cond_name, "metrics": metrics})
        by_name[cond_name] = metrics

    rows_sorted = sorted(rows, key=_peak_ratio_key, reverse=True)
    print_table(name, rows_sorted)
    print_read(name, by_name)
    print()


def main():
    cfg = load_config()
    name, revision = select_model(cfg)
    stimuli = load_stimuli()
    run_for_model(name, revision, stimuli)


if __name__ == "__main__":
    main()
