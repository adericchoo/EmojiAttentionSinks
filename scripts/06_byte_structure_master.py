"""Leg 3: byte-structure characterisation — does the sink track a byte-token?

    python scripts/06_byte_structure.py              # primary model (Pythia)
    python scripts/06_byte_structure.py replication  # Qwen

Legs 1 and 2 showed an emoji draws attention beyond its meaning and that the pull
lives in a single fragment, not the span mean. This leg asks WHICH fragment: the
hypothesis is that the sink rides the leading byte-token (the "F0 9F" lead of the
supplementary-plane emoji), not "emoji-ness" in general. Stimuli are therefore
BINNED BY THE ACTUAL LEAD BYTE-TOKEN the tokenizer produces, never by the Unicode
annotation in the config.

This is additive. It reuses, without modifying:
  - the structured probe with the fixed 64-token far block     (c2.probes, via 04)
  - attention capture and span location/validation             (eas, via 04)
  - script 05's per-fragment, head-aware reduction             (05_fragment_breakdown)
  - the tokenizer's byte-level decoder                         (c2.tokaudit.token_bytes)
  - the head-aware sink rule from the BOS / positive-control checks
    (metric_validation.common.share_positive vs common.PASS_BAR)

Following the convention of scripts 04 and 05, ONE model is run per invocation
(primary by default, "replication" for Qwen); run it twice to cover both models.
"""
import importlib.util
import os
import sys

from _bootstrap import load_config, select_model

import numpy as np
import yaml
from c2.probes import Rejected
from c2.tokaudit import token_bytes
from eas.span import SpanNotFound
from metric_validation import common

# Make emoji printable on consoles whose default code page is not UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


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
    """The byte-structure stimulus set, grouped by lead byte-token."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "config", "master_stimuli.yaml")
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# ---------------------------------------------------------------------------
# Lead byte-token tagging (ground truth = the tokenizer's byte-level split)
# ---------------------------------------------------------------------------
LEAD_F0_9F = "F0 9F"
LEAD_E2 = "E2"
LEAD_OTHER = "other"
BIN_ORDER = [LEAD_F0_9F, LEAD_E2, LEAD_OTHER]


def lead_byte_tag(raw_bytes):
    """Tag a fragment by the LEADING byte(s) of its raw bytes.

    "F0 9F" is the lead pair of the supplementary-plane pictographs (U+1Fxxx);
    "E2" is the lead byte of the three-byte BMP symbols (and of the ZWJ joiner
    and variation selectors, which also start E2); everything else, including an
    ordinary ASCII word token, is "other". The actual bytes decide the tag, so a
    stimulus the config calls "E2" is still binned "other" if the tokenizer
    disagrees.
    """
    if len(raw_bytes) >= 2 and raw_bytes[0] == 0xF0 and raw_bytes[1] == 0x9F:
        return LEAD_F0_9F
    if len(raw_bytes) >= 1 and raw_bytes[0] == 0xE2:
        return LEAD_E2
    return LEAD_OTHER


def format_bytes(raw_bytes):
    """Space-separated uppercase hex, e.g. b'\\xf0\\x9f' -> 'F0 9F'."""
    parts = []
    for b in raw_bytes:
        parts.append("%02X" % b)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Head-aware sink rule (the project-wide definition of "is a sink")
# ---------------------------------------------------------------------------
def is_sink(fragment_lh, reference_lh):
    """Head-aware sink test, the SAME rule as the BOS / positive-control checks.

    `fragment_lh` and `reference_lh` are [L, H] arrays of received attention over
    the same fixed far block. The fragment is a sink when it out-attracts the
    ordinary-token reference on MORE than common.PASS_BAR of (layer, head) cells
    — i.e. common.share_positive(diff) > common.PASS_BAR, identical to
    metric_validation.first_token.verdict. Returns (is_sink, share).
    """
    difference = fragment_lh - reference_lh
    share = common.share_positive(difference)
    passed = bool(share > common.PASS_BAR)
    return passed, share


# ---------------------------------------------------------------------------
# Locating and scoring one stimulus
# ---------------------------------------------------------------------------
def locate_single(model, tok, contexts, filler):
    """Locate ONE filler in the structured probe, trying carriers in order.

    Mirrors 04's locate_pair but for a single condition. Returns
    (cap, span, queries) for the first carrier that locates and validates
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


def score_fragments(cap, tok, span, queries):
    """Per-fragment records for one located span.

    frag_mean / frag_max come straight from script 05's head-aware reduction.
    The [L, H] received-attention array used for the sink test is 05's
    fragment_received for the same single key. The raw bytes and the lead-byte
    tag come from the tokenizer's byte-level decoder, not from the glyph.
    """
    reduced = frag.per_fragment_vector(cap, tok, span, queries)
    records = []
    index = 0
    for token_index in span.token_indices:
        base = reduced[index]
        frag_tensor = frag.fragment_received(cap.attentions, token_index, queries)
        frag_lh = frag_tensor.detach().numpy()
        raw_bytes = token_bytes(tok, cap.input_ids[token_index])
        record = {
            "position": base["position"],
            "token_index": token_index,
            "token": base["token"],
            "decoded": base["decoded"],
            "frag_mean": base["frag_mean"],
            "frag_max": base["frag_max"],
            "lead_tag": lead_byte_tag(raw_bytes),
            "raw_bytes": format_bytes(raw_bytes),
            "frag_lh": frag_lh,
        }
        records.append(record)
        index = index + 1
    return records


def score_stimulus(model, tok, contexts, item, baseline_lh):
    """Locate one stimulus item, score every fragment, flag sinks.

    Handles both config item shapes with explicit branches:
      {emoji, name, codepoints, optional f0_9f_count}   -> an emoji stimulus
      {text, kind}                                       -> a reference string
    Returns None (after a warning) when the span cannot be located and
    validated, which is expected for some multi-codepoint group C entries.
    """
    if "emoji" in item:
        filler = item["emoji"]
        label = item["emoji"]
        name = item.get("name", "")
        f0_9f_count = item.get("f0_9f_count")
    else:
        filler = item["text"]
        label = item["text"]
        name = item.get("kind", "")
        f0_9f_count = None

    try:
        cap, span, queries = locate_single(model, tok, contexts, filler)
    except SpanNotFound as error:
        print(f"  [skip] {label!r} ({name}): {error}")
        return None

    records = score_fragments(cap, tok, span, queries)
    sink_positions = []
    for record in records:
        passed, share = is_sink(record["frag_lh"], baseline_lh)
        record["is_sink"] = passed
        record["sink_share"] = share
        if passed:
            sink_positions.append(record["position"])

    stimulus = {
        "label": label,
        "name": name,
        "filler": filler,
        "f0_9f_count": f0_9f_count,
        "records": records,
        "sink_positions": sink_positions,
        "sink_count": len(sink_positions),
    }
    return stimulus


# ---------------------------------------------------------------------------
# Reference baseline (group R_reference)
# ---------------------------------------------------------------------------
def build_reference(model, tok, contexts, reference_group):
    """From group R_reference, build the ordinary-token baseline and hex contrast.

    The ordinary-word baseline [L, H] is the mean received attention of every
    fragment of the `ordinary_word` items; it is the "ordinary token" that the
    head-aware sink rule compares each fragment against. The `hex_fragment`
    item's LEAD fragment is kept separately as the length-matched byte-fragment
    contrast reported on the reference line.
    """
    items = reference_group["items"]
    ordinary_frag_arrays = []
    ordinary_frag_max = []
    hex_lead_lh = None
    hex_lead_max = None
    hex_text = None

    for item in items:
        text = item["text"]
        kind = item.get("kind", "")
        try:
            cap, span, queries = locate_single(model, tok, contexts, text)
        except SpanNotFound as error:
            print(f"  [skip] reference {text!r}: {error}")
            continue
        records = score_fragments(cap, tok, span, queries)
        if kind == "ordinary_word":
            for record in records:
                ordinary_frag_arrays.append(record["frag_lh"])
                ordinary_frag_max.append(record["frag_max"])
        if kind == "hex_fragment" and hex_lead_lh is None:
            hex_lead_lh = records[0]["frag_lh"]
            hex_lead_max = records[0]["frag_max"]
            hex_text = text

    if len(ordinary_frag_arrays) == 0:
        raise SpanNotFound("no ordinary-word reference fragments could be located")

    stacked = np.stack(ordinary_frag_arrays, axis=0)
    baseline_lh = stacked.mean(axis=0)
    reference = {
        "baseline_lh": baseline_lh,
        "baseline_max": _mean(ordinary_frag_max),
        "hex_lead_lh": hex_lead_lh,
        "hex_lead_max": hex_lead_max,
        "hex_text": hex_text,
    }
    return reference


# ---------------------------------------------------------------------------
# Binning by lead byte-token
# ---------------------------------------------------------------------------
def bin_by_lead_token(stimuli):
    """Collect every scored fragment into lead-byte-token bins.

    Returns {lead_tag: {"n": int, "frag_max_values": [...], "n_sink": int}}.
    """
    bins = {}
    for tag in BIN_ORDER:
        bins[tag] = {"n": 0, "frag_max_values": [], "n_sink": 0}
    for stimulus in stimuli:
        for record in stimulus["records"]:
            tag = record["lead_tag"]
            if tag not in bins:
                bins[tag] = {"n": 0, "frag_max_values": [], "n_sink": 0}
            bins[tag]["n"] = bins[tag]["n"] + 1
            bins[tag]["frag_max_values"].append(record["frag_max"])
            if record["is_sink"]:
                bins[tag]["n_sink"] = bins[tag]["n_sink"] + 1
    return bins


# ---------------------------------------------------------------------------
# Small arithmetic helpers
# ---------------------------------------------------------------------------
def _mean(values):
    if len(values) == 0:
        return float("nan")
    total = 0.0
    for value in values:
        total = total + value
    return total / len(values)


def _lead_record(stimulus):
    if len(stimulus["records"]) == 0:
        return None
    return stimulus["records"][0]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_stimulus_table(stimulus):
    print(f"  {stimulus['label']}  ({stimulus['name']})  |S| = {len(stimulus['records'])}")
    print(f"    {'pos':>3}  {'lead':<6} {'raw bytes':<14} {'decoded':<10} "
          f"{'frag_max':>10}  {'sink':>4}")
    for record in stimulus["records"]:
        decoded = repr(record["decoded"])
        if len(decoded) > 10:
            decoded = decoded[:9] + "…"
        raw = record["raw_bytes"]
        if len(raw) > 14:
            raw = raw[:13] + "…"
        sink_mark = "yes" if record["is_sink"] else "."
        print(f"    {record['position']:>3}  {record['lead_tag']:<6} {raw:<14} "
              f"{decoded:<10} {record['frag_max']:>10.6f}  {sink_mark:>4}")
    if stimulus["f0_9f_count"] is not None:
        print(f"    => sink_count = {stimulus['sink_count']}  vs f0_9f_count = "
              f"{stimulus['f0_9f_count']}  (sink positions {stimulus['sink_positions']})")


def _print_bins(bins):
    print("  BINNED BY LEAD BYTE-TOKEN  (all A/B/C/D fragments)")
    print(f"    {'lead':<8} {'n_frag':>7} {'mean frag_max':>14} {'sink share':>12}")
    for tag in BIN_ORDER:
        if tag not in bins:
            continue
        info = bins[tag]
        if info["n"] == 0:
            continue
        mean_max = _mean(info["frag_max_values"])
        share = info["n_sink"] / info["n"]
        print(f"    {tag:<8} {info['n']:>7} {mean_max:>14.6f} {share:>12.3f}")


def _lead_sink_share(stimuli):
    """Share of stimuli whose LEAD fragment is flagged a sink (ignores skips)."""
    total = 0
    sink = 0
    for stimulus in stimuli:
        lead = _lead_record(stimulus)
        if lead is None:
            continue
        total = total + 1
        if lead["is_sink"]:
            sink = sink + 1
    if total == 0:
        return float("nan"), 0, 0
    return sink / total, sink, total


def check_prediction_a(group_a):
    print("  (A) group A — the lead F0 9F fragment sinks consistently")
    total = 0
    lead_f0 = 0
    lead_sink = 0
    for stimulus in group_a:
        lead = _lead_record(stimulus)
        if lead is None:
            continue
        total = total + 1
        if lead["lead_tag"] == LEAD_F0_9F:
            lead_f0 = lead_f0 + 1
            if lead["is_sink"]:
                lead_sink = lead_sink + 1
    if total == 0:
        print("      no group A stimuli measured")
        return
    print(f"      {lead_f0}/{total} stimuli have an F0 9F lead fragment; "
          f"{lead_sink}/{lead_f0} of those lead fragments are sinks")
    holds = lead_f0 > 0 and lead_sink == lead_f0
    verdict = "HOLDS" if holds else "MIXED / FAILS"
    print(f"      prediction: {verdict}")


def check_prediction_bd(group_b, group_d, group_a):
    print("  (B/D) groups B and D — the E2 lead fragment does NOT sink like F0 9F")
    combined = []
    for stimulus in group_b:
        combined.append(stimulus)
    for stimulus in group_d:
        combined.append(stimulus)
    bd_share, bd_sink, bd_total = _lead_sink_share(combined)
    if bd_total == 0:
        print("      no group B/D stimuli measured")
        return
    lead_e2 = 0
    for stimulus in combined:
        lead = _lead_record(stimulus)
        if lead is not None and lead["lead_tag"] == LEAD_E2:
            lead_e2 = lead_e2 + 1
    a_share, _a_sink, a_total = _lead_sink_share(group_a)
    print(f"      {lead_e2}/{bd_total} B/D stimuli have an E2 lead fragment; "
          f"lead-fragment sink share (B/D) = {bd_share:.3f}")
    if a_total > 0:
        print(f"      contrast: group A lead-fragment sink share = {a_share:.3f}")
    holds = a_total > 0 and bd_share < a_share
    verdict = "HOLDS" if holds else "NOT SHOWN"
    print(f"      prediction: {verdict}")


def check_prediction_c(group_c):
    print("  (C) group C — sink_count ~= f0_9f_count; sinks on F0 9F, not the E2 joiner")
    if len(group_c) == 0:
        print("      no group C stimuli measured")
        return
    matches = 0
    measured = 0
    for stimulus in group_c:
        measured = measured + 1
        predicted = stimulus["f0_9f_count"]
        n_sink_on_f0 = 0
        n_sink_on_e2 = 0
        n_sink_on_other = 0
        for record in stimulus["records"]:
            if not record["is_sink"]:
                continue
            if record["lead_tag"] == LEAD_F0_9F:
                n_sink_on_f0 = n_sink_on_f0 + 1
            elif record["lead_tag"] == LEAD_E2:
                n_sink_on_e2 = n_sink_on_e2 + 1
            else:
                n_sink_on_other = n_sink_on_other + 1
        n_sink = stimulus["sink_count"]
        count_ok = predicted is not None and n_sink == predicted
        placement_ok = n_sink_on_e2 == 0 and n_sink_on_other == 0 and n_sink_on_f0 == n_sink
        if count_ok and placement_ok:
            matches = matches + 1
        print(f"      {stimulus['label']:6} {stimulus['name']:24} "
              f"sinks={n_sink} (F0 9F:{n_sink_on_f0} E2:{n_sink_on_e2} other:{n_sink_on_other}) "
              f"vs f0_9f_count={predicted}")
    print(f"      {matches}/{measured} stimuli match (count == f0_9f_count AND all sinks on F0 9F)")


def _print_reference_line(f0_lead_max, reference):
    print("  REFERENCE LINE")
    if len(f0_lead_max) > 0:
        print(f"    mean frag_max, emoji F0 9F lead fragments   = {_mean(f0_lead_max):.6f}")
    else:
        print("    (no F0 9F lead fragments measured)")
    print(f"    mean frag_max, ordinary-word reference token  = {reference['baseline_max']:.6f}")
    if reference["hex_lead_max"] is not None:
        print(f"    frag_max, hex_fragment lead ({reference['hex_text']}) = {reference['hex_lead_max']:.6f}")
    else:
        print("    (no hex_fragment reference located)")


def _print_read(model_name, bins):
    empty = {"n": 0, "n_sink": 0, "frag_max_values": []}
    f0 = bins.get(LEAD_F0_9F, empty)
    e2 = bins.get(LEAD_E2, empty)
    other = bins.get(LEAD_OTHER, empty)
    f0_max = _mean(f0["frag_max_values"]) if f0["n"] else float("nan")
    e2_max = _mean(e2["frag_max_values"]) if e2["n"] else float("nan")
    other_max = _mean(other["frag_max_values"]) if other["n"] else float("nan")
    f0_flag = f0["n_sink"] / f0["n"] if f0["n"] else float("nan")
    e2_flag = e2["n_sink"] / e2["n"] if e2["n"] else float("nan")
    # The byte-token localisation signal is the per-fragment peak (frag_max): the
    # sink tracks F0 9F when F0 9F fragments peak far higher than E2 and other.
    # The head-aware flag is the project's strict "broad across heads" sink bar;
    # a concentrated peak can dominate frag_max yet stay under it, exactly as the
    # punctuation positive control does.
    tracks = f0["n"] > 0 and e2["n"] > 0 and f0_max > e2_max and f0_max > other_max
    answer = "YES" if tracks else "NO"
    print(f"  READ [{model_name}]: does the sink track the F0 9F byte-token specifically? {answer}")
    print(f"    by peak (mean frag_max):   F0 9F = {f0_max:.3f}  E2 = {e2_max:.3f}  other = {other_max:.3f}")
    print(f"    by head-aware flag (share reaching the > {common.PASS_BAR:.2f} sink bar): "
          f"F0 9F = {f0_flag:.3f}  E2 = {e2_flag:.3f}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
EMOJI_GROUPS = ["A_smp_pictographs", "B_bmp_e2_symbols", "C_multi_f0_9f", "D_nonemoji_e2"]


def run_for_model(name, revision, stimuli):
    model, tok = frag.mmc.load_model(name, revision=revision)
    contexts = frag.mmc.build_contexts(tok)
    groups = stimuli["groups"]

    print(f"\nBYTE-STRUCTURE CHARACTERISATION  —  {name}")
    print("structured probe, fixed 64-token far block; per-fragment head-aware reduction (05)")
    print("binned by ACTUAL lead byte-token; sink = head-aware rule vs the R_reference ordinary word\n")

    reference = build_reference(model, tok, contexts, groups["R_reference"])
    baseline_lh = reference["baseline_lh"]

    scored = {}
    for group_name in EMOJI_GROUPS:
        group = groups[group_name]
        print(f"GROUP {group_name}")
        print(f"  tests: {group['tests']}")
        stimuli_list = []
        for item in group["items"]:
            stimulus = score_stimulus(model, tok, contexts, item, baseline_lh)
            if stimulus is None:
                continue
            _print_stimulus_table(stimulus)
            stimuli_list.append(stimulus)
        scored[group_name] = stimuli_list
        print()

    all_stimuli = []
    for group_name in EMOJI_GROUPS:
        for stimulus in scored[group_name]:
            all_stimuli.append(stimulus)
    bins = bin_by_lead_token(all_stimuli)
    _print_bins(bins)
    print()

    print("PREDICTION CHECKS")
    check_prediction_a(scored["A_smp_pictographs"])
    check_prediction_bd(scored["B_bmp_e2_symbols"], scored["D_nonemoji_e2"], scored["A_smp_pictographs"])
    check_prediction_c(scored["C_multi_f0_9f"])
    print()

    f0_lead_max = []
    for group_name in ("A_smp_pictographs", "C_multi_f0_9f"):
        for stimulus in scored[group_name]:
            lead = _lead_record(stimulus)
            if lead is not None and lead["lead_tag"] == LEAD_F0_9F:
                f0_lead_max.append(lead["frag_max"])
    _print_reference_line(f0_lead_max, reference)
    print()

    _print_read(name, bins)
    print()


def main():
    cfg = load_config()
    name, revision = select_model(cfg)
    stimuli = load_stimuli()
    run_for_model(name, revision, stimuli)


if __name__ == "__main__":
    main()
