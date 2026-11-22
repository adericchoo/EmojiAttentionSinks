"""Leg 11: WHO attends to F0 9F — a confirming figure for the source/target
decoupling the causal ablation (leg 10) found.

    python scripts/11_who_attends.py              # primary model (Pythia-410m)

RE-ANALYSIS ONLY. Clean forward passes, no intervention, no ablation. Leg 10 showed
that blocking the F0 9F sink-head attention edges is a near-no-op, while the token's
VALUE and ordinary (non-sink) heads do the real local assembly read. This script
shows the SOURCE side descriptively: where the attention TO the F0 9F key comes
from, by query distance, split into the SAME sink / non-sink head groups leg 10 used.

Expected shape if the decoupling holds:
  - non-sink heads attend to F0 9F PEAKED at the nearby continuation bytes (the
    functional assembly read), and
  - sink heads attend to F0 9F broadly / sustained at long range (the idle sink,
    roughly uniform across distant positions = default "parking").

This is additive. It reuses, without modifying:
  - clean capture, span location, model loading              (eas, via scripts/10)
  - the SAME natural carrier stimuli and the SAME sink-head set as leg 10
    (scripts/10_causal_ablation.py: prepare_stimulus, sink_head_mask, ...)
  - script 06's byte-level F0 9F tagging                      (06, reached via 10)
Nothing in src/eas or scripts 04-10 is changed; helpers live here.

Model: Pythia-410m (same as leg 10).
"""
import importlib.util
import os
import sys

from _bootstrap import load_config, select_model

import numpy as np

from eas import load_model

# Make emoji printable on consoles whose default code page is not UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _load_causal_ablation_module():
    """Import script 10 (name begins with a digit -> load by path). It carries the
    leg-10 stimuli, prepare_stimulus, and the sink-head rule, so this re-analysis
    lines up with the ablation exactly."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "10_causal_ablation.py")
    spec = importlib.util.spec_from_file_location("causal_ablation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ca = _load_causal_ablation_module()


# ---------------------------------------------------------------------------
# Distance bins for the source-side profile. d = q - k, where k is the F0 9F key
# position and q is a query at or after it. d = 0 is the F0 9F token attending to
# itself; d = 1, 2 are normally the continuation bytes; larger d is "far".
# ---------------------------------------------------------------------------
DISTANCE_BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 10 ** 9)]
# Bins counted as "far" for the long-range uniformity read (low edge >= this).
FAR_LOW_EDGE = 5
# A coefficient of variation at or below this, across the far bins, reads as roughly
# uniform (default parking); clearly above it reads as structured / peaked.
UNIFORM_COV_BAR = 0.25
# How many top non-sink "assembly reader" heads to name.
TOP_READERS = 6


def distance_bin(distance):
    for low, high in DISTANCE_BINS:
        if low <= distance <= high:
            return (low, high)
    return None


# ---------------------------------------------------------------------------
# Per-stimulus collection (pure; explicit loops)
# ---------------------------------------------------------------------------
def collect_distance_values(attentions, f0_index, sink_mask):
    """Every (distance, is_sink, attention) triple for one stimulus.

    For each query position q at or after the F0 9F key, and each (layer, head),
    record the attention weight A[l, h, q, f0] and the distance d = q - f0, tagged
    by whether that head is a sink head. Returns a list of (distance, is_sink,
    value) tuples.
    """
    n_layers = attentions.shape[0]
    n_heads = attentions.shape[1]
    n = attentions.shape[-1]
    triples = []
    for query in range(f0_index, n):
        distance = query - f0_index
        for layer in range(n_layers):
            for head in range(n_heads):
                value = float(attentions[layer, head, query, f0_index])
                is_sink = bool(sink_mask[layer, head])
                triples.append((distance, is_sink, value))
    return triples


def collect_region_values(attentions, f0_index, sink_mask, query_positions):
    """Attention to F0 9F from a specific set of query positions, split by head
    group. Returns (sink_values, non_sink_values) as flat lists over (head, query).
    """
    n_layers = attentions.shape[0]
    n_heads = attentions.shape[1]
    sink_values = []
    non_sink_values = []
    for query in query_positions:
        for layer in range(n_layers):
            for head in range(n_heads):
                value = float(attentions[layer, head, query, f0_index])
                if sink_mask[layer, head]:
                    sink_values.append(value)
                else:
                    non_sink_values.append(value)
    return sink_values, non_sink_values


def collect_head_near_attention(attentions, f0_index, sink_mask, continuation_positions):
    """Per NON-sink head, its mean attention to F0 9F from the continuation-byte
    queries (the near / assembly read). Returns {(layer, head): [values]} appended
    across queries, for later aggregation over stimuli."""
    n_layers = attentions.shape[0]
    n_heads = attentions.shape[1]
    per_head = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            if sink_mask[layer, head]:
                continue
            values = []
            for query in continuation_positions:
                values.append(float(attentions[layer, head, query, f0_index]))
            per_head[(layer, head)] = values
    return per_head


# ---------------------------------------------------------------------------
# Small arithmetic helpers (explicit)
# ---------------------------------------------------------------------------
def mean_of(values):
    if len(values) == 0:
        return float("nan")
    total = 0.0
    for value in values:
        total = total + value
    return total / len(values)


def std_of(values):
    if len(values) < 2:
        return float("nan")
    average = mean_of(values)
    total = 0.0
    for value in values:
        total = total + (value - average) * (value - average)
    return (total / (len(values) - 1)) ** 0.5


def coefficient_of_variation(values):
    average = mean_of(values)
    if average == 0.0 or average != average:
        return float("nan")
    spread = std_of(values)
    if spread != spread:
        return float("nan")
    return spread / average


def continuation_positions_of(record):
    positions = []
    for position in range(len(record["classes"])):
        if record["classes"][position] == ca.CLASS_CONTINUATION:
            positions.append(position)
    return positions


# ---------------------------------------------------------------------------
# Aggregation across stimuli
# ---------------------------------------------------------------------------
def empty_profile():
    profile = {}
    for low, high in DISTANCE_BINS:
        profile[(low, high)] = {"sink": [], "non": []}
    return profile


def add_to_profile(profile, triples):
    for distance, is_sink, value in triples:
        key = distance_bin(distance)
        if key is None:
            continue
        if is_sink:
            profile[key]["sink"].append(value)
        else:
            profile[key]["non"].append(value)


def merge_head_near(total_head_near, per_head):
    for key in per_head:
        if key not in total_head_near:
            total_head_near[key] = []
        for value in per_head[key]:
            total_head_near[key].append(value)


# ---------------------------------------------------------------------------
# Optional figure
# ---------------------------------------------------------------------------
def try_save_figure(model_name, profile):
    """Save a simple two-series line plot to outputs/ if matplotlib is importable.
    Returns the saved path, or None when matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    labels = []
    sink_means = []
    non_means = []
    for low, high in DISTANCE_BINS:
        cell = profile[(low, high)]
        if len(cell["sink"]) == 0 and len(cell["non"]) == 0:
            continue
        labels.append(_bin_label((low, high)))
        sink_means.append(mean_of(cell["sink"]))
        non_means.append(mean_of(cell["non"]))

    xs = []
    for index in range(len(labels)):
        xs.append(index)
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.plot(xs, sink_means, marker="o", label="sink heads")
    axis.plot(xs, non_means, marker="s", label="non-sink heads")
    axis.set_xticks(xs)
    axis.set_xticklabels(labels, rotation=0)
    axis.set_xlabel("query distance from the F0 9F key (d = q - k)")
    axis.set_ylabel("mean attention to the F0 9F key")
    axis.set_title("Who attends to F0 9F — source side (clean)")
    axis.legend()
    figure.tight_layout()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "outputs")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    safe_name = model_name.replace("/", "_")
    path = os.path.join(out_dir, f"leg11_who_attends_{safe_name}.png")
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _bin_label(key):
    low, high = key
    if low == high:
        return f"{low}"
    if high >= 10 ** 9:
        return f"{low}+"
    return f"{low}-{high}"


def print_profile(model_name, profile, sink_count, n_cells):
    print(f"\nWHO ATTENDS TO F0 9F  —  {model_name}  (clean, no intervention)")
    print(f"sink-head set reused from leg 10: {sink_count} of {n_cells} cells\n")
    print("  DISTANCE PROFILE — mean attention to the F0 9F key by query distance d = q - k")
    header = (f"    {'distance':>9} {'sink mean':>12} {'sink sd':>10} {'non mean':>12} "
              f"{'non sd':>10} {'n(head*q)':>10}")
    print(header)
    print("    " + "-" * (len(header) - 4))
    for low, high in DISTANCE_BINS:
        cell = profile[(low, high)]
        count = len(cell["sink"]) + len(cell["non"])
        if count == 0:
            continue
        label = _bin_label((low, high))
        print(f"    {label:>9} {mean_of(cell['sink']):>12.6f} {std_of(cell['sink']):>10.6f} "
              f"{mean_of(cell['non']):>12.6f} {std_of(cell['non']):>10.6f} {count:>10}")


def print_region_table(region):
    print("\n  REGION TABLE — mean attention to F0 9F, head group x query region "
          "(ties to leg 10's target split)")
    header = (f"    {'head group':14} {'continuation-byte q':>22} {'following/distant q':>22}")
    print(header)
    print("    " + "-" * (len(header) - 4))
    sink_cont = mean_of(region["continuation"]["sink"])
    sink_foll = mean_of(region["following"]["sink"])
    non_cont = mean_of(region["continuation"]["non"])
    non_foll = mean_of(region["following"]["non"])
    print(f"    {'sink':14} {sink_cont:>22.6f} {sink_foll:>22.6f}")
    print(f"    {'non-sink':14} {non_cont:>22.6f} {non_foll:>22.6f}")
    return sink_cont, sink_foll, non_cont, non_foll


def far_sink_means(profile):
    means = []
    for low, high in DISTANCE_BINS:
        if low < FAR_LOW_EDGE:
            continue
        cell = profile[(low, high)]
        if len(cell["sink"]) == 0:
            continue
        means.append(mean_of(cell["sink"]))
    return means


def print_top_readers(total_head_near):
    scored = []
    for key in total_head_near:
        scored.append((mean_of(total_head_near[key]), key))
    scored.sort(reverse=True)
    print(f"\n  TOP NON-SINK 'ASSEMBLY READERS' (highest near / continuation-byte attention to F0 9F)")
    shown = 0
    for average, key in scored:
        if shown >= TOP_READERS:
            break
        layer, head = key
        print(f"    layer {layer:>2} head {head:>2}   mean near attention = {average:.6f}")
        shown = shown + 1


def print_read(model_name, region_cells, cov_far):
    sink_cont, sink_foll, non_cont, non_foll = region_cells
    print(f"\n  READ [{model_name}]:")
    non_is_near_peaked = non_cont > non_foll
    sink_is_far_sustained = sink_foll >= sink_cont or sink_foll > 0.0
    print(f"    non-sink heads: continuation-byte attention {non_cont:.6f} vs following/distant "
          f"{non_foll:.6f} -> {'PEAKED near (assembly read)' if non_is_near_peaked else 'not near-peaked'}.")
    print(f"    sink heads: continuation-byte attention {sink_cont:.6f} vs following/distant "
          f"{sink_foll:.6f} -> {'broad / sustained at range' if sink_is_far_sustained else 'also local'}.")

    if cov_far != cov_far:
        print("    long-range sink shape: not enough far bins to judge uniformity.")
    elif cov_far <= UNIFORM_COV_BAR:
        print(f"    long-range sink shape: CoV across far distances = {cov_far:.3f} (<= {UNIFORM_COV_BAR}) "
              "-> roughly UNIFORM, consistent with default 'parking' behaviour.")
    else:
        print(f"    long-range sink shape: CoV across far distances = {cov_far:.3f} (> {UNIFORM_COV_BAR}) "
              "-> structured / peaked, not a flat park.")

    if non_is_near_peaked and sink_is_far_sustained:
        print("    => CONFIRMS the decoupling: non-sink heads do a LOCAL functional read at the "
              "continuation bytes, while the sink heads hold a broad long-range attraction — the "
              "idle sink leg 10 could ablate with almost no loss change.")
    elif not non_is_near_peaked:
        print("    => the non-sink near read is NOT peaked; the source-side picture does not cleanly "
              "mirror the ablation and should be read with care.")
    else:
        print("    => the sink attention is also local here; this COMPLICATES the ablation's story "
              "and is worth flagging.")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_for_model(name, revision):
    model, tok = load_model(name, revision=revision)
    model_type = model.config.model_type
    n_layers = int(model.config.num_hidden_layers)
    n_heads = int(model.config.num_attention_heads)
    n_cells = n_layers * n_heads

    print(f"\nWHO ATTENDS TO F0 9F (SOURCE SIDE)  —  {name}  ({model_type})")
    print("clean captures; same stimuli and sink-head set as leg 10; no intervention\n")

    # ---- Build the leg-10 stimuli and sink-head set (clean passes only) ----
    records = []
    emoji_items = ca.bs06.load_stimuli()["groups"]["A_smp_pictographs"]["items"]
    template_index = 0
    for item in emoji_items:
        emoji = item["emoji"]
        template = ca.CARRIER_TEMPLATES[template_index % len(ca.CARRIER_TEMPLATES)]
        template_index = template_index + 1
        record = ca.prepare_stimulus(model, tok, template, emoji)
        if record is not None:
            records.append(record)

    if len(records) == 0:
        raise RuntimeError("no F0 9F stimulus could be prepared")

    emoji_received_list = []
    baseline_received_list = []
    for record in records:
        emoji_received_list.append(record["emoji_received"])
        baseline_received_list.append(record["baseline_received"])
    emoji_received_mean = ca._mean_arrays(emoji_received_list)
    baseline_received_mean = ca._mean_arrays(baseline_received_list)
    sink_mask = ca.sink_head_mask(emoji_received_mean, baseline_received_mean, ca.SINK_HEAD_RATIO)
    sink_count = ca.count_true(sink_mask)

    print(f"  prepared {len(records)} F0 9F stimuli; sink-head set = {sink_count} of {n_cells} cells\n")

    # ---- Aggregate the source-side attention over stimuli ----
    profile = empty_profile()
    region = {
        "continuation": {"sink": [], "non": []},
        "following": {"sink": [], "non": []},
    }
    total_head_near = {}

    for record in records:
        attentions = record["clean_attentions"]
        f0_index = record["f0_index"]

        triples = collect_distance_values(attentions, f0_index, sink_mask)
        add_to_profile(profile, triples)

        continuation_positions = continuation_positions_of(record)
        following_positions = record["query_indices"]

        cont_sink, cont_non = collect_region_values(
            attentions, f0_index, sink_mask, continuation_positions
        )
        foll_sink, foll_non = collect_region_values(
            attentions, f0_index, sink_mask, following_positions
        )
        for value in cont_sink:
            region["continuation"]["sink"].append(value)
        for value in cont_non:
            region["continuation"]["non"].append(value)
        for value in foll_sink:
            region["following"]["sink"].append(value)
        for value in foll_non:
            region["following"]["non"].append(value)

        per_head = collect_head_near_attention(
            attentions, f0_index, sink_mask, continuation_positions
        )
        merge_head_near(total_head_near, per_head)

    # ---- Report ----
    print_profile(name, profile, sink_count, n_cells)
    region_cells = print_region_table(region)
    cov_far = coefficient_of_variation(far_sink_means(profile))
    print_top_readers(total_head_near)
    print_read(name, region_cells, cov_far)

    figure_path = try_save_figure(name, profile)
    if figure_path is not None:
        print(f"\n  figure saved to {figure_path}")
    else:
        print("\n  (matplotlib unavailable; table only, no figure saved)")
    print()


def main():
    cfg = load_config()
    name, revision = select_model(cfg)
    run_for_model(name, revision)


if __name__ == "__main__":
    main()
