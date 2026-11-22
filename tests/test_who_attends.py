"""Offline test for the leg-11 source-side binning and sink/non-sink aggregation.

No model is loaded and nothing is downloaded. A constructed attention tensor with a
known F0 9F key column is fed through the per-stimulus collectors, and the distance
binning + sink/non-sink split must return the expected profile and region values.
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Insert scripts first, then src, so src/ is at the FRONT: scripts/c2.py would
# otherwise shadow the src/c2 package (the numbered scripts avoid this because
# _bootstrap puts src/ at the front before c2 is first imported).
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import torch


def _load_script_module():
    path = os.path.join(ROOT, "scripts", "11_who_attends.py")
    spec = importlib.util.spec_from_file_location("who_attends", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wa = _load_script_module()


def _constructed_attention():
    """A tiny attentions tensor [L=1, H=2, n=6, n=6] with a KNOWN F0 9F key column.

    The F0 9F key sits at position f0 = 2. Head 0 is the sink head, head 1 is the
    non-sink head. The attention A[0, h, q, f0] for queries q >= f0 is set to a
    distinct, easy-to-check value per (head, distance):
      - head 0 (sink):     0.10 at every distance (flat / sustained)
      - head 1 (non-sink): 0.90 at distance 1, else 0.00 (peaked near)
    Every other entry is 0.
    """
    attentions = torch.zeros(1, 2, 6, 6)
    f0 = 2
    for query in range(f0, 6):
        distance = query - f0
        attentions[0, 0, query, f0] = 0.10
        if distance == 1:
            attentions[0, 1, query, f0] = 0.90
        else:
            attentions[0, 1, query, f0] = 0.00
    return attentions, f0


def _sink_mask():
    mask = np.zeros((1, 2), dtype=bool)
    mask[0, 0] = True   # head 0 is the sink head
    mask[0, 1] = False  # head 1 is non-sink
    return mask


def test_distance_binning_and_group_split():
    attentions, f0 = _constructed_attention()
    sink_mask = _sink_mask()

    triples = wa.collect_distance_values(attentions, f0, sink_mask)
    profile = wa.empty_profile()
    wa.add_to_profile(profile, triples)

    # Distance 1 bin: sink head flat 0.10, non-sink head peaked 0.90.
    cell_one = profile[(1, 1)]
    assert abs(wa.mean_of(cell_one["sink"]) - 0.10) < 1e-6
    assert abs(wa.mean_of(cell_one["non"]) - 0.90) < 1e-6

    # Distance 0 (self): sink 0.10, non-sink 0.00.
    cell_zero = profile[(0, 0)]
    assert abs(wa.mean_of(cell_zero["sink"]) - 0.10) < 1e-6
    assert abs(wa.mean_of(cell_zero["non"]) - 0.00) < 1e-6

    # Distance 2 bin: non-sink back to 0.00, sink still 0.10.
    cell_two = profile[(2, 2)]
    assert abs(wa.mean_of(cell_two["sink"]) - 0.10) < 1e-6
    assert abs(wa.mean_of(cell_two["non"]) - 0.00) < 1e-6


def test_region_values_split_by_group():
    attentions, f0 = _constructed_attention()
    sink_mask = _sink_mask()

    # Continuation-byte queries = positions 3, 4 (distance 1, 2); following = 5.
    cont_sink, cont_non = wa.collect_region_values(attentions, f0, sink_mask, [3, 4])
    foll_sink, foll_non = wa.collect_region_values(attentions, f0, sink_mask, [5])

    # Sink head is flat 0.10 everywhere.
    assert abs(wa.mean_of(cont_sink) - 0.10) < 1e-6
    assert abs(wa.mean_of(foll_sink) - 0.10) < 1e-6
    # Non-sink head: 0.90 at position 3 (distance 1), 0.00 at 4 -> mean 0.45; 0.00 at 5.
    assert abs(wa.mean_of(cont_non) - 0.45) < 1e-6
    assert abs(wa.mean_of(foll_non) - 0.00) < 1e-6


def test_top_reader_is_the_peaked_non_sink_head():
    attentions, f0 = _constructed_attention()
    sink_mask = _sink_mask()
    per_head = wa.collect_head_near_attention(attentions, f0, sink_mask, [3, 4])
    # Only the non-sink head (0, 1) is scored; the sink head is excluded.
    assert (0, 1) in per_head
    assert (0, 0) not in per_head
    # Its near attention over continuation queries 3, 4 is [0.90, 0.00].
    assert abs(wa.mean_of(per_head[(0, 1)]) - 0.45) < 1e-6
