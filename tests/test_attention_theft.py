"""Offline test for the attention-theft measurement helpers (leg 4).

No model is loaded. A synthetic AttentionCapture places known attention mass from a
fixed far block onto a slot token and a target token; the measured received
attention must match the constructed values, and the per-distance bucketing must
place each distance in the right bin.
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
from eas.types import AttentionCapture, SpanLoc


def _load_script_module():
    path = os.path.join(ROOT, "scripts", "09_attention_theft.py")
    spec = importlib.util.spec_from_file_location("attention_theft", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


at = _load_script_module()


def _capture_with_masses(n=10, target_index=3, slot_index=5, queries=(7, 8, 9)):
    """One layer, two heads. Each far-block query sends a fixed mass to the target
    (0.1 on head 0, 0.05 on head 1) and to the slot (0.2 on head 0, 0.4 on head 1).
    """
    att = torch.zeros(1, 2, n, n)
    for q in queries:
        att[0, 0, q, target_index] = 0.1
        att[0, 1, q, target_index] = 0.05
        att[0, 0, q, slot_index] = 0.2
        att[0, 1, q, slot_index] = 0.4
    input_ids = []
    tokens = []
    offsets = []
    for i in range(n):
        input_ids.append(i)
        tokens.append("t%d" % i)
        offsets.append((0, 0))
    cap = AttentionCapture(text="", input_ids=input_ids, tokens=tokens,
                           offsets=offsets, attentions=att)
    return cap


def test_received_attention_matches_constructed_masses():
    cap = _capture_with_masses()
    queries = [7, 8, 9]
    target_span = SpanLoc(3, 4, [3])
    slot_span = SpanLoc(5, 6, [5])

    target = at.received_lh(cap, target_span, queries)
    slot = at.received_lh(cap, slot_span, queries)

    # summed over the three queries
    assert abs(float(target[0, 0]) - 0.3) < 1e-6
    assert abs(float(target[0, 1]) - 0.15) < 1e-6
    assert abs(float(slot[0, 0]) - 0.6) < 1e-6
    assert abs(float(slot[0, 1]) - 1.2) < 1e-6

    # all-head mean
    assert abs(at.mean_all(target) - 0.225) < 1e-6
    assert abs(at.mean_all(slot) - 0.9) < 1e-6


def test_mean_over_mask_selects_heads():
    cap = _capture_with_masses()
    slot = at.received_lh(cap, SpanLoc(5, 6, [5]), [7, 8, 9])
    mask = np.zeros((1, 2), dtype=bool)
    mask[0, 1] = True  # only head 1 -> value 1.2
    assert abs(at.mean_over_mask(slot, mask) - 1.2) < 1e-6


def test_distance_bucketing():
    assert at.distance_bucket(1) == (1, 2)
    assert at.distance_bucket(2) == (1, 2)
    assert at.distance_bucket(3) == (3, 4)
    assert at.distance_bucket(6) == (5, 8)
    assert at.distance_bucket(12) == (9, 16)
    assert at.distance_bucket(99) is None
