"""Offline test for the narrow-sink classifier (leg 3b).

No model is loaded and nothing is downloaded. A synthetic AttentionCapture carries
hand-built attention so that three keys reduce (via script 05's fragment_received,
the same reduction the script uses) to known [L, H] vectors:

  - a baseline key with a small, flat value on every head,
  - a NARROW key: one extreme head, near-zero everywhere else,
  - a BROAD key: a little above the baseline on every head.

The narrow key must classify NARROW (not BROAD); the broad key must classify BROAD.
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

import torch
from eas.types import AttentionCapture, SpanLoc


def _load_script_module():
    path = os.path.join(ROOT, "scripts", "07_narrow_sink.py")
    spec = importlib.util.spec_from_file_location("narrow_sink", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ns = _load_script_module()


def _capture_with_keys(n=6, n_layers=2, n_heads=3):
    """Build attention so a fixed far block [4, 5] reduces each key to a known
    [L, H] vector. Key 1 = flat baseline, key 2 = one extreme head, key 3 = a
    uniform small bump above baseline.
    """
    att = torch.zeros(n_layers, n_heads, n, n)
    queries = [4, 5]
    for q in queries:
        # baseline key: 0.1 on every head -> summed over 2 queries = 0.2 each
        for layer in range(n_layers):
            for head in range(n_heads):
                att[layer, head, q, 1] = 0.1
        # narrow key: a single extreme head (0, 0); zero elsewhere
        att[0, 0, q, 2] = 5.0
        # broad key: 0.2 on every head -> 0.4 each, above the 0.2 baseline
        for layer in range(n_layers):
            for head in range(n_heads):
                att[layer, head, q, 3] = 0.2
    input_ids = []
    tokens = []
    offsets = []
    for i in range(n):
        input_ids.append(i)
        tokens.append("t%d" % i)
        offsets.append((0, 0))
    cap = AttentionCapture(text="", input_ids=input_ids, tokens=tokens,
                           offsets=offsets, attentions=att)
    return cap, queries


def test_narrow_key_is_narrow_not_broad():
    cap, queries = _capture_with_keys()
    baseline_lh = ns.frag.fragment_received(cap.attentions, 1, queries).numpy()
    narrow_lh = ns.frag.fragment_received(cap.attentions, 2, queries).numpy()

    metrics = ns.condition_metrics(narrow_lh, baseline_lh, k=ns.EXTREME_K)

    # one head carries a huge peak (10.0) over a 0.2 baseline peak -> peak_ratio 50
    assert metrics["peak_ratio"] >= ns.PEAK_RATIO_SINK_BAR
    # only that single head beats the baseline, so breadth is well under the bar
    assert metrics["broad_share"] < ns.BROAD_SHARE_BAR
    assert metrics["extreme_count"] >= ns.EXTREME_MIN_HEADS
    assert metrics["cls"] == ns.CLASS_NARROW
    assert metrics["cls"] != ns.CLASS_BROAD


def test_broad_key_is_broad():
    cap, queries = _capture_with_keys()
    baseline_lh = ns.frag.fragment_received(cap.attentions, 1, queries).numpy()
    broad_lh = ns.frag.fragment_received(cap.attentions, 3, queries).numpy()

    metrics = ns.condition_metrics(broad_lh, baseline_lh, k=ns.EXTREME_K)

    # above the baseline on every head -> breadth over the bar -> BROAD
    assert metrics["broad_share"] >= ns.BROAD_SHARE_BAR
    assert metrics["cls"] == ns.CLASS_BROAD


def test_classify_bars_directly():
    # peak but no extreme head and not broad -> NON-sink
    assert ns.classify(5.0, 0.1, 0) == ns.CLASS_NON
    # no meaningful peak -> NON-sink regardless of a stray extreme head
    assert ns.classify(1.5, 0.1, 2) == ns.CLASS_NON
    # real peak, localised by an extreme head, not broad -> NARROW
    assert ns.classify(5.0, 0.1, 1) == ns.CLASS_NARROW
    # broad share over the bar -> BROAD (takes precedence)
    assert ns.classify(5.0, 0.95, 1) == ns.CLASS_BROAD
