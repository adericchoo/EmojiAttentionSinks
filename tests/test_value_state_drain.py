"""Offline test for the value-norm reduction and drain ratio (leg 3c).

No model is loaded and nothing is downloaded. A fake value tensor
[layers, heads, seq, head_dim] with one position zeroed must reduce to ~0 value
norm at that position, and the drain_ratio against a normal position must reflect
the drain (~0).
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
    path = os.path.join(ROOT, "scripts", "08_value_state_drain.py")
    spec = importlib.util.spec_from_file_location("value_state_drain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vsd = _load_script_module()


def _fake_values(n_layers=2, n_heads=3, seq=5, head_dim=4, drained_position=2):
    """Value of 1.0 on every head_dim entry everywhere, except one position that
    is zeroed on every layer and head."""
    values = torch.ones(n_layers, n_heads, seq, head_dim)
    for layer in range(n_layers):
        for head in range(n_heads):
            for d in range(head_dim):
                values[layer, head, drained_position, d] = 0.0
    return values


def test_value_norm_is_zero_at_drained_position():
    head_dim = 4
    values = _fake_values(head_dim=head_dim, drained_position=2)
    drained = vsd.value_norms_at(values, 2)
    normal = vsd.value_norms_at(values, 1)

    # the zeroed position has ~0 norm everywhere
    assert float(drained.max()) < 1e-6
    # a normal position: L2 norm of a length-head_dim vector of ones = sqrt(head_dim)
    expected = head_dim ** 0.5
    assert abs(float(normal[0, 0]) - expected) < 1e-6


def test_drain_ratio_reflects_the_drain():
    values = _fake_values(drained_position=2)
    drained = vsd.value_norms_at(values, 2)
    normal = vsd.value_norms_at(values, 1)

    # sink (drained) over ordinary (normal) -> ~0
    ratio_all = vsd.drain_ratio(drained, normal, None)
    assert ratio_all < 1e-6

    # a non-drained position over itself -> 1
    ratio_self = vsd.drain_ratio(normal, normal, None)
    assert abs(ratio_self - 1.0) < 1e-6


def test_drain_ratio_respects_the_head_mask():
    n_layers = 2
    n_heads = 3
    sink = np.ones((n_layers, n_heads))
    ordinary = np.full((n_layers, n_heads), 2.0)
    # only head (0, 0) selected -> ratio = 1/2
    mask = np.zeros((n_layers, n_heads), dtype=bool)
    mask[0, 0] = True
    ratio = vsd.drain_ratio(sink, ordinary, mask)
    assert abs(ratio - 0.5) < 1e-9
