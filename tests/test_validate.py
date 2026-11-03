"""Offline tests for the validation suite, using a synthetic causal attention
tensor (no model needed)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
from eas.validate import rows_sum_to_one, causal_mask_holds, padding_excluded


def _causal_attention(L=2, H=3, n=5):
    """Build a valid lower-triangular, row-normalised attention tensor."""
    raw = torch.rand(L, H, n, n)
    mask = torch.tril(torch.ones(n, n))          # keep k <= q
    raw = raw * mask
    raw = raw / raw.sum(dim=-1, keepdim=True)     # rows sum to 1
    return raw


def test_rows_sum_to_one_passes():
    att = _causal_attention()
    ok, (lo, hi) = rows_sum_to_one(att)
    assert ok, (lo, hi)


def test_causal_mask_holds_passes():
    att = _causal_attention()
    ok, leak = causal_mask_holds(att)
    assert ok and leak == 0.0


def test_causal_mask_detects_leak():
    att = _causal_attention()
    att[0, 0, 0, 4] = 0.5  # put mass on a future key
    ok, leak = causal_mask_holds(att)
    assert not ok and leak == 0.5


def test_padding_excluded_passes():
    # 5 positions, last 2 are padding; real queries put no mass there by construction
    att = _causal_attention(n=5)
    att[:, :, :3, 3:] = 0.0
    att[:, :, :3, :3] = att[:, :, :3, :3] / att[:, :, :3, :3].sum(-1, keepdim=True)
    ok, mass = padding_excluded(att, n_real=3)
    assert ok, mass
