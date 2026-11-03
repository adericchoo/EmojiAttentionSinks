"""Metric tests on tiny synthetic attention matrices that can be checked by hand.

No model needed.
"""
import numpy as np
import pytest
import torch

from c2.metric import heavy_hitter, m_from_attention, m_from_reduced, ratio, uniform_baseline

# 6 tokens, positions 0..5. Probe span S = {2, 3}. Query block Q = {4, 5}.
# Rows = queries, columns = keys. Causal: row q only has entries for k <= q.
A_EMOJI = np.array([
    #  k=0   k=1   k=2   k=3   k=4   k=5
    [1.00, 0.00, 0.00, 0.00, 0.00, 0.00],   # q=0
    [0.50, 0.50, 0.00, 0.00, 0.00, 0.00],   # q=1
    [0.40, 0.30, 0.30, 0.00, 0.00, 0.00],   # q=2  (inside S: never a query)
    [0.25, 0.25, 0.25, 0.25, 0.00, 0.00],   # q=3  (inside S: never a query)
    [0.40, 0.10, 0.20, 0.10, 0.20, 0.00],   # q=4  gives S 0.20 + 0.10 = 0.30
    [0.30, 0.10, 0.25, 0.15, 0.10, 0.10],   # q=5  gives S 0.25 + 0.15 = 0.40
])
A_CONTROL = np.array([
    [1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.50, 0.50, 0.00, 0.00, 0.00, 0.00],
    [0.40, 0.30, 0.30, 0.00, 0.00, 0.00],
    [0.25, 0.25, 0.25, 0.25, 0.00, 0.00],
    [0.60, 0.10, 0.05, 0.05, 0.20, 0.00],   # q=4  gives S 0.10
    [0.50, 0.10, 0.10, 0.10, 0.10, 0.10],   # q=5  gives S 0.20
])
S = [2, 3]
Q = [4, 5]

# Worked by hand:
#   M_e = (0.30 + 0.40) / 2 = 0.35
#   M_c = (0.10 + 0.20) / 2 = 0.15
#   Δ   = 0.35 - 0.15      = 0.20
#   U   = (2/5 + 2/6) / 2  = 0.366667      (|S|/(q+1) for q = 4, 5)
#   R_e = 0.35 / 0.366667  = 0.954545
M_E_HAND, M_C_HAND, DELTA_HAND = 0.35, 0.15, 0.20
U_HAND = (2 / 5 + 2 / 6) / 2


def _as_LH(a):
    return a[None, None]   # [L=1, H=1, n, n]


def test_synthetic_matrices_are_valid_attention():
    for a in (A_EMOJI, A_CONTROL):
        assert np.allclose(a.sum(-1), 1.0)
        assert np.all(np.triu(a, k=1) == 0)


def test_M_matches_hand_calculation():
    assert m_from_attention(_as_LH(A_EMOJI), S, Q)[0, 0] == pytest.approx(M_E_HAND)
    assert m_from_attention(_as_LH(A_CONTROL), S, Q)[0, 0] == pytest.approx(M_C_HAND)


def test_delta_matches_hand_calculation():
    d = m_from_attention(_as_LH(A_EMOJI), S, Q) - m_from_attention(_as_LH(A_CONTROL), S, Q)
    assert d[0, 0] == pytest.approx(DELTA_HAND)


def test_reduced_path_equals_full_path():
    a_S = _as_LH(A_EMOJI)[..., S].sum(-1)              # what the hook stores: [L, H, n]
    assert m_from_reduced(a_S, S, Q)[0, 0] == pytest.approx(M_E_HAND)


def test_uniform_baseline_and_ratio():
    assert uniform_baseline(2, Q) == pytest.approx(U_HAND)
    assert ratio(np.array([[M_E_HAND]]), 2, Q)[0, 0] == pytest.approx(M_E_HAND / U_HAND)


def test_probe_internal_attention_excluded():
    with pytest.raises(ValueError):
        m_from_attention(_as_LH(A_EMOJI), S, [3, 4, 5])


def test_heavy_hitter_hand_example():
    # keys visible to Q: 0..5. Accumulated over Q (rows 4, 5):
    #   k0 0.70, k1 0.20, k2 0.45, k3 0.25, k4 0.30, k5 0.10.  top ceil(10% of 6) = 1 key -> k0.
    recv = _as_LH(A_EMOJI)[..., Q, :].sum(-2)
    assert not heavy_hitter(recv, S, n_keys=6)[0, 0]
    # with top 30% (2 keys: k0, k2) the probe key k2 is a heavy hitter
    assert heavy_hitter(recv, S, n_keys=6, top_frac=0.3)[0, 0]


def test_agrees_with_c1_reference_metric():
    """eas.metric_reference.received_attention sums over Q; ours averages."""
    from eas.metric_reference import received_attention
    from eas.types import AttentionCapture, SpanLoc
    cap = AttentionCapture(text="", input_ids=list(range(6)), tokens=[""] * 6, offsets=[(0, 0)] * 6,
                           attentions=torch.tensor(_as_LH(A_EMOJI), dtype=torch.float32))
    ref = received_attention(cap, SpanLoc(2, 4, S), Q)[0, 0].item()
    assert ref / len(Q) == pytest.approx(m_from_attention(_as_LH(A_EMOJI), S, Q)[0, 0], abs=1e-6)
