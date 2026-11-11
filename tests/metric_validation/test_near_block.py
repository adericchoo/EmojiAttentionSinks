"""Tests for check 2 (near buffer vs far block contrasts).

The arithmetic tests need only numpy and reuse the pipeline metric. The model
test is skipped unless weights and probes are present.
"""
import numpy as np
import pytest

from c2.metric import m_from_reduced
from metric_validation import common, near_block


def test_contrasts_for_set_builds_expected_deltas():
    shape = (2, 2)
    m_far = {
        "e": np.full(shape, 0.50),
        "s": np.full(shape, 0.30),
        "w": np.full(shape, 0.20),
        "x": np.full(shape, 0.10),
        "p": np.full(shape, 0.40),
        "r": np.full(shape, 0.15),
    }
    m_near = {k: v + 0.05 for k, v in m_far.items()}
    far_out, near_out = near_block.contrasts_for_set(m_far, m_near)
    assert set(far_out) == {"d_es", "d_ew", "d_ex", "d_pr"}
    assert np.allclose(far_out["d_es"], 0.20)
    assert np.allclose(far_out["d_pr"], 0.25)
    # subtracting a constant offset from both sides leaves the delta unchanged
    assert np.allclose(near_out["d_pr"], 0.25)


def test_contrasts_skip_when_a_condition_is_missing():
    m = {"e": np.zeros((1, 1)), "s": np.zeros((1, 1))}   # no w, x, p, r
    far_out, near_out = near_block.contrasts_for_set(m, m)
    assert set(far_out) == {"d_es"}
    assert near_out.keys() == far_out.keys()


def test_near_and_far_select_different_query_sets():
    # a_S[l, h, q] = attention query q gives to the slot. Positions 0..39.
    # near buffer = {24..27}, far block = {28..31}; distinct, known means.
    a_S = np.zeros((1, 1, 40))
    for q in range(24, 28):
        a_S[0, 0, q] = 0.10        # near buffer queries
    for q in range(28, 32):
        a_S[0, 0, q] = 0.40        # far block queries
    slot = [20, 21]
    buffer = [24, 25, 26, 27]
    far_q = [28, 29, 30, 31]
    m_near = m_from_reduced(a_S, slot, buffer)
    m_far = m_from_reduced(a_S, slot, far_q)
    assert m_near[0, 0] == pytest.approx(0.10)
    assert m_far[0, 0] == pytest.approx(0.40)


def test_aggregate_contrasts_and_share():
    per_set = [
        {"d_pr": np.array([[0.1, -0.2]])},
        {"d_pr": np.array([[0.3, 0.4]])},
    ]
    agg = common.aggregate_contrasts(per_set)
    assert agg["d_pr"].shape == (2, 1, 2)
    mean = agg["d_pr"].mean(axis=0)
    assert np.allclose(mean, [[0.2, 0.1]])
    assert common.share_positive(mean) == pytest.approx(1.0)


@pytest.mark.slow
def test_near_block_runs_on_pythia(pythia_ready, tmp_path, monkeypatch):
    monkeypatch.setattr(common, "OUT_DIR", str(tmp_path))
    results = near_block.main("pythia", split="dev", device="cpu", limit=2)
    assert "d_pr" in results["contrasts"]
    pr = results["contrasts"]["d_pr"]
    assert 0.0 <= pr["share_far"] <= 1.0
    assert 0.0 <= pr["share_near"] <= 1.0
    assert results["max_causal_leak"] < 1e-4
