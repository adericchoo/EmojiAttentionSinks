"""Tests for check 1 (first-token sink).

The arithmetic tests need only numpy and reuse the pipeline metric. The model
test is skipped unless weights and probes are present.
"""
import numpy as np
import pytest

from c2.metric import m_from_reduced
from metric_validation import common, first_token


def test_reference_probes_selects_only_r():
    probes = [
        {"condition": "e", "id": "1"},
        {"condition": "r", "id": "2"},
        {"condition": "p", "id": "3"},
        {"condition": "r", "id": "4"},
    ]
    refs = first_token.reference_probes(probes)
    assert [p["id"] for p in refs] == ["2", "4"]


def test_metric_function_is_reused_for_first_token():
    # a_S[l, h, q] = attention query q gives to the reduced span.
    # For span {0}, a_S holds A[q, 0]. One layer, one head, 6 positions.
    a_S0 = np.zeros((1, 1, 6))
    a_S0[0, 0, 4] = 0.70   # query 4 sends 0.70 to position 0
    a_S0[0, 0, 5] = 0.50   # query 5 sends 0.50 to position 0
    far_q = [4, 5]
    m_first = m_from_reduced(a_S0, [0], far_q)
    assert m_first[0, 0] == pytest.approx((0.70 + 0.50) / 2)


def test_verdict_detects_a_strong_sink():
    # Sink mass far above the reference on every head -> share 1.0, pass.
    mean_first = np.full((24, 16), 0.40)
    mean_ref = np.full((24, 16), 0.02)
    share, passed = first_token.verdict(mean_first, mean_ref, common.PASS_BAR)
    assert share == pytest.approx(1.0)
    assert passed is True


def test_verdict_fails_when_metric_is_blind():
    # Reference wins slightly more often than the sink -> share below the bar.
    rng = np.random.default_rng(0)
    mean_first = rng.random((24, 16))
    mean_ref = mean_first + 0.01   # reference always higher
    share, passed = first_token.verdict(mean_first, mean_ref, common.PASS_BAR)
    assert share == pytest.approx(0.0)
    assert passed is False


def test_share_positive_counts_heads():
    diff = np.array([[1.0, -1.0], [2.0, 3.0]])
    assert common.share_positive(diff) == pytest.approx(0.75)


@pytest.mark.slow
def test_first_token_runs_on_pythia(pythia_ready, tmp_path, monkeypatch):
    monkeypatch.setattr(common, "OUT_DIR", str(tmp_path))
    results = first_token.main("pythia", split="dev", device="cpu", limit=2)
    assert results["n_reference_probes"] == 2
    # The first token is the canonical sink: it should dominate an ordinary word.
    assert results["aggregate_M_first"] > results["aggregate_M_ref"]
    assert results["max_causal_leak"] < 1e-4
