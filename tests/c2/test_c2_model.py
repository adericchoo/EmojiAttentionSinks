"""Sanity checks on real Pythia-410M attention (weights from the local HF cache).

Covers: rows sum to 1, causal zeros, axis order/direction, the streaming hook
reproducing eas.capture_attention exactly, and the hand-checked real example.
Run with `-s` to see the hand-check printout.
"""
import numpy as np
import pytest

from c2.metric import m_from_attention, m_from_reduced
from c2.reduce import SpanReducer
from eas import capture_attention
from eas.validate import causal_mask_holds, rows_sum_to_one

TEXT = "Archive record 2214, field ID:\n\U0001F44D\U0001F3FD | status: closed | owner: records office."


@pytest.fixture(scope="module")
def cap(pythia):
    model, tok = pythia
    return capture_attention(model, tok, TEXT)


def test_rows_sum_to_one(cap):
    ok, (lo, hi) = rows_sum_to_one(cap.attentions, atol=1e-5)
    assert ok, (lo, hi)


def test_future_positions_get_zero_attention(cap):
    ok, leak = causal_mask_holds(cap.attentions)
    assert ok and leak == 0.0


def test_axis_order_and_direction(pythia, cap):
    model, _ = pythia
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    n = cap.n
    # shape is (layers, heads, T, T) = (24, 16, n, n)
    assert (L, H) == (24, 16)
    assert cap.shape == (L, H, n, n)
    att = cap.attentions.numpy()
    # rows are QUERIES: query 0 can only see key 0, so row 0 is [1, 0, 0, ...] in every head.
    # (If the axes were swapped, column 0 would be the one-hot vector instead.)
    assert np.allclose(att[:, :, 0, 0], 1.0) and np.allclose(att[:, :, 0, 1:], 0.0)
    # every row sums to 1 but columns do not (column 0 collects far more than 1)
    assert np.allclose(att.sum(-1), 1.0, atol=1e-5)
    assert att[:, :, :, 0].sum(-1).max() > 1.5
    # the last query row has non-zero mass on the first key (direction: later -> earlier)
    assert (att[:, :, n - 1, 0] > 0).all()


def test_streaming_hook_equals_capture(pythia, cap):
    model, tok = pythia
    ids = cap.input_ids
    S = [i for i, (s, e) in enumerate(cap.offsets) if s < TEXT.index(" |") and e > TEXT.index("\n") + 1]
    Q = list(range(S[-1] + 1, cap.n))
    with SpanReducer(model) as red:
        r = red.run(ids, S, Q)
    att = cap.attentions.numpy()
    assert np.allclose(r["a_S"], att[..., S].sum(-1), atol=1e-6)
    assert np.allclose(r["recv_Q"], att[..., Q, :].sum(-2), atol=1e-6)
    assert np.allclose(m_from_reduced(r["a_S"], S, Q), m_from_attention(att, S, Q), atol=1e-6)
    assert r["rowsum_err"] < 1e-5 and r["causal_leak"] == 0.0


def test_hooks_do_not_change_model_outputs(pythia):
    import torch
    model, tok = pythia
    ids = tok(TEXT, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        ref = model(input_ids=torch.tensor([ids])).logits[0, -1].numpy()
    with SpanReducer(model) as red:
        r = red.run(ids, [5], [6, 7])
    assert np.allclose(ref, r["logits_last"], atol=1e-4)
    # and the hooks are removed afterwards
    from c2.reduce import attention_modules
    assert all(len(m._forward_hooks) == 0 for m in attention_modules(model))


def test_hand_checked_real_example(pythia):
    from c2.handcheck import real_example, real_text
    model, tok = pythia
    r = real_example(model, tok)
    print("\n" + real_text(r, tok))
    assert r["manual"] == pytest.approx(r["code"], abs=1e-6)
    assert r["code_reduced"] == pytest.approx(r["code"], abs=1e-6)
    assert 0.0 <= r["code"] <= 1.0
