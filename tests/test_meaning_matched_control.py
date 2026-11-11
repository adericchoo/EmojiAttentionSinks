"""Offline test for the meaning-matched control's per-token normalization.

No model is loaded: a synthetic AttentionCapture with uniform causal attention is
built by hand, so the per-token metric can be checked against a closed form. Under
uniform attention, a 2-token span and a 4-token span must yield the SAME per-token
value, because per-token = summed received attention / span length.
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import torch
from eas.metric_reference import received_attention
from eas.types import AttentionCapture, SpanLoc


def _load_script_module():
    path = os.path.join(ROOT, "scripts", "04_meaning_matched_control.py")
    spec = importlib.util.spec_from_file_location("meaning_matched_control", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mmc = _load_script_module()


def _uniform_causal_capture(n=8):
    """attention[q, k] = 1/(q+1) for k <= q, else 0. One layer, one head."""
    att = torch.zeros(1, 1, n, n)
    for q in range(n):
        value = 1.0 / (q + 1)
        for k in range(q + 1):
            att[0, 0, q, k] = value
    input_ids = []
    tokens = []
    offsets = []
    for i in range(n):
        input_ids.append(i)
        tokens.append("x")
        offsets.append((0, 0))
    return AttentionCapture(text="x" * n, input_ids=input_ids, tokens=tokens,
                            offsets=offsets, attentions=att)


def test_per_token_divides_summed_metric_by_span_length():
    cap = _uniform_causal_capture(n=8)
    queries = [5, 6, 7]

    span2 = SpanLoc(2, 4, [2, 3])
    span4 = SpanLoc(1, 5, [1, 2, 3, 4])

    received2 = received_attention(cap, span2, queries)
    received4 = received_attention(cap, span4, queries)

    per_token2 = mmc.per_token_metric(received2, 2)
    per_token4 = mmc.per_token_metric(received4, 4)

    # per-token M is exactly summed M divided by the span-token count
    assert torch.allclose(per_token2, received2 / 2)
    assert torch.allclose(per_token4, received4 / 4)

    # under uniform attention, the 2-token and 4-token spans give the SAME per-token
    expected = (1.0 / 6) + (1.0 / 7) + (1.0 / 8)
    assert torch.allclose(per_token2, torch.full_like(per_token2, expected))
    assert torch.allclose(per_token4, torch.full_like(per_token4, expected))


def test_per_token_shape_is_layers_by_heads():
    cap = _uniform_causal_capture(n=8)
    span = SpanLoc(2, 4, [2, 3])
    received = received_attention(cap, span, [5, 6, 7])
    per_token = mmc.per_token_metric(received, 2)
    assert per_token.shape == received.shape
    assert tuple(per_token.shape) == (1, 1)


def test_per_token_rejects_empty_span():
    raised = False
    try:
        mmc.per_token_metric(torch.tensor([[0.5]]), 0)
    except ValueError:
        raised = True
    assert raised
