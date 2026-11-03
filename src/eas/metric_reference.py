"""REFERENCE metric only. Do not build the study's scoring on top of
this; it lives here purely as the shared reference for the interface.
"""
from __future__ import annotations

from typing import Sequence

import torch

from .types import AttentionCapture, SpanLoc


def received_attention(
    cap: AttentionCapture,
    span: SpanLoc,
    query_indices: Sequence[int],
) -> torch.Tensor:
    """M[l, h] = sum over q in Q, k in S of attention[l, h, q, k].

    Probe-internal attention (q in S) must NOT be in `query_indices` for the
    primary outcome — pass a Q disjoint from S (e.g. cap.later_queries(span)).
    Returns a [L, H] tensor in attention-probability units.
    """
    q = list(query_indices)
    s = span.token_indices
    if set(q) & set(s):
        raise ValueError(
            "query set overlaps the span; exclude probe-internal attention from "
            "the primary metric."
        )
    sub = cap.attentions[:, :, q][:, :, :, s]  # [L, H, |Q|, |S|]
    return sub.sum(dim=(-1, -2))  # [L, H]
