"""The study metric M, contrasts Δ, and diagnostics (R, heavy-hitter).

Notation. A(l,h)[q,k] = attention FROM query q (row) TO key k (column),
0-indexed positions, causal (k <= q). S = probe span, Q = query block.

  M(l,h) = (1/|Q|) * sum_{q in Q} sum_{k in S} A(l,h)[q,k]          in [0, 1]

i.e. the average share of a later token's attention that lands on the probe.
Probe-internal attention is excluded by construction: Q must not overlap S.

  M_far   uses Q      = the 64-token query block       (primary)
  M_near  uses Q_near = the 16-token buffer after S    (RQ2)

Contrasts per matched set:  Δ_{e,c} = M_e − M_c  for c in {w, s, x}
  (count-matched; confirmatory: Δ_ew, Δ_es; secondary: Δ_ex).
The ordinary-word reference r is NOT count-matched, so it is compared per
token: M/|S|.

Diagnostics only (never headline numbers):
  U = (1/|Q|) sum_{q in Q} |S|/(q+1)   what S would get if every query spread its
                                       attention uniformly over the q+1 keys it can see
  R = M / U
  heavy_hitter = any probe token among the top 10% of keys (k < end of Q) by
                 attention accumulated over the queries in Q.

Everything works on "reduced" per-layer arrays so the full n x n maps never have
to be kept (see reduce.py):
  a_S[l, h, q]   = sum_{k in S} A(l,h)[q,k]     (attention each query gives S)
  recv_Q[l, h, k] = sum_{q in Q} A(l,h)[q,k]    (attention each key receives from Q)
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

HEAVY_TOP_FRAC = 0.10


def _check_disjoint(span: Sequence[int], queries: Sequence[int]) -> None:
    if set(span) & set(queries):
        raise ValueError("query set overlaps the probe span; probe-internal attention must be excluded")


def m_from_reduced(a_S: np.ndarray, span: Sequence[int], queries: Sequence[int]) -> np.ndarray:
    """a_S: [L, H, n] -> M: [L, H]."""
    _check_disjoint(span, queries)
    return a_S[..., list(queries)].mean(axis=-1)


def m_from_attention(att: np.ndarray, span: Sequence[int], queries: Sequence[int]) -> np.ndarray:
    """Same metric from full maps att: [L, H, n, n] (rows = queries, cols = keys)."""
    _check_disjoint(span, queries)
    att = np.asarray(att)
    sub = att[..., list(queries), :][..., list(span)]   # [L, H, |Q|, |S|]
    return sub.sum(axis=-1).mean(axis=-1)


def uniform_baseline(n_span: int, queries: Sequence[int]) -> float:
    """U = (1/|Q|) * sum_{q in Q} |S| / (q + 1), 0-indexed positions."""
    q = np.asarray(list(queries), dtype=np.float64)
    return float(np.mean(n_span / (q + 1.0)))


def ratio(m: np.ndarray, n_span: int, queries: Sequence[int]) -> np.ndarray:
    return m / uniform_baseline(n_span, queries)


def heavy_hitter(recv_Q: np.ndarray, span: Sequence[int], n_keys: int, top_frac: float = HEAVY_TOP_FRAC) -> np.ndarray:
    """recv_Q: [L, H, n]. Keys considered: 0..n_keys-1 (everything Q can see).
    Returns [L, H] bool: is any span key in the top ceil(top_frac * n_keys)?"""
    r = recv_Q[..., :n_keys]
    k = max(1, math.ceil(top_frac * n_keys))
    top = np.argsort(-r, axis=-1, kind="stable")[..., :k]          # [L, H, k]
    return np.isin(top, list(span)).any(axis=-1)


def per_token(m: np.ndarray, n_span: int) -> np.ndarray:
    return m / n_span
