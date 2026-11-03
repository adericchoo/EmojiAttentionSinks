"""Extraction-correctness suite. C1 owns 'the attention numbers are right'.

Each check returns (passed: bool, detail) so scripts can print diagnostics.
"""
from __future__ import annotations

from typing import Tuple

import torch

from .types import AttentionCapture


def rows_sum_to_one(att: torch.Tensor, atol: float = 1e-4) -> Tuple[bool, Tuple[float, float]]:
    """Every query row sums to ~1 over its keys (causal: keys k <= q)."""
    sums = att.sum(dim=-1)  # [L, H, n]
    ok = torch.allclose(sums, torch.ones_like(sums), atol=atol)
    return bool(ok), (sums.min().item(), sums.max().item())


def causal_mask_holds(att: torch.Tensor, atol: float = 1e-6) -> Tuple[bool, float]:
    """A[.., q, k] == 0 for all future keys k > q."""
    n = att.shape[-1]
    future = torch.triu(torch.ones(n, n), diagonal=1).bool()  # k > q
    upper = att[..., future]
    max_leak = upper.abs().max().item() if upper.numel() else 0.0
    return max_leak <= atol, max_leak


def deterministic(cap_a: AttentionCapture, cap_b: AttentionCapture) -> Tuple[bool, float]:
    """Same input twice -> identical attention (eval mode, no dropout)."""
    if cap_a.attentions.shape != cap_b.attentions.shape:
        return False, float("inf")
    diff = (cap_a.attentions - cap_b.attentions).abs().max().item()
    return diff == 0.0, diff


def padding_excluded(att: torch.Tensor, n_real: int, atol: float = 1e-6) -> Tuple[bool, float]:
    """For a right-padded sequence, real queries must put ~0 mass on pad keys.

    `n_real` is the count of non-pad tokens; keys [n_real:] are padding.
    """
    if att.shape[-1] <= n_real:
        return True, 0.0
    pad_mass = att[:, :, :n_real, n_real:].abs().max().item()
    return pad_mass <= atol, pad_mass
