"""The data contract between capturing and metric.

`AttentionCapture` is what `capture_attention` returns
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class SpanLoc:
    """Token-index location of a probe span S inside a captured sequence."""
    start: int               # first token index (inclusive)
    end: int                 # last token index (exclusive)
    token_indices: List[int] # explicit list == list(range(start, end))

    def __post_init__(self) -> None:
        assert self.token_indices == list(range(self.start, self.end)), (
            "span token indices must be contiguous"
        )

    def __len__(self) -> int:
        return self.end - self.start


@dataclass
class AttentionCapture:
    """One forward pass, with everything C2 needs to compute received attention.

    attentions: tensor [L, H, n, n] on CPU, float32.
        attentions[l, h, q, k] = attention weight from query position q to key k.
    offsets: per-token (char_start, char_end) from the fast tokenizer; special
        tokens carry (0, 0).
    """
    text: str
    input_ids: List[int]
    tokens: List[str]
    offsets: List[Tuple[int, int]]
    attentions: torch.Tensor

    @property
    def n(self) -> int:
        return len(self.input_ids)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return tuple(self.attentions.shape)  # (L, H, n, n)

    def later_queries(self, span: SpanLoc) -> List[int]:
        """Default query set Q: all positions strictly after the span end.

        Disjoint from S by construction. R3/C2 may define a narrower Q (e.g.
        near-vs-far); this is the sensible default.
        """
        return list(range(span.end, self.n))
