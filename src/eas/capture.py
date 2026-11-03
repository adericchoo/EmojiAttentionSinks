"""Attention capture

`capture_attention` runs one forward pass and returns an `AttentionCapture`.
`build_stimulus` + `locate` give you a validated `SpanLoc` for the probe span.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .span import SpanNotFound, locate_span_by_offsets, validate_span
from .types import AttentionCapture, SpanLoc


@dataclass
class Stimulus:
    """A probe built as prefix + span + suffix, with known char offsets."""
    prefix: str
    span_text: str
    suffix: str

    @property
    def text(self) -> str:
        return f"{self.prefix}{self.span_text}{self.suffix}"

    @property
    def char_start(self) -> int:
        return len(self.prefix)

    @property
    def char_end(self) -> int:
        return len(self.prefix) + len(self.span_text)


def build_stimulus(prefix: str, span_text: str, suffix: str) -> Stimulus:
    """Construct a stimulus. Pad the span with spaces in prefix/suffix to reduce
    BPE merges across the boundary, e.g. build_stimulus("I saw a ", "cat", " today")."""
    return Stimulus(prefix=prefix, span_text=span_text, suffix=suffix)


def capture_attention(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    device: Optional[str] = None,
    max_tokens: int = 256,
) -> AttentionCapture:
    """One forward pass -> attentions [L, H, n, n] (CPU, float32) + token info."""
    enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
    n = enc["input_ids"].shape[1]
    if n > max_tokens:
        raise ValueError(
            f"sequence length {n} exceeds max_tokens={max_tokens}; captured "
            f"attention is O(L*H*n^2). Shorten the carrier or raise max_tokens "
            f"deliberately."
        )
    if device:
        enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc, output_attentions=True, use_cache=False)

    if out.attentions is None:
        raise RuntimeError(
            "model returned attentions=None — load it with "
            "attn_implementation='eager' (see eas.models.load_model)."
        )

    # stack tuple(L) of [1, H, n, n] -> [L, H, n, n], drop batch, to CPU fp32
    att = torch.stack([a[0] for a in out.attentions]).to("cpu", torch.float32)

    ids = enc["input_ids"][0].tolist()
    toks = tokenizer.convert_ids_to_tokens(ids)
    return AttentionCapture(
        text=text, input_ids=ids, tokens=toks, offsets=offsets, attentions=att
    )


def locate(tokenizer, cap: AttentionCapture, stim: Stimulus) -> SpanLoc:
    """Validated SpanLoc for `stim`'s span inside an already-captured sequence.

    Raises SpanNotFound (exclude + log the stimulus) on any mismatch.
    """
    idx = locate_span_by_offsets(cap.offsets, stim.char_start, stim.char_end)
    validate_span(tokenizer, cap.input_ids, idx, stim.span_text)
    return SpanLoc(start=idx[0], end=idx[-1] + 1, token_indices=idx)
