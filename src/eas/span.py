"""Robust probe-span location.

Byte-level BPE makes naive span-finding fragile:

  * Tokenising the span in isolation differs from its tokenisation in context,
    because a leading space is encoded onto the first token (GPT-2/NeoX write it
    as the marker "Ġ"). So "cat" alone is one token, but " cat" inside a sentence
    is a *different* token.
  * Offsets for multi-byte characters (emoji fragment into several byte-tokens)
    are awkward, and — critically — the span's first token often *starts at the
    preceding space*, one character before the span's own text.

Because we CONSTRUCT each stimulus as ``prefix + span + suffix`` we know the
span's character range exactly. The method is therefore:

  1. select every token whose character offset **overlaps** that range
     (overlap, not strict containment, so a token that swallows the leading
     space is still caught);
  2. **validate** by decoding the selected tokens and checking they reconstruct
     the span text (after stripping whitespace).

The decode-validation in step 2 is the real guarantee of correctness; the offset
overlap in step 1 only nominates candidates. Anything that fails validation is
excluded and logged, never silently scored.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple


class SpanNotFound(Exception):
    """Raised when the probe span cannot be cleanly located and validated."""


def _overlaps(tok_start: int, tok_end: int, char_start: int, char_end: int) -> bool:
    """True if token's char range [tok_start, tok_end) intersects the span range."""
    return tok_start < char_end and tok_end > char_start


def locate_span_by_offsets(
    offsets: Sequence[Tuple[int, int]],
    char_start: int,
    char_end: int,
) -> List[int]:
    """Token indices whose char offsets OVERLAP [char_start, char_end).

    Overlap (not strict containment) is deliberate: byte-level BPE attaches a
    leading space to the span's first token, so that token's offset can begin
    one character *before* char_start. Special tokens (offset (0, 0)) are
    skipped. Selection is validated separately by `validate_span`.
    """
    idx = []
    for i, (s, e) in enumerate(offsets):
      if not (s == 0 and e == 0) and _overlaps(s, e, char_start, char_end):
          idx.append(i)
    return idx


def validate_span(
    tokenizer,
    input_ids: Sequence[int],
    token_indices: List[int],
    span_text: str,
) -> None:
    """Assert the selected tokens are contiguous and decode back to span_text.

    Whitespace is stripped on both sides before comparing, which absorbs the
    leading space that byte-level BPE glues onto the first token. A mismatch
    means the span did not align to token boundaries (e.g. a BPE merge pulled in
    a character from the suffix) — raise so the caller excludes the stimulus.
    """
    if len(token_indices) == 0:
        raise SpanNotFound("no tokens overlapped the span's character range")
    expected = list(range(token_indices[0], token_indices[-1] + 1))
    if token_indices != expected:
        raise SpanNotFound(f"span tokens not contiguous: {token_indices}")
    decoded = tokenizer.decode([input_ids[i] for i in token_indices])
    if decoded.strip() != span_text.strip():
        raise SpanNotFound(
            f"decoded span {decoded.strip()!r} != expected {span_text.strip()!r} "
            "— the span did not land on clean token boundaries. Pad the span with "
            "spaces in the carrier template, or choose a different carrier."
        )
