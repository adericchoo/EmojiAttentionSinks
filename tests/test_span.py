"""Offline tests for span location — no model/tokenizer download needed.

We feed synthetic offset maps that mimic what a fast tokenizer returns, so the
selection logic is tested in isolation.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
from eas.span import locate_span_by_offsets, validate_span, SpanNotFound


def test_selects_tokens_inside_char_range():
    # text "A cat sat.": prefix "A " (0..2), span "cat" (2..5), suffix " sat." (5..10)
    offsets = [(0, 1), (1, 2), (2, 5), (5, 9), (9, 10)]  # one token for "cat"
    assert locate_span_by_offsets(offsets, 2, 5) == [2]


def test_multi_token_span():
    # span "cat" fragmented into two tokens "ca" (2..4) and "t" (4..5)
    offsets = [(0, 2), (2, 4), (4, 5), (5, 10)]
    assert locate_span_by_offsets(offsets, 2, 5) == [1, 2]


def test_skips_special_tokens():
    # leading special token with (0, 0) offset must be ignored
    offsets = [(0, 0), (0, 2), (2, 5), (5, 10)]
    assert locate_span_by_offsets(offsets, 2, 5) == [2]


def test_includes_leading_space_token():
    # byte-level BPE glues the preceding space onto the span's first token, so a
    # token (1..3) that straddles the span start MUST be included (overlap match).
    offsets = [(0, 1), (1, 3), (3, 5), (5, 10)]
    assert locate_span_by_offsets(offsets, 2, 5) == [1, 2]


class _FakeTok:
    """Minimal decode stub: maps ids to fixed strings."""
    def __init__(self, mapping):
        self.mapping = mapping

    def decode(self, ids):
        return "".join(self.mapping[i] for i in ids)


def test_validate_span_ok():
    tok = _FakeTok({10: "ca", 11: "t"})
    validate_span(tok, [10, 11], [0, 1], "cat")  # no raise


def test_validate_span_non_contiguous():
    tok = _FakeTok({10: "c", 11: "a", 12: "t"})
    with pytest.raises(SpanNotFound):
        validate_span(tok, [10, 11, 12], [0, 2], "ct")


def test_validate_span_decode_mismatch():
    tok = _FakeTok({10: "do", 11: "g"})
    with pytest.raises(SpanNotFound):
        validate_span(tok, [10, 11], [0, 1], "cat")
