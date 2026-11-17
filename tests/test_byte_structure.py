"""Offline test for the byte-structure leg's lead-byte tagging and binning.

No model is loaded and nothing is downloaded. A fake tokenizer maps token ids to
raw bytes (encoded as byte-level BPE pieces, the same form the real tokenizer
emits), and a synthetic AttentionCapture carries uniform causal attention. A span
holding a known F0 9F-lead token and an E2-lead token must be tagged and binned to
the correct lead byte-token.
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Insert scripts first, then src, so src/ is at the FRONT: scripts/c2.py would
# otherwise shadow the src/c2 package (the numbered scripts avoid this because
# _bootstrap puts src/ at the front before c2 is first imported).
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch
from c2.tokaudit import _byte_decoder
from eas.types import AttentionCapture, SpanLoc


def _load_script_module():
    path = os.path.join(ROOT, "scripts", "06_byte_structure.py")
    spec = importlib.util.spec_from_file_location("byte_structure", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bs = _load_script_module()


class FakeTokenizer:
    """Minimal stand-in: ids carry raw bytes, exposed as byte-level BPE pieces.

    `convert_ids_to_tokens` returns the GPT-2 byte-level encoding of the id's raw
    bytes, which is exactly what c2.tokaudit.token_bytes decodes back. `decode`
    turns ids straight into text from the same raw bytes.
    """

    def __init__(self, raw_by_id):
        self._raw = raw_by_id
        self._to_piece = {}
        for piece_char, byte_value in _byte_decoder().items():
            self._to_piece[byte_value] = piece_char

    def convert_ids_to_tokens(self, token_id):
        piece = ""
        for byte_value in self._raw[token_id]:
            piece = piece + self._to_piece[byte_value]
        return piece

    def decode(self, ids):
        raw = bytearray()
        for token_id in ids:
            raw.extend(self._raw[token_id])
        return bytes(raw).decode("utf-8", errors="replace")


def _uniform_capture(input_ids):
    """attention[q, k] = 1/(q+1) for k <= q, else 0. Two layers, two heads."""
    n = len(input_ids)
    att = torch.zeros(2, 2, n, n)
    for q in range(n):
        value = 1.0 / (q + 1)
        for k in range(q + 1):
            att[:, :, q, k] = value
    tokens = []
    offsets = []
    for i in range(n):
        tokens.append("t%d" % i)
        offsets.append((0, 0))
    return AttentionCapture(text="", input_ids=list(input_ids), tokens=tokens,
                            offsets=offsets, attentions=att)


def test_lead_byte_tag_from_raw_bytes():
    assert bs.lead_byte_tag(bytes([0xF0, 0x9F, 0x98, 0x80])) == "F0 9F"
    assert bs.lead_byte_tag(bytes([0xE2, 0x98, 0x80])) == "E2"
    # F0 not followed by 9F is not the supplementary-plane lead
    assert bs.lead_byte_tag(bytes([0xF0, 0x80])) == "other"
    assert bs.lead_byte_tag(b"rocket") == "other"


def test_span_fragments_tagged_and_binned_by_lead_byte_token():
    raw_by_id = {
        10: b"\n",
        101: bytes([0xF0, 0x9F, 0x98, 0x80]),   # grinning face -> F0 9F lead
        102: bytes([0xE2, 0x98, 0x80]),          # sun -> E2 lead
        20: b" the",
        21: b" far",
        22: b" end",
    }
    tok = FakeTokenizer(raw_by_id)
    input_ids = [10, 101, 102, 20, 21, 22]
    cap = _uniform_capture(input_ids)
    span = SpanLoc(1, 3, [1, 2])
    queries = [4, 5]

    records = bs.score_fragments(cap, tok, span, queries)
    tags = []
    for record in records:
        tags.append(record["lead_tag"])
    assert tags == ["F0 9F", "E2"]
    assert records[0]["raw_bytes"] == "F0 9F 98 80"
    assert records[1]["raw_bytes"] == "E2 98 80"

    # Flag sinks by hand (no model baseline here) so binning can be checked.
    records[0]["is_sink"] = True
    records[1]["is_sink"] = False
    stimulus = {"records": records}
    bins = bs.bin_by_lead_token([stimulus])

    assert bins["F0 9F"]["n"] == 1
    assert bins["F0 9F"]["n_sink"] == 1
    assert bins["E2"]["n"] == 1
    assert bins["E2"]["n_sink"] == 0
    assert bins["other"]["n"] == 0
