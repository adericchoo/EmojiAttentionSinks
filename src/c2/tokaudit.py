"""Tokenisation audit: how many model tokens does a string become *in context*?

Counting rule (used everywhere in C2, including the probe generator):
  * the filler is placed in a carrier, the whole text is tokenised once with
    ``add_special_tokens=False``;
  * span tokens = tokens whose character offsets OVERLAP the filler's character
    range (eas.span.locate_span_by_offsets);
  * boundary-crossing = a span token whose offsets also cover a NON-whitespace
    character outside the filler. The leading space that byte-level BPE glues
    onto the first token ("Ġ") is whitespace, so it does not count;
  * a token is a "partial UTF-8 byte fragment" if decoding it alone yields
    U+FFFD; otherwise it is "whole character(s)".
"""
from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

from eas.span import locate_span_by_offsets

AUDIT_CARRIER = "The note contains the marker [PROBE]. The remaining text discusses a routine meeting."
# Same sentence with the slot at the start of a line, which is how every probe
# carrier places it (see carriers.py). A newline is always its own token, so the
# filler tokenises exactly as it would alone ("isolated"), with no leading space.
LINE_CARRIER = "The note contains the marker:\n[PROBE] and the remaining text discusses a routine meeting."
POOL_RANGE = (2, 6)  # candidate pool: 2-6 tokens in BOTH models


@dataclass
class InContext:
    text: str
    char_span: Tuple[int, int]
    input_ids: List[int]
    offsets: List[Tuple[int, int]]
    span_idx: List[int]          # token indices overlapping the filler
    boundary: bool               # a span token covers non-whitespace outside the filler
    byte_frag: List[bool]        # per span token
    pieces: List[str]            # readable per span token

    @property
    def n_span(self) -> int:
        return len(self.span_idx)

    @property
    def n_frag(self) -> int:
        return sum(self.byte_frag)

    @property
    def span_ids(self) -> List[int]:
        return [self.input_ids[i] for i in self.span_idx]


@lru_cache(maxsize=1)
def _byte_decoder() -> Dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode (Pythia and Qwen both use it)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_bytes(tok, tid: int) -> bytes:
    piece = tok.convert_ids_to_tokens(tid)
    dec = _byte_decoder()
    try:
        return bytes(dec[ch] for ch in piece)
    except KeyError:  # not a byte-level piece (should not happen for our two models)
        return piece.encode("utf-8")


def is_byte_fragment(tok, tid: int) -> bool:
    return "�" in tok.decode([tid])


def readable_piece(tok, tid: int) -> str:
    """'␣cat' for whole characters, '‹F0 9F›' for raw bytes, mixed when both."""
    raw = token_bytes(tok, tid)
    out, i = "", 0
    while i < len(raw):
        # take the longest valid UTF-8 prefix starting at i
        for j in range(min(len(raw), i + 4), i, -1):
            try:
                ch = raw[i:j].decode("utf-8")
            except UnicodeDecodeError:
                continue
            out += ch.replace(" ", "␣")
            i = j
            break
        else:
            out += f"‹{raw[i]:02X}›"
            i += 1
    return out.replace("›‹", " ")


def tokenize_in_context(tok, before: str, filler: str, after: str) -> InContext:
    text = before + filler + after
    cs, ce = len(before), len(before) + len(filler)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    idx = locate_span_by_offsets(offsets, cs, ce)
    boundary = False
    for i in idx:
        s, e = offsets[i]
        outside = (text[s:cs] if s < cs else "") + (text[ce:e] if e > ce else "")
        if outside.strip():
            boundary = True
    return InContext(
        text=text, char_span=(cs, ce), input_ids=ids, offsets=offsets, span_idx=idx,
        boundary=boundary,
        byte_frag=[is_byte_fragment(tok, ids[i]) for i in idx],
        pieces=[readable_piece(tok, ids[i]) for i in idx],
    )


def audit_string(tok, s: str, carrier: str = AUDIT_CARRIER) -> InContext:
    before, after = carrier.split("[PROBE]")
    return tokenize_in_context(tok, before, s, after)


def audit_emoji(tok, model_name: str, emoji: Sequence) -> List[dict]:
    """One row per emoji. Primary columns use the protocol's AUDIT_CARRIER;
    the *_line columns use LINE_CARRIER (the probe-slot context)."""
    rows = []
    for e in emoji:
        r = audit_string(tok, e.text)
        r_line = audit_string(tok, e.text, LINE_CARRIER)
        rows.append({
            "model": model_name,
            "emoji": e.text,
            "name": e.name,
            "codepoints": e.codepoints_str,
            "n_codepoints": len(e.codepoints),
            "utf8_bytes": e.utf8_bytes,
            "seq_type": e.seq_type,
            "n_tokens": r.n_span,
            "n_byte_fragments": r.n_frag,
            "n_whole_char_tokens": r.n_span - r.n_frag,
            "token_pieces": " | ".join(r.pieces),
            "token_ids": " ".join(str(t) for t in r.span_ids),
            "boundary_crossing": r.boundary,
            "n_tokens_line": r_line.n_span,
            "n_byte_fragments_line": r_line.n_frag,
            "token_pieces_line": " | ".join(r_line.pieces),
            "boundary_crossing_line": r_line.boundary,
            "n_tokens_isolated": len(tok(e.text, add_special_tokens=False)["input_ids"]),
            "group": e.group,
            "subgroup": e.subgroup,
        })
    return rows


def summarise(rows: List[dict], suffix: str = "", exclude=()) -> dict:
    """Audit statistics. suffix="" -> audit carrier, "_line" -> probe-slot context.

    `exclude` drops emoji by text on top of boundary crossings (used only to
    show the effect of the proposal's #/* keycap exclusion).
    """
    def is_out(r):
        return r["boundary_crossing" + suffix] or r["emoji"] in exclude

    excluded = [r for r in rows if is_out(r)]
    kept = [r for r in rows if not is_out(r)]
    counts = [r["n_tokens" + suffix] for r in kept]
    multi = sum(c > 1 for c in counts)
    return {
        "n_emoji": len(rows),
        "n_boundary": len(excluded),
        "boundary_emoji": [f"{r['emoji']} ({r['name']})" for r in excluded],
        "n_kept": len(kept),
        "n_multi": multi,
        "pct_multi": 100.0 * multi / len(kept),
        "median": statistics.median(counts),
        "max": max(counts),
        "n_any_byte_fragment": sum(r["n_byte_fragments" + suffix] > 0 for r in kept),
        "counts": counts,
    }


def candidate_pool(rows_by_model: Dict[str, List[dict]], suffix: str = "_line",
                   lo: int = POOL_RANGE[0], hi: int = POOL_RANGE[1]) -> List[str]:
    """Emoji with lo..hi tokens (and no boundary crossing) in EVERY model, file order.

    Defaults to the probe-slot context ("_line"), because that is the
    tokenisation the probes actually see.
    """
    ok_sets = []
    for rows in rows_by_model.values():
        ok_sets.append({r["emoji"] for r in rows
                        if not r["boundary_crossing" + suffix] and lo <= r["n_tokens" + suffix] <= hi})
    first = next(iter(rows_by_model.values()))
    return [r["emoji"] for r in first if all(r["emoji"] in s for s in ok_sets)]


def write_csv(rows: List[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("n_codepoints", "utf8_bytes", "n_tokens", "n_byte_fragments", "n_whole_char_tokens",
                  "n_tokens_line", "n_byte_fragments_line", "n_tokens_isolated"):
            r[k] = int(r[k])
        for k in ("boundary_crossing", "boundary_crossing_line"):
            r[k] = r[k] == "True"
    return rows
