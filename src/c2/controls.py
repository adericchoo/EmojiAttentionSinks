"""Slot fillers for every condition, and the per-model screen that checks them.

Conditions (one-letter codes used everywhere):
  e  emoji                 test condition
  w  English word          lexical control, naturally split by the tokenizer
  s  rare characters       MAIN non-lexical control: byte fragments like emoji
  x  hex string            SECONDARY non-lexical control (e.g. "3f9a")
  p  punctuation           positive control (. , ! ;)
  r  ordinary word         single-token reference (e.g. "cat")

e, w, s, x are matched on contextual token count (probes.py). This module only
produces candidates and screens each one in the probe-slot context, per model:
  w, s, x  need >= 2 tokens and no boundary crossing
  s        additionally needs >= 1 partial UTF-8 byte-fragment token
  p, r     need exactly 1 token
"""
from __future__ import annotations

import os
import random
import string
import unicodedata
from typing import Dict, List

from . import DATA_DIR
from .tokaudit import LINE_CARRIER, audit_string

CONDITIONS = ["e", "w", "s", "x", "p", "r"]
MATCHED = ["w", "s", "x"]  # count-matched to e
COND_NAMES = {
    "e": "emoji", "w": "English word (split)", "s": "rare chars (byte fragments)",
    "x": "hex string", "p": "punctuation", "r": "ordinary word",
}

WORDS_PATH = os.path.join(DATA_DIR, "c2", "english_words.txt")
SYSTEM_DICT = "/usr/share/dict/words"

PUNCT = [".", ",", "!", ";"]
REF_WORDS = ["cat", "dog", "book", "table", "chair", "river", "apple", "house", "tree", "door",
             "car", "bird", "cup", "pen", "road", "box"]

# Rare scripts. Each filler is 1-4 characters from one block, plus mixed
# 2-3 character fillers across blocks. The 2-byte scripts are needed because in
# Pythia every 3/4-byte rare character is exactly 3/4 byte tokens, so without
# them no rare-character filler could reach 2 or 5 tokens.
RARE_BLOCKS = {
    "nko": (0x07C0, 0x07EA),            # 2-byte UTF-8
    "thaana": (0x0780, 0x07A5),         # 2-byte
    "syriac": (0x0710, 0x072F),         # 2-byte
    "cjk_ext_a": (0x3400, 0x4DBF),      # 3-byte
    "cjk_ext_b": (0x20000, 0x2A6DF),    # 4-byte
    "runic": (0x16A0, 0x16EA),          # 3-byte
    "ogham": (0x1681, 0x169A),          # 3-byte
    "linear_b": (0x10000, 0x1005D),     # 4-byte
    "gothic": (0x10330, 0x1034A),       # 4-byte
    "old_italic": (0x10300, 0x1031F),   # 4-byte
    "cuneiform": (0x12000, 0x12399),    # 4-byte
    "egyptian": (0x13000, 0x1342E),     # 4-byte
}


def build_word_list(n: int = 4000, seed: int = 0) -> None:
    """One-off: derive data/c2/english_words.txt from the system dictionary.

    Webster's 2nd International (public domain, ships as /usr/share/dict/words on
    macOS/BSD). Lowercase alphabetic words of 5-14 letters, seeded sample, so
    the committed list is reproducible and does not depend on the OS.
    """
    with open(SYSTEM_DICT, encoding="utf-8") as f:
        words = sorted({w.strip() for w in f if w.strip().isalpha() and w.strip().islower()
                        and 5 <= len(w.strip()) <= 14 and w.strip().isascii()})
    sample = sorted(random.Random(seed).sample(words, n))
    os.makedirs(os.path.dirname(WORDS_PATH), exist_ok=True)
    with open(WORDS_PATH, "w", encoding="utf-8") as f:
        f.write(f"# {n} words sampled (seed={seed}) from Webster's 2nd International ({SYSTEM_DICT}),\n")
        f.write("# lowercase alphabetic, 5-14 letters. Built by c2.controls.build_word_list.\n")
        f.write("\n".join(sample) + "\n")


def word_candidates() -> List[str]:
    with open(WORDS_PATH, encoding="utf-8") as f:
        return [w.strip() for w in f if w.strip() and not w.startswith("#")]


def _assigned(cp: int) -> bool:
    return unicodedata.category(chr(cp)) not in ("Cn", "Co", "Cs")


def rare_candidates(per_len: int = 30, max_len: int = 4, seed: int = 0) -> List[str]:
    rng = random.Random(seed)
    out = []
    blocks = {}
    for name, (lo, hi) in RARE_BLOCKS.items():
        # letters/symbols only: combining marks would attach to the neighbouring character
        blocks[name] = [c for c in range(lo, hi + 1)
                        if _assigned(c) and not unicodedata.category(chr(c)).startswith("M")]
    for cps in blocks.values():
        for k in range(1, max_len + 1):
            for _ in range(per_len):
                out.append("".join(chr(rng.choice(cps)) for _ in range(k)))
    names = sorted(blocks)
    for k in (2, 3):
        for _ in range(per_len * 4):
            out.append("".join(chr(rng.choice(blocks[rng.choice(names)])) for _ in range(k)))
    return list(dict.fromkeys(out))  # dedupe, keep order


def hex_candidates(per_len: int = 40, lengths=range(2, 17), seed: int = 0) -> List[str]:
    """Random lowercase hex strings with at least one digit AND one letter, so
    none is an English word ("cafe", "bead") or a plain number."""
    rng = random.Random(seed)
    out = []
    for n in lengths:
        made = 0
        while made < per_len:
            s = "".join(rng.choice("0123456789abcdef") for _ in range(n))
            if any(c in string.digits for c in s) and any(c in "abcdef" for c in s):
                out.append(s)
                made += 1
    return list(dict.fromkeys(out))


def candidates() -> Dict[str, List[str]]:
    return {"w": word_candidates(), "s": rare_candidates(), "x": hex_candidates(),
            "p": list(PUNCT), "r": list(REF_WORDS)}


def screen_one(tok, cond: str, filler: str) -> dict:
    r = audit_string(tok, filler, LINE_CARRIER)
    reason = ""
    if r.boundary:
        reason = "boundary_crossing"
    elif cond in ("p", "r") and r.n_span != 1:
        reason = f"not_single_token ({r.n_span})"
    elif cond in MATCHED and r.n_span < 2:
        reason = "single_token"
    elif cond == "s" and r.n_frag == 0:
        reason = "no_byte_fragment"
    return {
        "condition": cond, "filler": filler,
        "codepoints": " ".join(f"U+{ord(c):04X}" for c in filler),
        "utf8_bytes": len(filler.encode("utf-8")),
        "n_tokens": r.n_span, "n_byte_fragments": r.n_frag,
        "token_pieces": " | ".join(r.pieces),
        "eligible": reason == "", "reason": reason,
    }


def screen(tok, cands: Dict[str, List[str]] = None) -> List[dict]:
    cands = cands or candidates()
    return [screen_one(tok, c, f) for c, fs in cands.items() for f in fs]
