"""Emoji inventory: the official Unicode Emoji 15.1 test file.

We use only the ``fully-qualified`` lines (3,773 in v15.1). The file is committed
under data/unicode/ and its SHA-256 is checked on every load, so every run is
provably on the same inventory as the proposal.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import List

from . import DATA_DIR

EMOJI_TEST_URL = "https://www.unicode.org/Public/emoji/15.1/emoji-test.txt"
EMOJI_TEST_PATH = os.path.join(DATA_DIR, "unicode", "emoji-test-15.1.txt")
EXPECTED_SHA256 = "d876ee249aa28eaa76cfa6dfaa702847a8d13b062aa488d465d0395ee8137ed9"
EXPECTED_FULLY_QUALIFIED = 3773

ZWJ = 0x200D
VS16 = 0xFE0F
KEYCAP = 0x20E3
SKIN_TONES = range(0x1F3FB, 0x1F3FF + 1)
REGIONAL_INDICATORS = range(0x1F1E6, 0x1F1FF + 1)
TAGS = range(0xE0020, 0xE007F + 1)


@dataclass
class Emoji:
    text: str            # the grapheme cluster itself
    codepoints: List[int]
    name: str            # CLDR short name from the file, e.g. "thumbs up: medium skin tone"
    group: str
    subgroup: str

    @property
    def codepoints_str(self) -> str:
        return " ".join(f"U+{c:04X}" for c in self.codepoints)

    @property
    def utf8_bytes(self) -> int:
        return len(self.text.encode("utf-8"))

    @property
    def seq_type(self) -> str:
        return sequence_type(self.codepoints)


def sequence_type(cps: List[int]) -> str:
    """One of: keycap, flag, zwj, skin_tone, single.

    Checked in that order, so a ZWJ sequence that also carries a skin tone
    (e.g. 👩🏽‍💻) is "zwj", and the rainbow flag 🏳️‍🌈 (a ZWJ sequence) is "zwj".
    "flag" means regional-indicator pairs (🇸🇬) and tag sequences (🏴󠁧󠁢󠁥󠁮󠁧󠁿).
    "single" is one emoji code point, optionally followed by VS16 (❤️).
    """
    if KEYCAP in cps:
        return "keycap"
    if all(c in REGIONAL_INDICATORS for c in cps) or any(c in TAGS for c in cps):
        return "flag"
    if ZWJ in cps:
        return "zwj"
    if any(c in SKIN_TONES for c in cps):
        return "skin_tone"
    return "single"


def file_sha256(path: str = EMOJI_TEST_PATH) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_emoji(path: str = EMOJI_TEST_PATH, check: bool = True) -> List[Emoji]:
    """Parse the fully-qualified emoji, in file order."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing; download it from {EMOJI_TEST_URL}")
    if check:
        sha = file_sha256(path)
        if sha != EXPECTED_SHA256:
            raise ValueError(f"emoji-test.txt SHA-256 {sha} != expected {EXPECTED_SHA256}")

    out = []
    group = subgroup = ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# group:"):
                group = line.split(":", 1)[1].strip()
            elif line.startswith("# subgroup:"):
                subgroup = line.split(":", 1)[1].strip()
            if not line or line.startswith("#"):
                continue
            # "1F44D 1F3FD ; fully-qualified # 👍🏽 E1.0 thumbs up: medium skin tone"
            fields, comment = line.split("#", 1)
            cps_str, status = fields.split(";")
            if status.strip() != "fully-qualified":
                continue
            cps = [int(c, 16) for c in cps_str.split()]
            name = comment.strip().split(" ", 2)[2]  # drop the glyph and "E1.0"
            out.append(Emoji("".join(chr(c) for c in cps), cps, name, group, subgroup))

    if check and len(out) != EXPECTED_FULLY_QUALIFIED:
        raise ValueError(f"found {len(out)} fully-qualified emoji, expected {EXPECTED_FULLY_QUALIFIED}")
    return out
