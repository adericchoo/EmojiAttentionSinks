"""Build the F0 control set: non-emoji characters that are 4 bytes in UTF-8.

    python scripts/build_f0_control.py                  # writes config/f0_control_stimuli.yaml
    python scripts/build_f0_control.py --output PATH    # write somewhere else, e.g. for review
    python scripts/build_f0_control.py --check          # rebuild and compare with the committed file

Why this set exists. Leg 3 (script 06) compares emoji that start with the [F0 9F]
byte token (group A) against symbols that start with E2 (groups B and D). In every
existing group, "is an emoji" and "starts with [F0 9F]" go together, so the two
cannot be told apart. This set fills the two missing cells:

  E1_f0_9f_nonemoji   NOT emoji, but the SAME lead token [F0 9F] as group A
                      (dominoes, alchemical symbols, arrows, chess, legacy computing)
  E2_f0_other_4byte   NOT emoji, also 4 bytes, but a DIFFERENT lead token
                      (CJK Extension B/C, cuneiform, hieroglyphs, Linear B, maths letters)

A vs E1 asks "emoji, or the [F0 9F] token?". A vs E2 asks "[F0 9F] specifically,
or the start of any 4-byte character?". The set is scored by scripts/12_f0_control.py.

Selection is deterministic: a seeded order inside each Unicode block, then
round-robin across blocks until each group is full. Every kept character is checked
with BOTH pinned tokenizers in all 16 carrier/position slots of the structured
probe, with the same token IDs before and after the slot as a group-A emoji.

Additive: nothing in config/emoji_stimuli.yaml, config/master_stimuli.yaml or any
other team file is changed.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone

from _bootstrap import ROOT

import yaml
from c2.config import load_tokenizer, model_entry
from c2.probes import Rejected, build_contexts, check_against, make_probe
from c2.tokaudit import token_bytes

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


OUT_PATH = os.path.join(ROOT, "config", "f0_control_stimuli.yaml")
EMOJI_TEST = os.path.join(ROOT, "data", "unicode", "emoji-test-15.1.txt")

VERSION_ID = "f0-control-v1.0"
# Not frozen until the team has reviewed the list. Runs on a draft set are
# development runs (12_f0_control.py labels them DEVELOPMENT PILOT).
STATUS = "draft_for_team_review"
SEED = 2026
N_PER_GROUP = 24
# Every candidate must share the token IDs before and after the slot with this
# emoji (it is in group A of both team stimulus files), in every slot.
REFERENCE_EMOJI = "\U0001F680"
# Letters, numbers and symbols only: no marks, format characters or unassigned code points.
KEEP_CATEGORY_PREFIXES = ("L", "N", "S")
MODELS = ("pythia", "qwen")

F0_9F = b"\xf0\x9f"
F0 = b"\xf0"

# All non-emoji blocks inside U+1F000-1FFFF are screened. Only some pass the
# both-models rule: Qwen merges [F0 9F] with the next byte for the others.
E1_BLOCKS = [
    ("Mahjong Tiles", 0x1F000, 0x1F02F),
    ("Domino Tiles", 0x1F030, 0x1F09F),
    ("Playing Cards", 0x1F0A0, 0x1F0FF),
    ("Enclosed Alphanumeric Supplement", 0x1F100, 0x1F1FF),
    ("Enclosed Ideographic Supplement", 0x1F200, 0x1F2FF),
    ("Miscellaneous Symbols and Pictographs", 0x1F300, 0x1F5FF),
    ("Ornamental Dingbats", 0x1F650, 0x1F67F),
    ("Transport and Map Symbols", 0x1F680, 0x1F6FF),
    ("Alchemical Symbols", 0x1F700, 0x1F77F),
    ("Geometric Shapes Extended", 0x1F780, 0x1F7FF),
    ("Supplemental Arrows-C", 0x1F800, 0x1F8FF),
    ("Supplemental Symbols and Pictographs", 0x1F900, 0x1F9FF),
    ("Chess Symbols", 0x1FA00, 0x1FA6F),
    ("Symbols for Legacy Computing", 0x1FB00, 0x1FBFF),
]

E2_BLOCKS = [
    ("Linear B Syllabary", 0x10000, 0x1007F),
    ("Old Italic", 0x10300, 0x1032F),
    ("Gothic", 0x10330, 0x1034F),
    ("Cuneiform", 0x12000, 0x123FF),
    ("Egyptian Hieroglyphs", 0x13000, 0x1342F),
    ("Mathematical Alphanumeric Symbols", 0x1D400, 0x1D7FF),
    ("CJK Unified Ideographs Extension B", 0x20000, 0x2A6DF),
    ("CJK Unified Ideographs Extension C", 0x2A700, 0x2B73F),
]

GROUPS = {
    "E1_f0_9f_nonemoji": {
        "tests": "Same lead token [F0 9F] as group A, but not emoji: is the sink about emoji, or about the token?",
        "lead_byte_token": "F0 9F",
        "blocks": E1_BLOCKS,
    },
    "E2_f0_other_4byte": {
        "tests": "4-byte, not emoji, different lead token: is the sink [F0 9F] specifically, or any 4-byte start?",
        "lead_byte_token": "F0 (not followed by 9F)",
        "blocks": E2_BLOCKS,
    },
}

SELECTION_RULES = [
    "Candidates: assigned code points (letters, numbers, symbols) in the declared blocks, 4 bytes in UTF-8, "
    "never appearing in any sequence of emoji-test-15.1.txt (so no emoji and no emoji components).",
    "Order: within each block, sorted by sha256('<seed>:<group>:<code point in hex>'); blocks are visited "
    "round-robin in declared order until the group holds N_PER_GROUP characters.",
    "Every kept character builds a clean probe in all 16 carrier/position slots in both models, with the "
    "same token IDs before and after the slot as the reference emoji U+1F680, and the same split in every slot.",
    "E1: in BOTH models the first slot token is exactly the bytes F0 9F and the character is exactly 3 tokens "
    "(the same lead token, and in Pythia the same token count, as group A).",
    "E2: the first byte is F0 and the second is not 9F. In Pythia the first token is the lone byte F0 and the "
    "character is 4 tokens (one per byte, Pythia's normal split); in Qwen the first token starts with F0.",
]


# ---------------------------------------------------------------------------
# Pure rules (no tokenizer needed; covered by tests/test_f0_control.py)
# ---------------------------------------------------------------------------
def emoji_codepoints(path=EMOJI_TEST):
    """Every code point that appears in any line of emoji-test.txt."""
    cps = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            match = re.match(r"^([0-9A-F ]+);", line)
            if match:
                for cp in match.group(1).split():
                    cps.add(int(cp, 16))
    return cps


def is_candidate(cp, emoji_cps):
    ch = chr(cp)
    if cp in emoji_cps or len(ch.encode("utf-8")) != 4:
        return False
    if not unicodedata.category(ch).startswith(KEEP_CATEGORY_PREFIXES):
        return False
    return unicodedata.name(ch, "") != ""


def rank(group, cp):
    return int(hashlib.sha256(f"{SEED}:{group}:{cp:X}".encode()).hexdigest()[:16], 16)


def e1_split_ok(splits):
    """Same lead token as group A ([F0 9F]) and exactly 3 tokens, in every model."""
    return all(pieces[0] == F0_9F and len(pieces) == 3 for pieces in splits.values())


def e2_split_ok(splits):
    """A lead token that is NOT [F0 9F]: the lone byte F0 + 3 bytes in Pythia; F0-initial in Qwen."""
    pythia, qwen = splits["pythia"], splits["qwen"]
    pythia_ok = pythia[0] == F0 and len(pythia) == 4
    qwen_ok = qwen[0].startswith(F0) and not qwen[0].startswith(F0_9F)
    return pythia_ok and qwen_ok


SPLIT_RULES = {"E1_f0_9f_nonemoji": e1_split_ok, "E2_f0_other_4byte": e2_split_ok}


def round_robin(blocks, accept, n):
    """Visit blocks in order, taking each block's next accepted candidate, until n are kept.

    `blocks` is [(name, [candidate, ...])] already in seeded order; `accept(c)` returns
    (ok, info_or_reason). Returns (kept, per-block screen log).
    """
    iters = [(name, iter(cands)) for name, cands in blocks]
    log = {name: {"candidates": len(cands), "tried": 0, "kept": 0, "reasons": collections.Counter()}
           for name, cands in blocks}
    kept = []
    active = list(iters)
    while len(kept) < n and active:
        still = []
        for name, it in active:
            if len(kept) >= n:
                still.append((name, it))
                continue
            for cand in it:
                log[name]["tried"] += 1
                ok, info = accept(cand)
                if ok:
                    log[name]["kept"] += 1
                    kept.append((name, cand, info))
                    still.append((name, it))
                    break
                log[name]["reasons"][info] += 1
        active = still
    return kept, log


def content_sha256(groups):
    blob = json.dumps(groups, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def hex_bytes(raw):
    return " ".join("%02X" % b for b in raw)


# ---------------------------------------------------------------------------
# Tokenizer checks
# ---------------------------------------------------------------------------
class Slots:
    """The 16 structured-probe slots for one tokenizer, plus the reference emoji's probes."""

    def __init__(self, model):
        self.tok = load_tokenizer(model)
        self.contexts = build_contexts(self.tok)
        self.reference = [make_probe(self.tok, ctx, REFERENCE_EMOJI) for ctx in self.contexts]

    def split(self, ch):
        """The character's slot tokens (as raw bytes) and lead token ID; raises Rejected."""
        pieces = None
        lead_id = None
        for ctx, ref in zip(self.contexts, self.reference):
            probe = make_probe(self.tok, ctx, ch)
            check_against(ref, probe, match_count=False)
            s0, s1 = probe["token_span"]
            ids = probe["input_ids"][s0:s1]
            here = [token_bytes(self.tok, i) for i in ids]
            if pieces is None:
                pieces, lead_id = here, ids[0]
            elif here != pieces:
                raise Rejected("split_varies", f"{ctx.carrier}/{ctx.position}")
        return pieces, lead_id


def build():
    emoji_cps = emoji_codepoints()
    slots = {m: Slots(m) for m in MODELS}

    groups = {}
    screen = {}
    for group, spec in GROUPS.items():
        rule = SPLIT_RULES[group]
        blocks = []
        for name, lo, hi in spec["blocks"]:
            cands = [cp for cp in range(lo, hi + 1) if is_candidate(cp, emoji_cps)]
            cands.sort(key=lambda cp: rank(group, cp))
            blocks.append((f"U+{lo:04X}-U+{hi:04X} {name}", cands))

        def accept(cp):
            ch = chr(cp)
            splits, lead_ids = {}, {}
            for m in MODELS:
                try:
                    splits[m], lead_ids[m] = slots[m].split(ch)
                except Rejected as err:
                    return False, f"{m}:{err.reason}"
            if not rule(splits):
                return False, "split_rule"
            return True, {"splits": splits, "lead_ids": lead_ids}

        kept, log = round_robin(blocks, accept, N_PER_GROUP)
        items = []
        for block, cp, info in kept:
            ch = chr(cp)
            items.append({
                "char": ch,
                "name": unicodedata.name(ch).lower(),
                "codepoint": f"U+{cp:04X}",
                "unicode_block": block,
                "utf8": hex_bytes(ch.encode("utf-8")),
                "token_split": {
                    m: {"pieces": [hex_bytes(p) for p in info["splits"][m]],
                        "n_tokens": len(info["splits"][m]),
                        "lead_token_id": int(info["lead_ids"][m])}
                    for m in MODELS
                },
                "slots_ok": len(slots["pythia"].contexts),
            })
        groups[group] = {"tests": spec["tests"], "lead_byte_token": spec["lead_byte_token"], "items": items}
        screen[group] = {name: {"candidates": v["candidates"], "tried": v["tried"], "kept": v["kept"],
                                "rejected": dict(v["reasons"])} for name, v in log.items()}

    with open(EMOJI_TEST, "rb") as handle:
        emoji_sha = hashlib.sha256(handle.read()).hexdigest()
    tokenizers = {}
    for m in MODELS:
        _, name, rev = model_entry(m)
        tokenizers[m] = {"name": name, "revision": rev}

    return {
        "pre_registration": {
            "version_id": VERSION_ID,
            "status": STATUS,
            "build_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "content_sha256": content_sha256(groups),
            "unicode_data": f"python unicodedata {unicodedata.unidata_version}",
            "emoji_data": {"file": "data/unicode/emoji-test-15.1.txt", "sha256": emoji_sha},
            "tokenizers": tokenizers,
            "seed": SEED,
            "n_per_group": N_PER_GROUP,
            "reference_emoji": "U+1F680",
            "selection_rules": SELECTION_RULES,
        },
        "meta": {
            "reduction": "per_fragment_head_aware",
            "carrier": "structured_probe",
            "fragment_scored": "lead (first slot token)",
            "used_by": "scripts/12_f0_control.py",
        },
        "groups": groups,
        "screen": screen,
    }


HEADER = """# config/f0_control_stimuli.yaml  (generated by scripts/build_f0_control.py; do not hand-edit)
# Non-emoji 4-byte controls for the F0 9F sink: E1 = same [F0 9F] lead token as
# group A but not emoji; E2 = 4-byte, not emoji, different lead token.
# Status "draft_for_team_review": not frozen. Freeze it (new status, same hash)
# before any run whose numbers are reported.
"""


def write(data, path):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(HEADER)
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False, width=110)


def summarise(data):
    pre = data["pre_registration"]
    print(f"{pre['version_id']}  status={pre['status']}  content_sha256={pre['content_sha256']}")
    for group, g in data["groups"].items():
        print(f"\n{group}: {len(g['items'])} items  ({g['tests']})")
        for block, s in data["screen"][group].items():
            print(f"  {block:<52} candidates {s['candidates']:>6}  tried {s['tried']:>4}  kept {s['kept']:>2}"
                  + (f"  rejected {s['rejected']}" if s["rejected"] else ""))
        for it in g["items"]:
            ts = it["token_split"]
            print(f"    {it['char']}  {it['codepoint']:<8} {it['name'][:40]:<40} "
                  f"pythia [{'] ['.join(ts['pythia']['pieces'])}]  qwen [{'] ['.join(ts['qwen']['pieces'])}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=OUT_PATH)
    ap.add_argument("--check", action="store_true", help="rebuild and compare with the committed file")
    a = ap.parse_args()

    data = build()
    summarise(data)
    if a.check:
        with open(OUT_PATH, encoding="utf-8") as handle:
            committed = yaml.safe_load(handle)
        same = committed["pre_registration"]["content_sha256"] == data["pre_registration"]["content_sha256"]
        print(f"\n--check: committed {committed['pre_registration']['content_sha256']}\n"
              f"         rebuilt   {data['pre_registration']['content_sha256']}  -> {'MATCH' if same else 'DIFFERS'}")
        raise SystemExit(0 if same else 1)
    write(data, a.output)
    print(f"\nwrote {os.path.relpath(a.output, ROOT)}")


if __name__ == "__main__":
    main()
