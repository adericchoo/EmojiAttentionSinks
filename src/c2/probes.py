"""Probe generator: matched sets, rejection rules, exclusion log, dev/confirm split.

A matched set = one emoji x one carrier x one position, plus its controls:
  w, s, x  same span token count as the emoji (rule 2), and
  p, r     single-token references,
all sharing IDENTICAL token IDs before the probe (same start index) and after it.

Sequence layout (token indices, half-open [start, end)):
  [0, start)               neutral prefix + carrier header + "\\n"   (start = 24 or 96)
  [start, end)             probe span S (the filler)
  [end, end + 16)          buffer / "near block" (queries for M_near, RQ2)
  [end + 16, end + 80)     query block Q (the 64 "voters" for M_far)
  [end + 80, seq_len)      0-2 trailing tokens, so the text ends on a word.
                           Causal attention means they cannot affect Q.

Rejection rules (each failure is logged; never silently scored):
  1 boundary_crossing      a span token covers a non-space char outside the filler
  2 span_count_differs     control span count != emoji span count
  3 post_probe_ids_differ  token IDs after the probe differ from the emoji's
  4 truncation_cuts_Q      the sequence cannot hold the full Q within max_tokens
  + pre_probe_ids_differ   token IDs before the probe differ (would shift start)
  + no_candidate_with_count  no screened filler has the emoji's token count
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .carriers import CARRIERS, CONTINUATION, PREFIX, Carrier
from .controls import MATCHED, PUNCT, REF_WORDS
from .tokaudit import tokenize_in_context

POSITIONS = {"pos24": 24, "pos96": 96}
BUFFER = 16
QLEN = 64
MAX_TOKENS = 256
SEED = 2026
N_DEV = 5
N_CONFIRM = 48

R_BOUNDARY = "boundary_crossing"
R_COUNT = "span_count_differs"
R_POST = "post_probe_ids_differ"
R_TRUNC = "truncation_cuts_Q"
R_PRE = "pre_probe_ids_differ"
R_NOCAND = "no_candidate_with_count"


class Rejected(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


@dataclass(frozen=True)
class Context:
    """Fixed text around the slot for one (model, carrier, position)."""
    carrier: str
    position: str
    before: str
    after: str
    start: int   # actual token index of the slot (target 24 / 96)


def _n_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def fit_prefix(tok, header: str, target: int) -> Tuple[str, int]:
    """Shortest prefix (whole words; closed with '.' where that still hits the
    target) so that the text before the slot is exactly `target` tokens. Falls
    back to the closest count."""
    words = PREFIX.split()
    best = None
    for k in range(len(words) + 1):
        p = " ".join(words[:k])
        variants = [p + "." if p and p[-1] not in ".!?" else p, p]
        for v in variants:
            before = (v + " " if v else "") + header + "\n"
            n = _n_tokens(tok, before)
            if best is None or abs(n - target) < abs(best[1] - target):
                best = (before, n)
            if n == target:
                return best
        if n > target + 4:
            break
    return best


def build_contexts(tok, carriers: List[Carrier] = CARRIERS) -> List[Context]:
    out = []
    for c in carriers:
        assert c.tail.startswith(" "), c.name
        for pos, target in POSITIONS.items():
            before, n = fit_prefix(tok, c.header, target)
            out.append(Context(c.name, pos, before, c.tail + " " + CONTINUATION, n))
    return out


def make_probe(tok, ctx: Context, filler: str, max_tokens: int = MAX_TOKENS) -> dict:
    """Tokenise one probe, cut it just after Q, and check rules 1 and 4.

    Returns the probe record (without set-level fields). Raises Rejected.
    """
    r = tokenize_in_context(tok, ctx.before, filler, ctx.after)
    if r.boundary:
        raise Rejected(R_BOUNDARY, " | ".join(r.pieces))
    s0, s1 = r.span_idx[0], r.span_idx[-1] + 1
    q_end = s1 + BUFFER + QLEN
    if len(r.input_ids) < q_end:
        raise Rejected(R_TRUNC, f"only {len(r.input_ids) - s1} tokens after the probe")
    # cut the text at the end of the word containing the last Q token
    cut = r.offsets[q_end - 1][1]
    while cut < len(r.text) and not r.text[cut].isspace():
        cut += 1
    text = r.text[:cut]
    enc = tok(text, add_special_tokens=False)
    ids = list(enc["input_ids"])
    if ids[:q_end] != r.input_ids[:q_end]:
        raise Rejected(R_TRUNC, "cutting the text changed tokens inside Q")
    if len(ids) > max_tokens:
        raise Rejected(R_TRUNC, f"seq_len {len(ids)} > max_tokens {max_tokens}")
    return {
        "carrier": ctx.carrier, "position": ctx.position, "start_target": POSITIONS[ctx.position],
        "filler": filler, "text": text, "input_ids": ids,
        "char_span": list(r.char_span),
        "token_span": [s0, s1],
        "n_span_tokens": s1 - s0,
        "n_byte_fragments": r.n_frag,
        "filler_pieces": r.pieces,
        "buffer_span": [s1, s1 + BUFFER],
        "query_span": [s1 + BUFFER, q_end],
        "seq_len": len(ids),
    }


def check_against(ref: dict, p: dict, match_count: bool) -> None:
    """Set-level rules: identical pre/post token IDs (and equal span count)."""
    s0, s1 = p["token_span"]
    r0, r1 = ref["token_span"]
    if s0 != r0 or p["input_ids"][:s0] != ref["input_ids"][:r0]:
        raise Rejected(R_PRE, f"start {s0} vs {r0}")
    if match_count and (s1 - s0) != (r1 - r0):
        raise Rejected(R_COUNT, f"{s1 - s0} vs {r1 - r0} tokens")
    if p["input_ids"][s1:s1 + BUFFER + QLEN] != ref["input_ids"][r1:r1 + BUFFER + QLEN]:
        raise Rejected(R_POST)


def _rank(key: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()[:12], 16)


# ---------------------------------------------------------------------------
# Emoji sample + split
# ---------------------------------------------------------------------------
def sample_emoji(pool: List[str], strata_of: Dict[str, Tuple], n_dev: int = N_DEV,
                 n_confirm: int = N_CONFIRM, seed: int = SEED) -> Dict[str, List[str]]:
    """Round-robin over strata (sequence type x primary-model token count), so
    every type/length combination is represented before any repeats. Dev emoji
    are drawn first and never appear in the confirmatory set."""
    rng = random.Random(seed)
    strata = collections.defaultdict(list)
    for e in pool:
        strata[strata_of[e]].append(e)
    for k in strata:
        rng.shuffle(strata[k])
    order = sorted(strata)
    rng.shuffle(order)
    picked = []
    while len(picked) < n_dev + n_confirm and any(strata.values()):
        for k in order:
            if strata[k] and len(picked) < n_dev + n_confirm:
                picked.append(strata[k].pop())
    return {"dev": picked[:n_dev], "confirm": picked[n_dev:]}


# ---------------------------------------------------------------------------
# Matched-set generation
# ---------------------------------------------------------------------------
def generate(tok, model: str, tag: str, emoji_split: Dict[str, List[str]],
             emoji_meta: Dict[str, dict], eligible: Dict[str, Dict[int, List[str]]],
             contexts: Optional[List[Context]] = None) -> Tuple[List[dict], List[dict]]:
    """Returns (probes, exclusions).

    eligible[cond][n_tokens] = fillers that passed the controls screen.
    For each (emoji, condition) the control filler is chosen ONCE in a seeded
    order (least-used first, so fillers spread across emoji), and reused in every
    carrier/position where it still passes; otherwise the next candidate is tried.
    """
    contexts = contexts or build_contexts(tok)
    probes, excl = [], []
    usage = collections.Counter()

    def log(emoji, ctx, cond, reason, detail="", filler=""):
        excl.append({"model": model, "emoji": emoji, "emoji_name": emoji_meta[emoji]["name"],
                     "carrier": ctx.carrier, "position": ctx.position, "condition": cond,
                     "filler": filler, "reason": reason, "detail": detail})

    for split in ("dev", "confirm"):
        for e in emoji_split[split]:
            eid = emoji_meta[e]["index"]
            preferred: Dict[str, List[str]] = {}
            for ctx in contexts:
                set_id = f"{tag}-e{eid:04d}-{ctx.carrier}-{ctx.position}"
                base = {"model": model, "matched_set_id": set_id, "split": split,
                        "emoji": e, "emoji_name": emoji_meta[e]["name"], "seq_type": emoji_meta[e]["seq_type"]}
                try:
                    pe = make_probe(tok, ctx, e)
                except Rejected as err:
                    log(e, ctx, "e", err.reason, err.detail, e)
                    continue
                members = {"e": pe}
                n = pe["n_span_tokens"]
                for cond in MATCHED:
                    if cond not in preferred or not preferred[cond]:
                        cands = eligible[cond].get(n, [])
                        preferred[cond] = sorted(cands, key=lambda f: (usage[(cond, f)], _rank(f"{tag}:{e}:{cond}:{f}")))
                    if not preferred[cond]:
                        log(e, ctx, cond, R_NOCAND, f"no screened {cond} filler with {n} tokens")
                        continue
                    fails = collections.Counter()
                    for f in preferred[cond]:
                        try:
                            pc = make_probe(tok, ctx, f)
                            check_against(pe, pc, match_count=True)
                        except Rejected as err:
                            fails[err.reason] += 1
                            continue
                        members[cond] = pc
                        if f != preferred[cond][0]:   # make the working filler the default
                            preferred[cond].remove(f)
                            preferred[cond].insert(0, f)
                        break
                    if cond in members:
                        usage[(cond, members[cond]["filler"])] += 1
                    else:
                        top = fails.most_common(1)[0][0]
                        log(e, ctx, cond, top, f"tried {sum(fails.values())}: {dict(fails)}")
                # single-token references, rotated so each appears equally often
                k = CARRIERS.index(next(c for c in CARRIERS if c.name == ctx.carrier))
                for cond, pool, offset in (("p", PUNCT, k), ("r", REF_WORDS, k + eid)):
                    fails = collections.Counter()
                    for j in range(len(pool)):
                        f = pool[(offset + j) % len(pool)]
                        try:
                            pc = make_probe(tok, ctx, f)
                            check_against(pe, pc, match_count=False)
                            if pc["n_span_tokens"] != 1:
                                raise Rejected(R_COUNT, f"{pc['n_span_tokens']} tokens, need 1")
                        except Rejected as err:
                            fails[err.reason] += 1
                            continue
                        members[cond] = pc
                        break
                    if cond not in members:
                        log(e, ctx, cond, fails.most_common(1)[0][0], f"tried {dict(fails)}")
                for cond, p in members.items():
                    probes.append({"id": f"{set_id}-{cond}", **base, "condition": cond, **p})
    return probes, excl


def coverage(probes: List[dict], excl: List[dict], n_attempted: Dict[str, int]) -> str:
    """Matched sets achieved per condition, and exclusions by reason."""
    lines = []
    by = collections.defaultdict(set)
    for p in probes:
        by[(p["split"], p["condition"])].add(p["matched_set_id"])
    lines.append(f"  {'condition':28}" + "".join(f"{s:>14}" for s in ("dev", "confirm")))
    names = {"e": "e emoji (set kept)", "w": "w English word", "s": "s rare chars", "x": "x hex string",
             "p": "p punctuation", "r": "r ordinary word"}
    for c in ["e", "w", "s", "x", "p", "r"]:
        cells = "".join(f"{len(by[(s, c)]):>7} / {n_attempted[s]:<5}" for s in ("dev", "confirm"))
        lines.append(f"  {names[c]:28}{cells}")
    sets = collections.defaultdict(set)
    for p in probes:
        sets[p["matched_set_id"]].add(p["condition"])
    complete = collections.Counter()
    split_of = {p["matched_set_id"]: p["split"] for p in probes}
    for sid, cs in sets.items():
        if set("ewsxpr") <= cs:
            complete[split_of[sid]] += 1
    lines.append(f"  {'all 6 conditions':28}" + "".join(f"{complete[s]:>7} / {n_attempted[s]:<5}" for s in ("dev", "confirm")))
    lines.append("")
    lines.append("  exclusions by reason (one row per rejected/unmatchable set x condition):")
    reasons = collections.Counter((x["reason"], x["condition"]) for x in excl)
    if not reasons:
        lines.append("    (none)")
    for (reason, cond), n in sorted(reasons.items()):
        lines.append(f"    {reason:28} condition {cond}: {n}")
    return "\n".join(lines)


def write_jsonl(rows: List[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_exclusions(rows: List[dict], path: str) -> None:
    fields = ["model", "emoji", "emoji_name", "carrier", "position", "condition", "filler", "reason", "detail"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
