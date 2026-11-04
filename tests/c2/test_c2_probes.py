"""Probe-generator tests with the real (cached) tokenizers. No model weights."""
import collections

import pytest

from c2.carriers import CARRIERS
from c2.controls import MATCHED, candidates, screen
from c2.probes import (BUFFER, QLEN, R_BOUNDARY, R_COUNT, R_POST, R_PRE, R_TRUNC, Context, Rejected,
                       build_contexts, check_against, generate, make_probe)
from c2.tokaudit import AUDIT_CARRIER, LINE_CARRIER, audit_string

THUMBS_MED = "\U0001F44D\U0001F3FD"                                   # 👍🏽
FLAG_SG = "\U0001F1F8\U0001F1EC"                                       # flag: Singapore (in the 2-6 pool)
FAMILY = "\U0001F468\u200d\U0001F469\u200d\U0001F467"                  # family: man, woman, girl (too long for the pool)
HASH_KEYCAP, STAR_KEYCAP = "#\ufe0f\u20e3", "*\ufe0f\u20e3"


# ---- rule 1: boundary crossing --------------------------------------------------
@pytest.mark.parametrize("filler, after, merged", [("cat", "s and more text.", "cats"), (".", ". and more.", "..")])
def test_boundary_crossing_is_rejected(any_tok, filler, after, merged):
    """A token that merges the filler with the next character must be rejected."""
    r = audit_string(any_tok, filler, "Header:\n[PROBE]" + after)
    assert r.boundary and merged in "".join(r.pieces)
    ctx = Context("t", "pos24", "Header:\n", after + " filler" * 120, start=3)
    with pytest.raises(Rejected) as e:
        make_probe(any_tok, ctx, filler)
    assert e.value.reason == R_BOUNDARY


@pytest.mark.parametrize("keycap", [HASH_KEYCAP, STAR_KEYCAP])
def test_hash_star_keycaps_do_not_cross_under_our_rule(any_tok, keycap):
    """DOCUMENTS a discrepancy: the proposal excluded the # and * keycaps as
    boundary cases, but with the pinned tokenizers no token covers a non-space
    character outside them, in the audit carrier or the probe slot. Their first
    token is '#'/'*' (plus the leading space in the audit carrier), followed by
    the bytes of U+FE0F U+20E3. If this test ever fails, the tokenizer changed."""
    for carrier in (AUDIT_CARRIER, LINE_CARRIER):
        r = audit_string(any_tok, keycap, carrier)
        assert not r.boundary
        assert any_tok.decode([r.input_ids[i] for i in r.span_idx]).strip() == keycap


# ---- rules 2, 3, 4 and pre-probe identity --------------------------------------
def _ctx(tok, i=0, pos="pos24"):
    return next(c for c in build_contexts(tok, [CARRIERS[i]]) if c.position == pos)


def test_rule2_count_mismatch_rejected(pythia_tok):
    ctx = _ctx(pythia_tok)
    pe, pc = make_probe(pythia_tok, ctx, THUMBS_MED), make_probe(pythia_tok, ctx, "cat")
    with pytest.raises(Rejected) as e:
        check_against(pe, pc, match_count=True)
    assert e.value.reason == R_COUNT


def test_rule3_post_probe_ids_must_match(pythia_tok):
    ctx = _ctx(pythia_tok)
    other = Context(ctx.carrier, ctx.position, ctx.before, " differently" + ctx.after, ctx.start)
    pe, pc = make_probe(pythia_tok, ctx, THUMBS_MED), make_probe(pythia_tok, other, THUMBS_MED)
    with pytest.raises(Rejected) as e:
        check_against(pe, pc, match_count=True)
    assert e.value.reason == R_POST


def test_pre_probe_ids_must_match(pythia_tok):
    a, b = _ctx(pythia_tok, 0), _ctx(pythia_tok, 1)
    with pytest.raises(Rejected) as e:
        check_against(make_probe(pythia_tok, a, THUMBS_MED), make_probe(pythia_tok, b, THUMBS_MED), match_count=True)
    assert e.value.reason == R_PRE


def test_rule4_truncation_into_Q_rejected(pythia_tok):
    ctx = _ctx(pythia_tok, pos="pos96")
    with pytest.raises(Rejected) as e:
        make_probe(pythia_tok, ctx, THUMBS_MED, max_tokens=150)    # pos96 probes need ~180 tokens
    assert e.value.reason == R_TRUNC
    short = Context(ctx.carrier, ctx.position, ctx.before, " only a few words follow.", ctx.start)
    with pytest.raises(Rejected) as e:
        make_probe(pythia_tok, short, THUMBS_MED)
    assert e.value.reason == R_TRUNC


# ---- whole matched sets ---------------------------------------------------------
@pytest.fixture(scope="module", params=["pythia", "qwen"])
def generated(request):
    from c2.config import model_entry
    tok = request.getfixturevalue(f"{request.param}_tok")
    cands = candidates()
    cands["w"] = cands["w"][:600]
    eligible = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in screen(tok, cands):
        if r["eligible"]:
            eligible[r["condition"]][r["n_tokens"]].append(r["filler"])
    meta = {THUMBS_MED: {"index": 0, "name": "thumbs up: medium skin tone", "seq_type": "skin_tone"},
            FLAG_SG: {"index": 1, "name": "flag: Singapore", "seq_type": "flag"},
            FAMILY: {"index": 2, "name": "family: man, woman, girl", "seq_type": "zwj"}}
    probes, excl = generate(tok, model_entry(request.param)[1], request.param,
                            {"dev": [THUMBS_MED], "confirm": [FLAG_SG, FAMILY]}, meta, eligible)
    return tok, probes, excl


def _sets(probes):
    sets = collections.defaultdict(dict)
    for p in probes:
        sets[p["matched_set_id"]][p["condition"]] = p
    return sets


def test_matched_sets_have_equal_span_counts(generated):
    _, probes, _ = generated
    for sid, m in _sets(probes).items():
        counts = {c: m[c]["n_span_tokens"] for c in ["e", *MATCHED] if c in m}
        assert len(set(counts.values())) == 1, (sid, counts)
        assert m["p"]["n_span_tokens"] == 1 and m["r"]["n_span_tokens"] == 1


def test_matched_sets_have_identical_pre_and_post_token_ids(generated):
    _, probes, _ = generated
    for sid, m in _sets(probes).items():
        e = m["e"]
        s0, s1 = e["token_span"]
        for c, p in m.items():
            a0, a1 = p["token_span"]
            assert a0 == s0, (sid, c)
            assert p["input_ids"][:a0] == e["input_ids"][:s0], (sid, c)
            assert p["input_ids"][a1:a1 + BUFFER + QLEN] == e["input_ids"][s1:s1 + BUFFER + QLEN], (sid, c)


def test_buffer_and_query_indices(generated):
    tok, probes, _ = generated
    for p in probes:
        s0, s1 = p["token_span"]
        assert s0 == p["start_target"]                          # starts at exactly 24 / 96
        assert p["buffer_span"] == [s1, s1 + 16]
        assert p["query_span"] == [s1 + 16, s1 + 80]
        assert p["query_span"][1] <= p["seq_len"] == len(p["input_ids"]) <= 256
        # the span decodes back to the filler, and the text re-tokenises to the same IDs
        assert tok.decode(p["input_ids"][s0:s1]) == p["filler"]
        assert tok(p["text"], add_special_tokens=False)["input_ids"] == p["input_ids"]


def test_byte_fragment_control_really_fragments(generated):
    _, probes, _ = generated
    assert all(p["n_byte_fragments"] >= 1 for p in probes if p["condition"] == "s")


def test_pool_emoji_get_full_coverage(generated):
    _, probes, excl = generated
    assert not [x for x in excl if x["emoji"] in (THUMBS_MED, FLAG_SG)]
    complete = [m for m in _sets(probes).values() if set(m) == set("ewsxpr") and m["e"]["emoji"] != FAMILY]
    assert len(complete) == 2 * len(CARRIERS) * 2


def test_unmatchable_conditions_are_logged(generated):
    """Every (set, condition) that did not get a probe has exactly one exclusion
    row with a reason, so nothing is silently dropped. The family emoji is longer
    than most screened controls, so it exercises this path."""
    _, probes, excl = generated
    have = {(p["matched_set_id"].split("-", 1)[1], p["condition"]) for p in probes}
    logged = collections.Counter((f"e{2:04d}-{x['carrier']}-{x['position']}", x["condition"]) for x in excl)
    fam = [x for x in excl if x["emoji"] == FAMILY]
    assert fam, "expected at least one unmatchable condition for the family emoji"
    for x in fam:
        key = (f"e{2:04d}-{x['carrier']}-{x['position']}", x["condition"])
        assert key not in have and logged[key] == 1 and x["reason"]
    for c in "ewsxpr":
        for carrier in CARRIERS:
            for pos in ("pos24", "pos96"):
                key = (f"e{2:04d}-{carrier.name}-{pos}", c)
                assert (key in have) != (key in logged), key   # exactly one of: probe, or logged exclusion
