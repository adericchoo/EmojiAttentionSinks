"""Offline tests for the F0 control: the builder's rules, the committed set, and
the runner's analysis helpers.

No model or tokenizer is loaded and nothing is downloaded. The committed config is
checked against the rules it claims to follow; the analysis helpers run on small
synthetic [L, H] arrays whose answers are known.
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# scripts first, then src, so src/ ends up in FRONT: scripts/c2.py would otherwise
# shadow the src/c2 package (same note as tests/test_byte_structure.py).
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pytest
import yaml


def _load(filename, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_f0_control.py", "build_f0_control")
f0 = _load("12_f0_control.py", "f0_control")

F0_9F = b"\xf0\x9f"


# ---------------------------------------------------------------------------
# Builder rules
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def emoji_cps():
    return build.emoji_codepoints()


def test_emoji_test_file_covers_emoji_and_components(emoji_cps):
    assert 0x1F680 in emoji_cps      # rocket
    assert 0x1F3FD in emoji_cps      # skin-tone modifier (a component)
    assert 0x1F1F8 in emoji_cps      # regional indicator S (only appears inside flags)
    assert 0x1F047 not in emoji_cps  # a domino tile


def test_is_candidate(emoji_cps):
    assert build.is_candidate(0x1F047, emoji_cps)        # domino tile: 4 bytes, symbol, not emoji
    assert build.is_candidate(0x20000, emoji_cps)        # CJK Ext B: 4 bytes, letter
    assert not build.is_candidate(0x1F680, emoji_cps)    # emoji
    assert not build.is_candidate(0x2122, emoji_cps)     # trade mark: only 3 bytes
    assert not build.is_candidate(0x1D167, emoji_cps)    # combining musical mark (category Mn)
    assert not build.is_candidate(0x1F0FF, emoji_cps)    # unassigned


def test_split_rules():
    ok = {"pythia": [F0_9F, b"\xa0", b"\x80"], "qwen": [F0_9F, b"\xa0", b"\x80"]}
    assert build.e1_split_ok(ok)
    assert not build.e1_split_ok({"pythia": ok["pythia"], "qwen": [b"\xf0\x9f\x80", b"\x80"]})  # Qwen merged
    assert not build.e1_split_ok({"pythia": [F0_9F, b"\xa0\x80"], "qwen": ok["qwen"]})        # 2 tokens

    e2 = {"pythia": [b"\xf0", b"\xa0", b"\x80", b"\x80"], "qwen": [b"\xf0\xa0", b"\x80", b"\x80"]}
    assert build.e2_split_ok(e2)
    assert not build.e2_split_ok({"pythia": e2["pythia"][:3], "qwen": e2["qwen"]})             # 3 tokens
    assert not build.e2_split_ok({"pythia": [F0_9F, b"\x80", b"\x80"], "qwen": e2["qwen"]})    # F0 9F lead


def test_round_robin_spreads_and_skips_empty_blocks():
    blocks = [("b1", [1, 3, 5]), ("b2", [2, 4, 6, 8]), ("b3", [10, 12])]

    def accept(c):
        return (True, None) if c % 2 == 0 else (False, "odd")

    kept, log = build.round_robin(blocks, accept, 4)
    assert [c for _, c, _ in kept] == [2, 10, 4, 12]  # b1 has nothing acceptable and drops out
    assert log["b1"]["kept"] == 0 and log["b1"]["reasons"]["odd"] == 3
    assert log["b2"]["kept"] == 2 and log["b3"]["kept"] == 2


def test_content_hash_ignores_key_order():
    a = {"g": {"items": [{"char": "x", "name": "n"}], "tests": "t"}}
    b = {"g": {"tests": "t", "items": [{"name": "n", "char": "x"}]}}
    assert build.content_sha256(a) == build.content_sha256(b)


# ---------------------------------------------------------------------------
# The committed set follows its own rules
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def committed():
    with open(build.OUT_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_committed_hash_and_counts(committed):
    pre = committed["pre_registration"]
    assert pre["content_sha256"] == build.content_sha256(committed["groups"])
    assert pre["status"] == build.STATUS
    for group in build.GROUPS:
        items = committed["groups"][group]["items"]
        assert len(items) == build.N_PER_GROUP
        assert len({it["codepoint"] for it in items}) == len(items)


def test_committed_items_obey_group_rules(committed, emoji_cps):
    for it in committed["groups"]["E1_f0_9f_nonemoji"]["items"]:
        cp = ord(it["char"])
        assert it["utf8"].startswith("F0 9F") and cp not in emoji_cps
        for m in build.MODELS:
            assert it["token_split"][m]["pieces"][0] == "F0 9F"
            assert it["token_split"][m]["n_tokens"] == 3
    for it in committed["groups"]["E2_f0_other_4byte"]["items"]:
        cp = ord(it["char"])
        assert it["utf8"].startswith("F0") and not it["utf8"].startswith("F0 9F") and cp not in emoji_cps
        assert it["token_split"]["pythia"]["pieces"][0] == "F0"
        assert it["token_split"]["pythia"]["n_tokens"] == 4


def test_e1_shares_one_lead_token_id_per_model(committed):
    for m in build.MODELS:
        ids = {it["token_split"][m]["lead_token_id"] for it in committed["groups"]["E1_f0_9f_nonemoji"]["items"]}
        assert len(ids) == 1


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------
def test_clean_lead():
    assert f0.clean_lead(f0.GROUP_A, F0_9F)
    assert not f0.clean_lead(f0.GROUP_A, b"\xf0\x9f\x98\x80")   # Qwen whole-token emoji
    assert f0.clean_lead(f0.GROUP_E1, F0_9F)
    assert f0.clean_lead(f0.GROUP_E2, b"\xf0")
    assert f0.clean_lead(f0.GROUP_E2, b"\xf0\x90\x8c")
    assert not f0.clean_lead(f0.GROUP_E2, F0_9F)


def test_label_for():
    assert f0.label_for("draft_for_team_review", None) == f0.PILOT_LABEL
    assert f0.label_for("frozen_reviewed", 3) == f0.PILOT_LABEL
    assert f0.label_for("frozen_reviewed", None) == f0.DESCRIPTIVE_LABEL


def test_mean_se():
    m, se = f0.mean_se([1.0, 2.0, 3.0])
    assert m == pytest.approx(2.0) and se == pytest.approx(1.0 / np.sqrt(3))
    assert np.isnan(f0.mean_se([5.0])[1])


def test_overlap():
    a = np.array([[True, True], [False, False]])
    b = np.array([[True, False], [True, False]])
    o = f0.overlap(a, b)
    assert (o["n_a"], o["n_b"], o["both"]) == (2, 2, 1)
    assert o["share_of_a"] == pytest.approx(0.5) and o["jaccard"] == pytest.approx(1 / 3)


def _synthetic(n, strong):
    """n items with value `strong` on heads (0,0) and (1,1), 1.0 elsewhere, small noise."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        arr = np.ones((3, 4)) + rng.normal(0, 0.01, (3, 4))
        arr[0, 0] = arr[1, 1] = strong
        out.append(arr)
    return out


def test_crossfit_known_ratios():
    baseline = np.ones((3, 4))
    a = _synthetic(6, 10.0)
    cf = f0.crossfit(a, {"E1": _synthetic(5, 10.0), "E2": _synthetic(5, 1.0)}, baseline, 3.0)
    assert cf["n_heads"] == [2, 2]
    a_mean = np.mean(cf["A"])
    assert np.mean(cf["E1"]) / a_mean == pytest.approx(1.0, abs=0.01)
    assert np.mean(cf["E2"]) / a_mean == pytest.approx(0.1, abs=0.01)
    assert cf["baseline"] == pytest.approx(1.0)


def test_crossfit_needs_two_items_and_a_sink_head():
    baseline = np.ones((3, 4))
    assert f0.crossfit(_synthetic(1, 10.0), {}, baseline, 3.0) is None
    assert f0.crossfit(_synthetic(4, 1.0), {}, baseline, 3.0) is None  # no head above 3x


def test_provisional_read():
    assert f0.provisional_read(1.0, 0.2).startswith("token detector")
    assert f0.provisional_read(0.95, 0.9).startswith("general 4-byte")
    assert f0.provisional_read(0.3, 0.2).startswith("emoji-specific")
    assert f0.provisional_read(0.65, 0.2).startswith("in between")
    assert f0.provisional_read(float("nan"), 0.2).startswith("no call")
