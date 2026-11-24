"""Leg 6: the F0 control. Is the F0 9F sink about emoji, or about the byte token?

    python scripts/12_f0_control.py                       # primary model (Pythia)
    python scripts/12_f0_control.py replication           # Qwen
    python scripts/12_f0_control.py --limit 3             # smoke test: first 3 items per group
    python scripts/12_f0_control.py --slots first         # one slot per item, as script 06 does
    python scripts/12_f0_control.py --stimuli config/master_stimuli.yaml   # A and R from the master set

Groups compared:
  A   emoji whose lead token is [F0 9F]           team stimulus file, A_smp_pictographs
  E1  NOT emoji, SAME lead token [F0 9F]          config/f0_control_stimuli.yaml
  E2  NOT emoji, 4 bytes, DIFFERENT lead token    config/f0_control_stimuli.yaml
  R   ordinary words: the per-head baseline       team stimulus file, R_reference

For every item, the attention its LEAD token receives from the fixed 64-token far
block is measured per (layer, head), in each of the 16 carrier/position slots of the
structured probe, and averaged over slots. Then, per model:

  1  token check      lead token IDs per group (A and E1 must share one ID)
  2  identity check   the model's internal state at the lead token, A vs E1. When the
                      model reads [F0 9F] it has not seen the next bytes yet, so with
                      the same text before the slot the states must be identical.
                      Any A-vs-E1 difference must then come from the later reader tokens.
  3  sink flags       06's head-aware sink rule and 07's broad / narrow class, per group
  4  sink heads       heads above 3x the ordinary-word baseline (the rule of 08/09/10),
                      per group, and how much the groups' head sets overlap
  5  A's sink heads   attention E1 and E2 get on the heads that sink for A, as a share of
                      A's own. Cross-fitted: heads are picked on one half of A and A is
                      scored on the other half, so A gets no head start from the picking.

Descriptive only, never the pre-registered test. While the control set is a draft
(or with --limit), every output is labelled DEVELOPMENT PILOT – not confirmatory results.

Additive. It reuses, without modifying:
  - capture, span location and the fixed far block   (04 via 05)
  - 05's single-key received attention               (fragment_received)
  - 06's head-aware sink rule and lead-byte tagging  (is_sink, lead_byte_tag, format_bytes)
  - 07's broad / narrow metrics and class            (condition_metrics)
  - 08's sink-head rule                              (sink_head_mask, SINK_HEAD_RATIO)
Outputs go to outputs/f0_control/.
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from datetime import datetime

from _bootstrap import ROOT, load_config

import numpy as np
import torch
import transformers
import yaml
from c2.probes import Rejected, build_contexts
from c2.runner import _git_commit, sha256_file
from c2.tokaudit import token_bytes
from eas.span import SpanNotFound
from metric_validation import common

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _load_script(filename, name):
    """Import a numbered script by path (names starting with a digit cannot be imported)."""
    path = os.path.join(ROOT, "scripts", filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bs = _load_script("06_byte_structure.py", "byte_structure")
ns = _load_script("07_narrow_sink.py", "narrow_sink")
vd = _load_script("08_value_state_drain.py", "value_state_drain")
frag = bs.frag  # script 05


CONTROL_PATH = os.path.join(ROOT, "config", "f0_control_stimuli.yaml")
MASTER_PATH = os.path.join(ROOT, "config", "master_stimuli.yaml")
DEFAULT_PATH = os.path.join(ROOT, "config", "emoji_stimuli.yaml")
OUT_ROOT = os.path.join(ROOT, "outputs", "f0_control")

GROUP_A = "A_smp_pictographs"
GROUP_E1 = "E1_f0_9f_nonemoji"
GROUP_E2 = "E2_f0_other_4byte"
SHORT = {GROUP_A: "A", GROUP_E1: "E1", GROUP_E2: "E2"}
ORDER = [GROUP_A, GROUP_E1, GROUP_E2]

F0_9F = b"\xf0\x9f"
F0 = b"\xf0"

PILOT_LABEL = "DEVELOPMENT PILOT – not confirmatory results"
DESCRIPTIVE_LABEL = "DESCRIPTIVE – means and ratios only, not the pre-registered test"
# Internal states closer than this count as identical (the expected value is exactly 0).
IDENTITY_TOL = 1e-5

# PROVISIONAL bars for the one-line read in section 5. Not agreed by the team yet:
# confirm or replace them before the control set is frozen.
SIMILAR_BAR = 0.8   # E1 (or E2) gets at least 80% of A's attention on A's sink heads
LOW_BAR = 0.5       # E1 (or E2) gets under half of A's


# ---------------------------------------------------------------------------
# Stimuli
# ---------------------------------------------------------------------------
def load_yaml(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def default_stimuli_path():
    """The master set once it is merged, otherwise the leg-3 set."""
    return MASTER_PATH if os.path.exists(MASTER_PATH) else DEFAULT_PATH


def clean_lead(group, lead_bytes):
    """Does this item's lead token belong in its group's comparison?

    A and E1 must start with exactly the token [F0 9F] (in Qwen many emoji are one
    whole token, which has no [F0 9F] lead and is left out). E2 must start with a
    token that begins with F0 but is not [F0 9F].
    """
    if group in (GROUP_A, GROUP_E1):
        return lead_bytes == F0_9F
    return lead_bytes.startswith(F0) and not lead_bytes.startswith(F0_9F)


def label_for(control_status, limit):
    frozen = control_status.startswith(("frozen", "pre_registered"))
    return DESCRIPTIVE_LABEL if frozen and limit is None else PILOT_LABEL


# ---------------------------------------------------------------------------
# Measuring one item
# ---------------------------------------------------------------------------
def measure(model, tok, contexts, filler, first_only=False):
    """Received attention per span token, averaged over every slot that locates.

    Returns None if no slot works, else a dict with `frags` [n_tokens, L, H], the
    lead token's bytes / ID, and the first slot's input IDs (for the identity check).
    """
    per_slot = []
    lead_bytes = lead_id = first = None
    for ctx in contexts:
        try:
            cap, span, queries = frag.mmc.capture_and_locate(model, tok, ctx, filler)
        except (SpanNotFound, Rejected):
            continue
        frags = np.stack([frag.fragment_received(cap.attentions, k, queries).numpy()
                          for k in span.token_indices])
        if per_slot and frags.shape != per_slot[0].shape:
            continue  # a different split in this slot; the builder rules this out for E1/E2
        lead = span.token_indices[0]
        if first is None:
            lead_bytes = token_bytes(tok, cap.input_ids[lead])
            lead_id = cap.input_ids[lead]
            first = {"slot": f"{ctx.carrier}/{ctx.position}", "input_ids": list(cap.input_ids), "lead": lead}
        per_slot.append(frags)
        if first_only:
            break
    if not per_slot:
        return None
    return {"frags": np.mean(per_slot, axis=0), "n_slots": len(per_slot),
            "lead_bytes": lead_bytes, "lead_id": int(lead_id), "first": first}


def build_baseline(model, tok, contexts, reference_group, first_only):
    """06's ordinary-word baseline (every fragment of every ordinary word), over slots."""
    arrays = []
    for item in reference_group["items"]:
        if item.get("kind") != "ordinary_word":
            continue
        m = measure(model, tok, contexts, item["text"], first_only)
        if m is None:
            print(f"  [skip] reference {item['text']!r}: no slot located")
            continue
        arrays.extend(list(m["frags"]))
    if not arrays:
        raise SystemExit("no ordinary-word reference could be located")
    return np.mean(arrays, axis=0)


def score_item(group, item, m, baseline):
    lead_lh = m["frags"][0]
    is_sink, share = bs.is_sink(lead_lh, baseline)
    narrow = ns.condition_metrics(lead_lh, baseline)
    return {
        "group": SHORT[group], "char": item["char"], "name": item.get("name", ""),
        "codepoint": item.get("codepoint", ""), "lead_bytes": bs.format_bytes(m["lead_bytes"]),
        "lead_token_id": m["lead_id"], "n_tokens": int(m["frags"].shape[0]), "n_slots": m["n_slots"],
        "clean": clean_lead(group, m["lead_bytes"]),
        "frag_mean": float(lead_lh.mean()), "frag_max": float(lead_lh.max()),
        "sink_share": float(share), "is_sink": bool(is_sink),
        "peak_ratio": float(narrow["peak_ratio"]), "extreme_count": int(narrow["extreme_count"]),
        "cls": narrow["cls"], "lead_lh": lead_lh, "first": m["first"],
    }


def control_items(control, group):
    return control["groups"][group]["items"]


def emoji_items(stimuli):
    return [{"char": it["emoji"], "name": it.get("name", ""),
             "codepoint": " ".join(it.get("codepoints", []))} for it in stimuli["groups"][GROUP_A]["items"]]


# ---------------------------------------------------------------------------
# Analysis (pure numpy; covered by tests/test_f0_control.py)
# ---------------------------------------------------------------------------
def mean_se(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float("nan")
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


def overlap(mask_a, mask_b):
    """Counts and shares for two boolean [L, H] head sets."""
    a, b = int(mask_a.sum()), int(mask_b.sum())
    both = int((mask_a & mask_b).sum())
    union = int((mask_a | mask_b).sum())
    return {"n_a": a, "n_b": b, "both": both,
            "share_of_a": both / a if a else float("nan"),
            "jaccard": both / union if union else float("nan")}


def head_mean(lead_lh, mask):
    return float(lead_lh[mask].mean()) if mask.any() else float("nan")


def crossfit(a_arrays, other, baseline, ratio):
    """Score every group on A's sink heads without letting A pick its own heads.

    A is split into two halves by position (even / odd). For each half, the sink
    heads are picked from that half's mean; the OTHER half of A, and every item of
    the other groups, are scored on those heads. Each A item is scored once (in the
    fold where it was held out); other items average their two fold scores.
    Returns {"A": [...], name: [...], "baseline": float, "n_heads": [fold0, fold1]}
    or None when A has fewer than two items or a fold finds no sink head.
    """
    if len(a_arrays) < 2:
        return None
    halves = [list(range(0, len(a_arrays), 2)), list(range(1, len(a_arrays), 2))]
    a_scores = [float("nan")] * len(a_arrays)
    other_scores = {name: np.zeros(len(arrs)) for name, arrs in other.items()}
    base_scores, n_heads = [], []
    for fold in (0, 1):
        pick, held = halves[fold], halves[1 - fold]
        mask = vd.sink_head_mask(np.mean([a_arrays[i] for i in pick], axis=0), baseline, ratio)
        n_heads.append(int(mask.sum()))
        if not mask.any():
            return None
        for i in held:
            a_scores[i] = head_mean(a_arrays[i], mask)
        for name, arrs in other.items():
            for j, arr in enumerate(arrs):
                other_scores[name][j] += head_mean(arr, mask) / 2
        base_scores.append(head_mean(baseline, mask))
    out = {"A": a_scores, "baseline": float(np.mean(base_scores)), "n_heads": n_heads}
    for name, scores in other_scores.items():
        out[name] = list(scores)
    return out


def provisional_read(r_e1, r_e2):
    """One line mapping the two ratios to the outcome table (PROVISIONAL bars)."""
    if np.isnan(r_e1) or np.isnan(r_e2):
        return "no call: a ratio could not be computed"
    if r_e1 < LOW_BAR:
        return "emoji-specific: E1 gets much less than A (so the difference comes from the reader tokens)"
    if r_e1 >= SIMILAR_BAR and r_e2 < LOW_BAR:
        return "token detector: [F0 9F] draws the attention whether or not an emoji follows; other 4-byte starts do not"
    if r_e1 >= SIMILAR_BAR and r_e2 >= SIMILAR_BAR:
        return "general 4-byte detector: E1 and E2 both get about as much as A"
    return "in between: no clear call at these bars"


# ---------------------------------------------------------------------------
# Identity check
# ---------------------------------------------------------------------------
def lead_states(model, first):
    """Hidden states at the lead token, all layers: [n_layers + 1, d_model]."""
    ids = torch.tensor([first["input_ids"]])
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    return torch.stack([h[0, first["lead"]] for h in out.hidden_states])


def identity_check(model, rows):
    """Max |state - reference state| at the lead token, for clean A and E1 items
    measured in the same first slot as the reference (the first clean A item)."""
    pool = [r for r in rows if r["group"] in ("A", "E1") and r["clean"]]
    refs = [r for r in pool if r["group"] == "A"]
    if not refs:
        return None
    ref = refs[0]
    ref_state = lead_states(model, ref["first"])
    diffs = {"A": [], "E1": []}
    for r in pool:
        if r is ref or r["first"]["slot"] != ref["first"]["slot"] or r["first"]["lead"] != ref["first"]["lead"]:
            continue
        diffs[r["group"]].append(float((lead_states(model, r["first"]) - ref_state).abs().max()))
    return {"slot": ref["first"]["slot"], "reference": ref["char"], "diffs": diffs}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
class Report:
    """Print and keep every line, so summary.txt is exactly what was shown."""

    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)


def fmt(value, digits=3):
    return "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}f}"


def run(model_key, stimuli_path, slots, limit, device):
    cfg = load_config()
    if model_key not in cfg:
        raise SystemExit(f"unknown model key {model_key!r}; use 'primary' or 'replication'")
    name, revision = cfg[model_key]["name"], cfg[model_key].get("revision")
    tag = {"primary": "pythia410m", "replication": "qwen25_05b"}.get(model_key, model_key)

    stimuli = load_yaml(stimuli_path)
    control = load_yaml(CONTROL_PATH)
    status = control["pre_registration"]["status"]
    label = label_for(status, limit)
    say = Report()
    t0 = time.time()

    model, tok = frag.mmc.load_model(name, revision=revision, device=device)
    contexts = build_contexts(tok)
    first_only = slots == "first"

    say(label)
    say(DESCRIPTIVE_LABEL if label != DESCRIPTIVE_LABEL else "")
    say(f"\nF0 CONTROL  —  {name}")
    say(f"A and R from {os.path.relpath(stimuli_path, ROOT)}; E1 and E2 from "
        f"{os.path.relpath(CONTROL_PATH, ROOT)} ({control['pre_registration']['version_id']}, {status})")
    say(f"lead token only; fixed 64-token far block; {'first slot only' if first_only else 'mean over all 16 slots'}"
        + (f"; first {limit} items per group" if limit else ""))

    baseline = build_baseline(model, tok, contexts, stimuli["groups"]["R_reference"], first_only)

    sources = {GROUP_A: emoji_items(stimuli), GROUP_E1: control_items(control, GROUP_E1),
               GROUP_E2: control_items(control, GROUP_E2)}
    rows = []
    for group in ORDER:
        items = sources[group][:limit] if limit else sources[group]
        for item in items:
            m = measure(model, tok, contexts, item["char"], first_only)
            if m is None:
                say(f"  [skip] {SHORT[group]} {item['char']} ({item.get('name', '')}): no slot located")
                continue
            rows.append(score_item(group, item, m, baseline))
    by = {g: [r for r in rows if r["group"] == g and r["clean"]] for g in ("A", "E1", "E2")}

    # 1 token check
    say("\n1  TOKEN CHECK  (lead token of each item, as the tokenizer actually splits it)")
    for g in ("A", "E1", "E2"):
        all_g = [r for r in rows if r["group"] == g]
        ids = sorted({r["lead_token_id"] for r in by[g]})
        leads = sorted({r["lead_bytes"] for r in by[g]})
        counts = sorted({r["n_tokens"] for r in by[g]})
        say(f"   {g:<3} measured {len(all_g):>3}  in comparison {len(by[g]):>3}   lead bytes {leads}  "
            f"lead token IDs {ids if len(ids) <= 4 else str(len(ids)) + ' distinct'}  tokens {counts}")
    shared = {r["lead_token_id"] for r in by["A"]} & {r["lead_token_id"] for r in by["E1"]}
    say(f"   A and E1 share lead token ID: {sorted(shared) if shared else 'NO (check the tokenizer)'}")
    left_out = [r for r in rows if not r["clean"]]
    if left_out:
        say(f"   left out (lead token does not fit the group): "
            + ", ".join(f"{r['group']} {r['char']} [{r['lead_bytes']}]" for r in left_out))

    # 2 identity check
    say("\n2  IDENTITY CHECK  (internal state at the lead token, vs the first A item, all layers)")
    ident = identity_check(model, rows)
    if ident is None:
        say("   skipped: no A item with a [F0 9F] lead in this model")
    else:
        for g in ("A", "E1"):
            d = ident["diffs"][g]
            if d:
                verdict = "identical" if max(d) <= IDENTITY_TOL else "NOT identical"
                say(f"   {g:<3} {len(d):>3} items in slot {ident['slot']}: max difference {max(d):.2e}  -> {verdict}")
        say("   identical means the [F0 9F] token looks the same to every head whatever follows;")
        say("   any A-vs-E1 difference below can only come from the later reader tokens")

    # 3 sink flags
    say("\n3  SINK FLAGS PER GROUP  (lead token vs the ordinary-word baseline)")
    say(f"   {'grp':<4}{'n':>4}  {'frag_max mean ± SE':>22}  {'06 sink share':>13}  "
        f"{'07 peak_ratio':>13}  {'broad_share':>11}  {'extreme':>7}  class")
    group_means = {}
    for g in ("A", "E1", "E2"):
        if not by[g]:
            say(f"   {g:<4}{0:>4}  (none)")
            continue
        mean_lh = np.mean([r["lead_lh"] for r in by[g]], axis=0)
        group_means[g] = mean_lh
        m, se = mean_se([r["frag_max"] for r in by[g]])
        flagged = sum(r["is_sink"] for r in by[g]) / len(by[g])
        nm = ns.condition_metrics(mean_lh, baseline)
        say(f"   {g:<4}{len(by[g]):>4}  {fmt(m):>12} ± {fmt(se):<7}  {flagged:>13.3f}  "
            f"{nm['peak_ratio']:>13.2f}  {nm['broad_share']:>11.3f}  {nm['extreme_count']:>7}  {nm['cls']}")
    say(f"   06 sink share = share of items whose lead beats the baseline on > {common.PASS_BAR:.0%} of heads;")
    say(f"   07 numbers are for the group's mean vector (extreme = heads above {ns.EXTREME_K:g}x the baseline)")

    # 4 sink heads
    say(f"\n4  SINK HEADS  (heads where the group's mean lead attention > {vd.SINK_HEAD_RATIO:g}x the baseline)")
    masks = {g: vd.sink_head_mask(v, baseline, vd.SINK_HEAD_RATIO) for g, v in group_means.items()}
    for g, mask in masks.items():
        say(f"   {g:<3} {int(mask.sum()):>4} of {mask.size} heads")
    for x, y in (("A", "E1"), ("A", "E2"), ("E1", "E2")):
        if x in masks and y in masks:
            o = overlap(masks[x], masks[y])
            say(f"   {x} vs {y}: {o['both']} heads in both; {fmt(o['share_of_a'])} of {x}'s sink heads are also {y}'s;"
                f" Jaccard {fmt(o['jaccard'])}")

    # 5 cross-fitted ratio on A's sink heads
    say("\n5  ON A'S SINK HEADS  (cross-fitted; mean lead attention per head, ± SE across items)")
    cf = None
    if by["A"]:
        cf = crossfit([r["lead_lh"] for r in by["A"]],
                      {g: [r["lead_lh"] for r in by[g]] for g in ("E1", "E2") if by[g]},
                      baseline, vd.SINK_HEAD_RATIO)
    if cf is None:
        say("   skipped: needs at least 2 A items with a [F0 9F] lead and a sink head in each half")
        r_e1 = r_e2 = float("nan")
    else:
        a_m, a_se = mean_se(cf["A"])
        say(f"   sink heads picked per half of A: {cf['n_heads']}")
        say(f"   A   {fmt(a_m, 4)} ± {fmt(a_se, 4)}")
        ratios = {}
        for g in ("E1", "E2"):
            if g in cf:
                g_m, g_se = mean_se(cf[g])
                ratios[g] = g_m / a_m if a_m else float("nan")
                say(f"   {g:<3} {fmt(g_m, 4)} ± {fmt(g_se, 4)}   = {fmt(ratios[g], 2)} of A")
        say(f"   ordinary-word baseline on the same heads: {fmt(cf['baseline'], 4)}")
        r_e1, r_e2 = ratios.get("E1", float("nan")), ratios.get("E2", float("nan"))
        for i, r in enumerate(by["A"]):
            r["crossfit_score"] = cf["A"][i]
        for g in ("E1", "E2"):
            for i, r in enumerate(by[g]):
                r["crossfit_score"] = cf[g][i] if g in cf else float("nan")
    say(f"\n   PROVISIONAL READ (bars not agreed: similar >= {SIMILAR_BAR}, low < {LOW_BAR} of A):")
    say(f"   {provisional_read(r_e1, r_e2)}")

    # outputs
    os.makedirs(OUT_ROOT, exist_ok=True)
    run_dir = os.path.join(OUT_ROOT, f"{tag}_{slots}{'_limit' + str(limit) if limit else ''}_"
                                     f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir)
    fields = ["group", "char", "name", "codepoint", "clean", "lead_bytes", "lead_token_id", "n_tokens", "n_slots",
              "frag_mean", "frag_max", "sink_share", "is_sink", "peak_ratio", "extreme_count", "cls", "crossfit_score"]
    with open(os.path.join(run_dir, "items.csv"), "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    np.savez_compressed(os.path.join(run_dir, "heads.npz"), baseline=baseline,
                        **{f"lead_{g}": np.array([r["lead_lh"] for r in by[g]]) for g in by if by[g]},
                        **{f"sink_heads_{g}": m for g, m in masks.items()})
    say(f"\nwrote {os.path.relpath(run_dir, ROOT)}  ({time.time() - t0:.0f} s)")
    with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(say.lines) + "\n")
    manifest = {
        "label": label, "model": name, "revision": revision, "model_key": model_key, "device": device,
        "slots": slots, "limit": limit,
        "stimuli": {"file": os.path.relpath(stimuli_path, ROOT), "sha256": sha256_file(stimuli_path)},
        "control": {"file": os.path.relpath(CONTROL_PATH, ROOT), "sha256": sha256_file(CONTROL_PATH),
                    "version_id": control["pre_registration"]["version_id"], "status": status,
                    "content_sha256": control["pre_registration"]["content_sha256"]},
        "bars": {"sink_head_ratio": vd.SINK_HEAD_RATIO, "pass_bar": common.PASS_BAR, "extreme_k": ns.EXTREME_K,
                 "similar_bar_provisional": SIMILAR_BAR, "low_bar_provisional": LOW_BAR},
        "n_items": {g: len(by[g]) for g in by}, "identity": ident,
        "repo_commit": _git_commit(), "seconds": round(time.time() - t0, 1),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__},
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return run_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", default="primary", help="primary (Pythia) | replication (Qwen)")
    ap.add_argument("--stimuli", default=None, help="team file for groups A and R (default: master set if present)")
    ap.add_argument("--slots", default="all", choices=["all", "first"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N items of A, E1 and E2")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    path = os.path.join(ROOT, a.stimuli) if a.stimuli and not os.path.isabs(a.stimuli) else a.stimuli
    run(a.model, path or default_stimuli_path(), a.slots, a.limit, a.device)


if __name__ == "__main__":
    main()
