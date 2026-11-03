"""Stage 3: generate matched probes for one model -> JSONL, exclusion log, coverage.

    python scripts/c2.py probes --model pythia
"""
from __future__ import annotations

import collections
import csv
import json
import os

from . import OUT_DIR
from .audit_report import DISPLAY
from .config import load_tokenizer, model_entry, model_tag
from .controls import MATCHED
from .probes import (N_CONFIRM, N_DEV, SEED, build_contexts, coverage, generate, sample_emoji,
                     write_exclusions, write_jsonl)
from .tokaudit import read_csv

PROBE_DIR = os.path.join(OUT_DIR, "probes")


def load_eligible(tag: str):
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(os.path.join(OUT_DIR, "controls", f"controls_{tag}.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["eligible"] == "True":
                out[r["condition"]][int(r["n_tokens"])].append(r["filler"])
    return out


def emoji_sample() -> dict:
    """The dev/confirm emoji, shared by both models. Stratified on the PRIMARY
    model's token count x sequence type. Created once, then reused."""
    path = os.path.join(PROBE_DIR, "emoji_sample.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    pool = json.load(open(os.path.join(OUT_DIR, "audit", "candidate_pool.json"), encoding="utf-8"))["emoji"]
    prim = {r["emoji"]: r for r in read_csv(os.path.join(OUT_DIR, "audit", f"audit_{model_tag('primary')}.csv"))}
    split = sample_emoji(pool, {e: (prim[e]["seq_type"], prim[e]["n_tokens_line"]) for e in pool})
    out = {"seed": SEED, "n_dev": N_DEV, "n_confirm": N_CONFIRM,
           "strata": "sequence type x Pythia-410M token count (probe-slot context)", **split}
    os.makedirs(PROBE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


def main(model: str) -> str:
    os.makedirs(PROBE_DIR, exist_ok=True)
    tag = model_tag(model)
    name = model_entry(model)[1]
    tok = load_tokenizer(model)
    rows = read_csv(os.path.join(OUT_DIR, "audit", f"audit_{tag}.csv"))
    meta = {r["emoji"]: {"index": i, "name": r["name"], "seq_type": r["seq_type"]} for i, r in enumerate(rows)}
    sample = emoji_sample()
    split = {"dev": sample["dev"], "confirm": sample["confirm"]}

    contexts = build_contexts(tok)
    probes, excl = generate(tok, name, tag, split, meta, load_eligible(tag), contexts)

    p_path = os.path.join(PROBE_DIR, f"probes_{tag}.jsonl")
    x_path = os.path.join(PROBE_DIR, f"exclusions_{tag}.csv")
    write_jsonl(probes, p_path)
    write_exclusions(excl, x_path)

    n_ctx = len(contexts)
    attempted = {s: len(split[s]) * n_ctx for s in split}
    lines = [f"MATCHING / COVERAGE REPORT  -  {DISPLAY[tag]}", ""]
    lines.append(f"  emoji: {len(split['dev'])} dev + {len(split['confirm'])} confirmatory "
                 f"(seed {SEED}), x {n_ctx // 2} carriers x 2 positions = {sum(attempted.values())} candidate sets")
    lines.append("  dev emoji: " + "  ".join(f"{e} ({meta[e]['name']})" for e in split["dev"]))
    starts = collections.Counter((c.position, c.start) for c in contexts)
    lines.append("  probe start index per position: " +
                 ", ".join(f"{p}: token {s} in {n} carriers" for (p, s), n in sorted(starts.items())))
    lens = [p["seq_len"] for p in probes]
    lines.append(f"  seq_len: min {min(lens)}, max {max(lens)}  (limit 256)")
    lines.append("")
    lines.append("  matched sets achieved per condition (achieved / attempted)")
    lines.append(coverage(probes, excl, attempted))
    span_counts = collections.Counter(p["n_span_tokens"] for p in probes if p["condition"] == "e")
    lines.append("")
    lines.append("  emoji span token counts across sets: " + ", ".join(f"{k} tok: {v}" for k, v in sorted(span_counts.items())))
    reuse = {c: len({p["filler"] for p in probes if p["condition"] == c}) for c in MATCHED}
    lines.append("  distinct control fillers used: " + ", ".join(f"{c}: {n}" for c, n in reuse.items()))
    text = "\n".join(lines)
    with open(os.path.join(PROBE_DIR, f"coverage_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nwrote {p_path}  ({len(probes)} probes)\n      {x_path}  ({len(excl)} rows)")
    return p_path
