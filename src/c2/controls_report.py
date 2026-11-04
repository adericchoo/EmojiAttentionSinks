"""Stage 2: screen every control filler per model -> CSVs + matchability table.

    python scripts/c2.py controls
"""
from __future__ import annotations

import collections
import json
import os

from . import OUT_DIR
from .audit_report import DISPLAY, MODELS
from .config import load_tokenizer, model_tag
from .controls import COND_NAMES, WORDS_PATH, build_word_list, candidates, screen
from .tokaudit import read_csv, write_csv

COUNTS = range(2, 7)


def main(out_dir: str = os.path.join(OUT_DIR, "controls")) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(WORDS_PATH):
        build_word_list()
    pool = json.load(open(os.path.join(OUT_DIR, "audit", "candidate_pool.json"), encoding="utf-8"))["emoji"]
    cands = candidates()
    lines = ["CONTROL FILLERS  -  screened in the probe-slot context, per model", ""]
    for m in MODELS:
        tag = model_tag(m)
        tok = load_tokenizer(m)
        rows = screen(tok, cands)
        write_csv(rows, os.path.join(out_dir, f"controls_{tag}.csv"))
        audit = {r["emoji"]: r for r in read_csv(os.path.join(OUT_DIR, "audit", f"audit_{tag}.csv"))}
        pool_counts = collections.Counter(audit[e]["n_tokens_line"] for e in pool)

        lines.append(f"{DISPLAY[tag]}")
        lines.append(f"  {'condition':30}{'tried':>7}{'eligible':>10}   eligible fillers by token count")
        lines.append(f"  {'':30}{'':>7}{'':>10}   " + "".join(f"{c:>7}" for c in COUNTS))
        lines.append(f"  {'e  pool emoji (to be matched)':30}{len(pool):>7}{len(pool):>10}   "
                     + "".join(f"{pool_counts.get(c, 0):>7}" for c in COUNTS))
        for cond in ["w", "s", "x", "p", "r"]:
            cr = [r for r in rows if r["condition"] == cond]
            ok = [r for r in cr if r["eligible"]]
            by_n = collections.Counter(r["n_tokens"] for r in ok)
            cells = "".join(f"{by_n.get(c, 0):>7}" for c in COUNTS) if cond in "wsx" else f"   (all 1 token)"
            lines.append(f"  {cond + '  ' + COND_NAMES[cond]:30}{len(cr):>7}{len(ok):>10}   {cells}")
        reasons = collections.Counter((r["condition"], r["reason"]) for r in rows if not r["eligible"])
        lines.append("  rejected fillers: " + ", ".join(f"{c}:{why} x{n}" for (c, why), n in sorted(reasons.items())))
        s_ok = [r for r in rows if r["condition"] == "s" and r["eligible"]]
        frac = sum(r["n_byte_fragments"] == r["n_tokens"] for r in s_ok) / max(1, len(s_ok))
        lines.append(f"  s check: every eligible rare-char filler has >= 1 byte-fragment token "
                     f"({100 * frac:.0f}% are byte fragments in EVERY token)")
        lines.append("")
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "controls_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"wrote {out_dir}/  (controls_*.csv, controls_summary.txt)")
