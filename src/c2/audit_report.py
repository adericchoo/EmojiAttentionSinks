"""Stage 1: tokenisation audit for both models -> CSVs, summary table, figures.

    python scripts/c2.py audit
"""
from __future__ import annotations

import collections
import json
import os
from typing import Dict, List

from . import OUT_DIR
from .config import load_tokenizer, model_entry, model_tag
from .inventory import EXPECTED_SHA256, file_sha256, load_emoji
from .tokaudit import (AUDIT_CARRIER, LINE_CARRIER, POOL_RANGE, audit_emoji, audit_string,
                       candidate_pool, summarise, write_csv)

MODELS = ["pythia", "qwen"]
DISPLAY = {"pythia410m": "Pythia-410M", "qwen25_05b": "Qwen2.5-0.5B"}

# Numbers reported in the submitted proposal (to reproduce as a check).
PROPOSAL = {
    "pythia410m": {"n_boundary": 2, "n_multi": 3771, "n_kept": 3771, "pct_multi": 100.0, "median": 6, "max": 27},
    "qwen25_05b": {"n_boundary": 2, "n_multi": 2895, "n_kept": 3771, "pct_multi": 76.77, "median": 2, "max": 19},
    "pool": 1414,
}
PROPOSAL_EXCLUDED = ("#️⃣", "*️⃣")  # the # and * keycaps

# "How emoji get chopped" examples: (label, string, kind)
CHOP_EXAMPLES = [
    ("thumbs up: medium skin tone", "\U0001F44D\U0001F3FD", "emoji (skin-tone modifier)"),
    ("family: man, woman, girl, boy", "\U0001F468‍\U0001F469‍\U0001F467‍\U0001F466", "emoji (ZWJ sequence)"),
    ("flag: Singapore", "\U0001F1F8\U0001F1EC", "emoji (flag)"),
    ("CJK Ext. A U+3400", "㐀", "rare CJK character"),
    ("hex string", "3f9a", "hex / ASCII string"),
    ("English word", "kaleidoscope", "English word"),
]

ANSI_FRAG = "\033[1;38;5;208m"  # orange = partial UTF-8 byte fragment
ANSI_WHOLE = "\033[1;34m"       # blue = whole character(s)
ANSI_OFF = "\033[0m"


def _fmt_stats(s: dict) -> List[str]:
    return [
        f"{s['n_boundary']}",
        f"{s['n_multi']:,}/{s['n_kept']:,} ({s['pct_multi']:.2f}%)",
        f"{s['median']:g} / {s['max']}",
    ]


def summary_table(rows_by_tag: Dict[str, List[dict]], pools: Dict[str, int]) -> str:
    tags = list(rows_by_tag)
    sha = file_sha256()
    lines = [
        "TOKENISATION AUDIT  -  Unicode Emoji 15.1, fully-qualified sequences",
        f"  inventory: {len(next(iter(rows_by_tag.values()))):,} emoji, SHA-256 {sha[:8]}…{sha[-7:]}"
        f"  ({'matches' if sha == EXPECTED_SHA256 else 'DOES NOT MATCH'} proposal)",
        "  unit counted: MODEL TOKENS (span = tokens whose char offsets overlap the emoji)",
        "",
    ]
    labels = ["boundary exclusions", "multi-token (excl. boundary)", "median / max tokens"]
    blocks = [
        ("A. Protocol audit carrier (space before emoji)", "", (), AUDIT_CARRIER),
        ("B. Probe-slot context (emoji starts a line = tokenised alone)", "_line", (), LINE_CARRIER),
        ("B'. As B, also dropping the # and * keycaps like the proposal did", "_line", PROPOSAL_EXCLUDED, None),
    ]
    head = f"  {'':32}" + "".join(f"{DISPLAY[t]:>26}" for t in tags)
    for title, suffix, excl, carrier in blocks:
        lines.append(title)
        if carrier:
            lines.append(f"  carrier: {carrier!r}")
        lines.append(head)
        stats = [_fmt_stats(summarise(rows_by_tag[t], suffix, excl)) for t in tags]
        for i, lab in enumerate(labels):
            lines.append(f"  {lab:32}" + "".join(f"{st[i]:>26}" for st in stats))
        lines.append("")
    lines.append("Proposal-reported numbers")
    lines.append(head)
    prop = [_fmt_stats(PROPOSAL[t]) for t in tags]
    for i, lab in enumerate(labels):
        lines.append(f"  {lab:32}" + "".join(f"{st[i]:>26}" for st in prop))
    lines.append("")
    lo, hi = POOL_RANGE
    lines.append(f"Candidate pool ({lo}-{hi} tokens in BOTH models)")
    lines.append(f"  A. audit carrier            : {pools['A']:,}")
    lines.append(f"  B. probe-slot context (USED): {pools['B']:,}")
    lines.append(f"  proposal                    : {PROPOSAL['pool']:,}")
    return "\n".join(lines)


def boundary_explainer(rows_by_tag: Dict[str, List[dict]]) -> str:
    out = ["", "Why the numbers differ from the proposal:"]
    out.append("  * Proposal = block B': emoji tokenised ALONE (no leading space), then # and * removed.")
    out.append("  * Block A puts a space before the emoji. Qwen has many emoji as single tokens, but only")
    out.append("    without a leading space (' 😀' is fine, ' 👍🏽' -> '␣‹F0 9F 91›' '‹8D›' '🏽' instead of 2).")
    qa = collections.Counter(r["n_tokens"] - r["n_tokens_line"] for r in rows_by_tag["qwen25_05b"])
    out.append(f"    Qwen tokens(A) - tokens(B): {dict(sorted(qa.items()))}  (emoji count per difference)")
    out.append("  * # and * keycaps: NO counting rule we tested flags them as boundary-crossing, in either model")
    out.append("    or tokenizers 0.19.1 / 0.23.3. They begin with an ASCII punctuation character, which may be")
    out.append("    why they were excluded by hand. Either way they have 7 (Pythia) / 5 (Qwen) tokens, so they")
    out.append("    are outside the 2-6 pool and the pool size does not depend on them.")
    for tag, rows in rows_by_tag.items():
        for r in rows:
            if r["emoji"] in PROPOSAL_EXCLUDED:
                out.append(f"    {DISPLAY[tag]:13} {r['emoji']}  {r['token_pieces_line']}")
    return "\n".join(out)


def chopped_table(toks: Dict[str, object], color: bool = True) -> str:
    """Per example: token pieces in each model, fragments marked."""
    out = ["HOW STRINGS GET CHOPPED  (probe-slot context; " +
           (f"{ANSI_FRAG}orange{ANSI_OFF} = partial UTF-8 byte fragment, {ANSI_WHOLE}blue{ANSI_OFF} = whole character(s))"
            if color else "[x] = partial UTF-8 byte fragment, plain = whole character(s))"), ""]
    for label, s, kind in CHOP_EXAMPLES:
        out.append(f"{s}   {label}  -  {kind}   ({len(s)} code point(s), {len(s.encode('utf-8'))} UTF-8 bytes)")
        for tag, tok in toks.items():
            r = audit_string(tok, s, LINE_CARRIER)
            cells = []
            for p, frag in zip(r.pieces, r.byte_frag):
                if color:
                    cells.append(f"{ANSI_FRAG if frag else ANSI_WHOLE}{p}{ANSI_OFF}")
                else:
                    cells.append(f"[{p}]" if frag else p)
            out.append(f"    {DISPLAY[tag]:13} {r.n_span} tokens, {r.n_frag} byte fragments :  " + "  ".join(cells))
        out.append("")
    return "\n".join(out)


def plot_histogram(rows_by_tag: Dict[str, List[dict]], path: str) -> None:
    import numpy as np
    from . import plotstyle
    plt = plotstyle.plt
    plotstyle.apply()
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    mx = max(max(r["n_tokens"], r["n_tokens_line"]) for rows in rows_by_tag.values() for r in rows)
    xs = np.arange(1, mx + 1)
    w = 0.4
    for ax, (tag, rows) in zip(axes, rows_by_tag.items()):
        for j, (suffix, lab) in enumerate([("", "audit carrier (space before emoji)"),
                                           ("_line", "probe-slot context (used for pool)")]):
            c = collections.Counter(r["n_tokens" + suffix] for r in rows if not r["boundary_crossing" + suffix])
            ax.bar(xs + (j - 0.5) * w, [c.get(x, 0) for x in xs], width=w - 0.05,
                   color=plotstyle.SERIES[j], label=lab, zorder=2)
        lo, hi = POOL_RANGE
        ax.axvspan(lo - 0.5, hi + 0.5, color=plotstyle.GRID, alpha=0.5, zorder=0, lw=0,
                   label=f"shaded: {lo}-{hi} token pool")
        ax.set_title(DISPLAY[tag], loc="left")
        ax.set_ylabel("number of emoji")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper right")
    axes[1].set_xlabel("model tokens per emoji (n = 3,773 fully-qualified emoji)")
    axes[1].set_xticks(xs)
    fig.suptitle("Emoji fragmentation: tokens per emoji, by model", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _png_safe(p: str) -> str:
    """Matplotlib's fonts cannot draw emoji/CJK glyphs, so show those as U+XXXX."""
    return "".join(ch if ord(ch) < 0x2E80 else f"U+{ord(ch):04X}" for ch in p)


def plot_chopped(toks: Dict[str, object], path: str) -> None:
    from matplotlib.patches import FancyBboxPatch
    from . import plotstyle
    plt = plotstyle.plt
    plotstyle.apply()

    def box_text(p, frag):
        # colour already says "byte fragment", so drop the ‹› brackets inside boxes
        return _png_safe(p.replace("‹", "").replace("›", "") if frag else p)

    def box_w(t):
        return 0.085 * len(t) + 0.2

    gap, x0 = 0.05, 5.0
    rows = []
    for label, s, kind in CHOP_EXAMPLES:
        per_model = [(tag, audit_string(tok, s, LINE_CARRIER)) for tag, tok in toks.items()]
        rows.append((label, s, kind, per_model))
    width = max(x0 + sum(box_w(box_text(p, f)) + gap for p, f in zip(r.pieces, r.byte_frag))
                for *_, pm in rows for _, r in pm) + 0.3
    width = max(width, 13)
    n = len(rows)
    fig, ax = plt.subplots(figsize=(width, 0.95 * n + 1.5))
    ax.set_xlim(0, width)
    ax.set_ylim(-0.6, n * 2 + 1.0)
    ax.axis("off")
    for i, (label, s, kind, per_model) in enumerate(rows):
        y0 = (n - 1 - i) * 2 + 1
        cps = " ".join(f"U+{ord(c):04X}" for c in s) if not s.isascii() else repr(s)
        ax.text(0, y0 + 0.45, label, fontsize=10, fontweight="bold", color=plotstyle.INK, va="center")
        ax.text(0, y0 - 0.05, f"{kind}  ·  {len(s.encode('utf-8'))} UTF-8 bytes", fontsize=8, color=plotstyle.INK_2, va="center")
        ax.text(0, y0 - 0.45, cps if len(cps) < 34 else cps[:31] + "…", fontsize=7, color=plotstyle.MUTED, va="center")
        for k, (tag, r) in enumerate(per_model):
            y = y0 + 0.4 - k * 0.8
            ax.text(3.3, y, f"{DISPLAY[tag]}  ({r.n_span} tok)", fontsize=8, color=plotstyle.INK_2, va="center")
            x = x0
            for p, frag in zip(r.pieces, r.byte_frag):
                txt = box_text(p, frag)
                w = box_w(txt)
                col = plotstyle.SERIES[1] if frag else plotstyle.SERIES[0]
                ax.add_patch(FancyBboxPatch((x, y - 0.25), w, 0.5, boxstyle="round,pad=0,rounding_size=0.08",
                                            fc=col, ec=plotstyle.SURFACE, lw=2))
                ax.text(x + w / 2, y, txt, fontsize=7.5, color="white", ha="center", va="center", family="DejaVu Sans Mono")
                x += w + gap
    top = n * 2 + 0.75
    ax.text(0, top, "How strings get chopped into model tokens  (probe-slot context: the string starts a line)",
            fontsize=12, fontweight="bold", va="center", color=plotstyle.INK)
    leg_y = top - 0.5
    for j, (col, lab) in enumerate([(plotstyle.SERIES[0], "whole character(s)"),
                                    (plotstyle.SERIES[1], "partial UTF-8 byte fragment (hex bytes shown)")]):
        lx = j * 2.6
        ax.add_patch(FancyBboxPatch((lx, leg_y - 0.13), 0.26, 0.26, boxstyle="round,pad=0", fc=col, ec="none"))
        ax.text(lx + 0.38, leg_y, lab, fontsize=8, va="center", color=plotstyle.INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(out_dir: str = os.path.join(OUT_DIR, "audit")) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    emoji = load_emoji()
    rows_by_tag, toks = {}, {}
    for m in MODELS:
        tag = model_tag(m)
        toks[tag] = load_tokenizer(m)
        rows_by_tag[tag] = audit_emoji(toks[tag], model_entry(m)[1], emoji)
        write_csv(rows_by_tag[tag], os.path.join(out_dir, f"audit_{tag}.csv"))

    pool_a = candidate_pool(rows_by_tag, suffix="")
    pool_b = candidate_pool(rows_by_tag, suffix="_line")
    with open(os.path.join(out_dir, "candidate_pool.json"), "w", encoding="utf-8") as f:
        json.dump({"rule": "2-6 model tokens in both models, probe-slot (line-initial) context",
                   "n": len(pool_b), "emoji": pool_b}, f, ensure_ascii=False, indent=0)

    table = summary_table(rows_by_tag, {"A": len(pool_a), "B": len(pool_b)}) + "\n" + boundary_explainer(rows_by_tag)
    with open(os.path.join(out_dir, "audit_summary.txt"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(table)

    chop = chopped_table(toks, color=True)
    with open(os.path.join(out_dir, "chopped_examples.txt"), "w", encoding="utf-8") as f:
        f.write(chopped_table(toks, color=False) + "\n")
    print("\n" + chop)

    plot_histogram(rows_by_tag, os.path.join(out_dir, "audit_token_hist.png"))
    plot_chopped(toks, os.path.join(out_dir, "chopped_examples.png"))
    print(f"\nwrote {out_dir}/  (audit_*.csv, audit_summary.txt, candidate_pool.json, "
          "chopped_examples.txt/.png, audit_token_hist.png)")
    return {"rows": rows_by_tag, "pool": pool_b}
