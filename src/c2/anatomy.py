"""Probe anatomy: one matched set, regions marked, conditions stacked.

    python scripts/c2.py anatomy --model pythia [--set <matched_set_id>]
"""
from __future__ import annotations

import hashlib
import os
from typing import List

from . import OUT_DIR
from .audit_report import DISPLAY
from .config import load_tokenizer, model_tag
from .probes import read_jsonl
from .tokaudit import readable_piece

ORDER = ["e", "w", "s", "x", "p", "r"]
LABEL = {"e": "e emoji", "w": "w word", "s": "s rare", "x": "x hex", "p": "p punct", "r": "r ref"}
C = {"prefix": "\033[38;5;245m", "span": "\033[1;38;5;208m", "buffer": "\033[38;5;36m",
     "q": "\033[38;5;33m", "off": "\033[0m"}


def _piece(tok, tid: int) -> str:
    return readable_piece(tok, tid).replace("\n", "⏎")


def _hash(ids: List[int]) -> str:
    return hashlib.sha1(" ".join(map(str, ids)).encode()).hexdigest()[:8]


def regions(p: dict):
    s0, s1 = p["token_span"]
    b0, b1 = p["buffer_span"]
    q0, q1 = p["query_span"]
    return s0, s1, b0, b1, q0, q1


def full_view(tok, p: dict, color: bool) -> str:
    s0, s1, b0, b1, q0, q1 = regions(p)
    ids = p["input_ids"]
    out = []
    for i, t in enumerate(ids):
        piece = _piece(tok, t)
        if i == s0:
            out.append(f"{C['span']}⟦S " if color else "⟦S ")
        if i == b0:
            out.append(f"{C['buffer']}⟨buffer " if color else "⟨buffer ")
        if i == q0:
            out.append(f"{C['q']}{{Q " if color else "{Q ")
        if i == q1:
            out.append(f"{C['prefix']}" if color else "")
        out.append(piece + "·")
        if i == s1 - 1:
            out.append(f" S⟧{C['off']}" if color else " S⟧")
        if i == b1 - 1:
            out.append(f" buffer⟩{C['off']}" if color else " buffer⟩")
        if i == q1 - 1:
            out.append(f" Q}}{C['off']}" if color else " Q}")
    body = "".join(out)
    return (f"{C['prefix']}" if color else "") + body + (C["off"] if color else "")


def compact_row(tok, p: dict, color: bool) -> str:
    s0, s1, b0, b1, q0, q1 = regions(p)
    ids = p["input_ids"]
    pc = lambda a, b: "·".join(_piece(tok, t) for t in ids[a:b])
    c = C if color else {k: "" for k in C}
    return (f"{c['prefix']}[{s0} tok] …{pc(s0 - 3, s0)}{c['off']} "
            f"{c['span']}⟦{pc(s0, s1)}⟧{c['off']} "
            f"{c['buffer']}⟨{pc(b0, b0 + 4)}…{b1 - b0}⟩{c['off']} "
            f"{c['q']}{{{pc(q0, q0 + 3)}…{pc(q1 - 2, q1)}}}{c['off']}")


def report(tok, members: dict, tag: str, color: bool) -> str:
    e = members["e"]
    s0, s1, b0, b1, q0, q1 = regions(e)
    lines = [
        f"PROBE ANATOMY  -  {DISPLAY[tag]}, matched set {e['matched_set_id']} ({e['split']})",
        f"  carrier = {e['carrier']}, position = {e['position']} (probe starts at token {s0}), "
        f"emoji = {e['emoji']} {e['emoji_name']} ({e['seq_type']})",
        f"  regions (token indices, [start, end)):  prefix [0,{s0})   S [{s0},{s1})   "
        f"buffer [{b0},{b1}) = {b1 - b0}   Q [{q0},{q1}) = {q1 - q0}   seq_len {e['seq_len']}",
        "",
        "Full token sequence of the emoji probe ( · separates tokens, ␣ = space, ⏎ = newline,"
        " ‹F0 9F› = raw bytes ):",
        full_view(tok, e, color),
        "",
        "All conditions, one above another (only the slot ⟦…⟧ differs):",
    ]
    for cnd in ORDER:
        if cnd in members:
            lines.append(f"  {LABEL[cnd]:9} {compact_row(tok, members[cnd], color)}")
    lines.append("")
    lines.append(f"  {'cond':9} {'filler':>16} {'|S|':>4} {'frag':>5}  {'start':>5}  "
                 f"{'pre-probe ids':>14}  {'post-probe ids (buffer+Q)':>26}")
    pre_ref = e["input_ids"][:s0]
    post_ref = e["input_ids"][s1:s1 + 80]
    for cnd in ORDER:
        if cnd not in members:
            continue
        p = members[cnd]
        a0, a1 = p["token_span"]
        pre, post = p["input_ids"][:a0], p["input_ids"][a1:a1 + 80]
        ok_pre = "✓ same" if pre == pre_ref else "✗ DIFFERENT"
        ok_post = "✓ same" if post == post_ref else "✗ DIFFERENT"
        lines.append(f"  {LABEL[cnd]:9} {p['filler']!s:>16} {p['n_span_tokens']:>4} {p['n_byte_fragments']:>5}  {a0:>5}  "
                     f"{_hash(pre)} {ok_pre:>6}  {_hash(post):>17} {ok_post:>8}")
    lines.append("  (8-character hash of the token-ID list; identical hash = identical tokens)")
    return "\n".join(lines)


def plot(tok, members: dict, tag: str, path: str) -> None:
    from matplotlib.patches import FancyBboxPatch
    from . import plotstyle
    plt = plotstyle.plt
    plotstyle.apply()
    cols = {"prefix": "#898781", "span": plotstyle.SERIES[1], "buffer": plotstyle.SERIES[2], "q": plotstyle.SERIES[0]}
    rows = [c for c in ORDER if c in members]
    fig, ax = plt.subplots(figsize=(16, 0.62 * len(rows) + 1.9))
    ax.axis("off")
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.8, len(rows) + 1.2)
    safe = lambda s: "".join(ch if ord(ch) < 0x2E80 else f"U+{ord(ch):04X}" for ch in s)

    def boxes(x, y, pieces, color, draw=True):
        for t in pieces:
            t = safe(t.replace("‹", "").replace("›", ""))
            w = 0.075 * len(t) + 0.18
            if not draw:
                x += w + 0.03
                continue
            ax.add_patch(FancyBboxPatch((x, y - 0.2), w, 0.4, boxstyle="round,pad=0,rounding_size=0.06",
                                        fc=color, ec=plotstyle.SURFACE, lw=1.5))
            ax.text(x + w / 2, y, t, ha="center", va="center", fontsize=6.5, color="white", family="DejaVu Sans Mono")
            x += w + 0.03
        return x

    e = members["e"]
    # common x where the buffer starts, so buffer and Q line up across rows
    x_after = 0
    for cnd in rows:
        p = members[cnd]
        ids = p["input_ids"]
        s0, s1 = p["token_span"]
        x = boxes(2.35, 0, [_piece(tok, t) for t in ids[s0 - 3:s0]], None, draw=False) + 0.15
        x_after = max(x_after, boxes(x, 0, [_piece(tok, t) for t in ids[s0:s1]], None, draw=False) + 0.15)
    for i, cnd in enumerate(rows):
        p = members[cnd]
        s0, s1, b0, b1, q0, q1 = regions(p)
        ids = p["input_ids"]
        pcs = lambda a, b: [_piece(tok, t) for t in ids[a:b]]
        y = len(rows) - i - 0.3
        ax.text(0, y, LABEL[cnd], va="center", fontsize=9, fontweight="bold", color=plotstyle.INK)
        ax.text(0.85, y, f"[{s0} prefix tokens] …", va="center", fontsize=7.5, color=plotstyle.INK_2)
        x = boxes(2.35, y, pcs(s0 - 3, s0), cols["prefix"]) + 0.15
        boxes(x, y, pcs(s0, s1), cols["span"])
        x = boxes(x_after, y, pcs(b0, b0 + 5), cols["buffer"])
        ax.text(x + 0.05, y, f"…+{b1 - b0 - 5}", va="center", fontsize=7, color=plotstyle.INK_2)
        x = boxes(x + 0.55, y, pcs(q0, q0 + 4), cols["q"])
        ax.text(x + 0.05, y, f"…+{q1 - q0 - 4}", va="center", fontsize=7, color=plotstyle.INK_2)
        same = (ids[:s0] == e["input_ids"][:e["token_span"][0]]
                and ids[s1:s1 + 80] == e["input_ids"][e["token_span"][1]:e["token_span"][1] + 80])
        ax.text(14.1, y, ("✓ " if same else "✗ ") + f"|S| = {s1 - s0}, same pre/post IDs", va="center",
                fontsize=8, color="#006300" if same else "#d03b3b")
    s0, s1, b0, b1, q0, q1 = regions(e)
    ax.text(0, len(rows) + 0.85, f"Probe anatomy: one matched set ({DISPLAY[tag]}, {e['carrier']}, probe at token {s0})"
            f"  ·  emoji: {e['emoji_name']}", fontsize=12, fontweight="bold", color=plotstyle.INK)
    lx = 0
    for k, lab in [("prefix", f"prefix + carrier header [0,{s0})"), ("span", "probe span S (only this differs)"),
                   ("buffer", f"16-token buffer = near block [{b0},{b1})"), ("q", f"64-token query block Q [{q0},{q1})")]:
        ax.add_patch(FancyBboxPatch((lx, len(rows) + 0.28), 0.22, 0.22, boxstyle="round,pad=0", fc=cols[k], ec="none"))
        ax.text(lx + 0.3, len(rows) + 0.39, lab, fontsize=8, va="center", color=plotstyle.INK_2)
        lx += 3.9
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(model: str, set_id: str = None) -> None:
    tag = model_tag(model)
    tok = load_tokenizer(model)
    probes = read_jsonl(os.path.join(OUT_DIR, "probes", f"probes_{tag}.jsonl"))
    if set_id is None:
        # default: a dev set whose emoji has the most tokens (the most interesting picture)
        dev = [p for p in probes if p["split"] == "dev" and p["condition"] == "e" and p["carrier"] == "label"]
        set_id = max(dev, key=lambda p: (p["n_span_tokens"], p["position"] == "pos24"))["matched_set_id"]
    members = {p["condition"]: p for p in probes if p["matched_set_id"] == set_id}
    out_dir = os.path.join(OUT_DIR, "probes")
    print(report(tok, members, tag, color=True))
    with open(os.path.join(out_dir, f"anatomy_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(report(tok, members, tag, color=False) + "\n")
    plot(tok, members, tag, os.path.join(out_dir, f"anatomy_{tag}.png"))
    print(f"\nwrote {out_dir}/anatomy_{tag}.txt and .png")
