"""Hand-checkable metric examples (synthetic + one real Pythia sentence).

    python scripts/c2.py handcheck

The real example prints every attention value that enters M for one layer/head,
the manual sum done with plain Python floats, and the value from c2.metric,
so the two can be compared on a slide.
"""
from __future__ import annotations

import os

import numpy as np

from . import OUT_DIR
from .metric import m_from_attention, m_from_reduced

REAL_BEFORE = "The note says:\n"
REAL_FILLER = "\U0001F44D"          # 👍
REAL_AFTER = " is the marker we used today."


def synthetic_text() -> str:
    from importlib import util
    # reuse the exact matrices the unit test uses
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "tests", "c2", "test_c2_metric.py")
    spec = util.spec_from_file_location("t", path)
    t = util.module_from_spec(spec)
    spec.loader.exec_module(t)
    lines = ["SYNTHETIC EXAMPLE (6 tokens, S = {2,3}, Q = {4,5}; rows = queries, cols = keys)", ""]
    for name, a in (("emoji", t.A_EMOJI), ("control", t.A_CONTROL)):
        lines.append(f"  A_{name}:")
        for q, row in enumerate(a):
            tag = "  <- in Q" if q in t.Q else ("  <- in S (not a query)" if q in t.S else "")
            lines.append("    q=%d  " % q + " ".join(f"{v:5.2f}" for v in row) + tag)
        m = m_from_attention(a[None, None], t.S, t.Q)[0, 0]
        hand = " + ".join(f"({a[q, 2]:.2f}+{a[q, 3]:.2f})" for q in t.Q)
        lines.append(f"    M_{name} by hand = [{hand}] / {len(t.Q)} = {sum(a[q, k] for q in t.Q for k in t.S) / len(t.Q):.4f}"
                     f"   code = {m:.4f}")
        lines.append("")
    d = m_from_attention(t.A_EMOJI[None, None], t.S, t.Q)[0, 0] - m_from_attention(t.A_CONTROL[None, None], t.S, t.Q)[0, 0]
    lines.append(f"  Δ by hand = {t.M_E_HAND:.2f} - {t.M_C_HAND:.2f} = {t.DELTA_HAND:.2f}   code = {d:.4f}")
    lines.append(f"  U by hand = (2/5 + 2/6)/2 = {t.U_HAND:.4f},  R_emoji = 0.35 / U = {0.35 / t.U_HAND:.4f}")
    return "\n".join(lines)


def real_example(model, tok, layer: int = None, head: int = None) -> dict:
    """One short sentence through the model via eas.capture_attention."""
    from eas import capture_attention
    from eas.span import locate_span_by_offsets, validate_span

    text = REAL_BEFORE + REAL_FILLER + REAL_AFTER
    cap = capture_attention(model, tok, text)
    cs, ce = len(REAL_BEFORE), len(REAL_BEFORE) + len(REAL_FILLER)
    S = locate_span_by_offsets(cap.offsets, cs, ce)
    validate_span(tok, cap.input_ids, S, REAL_FILLER)
    Q = list(range(S[-1] + 1, cap.n))
    att = cap.attentions.numpy()
    M = m_from_attention(att, S, Q)
    if layer is None:   # a head with a mid-sized M, so the hand sum is not trivially ~0 or ~1
        layer, head = np.unravel_index(np.argmin(np.abs(M - 0.25)), M.shape)
    A = att[layer, head]
    manual = 0.0
    for q in Q:
        for k in S:
            manual += float(A[q, k])
    manual /= len(Q)
    return {"cap": cap, "S": S, "Q": Q, "layer": int(layer), "head": int(head), "A": A,
            "manual": manual, "code": float(M[layer, head]),
            "code_reduced": float(m_from_reduced(att[..., S].sum(-1), S, Q)[layer, head])}


def real_text(r: dict, tok) -> str:
    from .tokaudit import readable_piece
    cap, S, Q, A = r["cap"], r["S"], r["Q"], r["A"]
    pieces = [readable_piece(tok, t).replace("\n", "⏎") for t in cap.input_ids]
    lines = [f"REAL EXAMPLE  -  Pythia-410M, text = {cap.text!r}",
             f"  tokens: " + "  ".join(f"{i}:{p}" for i, p in enumerate(pieces)),
             f"  S (emoji tokens) = {S}   Q (all later tokens) = {Q}",
             f"  layer {r['layer']}, head {r['head']}  (the head whose M is closest to 0.25; an arithmetic check, not a result)",
             "",
             "  A[q, k] for q in Q (rows) and k in S (cols), plus the row's sum over S:"]
    lines.append("        " + "".join(f"   k={k:<2}({pieces[k][:6]:>6})" for k in S) + "     sum_S")
    total = 0.0
    for q in Q:
        row = [float(A[q, k]) for k in S]
        total += sum(row)
        lines.append(f"  q={q:<2}({pieces[q][:8]:>8}) " + "".join(f"{v:17.6f}" for v in row) + f"   {sum(row):.6f}")
    lines.append("")
    lines.append(f"  manual: sum of all values above = {total:.6f};  / |Q| = {total:.6f} / {len(Q)} = {r['manual']:.6f}")
    lines.append(f"  code  : c2.metric.m_from_attention = {r['code']:.6f}   (reduced path: {r['code_reduced']:.6f})")
    lines.append(f"  match : {'YES' if abs(r['manual'] - r['code']) < 1e-6 else 'NO'}  (|diff| = {abs(r['manual'] - r['code']):.2e})")
    return "\n".join(lines)


def main() -> None:
    from .config import model_entry
    from eas import load_model
    _, name, rev = model_entry("pythia")
    model, tok = load_model(name, revision=rev)
    text = synthetic_text() + "\n\n" + real_text(real_example(model, tok), tok)
    out = os.path.join(OUT_DIR, "tests")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "handcheck.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nwrote {out}/handcheck.txt")
