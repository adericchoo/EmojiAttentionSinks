"""Stage 6: development-pilot figures and summary. NOT confirmatory results.

    python scripts/c2.py figures --model pythia [--run outputs/c2/runs/<dir>]

Uncertainty bands are ±1 SE across EMOJI (the 5 dev emoji), not across the
80 matched sets: sets sharing an emoji are not independent, so treating them as
80 samples would overstate certainty. No p-values are reported for the pilot.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

from . import OUT_DIR, PILOT_LABEL
from .audit_report import DISPLAY
from .config import model_tag

PC_MIN_HEAD_FRAC = 0.60   # positive control "passes" only if > 60% of heads favour punctuation


def latest_run(tag: str, split: str = "dev") -> str:
    runs = sorted(glob.glob(os.path.join(OUT_DIR, "runs", f"{tag}_{split}_*")))
    if not runs:
        raise SystemExit(f"no {split} run for {tag}; run `python scripts/c2.py run --model ... --split {split}` first")
    return runs[-1]


def _emoji_of(set_id: str) -> str:
    return set_id.split("-")[1]          # "pythia410m-e0190-label-pos24" -> "e0190"


def _by_emoji(arr: np.ndarray, set_ids) -> np.ndarray:
    """[n_sets, ...] -> [n_emoji, ...] (mean within emoji)."""
    groups = defaultdict(list)
    for i, s in enumerate(set_ids):
        groups[_emoji_of(s)].append(i)
    return np.stack([np.nanmean(arr[idx], axis=0) for _, idx in sorted(groups.items())])


def _mean_se(per_emoji: np.ndarray):
    m = per_emoji.mean(0)
    se = per_emoji.std(0, ddof=1) / np.sqrt(per_emoji.shape[0]) if per_emoji.shape[0] > 1 else np.zeros_like(m)
    return m, se


def heatmap(d: np.ndarray, name: str, n_sets: int, n_emoji: int, vmax: float, path: str, model: str) -> None:
    from . import plotstyle
    plt = plotstyle.plt
    plotstyle.apply()
    L, H = d.shape
    fig, ax = plt.subplots(figsize=(7.6, 7.4))
    im = ax.imshow(np.clip(d, -vmax, vmax), origin="lower", cmap=plotstyle.DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    for i, j in zip(*np.where(np.abs(d) > vmax)):     # cells beyond the colour scale get their value written in
        ax.text(j, i, f"{d[i, j]:+.2f}", ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")
    ax.set_xticks(range(H))
    ax.set_yticks(range(0, L, 2))
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, extend="both")
    cb.set_label(f"mean {name}  (red: emoji gets more attention than the control)", color=plotstyle.INK_2)
    cb.outline.set_visible(False)
    i, j = np.unravel_index(np.argmax(np.abs(d)), d.shape)
    ax.text(0.0, -0.11, f"{model}, M_far (64-token Q). Mean over {n_sets} matched sets from {n_emoji} dev emoji. "
            f"Largest |Δ| at layer {i}, head {j}: {d[i, j]:+.4f}.\nColour scale clipped at ±{vmax:.3f} (99th percentile of |Δ|); "
            "cells beyond it show their value.", transform=ax.transAxes, fontsize=8, color=plotstyle.INK_2)
    plotstyle.pilot_title(fig, f"Mean {name} per layer × head", n_sets)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def layer_lines(series: dict, n_sets: int, n_emoji: int, path: str, model: str) -> None:
    from . import plotstyle
    plt = plotstyle.plt
    plotstyle.apply()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.axhline(0, color=plotstyle.AXIS, lw=1, zorder=1)
    for k, (lab, (m, se)) in enumerate(series.items()):
        x = np.arange(len(m))
        c = plotstyle.SERIES[k]
        ax.fill_between(x, m - se, m + se, color=c, alpha=0.15, lw=0, zorder=2)
        ax.plot(x, m, color=c, label=lab, zorder=3)
    ax.set_xlim(-0.5, len(m) - 0.5)
    ax.set_xticks(range(0, len(m), 2))
    ax.set_xlabel("layer")
    ax.set_ylabel("head-averaged Δ in M_far  (emoji − control)")
    ax.legend(loc="upper left")
    ax.text(0.0, -0.16, f"{model}. Mean over all heads, then over {n_sets} matched sets; band = ±1 SE across "
            f"the {n_emoji} dev emoji. Δ_ew, Δ_es confirmatory; Δ_ex secondary.", transform=ax.transAxes,
            fontsize=8, color=plotstyle.INK_2)
    plotstyle.pilot_title(fig, "Per-layer contrast, emoji vs count-matched controls", n_sets)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def positive_control(mp: np.ndarray, mr: np.ndarray, set_ids, n_sets: int, path: str, model: str) -> dict:
    """mp, mr: [n_sets, L, H] M_far for punctuation and the ordinary word."""
    from . import plotstyle
    plt = plotstyle.plt
    plotstyle.apply()
    d = mp - mr
    head_mean = d.mean(0)                                 # [L, H]
    n_pos = int((head_mean > 0).sum())
    overall = float(head_mean.mean())
    pe, re_, de = _by_emoji(mp.mean(-1), set_ids), _by_emoji(mr.mean(-1), set_ids), _by_emoji(d.mean(-1), set_ids)
    n_emoji = pe.shape[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(mp.shape[1])
    for k, (lab, arr) in enumerate([("punctuation (. , ! ;)", pe), ("ordinary word (cat, dog, …)", re_)]):
        m, se = _mean_se(arr)
        c = plotstyle.SERIES[k]
        a1.fill_between(x, m - se, m + se, color=c, alpha=0.15, lw=0)
        a1.plot(x, m, color=c, label=lab)
    a1.set_xlabel("layer")
    a1.set_ylabel("head-averaged M_far")
    a1.set_title("Distant attention received (both are 1 token)", loc="left", fontsize=10)
    a1.legend(loc="upper left")
    m, se = _mean_se(de)
    a2.axhline(0, color=plotstyle.AXIS, lw=1)
    a2.fill_between(x, m - se, m + se, color=plotstyle.SERIES[2], alpha=0.15, lw=0)
    a2.plot(x, m, color=plotstyle.SERIES[2])
    a2.set_xlabel("layer")
    a2.set_ylabel("Δ M_far = punctuation − ordinary word")
    a2.set_title("Positive-control contrast", loc="left", fontsize=10)
    # pass = clearly positive on average AND in a clear majority of heads (> 60%)
    frac_pos = n_pos / head_mean.size
    verdict = "YES" if overall > 0 and frac_pos > PC_MIN_HEAD_FRAC else "NOT ESTABLISHED"
    order = np.argsort(head_mean, axis=None)
    top = [np.unravel_index(k, head_mean.shape) for k in order[::-1][:3]]
    bot = [np.unravel_index(k, head_mean.shape) for k in order[:2]]
    fmt = lambda ij: f"L{ij[0]}H{ij[1]} {head_mean[ij]:+.3f}"
    a2.text(0.02, 0.04, f"punctuation > word in {n_pos}/{head_mean.size} heads ({100 * frac_pos:.0f}%)\n"
            f"mean Δ over all heads = {overall:+.4f}\nstrongest: {', '.join(fmt(t) for t in top)}\n"
            f"most negative: {', '.join(fmt(b) for b in bot)}\npositive control (> {100 * PC_MIN_HEAD_FRAC:.0f}% of heads): {verdict}",
            transform=a2.transAxes, va="bottom", fontsize=8.5, color=plotstyle.INK,
            bbox=dict(fc=plotstyle.SURFACE, ec=plotstyle.GRID, boxstyle="round,pad=0.4"))
    fig.text(0.01, 0.005, f"{model}. Same carrier/position as each emoji set; band = ±1 SE across the {n_emoji} dev emoji "
             "(understates spread: the same 4 punctuation marks / reference words recur for every emoji).",
             fontsize=8, color=plotstyle.INK_2)
    plotstyle.pilot_title(fig, "Positive control: punctuation vs ordinary word", n_sets)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return {"n_heads_p_gt_r": n_pos, "n_heads": int(head_mean.size), "mean_delta_pr_far": overall,
            "passes": verdict == "YES", "per_layer_mean": m.tolist(),
            "top_heads": [fmt(t) for t in top], "bottom_heads": [fmt(b) for b in bot]}


def main(model: str, run_dir: str = None) -> None:
    tag = model_tag(model)
    run_dir = run_dir or latest_run(tag, "dev")
    manifest = json.load(open(os.path.join(run_dir, "manifest.json"), encoding="utf-8"))
    if manifest["split"] != "dev":
        raise SystemExit("figures are for the development pilot only (--split dev runs)")
    out = os.path.join(OUT_DIR, "pilot")
    os.makedirs(out, exist_ok=True)
    name = DISPLAY[tag]

    D = np.load(os.path.join(run_dir, "deltas.npz"))
    sets = list(D["matched_set_id"])
    P = np.load(os.path.join(run_dir, "per_head.npz"))
    pid = list(P["probe_id"])
    cond_of = {p: p.rsplit("-", 1)[1] for p in pid}
    set_of = {p: p.rsplit("-", 1)[0] for p in pid}
    n_emoji = len({_emoji_of(s) for s in sets})

    def complete(key):
        return ~np.isnan(D[key]).any(axis=(1, 2))

    # 1. heatmaps (shared colour scale so the two are comparable)
    hm = {}
    for key, lab in (("d_es_far", "Δ_es (emoji − rare chars)"), ("d_ew_far", "Δ_ew (emoji − English word)")):
        ok = complete(key)
        hm[key] = (D[key][ok].mean(0), int(ok.sum()), lab)
    vmax = float(np.percentile(np.abs(np.stack([v[0] for v in hm.values()])), 99))
    for key, (d, n, lab) in hm.items():
        heatmap(d, lab, n, n_emoji, vmax, os.path.join(out, f"heatmap_{key}_{tag}.png"), name)

    # 2. per-layer lines (head-averaged), SE across emoji
    series, n_line = {}, None
    for key, lab in (("d_ew_far", "Δ_ew  vs English word"), ("d_es_far", "Δ_es  vs rare chars"), ("d_ex_far", "Δ_ex  vs hex string")):
        ok = complete(key)
        series[lab] = _mean_se(_by_emoji(D[key][ok].mean(-1), [s for s, o in zip(sets, ok) if o]))
        n_line = int(ok.sum()) if n_line is None else min(n_line, int(ok.sum()))
    layer_lines(series, n_line, n_emoji, os.path.join(out, f"layers_delta_{tag}.png"), name)

    # 3. positive control
    idx = defaultdict(dict)
    for i, p in enumerate(pid):
        idx[set_of[p]][cond_of[p]] = i
    pr_sets = [s for s in sets if {"p", "r"} <= set(idx[s])]
    mp = np.stack([P["M_far"][idx[s]["p"]] for s in pr_sets])
    mr = np.stack([P["M_far"][idx[s]["r"]] for s in pr_sets])
    pc = positive_control(mp, mr, pr_sets, len(pr_sets), os.path.join(out, f"positive_control_{tag}.png"), name)

    # 4. text summary
    lines = [f"{PILOT_LABEL}", f"{name}, run {os.path.basename(run_dir)}", "",
             f"  matched sets: {len(sets)} from {n_emoji} dev emoji; probes: {manifest['n_probes']}",
             f"  validation: max row-sum error {manifest['validation']['max_rowsum_error']:.1e}, "
             f"causal leak {manifest['validation']['max_causal_leak']}, stream vs capture "
             f"{manifest['validation']['max_abs_diff_stream_vs_capture_M']:.1e}", "",
             "  Mean contrast in M_far (all layers x heads), ±1 SE across emoji:"]
    for key, lab in (("d_ew_far", "Δ_ew"), ("d_es_far", "Δ_es"), ("d_ex_far", "Δ_ex"),
                     ("d_ew_near", "Δ_ew near"), ("d_es_near", "Δ_es near"), ("d_ex_near", "Δ_ex near"),
                     ("d_er_far_per_token", "Δ_er per token")):
        ok = complete(key)
        per_e = _by_emoji(D[key][ok].mean(axis=(1, 2)), [s for s, o in zip(sets, ok) if o])
        m, se = _mean_se(per_e)
        frac = float((D[key][ok].mean(0) > 0).mean())
        lines.append(f"    {lab:16} {float(m):+.5f} ± {float(se):.5f}   (heads with mean > 0: {100 * frac:.0f}%)")
    lines.append("")
    lines.append(f"  Positive control (punctuation vs ordinary word, M_far):")
    lines.append(f"    punctuation > word in {pc['n_heads_p_gt_r']}/{pc['n_heads']} heads; "
                 f"mean Δ_pr = {pc['mean_delta_pr_far']:+.5f}")
    lines.append(f"    strongest heads: {', '.join(pc['top_heads'])};  most negative: {', '.join(pc['bottom_heads'])}")
    if pc["passes"]:
        lines.append("    -> punctuation attracts more distant attention than the ordinary word in a clear")
        lines.append("       majority of heads: the metric detects the known effect on this dev pilot.")
    else:
        lines.append(f"    -> NOT ESTABLISHED: the head-averaged difference is positive but small, and only")
        lines.append(f"       {100 * pc['n_heads_p_gt_r'] / pc['n_heads']:.0f}% of heads favour punctuation "
                     f"(pass needs > {100 * PC_MIN_HEAD_FRAC:.0f}%). The effect sits in a few heads.")
        lines.append("       Metric sensitivity is therefore UNESTABLISHED on this pilot; do not interpret Δ yet.")
    lines.append("")
    lines.append("  These are development numbers from 5 emoji. They are for checking the pipeline, not for claims.")
    text = "\n".join(lines)
    with open(os.path.join(out, f"pilot_summary_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nwrote {out}/  (heatmap_d_es_far, heatmap_d_ew_far, layers_delta, positive_control, pilot_summary)")
