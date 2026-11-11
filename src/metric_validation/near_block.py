"""Check 2: the key contrasts measured on the near buffer instead of the far block.

The positive control Delta(p,r) and the emoji contrasts are defined on the far
query block Q. The near buffer — the 16 tokens immediately after the slot — showed
larger contrasts in the pilot. This check recomputes each contrast with the near
buffer as the query set and places it side by side with the far-block value.

Only the query set changes. The slot spans, the probe files and the metric are the
pipeline's own: for each probe a single forward pass is reduced to the slot, and
the SAME metric function is averaged once over the far block and once over the near
buffer.

Contrasts, each as M(test) - M(control), per layer x head:
    Delta(e,s), Delta(e,w), Delta(e,x)   emoji vs its count-matched controls
    Delta(p,r)                           punctuation vs ordinary word (positive control)
"""
from __future__ import annotations

import csv
import os
import time
from typing import Dict, List, Tuple

import numpy as np

from c2.metric import m_from_reduced

from . import common

# test condition -> control condition for each contrast
CONTRASTS = [
    ("d_es", "e", "s"),
    ("d_ew", "e", "w"),
    ("d_ex", "e", "x"),
    ("d_pr", "p", "r"),
]


def contrasts_for_set(m_far: Dict[str, np.ndarray],
                      m_near: Dict[str, np.ndarray]
                      ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Per matched set: {name: [L, H]} for the far block and the near buffer.

    A contrast is produced only when both of its conditions are present.
    """
    far_out: Dict[str, np.ndarray] = {}
    near_out: Dict[str, np.ndarray] = {}
    for name, test, control in CONTRASTS:
        if test in m_far and control in m_far:
            far_out[name] = m_far[test] - m_far[control]
            near_out[name] = m_near[test] - m_near[control]
    return far_out, near_out


def _write_per_head(path: str, mean_far: Dict[str, np.ndarray],
                    mean_near: Dict[str, np.ndarray], n_layers: int, n_heads: int) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["contrast", "layer", "head", "delta_far", "delta_near"])
        for name, _test, _control in CONTRASTS:
            if name not in mean_far:
                continue
            far = mean_far[name]
            near = mean_near[name]
            for layer in range(n_layers):
                for head in range(n_heads):
                    writer.writerow([
                        name, layer, head,
                        f"{far[layer, head]:.6g}", f"{near[layer, head]:.6g}",
                    ])


def _write_summary(path: str, manifest: dict) -> None:
    results = manifest["results"]
    lines = []
    lines.append(manifest["label"])
    lines.append("")
    lines.append("CHECK 2 — key contrasts on the near buffer vs the far block")
    lines.append(f"model        {manifest['model_id']}  rev {manifest['model_revision_pinned']}")
    lines.append(f"matched sets {manifest['n_matched_sets']}  (split {manifest['split']})")
    lines.append(f"pass bar     share of heads with delta > 0 greater than {results['pass_bar']:.2f}")
    lines.append("")
    header = f"  {'contrast':10}  {'mean_far':>10}  {'mean_near':>10}  {'share_far':>10}  {'share_near':>10}  verdict"
    lines.append(header)
    for name, _test, _control in CONTRASTS:
        if name not in results["contrasts"]:
            continue
        row = results["contrasts"][name]
        verdict = ""
        if name == "d_pr":
            far_flag = "PASS" if row["pass_far"] else "FAIL"
            near_flag = "PASS" if row["pass_near"] else "FAIL"
            verdict = f"far {far_flag} | near {near_flag}"
        lines.append(
            f"  {name:10}  {row['mean_far']:>10.6f}  {row['mean_near']:>10.6f}  "
            f"{row['share_far']:>10.3f}  {row['share_near']:>10.3f}  {verdict}"
        )
    lines.append("")
    lines.append("extraction validity:")
    lines.append(f"  max row-sum error = {results['max_rowsum_error']:.2e}")
    lines.append(f"  max causal leak   = {results['max_causal_leak']:.2e}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main(model: str, split: str = "dev", device: str = "cpu", limit: int = None) -> dict:
    """Run the near-vs-far contrast check for one model. Returns a results dict."""
    from c2.reduce import SpanReducer

    tag = common.model_tag(model)
    probes_path, probes = common.load_probes(tag, split)
    by_set = common.group_by_set(probes)
    set_ids = sorted(by_set)
    if limit:
        set_ids = set_ids[:limit]

    key, name, rev, mdl, tok = common.load_pinned_model(model, device)
    n_layers = int(mdl.config.num_hidden_layers)
    n_heads = int(mdl.config.num_attention_heads)
    run_device = None if device == "cpu" else device

    per_set_far: List[Dict[str, np.ndarray]] = []
    per_set_near: List[Dict[str, np.ndarray]] = []
    worst_row = 0.0
    worst_leak = 0.0
    n_probes_used = 0

    start = time.time()
    with SpanReducer(mdl) as reducer:
        for sid in set_ids:
            members = by_set[sid]
            m_far: Dict[str, np.ndarray] = {}
            m_near: Dict[str, np.ndarray] = {}
            for cond, probe in members.items():
                common.check_retokenises(tok, probe)
                spans = common.span_indices(probe)
                slot = spans["slot"]
                buffer = spans["buffer"]
                far_q = spans["far_q"]
                # one pass reduced to the slot; the same a_S is averaged over both
                # query sets, so near and far share an identical forward pass.
                run = reducer.run(probe["input_ids"], slot, far_q, run_device)
                m_far[cond] = m_from_reduced(run["a_S"], slot, far_q)
                m_near[cond] = m_from_reduced(run["a_S"], slot, buffer)
                worst_row = max(worst_row, run["rowsum_err"])
                worst_leak = max(worst_leak, run["causal_leak"])
                n_probes_used += 1
            far_out, near_out = contrasts_for_set(m_far, m_near)
            if far_out:
                per_set_far.append(far_out)
                per_set_near.append(near_out)
    runtime_s = time.time() - start

    agg_far = common.aggregate_contrasts(per_set_far)
    agg_near = common.aggregate_contrasts(per_set_near)
    mean_far: Dict[str, np.ndarray] = {}
    mean_near: Dict[str, np.ndarray] = {}
    contrasts_summary: Dict[str, dict] = {}
    for name_c, _test, _control in CONTRASTS:
        if name_c not in agg_far:
            continue
        mf = agg_far[name_c].mean(axis=0)
        mn = agg_near[name_c].mean(axis=0)
        mean_far[name_c] = mf
        mean_near[name_c] = mn
        share_far = common.share_positive(mf)
        share_near = common.share_positive(mn)
        entry = {
            "mean_far": float(mf.mean()),
            "mean_near": float(mn.mean()),
            "se_far_across_sets": common.se_over_sets(list(agg_far[name_c])),
            "se_near_across_sets": common.se_over_sets(list(agg_near[name_c])),
            "share_far": share_far,
            "share_near": share_near,
        }
        if name_c == "d_pr":
            entry["pass_far"] = bool(share_far > common.PASS_BAR)
            entry["pass_near"] = bool(share_near > common.PASS_BAR)
        contrasts_summary[name_c] = entry

    out_dir = common.ensure_out_dir("near_block", tag, split)
    _write_per_head(os.path.join(out_dir, "near_vs_far_per_head.csv"),
                    mean_far, mean_near, n_layers, n_heads)
    save_arrays = {}
    for name_c in mean_far:
        save_arrays[f"{name_c}_far"] = mean_far[name_c]
        save_arrays[f"{name_c}_near"] = mean_near[name_c]
    np.savez_compressed(os.path.join(out_dir, "near_vs_far.npz"), **save_arrays)

    results = {
        "contrasts": contrasts_summary,
        "pass_bar": common.PASS_BAR,
        "max_rowsum_error": worst_row,
        "max_causal_leak": worst_leak,
    }
    params = {"near_queries": "16-token buffer", "far_queries": "64-token Q", "pass_bar": common.PASS_BAR}
    manifest = common.base_manifest(
        "near_block", model, key, name, rev, mdl, probes_path,
        n_probes_used, len(per_set_far), split, device, runtime_s, params, results,
    )
    manifest["out_dir"] = os.path.relpath(out_dir, common.ROOT)
    common.write_manifest(out_dir, manifest)
    _write_summary(os.path.join(out_dir, "summary.txt"), manifest)

    pr = contrasts_summary.get("d_pr")
    if pr is not None:
        print(f"[near_block] {tag} {split}: Delta(p,r) far share={pr['share_far']:.3f} "
              f"({'PASS' if pr['pass_far'] else 'FAIL'}) | near share={pr['share_near']:.3f} "
              f"({'PASS' if pr['pass_near'] else 'FAIL'})  -> {out_dir}")
    else:
        print(f"[near_block] {tag} {split}: no p/r sets present  -> {out_dir}")
    results["out_dir"] = out_dir
    results["tag"] = tag
    results["model"] = model
    return results
