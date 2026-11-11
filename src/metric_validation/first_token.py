"""Check 1: can the metric detect the strongest known attention sink?

The first token of a sequence is the canonical attention sink. Because the models
here are tokenised without special tokens, "first token" means position 0.

If the study metric M registers much more attention on position 0 than on an
ordinary single-token word placed in the probe slot, then M is demonstrably
sensitive to a genuine sink — and punctuation's weakness in the positive control
becomes a fact about punctuation, not evidence that the metric is blind.

For every single-token reference probe (condition r), using the SAME metric
function and the SAME far query block Q as the main pipeline:

    M_first = M(span = {position 0},               queries = far block Q)
    M_ref   = M(span = {reference word in slot},   queries = far block Q)

Both go through c2.metric.m_from_reduced on the reduced arrays produced by the
shared SpanReducer. Results are reported per layer x head, per layer, in
aggregate, and as the share of heads where the first-token attention exceeds the
reference. A clear pass/fail is stated against PASS_BAR.
"""
from __future__ import annotations

import csv
import os
import time
from typing import Dict, List, Tuple

import numpy as np

from c2.metric import m_from_reduced

from . import common


def reference_probes(probes: List[dict]) -> List[dict]:
    """The single-token ordinary-word probes (condition r)."""
    out = []
    for probe in probes:
        if probe["condition"] == "r":
            out.append(probe)
    return out


def verdict(mean_first: np.ndarray, mean_ref: np.ndarray, bar: float) -> Tuple[float, bool]:
    """Share of heads where the sink beats the reference, and the pass flag."""
    diff = mean_first - mean_ref
    share = common.share_positive(diff)
    return share, bool(share > bar)


def _write_per_head(path: str, mean_first: np.ndarray, mean_ref: np.ndarray,
                    n_layers: int, n_heads: int) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "head", "M_first", "M_ref", "diff", "first_exceeds_ref"])
        for layer in range(n_layers):
            for head in range(n_heads):
                first = mean_first[layer, head]
                ref = mean_ref[layer, head]
                writer.writerow([
                    layer, head, f"{first:.6g}", f"{ref:.6g}",
                    f"{first - ref:.6g}", int(first > ref),
                ])


def _write_summary(path: str, manifest: dict, mean_first: np.ndarray,
                   mean_ref: np.ndarray) -> None:
    results = manifest["results"]
    layers_first = common.per_layer_means(mean_first)
    layers_ref = common.per_layer_means(mean_ref)
    lines = []
    lines.append(manifest["label"])
    lines.append("")
    lines.append("CHECK 1 — first-token (position 0) sink detection")
    lines.append(f"model            {manifest['model_id']}  rev {manifest['model_revision_pinned']}")
    lines.append(f"reference probes {results['n_reference_probes']} (condition r, single-token words)")
    lines.append(f"query block      far Q ({manifest['params']['query_block']})")
    lines.append("")
    lines.append(f"aggregate M_first (attention Q -> position 0) = {results['aggregate_M_first']:.6f}")
    lines.append(f"aggregate M_ref   (attention Q -> slot word)  = {results['aggregate_M_ref']:.6f}")
    lines.append(f"aggregate diff    (M_first - M_ref)           = {results['aggregate_diff']:.6f}")
    lines.append("")
    verdict_text = "PASS — the metric detects the first-token sink" if results["pass"] \
        else "FAIL — the metric does not separate the sink from an ordinary token"
    lines.append(f"share of heads first > ref = {results['share_heads_first_exceeds_ref']:.3f} "
                 f"(bar > {results['pass_bar']:.2f})  -> {verdict_text}")
    lines.append("")
    lines.append("per layer (mean over heads):")
    lines.append(f"  {'layer':>5}  {'M_first':>10}  {'M_ref':>10}  {'diff':>10}")
    for layer in range(mean_first.shape[0]):
        lines.append(f"  {layer:>5}  {layers_first[layer]:>10.6f}  "
                     f"{layers_ref[layer]:>10.6f}  {layers_first[layer] - layers_ref[layer]:>10.6f}")
    lines.append("")
    lines.append("extraction validity:")
    lines.append(f"  max row-sum error = {results['max_rowsum_error']:.2e}")
    lines.append(f"  max causal leak   = {results['max_causal_leak']:.2e}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main(model: str, split: str = "dev", device: str = "cpu", limit: int = None) -> dict:
    """Run the first-token check for one model. Returns a results dict."""
    from c2.reduce import SpanReducer

    tag = common.model_tag(model)
    probes_path, probes = common.load_probes(tag, split)
    refs = reference_probes(probes)
    if limit:
        refs = refs[:limit]
    if not refs:
        raise SystemExit("no condition-r (single-token reference) probes found")

    key, name, rev, mdl, tok = common.load_pinned_model(model, device)
    n_layers = int(mdl.config.num_hidden_layers)
    n_heads = int(mdl.config.num_attention_heads)
    run_device = None if device == "cpu" else device

    first_arrays: List[np.ndarray] = []
    ref_arrays: List[np.ndarray] = []
    worst_row = 0.0
    worst_leak = 0.0

    start = time.time()
    with SpanReducer(mdl) as reducer:
        for probe in refs:
            common.check_retokenises(tok, probe)
            spans = common.span_indices(probe)
            slot = spans["slot"]
            far_q = spans["far_q"]

            # Two forward passes so each quantity goes through the identical metric
            # call: one reduced to position 0, one reduced to the slot word.
            run_first = reducer.run(probe["input_ids"], [0], far_q, run_device)
            run_ref = reducer.run(probe["input_ids"], slot, far_q, run_device)

            m_first = m_from_reduced(run_first["a_S"], [0], far_q)
            m_ref = m_from_reduced(run_ref["a_S"], slot, far_q)
            first_arrays.append(m_first)
            ref_arrays.append(m_ref)

            worst_row = max(worst_row, run_first["rowsum_err"], run_ref["rowsum_err"])
            worst_leak = max(worst_leak, run_first["causal_leak"], run_ref["causal_leak"])
    runtime_s = time.time() - start

    mean_first = common.mean_over_sets(first_arrays)
    mean_ref = common.mean_over_sets(ref_arrays)
    diff = mean_first - mean_ref
    share, passed = verdict(mean_first, mean_ref, common.PASS_BAR)

    out_dir = common.ensure_out_dir("first_token", tag, split)
    _write_per_head(os.path.join(out_dir, "first_token_per_head.csv"),
                    mean_first, mean_ref, n_layers, n_heads)
    np.savez_compressed(
        os.path.join(out_dir, "first_token.npz"),
        mean_first=mean_first, mean_ref=mean_ref, diff=diff,
    )

    results = {
        "n_reference_probes": len(refs),
        "aggregate_M_first": float(mean_first.mean()),
        "aggregate_M_ref": float(mean_ref.mean()),
        "aggregate_diff": float(diff.mean()),
        "aggregate_diff_se_across_sets": common.se_over_sets(
            [f - r for f, r in zip(first_arrays, ref_arrays)]
        ),
        "share_heads_first_exceeds_ref": share,
        "pass_bar": common.PASS_BAR,
        "pass": passed,
        "max_rowsum_error": worst_row,
        "max_causal_leak": worst_leak,
    }
    params = {"reference_condition": "r", "query_block": "far_Q", "pass_bar": common.PASS_BAR}
    manifest = common.base_manifest(
        "first_token", model, key, name, rev, mdl, probes_path,
        len(refs), len(refs), split, device, runtime_s, params, results,
    )
    manifest["out_dir"] = os.path.relpath(out_dir, common.ROOT)
    common.write_manifest(out_dir, manifest)
    _write_summary(os.path.join(out_dir, "summary.txt"), manifest, mean_first, mean_ref)

    print(f"[first_token] {tag} {split}: share={share:.3f} "
          f"({'PASS' if passed else 'FAIL'})  -> {out_dir}")
    results["out_dir"] = out_dir
    results["tag"] = tag
    results["model"] = model
    return results
