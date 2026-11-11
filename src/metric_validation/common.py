"""Shared helpers for the two metric-validation checks.

Everything heavy is reused from the existing pipeline:
  - model loading at a pinned revision        (eas.load_model)
  - the streaming attention reducer           (c2.reduce.SpanReducer)
  - the study metric M                        (c2.metric.m_from_reduced)
  - the generated probe files and their spans (c2.probes.read_jsonl)

Imports of torch and the reducer are deferred into the functions that need them,
so this module (and the pure-arithmetic helpers below) import cleanly on a
machine without the model stack installed.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
from typing import Dict, List, Sequence, Tuple

import numpy as np

from c2 import OUT_DIR as PIPELINE_OUT_DIR, ROOT
from c2.config import model_entry, model_tag
from c2.probes import SEED, read_jsonl

OUT_DIR = os.path.join(ROOT, "outputs", "metric_validation")
PROBE_DIR = os.path.join(PIPELINE_OUT_DIR, "probes")

# Share of attention heads a contrast must win to count as detected. Kept equal
# to the positive-control bar already used by the pipeline, so the first-token
# check and the near-block check are read on the same scale.
PASS_BAR = 0.60


# ---------------------------------------------------------------------------
# Probe loading and span bookkeeping
# ---------------------------------------------------------------------------
def probe_path(tag: str) -> str:
    return os.path.join(PROBE_DIR, f"probes_{tag}.jsonl")


def load_probes(tag: str, split: str = "dev") -> Tuple[str, List[dict]]:
    """The generated probes for one model tag and split (default the pilot)."""
    path = probe_path(tag)
    if not os.path.exists(path):
        raise SystemExit(
            f"probe file not found: {path}\n"
            "generate it first, e.g.  python scripts/c2.py probes --model <model>"
        )
    rows = []
    for probe in read_jsonl(path):
        if split == "all" or probe["split"] == split:
            rows.append(probe)
    if not rows:
        raise SystemExit(f"no probes with split={split!r} in {path}")
    return path, rows


def span_indices(probe: dict) -> Dict[str, List[int]]:
    """Token indices for the slot, the near buffer and the far query block."""
    slot = list(range(probe["token_span"][0], probe["token_span"][1]))
    buffer = list(range(probe["buffer_span"][0], probe["buffer_span"][1]))
    far_q = list(range(probe["query_span"][0], probe["query_span"][1]))
    return {"slot": slot, "buffer": buffer, "far_q": far_q}


def group_by_set(probes: List[dict]) -> Dict[str, Dict[str, dict]]:
    """matched_set_id -> {condition: probe}."""
    by_set: Dict[str, Dict[str, dict]] = {}
    for probe in probes:
        sid = probe["matched_set_id"]
        if sid not in by_set:
            by_set[sid] = {}
        by_set[sid][probe["condition"]] = probe
    return by_set


# ---------------------------------------------------------------------------
# Model loading and one forward pass (deferred imports)
# ---------------------------------------------------------------------------
def load_pinned_model(model: str, device: str):
    """Load one model and tokenizer at the pinned revision, seeded."""
    import torch
    from eas import load_model

    key, name, rev = model_entry(model)
    torch.manual_seed(SEED)
    mdl, tok = load_model(name, revision=rev, device=None if device == "cpu" else device)
    return key, name, rev, mdl, tok


def check_retokenises(tok, probe: dict) -> None:
    """Guard copied in spirit from the runner: the stored text must reproduce the
    stored input_ids, so what is measured is exactly what was generated."""
    got = tok(probe["text"], add_special_tokens=False)["input_ids"]
    if got != probe["input_ids"]:
        raise RuntimeError(f"{probe['id']}: text does not re-tokenise to stored input_ids")


# ---------------------------------------------------------------------------
# Aggregation (pure numpy, no model needed)
# ---------------------------------------------------------------------------
def mean_over_sets(arrays: List[np.ndarray]) -> np.ndarray:
    """Mean of a list of [L, H] arrays -> [L, H]."""
    stacked = np.stack(arrays, axis=0)
    return stacked.mean(axis=0)


def se_over_sets(arrays: List[np.ndarray]) -> float:
    """Standard error, across sets, of the head-and-layer-averaged value.

    The spread is taken across matched sets because probes that share an emoji
    are not independent samples.
    """
    per_set = []
    for array in arrays:
        per_set.append(float(array.mean()))
    values = np.asarray(per_set, dtype=np.float64)
    if values.size < 2:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(values.size))


def share_positive(delta_lh: np.ndarray) -> float:
    """Fraction of (layer, head) cells where the aggregate contrast is positive."""
    total = int(delta_lh.size)
    if total == 0:
        return 0.0
    positive = int((delta_lh > 0).sum())
    return positive / total


def aggregate_contrasts(per_set: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """List of {name: [L, H]} per set -> {name: stacked [N, L, H]}."""
    names = set()
    for one in per_set:
        for name in one:
            names.add(name)
    out: Dict[str, np.ndarray] = {}
    for name in sorted(names):
        collected = []
        for one in per_set:
            if name in one:
                collected.append(one[name])
        out[name] = np.stack(collected, axis=0)
    return out


# ---------------------------------------------------------------------------
# Reproducibility: output dir, hashing, versions, manifest
# ---------------------------------------------------------------------------
def ensure_out_dir(check: str, tag: str, split: str) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(OUT_DIR, f"{check}_{tag}_{split}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def peak_rss_mb():
    """Best-effort peak resident memory; None where the OS does not expose it."""
    try:
        import resource
    except Exception:
        return None
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if platform.system() == "Darwin" else raw / 1e3


def version_info(device: str) -> dict:
    import tokenizers
    import torch
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "seed": SEED,
    }


def base_manifest(check, model, key, name, rev, mdl, probes_path, n_probes,
                  n_sets, split, device, runtime_s, params, results) -> dict:
    manifest = {
        "check": check,
        "label": "metric validation check — not a confirmatory result",
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "model_config_key": key,
        "model_id": name,
        "model_revision_pinned": rev,
        "model_revision_loaded": getattr(mdl.config, "_commit_hash", None),
        "attn_implementation": getattr(mdl.config, "_attn_implementation", None),
        "dtype": str(next(mdl.parameters()).dtype),
        "layers": int(mdl.config.num_hidden_layers),
        "heads": int(mdl.config.num_attention_heads),
        "split": split,
        "probe_file": os.path.relpath(probes_path, ROOT),
        "probe_file_sha256": sha256_file(probes_path),
        "n_probes_used": n_probes,
        "n_matched_sets": n_sets,
        "runtime_s": round(runtime_s, 2),
        "peak_memory_mb": peak_rss_mb(),
        "params": params,
        "results": results,
        "repo_commit": git_commit(),
    }
    manifest.update(version_info(device))
    return manifest


def write_manifest(out_dir: str, manifest: dict) -> None:
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, ensure_ascii=False)


def per_layer_means(array_lh: np.ndarray) -> np.ndarray:
    """Mean over heads for each layer -> [L]."""
    return array_lh.mean(axis=1)


def verdict_line(label: str, share: float, passed: bool) -> str:
    return f"  {label:14} share {share:.3f}  {'PASS' if passed else 'FAIL'}"
