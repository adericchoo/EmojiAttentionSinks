"""Stage 5: experiment runner. Probes -> forward passes -> per-head metrics + Δ.

    python scripts/c2.py run --model pythia --split dev

For one model and probe file:
  1. load the model through eas.load_model (pinned revision, eager, fp32, eval)
  2. run each probe at batch size 1 with the streaming SpanReducer
  3. compute M_far, M_near, R, heavy-hitter per layer x head; Δ per matched set
  4. write results + a run manifest (versions, hashes, runtime, peak memory)

A small diagnostic subset (first matched set) is ALSO run through
eas.capture_attention; its full maps are saved and its M must equal the
streaming M (recorded in the manifest).
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np

from . import OUT_DIR, PILOT_LABEL, ROOT
from .config import model_entry, model_tag
from .controls import MATCHED
from .metric import heavy_hitter, m_from_attention, m_from_reduced, ratio
from .probes import SEED, read_jsonl
from .reduce import SpanReducer

RUN_DIR = os.path.join(OUT_DIR, "runs")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cached_commit(name: str, filename: str, revision: str = None):
    """Commit hash of the snapshot a file was actually loaded from (HF cache)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(name, filename, revision=revision)
        if isinstance(p, str) and "/snapshots/" in p:
            return p.split("/snapshots/")[1].split("/")[0]
    except Exception:
        pass
    return None


def _git_commit():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if platform.system() == "Darwin" else r / 1e3  # bytes on macOS, KiB on Linux


def probe_metrics(red: dict, p: dict) -> Dict[str, np.ndarray]:
    s = list(range(*p["token_span"]))
    buf = list(range(*p["buffer_span"]))
    q = list(range(*p["query_span"]))
    m_far = m_from_reduced(red["a_S"], s, q)
    m_near = m_from_reduced(red["a_S"], s, buf)
    return {
        "M_far": m_far,
        "M_near": m_near,
        "M_far_per_token": m_far / len(s),
        "M_near_per_token": m_near / len(s),
        "R_far": ratio(m_far, len(s), q),
        "R_near": ratio(m_near, len(s), buf),
        "heavy_hitter": heavy_hitter(red["recv_Q"], s, n_keys=q[-1] + 1),
    }


def deltas(metrics: Dict[str, Dict[str, np.ndarray]], probes: List[dict]) -> List[dict]:
    """Per matched set: arrays [L, H] for each contrast that set supports."""
    by_set = defaultdict(dict)
    for p in probes:
        by_set[p["matched_set_id"]][p["condition"]] = p
    out = []
    for sid, mem in by_set.items():
        if "e" not in mem:
            continue
        me = metrics[mem["e"]["id"]]
        d = {}
        for c in MATCHED:                       # count-matched: raw M
            if c in mem:
                mc = metrics[mem[c]["id"]]
                d[f"d_e{c}_far"] = me["M_far"] - mc["M_far"]
                d[f"d_e{c}_near"] = me["M_near"] - mc["M_near"]
        if "r" in mem:                          # not count-matched: per token
            mr = metrics[mem["r"]["id"]]
            d["d_er_far_per_token"] = me["M_far_per_token"] - mr["M_far_per_token"]
            d["d_er_near_per_token"] = me["M_near_per_token"] - mr["M_near_per_token"]
        if "p" in mem and "r" in mem:           # positive control (both 1 token)
            d["d_pr_far"] = metrics[mem["p"]["id"]]["M_far"] - metrics[mem["r"]["id"]]["M_far"]
            d["d_pr_near"] = metrics[mem["p"]["id"]]["M_near"] - metrics[mem["r"]["id"]]["M_near"]
        e = mem["e"]
        out.append({"matched_set_id": sid, "split": e["split"], "emoji": e["emoji"], "carrier": e["carrier"],
                    "position": e["position"], "n_span_tokens": e["n_span_tokens"], "conditions": "".join(sorted(mem)),
                    "arrays": d})
    return out


def main(model: str, split: str = "dev", probes_path: str = None, device: str = "cpu",
         limit: int = None, n_diag_sets: int = 1) -> str:
    import torch
    import transformers
    import tokenizers
    from eas import load_model, capture_attention

    key, name, rev = model_entry(model)
    tag = model_tag(model)
    probes_path = probes_path or os.path.join(OUT_DIR, "probes", f"probes_{tag}.jsonl")
    probes = [p for p in read_jsonl(probes_path) if split == "all" or p["split"] == split]
    if limit:
        keep = sorted({p["matched_set_id"] for p in probes})[:limit]
        probes = [p for p in probes if p["matched_set_id"] in keep]
    if split == "confirm":
        print("[note] running CONFIRMATORY probes. The pilot uses --split dev only.")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(RUN_DIR, f"{tag}_{split}_{stamp}")
    os.makedirs(os.path.join(out_dir, "diag"), exist_ok=True)

    torch.manual_seed(SEED)
    t_load = time.time()
    mdl, tok = load_model(name, revision=rev, device=None if device == "cpu" else device)
    t_load = time.time() - t_load
    L, H = mdl.config.num_hidden_layers, mdl.config.num_attention_heads

    diag_sets = sorted({p["matched_set_id"] for p in probes})[:n_diag_sets]
    metrics, worst_row, worst_leak, diag_diff = {}, 0.0, 0.0, 0.0
    t0 = time.time()
    per_probe = []
    with SpanReducer(mdl) as red:
        for i, p in enumerate(probes):
            # the text re-tokenises to exactly the stored IDs (what capture_attention would see)
            if tok(p["text"], add_special_tokens=False)["input_ids"] != p["input_ids"]:
                raise RuntimeError(f"{p['id']}: text does not re-tokenise to stored input_ids")
            ts = time.time()
            r = red.run(p["input_ids"], list(range(*p["token_span"])), list(range(*p["query_span"])),
                        None if device == "cpu" else device)
            per_probe.append(time.time() - ts)
            worst_row = max(worst_row, r["rowsum_err"])
            worst_leak = max(worst_leak, r["causal_leak"])
            metrics[p["id"]] = probe_metrics(r, p)
            if p["matched_set_id"] in diag_sets:
                red.__exit__()                    # plain forward for the groupmate's capture
                cap = capture_attention(mdl, tok, p["text"], device=None if device == "cpu" else device)
                red.__enter__()
                att = cap.attentions.numpy()
                m_cap = m_from_attention(att, list(range(*p["token_span"])), list(range(*p["query_span"])))
                diag_diff = max(diag_diff, float(np.abs(m_cap - metrics[p["id"]]["M_far"]).max()))
                np.savez_compressed(os.path.join(out_dir, "diag", f"{p['id']}.npz"), attentions=att.astype(np.float32),
                                    input_ids=np.array(p["input_ids"]), token_span=p["token_span"],
                                    buffer_span=p["buffer_span"], query_span=p["query_span"])
            if (i + 1) % 50 == 0 or i + 1 == len(probes):
                el = time.time() - t0
                print(f"  {i + 1}/{len(probes)} probes  {el:.0f}s  ({el / (i + 1):.2f}s/probe)", flush=True)
    t_run = time.time() - t0

    # ---- per probe x layer x head -------------------------------------------------
    names = ["M_far", "M_near", "M_far_per_token", "M_near_per_token", "R_far", "R_near", "heavy_hitter"]
    ids = [p["id"] for p in probes]
    np.savez_compressed(os.path.join(out_dir, "per_head.npz"), probe_id=np.array(ids),
                        **{k: np.stack([metrics[i][k] for i in ids]) for k in names})
    head_cols = ["probe_id", "matched_set_id", "split", "condition", "filler", "carrier", "position",
                 "n_span_tokens", "layer", "head"] + names
    with open(os.path.join(out_dir, "per_head.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head_cols)
        for p in probes:
            m = metrics[p["id"]]
            for l in range(L):
                for h in range(H):
                    w.writerow([p["id"], p["matched_set_id"], p["split"], p["condition"], p["filler"], p["carrier"],
                                p["position"], p["n_span_tokens"], l, h]
                               + [f"{m[k][l, h]:.6g}" if k != "heavy_hitter" else int(m[k][l, h]) for k in names])

    # ---- Δ per matched set x layer x head -----------------------------------------
    ds = deltas(metrics, probes)
    d_names = sorted({k for d in ds for k in d["arrays"]})
    with open(os.path.join(out_dir, "deltas.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["matched_set_id", "split", "emoji", "carrier", "position", "n_span_tokens", "layer", "head"] + d_names)
        for d in ds:
            for l in range(L):
                for h in range(H):
                    w.writerow([d["matched_set_id"], d["split"], d["emoji"], d["carrier"], d["position"],
                                d["n_span_tokens"], l, h]
                               + [f"{d['arrays'][k][l, h]:.6g}" if k in d["arrays"] else "" for k in d_names])
    np.savez_compressed(os.path.join(out_dir, "deltas.npz"), matched_set_id=np.array([d["matched_set_id"] for d in ds]),
                        **{k: np.stack([d["arrays"].get(k, np.full((L, H), np.nan)) for d in ds]) for k in d_names})

    # ---- manifest ------------------------------------------------------------------
    by_cond = defaultdict(int)
    for p in probes:
        by_cond[p["condition"]] += 1
    mps_mem = None
    if device == "mps":
        mps_mem = torch.mps.driver_allocated_memory() / 1e6
    manifest = {
        "label": PILOT_LABEL if split == "dev" else "confirmatory run",
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "model_id": name, "model_config_key": key,
        "model_revision_pinned": rev,
        "model_revision_loaded": getattr(mdl.config, "_commit_hash", None),
        "tokenizer_revision_pinned": rev,
        "tokenizer_revision_loaded": _cached_commit(name, "tokenizer.json", rev),
        "layers": L, "heads": H,
        "attn_implementation": mdl.config._attn_implementation,
        "dtype": str(next(mdl.parameters()).dtype), "device": device,
        "torch": torch.__version__, "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
        "numpy": np.__version__, "python": platform.python_version(), "platform": platform.platform(),
        "repo_commit": _git_commit(),
        "seed": SEED,
        "split": split,
        "probe_file": os.path.relpath(probes_path, ROOT), "probe_file_sha256": sha256_file(probes_path),
        "n_probes": len(probes), "n_matched_sets": len({p["matched_set_id"] for p in probes}),
        "n_probes_by_condition": dict(sorted(by_cond.items())),
        "validation": {"max_rowsum_error": worst_row, "max_causal_leak": worst_leak,
                       "diag_sets": diag_sets, "max_abs_diff_stream_vs_capture_M": diag_diff},
        "runtime": {"model_load_s": round(t_load, 2), "forward_total_s": round(t_run, 2),
                    "per_probe_mean_s": round(float(np.mean(per_probe)), 4),
                    "per_probe_max_s": round(float(np.max(per_probe)), 4),
                    "estimate_full_run_4608_probes_min": round(4608 * float(np.mean(per_probe)) / 60, 1)},
        "peak_memory": {"process_peak_rss_mb": round(_peak_rss_mb(), 1), "mps_driver_allocated_mb": mps_mem},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: manifest[k] for k in ("n_probes", "n_matched_sets", "validation", "runtime", "peak_memory")}, indent=1))
    print(f"wrote {out_dir}/  (per_head.csv/.npz, deltas.csv/.npz, manifest.json, diag/)")
    return out_dir
