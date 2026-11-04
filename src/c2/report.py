"""Stage 7: one local web page to browse everything C2 has produced.

    python scripts/c2.py report          (also runs automatically after `run` and `figures`)

Writes outputs/c2/report/index.html plus data/*.js next to it. Open index.html in
any browser: it works offline, straight from the file system (no CDN, no server).
The page is rebuilt from whatever is on disk, so every new audit, probe set or run
(dev or confirm, Pythia or Qwen) shows up the next time this command runs.

Each run's numbers live in their own data/run_<name>.js, loaded only when that run
is selected, so a page with many runs still opens quickly.

Aggregation matches c2.figures: values are averaged within each emoji first, and
uncertainty is ±1 SE across emoji (sets sharing an emoji are not independent).
"""
from __future__ import annotations

import base64
import csv
import datetime as dt
import glob
import json
import os
import shutil
from collections import defaultdict

import numpy as np

from . import OUT_DIR, PILOT_LABEL, ROOT
from .audit_report import DISPLAY
from .config import TAGS
from .controls import COND_NAMES, CONDITIONS
from .figures import PC_MIN_HEAD_FRAC, _emoji_of
from .probes import BUFFER, QLEN, read_jsonl

REPORT_DIR = os.path.join(OUT_DIR, "report")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_template.html")
MODEL_OF_TAG = {tag: alias for alias, tag in (("pythia", "pythia410m"), ("qwen", "qwen25_05b"))}
CONFIRM_LABEL = "CONFIRMATORY RUN – descriptive view only (means ± 1 SE across emoji); the pre-registered test is not computed here"
MIXED_LABEL = "MIXED dev + confirm run – not for claims"


def _b64(a: np.ndarray) -> str:
    """float32 little-endian, base64; the page decodes it into a Float32Array."""
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def _read(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def _csv(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# stage 1-3 data: audit, controls, probes
# ---------------------------------------------------------------------------
def audit_data() -> dict:
    tags = [t for t in TAGS.values() if os.path.exists(os.path.join(OUT_DIR, "audit", f"audit_{t}.csv"))]
    if not tags:
        return None
    pool = json.load(open(os.path.join(OUT_DIR, "audit", "candidate_pool.json"), encoding="utf-8"))
    in_pool = set(pool["emoji"])
    rows, order = {}, []
    for t in tags:
        for r in _csv(os.path.join(OUT_DIR, "audit", f"audit_{t}.csv")):
            e = r["emoji"]
            if e not in rows:
                order.append(e)
                rows[e] = {"e": e, "name": r["name"], "seq": r["seq_type"], "group": r["group"], "pool": e in in_pool}
            rows[e][t] = [int(r["n_tokens_line"]), int(r["n_byte_fragments_line"]), r["token_pieces_line"],
                          int(r["n_tokens"]), r["boundary_crossing_line"] == "True"]
    return {"tags": tags, "pool_rule": pool["rule"], "pool_n": pool["n"],
            "rows": [rows[e] for e in order],
            "summary": _read(os.path.join(OUT_DIR, "audit", "audit_summary.txt"))}


def controls_data() -> dict:
    out = {}
    for t in TAGS.values():
        rows = _csv(os.path.join(OUT_DIR, "controls", f"controls_{t}.csv"))
        if rows:
            out[t] = [[r["condition"], r["filler"], int(r["n_tokens"]), int(r["n_byte_fragments"]),
                       r["token_pieces"], r["eligible"] == "True", r["reason"]] for r in rows]
    if not out:
        return None
    return {"rows": out, "summary": _read(os.path.join(OUT_DIR, "controls", "controls_summary.txt"))}


def _tokenizer(tag: str):
    try:
        from .config import load_tokenizer
        return load_tokenizer(MODEL_OF_TAG[tag])
    except Exception as exc:  # offline with no cache: the page falls back to token ids
        print(f"[report] {tag}: tokenizer unavailable ({exc.__class__.__name__}); showing token ids")
        return None


def probes_data(tag: str) -> dict:
    path = os.path.join(OUT_DIR, "probes", f"probes_{tag}.jsonl")
    if not os.path.exists(path):
        return None
    from .tokaudit import readable_piece
    tok = _tokenizer(tag)
    cache = {}

    def piece(tid):
        if tid not in cache:
            cache[tid] = readable_piece(tok, tid).replace("\n", "⏎") if tok else f"#{tid}"
        return cache[tid]

    by_set = defaultdict(dict)
    for p in read_jsonl(path):
        by_set[p["matched_set_id"]][p["condition"]] = p
    # everything outside the slot is shared within a set, and mostly across sets
    # with the same carrier/position, so the token strips are stored once each
    ctx_index, contexts, sets = {}, [], []
    for sid, members in by_set.items():
        e = members.get("e") or next(iter(members.values()))
        pre = tuple(e["input_ids"][:e["token_span"][0]])
        post = tuple(e["input_ids"][e["token_span"][1]:])
        tpre, tpost = e["text"][:e["char_span"][0]], e["text"][e["char_span"][1]:]
        same = all(tuple(m["input_ids"][:m["token_span"][0]]) == pre and
                   tuple(m["input_ids"][m["token_span"][1]:]) == post and
                   m["text"][:m["char_span"][0]] == tpre and m["text"][m["char_span"][1]:] == tpost
                   for m in members.values())
        key = (pre, post, tpre, tpost)
        if key not in ctx_index:
            ctx_index[key] = len(contexts)
            contexts.append({"pre": [piece(i) for i in pre], "post": [piece(i) for i in post],
                             "tpre": tpre, "tpost": tpost})
        conds = {}
        for c, m in members.items():
            a, b = m["token_span"]
            conds[c] = {"f": m["filler"], "p": [piece(i) for i in m["input_ids"][a:b]], "frag": m["n_byte_fragments"]}
        sets.append({"id": sid, "e": e["emoji"], "name": e["emoji_name"], "seq": e["seq_type"],
                     "split": e["split"], "carrier": e["carrier"], "pos": e["position"],
                     "start": e["token_span"][0], "n": e["n_span_tokens"], "len": e["seq_len"],
                     "ctx": ctx_index[key], "same": same, "conds": conds})
    excl = _csv(os.path.join(OUT_DIR, "probes", f"exclusions_{tag}.csv")) or []
    return {"contexts": contexts, "sets": sets, "buffer": BUFFER, "qlen": QLEN, "exclusions": excl,
            "coverage": _read(os.path.join(OUT_DIR, "probes", f"coverage_{tag}.txt"))}


# ---------------------------------------------------------------------------
# stage 5 data: one runs/<dir>
# ---------------------------------------------------------------------------
def _label(split: str) -> str:
    return {"dev": PILOT_LABEL, "confirm": CONFIRM_LABEL}.get(split, MIXED_LABEL)


def run_data(run_dir: str) -> dict:
    man = json.load(open(os.path.join(run_dir, "manifest.json"), encoding="utf-8"))
    tag = next(t for t in TAGS.values() if os.path.basename(run_dir).startswith(t + "_"))
    meta = {}
    probe_file = os.path.join(ROOT, man["probe_file"])
    for p in read_jsonl(probe_file if os.path.exists(probe_file) else os.path.join(OUT_DIR, "probes", f"probes_{tag}.jsonl")):
        if p["condition"] == "e":
            meta[p["matched_set_id"]] = p
    D = np.load(os.path.join(run_dir, "deltas.npz"))
    P = np.load(os.path.join(run_dir, "per_head.npz"))
    sets = list(D["matched_set_id"])
    codes = sorted({_emoji_of(s) for s in sets})
    code_ix = {c: i for i, c in enumerate(codes)}
    emoji_of_code = {_emoji_of(s): meta[s] for s in sets}
    groups = defaultdict(list)
    for i, s in enumerate(sets):
        groups[code_ix[_emoji_of(s)]].append(i)
    L, H = int(man["layers"]), int(man["heads"])

    def by_emoji(arr):          # [n_sets, L, H] with NaN rows for incomplete sets -> [E, L, H]
        out = np.full((len(codes), L, H), np.nan, dtype=np.float32)
        for ei, idx in groups.items():
            sub = arr[idx]
            ok = ~np.isnan(sub).any(axis=(1, 2))
            if ok.any():
                out[ei] = sub[ok].mean(0)
        return out

    def pack(arr):
        complete = ~np.isnan(arr).any(axis=(1, 2))
        return {"by_emoji": _b64(by_emoji(arr)), "layer_sets": _b64(arr.mean(-1)), "n_ok": int(complete.sum())}

    measures = {k: pack(D[k]) for k in D.files if k != "matched_set_id"}

    # per-condition M (and heavy-hitter rate), aligned to the set order above
    pid = list(P["probe_id"])
    row = {(p.rsplit("-", 1)[0], p.rsplit("-", 1)[1]): i for i, p in enumerate(pid)}
    cond = {"M_far": {}, "M_near": {}, "heavy_hitter": {}}
    for c in CONDITIONS:
        idx = [row.get((s, c)) for s in sets]
        if all(i is None for i in idx):
            continue
        for key in cond:
            arr = np.full((len(sets), L, H), np.nan, dtype=np.float32)
            for j, i in enumerate(idx):
                if i is not None:
                    arr[j] = P[key][i]
            cond[key][c] = pack(arr) if key != "heavy_hitter" else {"by_emoji": _b64(by_emoji(arr))}

    # positive control, same rule as c2.figures
    pc = None
    pr = [j for j, s in enumerate(sets) if (s, "p") in row and (s, "r") in row]
    if pr:
        d = np.stack([P["M_far"][row[(sets[j], "p")]] - P["M_far"][row[(sets[j], "r")]] for j in pr])
        hm = d.mean(0)
        frac = float((hm > 0).mean())
        pc = {"n_pos": int((hm > 0).sum()), "n_heads": int(hm.size), "mean": float(hm.mean()), "frac": frac,
              "min_frac": PC_MIN_HEAD_FRAC, "passes": bool(hm.mean() > 0 and frac > PC_MIN_HEAD_FRAC)}

    name = os.path.basename(run_dir)
    return {"name": name, "tag": tag, "model": DISPLAY.get(tag, tag), "split": man["split"], "label": _label(man["split"]),
            "L": L, "H": H, "n_sets": len(sets), "n_probes": man["n_probes"],
            "emoji": [{"e": emoji_of_code[c]["emoji"], "name": emoji_of_code[c]["emoji_name"],
                       "seq": emoji_of_code[c]["seq_type"]} for c in codes],
            "sets": {"emoji": [code_ix[_emoji_of(s)] for s in sets],
                     "carrier": [meta[s]["carrier"] for s in sets], "pos": [meta[s]["position"] for s in sets],
                     "n": [meta[s]["n_span_tokens"] for s in sets], "seq": [meta[s]["seq_type"] for s in sets]},
            "measures": measures, "cond": cond, "pc": pc, "manifest": man,
            "summary": _read(os.path.join(OUT_DIR, "pilot", f"pilot_summary_{tag}.txt")) if man["split"] == "dev" else None}


# ---------------------------------------------------------------------------
def _write_js(path: str, var: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{var} = ")
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"), allow_nan=True)
        f.write(";\n")


def main() -> str:
    data_dir = os.path.join(REPORT_DIR, "data")
    shutil.rmtree(data_dir, ignore_errors=True)
    os.makedirs(data_dir)
    runs = []
    for d in sorted(glob.glob(os.path.join(OUT_DIR, "runs", "*"))):
        if not os.path.exists(os.path.join(d, "manifest.json")) or not os.path.exists(os.path.join(d, "deltas.npz")):
            continue
        r = run_data(d)
        _write_js(os.path.join(data_dir, f"run_{r['name']}.js"), f"window.C2_RUN[{json.dumps(r['name'])}]", r)
        runs.append({k: r[k] for k in ("name", "tag", "model", "split", "label", "n_sets", "n_probes")} |
                    {"n_emoji": len(r["emoji"]), "timestamp": r["manifest"]["timestamp"],
                     "file": f"data/run_{r['name']}.js"})
        print(f"[report] run {r['name']}: {r['n_sets']} sets, {len(r['emoji'])} emoji")
    images = [{"src": os.path.relpath(p, REPORT_DIR), "section": os.path.basename(os.path.dirname(p)),
               "name": os.path.basename(p)}
              for p in sorted(glob.glob(os.path.join(OUT_DIR, "**", "*.png"), recursive=True))
              if not p.startswith(REPORT_DIR)]
    base = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "pilot_label": PILOT_LABEL,
        "display": DISPLAY, "cond_names": COND_NAMES, "conditions": CONDITIONS,
        "audit": audit_data(), "controls": controls_data(),
        "probes": {t: v for t in TAGS.values() if (v := probes_data(t))},
        "runs": runs, "images": images,
        "texts": {"pytest": _read(os.path.join(OUT_DIR, "tests", "pytest_output.txt")),
                  "handcheck": _read(os.path.join(OUT_DIR, "tests", "handcheck.txt")),
                  **{f"anatomy_{t}": _read(os.path.join(OUT_DIR, "probes", f"anatomy_{t}.txt")) for t in TAGS.values()}},
    }
    _write_js(os.path.join(data_dir, "base.js"), "window.C2", base)
    shutil.copyfile(TEMPLATE, os.path.join(REPORT_DIR, "index.html"))
    out = os.path.join(REPORT_DIR, "index.html")
    print(f"wrote {out}\n  open it with:  open \"{out}\"")
    return out
