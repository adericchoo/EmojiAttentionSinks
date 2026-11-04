# C2 – Probes, metric and experiment runner (Isaac)

This is the guide to **role C2** in *Do Emoji Subtokens Attract Sink-Like Attention?*
It explains what the C2 code does, how to run it, how to read every output, and
how each part covers one of my four C2 jobs.

> **Where things live**
> - `probe/c2_probe_prototype.py`: my first standalone prototype. It is kept for
>   reference and is **not used** by the pipeline.
> - `src/c2/`: the real implementation (this README describes it).
> - `scripts/c2.py`: the one command-line entry point.
> - `tests/c2/`: the C2 tests.
> - `outputs/c2/`: every output (gitignored, rebuilt by the commands below).
> - `outputs/c2/report/index.html`: a page for browsing all of the above (see [§9](#9-viewing-the-data-the-results-viewer)).

---

## 1. The question, in plain words

A language model reads text as **tokens** (small chunks). "cat" is one token.
Most emoji are chopped into several meaningless **byte fragments**: 👍🏽 becomes
6 of them in Pythia.

When the model reads a token, it **looks back** at earlier tokens. This is called
**attention**. Some tokens act like magnets and soak up far more attention than
their meaning deserves ("attention sinks").

**Question:** do an emoji's fragments act like magnets?

The catch is that "emoji get lots of attention" could have other causes. They
could simply be split into many pieces, sit in an unusual spot, or be followed by
different words. So every emoji is compared with **look-alikes that share all of
those properties but are not emoji**, inside an *identical* sentence. If the emoji
still gets more attention, that is evidence the effect is about being an emoji.

Models: **Pythia-410M** (primary, 24 layers × 16 heads) and **Qwen2.5-0.5B**
(replication, 24 layers × 14 heads). Both run at pinned revisions from
`config/models.yaml`, in eager attention mode, without special tokens.

---

## 2. How this covers my C2 jobs

| C2 job (from the team plan) | Done by | Proof / output |
|---|---|---|
| 1. Tokenisation audit | `src/c2/inventory.py`, `tokaudit.py`, `audit_report.py` | `outputs/c2/audit/`: per-emoji CSVs, summary table, histogram, "how emoji get chopped" figure |
| 2. Probe generator, with matched probes, rejection rules and an exclusion log | `controls.py`, `carriers.py`, `probes.py`, `probes_report.py`, `anatomy.py` | `outputs/c2/probes/`: probes JSONL, exclusion CSV, coverage report, anatomy figure |
| 3. Metric (M, Δ, diagnostics) | `metric.py`, `reduce.py`, `handcheck.py` | `tests/c2/` (49 tests pass), `outputs/c2/tests/handcheck.txt` (hand sum = code) |
| 4. Experiment runner, plugged into my groupmate's pipeline | `runner.py`, which calls `eas.load_model` and `eas.capture_attention` | `outputs/c2/runs/<run>/`: per-head results, Δ table, manifest |
| Presentation outputs | `figures.py`, `report.py` | `outputs/c2/pilot/` PNGs and the results viewer |

**My groupmate's code (`src/eas/`) was not changed.** The only shared files I
touched are `environment.yml` and `requirements.txt`, where I added one line each
for `matplotlib` (needed for the figures). Please mention this to them.

---

## 3. Running it (in order)

Run everything from the repo root, inside the project environment:

```bash
conda activate eas            # important: the base Anaconda Python has an older transformers that cannot load Qwen

python scripts/c2.py audit                          # 1. tokenisation audit, both models            (~1 min)
python scripts/c2.py controls                       # 2. build and screen look-alike fillers         (~1 min)
python scripts/c2.py probes  --model pythia         # 3. matched probes + exclusion log              (~1 min)
python scripts/c2.py probes  --model qwen
python scripts/c2.py anatomy --model pythia         #    print/draw one matched set
python scripts/c2.py handcheck                      # 4. metric checked by hand on a real sentence
pytest -v tests | tee outputs/c2/tests/pytest_output.txt   #  all tests (C1's + mine)
python scripts/c2.py run     --model pythia --split dev      # 5. DEV PILOT run, 480 probes   (~2 min, CPU)
python scripts/c2.py figures --model pythia                  # 6. pilot figures (labelled DEVELOPMENT PILOT)
python scripts/c2.py report                                  # 7. rebuild the results viewer
open outputs/c2/report/index.html
```

`run` and `figures` rebuild the viewer automatically, so you only need `report`
by hand after the audit, controls or probe steps.

The real (confirmatory) run is **not done yet**. When the team is ready:

```bash
python scripts/c2.py run --model pythia --split confirm     # 4,608 probes, ~16 min on a laptop CPU
python scripts/c2.py run --model qwen   --split confirm
```

Useful flags for `run`: `--limit 2` runs only the first 2 matched sets (a quick
smoke test), and `--device mps` uses Apple GPU (it was slower than CPU in the pilot).

---

## 4. Step 1: tokenisation audit

**What it does.** It reads the official Unicode Emoji 15.1 list
(`data/unicode/emoji-test-15.1.txt`). There are 3,773 fully-qualified emoji; the
SHA-256 is checked against the proposal (`d876ee24…8137ed9`, matches). It then
counts how many tokens each emoji becomes in each model.

- The span is found with **offset mappings**: a token belongs to the emoji if its
  character range overlaps the emoji's.
- Each token is labelled **whole character** or **byte fragment**. A byte fragment
  is a token that decodes to `�` on its own.
- Each emoji gets a type: single, skin tone, ZWJ (joined emoji like 👨‍👩‍👧), flag
  or keycap.
- Code points, UTF-8 bytes and model tokens are kept as separate columns.
  "Fragmentation" always means model tokens.

**Outputs** (`outputs/c2/audit/`):

| File | Show it for |
|---|---|
| `audit_summary.txt` | The comparison with the proposal's numbers |
| `audit_token_hist.png` | "How many tokens does an emoji cost?" for both models |
| `chopped_examples.png/.txt` | 👍🏽, a ZWJ family, a flag, a rare CJK character, a hex string and an English word, chopped by each model |
| `audit_pythia410m.csv`, `audit_qwen25_05b.csv` | Every emoji, every token piece |
| `candidate_pool.json` | The 1,414 emoji the experiment can sample from |

**Numbers vs the proposal.** I was asked to report differences honestly, not force a match.

| | Pythia | Qwen | Proposal |
|---|---|---|---|
| Inside the protocol carrier "…the marker [PROBE]. …" (space before emoji) | 99.97% multi-token, median 6, max 27 | **99.92%**, median **4**, max **21** | – |
| Emoji at the **start of a line** (= tokenised alone) | 100%, median 6, max 27 | 76.78%, median 2, max 19 | – |
| Same, also dropping # and * like the proposal | **3,771/3,771**, median 6, max 27 | **2,895/3,771 (76.77%)**, median 2, max 19 | identical |
| Pool: 2–6 tokens in both models | – | – | **1,414** with the line-start rule (2,287 with the space carrier) |

*Why they differ:* the proposal's numbers come from tokenising each emoji **on
its own**. With a space in front, Qwen splits many emoji further. For example,
`" 👍🏽"` becomes `␣‹F0 9F 91› ‹8D› 🏽` instead of 2 tokens. Our probes put the emoji
at the **start of a line** (see §6), which tokenises exactly like "on its own", so
the **1,414 pool is the right one for our probes**.

The proposal excluded # and * as "boundary-crossing", but no rule I tried flags
them in either model. They have 7 (Pythia) and 5 (Qwen) tokens, so they fall
outside the 2–6 pool anyway, and the pool does not depend on them.

---

## 5. Step 2: the look-alikes (control conditions)

Every emoji is compared with five other slot fillers:

| Code | Condition | Example | Why it is there |
|---|---|---|---|
| **e** | emoji | 👍🏽 | The thing being tested |
| **w** | English word the tokenizer splits on its own | `outhyperbolize` → outh·y·per·bol·ize | "Is it just being split into pieces?" (confirmatory contrast) |
| **s** | rare non-emoji characters (old scripts, rare CJK) | NKo, Runic, CJK Ext A/B, Cuneiform… → byte fragments | **Main control**: chopped into bytes like an emoji, but not an emoji (confirmatory) |
| **x** | hex string | `28324f17b` | Secondary non-word control |
| **p** | punctuation | `.` `,` `!` `;` | **Positive control**: known to attract attention, so it tests whether the metric can detect anything at all |
| **r** | ordinary 1-token word | `cat`, `road` | Reference baseline |

w, s and x must have **exactly the same number of tokens as the emoji, in that
model**. `controls.py` generates large candidate lists (about 4,000 dictionary
words, about 1,600 rare-character strings, about 600 hex strings) and screens
each one inside the probe slot. It checks token count and byte fragments (s must
actually produce fragments; w and x must not), and no word is ever split by hand.
`outputs/c2/controls/controls_summary.txt` shows there are candidates for every
needed count (2–6) in both models.

---

## 6. Step 3: the probe generator

A **probe** is one sentence with one filler in the slot. Here is its layout:

```
[ neutral office-notes prefix ... carrier header ]⏎[ SLOT ][ 16-token buffer ][ 64-token Q ][ rest ]
                                                     ↑
                         always starts at token 24 or token 96 (identical for all six conditions)
```

- **8 carriers** (neutral places where any odd string looks equally natural):
  label, list item, quoted marker, ID field, annotation, tag, table cell and
  subject line (`src/c2/carriers.py`). One example: *"The storage box on the top
  shelf has this label:⏎[SLOT] written on a strip of tape across the lid."*
- **2 positions**: the slot starts at **token 24** or **token 96**. `fit_prefix`
  trims the neutral prefix until the slot lands exactly there.
- The slot starts **on a new line**. This was a design choice I had to make.
  With a space before the slot, some fillers changed how the preceding space was
  tokenised, which moved the start index (9 Pythia and 19 Qwen fillers). With a
  newline, no filler moved it.
- The **buffer** (16 tokens right after the slot) is skipped for the main measure,
  so "just reading the next word" attention doesn't count. It is analysed
  separately as the "near" measure for RQ2.
- **Q** (the next 64 tokens) are the "readers" whose looking-back is measured.
  Sequences are 105–183 tokens (limit 256).

**Matched set** = one emoji × one carrier × one position, written six times
(e, w, s, x, p, r). The generator **rejects and logs** a set if any of these happen:

1. a token crosses the slot boundary (`boundary_crossing`);
2. span token counts differ between e, w, s and x (`span_count_differs`);
3. the token IDs **after** the slot differ between conditions (`post_probe_ids_differ`);
4. truncation would cut into Q (`truncation_cuts_Q`);
5. extra check: the token IDs **before** the slot differ (`pre_probe_ids_differ`);
6. no control candidate has the needed token count (`no_candidate_with_count`).

Every rejection is one row in `exclusions_<model>.csv`, with the model, emoji,
carrier, position, condition, reason and detail.

**Dev vs confirm split** (seed 2026). There are 5 **dev** emoji for checking the
pipeline (☝🏼 🐏 ✋🏾 🇲🇸 ™️) and 48 **confirm** emoji for the real experiment. The
confirm emoji are sampled from the 1,414 pool, spread over sequence types and
token counts. Both models use the same emoji list (`emoji_sample.json`).

**Result:** 848 sets per model (80 dev, 768 confirm) and 5,088 probes. Coverage is
**100%** with **0 exclusions** in both models. The confirm split is exactly the
4,608 forward passes per model from the proposal.

**Outputs** (`outputs/c2/probes/`):
- `probes_<model>.jsonl` has one record per probe, with every field the protocol
  asks for: ids, spans, buffer, Q, seq_len and so on.
- `coverage_<model>.txt`
- `exclusions_<model>.csv`
- `anatomy_<model>.png/.txt` shows the six versions stacked. Only the orange
  slot differs, and hashes prove the rest is token-for-token identical.

---

## 7. Step 4: the measurement (metric) and its checks

For one layer ℓ and head h, with slot tokens S and reader tokens Q:

**M(ℓ,h) = (1/|Q|) · Σ_{q∈Q} Σ_{k∈S} A[q,k]**

In words: **for each reader, take the share of its attention that lands on the
slot, then average over readers.** M runs from 0 (ignored) to 1 (all attention on
the slot). `A[q,k]` is attention **from** query q (row) **to** key k (column).

| Quantity | Meaning | Role |
|---|---|---|
| `M_far` | M with Q = the 64 readers | Main measure |
| `M_near` | M with the 16 buffer tokens as readers | RQ2 |
| **Δ_es = M_e − M_s**, **Δ_ew = M_e − M_w** | Emoji minus look-alike, in the same sentence | **Confirmatory contrasts** |
| Δ_ex = M_e − M_x | vs hex | Secondary |
| Δ_er (per token) = M_e/\|S_e\| − M_r/1 | vs ordinary word, per token because r is not count-matched | Reference |
| Δ_pr = M_p − M_r | punctuation − word | Positive-control check |
| R = M / U, with U = (1/\|Q\|) Σ \|S\|/(q+1) | How M compares with a "uniform attention" baseline | Diagnostic only |
| heavy-hitter flag | Is any slot token in the top 10% of keys by attention received from Q? | Diagnostic only |

Every value is computed **per layer × head** (384 per probe in Pythia), never
only as a per-layer average.

**Memory trick** (`reduce.py`). Full attention is huge (layers × heads × n × n).
A forward hook on each attention layer immediately reduces it to the two sums the
metric needs, then throws the full matrix away. A test proves this gives the same
M as my groupmate's full `capture_attention` (difference 2e-8), and another proves
the hooks don't change the model's output.

**Tests** (`pytest -v tests`: **49 pass**, 11 of them my groupmate's):
- attention rows sum to 1;
- future tokens get 0 attention (causal mask);
- the axis order is right (rows = queries, columns = keys; row 0 is `[1, 0, 0, …]`);
- M and Δ on a tiny hand-made matrix (M_e = 0.35, M_c = 0.15, Δ = 0.20);
- a **real Pythia sentence**: the hand sum of the printed attention values is
  0.249838, and the code gives 0.249838 (`outputs/c2/tests/handcheck.txt`);
- probe rules: boundary rejection ("cat"+"s" → "cats" is caught), # and *
  keycaps, equal counts, identical IDs, and correct buffer and Q indices.

---

## 8. Steps 5–6: the runner and the development pilot

`python scripts/c2.py run` loads the model **through my groupmate's
`eas.load_model`**. It then runs each probe at batch size 1, after first checking
that its text re-tokenises to exactly the stored IDs. One diagnostic set also goes
through `eas.capture_attention`, with its full maps saved in `diag/`. Each run
writes `outputs/c2/runs/<model>_<split>_<timestamp>/`, containing:

- `per_head.csv/.npz`: one row per probe × layer × head (M_far, M_near, per-token, R, heavy hitter);
- `deltas.csv/.npz`: one row per matched set × layer × head (all Δs);
- `manifest.json`:
  - model and tokenizer revisions (pinned = loaded);
  - torch/transformers/tokenizers versions;
  - device, seed and timestamp;
  - probe-file SHA-256 and counts;
  - validation numbers, runtime and peak memory.

**DEVELOPMENT PILOT – not confirmatory results** (Pythia-410M, 5 dev emoji, 80
matched sets, 480 probes). Treat these as a pipeline check only.

- Extraction checks pass: row-sum error 7e-7, causal leak 0, stream vs full capture 2e-8.
- Runtime 0.21 s/probe on CPU, so the full confirm run is about 16 min per model. Peak memory is about 1.9 GB.
- Mean over all heads: Δ_es +0.0053 ± 0.0003, Δ_ew +0.0023 ± 0.0005, Δ_ex +0.0045 ± 0.0002 (± 1 SE across emoji).
- **Positive control NOT established.** Punctuation beat the plain word in only
  197/384 heads (51%; I set the pass mark at > 60%). So we **cannot yet show the
  metric is sensitive enough**, and Δ should not be interpreted yet. This is
  reported, not hidden.
- One thing to watch: **layer 22, head 1** gives 70–85% of the readers' attention
  to the emoji but 0–13% to the rare-character look-alike, for all 5 dev emoji.
  With 5 emoji this is a lead, not a finding.

Figures in `outputs/c2/pilot/` are all titled with the pilot label and n:
- Δ_es and Δ_ew heatmaps (24 layers × 16 heads);
- per-layer Δ lines;
- the positive-control plot;
- `pilot_summary_pythia410m.txt`.

---

## 9. Viewing the data: the results viewer

```bash
python scripts/c2.py report        # also runs automatically after `run` and `figures`
open outputs/c2/report/index.html  # any browser; works offline, no server needed
```

The page is rebuilt from whatever is in `outputs/c2/`, so **every new run you do
(dev or confirm, Pythia or Qwen) appears automatically**. Hover over any chart for
exact values.

| Tab | What you can do |
|---|---|
| Overview | Status of every pipeline step, plus a glossary |
| 1 · Audit | Token-count histogram for both models; search any emoji and see its pieces |
| 2 · Controls | Candidates available at each token count; browse all fillers |
| 3 · Probes | Pick any of the 848 matched sets: see the six versions, every token colour-coded (prefix / slot / buffer / Q), and the exclusion log |
| 4 · Runs | Pick a run: headline Δ tiles, positive-control verdict, extraction checks, a clickable layer × head heatmap for any Δ, raw M per condition for a chosen head, Δ by layer, the strongest heads, and the manifest. Filter by carrier, position, emoji type or slot length |
| 5 · Tests | pytest results per file and the hand-check |
| 6 · Figures | Every saved PNG |

Dev runs are bannered **DEVELOPMENT PILOT – not confirmatory results**.
Confirm runs are labelled as a descriptive view only. They show means ± SE
across emoji, not the pre-registered statistical test, which still needs to be
agreed with the team.

---

## 10. My prototype's gaps, and how they were fixed

| Gap in `c2_probe_prototype.py` | Fix in `src/c2/` |
|---|---|
| 5 carriers; early/mid/late prefixes | 8 carriers; slot at exactly token 24 or 96 |
| No exact token-count matching or hex-length search; no post-probe ID check | Per-model screening of every candidate; seeded matching; rejection rules 1–6 above |
| No 16-token buffer or fixed 64-token Q | Fixed buffer + Q, stored in every probe record |
| Per-layer means only; no Δ; no exclusion log | Per-head M_far/M_near, a Δ table and an exclusion CSV |
| Old "mass"/"ratio" definitions | M exactly as in the proposal; R is a diagnostic only |

---

## 11. Decisions I made that the team should confirm

1. **Report 1,414 as the pool**, with the line-start rule explained in §4,
   instead of the protocol carrier with a space (which gives 2,287).
2. **The slot starts on a new line** (the reason is in §6).
3. **The 8 carriers** listed in §6.
4. **The heavy-hitter definition**: the top 10% of keys by attention received from Q.
5. **The positive-control pass mark**: more than 60% of heads favour punctuation.
6. The English words come from the system dictionary, so many are obscure
   ("abaft"). A word-frequency list would make w more natural; this is optional.
7. Whether to commit this `probe/` folder (prototype + README) to the repo.

## 12. Glossary

- **token**: a chunk of text the model reads.
- **byte fragment**: a token that is only part of a character's UTF-8 bytes (shown as `‹F0 9F›`).
- **slot (S)**: the tokens of the filler being tested.
- **buffer**: the 16 tokens after the slot.
- **Q**: the 64 tokens after the buffer (the readers).
- **M**: the average share of the readers' attention on the slot.
- **Δ**: M(emoji) − M(look-alike).
- **head**: one of the model's separate attention mechanisms (Pythia has 384).
- **SE**: standard error across **emoji**, because sentences sharing an emoji are not independent samples.
- **dev / confirm**: the 5 emoji for testing the pipeline vs the 48 for the real experiment.
