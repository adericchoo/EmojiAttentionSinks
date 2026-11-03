# emoji-attention-sinks

Pipeline for _Do Emoji Subtokens Attract Sink-Like Attention?_ — a controlled,
inference-only attention-measurement study.

This repo is the **C1 (pipeline) skeleton**: model loading, attention capture,
span location, and the extraction-validation suite. Scoring/metrics (M, Δ) are
owned by C2 and plug in at the interface described below.

## Setup

Recreate the environment with conda (the project uses conda-managed PyTorch):

```bash
conda env create -f environment.yml
conda activate eas
```

Then confirm it works:

```bash
pytest -q          # should print "11 passed" — no model download needed
```

**GPU box vs laptop:** `environment.yml` pins the CPU build of PyTorch for
local development. On the lab GPU machine, edit the file to drop the `cpuonly`
line and install the CUDA build instead:
`conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia`. Everything else
is identical.

**Pin the models too.** Reproducibility needs both the environment _and_ the
model weights: set the commit SHAs in `config/models.yaml` (HuggingFace →
model page → Files and versions → History → copy the commit hash).

**Windows note:** don't use `python -m venv` here — on an Anaconda machine a
pip-installed torch clashes with Anaconda's DLLs (`WinError 1114` loading
`c10.dll`). Use the conda env above.

## The three starter scripts (run in order)

```bash
python scripts/00_feasibility.py     # loads pythia-410m, proves attentions come out, measures emoji fragmentation
python scripts/01_hand_check.py      # the hand-verifiable micro example — THE gate
python scripts/02_run_validation.py  # rows sum to 1, causal zeros, determinism
```

Offline unit tests (no model download) cover the span + validation logic:

```bash
pytest -q
```

## The gate for your proposal deliverable

- `scripts/00` prints a non-None attention tuple and the emoji fragmentation counts.
- `scripts/01` prints one head's matrix you can check by hand, and a reference
  received-attention number you and Isaac agree on.
- `scripts/02` passes all three sanity checks.
- `pytest` is green.

Once those hold on pythia-410m, repeat `scripts/00` against
`Qwen/Qwen2.5-0.5B` (note: GQA, so the head count differs) to confirm portability.

## Memory note

Captured attention is `O(L·H·n²)`. For pythia-410m (L=24, H=16): ~0.4 GB at
n=256, ~1.6 GB at n=512, ~6.4 GB at n=1024. `capture_attention` refuses
sequences longer than `max_tokens` (default 256) so a stray long carrier can't
OOM a sweep. Keep core-study carriers near 256 tokens.
