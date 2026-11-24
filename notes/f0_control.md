# F0 control: non-emoji 4-byte characters

**Status.** Control set `f0-control-v1.0`, status `draft_for_team_review`. No full run
has been made. Any run on the draft is a **DEVELOPMENT PILOT – not confirmatory
results**, and the runner labels it so.

## Question

Leg 3 (script 06) says the emoji sink rides the `[F0 9F]` byte token, not "emoji-ness".
But in every existing group the two go together, so they cannot be told apart:

| | lead token `[F0 9F]` | other lead token |
|---|---|---|
| **emoji** | A (🚀) | B (☀) |
| **not emoji** | **E1 (new)**: 🡐 arrow, 🁇 domino tile | D (™, one whole token), **E2 (new)**: 𦏕 CJK Ext B |

- **A vs E1**: is the sink about emoji, or about the `[F0 9F]` token?
- **A vs E2**: is it `[F0 9F]` specifically, or the start of any 4-byte character?

## The two new groups (`config/f0_control_stimuli.yaml`)

| Group | What | Blocks (3–4 items each) | Pythia split | Qwen split |
|---|---|---|---|---|
| E1, 24 items | not emoji, same lead token as A | Domino Tiles, Alchemical Symbols, Geometric Shapes Extended, Supplemental Arrows-C, Chess Symbols, Symbols for Legacy Computing | `[F0 9F] [x] [y]`, 3 tokens, like A | `[F0 9F] [x] [y]`, 3 tokens |
| E2, 24 items | not emoji, 4 bytes, different lead token | Linear B, Old Italic, Gothic, Cuneiform, Egyptian Hieroglyphs, Mathematical Alphanumeric Symbols, CJK Ext B, CJK Ext C | `[F0] [x] [y] [z]`, 4 tokens | starts with `F0` but not `F0 9F`; 1–3 tokens |

Selection rules (all in the file's `pre_registration` block):

- "Not emoji" means the code point appears in no line of `emoji-test-15.1.txt`, so emoji components such as skin tones and regional indicators are excluded too.
- The order inside each block is seeded (`sha256('2026:<group>:<code point>')`). Blocks are visited round-robin.
- Every character must build a clean probe in all 16 carrier/position slots in **both** models, with the same token IDs before and after the slot as 🚀.
- E1 additionally needs the exact `[F0 9F]` lead and 3 tokens in both models. Eight other U+1Fxxx blocks (mahjong, playing cards, enclosed letters and so on) were screened and failed, because Qwen merges `[F0 9F]` with the next byte for them. The screen log is in the file.

Rebuild or verify: `python scripts/build_f0_control.py` / `python scripts/build_f0_control.py --check`.

## Why E1 is such a clean control: the identity check

When a model reads the `[F0 9F]` token, it has not yet seen the bytes that follow.
With the same text before the slot, its internal state at that token, and so that
token's key and value in every head, is **exactly the same** whether an emoji or a
domino tile follows. The runner checks this directly (section 2, expected difference 0).

So:
- any difference in attention between A and E1 can only come from the later
  "reader" tokens, which do see the whole character;
- leg 3c's value-size result at `[F0 9F]` is a property of the token, shared by E1.

## How it is measured (`scripts/12_f0_control.py`)

For each item, the attention its **lead token** receives from the fixed 64-token far
block, per layer × head. This uses script 05's single-key reduction, averaged over
all 16 slots. The ordinary words in `R_reference` give the per-head baseline, as in
script 06. Then:

1. **Token check.** The lead token IDs per group; A and E1 must share one ID. Items whose lead does not fit the group are listed and left out. Example: an emoji that Qwen keeps as one whole token has no `[F0 9F]` lead.
2. **Identity check.** The internal state at the lead token, A and E1 vs the first A item.
3. **Sink flags.** Script 06's head-aware sink rule per item, and script 07's peak ratio, broad share, extreme-head count and BROAD/NARROW/NON class for each group's mean.
4. **Sink heads.** The heads above 3× the baseline (the rule of scripts 08–10) per group, and how much the sets overlap (shared heads, Jaccard).
5. **On A's sink heads.** How much E1 and E2 get on A's sink heads, as a share of A. This is **cross-fitted**: the heads are picked on one half of A and A is scored on the other half, so A gets no head start from having chosen the heads.

The last line gives a one-line read using **provisional** bars: similar ≥ 0.8 of A, low < 0.5 of A.

| Result | Read |
|---|---|
| E1 ≥ 0.8 of A, E2 < 0.5 of A | token detector: `[F0 9F]` draws the attention whether or not an emoji follows |
| E1 and E2 ≥ 0.8 of A | general 4-byte detector |
| E1 < 0.5 of A | emoji-specific, and it must come from the reader tokens |
| otherwise | no clear call |

## Run it

```bash
conda activate eas
python scripts/12_f0_control.py                    # Pythia, all 16 slots
python scripts/12_f0_control.py replication        # Qwen
python scripts/12_f0_control.py --limit 3 --slots first   # smoke test, seconds
python scripts/12_f0_control.py --stimuli config/master_stimuli.yaml   # A and R from the master set
```

Groups A and R come from `config/master_stimuli.yaml` when that file exists (once the
master branch is merged); otherwise from `config/emoji_stimuli.yaml`, which has 10 A
emoji. A full Pythia run is roughly 1,000–1,800 forward passes, an estimate of
5–15 minutes on a laptop CPU.

Outputs go to `outputs/f0_control/<model>_<slots>_<time>/` (gitignored):

| File | What it holds |
|---|---|
| `summary.txt` | Everything printed |
| `items.csv` | One row per item |
| `heads.npz` | Baseline, per-item lead arrays, sink-head masks |
| `manifest.json` | Model revision, file hashes, bars, git commit, timings |

## To decide before any numbers are reported

1. **Review and freeze the set.** Check the 48 characters, then change `status`; the content hash stays the same. Alternatively, ask the master set's owner to add E1 and E2 to the master set as v1.1.
2. **Agree the decision rule.** Either keep the provisional bars or replace them with a test across items, e.g. a permutation test of E1 vs A on A's sink heads.
3. **Which A set.** The 60-item master set (better) or the 10-item leg-3 set.
4. **Slots.** All 16 (default; more robust) or the first slot only (matches script 06's numbers).

## Limits

- **Qwen.** Most emoji are one whole token, so only A items with an exact `[F0 9F]` lead enter the comparison. The master A set was verified only with Pythia's tokenizer. E2's lead tokens in Qwen vary (`F0 90`, `F0 9D 96`, ...).
- **Token counts.** E2 is 4 tokens in Pythia, against 3 for A and E1. Only the lead token is scored, and the far block is fixed, so the number of readers is the same.
- **Unicode version.** The environment's Unicode data is version 14.0, so characters added in Unicode 15/15.1 are not candidates.
- **Descriptive only.** Same metric definitions as scripts 06–08; not the pre-registered test.

## Files

`scripts/build_f0_control.py`, `config/f0_control_stimuli.yaml`, `scripts/12_f0_control.py`,
`tests/test_f0_control.py` (offline). No team file is changed.
