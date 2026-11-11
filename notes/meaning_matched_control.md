# Meaning-matched control — results note

**Scope.** A descriptive probe, not a confirmatory test. 7 emoji/word pairs, two
models. Treat the numbers as a direction, not a p-value.

## Question

Does an emoji draw more attention than its *meaning* warrants? Each emoji is
placed in the same structured probe slot as its spelled-out form (e.g. 👍 vs
`thumbs up`) and scored on how much attention the following text sends back to the
slot.

## Method (reuses the existing pipeline, nothing modified)

- **Structured probe, fixed far block.** Both conditions go in an identical
  carrier with a fixed **64-token far query block**, so `|Q|` is the same across
  conditions and across spans of different length. (A simple carrier whose query
  set grows with span length is not comparable here.)
- **Per-token score.** `M / |S|`: received attention (summed over the far block)
  divided by the number of span tokens, so a fragmented multi-token emoji and a
  short word are compared per token.
- **Per-fragment re-score.** The same captured attention, scored one span token at
  a time (`received_attention` left un-summed over `k`). Consistency vs the pooled
  metric was checked at runtime: `max|Σ_k frag − received|` = 3.8e-6 (Pythia),
  4.8e-7 (Qwen).
- Code: `scripts/04_meaning_matched_control.py` (span-mean),
  `scripts/05_fragment_breakdown.py` (per-fragment). Offline test:
  `tests/test_meaning_matched_control.py`.

## Result 1 — span mean (M / |S|)

| model | mean Mpt(emoji) | mean Mpt(spelled) | mean Δ (per pair) |
|---|---|---|---|
| Pythia-410M | 0.165 | 0.175 | **−0.010** |
| Qwen2.5-0.5B | 0.252 | 0.192 | **+0.060** (all 7 pairs positive) |

The two models disagree, and it tracks **fragmentation**: Qwen keeps these emoji
as ~1 token (mild sink, emoji > meaning); Pythia shatters them into 3–6 byte
fragments, and the per-token average lands at ≈ / slightly below the meaning.

## Result 2 — per fragment (the important one)

On **Pythia the span mean hides a sink**. The emoji's attention is concentrated in
a single fragment, which the `/|S|` average dilutes away:

- Hottest emoji fragment is **position 0 in 6/7 pairs** and decodes to **`�`
  (U+FFFD) in 7/7** — the leading byte piece `ðŁ` (bytes `F0 9F`, shared by every
  `U+1F3xx–1F6xx` emoji), with **frag_mean ≈ 0.45, frag_max ≈ 50** per head.
- **Dilution 2.25×** (hottest fragment 0.371 vs span-mean 0.165).
- **mean Δ_maxfrag = +0.171 > 0**: the hottest emoji fragment out-attracts the
  hottest token of its spelled meaning — **the opposite of the span-mean read**.
- Exception: ⭐ (Δ_maxfrag −0.094), because `star` is itself a strong single token.

On **Qwen** the emoji are whole tokens, so the per-fragment view degenerates
(dilution 1.03×, Δ_maxfrag +0.063, winners are the emoji glyph itself).

## Takeaway

On both models the hottest emoji fragment out-attracts the hottest token of its
meaning. On Pythia the sink is **localized to the first byte-fragment** (`F0 9F` →
`�`); dividing by `|S|` averages it into the noise. A fragment-level view is
needed to see it.

## Caveats

- n = 7 pairs; descriptive, not the pre-registered test.
- Emoji sit at a line start (the structured-probe convention); not mid-sentence.
- `max` values far exceed 1 because the metric sums over the 64 far queries — a
  single strong sink head can dominate; the means are over all heads.

## Reproduce

```bash
# env python (conda not on PATH): C:\Users\aderi\anaconda3\envs\eas\python.exe
# set HF_HUB_OFFLINE=1 and PYTHONUTF8=1
python scripts/04_meaning_matched_control.py              # span-mean, Pythia
python scripts/04_meaning_matched_control.py replication  # span-mean, Qwen
python scripts/05_fragment_breakdown.py                   # per-fragment, Pythia
python scripts/05_fragment_breakdown.py replication       # per-fragment, Qwen
```
