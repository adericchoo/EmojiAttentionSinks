# Master Stimulus Set: Project Handoff

Use this note to understand the master stimulus set and the copied runners. The repository is the EmojiAttentionSinks project.

## Branch and commits

- The master config, builder, and runner copies are on `main`.
- Master config and builder: commit `21204fe` (`Add master stimulus set and builder`)
- Master-config runner copies: commit `e810d75` (`Add master-config pipeline script copies`)
- This handoff explains how to fetch, install dependencies, and run the copied scripts.

To get the project from GitHub:

```powershell
git clone https://github.com/adericchoo/EmojiAttentionSinks.git
cd EmojiAttentionSinks
```

If the repository is already cloned:

```powershell
git fetch origin
git switch main
git pull origin main
```

## What the master set is

`config/master_stimuli.yaml` is the tokenizer-audited stimulus set. It has 115 main stimuli because the requested group targets add up to 115 (60 + 20 + 20 + 15). `R_reference` contains baselines and controls; it is not part of that count.

The config was verified with the pinned `EleutherAI/pythia-410m` tokenizer at revision `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f`. It records expected and actual token counts, token splits, token byte leads, glosses, and generated controls. Its content SHA-256 is `866ea758fbecf0467aae0c29f5b8929c220afba52fb3a88afd2471735a76dfe5`.

Group counts:

| Group | Count | Purpose |
|---|---:|---|
| `A_smp_pictographs` | 60 | Single-codepoint supplementary-plane pictographs, spread over five Unicode blocks |
| `B_bmp_e2_symbols` | 20 | E2-leading BMP symbols that fragment into multiple Pythia tokens; `stratum` identifies the requested `B_bmp_e2_fragmenting` group |
| `C_multi_f0_9f` | 20 | Skin-tone sequences, ZWJ/family sequences, and regional-indicator flags |
| `D_nonemoji_e2` | 15 | E2-leading, single-token symbol controls; `stratum` identifies `D_rare_whole_token` |
| `R_reference` | 4 fixed baselines plus 60 generated control pairs | Ordinary words, a fixed hex baseline, and controls linked to group A |

There are 15 group-A `gloss` values for the meaning-matched runner. Each group-A item has a `control_id`; its shared hex and rare-character controls live under `R_reference.generated_controls`.

## Builder and review output

The builder is `scripts/build_master_stimuli.py`. It loads the pinned Pythia tokenizer, checks byte/token splits and round trips, generates deterministic controls, writes the config, and prints a summary.

To inspect a fresh summary without overwriting the committed config:

```powershell
python scripts/build_master_stimuli.py --output outputs/master_stimuli_review.yaml
```

The recorded build summary had 13 expected-versus-actual token-count mismatches and 8 dropped candidates. The mismatches are retained and flagged; they reflect differences between the predeclared byte-split expectation and actual Pythia tokenization. Dropped candidates either were not fully-qualified Unicode 15.1 sequences or did not meet the required single-token property. Read the printed summary before changing the frozen list.

The builder and copied scripts compiled successfully. No full model experiment has been run for this master set.

## Master-config runner copies

Use the `_master.py` copies below. The originals are preserved. Run commands from the repository root:

```powershell
python scripts/04_meaning_matched_control_master.py
python scripts/06_byte_structure_master.py
python scripts/07_narrow_sink_master.py
python scripts/08_value_state_drain_master.py
python scripts/09_attention_theft_master.py
```

These scripts use the primary model configured in `config/models.yaml`. To select the configured replication model, add `replication` after the script path, for example:

```powershell
python scripts/06_byte_structure_master.py replication
```

`04` reads only A items with a `gloss`. `06`, `07`, and `08` use the master stimulus groups and fixed reference items. `09` reads the saved per-emoji hex and rare-character controls and checks their token counts in each carrier before measuring them. Those controls were generated for the Pythia tokenizer; another tokenizer may fail the runtime count check.

The full repository is needed because the copied scripts reuse shared project modules, including `scripts/05_fragment_breakdown.py`, `scripts/_bootstrap.py`, `scripts/c2.py`, and `src/eas/`.

## Environment and model files

Install the Python packages from `requirements.txt`. Install PyTorch separately with a build compatible with the machine's CPU/GPU; the requirements file intentionally does not pin a CUDA-specific PyTorch wheel. The builder needs internet access or a local Hugging Face cache for the pinned tokenizer. Running an experiment also needs access to the selected model weights and enough memory for that model.

The committed `config/models.yaml` includes `primary` (Pythia-410M) and `replication` (Qwen2.5-0.5B). Larger Pythia entries that may exist in another developer's local, uncommitted work are not guaranteed to be present on this branch. Add and pin a model entry before asking the runner to select it.

## Important scope note

The original runner files were not changed for the master-set work. The copies are the entry points that load `master_stimuli.yaml`. `06`–`08` use the shared stimulus and reference groups; the per-emoji generated control pairs are specifically consumed by `09`. If the study design requires those generated controls to be measured by `06`–`08` too, that needs a separate, explicit change.

Do not silently change the selected items, tokenizer revision, controls, or selection rules after treating this version as preregistered. Record any later changes as a new version with a new content hash.
