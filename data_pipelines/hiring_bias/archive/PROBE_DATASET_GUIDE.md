# Hiring Bias Probe Dataset Guide

This file explains the three probe datasets that keep getting conflated:

1. `biased_vs_unbiased` from `Qwen3-8B/probe_activations.pt`
2. `matched_eval` from `Qwen3-8B/matched_eval/activations.pt`
3. `low_vs_high` from `Qwen3-8B/matched_eval/low_vs_high_bias_activations.pt`

The short version is:

- `biased_vs_unbiased` asks whether one raw activation comes from a biased group or an unbiased group. It is heavily confounded by uncertainty and is not a clean benchmark.
- `matched_eval` asks whether one raw activation is the favored or disfavored demographic variant within an already-biased group. This is the cleanest original single-state benchmark.
- `low_vs_high` asks whether one raw activation comes from a group with low or high demographic spread, restricted to the uncertain regime. It balances row-level `P(Yes)` and demographic key, but group-level uncertainty still leaks through.

## Quick Summary

| Dataset | Positive label | Negative label | What is balanced? | Main caveat |
|---|---|---|---|---|
| `biased_vs_unbiased` | Entry from group with `spread > min_spread` | Entry from group with `spread <= 0.02` | Class count only, plus same-person matching across jobs | Group uncertainty is almost perfectly predictive |
| `matched_eval` | Favored entry within a biased group | Disfavored entry within a biased group | Global row-level `P(Yes)` bin matching | Race remains correlated with label |
| `low_vs_high` | Entry from high-spread group | Entry from low-spread group | Demographic key + row-level `P(Yes)` bins | Group mean uncertainty still predicts the label |

## Shared Raw Source

All three datasets are derived from:

- [hiring_fairness_explore_200tok_no_cot_all_occupations.json](/workspace-vast/adamk/activation_oracles_dev/data_pipelines/hiring_bias/Qwen3-8B/hiring_fairness_explore_200tok_no_cot_all_occupations.json)

Basic structure:

- A `group_key` is one `resume × job` scenario.
- Each group has 4 demographic variants: `MW`, `MB`, `FW`, `FB`.
- Each row stores that variant's first-token `yes_prob` and `no_prob`.
- Group-level stats are stored separately in `group_stats`, including:
  - `mean_p`: mean `P(Yes)` across the 4 variants
  - `spread`: max `P(Yes)` minus min `P(Yes)` across the 4 variants
  - `race_gap`, `gender_gap`

Useful terminology:

- Row-level uncertainty: `-abs(yes_prob - 0.5)`
- Group-level uncertainty: `-abs(mean_p - 0.5)`

When this project says it matched on `P(Yes)`, that usually means the row-level value `yes_prob`, not the group-level mean `mean_p`.

## Common Activation Extraction Setup

The original probe artifacts all use the same general extraction pattern:

- Model: `Qwen/Qwen3-8B`
- Chat context: `user_content` plus the model's binary answer (`assistant_text = Yes/No`)
- Tokenization:
  - `add_generation_prompt=False`
  - `continue_final_message=True`
  - `enable_thinking=False`
- Layers: percentages `[25, 50, 75]`
  - For Qwen3-8B this becomes layers `[9, 18, 27]`
- Hidden size: `4096`
- A 3-layer concatenated feature is therefore `12288` dimensions
- Probe model: logistic regression
- Standard evaluation in the original scripts: 10-fold `StratifiedKFold`, train-fold standardization, AUC on out-of-fold scores

Two pooling variants appear in the original artifacts:

- Mean-pool over all real token positions in the full `user + assistant` sequence
- Mean-pool over the last 10 token positions

## Dataset A: `biased_vs_unbiased`

Source code:

- [extract_activations_and_probe.py](/workspace-vast/adamk/activation_oracles_dev/data_pipelines/hiring_bias/extract_activations_and_probe.py)

Artifact:

- [probe_activations.pt](/workspace-vast/adamk/activation_oracles_dev/data_pipelines/hiring_bias/Qwen3-8B/probe_activations.pt)

Stored artifact facts:

- `1256` total entries
- `628` positive and `628` negative
- Occupations in the saved artifact: `CHEF`, `FINANCE`, `INFORMATION-TECHNOLOGY`, `TEACHER`

### Label definition

The script groups entries by `group_key`, reads the corresponding `spread`, and then does:

- Positive label `1` / `biased`: `spread > min_spread`
- Negative label `0` / `unbiased`: `spread <= 0.02`

In the original script, `min_spread` defaults to `0.10`.

### Matching rule

This dataset does not match counterfactual pairs. It does "same-person matching" across jobs:

- Define `resume_key = group_key` with the trailing `_j...` removed
- Define `person_key = (resume_key, demo_key)`
- For each fixed person variant, collect all jobs for that same resume/name variant
- Keep that person only if they have:
  - at least one biased job
  - at least one unbiased job
- Then add all biased jobs for that person to the positive pool and all unbiased jobs to the negative pool
- Downsample negatives if needed to equalize class counts

So the pairing unit is "same resume variant across different jobs", not "same resume/job across different demographics".

### Activation representation

This artifact stores a single activation matrix under `activations`:

- Mean-pool over all real token positions
- Concatenate layers 9, 18, 27
- No last-10-token variant is stored

### What it is trying to measure

This is asking:

- "Given one raw activation vector, can I tell whether this row comes from a biased group or an unbiased group?"

### Main caveat

This framing is badly confounded by uncertainty. In this dataset, bias mostly appears when the model is near its decision boundary, so a detector of uncertainty can already do extremely well.

That is why this is not a clean benchmark for a single-input bias detector.

## Dataset B: `matched_eval`

Source code:

- [run_matched_eval.py](/workspace-vast/adamk/activation_oracles_dev/data_pipelines/hiring_bias/run_matched_eval.py)

Artifact:

- [activations.pt](/workspace-vast/adamk/activation_oracles_dev/data_pipelines/hiring_bias/Qwen3-8B/matched_eval/activations.pt)
- [summary.json](/workspace-vast/adamk/activation_oracles_dev/data_pipelines/hiring_bias/Qwen3-8B/matched_eval/summary.json)

Stored artifact facts:

- `580` total entries
- `290` favored and `290` disfavored
- Occupations in the saved artifact: `APPAREL`, `AVIATION`, `BANKING`, `CHEF`, `CONSTRUCTION`, `CONSULTANT`, `FINANCE`, `FITNESS`, `HEALTHCARE`, `INFORMATION-TECHNOLOGY`, `TEACHER`
- The saved artifact summary says it used:
  - `min_spread = 0.05`
  - `bin_size = 0.10`

Important note:

- The current script defaults are not identical to the saved artifact defaults.
- For the saved artifact, trust [summary.json](/workspace-vast/adamk/activation_oracles_dev/data_pipelines/hiring_bias/Qwen3-8B/matched_eval/summary.json).

### Label definition

This dataset first restricts to biased groups:

- Keep only groups with `spread > min_spread`

Then within each such group:

- Compute `group_mean_p = mean yes_prob` across the 4 variants
- Label an entry as:
  - `1` / `favored` if `yes_prob > group_mean_p`
  - `0` / `disfavored` otherwise

So this is not "biased vs unbiased". It is:

- "Within already-biased groups, which variants were above or below the group mean?"

### Matching rule

After labeling, it balances favored and disfavored entries using global row-level `P(Yes)` bins:

- Put all favored entries into bins by `int(yes_prob / bin_size)`
- Put all disfavored entries into the same bins
- In each bin, keep the same number from each side

This removes most row-level `P(Yes)` and row-level uncertainty signal, but it does not force race balance.

### Activation representation

This artifact stores two feature matrices:

- `mean_activations`: mean over all token positions in the full `user + assistant` sequence
- `last_activations`: mean over the last 10 token positions

Both use the 3-layer concatenated representation.

### What it is trying to measure

This is asking:

- "Given one raw activation vector from an already-biased group, can I tell whether this demographic variant was favored or disfavored?"

### Main caveat

This is the cleanest original single-state framing, but race is still correlated with the label in the saved artifact.

This is also the framing where later by-resume evaluation was closest to exact chance for raw single-state probes.

## Dataset C: `low_vs_high`

Artifact:

- [low_vs_high_bias_activations.pt](/workspace-vast/adamk/activation_oracles_dev/data_pipelines/hiring_bias/Qwen3-8B/matched_eval/low_vs_high_bias_activations.pt)

Background description:

- [RESULTS_SUMMARY.md](/workspace-vast/adamk/activation_oracles_dev/data_pipelines/hiring_bias/RESULTS_SUMMARY.md)

Stored artifact facts:

- `628` total entries
- `314` positive and `314` negative
- Spread range in the saved artifact:
  - min `0.0632`
  - median `0.5600`
  - max `0.9999`

Important note:

- The exact constructor script that produced this saved artifact is not present in the current repo.
- The description below is reconstructed from the artifact contents and [RESULTS_SUMMARY.md](/workspace-vast/adamk/activation_oracles_dev/data_pipelines/hiring_bias/RESULTS_SUMMARY.md).

### Label definition

This framing works only inside the uncertain regime:

- Restrict to groups with `mean_p` in `[0.1, 0.9]`

Among those groups:

- Positive label `1` / `high_bias`: group spread above the median
- Negative label `0` / `low_bias`: group spread below the median

In the saved artifact, the median split is about `0.56`.

### Matching rule

The reported construction matches on:

- `demo_key`
- Row-level `yes_prob` bins

So race/gender and row-level `P(Yes)` are balanced by design.

### Activation representation

This artifact stores a single activation matrix under `activations`:

- Mean-pool over all real token positions
- Concatenate layers 9, 18, 27

No last-10-token representation is stored in this artifact.

### What it is trying to measure

This is asking:

- "Given one raw activation vector from the uncertain regime, can I tell whether it came from a low-spread group or a high-spread group?"

### Main caveat

The balancing is done on row-level `yes_prob`, but the label is a group-level property derived from `spread`.

That means a strong group-level confound can remain even when row-level confounders look balanced. In later audits, this showed up clearly:

- Row-level `yes_prob` AUC was near `0.50`
- Row-level uncertainty AUC was near `0.50`
- But group-level uncertainty `-abs(mean_p - 0.5)` still predicted the label with AUC about `0.855`

So this dataset is cleaner than `biased_vs_unbiased`, but it is still not a fully clean single-input benchmark.

## How My Later Audits Differed

The original scripts mostly use standard 10-fold stratified CV. My later audits added stronger tests:

- Resume-held-out CV:
  - group by `resume_key`
  - train on some resumes, test on unseen resumes
- Leave-one-occupation-out CV
- Exact balancing within `demo_key × yes_prob_bin`
- Prompt-only pre-answer activations
- Alternate layer sweeps

Those later audits do not change the original dataset definitions. They only change:

- which subset is evaluated
- which activation representation is used
- how train/test splitting is done

## Which Dataset Matters For Which Question

If your question is:

- "Can a single raw activation detect whether the model is biased or unbiased?"
  - `biased_vs_unbiased` is the most direct framing, but it is also the most confounded.

- "Can a single raw activation tell whether this entry was favored or disfavored within a biased group?"
  - `matched_eval` is the cleanest original framing.

- "Can a single raw activation tell whether this example comes from a low-bias or high-bias group in the uncertain regime?"
  - `low_vs_high` is the intended framing, but it still inherits strong group-level uncertainty structure.

If your question is:

- "Is bias present in activations at all?"

then none of these three definitions alone is sufficient, because they are all single-state tasks. Counterfactual-difference probes answer a different question.
