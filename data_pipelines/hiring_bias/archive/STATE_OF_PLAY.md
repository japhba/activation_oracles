# Hiring Bias Activations: State of Play

This document is meant to be a detailed handoff for other LLMs or collaborators. It combines:

- the raw behavioral facts from the `hiring_fairness` runs
- the definitions of the three probe datasets
- the original probe results
- later follow-up analyses on confounders and grouped holdouts
- the current best interpretation of what is and is not true

For a compact definition-only reference, see `data_pipelines/hiring_bias/PROBE_DATASET_GUIDE.md`.

## Executive Summary

The shortest honest summary is:

- Qwen3-8B does show real demographic bias in this hiring setup, especially pro-Black bias and much weaker, less coherent pro-Female bias.
- The bias is concentrated near the model's decision boundary. Across raw groups, spread and group-level uncertainty are extremely correlated.
- This makes some benchmark framings, especially `biased_vs_unbiased`, badly confounded.
- Even after cleaning the obvious row-level confounders, a linear probe on one raw activation vector is weak or near chance on the best single-input framings.
- But bias information is clearly present in activations in a counterfactual sense: matched difference probes are very strong.

So the strongest false statement is:

- "Bias is not in activations at all."

And the strongest incomplete statement is:

- "The single-state probe fails only because of uncertainty confounds."

The best current picture is:

- bias is present in activations as a small, context-dependent counterfactual effect
- many dataset framings are entangled with uncertainty
- there is little evidence for a strong, universal, resume-generalizing, single-state linear marker of "bias" on raw activations

## Key Files

Raw data and original artifacts:

- `data_pipelines/hiring_bias/Qwen3-8B/hiring_fairness_explore_200tok_no_cot_all_occupations.json`
- `data_pipelines/hiring_bias/Qwen3-8B/probe_activations.pt`
- `data_pipelines/hiring_bias/Qwen3-8B/matched_eval/activations.pt`
- `data_pipelines/hiring_bias/Qwen3-8B/matched_eval/low_vs_high_bias_activations.pt`
- `data_pipelines/hiring_bias/Qwen3-8B/matched_eval/summary.json`
- `data_pipelines/hiring_bias/PROBE_DATASET_GUIDE.md`

Follow-up analysis artifacts:

- `codex/runs/hiring_bias_audit_summary.json`
- `codex/runs/hiring_bias_balanced_raw_single_state.json`
- `codex/runs/hiring_bias_followup_checks.json`
- `codex/runs/hiring_bias_low_vs_high_per_demo_fast_singlethread.json`

## What The Raw Data Is

Base file:

- `data_pipelines/hiring_bias/Qwen3-8B/hiring_fairness_explore_200tok_no_cot_all_occupations.json`

Basic setup:

- model: `Qwen/Qwen3-8B`
- resumes truncated to 200 tokens
- all jobs for the matching occupation
- 4 demographic variants per `resume × job` group:
  - `MW`
  - `MB`
  - `FW`
  - `FB`

Stored metadata:

- `16100` total rows
- `4025` total `resume × job` groups
- `11` occupations

Each row stores that variant's first-token:

- `yes_prob`
- `no_prob`

Each group has group-level stats:

- `mean_p`: mean `P(Yes)` across the 4 variants
- `spread`: max `P(Yes)` minus min `P(Yes)` across the 4 variants
- `race_gap`
- `gender_gap`

Useful definitions used in later analysis:

- row-level uncertainty: `-abs(yes_prob - 0.5)`
- group-level uncertainty: `-abs(mean_p - 0.5)`

Larger values here mean "closer to 0.5" and therefore more uncertain.

## Raw Behavioral Findings

From the original run summary:

- White Male mean `P(Yes)`: `0.179`
- Black Male mean `P(Yes)`: `0.190`
- White Female mean `P(Yes)`: `0.180`
- Black Female mean `P(Yes)`: `0.191`

So the aggregate effect is:

- pro-Black overall
- very weak pro-Female overall

Groups with meaningful spread:

- `255 / 4025` groups have `spread > 0.10`

### The most important raw correlation

Across all `4025` groups:

- `corr(spread, group_uncertainty) = 0.939`

where:

- `group_uncertainty = -abs(mean_p - 0.5)`

Interpretation:

- groups near `mean_p = 0.5` tend to have very large demographic spread
- groups near `mean_p = 0` or `1` usually have tiny spread

This is the core reason that "bias vs unbiased" is hard to make clean in this dataset.

### How many low-spread uncertain groups exist?

Inside the uncertain regime `0.1 <= mean_p <= 0.9`:

- there are only `197` groups total
- minimum spread is `0.063`
- median spread is `0.560`

Counts of low-spread groups in that uncertain zone:

- `spread <= 0.05`: `0`
- `spread <= 0.10`: `1`
- `spread <= 0.20`: `10`
- `spread <= 0.30`: `35`

So if you want "middle `P(Yes)` but genuinely low-bias" negatives, this dataset has very little support.

## Is The Bias Coherent Or Just Noise?

This depends on whether you mean race or gender, and whether you condition on groups where bias is actually present.

### Race

Across groups with `spread > 0.10` (`255` groups):

- mean `race_gap = +0.167`
- positive race gap in `200`
- negative race gap in `55`

Across uncertain groups (`197` groups):

- mean `race_gap = +0.199`
- positive race gap in `155`
- negative race gap in `42`

Under a simple random-sign null, the sign asymmetry is extremely unlikely:

- `spread > 0.10`: exact two-sided binomial `p ≈ 1.6e-20`
- uncertain zone: exact two-sided binomial `p ≈ 2.0e-16`

So the race effect does not look like pure random fluctuation.

### Gender

Across groups with `spread > 0.10`:

- mean `gender_gap = +0.020`
- positive gender gap in `150`
- negative gender gap in `105`

Across uncertain groups:

- mean `gender_gap = +0.025`
- positive gender gap in `119`
- negative gender gap in `78`

This is weaker and much less coherent than the race effect. It is not obviously pure noise, but it is much closer to "small, sparse, heterogeneous effect" than the race bias is.

### Occupation heterogeneity

The direction is not identical across occupations.

Examples among groups with `spread > 0.10`:

- `BANKING` has negative mean race gap
- `CONSTRUCTION` and `HEALTHCARE` have negative mean gender gap
- `APPAREL` has mixed race signs

So the best description is:

- race bias: real, but sparse and concentrated near the boundary
- gender bias: weaker and more occupation-dependent

## The Three Probe Datasets

This section subsumes the standalone dataset guide. The three single-state probe datasets test different hypotheses and control different confounders.

### Shared activation extraction setup

The original probe artifacts all use the same basic extraction recipe:

- model: `Qwen/Qwen3-8B`
- context: `user_content` plus the model's binary answer `assistant_text = Yes/No`
- tokenization:
  - `add_generation_prompt=False`
  - `continue_final_message=True`
  - `enable_thinking=False`
- extracted layer percentages: `[25, 50, 75]`
- for Qwen3-8B this corresponds to layers `[9, 18, 27]`
- hidden size: `4096`
- concatenated 3-layer feature size: `12288`
- probe model: logistic regression
- original train/test recipe:
  - 10-fold `StratifiedKFold`
  - train-fold standardization
  - AUC on stitched out-of-fold probabilities

Depending on the artifact, the stored feature may be:

- mean pooled over all real token positions
- mean pooled over the last 10 token positions

### 1. `biased_vs_unbiased`

Source code:

- `data_pipelines/hiring_bias/extract_activations_and_probe.py`

Artifact:

- `data_pipelines/hiring_bias/Qwen3-8B/probe_activations.pt`

Stored artifact facts:

- `1256` total rows
- `628` positive, `628` negative
- occupations present in the saved artifact:
  - `CHEF`
  - `FINANCE`
  - `INFORMATION-TECHNOLOGY`
  - `TEACHER`

#### Label definition

This artifact labels a row by the spread of its `resume × job` group:

- positive `1` / `biased`: `spread > min_spread`
- negative `0` / `unbiased`: `spread <= 0.02`

In the script, `min_spread` defaults to `0.10`.

#### Matching rule

This dataset does not pair demographic counterfactuals. Instead it does same-person matching across jobs:

- define `resume_key = group_key` with the trailing `_j...` removed
- define `person_key = (resume_key, demo_key)`
- for each fixed resume variant, collect all jobs for that same named person
- keep that person only if they have:
  - at least one biased job
  - at least one unbiased job
- then add all biased jobs for that person to the positive pool
- add all unbiased jobs for that person to the negative pool
- downsample negatives if necessary to equalize class counts

So the matching unit is:

- same resume variant across different jobs

not:

- same resume/job across different demographic variants

#### Activation representation

This artifact stores one matrix:

- `activations`

It is:

- mean pooled over all real token positions
- concatenated across layers 9, 18, 27

There is no stored last-10-token variant in this artifact.

#### What this dataset is asking

- "Given one raw activation vector, can I tell whether this row comes from a biased group or an unbiased group?"

#### Main caveat

This framing is badly confounded by uncertainty. Because the model's bias is concentrated near the decision boundary, uncertainty alone is already a near-perfect classifier in this setup.

This is why `biased_vs_unbiased` is not a clean benchmark for a single-input bias detector.

### 2. `matched_eval`

Source code:

- `data_pipelines/hiring_bias/run_matched_eval.py`

Artifacts:

- `data_pipelines/hiring_bias/Qwen3-8B/matched_eval/activations.pt`
- `data_pipelines/hiring_bias/Qwen3-8B/matched_eval/summary.json`

Stored artifact facts:

- `580` total rows
- `290` favored, `290` disfavored
- occupations present in the saved artifact:
  - `APPAREL`
  - `AVIATION`
  - `BANKING`
  - `CHEF`
  - `CONSTRUCTION`
  - `CONSULTANT`
  - `FINANCE`
  - `FITNESS`
  - `HEALTHCARE`
  - `INFORMATION-TECHNOLOGY`
  - `TEACHER`

Saved artifact settings from `summary.json`:

- `min_spread = 0.05`
- `bin_size = 0.10`

Important note:

- the current script defaults are not exactly the same as the saved artifact settings
- for the saved artifact, `summary.json` is the source of truth

#### Label definition

This dataset first restricts to already-biased groups:

- keep only groups with `spread > min_spread`

Then within each such group:

- compute `group_mean_p = mean yes_prob` across the 4 variants
- assign:
  - `favored` if `yes_prob > group_mean_p`
  - `disfavored` otherwise

So this is not "biased vs unbiased." It is:

- "Within already-biased groups, which variants were above or below the group mean?"

#### Matching rule

After labeling, the script globally matches favored and disfavored rows by row-level `P(Yes)` bin:

- bin favored rows by `int(yes_prob / bin_size)`
- bin disfavored rows the same way
- in each bin, keep the same number from both sides

This removes most row-level `P(Yes)` and row-level uncertainty signal, but it does not balance race.

#### Activation representation

This artifact stores two matrices:

- `mean_activations`
- `last_activations`

They are:

- `mean_activations`: mean pooled over all real positions in the full `user + assistant` sequence
- `last_activations`: mean pooled over the last 10 positions in that sequence

Both use the same 3-layer concatenated representation.

#### What this dataset is asking

- "Given one raw activation vector from an already-biased group, can I tell whether this demographic variant was favored or disfavored?"

#### Main caveat

Race is strongly correlated with the label in the saved artifact. The saved demographic counts are:

- favored `FB`: `104`
- favored `MB`: `100`
- favored `FW`: `43`
- favored `MW`: `43`
- disfavored `FB`: `47`
- disfavored `MB`: `43`
- disfavored `FW`: `100`
- disfavored `MW`: `100`

So:

- race is a strong confound
- gender is not, in this artifact

This is still the cleanest original single-state framing, but it is not fully clean.

### 3. `low_vs_high`

Artifact:

- `data_pipelines/hiring_bias/Qwen3-8B/matched_eval/low_vs_high_bias_activations.pt`

Stored artifact facts:

- `628` total rows
- `314` positive, `314` negative
- occupations present in the saved artifact:
  - `APPAREL`
  - `AVIATION`
  - `BANKING`
  - `CHEF`
  - `CONSTRUCTION`
  - `CONSULTANT`
  - `FINANCE`
  - `FITNESS`
  - `HEALTHCARE`
  - `INFORMATION-TECHNOLOGY`
  - `TEACHER`
- spread range in the artifact:
  - min `0.0632`
  - median `0.5600`
  - max `0.9999`

Important note:

- the exact constructor script for the saved artifact is not present in the current repo
- the reconstruction below comes from the artifact contents plus `data_pipelines/hiring_bias/RESULTS_SUMMARY.md`

#### Label definition

This framing is defined only inside the uncertain regime:

- restrict to groups with `0.1 <= mean_p <= 0.9`

Then:

- positive `1` / `high_bias`: group spread above the median
- negative `0` / `low_bias`: group spread below the median

In the saved artifact, the split is around median spread `0.56`.

#### Matching rule

The reported construction matches on:

- `demo_key`
- row-level `yes_prob` bins

So race/gender and row-level `P(Yes)` are balanced by design.

#### Activation representation

This artifact stores one matrix:

- `activations`

It is:

- mean pooled over all real token positions
- concatenated across layers 9, 18, 27

There is no stored last-10-token variant in this artifact.

#### What this dataset is asking

- "Given one raw activation vector from the uncertain regime, can I tell whether it came from a low-spread or high-spread group?"

#### Main caveat

The label is group-level, but the matching is row-level.

That means the dataset can look balanced on:

- row `yes_prob`
- row uncertainty
- race
- gender

while still leaking a strong group-level confound:

- `group_uncertainty = -abs(mean_p - 0.5)`

In later audits, group-level uncertainty alone still predicted the `low_vs_high` label with AUC about `0.855`.

## Original Probe Results

### `biased_vs_unbiased`

Original and later audits agree on the main point:

- the dataset is dominated by uncertainty
- high AUC here is not meaningful evidence of bias detection

Later audit numbers:

- row-level uncertainty baseline AUC: `0.978`
- activation probe standard CV AUC: `0.952`
- activation probe resume-held-out AUC: `0.602`

That `0.602` should not be interpreted as a clean bias readout because the framing itself is not clean.

### `matched_eval`

Original saved summary:

- linear probe last-10 standard CV AUC: `0.603`
- `P(Yes)` confounder AUC: `0.516`
- uncertainty confounder AUC: `0.502`
- race confounder AUC: `0.697`

Later grouped evaluation on raw single-state activations:

- mean-pooled standard CV AUC: `0.546`
- mean-pooled resume-held-out AUC: `0.499`
- last-10 standard CV AUC: `0.603`
- last-10 resume-held-out AUC: `0.498`

Cleaned exact-balanced re-eval within `demo_key × yes_prob_bin`:

- best raw single-state result is only about `0.525` to `0.539`
- most feature variants remain near chance

So the consistent picture is:

- standard CV can look mildly positive
- resume-held-out raw single-state performance collapses to weak or chance-level

### `low_vs_high`

Original report:

- standard CV AUC: `0.994`
- group-level CV AUC: `0.487`

Later audits found that one split can vary, but the overall picture remains weak:

- one resume-held-out split on the original artifact: `0.539`
- 20-seed grouped sweep: about `0.504 ± 0.020`
- cleaned exact-balanced single-state re-eval: about `0.496` to `0.508`

So the robust conclusion is:

- there is no strong, stable, resume-generalizing single-state signal here

## Counterfactual-Difference Results

These are not the intended single-input tool, but they matter because they show the bias information is in the activations in some form.

Using matched favored-vs-disfavored pairs from the same `resume × job` group:

- post-answer pair-difference resume-held-out AUC: `0.981`
- pre-answer pair-difference resume-held-out AUC: `0.965`
- pre-answer same-answer resume-held-out AUC: `0.946`

That means:

- the signal is already present before the answer token
- it is not just reading whether the model literally outputs `Yes` or `No`
- bias information is clearly present in a counterfactual sense

So the strong false statement is:

- "There is no bias signal in activations."

The more precise true statement is:

- "There is no strong, reusable, single-state absolute linear marker of bias in these raw activations."

## Key Correlation And Confound Findings

### `biased_vs_unbiased`: uncertainty dominates

This is the clearest confounded case.

Even after balancing row-level `yes_prob`, later audits found:

- row-level uncertainty AUC stayed about `0.979`

So this dataset cannot cleanly answer whether a single activation reveals bias.

### `low_vs_high`: row-level confounders can be balanced, but group-level uncertainty remains

In the cleaned exact-balanced subset:

- row-level `yes_prob` AUC: `0.496`
- row-level uncertainty AUC: `0.495`
- race AUC: `0.500`
- gender AUC: `0.500`

So at the row level it looks clean.

But if you join back to the raw group stats and use:

- `group_uncertainty = -abs(mean_p - 0.5)`

then that scalar alone predicts the `low_vs_high` label with:

- AUC `0.855`

So `low_vs_high` is balanced on the wrong level if the goal is a clean single-input benchmark.

### `matched_eval`: race is a real label correlation

For the saved artifact:

- race AUC: `0.697`
- gender AUC: `0.500`

Demo counts:

- favored `FB`: `104`
- favored `MB`: `100`
- favored `FW`: `43`
- favored `MW`: `43`

- disfavored `FB`: `47`
- disfavored `MB`: `43`
- disfavored `FW`: `100`
- disfavored `MW`: `100`

So:

- race is a strong confound
- gender is not, in this artifact

This matches the raw-data picture: the race effect is much more coherent than the gender effect.

## Does Restricting To One Demographic Help?

Yes, a little, but not dramatically.

Fast per-demo single-state checks on `low_vs_high` gave:

- `FB`: `0.582` post-answer, `0.533` pre-answer
- `MB`: `0.546` post-answer, `0.549` pre-answer
- `FW`: `0.497` post-answer, `0.483` pre-answer
- `MW`: `0.512` post-answer, `0.494` pre-answer

Interpretation:

- mixing demographic subpopulations can wash out a modest signal
- but this is not a dramatic rescue
- the best subgroup is still only around `0.58`, nowhere near the counterfactual-difference results

## What I Think Is Closest To The Truth

There were two candidate interpretations:

1. the model is behaviorally biased, but the bias is not present in activations at all
2. the model's bias is tightly entangled with uncertainty or related confounds, so once those are controlled, a linear probe loses its easy signal

My current view is:

- interpretation 2 explains a lot of the dataset-level mess, especially for `biased_vs_unbiased` and `low_vs_high`
- but interpretation 1 is too strong, because counterfactual-difference probes are very strong
- there is a third view that fits the evidence better:

> bias in this setup behaves like a counterfactual sensitivity near the decision boundary, and that sensitivity does not appear as a large, universal, single-state linear marker on raw activations

That reconciles the main observations:

- why the model can be clearly biased behaviorally
- why uncertainty is so entangled with spread
- why difference probes work
- why one-shot raw-state probes stay weak

## The Two Main Concerns You Flagged

### Concern A: uncertainty makes it hard to build a reasonable balanced dataset

I think this concern is absolutely real.

The evidence is:

- spread and group-level uncertainty are extremely correlated in the raw data
- genuinely low-spread uncertain groups are rare
- balancing row-level `yes_prob` does not remove group-level uncertainty leakage

This means many superficially "balanced" datasets in this domain are only balanced at the wrong level.

### Concern B: the bias may not be very coherent

I think this concern is partly right.

More precise version:

- race bias is coherent enough to clearly reject "just random fluctuation"
- gender bias is much weaker and more heterogeneous
- neither effect looks like a large universal activation feature that generalizes cleanly across resumes

So "not coherent" is too strong for race, but it is directionally right for gender and for the idea of one universal linear activation signature.

## What Would Actually Clarify Things Next

If the goal is a real skyline for a single-input tool, the most useful next experiments are:

1. Build a benchmark matched on group `mean_p`, not just row `yes_prob`

Why:

- this would directly control the group-level uncertainty confound that still leaks into `low_vs_high`

2. Try stricter low-vs-high spread splits with explicit support accounting

Examples in the uncertain zone:

- `spread <= 0.20` vs `>= 0.80` gives only `10` low groups
- `spread <= 0.35` vs `>= 0.75` gives `46` balanced groups

This makes the sample-size tradeoff explicit.

3. Continue per-demo evaluations with stronger holdouts

Why:

- `FB` and `MB` show modest signal
- `FW` and `MW` do not
- this may reveal whether demographic mixing is obscuring weak single-state structure

4. Try richer but still single-input linear features

Examples:

- multiple token-position summaries
- prompt-only and post-answer features together
- layerwise summaries that do not collapse everything into a single pooled vector

This would still respect the "single input only" constraint while testing whether the current pooling is too lossy.

## Bottom Line

If this needed to be compressed into a few sentences for another model:

- The behavioral bias is real, especially for race, but it is sparse and concentrated near the decision boundary.
- Several dataset framings are entangled with uncertainty in ways that make them poor clean benchmarks for single-input bias detection.
- After cleaning the obvious row-level confounders, raw single-state linear probes remain weak.
- But counterfactual-difference probes are very strong, so the bias information is in activations as a relational effect rather than as a strong absolute-state marker.
