# Context Length Sweep Results

**Checkpoint**: `500k_pl_31k_spqav2_199k_sqav3_126k_cls_100k_chatreg/final`
**Model**: Qwen/Qwen3-8B
**Date**: 2026-03-30

## MMLU Prediction (Binary — ROC AUC)

| segment_tokens | pre_answer AUC | post_answer AUC | time (s) |
| --- | --- | --- | --- |
| 10 (default) | 0.6990 | 0.7456 | 37 |
| 50 | 0.7172 | 0.7614 | 8 |
| full | 0.7378 | 0.8008 | 17 |

More context helps consistently. Post-answer is always better than pre-answer (as expected — it can see whether the model committed to the right letter). Full context gives a +0.039 AUC boost over default (pre) and +0.055 (post).

Note: accuracy_at_zero is ~0.50 across the board (balanced dataset), so AUC is the meaningful metric here.

## Number Prediction (Exact Match)

| segment_tokens | model_match_rate | true_match_rate | simple_2op | large_numbers | divmod | medium_3_4op | nested |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 (default) | 0.073 | 0.060 | 0.267 | 0.000 | 0.125 | 0.000 | 0.000 |
| full | **0.480** | **0.467** | **1.000** | **0.733** | **0.625** | 0.111 | 0.000 |

Massive improvement with full context: 7.3% → 48.0% model match rate. The AO can perfectly predict simple 2-operand calculations with full context (1.000). Large numbers and divmod also see huge gains. Medium 3-4 operand and nested problems remain very hard regardless of context.

## Backtracking (LLM Judge — 1-5 Scale)

| segment_tokens | mean_specificity | mean_correctness | specificity ≥3 rate | correctness ≥3 rate | time (s) |
| --- | --- | --- | --- | --- | --- |
| 20 (default) | 2.50 | 2.02 | 41.3% | 24.5% | 683 |
| 50 | 2.81 | 2.41 | 60.9% | 39.6% | 89 |
| 200 | 2.85 | 2.40 | 64.0% | 39.6% | 100 |
| full | **2.93** | **2.44** | **72.6%** | 39.6% | 241 |

The big jump is from 20 → 50 tokens. Going beyond 50 gives diminishing returns — mean correctness barely changes (2.41 → 2.44) and the ≥3 correctness rate is flat at 39.6%. Specificity continues to improve slightly with more context (≥3 rate: 41% → 61% → 64% → 73%).

## Summary

| Eval | Default tokens | Improvement with full context |
| --- | --- | --- |
| MMLU (post AUC) | 10 → 0.746 | full → 0.801 (+0.055) |
| Number prediction (match rate) | 10 → 7.3% | full → 48.0% (+40.7pp) |
| Backtracking (specificity) | 20 → 2.50 | full → 2.93 (+0.43) |

**Number prediction** benefits enormously from full context — the default 10 tokens simply don't contain enough information for the AO to determine the answer. **MMLU** sees moderate gains. **Backtracking** mostly saturates around 50 tokens with marginal improvements after that.
