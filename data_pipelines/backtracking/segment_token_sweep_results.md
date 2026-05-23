# Backtracking Eval: Segment Token Sweep

**Date**: 2026-03-25
**Checkpoint**: `checkpoints_500k_pl_31k_spqav2_199k_sqav3_126k_cls/final`
**Model**: Qwen/Qwen3-8B
**Setup**: Single rollout, temperature 0, 197 entries, LLM judge (Haiku)

## Results

| seg_tokens | mean_specificity | mean_correctness | spec>=3 rate | spec>=4 rate | corr>=3 rate | corr>=4 rate |
|-----------|-----------------|-----------------|-------------|-------------|-------------|-------------|
| 20        | 2.505           | 1.953           | 0.417       | 0.177       | 0.224       | 0.099       |
| 50        | 2.801           | 2.347           | 0.612       | 0.255       | 0.362       | 0.163       |
| 200       | 2.868           | 2.416           | 0.690       | 0.228       | 0.401       | 0.178       |
| 400       | 2.954           | 2.533           | 0.685       | 0.269       | 0.421       | 0.228       |

## Key Observations

- The biggest jump is from 20 to 50 tokens (specificity +0.30, correctness +0.39).
- Diminishing returns after 50 tokens: 50->200 is +0.07/+0.07, and 200->400 is +0.09/+0.12.
- The eval is moderately sensitive to segment length, but most signal is captured by ~50-200 tokens.
- The default of 20 tokens was leaving meaningful performance on the table.

## Raw Data

Full per-entry results saved in `experiments/backtracking_segment_sweep/`.
