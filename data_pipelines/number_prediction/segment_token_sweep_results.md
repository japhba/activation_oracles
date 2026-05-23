# Number Prediction Eval: Segment Token Sweep

**Date**: 2026-03-25
**Checkpoint**: `checkpoints_500k_pl_31k_spqav2_199k_sqav3_126k_cls/final`
**Model**: Qwen/Qwen3-8B
**Setup**: Single rollout, temperature 0, 50 entries (150 scored = 50 entries x 3 prompt variants)
**Context lengths**: 25-41 tokens (mean 30.8)

## Results

| seg_tokens | model_match_rate | true_match_rate | has_number_rate |
|-----------|-----------------|-----------------|-----------------|
| 5         | 0.067           | 0.067           | 1.000           |
| 10        | 0.087           | 0.067           | 1.000           |
| 20        | 0.133           | 0.127           | 1.000           |
| 30        | 0.427           | 0.467           | 1.000           |
| 40        | 0.433           | 0.487           | 1.000           |

## Key Observations

- Massive jump from 20 to 30 tokens (model match 0.133 -> 0.427, a 3x improvement).
- Minimal gain from 30 to 40 (0.427 -> 0.433), suggesting the signal saturates around 30 tokens.
- At 30 tokens, the segment covers most of the context (mean context is ~31 tokens), so the AO is seeing nearly the full prompt.
- The default of 10 tokens was severely limiting — the AO was only seeing the tail of the prompt without the actual math expression.
- The sharp threshold around 20-30 tokens likely corresponds to whether the segment includes the user's question content or just the chat template tokens.
- At seg=30, the `simple_2op` category hits 100% model match, while `nested` stays at 0% — the AO can extract simple answers but not complex ones.

## Raw Data

Full per-entry results saved in `experiments/number_prediction_segment_sweep/`.
