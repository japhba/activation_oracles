# MMLU Prediction Eval Dataset — Creation Process

## Goal
Test whether the AO can predict if Qwen3-8B will answer an MMLU question correctly or not, based on the model's internal activations before it produces an answer. This is a binary classification task (correct/incorrect) read from activations — complementary to the number prediction eval which tests if the AO can read the *specific* answer.

## Pipeline (`generate_dataset.py`)

Single script that runs end-to-end:

1. **Download full MMLU test set** (14,042 questions across all subjects) via HuggingFace `cais/mmlu`.
2. **Run Qwen3-8B at temperature 0**, thinking disabled, prompt: multiple-choice question formatted with lettered choices + "Answer with just the letter (A, B, C, or D), nothing else."
3. **Parse model answers** — extract first A/B/C/D letter from response, skip unparseable.
4. **Sample 100 balanced entries** — 50 correct + 50 incorrect, shuffled randomly.
5. **Save** `mmlu_prediction_eval_dataset.json`.

Usage:
```
.venv/bin/python data_pipelines/mmlu_prediction/generate_dataset.py
```

## Results
- **Full MMLU accuracy** (no thinking): 9,585/13,235 parseable = **72.4%**
  - 807/14,042 responses were unparseable (model didn't output a clean letter)
- **Dataset**: 100 entries (50 correct, 50 incorrect) across 34 subjects
- Top subjects: professional_law (15), moral_scenarios (9), high_school_psychology (6)

## Eval Design (for AO evaluation)
Two evaluation modes with different context construction:

### Pre-answer mode (AO sees question only, empty assistant turn)
- **predict_letter**: "What answer letter (A, B, C, or D) will the model choose?" → letter match scoring

### Post-answer mode (AO sees question + model's single-letter answer in assistant turn)
- **post_is_correct**: "Is the model's answer correct? Answer 'yes' or 'no'." → binary yes/no scoring
- **post_confidence**: "Is the model confident in its answer? Answer 'yes' or 'no'." → binary yes/no scoring

The post-answer mode tests whether the AO can read confidence signals from activations. The AO doesn't have access to the full question content (only activations), so it can't just look up the answer — it must detect internal confidence patterns.

Key metrics:
- `accuracy`: does AO's prediction match ground truth?
- Breakdown by prompt type, by subject, by model_correct (to detect bias toward always saying yes/no)
- For predict_letter: separate `matches_true_rate` to see if AO predicts the *correct* answer vs the *model's* answer

## Key Files
| File | Description |
|------|-------------|
| `generate_dataset.py` | End-to-end dataset generation script |
| `mmlu_prediction_eval_dataset.json` | **Final eval dataset** (100 entries) |
| `spot_check.py` | Interactive validation notebook |
