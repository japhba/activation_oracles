# Number Prediction Eval Dataset — Creation Process

## Goal
Test whether the AO can predict the specific number a model is about to output from its internal activations, before any answer tokens are generated. Initial finding (eval2.md) showed the AO confabulates the same numbers ("10", "12") regardless of the true answer.

## Pipeline (`generate_dataset.py`)

Single script that runs end-to-end:

1. **Generate 50 arithmetic problems** (seed=42) across 5 categories:
   - **simple_2op** (10): Two operands, basic ops (e.g. `18 * 7`). Answers range [-4, 126].
   - **medium_3_4op** (15): 3-4 operands with parentheses. Answers range [-6730, 102752].
   - **large_numbers** (10): Two large operands (e.g. `935 * 170`). Answers range [-662, 454150].
   - **divmod** (8): Division and modulo (e.g. `314 // 16`). Answers range [0, 111].
   - **nested** (7): Complex nested expressions. Answers range [-9072, 8800].

2. **Run Qwen3-8B at temperature 0**, thinking disabled, prompt: "What is {expression}? Answer with just the number, nothing else."

3. **Tokenize answers** to label single-token vs multi-token (using whatever the Qwen tokenizer's BPE vocabulary produces for that number string).

4. **Save** `number_prediction_eval_dataset.json`.

Usage:
```
.venv/bin/python data_pipelines/number_prediction/generate_dataset.py
```

## Results
- **Model accuracy**: 22/50 (44%)
  - simple_2op: 10/10 (100%)
  - large_numbers: 7/10 (70%)
  - divmod: 4/8 (50%)
  - medium_3_4op: 1/15 (6.7%)
  - nested: 0/7 (0%)
- **Single-token answers**: 4 (only answers 0-9)
- **Multi-token answers**: 46

## Eval Design (for AO evaluation)
- No LLM judge needed — just exact number match
- Feed AO the model's activations on the prompt (before any answer tokens)
- Ask AO to predict what number the model will output
- Score: does AO's predicted number match `model_answer`?
- Separate results by `is_single_token_answer` for analysis
- `model_correct` field lets us analyze: does the AO predict the *correct* answer or the *model's* answer?

## Key Files
| File | Description |
|------|-------------|
| `generate_dataset.py` | End-to-end dataset generation script |
| `number_prediction_eval_dataset.json` | **Final eval dataset** |
