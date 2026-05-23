# MMLU Prediction Prompt Selection

## Task
Predict whether the target model's answer to an MMLU question is correct or not,
using binary (yes/no) logit scoring on the AO's output.

## Selected Prompts
**Pre-answer:**
```
Answer with 'Yes' or 'No' only. Will the model likely answer this correctly?
```

**Post-answer:**
```
Answer with 'Yes' or 'No' only. Is the model's answer likely to be correct?
```

## Why These Prompts

### Round 1: Initial sweep (experiments/mmlu_prompt_sweep/)
Tested 6 post-answer prompts on 2 LoRAs (500k, past_lens_addition):
- "correct/confident" prompts (confident_in_answer, very_confident, is_correct, likely_correct)
  got AUC 0.62-0.74 but always 50% accuracy — AO always leans "yes."
- "uncertain" prompts (uncertain, at_all_uncertain) were anti-correlated (AUC 0.29-0.54).
  Negation framing confuses the AO.
- Best: `post_likely_correct` AUC=0.742 (500k), `post_is_correct` AUC=0.725.

### Round 2: Behavioral/reframed prompts
Tested 8 prompts including "guessing", "struggling", "recognizing", "lucky guess",
"completely sure", "any chance wrong":
- `post_lucky_guess` ("Did the model know the answer or was it a lucky guess?") got
  highest single AUC: 0.764 (500k).
- `post_completely_sure` got 0.751.
- Negative framings still anti-correlated: `any_chance_wrong` AUC=0.366, `struggling` AUC=0.542.
- Accuracy still stuck at 50% across all prompts.

### Round 3: Pre-answer prompts
Tested 8 pre-answer prompts (AO sees question only, no model answer):
- Pre-answer AUC surprisingly competitive with post-answer (best: 0.722 vs 0.764).
- `pre_will_correct` most consistent: AUC=0.722 on both LoRAs.
- `pre_confident_knows` best on past_lens (0.733) but inconsistent across LoRAs.
- Negative framings (`guessing`, `struggling`) again near chance or anti-correlated.

### Round 4: Cross-LoRA consistency check
Tested 4 finalists across 4 LoRAs (on_policy, 250k, 500k, past_lens_addition):

| Prompt | on_policy | 250k | 500k | past_lens |
|--------|-----------|------|------|-----------|
| `post_lucky_guess` | 0.623 | 0.731 | **0.763** | 0.668 |
| `post_likely_correct` | 0.597 | 0.727 | 0.728 | 0.670 |
| `pre_will_correct` | 0.639 | **0.744** | 0.724 | 0.719 |
| `pre_confident_knows` | **0.666** | 0.672 | 0.625 | **0.731** |

### Round 5: Final validation of "likely correct" for both pre and post

| Prompt | on_policy | 250k | 500k | past_lens |
|--------|-----------|------|------|-----------|
| `pre_likely_correct` | 0.629 | 0.714 | 0.701 | 0.713 |
| `post_likely_correct` | 0.600 | 0.727 | 0.730 | 0.662 |

### Why "likely correct" over higher-AUC alternatives
- `post_lucky_guess` had the highest peak AUC (0.763) but was inconsistent (0.623 on on_policy).
- `pre_will_correct` was the most consistent but uses different wording than post.
- Using "likely correct" for both pre and post keeps the prompt simple and parallel —
  same concept, just tense-shifted for pre vs post. Less cherry-picking.
- Both variants are robust: AUC 0.60-0.73 across all 4 LoRAs tested.

## Notes
- Accuracy at zero threshold is ~50% for all prompts — the AO always leans "yes"
  (positive margin for both correct and incorrect answers). The signal is only in the
  *degree* of yes. This differs from sycophancy where balanced accuracy was achievable.
- Negation/negative framings ("uncertain", "guessing", "struggling", "any chance wrong")
  consistently fail — either near chance or anti-correlated. The AO can't handle negation.
- Pre-answer AUC is surprisingly close to post-answer, suggesting the AO reads the model's
  internal confidence about the question itself, not just its confidence in a specific answer.

## Letter Prediction Prompt

Separate from the binary yes/no correctness prediction, the eval also asks the AO to
predict which answer letter (A-D) the model will choose, using the argmax output token.

**Selected prompt:**
```
Predict the model's answer. Output only A, B, C, or D.
```

### Why this prompt
Most prompt styles caused the AO to output "The" (a natural sentence starter) instead of
a letter. Tested 8 prompts — only completion-style prompts ("The model's answer is",
"Predict the model's answer...") produced parseable A/B/C/D tokens.

| Prompt | Parseable (500k) | Match Model | Match True |
|--------|------------------|-------------|------------|
| `Predict the model's answer. Output only A, B, C, or D.` | 100% | **81%** | 57% |
| `The model's answer is` | 100% | 77% | 62% |
| `The answer is` | 100% | 78% | 63% |
| `The model will say` | 100% | 78% | 60% |
| `The model chose` | 100% | 77% | 59% |

"Predict..." was chosen for highest model-match rate (81%) and 100% parseable across
both newer and older LoRAs (though the older LoRA collapses to only A/C predictions).
Random chance is 25%.
