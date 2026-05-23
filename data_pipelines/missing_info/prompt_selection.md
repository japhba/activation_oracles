# Missing Info Prompt Selection

## Task
Detect whether the model is missing information needed to solve a problem.
Three conditions per problem:
- A (complete): model has all info, reasons correctly
- B (incomplete): problem is missing info, model reasons confusedly
- C (forced): model has all info but is teacher-forced down wrong reasoning path

Two separate classification problems:
- **A-vs-B**: Can the AO detect the model is missing info? (different text, different activations)
- **A-vs-C**: Can the AO read activations vs surface tokens? (same text, different activations)

## Selected Prompt
```
Answer with 'Yes' or 'No' only. Does the model have incomplete information about this problem?
```

## Experimental Design
- **A-vs-B**: A has NO teacher-forced text appended, fixed 20-token window for both.
  Rationale: teacher-forced text is an artifact of the A-vs-C design, not relevant to
  A-vs-B. Fixed window avoids accidentally including explicit uncertainty language.
- **A-vs-C**: A and C both have the teacher-forced segment, matched segment size per
  problem (4-20 tokens). This is the controlled experiment — same tokens, different
  activations.

## Why This Prompt

### Prompt sweep (experiments/three_lora_comparison/missing_info_prompt_sweep/)
Tested 8 prompts across 6 LoRAs:
- `missing_info` and `missing_critical_info` had the best AUC (0.76-0.79) but poor
  balanced accuracy (~58-62%) due to strong "no" bias (A acc ~80%, B/C acc ~30-40%).
- `confused` and `guessing` were weak across the board.
- `enough_context` was strong on A-vs-C (0.827) but weak on A-vs-B (0.518-0.720).

### Sensitivity sweep (experiments/three_lora_comparison/missing_info_sensitivity_sweep/)
Tested 8 variants of the missing_info prompt to improve balanced accuracy:
- `incomplete_info` ("Does the model have incomplete information about this problem?")
  was the clear winner: maintained comparable AUC while dramatically improving balanced
  accuracy (73-80% vs 57-60% for `missing_info`).
- The key difference: `incomplete_info` shifts the threshold so B/C accuracy goes from
  30-40% to 80-100%, while A accuracy stays at 60-67%.
- `info_gap` was too aggressive — says "yes" to everything (A acc 13-47%).

### Results with selected prompt (experiments/missing_info_clean_eval/)

| LoRA | A-vs-B AUC | A-vs-B Acc | A-vs-C AUC | A-vs-C Acc |
|------|-----------|-----------|-----------|-----------|
| on_policy | 0.760 | 67% | 0.718 | 63% |
| past_lens_addition | 0.756 | 67% | 0.609 | 60% |
| 250k_plus | 0.800 | 70% | 0.787 | 80% |
| 250k_replace_lqa | 0.809 | 77% | 0.796 | 80% |
| 500k_plus | 0.807 | 70% | 0.778 | 73% |
| 500k_replace_lqa | 0.804 | 83% | 0.782 | 73% |

### Segment size experiment (experiments/three_lora_comparison/missing_info_segment_size/)
Tested segment sizes 10-200 on 500k_plus (A-vs-B with teacher-forced, old design):
- Larger segments help: AUC goes from 0.751 (10 tokens) to 0.880 (200 tokens).
- The default variable-length (4-20 tokens) gets 0.764.
- For the clean A-vs-B design we settled on 20 tokens as a balance between signal
  and avoiding explicit uncertainty language.
