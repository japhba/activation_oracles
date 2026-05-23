# Backtracking Eval Dataset — Creation Process

## Goal
Build an eval dataset of points in Qwen3-8B's chain-of-thought where the model expresses uncertainty, doubt, or desire to backtrack. Used to test whether Activation Oracles can identify the *specific reason* for uncertainty (vs. giving vague/unfalsifiable answers).

## Pipeline

### 1. Problem Generation
- 30 diverse, challenging reasoning problems across 12+ domains (number theory, combinatorics, logic, algebra, geometry, probability, etc.)
- Designed to elicit extended thinking with natural uncertainty points
- Saved to `problems.json`

### 2. Rollout Generation (`experiments/generate_backtracking_rollouts.py`)
- Model: **Qwen/Qwen3-8B** via vLLM
- 10 rollouts per problem (300 total), temperature=1.0, top_p=0.95, max_tokens=3000
- Thinking mode enabled (`enable_thinking=True` in chat template)
- ~892K tokens generated; avg thinking length ~9,700 chars
- Most rollouts truncated at 3000 tokens (model still thinking)
- Saved to `rollouts.json`

### 3. Backtracking Point Identification
- Identified points in rollouts where the model expresses uncertainty via keyword patterns ("Wait", "Actually", "No,", "Hold on", "Hmm", "let me re", "mistake", etc.)
- Extracted prefix (text up to the uncertainty point) and suffix (text after)
- Saved to `verification_points_large.json` (210 points)

### 4. Consistency Verification (`experiments/verify_backtracking_consistency.py`)
- For each identified point: feed the prefix back to Qwen3-8B via vLLM, generate 10 continuations of ~100 tokens each
- Check whether the model consistently backtracks at that point across re-generations
- Computed `backtrack_rate` (0.0–1.0) per point
- Results: mean rate 0.58; 23% always backtrack, 17% usually, 28% sometimes, 32% rarely/never
- Saved to `verification_results_large.json`

### 5. Uncertainty Description Generation
- Split 210 points into high consistency (rate >= 0.7, N=110) and low consistency (0 < rate < 0.7, N=92)
- Batched into 10 high batches + 9 low batches of ~11 items each (`batches/` folder)
- **19 parallel Claude Sonnet sub-agents** each reviewed one batch
- Each agent read the prefix, original continuation, and 10 sample continuations
- Produced a specific 1-2 sentence description of what the uncertainty is about
- For low-consistency entries, agents also judged `is_genuine_uncertainty` (true/false)
- Results: 54/92 low entries flagged as genuine uncertainty
- Saved to `agent_descriptions_high.json` and `agent_descriptions_low.json`

### 6. Dataset Assembly (`experiments/assemble_backtracking_dataset.py`)
- Combined verification results with agent-generated descriptions
- Matched by batch ordering (agent outputs mapped to batch files by prompt content)
- Final dataset: 110 high-consistency + 92 low-consistency entries
- Each entry contains: problem, prefix, all 10 continuations (full text), backtrack_rate, uncertainty_description
- Saved to `backtracking_eval_dataset.json` (2.0 MB)

### 7. Eval Implementation (`nl_probes/open_ended_eval/backtracking.py`)
- Standalone eval module following the taboo/personaqa pattern
- For each dataset entry:
  1. Formats the problem + prefix (truncated to last 2000 chars) as a user message
  2. Runs through `base_experiment.run_verbalizer()` to collect activations from Qwen3-8B and query the AO
  3. AO is asked: "What is the model uncertain or confused about at this point in its reasoning?"
  4. Uses segment activations (last 20 tokens) at layers [25, 50, 75]%, greedy decoding, max 150 new tokens
- No target LoRA — uses base model activations (the uncertainty is intrinsic, not LoRA-induced)
- **LLM judge scoring**: Claude Haiku (`claude-haiku-4-5-20251001`) rates each AO response on:
  - **Specificity** (1-5): How specific vs. vague/generic is the AO's answer?
  - **Correctness** (1-5): How well does it match the ground-truth `uncertainty_description`?
- Async judging with configurable concurrency (default 10 concurrent requests)
- Aggregate metrics: mean scores, rate of scores >=3 and >=4
- Initial results on 20 high-consistency entries: mean specificity 2.1, mean correctness 2.15 — confirms AO vagueness problem
- Run: `source .env && .venv/bin/python -m nl_probes.open_ended_eval.backtracking`

## Models Used
- **Target model**: Qwen/Qwen3-8B (rollout generation, consistency verification)
- **Uncertainty description agents**: Claude Sonnet (19 parallel sub-agents via Claude Code)
- **Eval judge**: Claude Haiku (`claude-haiku-4-5-20251001`) for scoring AO responses

## Key Files
| File | Description |
|------|-------------|
| `problems.json` | 30 input problems |
| `rollouts.json` | 300 CoT rollouts (30 x 10) |
| `verification_points_large.json` | 210 identified backtracking points |
| `verification_results_large.json` | Consistency verification (10 continuations each) |
| `batches/` | Slimmed data sent to sub-agents |
| `agent_descriptions_{high,low}.json` | Sub-agent outputs |
| `backtracking_eval_dataset.json` | **Final assembled dataset** |
