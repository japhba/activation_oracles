# Model Understanding Pipeline Guide

## What this pipeline does

We find and explain surprising behaviors in a target LLM. The end product is a dataset of verified causal explanations: "the model does X because of prompt feature Y, and here are the counterfactual experiments proving it." These can be used as training data for self-explanation (activation oracles) and as evaluations for interpretability methods.

**Models run so far:** Qwen3-8B, Qwen3-32B (FP8). The pipeline is model-agnostic — set `model` and optionally `quantization` in the run config.

Source data is WildChat-1M — real user conversations with LLMs. We use both single-turn prompts and multi-turn conversations (the model sees prior turns as context and generates a response to the final user message).

## Pipeline stages in detail

### Stage 0: Prompt preparation

**Script:** `slurm_scripts/prepare_shards.py`
**What it does:** Streams WildChat-1M, filters prompts, tokenizes, splits into shards for multi-GPU generation.

- Streams conversations from `allenai/WildChat-1M` (English, non-redacted)
- Filters by total character count across all turns (default 500-5000 chars)
- Includes both single-turn and multi-turn (up to 5 turns by default)
- Multi-turn conversations are truncated to end with a user message
- Filters out NAME_ placeholders (lmsys anonymization artifacts)
- Tokenizes using the target model's chat template with thinking disabled
- Splits into N shard files (one per GPU)

**Bottleneck:** Streaming WildChat + tokenization. ~5-10 min for 40k prompts.

**Deduplication:** Use `--exclude` to pass one or more `completions.json` files from previous runs. The script excludes prompts by both `conversation_hash` (WildChat's canonical ID) and content hash (catches WildChat duplicates with identical messages but different hashes). Example:

```bash
.venv/bin/python data_pipelines/model_understanding/slurm_scripts/prepare_shards.py \
    --model Qwen/Qwen3-32B --n-prompts 60000 --n-shards 6 \
    --min-chars 500 --max-chars 5000 --max-turns 5 --max-scan 500000 \
    --exclude data_pipelines/model_understanding/runs/qwen3_32b_run/completions.json \
    --output-dir data_pipelines/model_understanding/runs/qwen3_32b_run_2/shards
```

The pipeline config also supports `"exclude_files": [...]` for the same purpose.

**Output:** `shards/shard_0.json`, `shards/shard_1.json`, etc. Each contains pre-tokenized prompts with `{id, conversation_hash, user_message, messages, n_turns, total_chars, prompt_ids}`.

### Stage 1: Generate completions

**Script:** `generate_completions.py` (via Slurm) or inline in `run_pipeline.py` (vLLM server mode)
**What it does:** Generates 10 independent completions per prompt from Qwen3-14B at temperature 1.0.

Two modes:
- **Slurm mode** (production): Submits one Slurm GPU job per shard. Each job loads Qwen3-14B into vLLM (batch mode, not HTTP server), generates completions, saves a part file. Pipeline waits for all jobs, then combines part files via `slurm_scripts/combine_completions.py`.
- **vLLM server mode** (dev): Sends all prompts concurrently to an already-running vLLM HTTP server. Fast iteration, no model load overhead.

For multi-turn prompts, the full conversation history is sent as the prompt. The 10 completions are for the final assistant response given the conversation context.

**Bottleneck:** Model loading time (~8 min per GPU). Actual generation is fast once loaded. For <1000 prompts, `enforce_eager=True` speeds up loading.

**Output:** `completions.json` with `{metadata, prompts: [{id, user_message, messages, n_turns, total_chars, completions: [str], ...}]}`.

**Throughput:** 8 GPUs can generate 40k prompts × 10 completions in ~30-60 min (dominated by model load).

### Stage 2: Screening

**Script:** `screen_completions.py`
**What it does:** Opus with extended thinking reads each prompt + all 10 completions and judges whether the behavior is interesting enough to investigate.

For each prompt, Opus outputs:
- `interest_score` (1-5): 1=routine, 3=non-obvious with 2+ hypotheses, 5=deeply puzzling
- `behavior_summary`: what the model does across completions (e.g., "7/10 completions claim X despite evidence of Y")
- `question`: an investigable question about why ("Why does the model do X?")
- `second_person_question`: same question in second person ("Why did you do X?") for training data
- `hypotheses`: 2-4 plausible competing explanations
- `counterfactual`: a concrete experiment to distinguish hypotheses
- `justification`: why this score was given

Only prompts scoring >= 3 proceed to investigation.

Two API modes:
- **Async:** Concurrent API calls (configurable concurrency). Good for small/medium runs.
- **Batch:** Anthropic batch API. Cheaper, higher throughput for large runs. Chunks into batch_chunk_size requests.

For multi-turn prompts, the screening prompt shows the full conversation history labeled as `[USER]`/`[ASSISTANT]` turns, then the final user message, then all 10 completions.

**What gets filtered out (score 1-2):**
- Hallucinations (small model, expected)
- Uniform safety refusals (10/10 refuse) — split behavior IS interesting
- Single obvious keyword triggers
- Generic sycophancy/politeness
- Straightforward task execution with no variation

**Bottleneck:** API call latency × number of prompts. Batch API handles large runs efficiently.

**Output:** `screening.json` with sorted results + `screening.jsonl` for resume.

**Throughput:** Batch API processes 10k+ prompts in a few hours. Async at concurrency 40: ~200 prompts screened per minute.

**Pass rate:** ~55% of prompts score >= 3 (measured at 40k scale).

### Stage 3: Investigation (THE BOTTLENECK)

**Script:** `investigate.py`
**What it does:** An Opus agent runs counterfactual experiments to answer the screening question. It has two tools:

1. `run_prompt(user_message=..., messages=..., n=10, temperature=1.0)`: Sends a modified prompt to the vLLM server and gets back 10 completions. For multi-turn, accepts a full `messages` list so the agent can modify any conversation turn.

2. `submit_findings(question, baseline_behavior, answer, first_person_answer, commentary, confidence, key_evidence)`: Submits structured results at the end.

The agent loop:
1. Agent sees the original prompt + 10 completions + the screening question + hypotheses
2. Agent designs and runs counterfactual experiments (typically 8-15 per investigation)
3. Each experiment: modify one aspect of the prompt, send to vLLM, observe how behavior changes
4. Agent identifies the causal factor(s) and submits structured findings
5. Max 30 agent turns (safety limit, rarely hit)

For multi-turn prompts, the system prompt instructs the agent to modify any turn — user messages, assistant responses, or both. The agent uses the `messages` parameter (list of {role, content} objects) to construct full modified conversations.

The `answer` field contains only claims verifiable from the experiments. Speculation about training data, RLHF effects, etc. goes in the separate `commentary` field.

**Key output fields:**
- `answer`: 2-4 sentences, third person, percentages for rates. Only experiment-verified claims.
- `first_person_answer`: Same answer as if the model is explaining itself. For training data.
- `commentary`: Speculation beyond what experiments verify (training data hypotheses, etc.)
- `key_evidence`: 3-5 most decisive experiments with experiment number, what changed, baseline rate, counterfactual rate
- `experiment_log`: Every vLLM call with exact prompt/messages + all completions

**Error handling:** If an investigation fails (malformed tool call, API error), it's skipped with a count. The pipeline continues with remaining investigations. Skip count is tracked in progress output and status.json.

**Bottleneck:** This is where ~90% of the pipeline time goes. Each investigation involves multiple serial Opus API calls (each waiting for the previous response), plus vLLM calls. At concurrency N, N investigations run in parallel.

**Output:** `investigations.json` + `investigations.jsonl` (resume) + `transcripts/` (full agent message histories).

**Throughput:** Scales sublinearly with concurrency — at high concurrency, the vLLM server or API latency can become the bottleneck. With 3 GPUs (data-parallel, FP8, max-model-len 6000) and concurrency 250-500, measured ~1,200-1,300 investigations/hr. The pipeline logs `[rate]` lines every 60s showing input/output tokens per minute.

**Token usage per investigation (measured, 22k investigations):**
- Non-cached input: ~80k tokens (input + cache_create)
- Cache read: ~182k tokens (~70% cache hit rate)
- Output: ~8.5k tokens (includes thinking)
- vLLM calls: ~7.6 per investigation

### Stage 4: Verification

**Script:** `verify_investigations.py`
**What it does:** An independent Opus judge scores the investigation quality. It sees:
- The claimed question, baseline behavior, and causal answer
- The original prompt + completions
- 3-5 key experiments with the actual prompts sent and actual completions received

For each key experiment, the judge checks:
1. **Rate accuracy:** Does the claimed rate match the actual completions? (within ±1)
2. **Change description accuracy:** Does "what changed" match the actual prompt difference?
3. **Relevance to causal claim:** Does this experiment actually test the stated cause?

For multi-turn experiments, the judge sees the full conversation that was sent (all `[USER]`/`[ASSISTANT]` turns), not just the last user message.

Score guide:
- 1-2: Fundamentally unsupported — experiments irrelevant or contradict the answer
- 3-4: Major gaps — key experiments don't isolate claimed factors
- 5-6: Moderate issues — most experiments relevant but gaps remain
- 7-8: Minor issues — all experiments relevant, small rate miscounts
- 9-10: Fully verified — all rates match, experiments directly test the causal explanation

Uses extended thinking for careful rate-counting.

Two API modes: async (concurrency) and batch (for large runs).

**Bottleneck:** Not significant. Batch API handles thousands of verifications in hours.

**Output:** `verification.json` + `verification.jsonl` (resume).

**Score distribution (from qwen3_32b_run, 22,020 verifications):**
- Mean: 7.36/10
- Score >= 7: 78%
- Score >= 8: 44%

## End-to-end pipeline yield

Starting from raw WildChat prompts (measured from qwen3_32b_run, 40k prompts):

| Step | What happens | Pass rate | Cumulative |
|------|-------------|-----------|------------|
| Char filter | 500-5000 total chars, <= 5 turns | ~30-40% of WildChat | ~35% |
| Stage 2 | Opus scores behavior 1-5, need >= 3 | ~55% of screened | ~19% |
| Stage 3 | Opus investigates, produces findings | ~99.9% succeed (few skips) | ~19% |
| Stage 4 >= 8 | High quality, verified | ~44% of investigated | ~8% |
| Stage 4 >= 7 | Usable quality | ~78% of investigated | ~15% |

**From 40,000 stage 1 prompts, expect ~9,500 high-quality data points (score >= 8) or ~17,000 usable data points (score >= 7).**

## How to run

```bash
# 1. Make sure .env has the right ANTHROPIC_API_KEY
source .env

# 2. Make sure a vLLM server is running (needed for stage 3)
#    Check: squeue -u $USER
#    Start if needed: sbatch --time=72:00:00 data_pipelines/model_understanding/slurm_scripts/submit_vllm_server.sh

# 3. Run the pipeline
.venv/bin/python data_pipelines/model_understanding/run_pipeline.py \
    --config data_pipelines/model_understanding/runs/<run_name>/config.json
```

### Config format

The config JSON overrides values from the `DEFAULTS` dict at the top of `run_pipeline.py`. Example with all values explicit:

```json
{
  "run_name": "qwen3_14b_run",
  "stage1_mode": "slurm",
  "stage1_n_prompts": 40000,
  "stage1_n_gpus": 4,
  "stage1_min_chars": 500,
  "stage1_max_chars": 5000,
  "stage1_max_turns": 5,
  "stage1_max_scan": 500000,
  "vllm_max_tokens": 500,
  "vllm_n_completions": 10,
  "vllm_temperature": 1.0,
  "stage2_mode": "batch",
  "stage2_n": 40000,
  "judge_model": "claude-opus-4-6",
  "judge_max_tokens": 8192,
  "judge_thinking_budget": 4096,
  "stage3_min_score": 3,
  "stage3_model": "claude-opus-4-6",
  "stage3_max_tokens": 16384,
  "stage3_thinking_budget": 10000,
  "stage3_max_agent_turns": 30,
  "stage4_mode": "batch",
  "async_concurrency": 30,
  "batch_chunk_size": 10000
}
```

### Resuming

The pipeline is fully resumable. Each stage checks `status.json` and skips if "done". Within a stage, completed items are tracked in JSONL files. To re-run a specific stage:
1. Edit `status.json` in the run directory — set the stage's status to "pending"
2. Delete any output files for that stage if you want a clean re-run
3. Re-run the pipeline command

To change concurrency mid-stage-3: kill the pipeline, change `async_concurrency` in config.json, re-run. It resumes from the JSONL.

### Monitoring

```bash
# Check overall progress
cat data_pipelines/model_understanding/runs/<run_name>/status.json

# Watch stage 3 progress
tail -f <pipeline_output_log> | grep "Investigation progress"

# Count completed investigations
wc -l data_pipelines/model_understanding/runs/<run_name>/investigations.jsonl
```

### API key

The pipeline uses whatever `ANTHROPIC_API_KEY` is set in `.env`. To use the low-priority key, comment out the high-priority line and set ANTHROPIC_API_KEY to the low-priority value. Stage 2/4 batch API uses `ANTHROPIC_API_KEY_BATCH_API` (separate key).

## Key files

| File | Purpose |
|---|---|
| `run_pipeline.py` | Orchestrator — runs all stages, config-driven, all defaults defined here |
| `generate_completions.py` | Stage 1: vLLM batch completion generation |
| `slurm_scripts/prepare_shards.py` | Stage 0: prepare prompt shards for multi-GPU |
| `slurm_scripts/combine_completions.py` | Combine shard outputs |
| `screen_completions.py` | Stage 2: Opus screening |
| `investigate.py` | Stage 3: Opus investigation agent |
| `verify_investigations.py` | Stage 4: Opus verification judge |
| `few_shot_question_pool.json` | 185 good question examples for screening |
| `INVESTIGATION_QUALITY_GUIDE.md` | What makes investigations interesting |
| `slurm_scripts/submit_vllm_server.sh` | Slurm job for vLLM server (14B) |
| `slurm_scripts/submit_vllm_server_32b.sh` | Slurm job for vLLM server (32B, 3 GPU, FP8) |
| `runs/split_qwen3_14b_run.py` | Train/test split script (example) |
| `utility_scripts/backfill_conversation_hashes.py` | Backfill `conversation_hash` into older completions.json files that lack it |
| `utility_scripts/verify_no_overlap.py` | Verify zero prompt overlap between two runs (checks conversation_hash, content hash, user_message) |

## After the pipeline completes

### 1. Train/test split

Split the pipeline output into train and test sets. See `runs/split_qwen3_14b_run.py` for an example — it samples N examples per verification score bucket for the test set and puts the rest in train.

### 2. Training (Text SFT)

Train a LoRA on the pipeline output so the model can explain its own behavior in text.

- **Training script:** `nl_probes/text_sft/model_understanding.py`
- **Training configs:** `training_configs/model_understanding/`
- **Slurm job scripts:** `jobs/model_understanding/`
- **Full guide:** `docs/launching_training_jobs.md` § "Model Understanding Text SFT"
- **Data assembly:** read `nl_probes/text_sft/model_understanding.py` (the code is the source of truth)

### 3. Evaluation

Evaluate trained LoRAs, base model text baselines, or activation oracle verbalizers.

- **Eval script:** `nl_probes/open_ended_eval/model_understanding.py`
- **Three modes:** `ao` (activation oracle), `text_baseline` (base model), `text_sft` (trained LoRA)
- **Full guide:** `docs/running_evaluations.md`

### 4. AO Agent Eval

Multi-turn evaluation where an Opus agent interactively queries an Activation Oracle to explain model behavior. Instead of a single AO query, the agent formulates hypotheses, queries the AO about specific parts of the conversation, and uses response consistency (10 completions at temperature 1.0) to assess confidence.

**Script:** `nl_probes/open_ended_eval/model_understanding_ao_agent.py`

**How it works:**
1. Load the target model (e.g., Qwen3-8B) and AO adapter on GPU
2. For each eval entry: tokenize the conversation (user messages + assistant completion, no question appended)
3. Opus agent sees the transcript + behavioral question, has `query_ao` and `submit_findings` tools
4. Agent queries the AO ~5-9 times, selecting different token windows (named segments or text snippets)
5. Each AO query returns 10 temperature-1.0 completions -- consistency indicates confidence
6. Agent submits a final answer, Haiku judge scores against reference answer

**Running it:**

```bash
# Via Slurm (needs 1 GPU for the AO, API calls for Opus)
sbatch jobs/model_understanding/eval_ao_agent.sh

# Or directly
source .env
python -m nl_probes.open_ended_eval.model_understanding_ao_agent \
    --verbalizer-lora-path checkpoints/500k_pl_31k_spqav2_199k_sqav3_126k_cls/final \
    --run-dir data_pipelines/model_understanding/runs/qwen3_14b_run_test \
    --model-name Qwen/Qwen3-8B \
    --output-dir experiments/model_understanding_eval/qwen3_14b_run_test_v2_judge/ao_agent \
    --max-agent-turns 10 \
    --ao-completions-per-query 10 \
    --ao-max-context-tokens 1000 \
    --agent-concurrency 10 \
    --agent-model claude-opus-4-6 \
    --judge-concurrency 30 \
    --min-verification-score 6
```

**Key parameters:**
- `--verbalizer-lora-path`: AO checkpoint. See `docs/current_default_models.md` for available checkpoints.
- `--ao-max-context-tokens`: Max tokens in AO context window. The AO prompt is 3x this (one layer block per selected layer), so 1000 tokens = ~3000 AO input tokens. Going above 1000 may OOM.
- `--agent-concurrency`: Number of concurrent Opus agent loops. GPU is not the bottleneck -- Opus API latency is. 10-20 is reasonable. Watch for rate limits at higher concurrency.
- `--ao-completions-per-query`: Number of AO completions per query. 10 at temperature 1.0 gives good consistency signal.
- `--max-agent-turns`: Safety limit. Most investigations complete in 4-6 turns.

**Efficiency considerations:**
- GPU memory: AO batch size is 8 requests x 10 completions = 80 items per batch. On H200 (140GB) this uses ~82GB peak. Reduce `AO_EVAL_BATCH_SIZE` in the script if OOMing on smaller GPUs.
- API costs: Each entry uses ~5-9 Opus turns with thinking enabled. At 20 concurrency, 500 entries takes ~4-5 hours.
- The script has retry logic for rate limits (429) and overloaded errors (529) with exponential backoff.

**Resume support:** Results are saved incrementally to `agent_results.jsonl`. Resubmitting the same command skips already-completed prompt IDs.

**Output files:**
- `agent_results.jsonl`: One line per entry with prompt_id, response_text, structured_findings, turn/query counts
- `ao_agent_results.json`: Judged results with metrics (written after the judge phase at the end)
- `transcripts/{prompt_id}_transcript.json`: Full agent transcript per entry, including decoded tokenized context, segment boundaries, AO query log with decoded selected text, and the full Anthropic message history

**Judging partial results:** If the run is still going or was interrupted before the judge phase:

```bash
source .env
.venv/bin/python experiments/model_understanding_eval/judge_ao_agent_results.py \
    experiments/model_understanding_eval/qwen3_14b_run_test_v2_judge/ao_agent
```

**Comparing multiple runs:**

```bash
.venv/bin/python experiments/model_understanding_eval/judge_ao_agent_results.py \
    experiments/.../ao_agent \
    experiments/.../ao_agent_chatreg2ep \
    experiments/.../ao_agent_v2
```

**Job scripts for different AO checkpoints:** See `jobs/model_understanding/eval_ao_ag_*.sh`.

## Monitoring and operations

### Monitoring checklist

```bash
# Pipeline alive?
ps -p $(cat data_pipelines/model_understanding/runs/<run_name>/pipeline.pid)

# Stage progress
cat data_pipelines/model_understanding/runs/<run_name>/status.json

# vLLM server running? (needed for stage 3)
squeue -u $USER -o "%.10i %.12j %.2t %.10M %.20R %.10b"

# Investigation count (stage 3)
wc -l data_pipelines/model_understanding/runs/<run_name>/investigations.jsonl

# Token rates (stage 3)
grep "\[rate\]" data_pipelines/model_understanding/runs/<run_name>/pipeline.log | tail -3

# Errors
grep -i "SKIPPED\|Traceback\|rate_limit" data_pipelines/model_understanding/runs/<run_name>/pipeline.log
```

### Resuming after a crash

```bash
source .env
nohup .venv/bin/python data_pipelines/model_understanding/run_pipeline.py \
    --config data_pipelines/model_understanding/runs/<run_name>/config.json \
    > data_pipelines/model_understanding/runs/<run_name>/pipeline.log 2>&1 &
echo $! > data_pipelines/model_understanding/runs/<run_name>/pipeline.pid
```

The pipeline resumes from JSONL checkpoints automatically.

### Changing concurrency mid-run

1. `kill $(cat data_pipelines/model_understanding/runs/<run_name>/pipeline.pid)`
2. Edit `async_concurrency` in the run's `config.json`
3. Restart with the command above

### Expected timeline (40k prompts, measured from qwen3_32b_run)

- Stage 1 (4 GPUs, Slurm): ~1-2 hours
- Stage 2 (batch API, 40k prompts): ~30 min
- Stage 3 (concurrency 250-500, 3 GPUs): ~15 hours (the bottleneck)
- Stage 4 (batch API, ~22k verifications): ~30 min

### Tips

- **Check ~10 min after each stage starts** — failures surface early (bad GPU node, OOM, API errors)
- **Stage 1 shard failures**: GPU nodes can have insufficient memory. Resubmit failed shards with `--exclude=<bad_node>` — the pipeline skips completed parts on restart
- **Stage 4 batch chunk size**: use `stage4_batch_chunk_size: 500` in config (verification prompts are ~330KB each, default 10k exceeds the 256MB batch API limit)
- **Always set `max_model_len`** on vLLM servers — reduces KV cache allocation and increases throughput significantly (see AGENTS.md)
