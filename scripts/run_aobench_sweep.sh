#!/usr/bin/env bash
# Run AObench on each of the 5 layer checkpoints, one B200 per checkpoint.
# Uses Sonnet 4.6 as judge via native Anthropic API (with caching + low-prio key).
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

source /workspace/.env

# AObench isn't a packaged module — we set PYTHONPATH so `python -m AObench.*` resolves.
export PYTHONPATH="$REPO_ROOT/third_party/cot-oracle:${PYTHONPATH:-}"

mkdir -p /workspace/logs/aobench

declare -A GPU_FOR=( [20]=0 [21]=1 [22]=2 [23]=3 )

SAMPLE_PROFILE="${SAMPLE_PROFILE:-full}"
N_POSITIONS="${N_POSITIONS:-5}"
# Explicit include list — full benchmark minus taboo/personaqa (those need target LoRAs we don't have).
INCLUDE_EVALS="${INCLUDE_EVALS:-number_prediction mmlu_prediction backtracking missing_info sycophancy system_prompt_qa_hidden system_prompt_qa_latentqa vagueness domain_confusion activation_sensitivity hallucination}"

for layer in 20 21 22 23; do
  gpu=${GPU_FOR[$layer]}
  ckpt="/workspace/checkpoints/ao_qwen3_8b_L${layer}/final"
  log="/workspace/logs/aobench/layer${layer}.log"
  session="aob_L${layer}"

  if [ ! -d "$ckpt" ]; then
    echo "checkpoint not found: $ckpt — skipping" >&2
    continue
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already running, skipping"
    continue
  fi

  echo "launching AObench layer=$layer on GPU=$gpu -> $log"
  tmux new-session -d -s "$session" -c "$REPO_ROOT/third_party/cot-oracle" \
    "set -a; source /workspace/.env; set +a; \
     export CUDA_VISIBLE_DEVICES=$gpu; \
     export PYTHONPATH=$REPO_ROOT/third_party/cot-oracle:\${PYTHONPATH:-}; \
     /workspace/.venv/bin/python scripts/run_paper_collection_aobench.py \
       --verbalizer-lora $ckpt \
       --include $INCLUDE_EVALS \
       --sample-profile $SAMPLE_PROFILE \
       --n-positions $N_POSITIONS \
       --output-dir AObench/eval_results/layer_sweep_L${layer} \
       2>&1 | tee $log"
done

echo "AObench launched. tail logs: tail -f /workspace/logs/aobench/layer*.log"
