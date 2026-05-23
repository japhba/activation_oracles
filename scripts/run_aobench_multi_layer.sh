#!/usr/bin/env bash
# AObench eval on the multi-layer L21/22/23 checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
source /workspace/.env

export PYTHONPATH="$REPO_ROOT/third_party/cot-oracle:${PYTHONPATH:-}"
mkdir -p /workspace/logs/aobench_multi

ckpt="/workspace/checkpoints/ao_qwen3_8b_multi_L21_22_23/final"
log="/workspace/logs/aobench_multi/multi_L21_22_23.log"
session="aob_multi_L21_22_23"
outdir="AObench/eval_results/multi_L21_22_23"
GPU=${MULTI_GPU:-0}
SAMPLE_PROFILE="${SAMPLE_PROFILE:-full}"
N_POSITIONS="${N_POSITIONS:-5}"
INCLUDE_EVALS="${INCLUDE_EVALS:-number_prediction mmlu_prediction backtracking missing_info sycophancy system_prompt_qa_hidden system_prompt_qa_latentqa vagueness domain_confusion activation_sensitivity hallucination taboo personaqa}"

[ -d "$ckpt" ] || { echo "missing $ckpt"; exit 1; }
if tmux has-session -t "$session" 2>/dev/null; then echo "$session running"; exit 0; fi

tmux new-session -d -s "$session" -c "$REPO_ROOT/third_party/cot-oracle" \
  "set -a; source /workspace/.env; set +a; \
   export CUDA_VISIBLE_DEVICES=$GPU; \
   export PYTHONPATH=$REPO_ROOT/third_party/cot-oracle:\${PYTHONPATH:-}; \
   /workspace/.venv/bin/python scripts/run_paper_collection_aobench.py \
     --verbalizer-lora $ckpt \
     --include $INCLUDE_EVALS \
     --sample-profile $SAMPLE_PROFILE \
     --n-positions $N_POSITIONS \
     --output-dir $outdir \
     2>&1 | tee $log"
echo "launched. tail: tail -f $log"
