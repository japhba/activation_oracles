#!/usr/bin/env bash
# AObench eval on each L22+rsLoRA LR-sweep checkpoint.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
source /workspace/.env

export PYTHONPATH="$REPO_ROOT/third_party/cot-oracle:${PYTHONPATH:-}"
mkdir -p /workspace/logs/aobench_lr_sweep

declare -A GPU_FOR=( [1em5]=0 [3em5]=1 [1em4]=2 [3em4]=3 )
SAMPLE_PROFILE="${SAMPLE_PROFILE:-full}"
N_POSITIONS="${N_POSITIONS:-5}"
INCLUDE_EVALS="${INCLUDE_EVALS:-number_prediction mmlu_prediction backtracking missing_info sycophancy system_prompt_qa_hidden system_prompt_qa_latentqa vagueness domain_confusion activation_sensitivity hallucination taboo personaqa}"

for tag in 1em5 3em5 1em4 3em4; do
  gpu=${GPU_FOR[$tag]}
  ckpt="/workspace/checkpoints/ao_qwen3_8b_L22_rslora_lr${tag}/final"
  log="/workspace/logs/aobench_lr_sweep/lr${tag}.log"
  session="aob_lr_${tag}"
  outdir="AObench/eval_results/lr_sweep_L22_rslora_lr${tag}"

  if [ ! -d "$ckpt" ]; then echo "missing $ckpt"; continue; fi
  if tmux has-session -t "$session" 2>/dev/null; then echo "$session running, skip"; continue; fi

  echo "launching aobench lr=$tag on GPU=$gpu -> $log"
  tmux new-session -d -s "$session" -c "$REPO_ROOT/third_party/cot-oracle" \
    "set -a; source /workspace/.env; set +a; \
     export CUDA_VISIBLE_DEVICES=$gpu; \
     export PYTHONPATH=$REPO_ROOT/third_party/cot-oracle:\${PYTHONPATH:-}; \
     /workspace/.venv/bin/python scripts/run_paper_collection_aobench.py \
       --verbalizer-lora $ckpt \
       --include $INCLUDE_EVALS \
       --sample-profile $SAMPLE_PROFILE \
       --n-positions $N_POSITIONS \
       --output-dir $outdir \
       2>&1 | tee $log"
done

echo "AObench LR sweep launched. tail logs: tail -f /workspace/logs/aobench_lr_sweep/lr*.log"
