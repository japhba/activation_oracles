#!/usr/bin/env bash
# AObench eval on ablation checkpoints, 1 GPU per checkpoint.
# BATCH=ABCD or EFGH selects which 4 to launch.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
source /workspace/.env

export PYTHONPATH="$REPO_ROOT/third_party/cot-oracle:${PYTHONPATH:-}"
mkdir -p /workspace/logs/ablation_aobench

if [ "${BATCH:-ABCD}" = "ABCD" ]; then
  declare -A GPU_FOR=( [A_offpolicy]=0 [B_pastonly]=1 [C_finefineweb]=2 [D_fineweb]=3 )
  TAGS="A_offpolicy B_pastonly C_finefineweb D_fineweb"
else
  declare -A GPU_FOR=( [E_past_vllm]=0 [F_future_corpus]=1 [G_future_vllm_noinject]=2 [H_future_vllm_inject]=3 )
  TAGS="E_past_vllm F_future_corpus G_future_vllm_noinject H_future_vllm_inject"
fi

SAMPLE_PROFILE="${SAMPLE_PROFILE:-full}"
N_POSITIONS="${N_POSITIONS:-5}"
INCLUDE_EVALS="${INCLUDE_EVALS:-number_prediction mmlu_prediction backtracking missing_info sycophancy system_prompt_qa_hidden system_prompt_qa_latentqa vagueness domain_confusion activation_sensitivity hallucination taboo personaqa}"

for tag in $TAGS; do
  gpu=${GPU_FOR[$tag]}
  ckpt="/workspace/checkpoints/ao_q3_8b_multi_abl_${tag}/final"
  log="/workspace/logs/ablation_aobench/${tag}.log"
  session="ablobench_${tag}"
  outdir="AObench/eval_results/abl_${tag}"

  if [ ! -d "$ckpt" ]; then echo "missing ckpt: $ckpt — skip"; continue; fi
  if tmux has-session -t "$session" 2>/dev/null; then echo "$session running, skip"; continue; fi

  echo "launching aobench $tag on GPU=$gpu -> $log"
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
echo "aobenches launched. tail logs: tail -f /workspace/logs/ablation_aobench/*.log"
