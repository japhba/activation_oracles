#!/usr/bin/env bash
# Generic launcher for Phase 3 ablations.
#   STAGE=gen|train|aobench
#   TAGS="I_5layer_21_25 J_4layer_19_21_23_25 K_single_L23"
# Pins one B200 per run round-robin starting at GPU 0.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
source /workspace/.env

STAGE="${STAGE:-gen}"
: "${TAGS:?must set TAGS=\"tag1 tag2 ...\"}"

mkdir -p /workspace/logs/phase3

i=0
for tag in $TAGS; do
  gpu=$((i % 4))
  port=$((30000 + i))
  cfg="$REPO_ROOT/training_configs/phase3_ablations/phase3_${tag}.json"
  log="/workspace/logs/phase3/${STAGE}_${tag}.log"
  ckpt="/workspace/checkpoints/ao_q3_8b_phase3_${tag}/final"

  if [ ! -f "$cfg" ]; then echo "missing $cfg"; exit 1; fi

  if [ "$STAGE" = "gen" ]; then
    session="p3gen_${tag}"
    if tmux has-session -t "$session" 2>/dev/null; then echo "$session running, skip"; i=$((i+1)); continue; fi
    echo "[gen]   $tag GPU=$gpu -> $log"
    tmux new-session -d -s "$session" -c "$REPO_ROOT" \
      "set -a; source /workspace/.env; set +a; \
       export CUDA_VISIBLE_DEVICES=$gpu AO_USE_UNSLOTH=1 TOKENIZERS_PARALLELISM=false; \
       /workspace/.venv/bin/python nl_probes/sft.py --config $cfg --gen-only \
       2>&1 | tee $log"
  elif [ "$STAGE" = "train" ]; then
    session="p3tr_${tag}"
    if tmux has-session -t "$session" 2>/dev/null; then echo "$session running, skip"; i=$((i+1)); continue; fi
    echo "[train] $tag GPU=$gpu port=$port -> $log"
    tmux new-session -d -s "$session" -c "$REPO_ROOT" \
      "set -a; source /workspace/.env; set +a; \
       export CUDA_VISIBLE_DEVICES=$gpu AO_USE_UNSLOTH=1 TOKENIZERS_PARALLELISM=false; \
       /workspace/.venv/bin/torchrun --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:$port \
         nl_probes/sft.py --config $cfg \
       2>&1 | tee $log"
  elif [ "$STAGE" = "aobench" ]; then
    session="p3aob_${tag}"
    [ -d "$ckpt" ] || { echo "missing ckpt $ckpt for $tag — skip"; i=$((i+1)); continue; }
    if tmux has-session -t "$session" 2>/dev/null; then echo "$session running, skip"; i=$((i+1)); continue; fi
    INCLUDE_EVALS="${INCLUDE_EVALS:-number_prediction mmlu_prediction backtracking missing_info sycophancy system_prompt_qa_hidden system_prompt_qa_latentqa vagueness domain_confusion activation_sensitivity hallucination taboo personaqa}"
    echo "[aobench] $tag GPU=$gpu -> $log"
    tmux new-session -d -s "$session" -c "$REPO_ROOT/third_party/cot-oracle" \
      "set -a; source /workspace/.env; set +a; \
       export CUDA_VISIBLE_DEVICES=$gpu; \
       export PYTHONPATH=$REPO_ROOT/third_party/cot-oracle:\${PYTHONPATH:-}; \
       /workspace/.venv/bin/python scripts/run_paper_collection_aobench.py \
         --verbalizer-lora $ckpt \
         --include $INCLUDE_EVALS \
         --sample-profile full \
         --n-positions 5 \
         --output-dir AObench/eval_results/phase3_${tag} \
       2>&1 | tee $log"
  else
    echo "unknown STAGE=$STAGE"; exit 1
  fi
  i=$((i+1))
done
echo "done dispatching STAGE=$STAGE for [$TAGS]"
