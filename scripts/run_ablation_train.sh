#!/usr/bin/env bash
# Train 4 ablations in parallel, one B200 each.
# Usage: BATCH=ABCD bash run_ablation_train.sh   (or BATCH=EFGH)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
source /workspace/.env

mkdir -p /workspace/logs/ablation_train

if [ "${BATCH:-ABCD}" = "ABCD" ]; then
  declare -A GPU_FOR=( [A_offpolicy]=0 [B_pastonly]=1 [C_finefineweb]=2 [D_fineweb]=3 )
  TAGS="A_offpolicy B_pastonly C_finefineweb D_fineweb"
else
  declare -A GPU_FOR=( [E_past_vllm]=0 [F_future_corpus]=1 [G_future_vllm_noinject]=2 [H_future_vllm_inject]=3 )
  TAGS="E_past_vllm F_future_corpus G_future_vllm_noinject H_future_vllm_inject"
fi
PORT_BASE=29800

for tag in $TAGS; do
  gpu=${GPU_FOR[$tag]}
  port=$((PORT_BASE + gpu))
  cfg="$REPO_ROOT/training_configs/ablations_multi_L21_22_23/abl_${tag}.json"
  log="/workspace/logs/ablation_train/${tag}.log"
  session="ablt_${tag}"

  if tmux has-session -t "$session" 2>/dev/null; then echo "$session running, skip"; continue; fi
  [ -f "$cfg" ] || { echo "missing $cfg"; exit 1; }

  echo "launching train $tag on GPU=$gpu port=$port -> $log"
  tmux new-session -d -s "$session" -c "$REPO_ROOT" \
    "set -a; source /workspace/.env; set +a; \
     export CUDA_VISIBLE_DEVICES=$gpu AO_USE_UNSLOTH=1 TOKENIZERS_PARALLELISM=false; \
     export WANDB_RUN_GROUP=ao_ablations; \
     /workspace/.venv/bin/torchrun \
       --nproc_per_node=1 \
       --rdzv_backend=c10d \
       --rdzv_endpoint=127.0.0.1:$port \
       nl_probes/sft.py --config $cfg \
       2>&1 | tee $log"
done
echo "trains launched. tail logs: tail -f /workspace/logs/ablation_train/*.log"
