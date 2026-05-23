#!/usr/bin/env bash
# Pre-generate past_lens datasets for all 4 ablations, one B200 each.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
source /workspace/.env

mkdir -p /workspace/logs/ablation_gen

if [ "${BATCH:-ABCD}" = "ABCD" ]; then
  declare -A GPU_FOR=( [A_offpolicy]=0 [B_pastonly]=1 [C_finefineweb]=2 [D_fineweb]=3 )
  TAGS="A_offpolicy B_pastonly C_finefineweb D_fineweb"
else
  declare -A GPU_FOR=( [E_past_vllm]=0 [F_future_corpus]=1 [G_future_vllm_noinject]=2 [H_future_vllm_inject]=3 )
  TAGS="E_past_vllm F_future_corpus G_future_vllm_noinject H_future_vllm_inject"
fi

for tag in $TAGS; do
  gpu=${GPU_FOR[$tag]}
  cfg="$REPO_ROOT/training_configs/ablations_multi_L21_22_23/abl_${tag}.json"
  log="/workspace/logs/ablation_gen/${tag}.log"
  session="genabl_${tag}"

  if tmux has-session -t "$session" 2>/dev/null; then echo "$session running, skip"; continue; fi
  [ -f "$cfg" ] || { echo "missing $cfg"; exit 1; }

  echo "launching gen $tag on GPU=$gpu -> $log"
  tmux new-session -d -s "$session" -c "$REPO_ROOT" \
    "set -a; source /workspace/.env; set +a; \
     export CUDA_VISIBLE_DEVICES=$gpu AO_USE_UNSLOTH=1 TOKENIZERS_PARALLELISM=false; \
     /workspace/.venv/bin/python nl_probes/sft.py --config $cfg --gen-only \
     2>&1 | tee $log"
done
echo "all gens launched. tail logs: tail -f /workspace/logs/ablation_gen/*.log"
