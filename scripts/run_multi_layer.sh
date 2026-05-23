#!/usr/bin/env bash
# Multi-layer AO over L21/L22/L23 with rsLoRA + past_lens(CoT v5), lr=3e-5.
# Single B200 (gpu 0); other 3 GPUs free for parallel work.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
source /workspace/.env

mkdir -p /workspace/logs/multi_layer

cfg="$REPO_ROOT/training_configs/multi_layer_L21_22_23/multi_L21_22_23.json"
log="/workspace/logs/multi_layer/multi_L21_22_23.log"
session="multi_L21_22_23"
GPU=${MULTI_GPU:-0}
PORT=${MULTI_PORT:-29700}

if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session $session already running"; exit 0
fi
[ -f "$cfg" ] || { echo "missing $cfg" >&2; exit 1; }

echo "launching multi-layer L21/22/23 on GPU=$GPU port=$PORT -> $log"
tmux new-session -d -s "$session" -c "$REPO_ROOT" \
  "set -a; source /workspace/.env; set +a; \
   export CUDA_VISIBLE_DEVICES=$GPU; \
   export AO_USE_UNSLOTH=1 TOKENIZERS_PARALLELISM=false; \
   export WANDB_RUN_GROUP=multi_layer_L21_22_23; \
   /workspace/.venv/bin/torchrun \
     --nproc_per_node=1 \
     --rdzv_backend=c10d \
     --rdzv_endpoint=127.0.0.1:$PORT \
     nl_probes/sft.py --config $cfg \
     2>&1 | tee $log"
echo "launched. tail: tail -f $log"
