#!/usr/bin/env bash
# Launch 5 parallel trainings, one Qwen3-8B layer per B200 GPU.
# Designed for the RunPod 5x B200 box. Each layer gets its own tmux session,
# its own torchrun rendezvous port, and its own log file under /workspace/logs/.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

source /workspace/.env

mkdir -p /workspace/logs /workspace/checkpoints

declare -A GPU_FOR=( [20]=0 [21]=1 [22]=2 [23]=3 )
PORT_BASE=29500

for layer in 20 21 22 23; do
  gpu=${GPU_FOR[$layer]}
  port=$((PORT_BASE + layer))
  cfg="$REPO_ROOT/training_configs/layer_sweep/layer${layer}.json"
  log="/workspace/logs/layer${layer}.log"
  session="ao_L${layer}"

  if [ ! -f "$cfg" ]; then
    echo "missing config $cfg" >&2
    exit 1
  fi

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already running, skipping"
    continue
  fi

  echo "launching layer=$layer on GPU=$gpu port=$port -> $log"
  tmux new-session -d -s "$session" -c "$REPO_ROOT" \
    "set -a; source /workspace/.env; set +a; \
     export CUDA_VISIBLE_DEVICES=$gpu; \
     export AO_USE_UNSLOTH=1; \
     export TOKENIZERS_PARALLELISM=false; \
     export TORCHDYNAMO_DISABLE=0; \
     export WANDB_RUN_GROUP=layer_sweep_qwen3_8b; \
     /workspace/.venv/bin/torchrun \
       --nproc_per_node=1 \
       --rdzv_backend=c10d \
       --rdzv_endpoint=127.0.0.1:$port \
       nl_probes/sft.py --config $cfg \
       2>&1 | tee $log"
done

echo "all sessions launched. attach with: tmux a -t ao_L<layer>"
echo "tail logs with: tail -f /workspace/logs/layer*.log"
