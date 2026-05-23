#!/usr/bin/env bash
# LR sweep at L22 with rsLoRA. 4 configs, one B200 per run.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

source /workspace/.env

mkdir -p /workspace/logs/lr_sweep /workspace/checkpoints

# tag -> GPU index
declare -A GPU_FOR=( [1em5]=0 [3em5]=1 [1em4]=2 [3em4]=3 )
PORT_BASE=29600

for tag in 1em5 3em5 1em4 3em4; do
  gpu=${GPU_FOR[$tag]}
  port=$((PORT_BASE + gpu))
  cfg="$REPO_ROOT/training_configs/lr_sweep_L22_rslora/L22_rslora_lr${tag}.json"
  log="/workspace/logs/lr_sweep/lr${tag}.log"
  session="lr_${tag}"

  if [ ! -f "$cfg" ]; then
    echo "missing config $cfg" >&2; exit 1
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already running, skipping"; continue
  fi

  echo "launching lr=$tag on GPU=$gpu port=$port -> $log"
  tmux new-session -d -s "$session" -c "$REPO_ROOT" \
    "set -a; source /workspace/.env; set +a; \
     export CUDA_VISIBLE_DEVICES=$gpu; \
     export AO_USE_UNSLOTH=1; \
     export TOKENIZERS_PARALLELISM=false; \
     export WANDB_RUN_GROUP=lr_sweep_L22_rslora; \
     /workspace/.venv/bin/torchrun \
       --nproc_per_node=1 \
       --rdzv_backend=c10d \
       --rdzv_endpoint=127.0.0.1:$port \
       nl_probes/sft.py --config $cfg \
       2>&1 | tee $log"
done

echo "all sessions launched. tail logs: tail -f /workspace/logs/lr_sweep/lr*.log"
