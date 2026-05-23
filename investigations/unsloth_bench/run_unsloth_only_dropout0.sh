#!/usr/bin/env bash
set -uo pipefail
cd /workspace-vast/jbauer/activation_oracles_dev
source .env
VENV=/var/tmp/jbauer/venvs/loracles-unsloth-bench
PY="$VENV/bin/python"
RESULTS=investigations/unsloth_bench/results

echo "=== node info ==="
hostname
"$PY" -c "import torch, flash_attn; print('torch', torch.__version__, 'flash_attn', flash_attn.__version__)"

export PYTHONUNBUFFERED=1
"$PY" -u investigations/unsloth_bench/bench.py \
    --mode unsloth --model Qwen/Qwen3-8B \
    --batch-size 8 --seq-len 1024 \
    --num-steering-positions 10 --hook-layer 1 \
    --warmup-steps 3 --measured-steps 15 \
    --gradient-checkpointing --lora-dropout 0 \
    --out "$RESULTS/dropout0_unsloth.json"
echo "exit code: $?"
