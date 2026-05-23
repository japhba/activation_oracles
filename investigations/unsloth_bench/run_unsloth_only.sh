#!/usr/bin/env bash
# Just the unsloth realistic phase, with un-buffered output to catch any crash.

set -uo pipefail
cd /workspace-vast/jbauer/activation_oracles_dev
source .env

VENV=/var/tmp/jbauer/venvs/loracles-unsloth-bench
PY="$VENV/bin/python"
RESULTS=investigations/unsloth_bench/results

echo "=== node info ==="
hostname
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Force unbuffered stdout/stderr so we don't lose the last lines on a crash
export PYTHONUNBUFFERED=1

"$PY" -u investigations/unsloth_bench/bench.py \
    --mode unsloth \
    --model Qwen/Qwen3-8B \
    --batch-size 8 \
    --seq-len 1024 \
    --num-steering-positions 10 \
    --hook-layer 1 \
    --warmup-steps 3 \
    --measured-steps 15 \
    --gradient-checkpointing \
    --out "$RESULTS/realistic_unsloth.json"
echo "exit code: $?"
