#!/usr/bin/env bash
# Verify the AO extract-then-inject training-step pattern works under both backends.
# Then compare extracted activation norms (sanity: should be in the same ballpark).

set -uo pipefail

cd /workspace-vast/jbauer/activation_oracles_dev
source .env

VENV=/var/tmp/jbauer/venvs/loracles-unsloth-bench
PY="$VENV/bin/python"
UV=/home/jbauer/.local/bin/uv
RESULTS=investigations/unsloth_bench/results
mkdir -p "$RESULTS" /var/tmp/jbauer/venvs

echo "=== node info ==="
hostname; nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo

# Build venv on this node if missing (same logic as run_bench.sh)
if [ ! -x "$PY" ]; then
    echo "=== building unsloth venv on $(hostname) ==="
    LORACLES=/var/tmp/jbauer/venvs/loracles
    if [ -d "$LORACLES" ]; then
        cp -a --reflink=auto "$LORACLES" "$VENV"
        "$UV" pip install --python "$PY" unsloth || exit 2
    else
        "$UV" venv --python 3.13 "$VENV" || exit 2
        "$UV" pip install --python "$PY" \
            --extra-index-url https://download.pytorch.org/whl/cu128 \
            torch transformers peft accelerate huggingface_hub bitsandbytes unsloth || exit 2
    fi
fi

"$PY" -c "import torch; assert torch.cuda.is_available()" || exit 3

run_one() {
    local mode="$1"
    echo "============================================================"
    echo "[$mode] extract+inject training-step test"
    echo "============================================================"
    "$PY" investigations/unsloth_bench/extract_insert_test.py \
        --mode "$mode" \
        --model Qwen/Qwen3-8B \
        --batch-size 4 \
        --ctx-len 256 \
        --train-len 256 \
        --num-steering-positions 10 \
        --extract-layers 7 14 21 \
        --hook-layer 1 \
        --num-steps 3 \
        --out "$RESULTS/extract_insert_${mode}.json" || return $?
}

run_one hf      || exit 1
run_one unsloth || exit 1

echo
echo "=== compare extracted norms (sanity: HF and Unsloth should agree on the same base weights) ==="
"$PY" -c "
import json
hf = json.load(open('$RESULTS/extract_insert_hf.json'))
us = json.load(open('$RESULTS/extract_insert_unsloth.json'))
print('per-step mean extracted norm (per-batch-elem mean of L2 norms):')
print('  step  HF                            Unsloth')
for s in range(len(hf['extracted_norms_per_step'])):
    h = hf['extracted_norms_per_step'][s]
    u = us['extracted_norms_per_step'][s]
    print(f'  {s}     {[round(x,3) for x in h]}  {[round(x,3) for x in u]}')
print()
print(f'HF    losses: first={hf[\"loss_first\"]:.4f} last={hf[\"loss_last\"]:.4f}')
print(f'US    losses: first={us[\"loss_first\"]:.4f} last={us[\"loss_last\"]:.4f}')
print(f'HF    grads: lora={hf[\"lora_grad_params\"]} base={hf[\"base_grad_params\"]}')
print(f'US    grads: lora={us[\"lora_grad_params\"]} base={us[\"base_grad_params\"]}')
"
