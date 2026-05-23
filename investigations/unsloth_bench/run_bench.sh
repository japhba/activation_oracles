#!/usr/bin/env bash
# Run the unsloth-vs-HF AO training-step benchmark on one Slurm-allocated GPU.
# Builds the unsloth venv on the allocated node if it doesn't already exist.
# Smoke first on Qwen3-0.6B (both modes, 3 measured steps), then full bench on
# Qwen3-8B (both modes, 20 measured steps).

set -uo pipefail

cd /workspace-vast/jbauer/activation_oracles_dev
source .env

VENV=/var/tmp/jbauer/venvs/loracles-unsloth-bench
PY="$VENV/bin/python"
UV=/home/jbauer/.local/bin/uv
RESULTS=investigations/unsloth_bench/results
mkdir -p "$RESULTS" /var/tmp/jbauer/venvs

echo "=== node info ==="
hostname
echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv | head -5
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo

# ---- Build venv on this node if missing ----
if [ ! -x "$PY" ]; then
    echo "=== building unsloth venv on $(hostname) ==="
    LORACLES=/var/tmp/jbauer/venvs/loracles
    if [ -d "$LORACLES" ]; then
        echo "cloning from $LORACLES (reflink/copy)"
        cp -a --reflink=auto "$LORACLES" "$VENV"
        echo "installing unsloth on top of clone..."
        "$UV" pip install --python "$PY" unsloth || { echo "uv install failed"; exit 2; }
    else
        echo "loracles venv not on this node; installing fresh"
        "$UV" venv --python 3.13 "$VENV" || { echo "uv venv failed"; exit 2; }
        "$UV" pip install --python "$PY" \
            --extra-index-url https://download.pytorch.org/whl/cu128 \
            torch transformers peft accelerate huggingface_hub bitsandbytes unsloth \
            || { echo "fresh uv install failed"; exit 2; }
    fi
    echo "venv ready at $VENV"
fi

# Verify torch + driver are compatible BEFORE running anything heavy
"$PY" -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
x = torch.randn(4, 4, device='cuda')
print('cuda alloc ok', x.shape)
" || { echo "torch/cuda check failed on this node"; exit 3; }
echo

run_one() {
    local mode="$1"; local model="$2"; local steps="$3"; local tag="$4"
    local out="$RESULTS/${tag}_${mode}.json"
    echo "============================================================"
    echo "[$tag/$mode] model=$model measured-steps=$steps"
    echo "============================================================"
    "$PY" investigations/unsloth_bench/bench.py \
        --mode "$mode" \
        --model "$model" \
        --measured-steps "$steps" \
        --warmup-steps 3 \
        --batch-size 4 \
        --seq-len 512 \
        --num-steering-positions 10 \
        --hook-layer 1 \
        --out "$out"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$tag/$mode] FAILED (exit $rc)"
        return $rc
    fi
}

# Smoke first
run_one hf      Qwen/Qwen3-0.6B 3 smoke || exit 1
run_one unsloth Qwen/Qwen3-0.6B 3 smoke || exit 1

# Full bench
run_one hf      Qwen/Qwen3-8B 20 full || exit 1
run_one unsloth Qwen/Qwen3-8B 20 full || exit 1

echo
echo "=== summary ==="
for f in "$RESULTS"/*.json; do
    echo
    echo "--- $f ---"
    "$PY" -c "
import json, sys
d = json.load(open('$f'))
keys = ['mode','model','batch_size','seq_len','seconds_per_step','tokens_per_second','peak_memory_gb','loss_first','loss_last']
for k in keys: print(f'  {k}: {d[k]}')
"
done
