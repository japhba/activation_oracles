#!/usr/bin/env bash
# Realistic AO training-step bench: FA2 + grad checkpointing + larger batch/seq.
# Builds the unsloth venv on the allocated node if missing, including flash-attn.
# Note: flash-attn build is ~20 min on a fresh node.

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
nvidia-smi --query-gpu=index,memory.free,memory.total,driver_version --format=csv | head -3
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo

# ---- Build venv on this node if missing ----
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

# ---- Ensure flash-attn ----
if ! "$PY" -c "import flash_attn" >/dev/null 2>&1; then
    echo "=== building flash-attn on $(hostname) (slow, ~15-25 min) ==="
    "$UV" pip install --python "$PY" --no-build-isolation flash-attn || exit 4
fi
"$PY" -c "import flash_attn; print('flash_attn', flash_attn.__version__)"

# ---- Sanity ----
"$PY" -c "
import torch; assert torch.cuda.is_available()
print('torch', torch.__version__)
" || exit 3

run_one() {
    local mode="$1"; local tag="$2"
    local out="$RESULTS/${tag}_${mode}.json"
    local extra=()
    if [ "$mode" = "hf" ]; then
        extra+=(--attn-impl flash_attention_2)
    fi
    echo "============================================================"
    echo "[$tag/$mode] realistic bench: b=8 seq=1024 grad-ckpt=on FA2=on"
    echo "============================================================"
    "$PY" investigations/unsloth_bench/bench.py \
        --mode "$mode" \
        --model Qwen/Qwen3-8B \
        --batch-size 8 \
        --seq-len 1024 \
        --num-steering-positions 10 \
        --hook-layer 1 \
        --warmup-steps 3 \
        --measured-steps 15 \
        --gradient-checkpointing \
        "${extra[@]}" \
        --out "$out"
    local rc=$?
    [ $rc -ne 0 ] && { echo "[$tag/$mode] FAILED (exit $rc)"; return $rc; }
}

run_one hf      realistic || exit 1
run_one unsloth realistic || exit 1

echo
echo "=== realistic-bench summary ==="
"$PY" -c "
import json
hf = json.load(open('$RESULTS/realistic_hf.json'))
us = json.load(open('$RESULTS/realistic_unsloth.json'))
print(f'config: model=Qwen3-8B b={hf[\"batch_size\"]} seq={hf[\"seq_len\"]} '
      f'grad_ckpt={hf[\"gradient_checkpointing\"]} attn=HF:{hf[\"attn_impl\"]} US:builtin')
print()
print(f'                       HF+PEFT      Unsloth      Δ')
def row(k, fmt, better='lower'):
    a, b = hf[k], us[k]
    if better == 'higher':
        d = f'{b/a:.2f}x'
    else:
        d = f'{a/b:.2f}x faster' if 'second' in k else f'{(1-b/a)*100:+.1f}%'
    print(f'  {k:22s} {fmt.format(a):>10s}  {fmt.format(b):>10s}  {d}')
row('load_seconds', '{:.1f}s', 'lower')
row('seconds_per_step', '{:.4f}', 'lower')
row('tokens_per_second', '{:.0f}', 'higher')
row('peak_memory_gb', '{:.2f}', 'lower')
print(f'  loss_no_hook_eval       {hf[\"loss_no_hook_eval\"]:.4f}      {us[\"loss_no_hook_eval\"]:.4f}')
print(f'  loss_with_hook_eval     {hf[\"loss_with_hook_eval\"]:.4f}      {us[\"loss_with_hook_eval\"]:.4f}')
"
