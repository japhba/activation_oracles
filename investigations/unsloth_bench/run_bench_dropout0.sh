#!/usr/bin/env bash
# Same realistic bench, but with lora_dropout=0 — Unsloth's fused QKV/O/MLP
# kernels are only active in this case. Compares head-to-head against HF+PEFT
# at the same dropout to isolate Unsloth's gain.

set -uo pipefail

cd /workspace-vast/jbauer/activation_oracles_dev
source .env

VENV=/var/tmp/jbauer/venvs/loracles-unsloth-bench
PY="$VENV/bin/python"
RESULTS=investigations/unsloth_bench/results

echo "=== node info ==="
hostname
"$PY" -c "import torch, flash_attn; print('torch', torch.__version__, 'flash_attn', flash_attn.__version__)"
echo

export PYTHONUNBUFFERED=1

run_one() {
    local mode="$1"
    local extra=()
    [ "$mode" = "hf" ] && extra+=(--attn-impl flash_attention_2)
    echo "============================================================"
    echo "[dropout0/$mode] b=8 seq=1024 grad-ckpt=on FA2=on dropout=0"
    echo "============================================================"
    "$PY" -u investigations/unsloth_bench/bench.py \
        --mode "$mode" \
        --model Qwen/Qwen3-8B \
        --batch-size 8 \
        --seq-len 1024 \
        --num-steering-positions 10 \
        --hook-layer 1 \
        --warmup-steps 3 \
        --measured-steps 15 \
        --gradient-checkpointing \
        --lora-dropout 0 \
        "${extra[@]}" \
        --out "$RESULTS/dropout0_${mode}.json"
    local rc=$?
    [ $rc -ne 0 ] && { echo "[dropout0/$mode] FAILED (exit $rc)"; return $rc; }
}

run_one hf      || exit 1
run_one unsloth || exit 1

echo
echo "=== dropout=0 comparison ==="
"$PY" -c "
import json
hf = json.load(open('$RESULTS/dropout0_hf.json'))
us = json.load(open('$RESULTS/dropout0_unsloth.json'))
print(f'config: model=Qwen3-8B b={hf[\"batch_size\"]} seq={hf[\"seq_len\"]} '
      f'grad_ckpt={hf[\"gradient_checkpointing\"]} dropout=0 attn=HF:{hf[\"attn_impl\"]} US:builtin')
print()
print(f'                       HF+PEFT      Unsloth      Δ')
def row(k, fmt, kind='lower'):
    a, b = hf[k], us[k]
    if kind=='higher':
        d=f'{b/a:.2f}x'
    elif 'second' in k:
        d=f'{a/b:.2f}x faster'
    else:
        d=f'{(1-b/a)*100:+.1f}%'
    print(f'  {k:22s} {fmt.format(a):>10s}  {fmt.format(b):>10s}  {d}')
row('seconds_per_step','{:.4f}','lower')
row('tokens_per_second','{:.0f}','higher')
row('peak_memory_gb','{:.2f}','lower')
print(f'  loss_no_hook_eval       {hf[\"loss_no_hook_eval\"]:.4f}      {us[\"loss_no_hook_eval\"]:.4f}')
print(f'  loss_with_hook_eval     {hf[\"loss_with_hook_eval\"]:.4f}      {us[\"loss_with_hook_eval\"]:.4f}')
"
