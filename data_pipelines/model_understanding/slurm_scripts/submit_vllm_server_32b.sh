#!/bin/bash
#SBATCH --job-name=vllm_server_32b
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=120:00:00
#SBATCH --qos=high
#SBATCH --output=data_pipelines/model_understanding/logs/slurm_vllm_32b_%j.out

set -euo pipefail

cd /workspace-vast/adamk/activation_oracles_dev

MODEL="${1:-Qwen/Qwen3-32B}"
PORT="${2:-8000}"

echo "Starting vLLM server: model=${MODEL} port=${PORT} node=$(hostname)"
echo "  FP8 quantization, data-parallel-size=4, max-model-len=6000"
echo "Connect with: --vllm-url http://$(hostname):${PORT}"

.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --quantization fp8 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 6000
