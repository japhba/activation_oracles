#!/bin/bash
#SBATCH --job-name=vllm_server_llama8b
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --qos=high
#SBATCH --output=data_pipelines/model_understanding/logs/slurm_vllm_llama8b_%j.out

set -euo pipefail

cd /workspace-vast/adamk/activation_oracles_dev
source .env

MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${2:-8000}"

echo "Starting vLLM server: model=${MODEL} port=${PORT} node=$(hostname)"
echo "  BF16, data-parallel-size=2, max-model-len=6000"
echo "Connect with: --vllm-url http://$(hostname):${PORT}"

.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --data-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 6000
