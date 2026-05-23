#!/bin/bash
#SBATCH --job-name=vllm_server
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --qos=high
#SBATCH --output=data_pipelines/model_understanding/logs/slurm_vllm_%j.out

set -euo pipefail

cd /workspace-vast/adamk/activation_oracles_dev

MODEL="${1:-Qwen/Qwen3-14B}"
PORT="${2:-8000}"

echo "Starting vLLM server: model=${MODEL} port=${PORT} node=$(hostname)"
echo "Connect with: --vllm-url http://$(hostname):${PORT}"

.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization 0.9 \
    --max-model-len 6000
