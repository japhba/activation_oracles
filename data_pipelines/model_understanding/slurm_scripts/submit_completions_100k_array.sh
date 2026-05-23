#!/bin/bash
#SBATCH --job-name=mu_comp_100k
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --qos=high
#SBATCH --array=0-3
#SBATCH --output=data_pipelines/model_understanding/logs/slurm_%A_%a.out

set -euo pipefail

cd /workspace-vast/adamk/activation_oracles_dev
source .env

SHARD_DIR="data_pipelines/model_understanding/Qwen3-14B/shards_100k"
SHARD="${SHARD_DIR}/shard_${SLURM_ARRAY_TASK_ID}.json"
PART=$((SLURM_ARRAY_TASK_ID + 1))
OUTPUT="data_pipelines/model_understanding/Qwen3-14B/completions_100k_part${PART}.json"

echo "Array task ${SLURM_ARRAY_TASK_ID}: shard=${SHARD}, output=${OUTPUT}"

.venv/bin/python data_pipelines/model_understanding/generate_completions.py \
    --model Qwen/Qwen3-14B \
    --n-completions 10 \
    --input-shard ${SHARD} \
    --output ${OUTPUT}
