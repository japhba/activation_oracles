#!/bin/bash
#SBATCH --job-name=gen_train_data
#SBATCH --partition=general
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --qos=high
#SBATCH --output=data_pipelines/training_data/artifacts/slurm_%A_%a.out

# Generate training data for activation oracles (stages 1-2 only).
# Run `prep` first, then submit this script.
#
# Usage:
#   source .env
#   RUN=synthetic_qa_400k
#   .venv/bin/python data_pipelines/training_data/generate_training_data.py prep --run "$RUN" --n-prompts 400000
#   sbatch data_pipelines/training_data/submit_training_data.sh
#
# After all jobs finish:
#   .venv/bin/python data_pipelines/training_data/generate_training_data.py merge-stage12 --run "$RUN" --num-shards 4

set -euo pipefail

cd /workspace-vast/adamk/activation_oracles_dev
source .env

RUN=synthetic_qa_2gpu_100k_qwen3_14b
NUM_SHARDS=2

echo "=== Shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS} on $(hostname), GPU ${CUDA_VISIBLE_DEVICES} ==="

.venv/bin/python data_pipelines/training_data/generate_training_data.py \
    stage12 \
    --run "${RUN}" \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --num-shards "${NUM_SHARDS}" \
    --stop-after-stage 2
