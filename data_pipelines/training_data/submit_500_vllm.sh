#!/bin/bash
#SBATCH --job-name=gen_500_vllm
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --qos=high
#SBATCH --output=data_pipelines/training_data/artifacts/slurm_500_%j.out

# Generate 500 entries (stages 1-2 only) for extended-thinking QA experiments.

set -euo pipefail

cd /workspace-vast/adamk/activation_oracles_dev
source .env

RUN=debug_500_vllm

if [ ! -f "data_pipelines/training_data/artifacts/${RUN}/prompts.json" ]; then
    .venv/bin/python data_pipelines/training_data/generate_training_data.py \
        prep \
        --run "${RUN}" \
        --n-prompts 500
fi

.venv/bin/python data_pipelines/training_data/generate_training_data.py \
    stage12 \
    --run "${RUN}" \
    --stop-after-stage 2
