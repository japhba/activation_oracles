#!/bin/bash
#SBATCH --job-name=mu_prep_shards
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --qos=high
#SBATCH --output=data_pipelines/model_understanding/logs/slurm_prep_%j.out

set -euo pipefail

cd /workspace-vast/adamk/activation_oracles_dev
source .env

.venv/bin/python data_pipelines/model_understanding/prepare_shards.py \
    --model Qwen/Qwen3-14B \
    --n-prompts 100000 \
    --n-shards 4 \
    --output-dir data_pipelines/model_understanding/Qwen3-14B/shards_100k
