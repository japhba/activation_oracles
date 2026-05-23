#!/bin/bash
#SBATCH --job-name=mu_combine_100k
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --qos=high
#SBATCH --output=data_pipelines/model_understanding/logs/slurm_combine_%j.out

set -euo pipefail

cd /workspace-vast/adamk/activation_oracles_dev

DATA_DIR="data_pipelines/model_understanding/Qwen3-14B"

.venv/bin/python data_pipelines/model_understanding/combine_completions.py \
    --inputs \
        ${DATA_DIR}/completions_100k_part1.json \
        ${DATA_DIR}/completions_100k_part2.json \
        ${DATA_DIR}/completions_100k_part3.json \
        ${DATA_DIR}/completions_100k_part4.json \
    --output ${DATA_DIR}/completions_100000p_10c.json
