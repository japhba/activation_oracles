#!/bin/bash
#SBATCH --job-name=mu_comp_10k
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --qos=high
#SBATCH --output=data_pipelines/model_understanding/logs/slurm_%j.out

set -euo pipefail

cd /workspace-vast/adamk/activation_oracles_dev
source .env

.venv/bin/python data_pipelines/model_understanding/generate_completions.py \
    --model Qwen/Qwen3-14B \
    --n-prompts 10000 \
    --n-completions 10
