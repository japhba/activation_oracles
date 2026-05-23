#!/bin/bash
# Generate AITA sycophancy dataset for Qwen3-32B-FP8 with 1000 posts.
# Usage: sbatch data_pipelines/sycophancy_aita/submit_generate_32b_1k.sh

#SBATCH --job-name=gen_aita_32b_1k
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /workspace-vast/adamk/activation_oracles_dev
source .venv/bin/activate
source .env

echo "=== Generating AITA sycophancy dataset for Qwen3-32B-FP8 (1000 posts, no-cot) ==="
python data_pipelines/sycophancy_aita/generate_dataset.py \
    --model Qwen/Qwen3-32B-FP8 \
    --num-posts 1000 \
    --no-cot-only

echo "=== Done ==="
