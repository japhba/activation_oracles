#!/bin/bash
# Generate AITA sycophancy dataset for Qwen3-14B.
# Usage: sbatch data_pipelines/sycophancy_aita/submit_generate.sh

#SBATCH --job-name=gen_syc_aita
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /workspace-vast/adamk/activation_oracles_dev
source .venv/bin/activate
source .env

echo "=== Generating AITA sycophancy dataset for Qwen3-14B ==="
python data_pipelines/sycophancy_aita/generate_dataset.py --model Qwen/Qwen3-14B

echo "=== Done ==="
