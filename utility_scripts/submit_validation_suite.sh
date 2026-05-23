#!/bin/bash
#SBATCH --job-name=build_val_suite
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: sbatch utility_scripts/submit_validation_suite.sh <training_config.json> <suite_config.json> [extra builder args...]"
  exit 1
fi

TRAINING_CONFIG="$1"
SUITE_CONFIG="$2"
shift 2

cd /workspace-vast/adamk/activation_oracles_dev
source .venv/bin/activate
source .env
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "=== Validation Suite Job on $(hostname) ==="
nvidia-smi

python utility_scripts/build_validation_suite.py \
  --training-config "$TRAINING_CONFIG" \
  --suite-config "$SUITE_CONFIG" \
  "$@"
