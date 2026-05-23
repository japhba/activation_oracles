#!/bin/bash
# Launch 100k completion generation: prep shards -> 4 parallel GPU jobs -> combine.
#
# Three Slurm jobs chained with dependencies:
#   1. CPU prep: stream WildChat, tokenize, split into 4 shards (~10-30 min)
#   2. GPU array (0-3): each loads its shard, generates completions (hours)
#   3. CPU combine: merge 4 parts into one file (minutes)
#
# Usage: bash data_pipelines/model_understanding/launch_100k_parallel.sh
set -euo pipefail

cd /workspace-vast/adamk/activation_oracles_dev

echo "Step 1: Submitting shard prep job (CPU, ~10-30 min)..."
PREP_JOB=$(sbatch --parsable data_pipelines/model_understanding/submit_prepare_shards.sh)
echo "  Prep job: ${PREP_JOB}"

echo "Step 2: Submitting 4 parallel generation jobs (GPU, 48h timeout each)..."
ARRAY_JOB=$(sbatch --parsable --dependency=afterok:${PREP_JOB} data_pipelines/model_understanding/submit_completions_100k_array.sh)
echo "  Array job: ${ARRAY_JOB} (tasks 0-3, depends on ${PREP_JOB})"

echo "Step 3: Submitting combine job (CPU, runs after all 4 complete)..."
COMBINE_JOB=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB} data_pipelines/model_understanding/submit_combine_100k.sh)
echo "  Combine job: ${COMBINE_JOB} (depends on ${ARRAY_JOB})"

echo ""
echo "Monitor: squeue -u \$USER"
echo "Shard output: data_pipelines/model_understanding/Qwen3-14B/shards_100k/"
echo "Final output: data_pipelines/model_understanding/Qwen3-14B/completions_100000p_10c.json"
