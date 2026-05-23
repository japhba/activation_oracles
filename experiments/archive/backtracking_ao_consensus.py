"""
Consensus-of-10 AO for backtracking eval with full context.

Usage:
    source .env && .venv/bin/python experiments/backtracking_ao_consensus.py
"""

import json
import os
import random
import time

import torch

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from nl_probes.open_ended_eval.backtracking import run_backtracking_open_ended_eval
from nl_probes.utils.common import load_model, load_tokenizer

MODEL_NAME = "Qwen/Qwen3-8B"
CHECKPOINT_DIR = "checkpoints/500k_pl_31k_spqav2_199k_sqav3_126k_cls_100k_chatreg/final"
NUM_ROLLOUTS = 10
FULL = 99999
OUTPUT_DIR = "experiments/backtracking_ao_consensus_10_full_context"


def main():
    random.seed(42)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = load_tokenizer(MODEL_NAME)
    print(f"Loading model: {MODEL_NAME} on {device} with dtype={dtype}")
    model = load_model(MODEL_NAME, dtype)
    model.eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nRunning backtracking AO consensus-of-{NUM_ROLLOUTS} with full context")
    print(f"Checkpoint: {CHECKPOINT_DIR}")

    start = time.perf_counter()
    result = run_backtracking_open_ended_eval(
        model_name=MODEL_NAME,
        model=model,
        tokenizer=tokenizer,
        device=device,
        verbalizer_lora_paths=[CHECKPOINT_DIR],
        output_dir=OUTPUT_DIR,
        num_rollouts=NUM_ROLLOUTS,
        multi_rollout_mode="consensus",
        segment_tokens=FULL,
    )
    elapsed = time.perf_counter() - start

    print(f"\n{'='*60}")
    print(f"RESULTS (consensus-of-{NUM_ROLLOUTS}, full context)")
    print(f"{'='*60}")
    om = result.get("overall_metrics", {})
    print(f"  mean_correctness: {om.get('mean_correctness', 0):.3f}")
    print(f"  mean_specificity: {om.get('mean_specificity', 0):.3f}")
    print(f"  specificity_>=3_rate: {om.get('specificity_>=3_rate', 0):.1%}")
    print(f"  correctness_>=3_rate: {om.get('correctness_>=3_rate', 0):.1%}")
    print(f"  elapsed: {elapsed:.0f}s")

    summary = {
        "checkpoint": CHECKPOINT_DIR,
        "num_rollouts": NUM_ROLLOUTS,
        "mode": "consensus",
        "segment_tokens": "full",
        "elapsed_seconds": elapsed,
        "overall_metrics": om,
        "full_result": result,
    }
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
