"""
Sweep activation segment modes for system_prompt_qa evals.

Runs both hidden_instruction and latentqa datasets across multiple activation
modes to measure sensitivity to which tokens activations are extracted from.

Modes: user_and_assistant, assistant_only, user_only, user_last_30, assistant_last_30

Usage:
    source .env && .venv/bin/python experiments/system_prompt_qa_segment_sweep.py
"""

import json
import os
import random
import time

import torch

from nl_probes.open_ended_eval.system_prompt_qa import (
    ACTIVATION_MODES_EXTENDED,
    VERBALIZER_PROMPTS_HIDDEN_INSTRUCTION,
    VERBALIZER_PROMPTS_SYSTEM_PROMPT_QA,
    run_system_prompt_qa_open_ended_eval,
)
from nl_probes.utils.common import load_model, load_tokenizer

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = "Qwen/Qwen3-8B"

VERBALIZER_LORA_PATHS = [
    "checkpoints/500k_pl_31k_spqav2_199k_sqav3_126k_cls/final",
    "checkpoints/Qwen3-8B_full_mix_500k_past_lens_plus_synthetic_qa_v3/final",
]

EVAL_CONFIGS = [
    {
        "name": "hidden_instruction",
        "dataset_path": "data_pipelines/system_prompt_qa/Qwen3-8B/hidden_instruction_eval_dataset.json",
        "verbalizer_prompts": VERBALIZER_PROMPTS_HIDDEN_INSTRUCTION,
    },
    {
        "name": "latentqa",
        "dataset_path": "data_pipelines/system_prompt_qa/Qwen3-8B/latentqa_eval_dataset.json",
        "verbalizer_prompts": VERBALIZER_PROMPTS_SYSTEM_PROMPT_QA,
    },
]

MODES = ("user_and_assistant", "assistant_only", "user_only", "user_last_30", "assistant_last_30")

OUTPUT_DIR = "experiments/system_prompt_qa_segment_sweep_results"


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

    all_results: dict[str, dict[str, dict]] = {}  # lora -> eval_name -> result

    for lora_path in VERBALIZER_LORA_PATHS:
        lora_name = lora_path.split("/")[-2]  # e.g. "500k_pl_31k_..."
        all_results[lora_name] = {}

        for eval_cfg in EVAL_CONFIGS:
            eval_name = eval_cfg["name"]
            eval_output_dir = os.path.join(OUTPUT_DIR, lora_name, eval_name)
            os.makedirs(eval_output_dir, exist_ok=True)

            print(f"\n{'='*70}")
            print(f"  LoRA: {lora_name} | Eval: {eval_name}")
            print(f"  Modes: {MODES}")
            print(f"{'='*70}\n")

            start = time.perf_counter()
            result = run_system_prompt_qa_open_ended_eval(
                model_name=MODEL_NAME,
                model=model,
                tokenizer=tokenizer,
                device=device,
                output_dir=eval_output_dir,
                dataset_path=eval_cfg["dataset_path"],
                verbalizer_prompts=eval_cfg["verbalizer_prompts"],
                verbalizer_lora_paths=[lora_path],
                modes=MODES,
            )
            elapsed = time.perf_counter() - start
            result["elapsed_seconds"] = elapsed
            all_results[lora_name][eval_name] = result

            summary_path = os.path.join(eval_output_dir, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved summary to {summary_path} ({elapsed:.0f}s)")

    # Print comparison table
    print(f"\n{'='*90}")
    print("SEGMENT SENSITIVITY COMPARISON")
    print(f"{'='*90}")

    for lora_name, eval_results in all_results.items():
        for eval_name, result in eval_results.items():
            print(f"\n--- {lora_name} / {eval_name} ---")
            print(f"{'Mode':<25} {'Specificity':>12} {'Correctness':>12}")
            print("-" * 50)
            for mode, mode_result in result.get("mode_results", {}).items():
                om = mode_result.get("overall_metrics", {})
                spec = om.get("mean_specificity", 0)
                corr = om.get("mean_correctness", 0)
                print(f"{mode:<25} {spec:>12.2f} {corr:>12.2f}")

    # Save combined results
    combined_path = os.path.join(OUTPUT_DIR, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined_path}")


if __name__ == "__main__":
    main()
