"""
Best-of-N text baseline skyline for backtracking eval.

Runs the base model N times (temperature=1.0) on the full context, judges all
responses, and picks the best per entry. This is a skyline — the best the model
can do with full text access and multiple attempts.

Usage:
    source .env && .venv/bin/python experiments/backtracking_text_baseline_best_of_n.py
"""

import asyncio
import json
import os
import random
import time

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch

from nl_probes.open_ended_eval.backtracking import (
    VERBALIZER_PROMPTS,
    compute_judge_metrics,
    judge_text_baseline_responses,
    load_backtracking_dataset,
)
from nl_probes.open_ended_eval.eval_runner import (
    render_baseline_chat_prompt,
)
from nl_probes.utils.common import load_tokenizer

MODEL_NAME = "Qwen/Qwen3-8B"
NUM_ROLLOUTS = 10
OUTPUT_DIR = "experiments/backtracking_text_baseline_best_of_10_full_context"


def build_full_context_prompts(entries, tokenizer, verbalizer_prompt):
    """Build full-context text baseline prompts (same as the existing baseline)."""
    prompts = []
    for entry in entries:
        prompt_text = render_baseline_chat_prompt(
            tokenizer=tokenizer,
            messages=[
                {"role": "user", "content": entry["problem"]},
                {"role": "assistant", "content": entry["prefix"]},
                {"role": "user", "content": verbalizer_prompt},
            ],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt_text)
    return prompts


def main():
    random.seed(42)
    torch.manual_seed(42)

    tokenizer = load_tokenizer(MODEL_NAME)
    entries = load_backtracking_dataset(MODEL_NAME)
    verbalizer_prompt = VERBALIZER_PROMPTS[0]

    print(f"Loaded {len(entries)} entries")
    print(f"Verbalizer prompt: {verbalizer_prompt}")
    print(f"Num rollouts: {NUM_ROLLOUTS}")

    # Build prompts
    base_prompts = build_full_context_prompts(entries, tokenizer, verbalizer_prompt)

    # Replicate each prompt NUM_ROLLOUTS times for batched vLLM inference
    all_prompts = []
    prompt_to_entry_idx = []
    for idx, prompt in enumerate(base_prompts):
        for _ in range(NUM_ROLLOUTS):
            all_prompts.append(prompt)
            prompt_to_entry_idx.append(idx)

    print(f"Total prompts to run: {len(all_prompts)}")

    # Run vLLM inference
    import vllm
    sampling_params = vllm.SamplingParams(
        temperature=1.0,
        max_tokens=150,
    )

    print(f"\nLoading vLLM model {MODEL_NAME}...")
    llm = vllm.LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        enforce_eager=True,
        enable_lora=False,
        gpu_memory_utilization=0.7,
    )

    print("Generating responses...")
    start = time.perf_counter()
    outputs = llm.generate(all_prompts, sampling_params=sampling_params, use_tqdm=True)
    gen_elapsed = time.perf_counter() - start
    print(f"Generation done in {gen_elapsed:.0f}s")

    del llm
    torch.cuda.empty_cache()

    # Group responses by entry
    grouped_responses: list[list[str]] = [[] for _ in range(len(entries))]
    for output, entry_idx in zip(outputs, prompt_to_entry_idx):
        grouped_responses[entry_idx].append(output.outputs[0].text)

    # Judge ALL responses individually
    print(f"\nJudging {len(all_prompts)} responses...")

    # Flatten for judging
    flat_responses = []
    flat_entries = []
    flat_rollout_idx = []
    for entry_idx, (entry, responses) in enumerate(zip(entries, grouped_responses)):
        for rollout_idx, response in enumerate(responses):
            flat_responses.append(response)
            flat_entries.append(entry)
            flat_rollout_idx.append(rollout_idx)

    judge_start = time.perf_counter()

    # Use a simple TextBaselineGenerationResult-like object for judging
    from dataclasses import dataclass

    @dataclass
    class FakeResult:
        prompt_text: str
        response_text: str

    fake_results = [FakeResult(prompt_text="", response_text=r) for r in flat_responses]
    fake_metadata = [{"baseline_variant": "full_context"} for _ in flat_responses]

    scored = asyncio.run(
        judge_text_baseline_responses(
            results=fake_results,
            entries=flat_entries,
            concurrency=20,
            extra_metadata=fake_metadata,
        )
    )
    judge_elapsed = time.perf_counter() - judge_start
    print(f"Judging done in {judge_elapsed:.0f}s, {len(scored)} scored")

    # Group scored results back by entry and pick best
    scored_by_entry: list[list[dict]] = [[] for _ in range(len(entries))]
    for s, entry_idx in zip(scored, [prompt_to_entry_idx[i] for i in range(len(scored))]):
        scored_by_entry[entry_idx].append(s)

    # Wait - the scored list may be shorter if some failed. Let me track properly.
    # Actually, judge_text_baseline_responses iterates zip(results, entries) so it's 1:1.
    # Let me regroup properly.
    scored_by_entry = [[] for _ in range(len(entries))]
    idx = 0
    for entry_idx in range(len(entries)):
        for _rollout_idx in range(NUM_ROLLOUTS):
            if idx < len(scored):
                scored[idx]["entry_idx"] = entry_idx
                scored[idx]["rollout_idx"] = _rollout_idx
                scored_by_entry[entry_idx].append(scored[idx])
            idx += 1

    # Pick best per entry (max correctness, break ties with specificity)
    best_per_entry = []
    all_rollout_details = []
    for entry_idx, entry_scored in enumerate(scored_by_entry):
        if not entry_scored:
            continue
        best = max(entry_scored, key=lambda s: (s.get("correctness", 0), s.get("specificity", 0)))
        best["selection"] = "best_of_n"
        best_per_entry.append(best)
        all_rollout_details.append({
            "entry_id": entries[entry_idx].get("id", entry_idx),
            "num_rollouts_scored": len(entry_scored),
            "best_rollout_idx": best.get("rollout_idx"),
            "best_correctness": best.get("correctness"),
            "best_specificity": best.get("specificity"),
            "all_correctness": [s.get("correctness") for s in entry_scored],
            "all_specificity": [s.get("specificity") for s in entry_scored],
        })

    # Compute metrics on best-of-N selections
    best_metrics = compute_judge_metrics(best_per_entry)
    # Also compute greedy (first rollout only) for comparison
    greedy_per_entry = [entry_scored[0] for entry_scored in scored_by_entry if entry_scored]
    greedy_metrics = compute_judge_metrics(greedy_per_entry)
    # And average across all rollouts
    all_metrics = compute_judge_metrics(scored)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nGreedy (single rollout, temp=1):")
    print(f"  mean_correctness: {greedy_metrics.get('mean_correctness', 0):.3f}")
    print(f"  mean_specificity: {greedy_metrics.get('mean_specificity', 0):.3f}")
    print(f"\nAverage across all {NUM_ROLLOUTS} rollouts:")
    print(f"  mean_correctness: {all_metrics.get('mean_correctness', 0):.3f}")
    print(f"  mean_specificity: {all_metrics.get('mean_specificity', 0):.3f}")
    print(f"\nBest-of-{NUM_ROLLOUTS} (skyline):")
    print(f"  mean_correctness: {best_metrics.get('mean_correctness', 0):.3f}")
    print(f"  mean_specificity: {best_metrics.get('mean_specificity', 0):.3f}")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_data = {
        "model_name": MODEL_NAME,
        "num_rollouts": NUM_ROLLOUTS,
        "num_entries": len(entries),
        "verbalizer_prompt": verbalizer_prompt,
        "generation_elapsed_seconds": gen_elapsed,
        "judge_elapsed_seconds": judge_elapsed,
        "greedy_metrics": greedy_metrics,
        "average_metrics": all_metrics,
        "best_of_n_metrics": best_metrics,
        "best_per_entry": best_per_entry,
        "rollout_details": all_rollout_details,
    }
    output_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
