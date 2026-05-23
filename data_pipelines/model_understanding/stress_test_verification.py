"""
Stress test the verification judge by corrupting investigation findings.

Creates corrupted variants of real investigations where experiment numbers
are swapped to wrong ones, then asks Opus to rewrite the explanation to
sound coherent with the wrong experiments. The judge should still detect
that the causal logic doesn't hold.

Corruption modes:
- naive: Swap experiment numbers, keep claims unchanged (easy to catch)
- rewrite: Swap experiment numbers, then have Opus rewrite the explanation
  to sound coherent with the wrong experiments (harder to catch)

Usage:
    source .env

    # Naive corruption (swap experiments, keep claims)
    .venv/bin/python data_pipelines/model_understanding/stress_test_verification.py \
        --input data_pipelines/model_understanding/Qwen3-14B/investigation_results_wip.json \
        --mode naive --n 5

    # Rewrite corruption (swap experiments, rewrite explanation)
    .venv/bin/python data_pipelines/model_understanding/stress_test_verification.py \
        --input data_pipelines/model_understanding/Qwen3-14B/investigation_results_wip.json \
        --mode rewrite --n 5
"""

import argparse
import asyncio
import copy
import functools
import json
import random
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

print = functools.partial(print, flush=True)

import anthropic

from data_pipelines.pipeline_utils import async_api_call, load_dotenv, run_concurrent
from data_pipelines.model_understanding.verify_investigations import (
    VERIFICATION_SYSTEM_PROMPTS,
    VERIFICATION_TOOL,
    VerificationResult,
    build_verification_prompt,
    load_investigations,
    verify_async,
    save_results,
)

load_dotenv()

REWRITE_MODEL = "claude-opus-4-6"
REWRITE_MAX_TOKENS = 4096
REWRITE_CONCURRENCY = 30


def corrupt_naive(investigation: dict, n_corrupt: int) -> dict:
    """Swap n_corrupt key_evidence experiment numbers to wrong ones.

    Claims stay unchanged — the judge should notice the completions
    don't match what's claimed. This is the easy-to-catch corruption.
    """
    inv = json.loads(json.dumps(investigation))
    all_exp_nums = [e["exp_num"] for e in inv["experiment_log"]]
    key_exp_nums = {ev["experiment_number"]
                    for ev in inv["structured_findings"]["key_evidence"]}
    other_nums = [n for n in all_exp_nums if n not in key_exp_nums]
    random.shuffle(other_nums)

    evidence = inv["structured_findings"]["key_evidence"]
    for i in range(min(n_corrupt, len(evidence))):
        if i < len(other_nums):
            evidence[i]["experiment_number"] = other_nums[i]

    inv["prompt_id"] = f"{inv['prompt_id']}_naive_{n_corrupt}"
    return inv


DESCRIBE_SYSTEM_PROMPT = """\
You are describing what a counterfactual experiment on a language model \
shows. You will see the original prompt, the modified prompt that was \
sent, and the completions the model produced. Describe:
1. What was changed in the prompt (one sentence)
2. The behavior rate in the completions (e.g., "7/10 completions do X")

Be accurate — count the completions carefully."""

DESCRIBE_TOOL = {
    "name": "describe_experiment",
    "description": "Describe what this experiment shows.",
    "input_schema": {
        "type": "object",
        "properties": {
            "what_changed": {
                "type": "string",
                "description": "One sentence: what was modified in the prompt compared to the original",
            },
            "counterfactual_rate": {
                "type": "string",
                "description": "Behavior rate in these completions (e.g., '7/10')",
            },
        },
        "required": ["what_changed", "counterfactual_rate"],
    },
}


def build_describe_prompt(
    investigation: dict, experiment: dict, question: str,
) -> str:
    """Ask Opus to describe what a single experiment shows."""
    parts = []
    parts.append(f"## Question Being Investigated\n{question}\n")
    parts.append(f"\n## Original User Prompt\n{investigation['user_message']}\n")
    parts.append(f"\n## Experiment Prompt Sent to Model\n{experiment['prompt']}\n")
    if experiment.get("system_prompt"):
        parts.append(f"\n**System prompt:** {experiment['system_prompt']}\n")
    parts.append(f"\n## Completions ({len(experiment['completions'])} total)\n")
    for i, comp in enumerate(experiment["completions"], 1):
        parts.append(f"--- Completion {i}/{len(experiment['completions'])} ---\n{comp}\n")
    parts.append(
        "\n---\n\nDescribe: (1) what was changed in the prompt compared "
        "to the original, and (2) the behavior rate in the completions "
        "relevant to the question above."
    )
    return "\n".join(parts)


async def corrupt_rewrite_one(
    investigation: dict,
    n_corrupt: int,
    client: anthropic.AsyncAnthropic,
) -> dict:
    """Swap experiments, then have Opus accurately describe only the swapped ones.

    Keeps the original answer, baseline_behavior, and unswapped evidence
    entries completely untouched. Only updates what_changed and
    counterfactual_rate for the swapped evidence entries so they
    accurately describe the (wrong) experiment shown.

    The judge must catch that the answer doesn't follow from the evidence.
    """
    inv = json.loads(json.dumps(investigation))  # deep copy
    original_findings = investigation["structured_findings"]
    exp_by_num = {e["exp_num"]: e for e in inv["experiment_log"]}

    all_exp_nums = [e["exp_num"] for e in inv["experiment_log"]]
    key_exp_nums = {ev["experiment_number"]
                    for ev in original_findings["key_evidence"]}
    other_nums = [n for n in all_exp_nums if n not in key_exp_nums]
    random.shuffle(other_nums)

    # Determine which evidence indices to swap
    evidence = copy.deepcopy(original_findings["key_evidence"])
    swapped_indices = []
    for i in range(min(n_corrupt, len(evidence))):
        if i < len(other_nums):
            evidence[i]["experiment_number"] = other_nums[i]
            swapped_indices.append(i)

    # Have Opus describe only the swapped experiments
    question = original_findings["question"]
    tasks = []
    for idx in swapped_indices:
        exp_num = evidence[idx]["experiment_number"]
        exp = exp_by_num[exp_num]
        tasks.append((idx, exp_num, exp))

    for idx, exp_num, exp in tasks:
        user_content = build_describe_prompt(investigation, exp, question)
        resp = await async_api_call(
            client,
            model=REWRITE_MODEL,
            max_tokens=1024,
            system=DESCRIBE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            tools=[DESCRIBE_TOOL],
            tool_choice={"type": "tool", "name": "describe_experiment"},
        )
        from data_pipelines.pipeline_utils import extract_tool_input
        desc = extract_tool_input(resp)
        evidence[idx]["what_changed"] = desc["what_changed"]
        evidence[idx]["counterfactual_rate"] = desc["counterfactual_rate"]

    # Apply: only swapped evidence entries change, everything else untouched
    inv["structured_findings"] = copy.deepcopy(original_findings)
    inv["structured_findings"]["key_evidence"] = evidence
    inv["prompt_id"] = f"{inv['prompt_id']}_rewrite_{n_corrupt}"
    return inv


async def generate_corruptions(
    investigations: list[dict],
    mode: str,
    n_corrupt_per: int,
) -> list[dict]:
    """Generate corrupted variants of investigations."""
    corrupted = []

    if mode == "naive":
        for inv in investigations:
            n_evidence = len(inv["structured_findings"]["key_evidence"])
            n_other = len([
                e["exp_num"] for e in inv["experiment_log"]
                if e["exp_num"] not in {ev["experiment_number"]
                                         for ev in inv["structured_findings"]["key_evidence"]}
            ])
            for k in range(1, min(n_corrupt_per, n_evidence, n_other) + 1):
                corrupted.append(corrupt_naive(inv, k))
        return corrupted

    elif mode == "rewrite":
        client = anthropic.AsyncAnthropic()
        semaphore = asyncio.Semaphore(REWRITE_CONCURRENCY)
        results = []

        async def rewrite_one(inv: dict, k: int):
            async with semaphore:
                result = await corrupt_rewrite_one(inv, k, client)
                results.append(result)
                print(f"  Rewrote {result['prompt_id']}")

        tasks = []
        for inv in investigations:
            n_evidence = len(inv["structured_findings"]["key_evidence"])
            n_other = len([
                e["exp_num"] for e in inv["experiment_log"]
                if e["exp_num"] not in {ev["experiment_number"]
                                         for ev in inv["structured_findings"]["key_evidence"]}
            ])
            for k in range(1, min(n_corrupt_per, n_evidence, n_other) + 1):
                tasks.append(rewrite_one(inv, k))

        print(f"Generating {len(tasks)} rewritten corruptions...")
        await asyncio.gather(*tasks)
        return results

    else:
        raise ValueError(f"Unknown mode: {mode}")


async def run_stress_test(
    investigations: list[dict],
    corrupted: list[dict],
    output_path: Path,
    input_path: Path,
    model: str,
    max_tokens: int,
    thinking_budget: int,
    concurrency: int,
    variant: str = "default",
    max_baseline_completions: int = 10,
):
    """Run verification on originals + corrupted, then compare."""
    all_investigations = list(investigations) + corrupted
    print(f"\nTotal: {len(investigations)} originals + {len(corrupted)} corrupted "
          f"= {len(all_investigations)}")

    results = await verify_async(
        all_investigations, output_path,
        model=model,
        max_tokens=max_tokens,
        thinking_budget=thinking_budget,
        concurrency=concurrency,
        variant=variant,
        max_baseline_completions=max_baseline_completions,
    )

    # Split results
    original_results = [r for r in results if "_naive_" not in r.prompt_id
                        and "_rewrite_" not in r.prompt_id]
    corrupt_results = [r for r in results if "_naive_" in r.prompt_id
                       or "_rewrite_" in r.prompt_id]

    print(f"\n{'='*60}")
    print("STRESS TEST RESULTS")
    print(f"{'='*60}")

    print(f"\n--- Originals (n={len(original_results)}) ---")
    if original_results:
        mean = sum(r.score for r in original_results) / len(original_results)
        dist = {s: sum(1 for r in original_results if r.score == s)
                for s in range(1, 6)}
        print(f"  Mean: {mean:.2f}  Distribution: {dist}")

    print(f"\n--- Corrupted (n={len(corrupt_results)}) ---")
    by_level = defaultdict(list)
    for r in corrupt_results:
        # Extract corruption level from prompt_id
        for tag in ("_naive_", "_rewrite_"):
            if tag in r.prompt_id:
                level = int(r.prompt_id.rsplit(tag, 1)[1])
                mode = tag.strip("_")
                by_level[(mode, level)].append(r.score)
                break

    for (mode, level) in sorted(by_level.keys()):
        scores = by_level[(mode, level)]
        mean = sum(scores) / len(scores)
        dist = {s: scores.count(s) for s in range(1, 6) if scores.count(s) > 0}
        print(f"  {mode}_{level}: mean={mean:.1f}  n={len(scores)}  {dist}")

    save_results(results, output_path, input_path, model=model)


def main():
    parser = argparse.ArgumentParser(
        description="Stress test verification judge with corrupted investigations",
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to investigation results JSON",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Limit to first N investigations",
    )
    parser.add_argument(
        "--mode", type=str, required=True, choices=["naive", "rewrite"],
        help="Corruption mode: naive (swap experiments) or rewrite (swap + rewrite explanation)",
    )
    parser.add_argument(
        "--n-corrupt", type=int, default=3,
        help="Max number of experiments to corrupt per investigation (default 3)",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Anthropic model for verification",
    )
    parser.add_argument(
        "--max-tokens", type=int, required=True,
        help="Max response tokens",
    )
    parser.add_argument(
        "--thinking-budget", type=int, required=True,
        help="Extended thinking budget",
    )
    parser.add_argument(
        "--concurrency", type=int, required=True,
        help="Max concurrent API calls",
    )
    parser.add_argument(
        "--variant", type=str, default="default",
        choices=["default", "relevance"],
        help="Verification prompt variant (default: default)",
    )
    parser.add_argument(
        "--max-baseline-completions", type=int, default=10,
        help="Max original completions to include in prompt (default 10)",
    )
    args = parser.parse_args()

    investigations = load_investigations(args.input, n=args.n)
    if not investigations:
        print("No investigations with structured findings")
        return

    output_dir = args.input.parent
    output_path = output_dir / f"stress_test_{args.mode}_{args.variant}_results.json"

    async def run():
        corrupted = await generate_corruptions(
            investigations, mode=args.mode, n_corrupt_per=args.n_corrupt,
        )
        await run_stress_test(
            investigations, corrupted, output_path, args.input,
            model=args.model,
            max_tokens=args.max_tokens,
            thinking_budget=args.thinking_budget,
            concurrency=args.concurrency,
            variant=args.variant,
            max_baseline_completions=args.max_baseline_completions,
        )

    asyncio.run(run())


if __name__ == "__main__":
    main()
