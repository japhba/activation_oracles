"""
Stage 4: Verify investigation findings with an independent LLM judge.

Takes structured_findings + experiment_log from stage 3 and constructs
self-contained verification prompts. The judge checks whether the claimed
behavior rates match the actual completions and whether the causal
explanation follows from the evidence.

No transcript parsing — prompts are assembled entirely from structured data.

Usage:
    source .env

    # Preview prompts (dry run, no API calls)
    .venv/bin/python data_pipelines/model_understanding/verify_investigations.py \
        --input data_pipelines/model_understanding/Qwen3-14B/investigation_results.json \
        --dry-run --n 3

    # Run verification via async API
    .venv/bin/python data_pipelines/model_understanding/verify_investigations.py \
        --input data_pipelines/model_understanding/Qwen3-14B/investigation_results.json

    # Run verification via batch API
    .venv/bin/python data_pipelines/model_understanding/verify_investigations.py \
        --input data_pipelines/model_understanding/Qwen3-14B/investigation_results.json \
        --batch
"""

import argparse
import asyncio
import functools
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

print = functools.partial(print, flush=True)

import anthropic

from data_pipelines.pipeline_utils import async_api_call, extract_tool_input, load_dotenv, run_concurrent

load_dotenv()


VERIFICATION_SYSTEM_PROMPT_DEFAULT = """\
You are verifying the findings of an investigation into language model behavior.

An investigator ran counterfactual experiments on a language model to answer a \
question about why the model behaves a certain way. You will see:

1. The question that was investigated
2. The claimed baseline behavior (what the model does on the original prompt)
3. The investigator's answer (their causal explanation)
4. 3-5 key experiments with:
   - What the investigator claims they changed
   - The claimed behavior rates (e.g., "8/10" → "1/10")
   - The ACTUAL prompt that was sent to the model
   - The ACTUAL completions the model produced

## Your task

For EACH key experiment, check all three of these:

1. **Rate accuracy**: Count the completions that show the behavior and \
compare to the claimed rate. Be specific — state your count.

2. **Change description accuracy**: Compare the original prompt to the \
experiment's actual prompt. Does the "what changed" description \
accurately describe the real difference? Is there anything else that \
changed that the description omits?

3. **Relevance to the causal claim**: Does this experiment actually \
test the investigator's stated answer? A good experiment isolates a \
specific factor from the causal explanation and shows it matters. \
A bad experiment might have accurate rates but tests something \
unrelated to the stated cause, or changes too many things at once \
to isolate anything. Ask: "If I only saw this experiment, would it \
give me evidence for or against the stated answer?" If not, it's \
irrelevant padding.

Then assess overall:
4. **Does the evidence support the answer?** How many of the key \
experiments actually provide evidence for the causal claim? Are there \
obvious alternative explanations the experiments don't rule out? \
Does the investigator's own evidence contradict their conclusion?

Be a skeptical reviewer. Don't give credit for experiments that are \
tangentially related — demand that each key experiment directly tests \
a component of the stated causal explanation."""

VERIFICATION_SYSTEM_PROMPT_RELEVANCE = """\
You are verifying the findings of an investigation into language model behavior.

An investigator claims to have found why a model behaves a certain way, \
and presents 3-5 "key experiments" as evidence. Your primary job is to \
determine how many of these experiments ACTUALLY support the causal claim.

## Scoring rule

Your score should be driven primarily by this formula:
- Count how many key experiments are RELEVANT (directly test a component \
of the stated causal explanation)
- Count how many are IRRELEVANT (test something unrelated, or their \
results don't bear on the causal claim)
- An experiment with accurate rates but testing an unrelated factor is \
IRRELEVANT

Score guide based on relevant experiment count:
- 5/5 relevant + accurate rates = 9-10
- 4/5 relevant + accurate rates = 7-8
- 3/5 relevant = 5-6
- 2/5 relevant = 3-4
- 0-1/5 relevant = 1-2

Adjust down 1-2 points for rate inaccuracies or unaddressed confounds.

## How to assess relevance

For each experiment, read the causal answer, then ask: \
"Does this experiment test or isolate a specific factor mentioned in \
the answer?" If the experiment changes something the answer doesn't \
discuss, or if the experiment's results don't distinguish between the \
stated explanation and obvious alternatives, mark it irrelevant.

## What to check

For each key experiment:
1. Count the actual behavior rate in completions
2. Check if "what changed" matches the actual prompt difference
3. Decide: relevant or irrelevant to the causal claim?"""

VERIFICATION_SYSTEM_PROMPTS = {
    "default": VERIFICATION_SYSTEM_PROMPT_DEFAULT,
    "relevance": VERIFICATION_SYSTEM_PROMPT_RELEVANCE,
}

VERIFICATION_TOOL = {
    "name": "submit_verification",
    "description": "Submit your verification judgment for this investigation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": (
                    "1-2: Fundamentally unsupported — most experiments are "
                    "irrelevant to the causal claim, or the evidence actively "
                    "contradicts the stated answer. "
                    "3-4: Major gaps — key experiments don't isolate the "
                    "claimed factors, important confounds unaddressed, or "
                    "multiple rate claims are significantly wrong (off by 3+). "
                    "5-6: Moderate issues — most experiments are relevant but "
                    "one or two don't actually test the stated cause, or "
                    "rate miscounts (off by 2) affect interpretation. The "
                    "core claim is plausible but not fully locked down. "
                    "7-8: Minor issues — all experiments are relevant to the "
                    "claim, small rate miscounts (±1) that don't affect the "
                    "takeaway, or very minor gaps in the explanation. "
                    "9-10: Fully verified — all rates match, every experiment "
                    "directly tests a component of the causal explanation, "
                    "and the explanation follows cleanly from evidence."
                ),
            },
            "experiment_assessments": {
                "type": "array",
                "description": (
                    "One assessment per key experiment. For each: is the rate "
                    "accurate, is the change description accurate, and does "
                    "this experiment actually test the stated causal claim?"
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "experiment_number": {
                            "type": "integer",
                            "description": "Which experiment this assessment is for",
                        },
                        "rate_accurate": {
                            "type": "boolean",
                            "description": "Does the claimed rate match actual completions (within ±1)?",
                        },
                        "actual_rate": {
                            "type": "string",
                            "description": "Your count of the actual rate (e.g., '7/10')",
                        },
                        "change_description_accurate": {
                            "type": "boolean",
                            "description": "Does 'what changed' accurately describe the real prompt difference?",
                        },
                        "relevant_to_causal_claim": {
                            "type": "boolean",
                            "description": (
                                "Does this experiment directly test a component "
                                "of the stated causal explanation? false if it "
                                "tests something tangential or unrelated."
                            ),
                        },
                        "notes": {
                            "type": "string",
                            "description": "Brief explanation of any issues with this experiment",
                        },
                    },
                    "required": [
                        "experiment_number", "rate_accurate", "actual_rate",
                        "change_description_accurate", "relevant_to_causal_claim", "notes",
                    ],
                },
            },
            "explanation_assessment": {
                "type": "string",
                "description": (
                    "Does the causal explanation follow from the experimental "
                    "evidence? How many experiments are relevant vs irrelevant? "
                    "Note any gaps or alternative explanations."
                ),
            },
            "issues": {
                "type": "array",
                "description": "List of specific issues found (empty if none)",
                "items": {
                    "type": "string",
                },
            },
        },
        "required": ["score", "experiment_assessments", "explanation_assessment", "issues"],
    },
}


@dataclass
class VerificationResult:
    prompt_id: str
    score: int
    experiment_assessments: list[dict]
    explanation_assessment: str
    issues: list[str]
    verified_at: str


def build_verification_prompt(
    investigation: dict, max_baseline_completions: int = 10,
) -> str:
    """Build a self-contained verification prompt from structured data.

    Pulls key_evidence experiment numbers from structured_findings and
    looks them up in experiment_log to get actual prompts and completions.
    """
    findings = investigation["structured_findings"]
    experiment_log = investigation["experiment_log"]

    # Index experiments by number for lookup
    exp_by_num = {e["exp_num"]: e for e in experiment_log}

    parts = []

    # Header with the claims
    parts.append("## Investigation Claims\n")
    parts.append(f"**Question:** {findings['question']}\n")
    parts.append(f"**Claimed baseline behavior:** {findings['baseline_behavior']}\n")
    parts.append(f"**Answer (causal explanation):** {findings['answer']}\n")
    parts.append(f"**Confidence:** {findings['confidence']}\n")

    # Original prompt and completions for baseline reference
    parts.append("\n## Original Prompt\n")
    parts.append(f"{investigation['user_message']}\n")

    completions = investigation["completions"]
    n_show = min(max_baseline_completions, len(completions))
    parts.append(f"\n## Original Completions ({n_show} of {len(completions)} "
                 f"completions from the target model)\n")
    for i, comp in enumerate(completions[:n_show], 1):
        parts.append(f"--- Completion {i}/{n_show} ---\n{comp}\n")

    # Key experiments with actual data
    parts.append("\n## Key Experiments (investigator's 3-5 most decisive)\n")
    parts.append(
        "For each experiment below, verify that the claimed behavior rate "
        "matches what you see in the actual completions.\n"
    )

    for i, evidence in enumerate(findings["key_evidence"], 1):
        if not isinstance(evidence, dict):
            parts.append(f"\n### Key Experiment {i}\n(malformed evidence entry, skipping)\n")
            continue
        exp_num = evidence["experiment_number"]
        exp = exp_by_num.get(exp_num)

        parts.append(f"\n### Key Experiment {i} (Experiment #{exp_num})\n")
        parts.append(f"**Claim — what changed:** {evidence['what_changed']}\n")
        parts.append(f"**Claimed baseline rate:** {evidence['baseline_rate']}\n")
        parts.append(f"**Claimed counterfactual rate:** {evidence['counterfactual_rate']}\n")

        if exp is None:
            parts.append(
                f"\n⚠ Experiment #{exp_num} not found in experiment log.\n"
            )
            continue

        if exp.get("messages") and len(exp["messages"]) > 1:
            parts.append(f"\n**Actual conversation sent to the model "
                         f"({len(exp['messages'])} messages):**\n")
            for msg in exp["messages"]:
                role = msg["role"].upper()
                parts.append(f"[{role}]: {msg['content']}\n")
        else:
            parts.append(
                f"\n**Actual prompt sent to the model:**\n{exp['prompt']}\n"
            )
        if exp.get("system_prompt"):
            parts.append(f"\n**System prompt:** {exp['system_prompt']}\n")

        parts.append(f"\n**Actual completions ({len(exp['completions'])} total):**\n")
        for j, comp in enumerate(exp["completions"], 1):
            parts.append(f"--- Completion {j}/{len(exp['completions'])} ---\n{comp}\n")

    parts.append(
        "\n---\n\n"
        "Verify the claims above. Check each experiment's actual completions "
        "against the claimed rates, then assess whether the overall causal "
        "explanation follows from the evidence."
    )

    return "\n".join(parts)


def load_investigations(input_path: Path, n: int | None) -> list[dict]:
    """Load investigations that have structured_findings."""
    data = json.loads(input_path.read_text())
    results = data["results"] if "results" in data else data

    # Filter to investigations with structured findings
    with_findings = [r for r in results if r.get("structured_findings")]
    without = len(results) - len(with_findings)
    if without > 0:
        print(f"Skipping {without} investigations without structured_findings")

    if n is not None:
        with_findings = with_findings[:n]

    print(f"Loaded {len(with_findings)} investigations with structured findings")
    return with_findings


async def verify_async(
    investigations: list[dict],
    output_path: Path,
    model: str,
    max_tokens: int,
    thinking_budget: int,
    concurrency: int,
    variant: str = "default",
    max_baseline_completions: int = 10,
):
    """Verify investigations using the normal async API."""
    system_prompt = VERIFICATION_SYSTEM_PROMPTS[variant]
    print(f"\nVerifying {len(investigations)} investigations with "
          f"{model} (concurrency={concurrency}, "
          f"variant={variant}, max_baseline={max_baseline_completions})...")

    client = anthropic.AsyncAnthropic()
    results: list[VerificationResult] = []
    jsonl_path = output_path.with_suffix(".jsonl")

    # Resume from existing results
    completed_ids = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    completed_ids.add(record["prompt_id"])
                    results.append(VerificationResult(**record))
        print(f"  Resuming: {len(completed_ids)} already verified")

    remaining = [inv for inv in investigations
                 if inv["prompt_id"] not in completed_ids]
    if not remaining:
        print("  All investigations already verified")
        return results

    jsonl_file = open(jsonl_path, "a")
    write_lock = asyncio.Lock()
    skipped = {"count": 0}

    async def verify_one(i: int, investigation: dict):
        user_content = build_verification_prompt(
            investigation, max_baseline_completions=max_baseline_completions,
        )
        try:
            resp = await async_api_call(
                client,
                model=model,
                max_tokens=max_tokens,
                thinking={"type": "enabled", "budget_tokens": thinking_budget},
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
                tools=[VERIFICATION_TOOL],
            )
            tool_input = extract_tool_input(resp)

            result = VerificationResult(
                prompt_id=investigation["prompt_id"],
                score=tool_input["score"],
                experiment_assessments=tool_input.get("experiment_assessments", []),
                explanation_assessment=tool_input["explanation_assessment"],
                issues=tool_input.get("issues", []),
                verified_at=datetime.now(timezone.utc).isoformat(),
            )
            results.append(result)

            async with write_lock:
                jsonl_file.write(json.dumps(asdict(result)) + "\n")
                jsonl_file.flush()
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            skipped["count"] += 1
            print(f"  SKIPPING {investigation['prompt_id']}: malformed result ({e})")

    await run_concurrent(
        verify_one,
        remaining,
        concurrency=concurrency,
        label="Verification",
        progress_interval=10,
    )
    jsonl_file.close()

    if remaining:
        skip_rate = skipped["count"] / len(remaining)
        assert skip_rate < 0.001, (
            f"Verification skip rate {skip_rate:.1%} "
            f"({skipped['count']}/{len(remaining)}) exceeds 0.1% threshold"
        )

    return results


def verify_batch(
    investigations: list[dict],
    output_path: Path,
    model: str,
    max_tokens: int,
    batch_chunk_size: int,
    variant: str = "default",
):
    """Verify investigations using the batch API."""
    print(f"\nVerifying {len(investigations)} investigations with "
          f"{model} (batch API, variant={variant})...")

    batch_state_path = output_path.with_suffix(".batch_state.json")
    jsonl_path = output_path.with_suffix(".jsonl")

    system_prompt = VERIFICATION_SYSTEM_PROMPTS[variant]

    requests = []
    for inv in investigations:
        user_content = build_verification_prompt(inv)
        requests.append({
            "custom_id": inv["prompt_id"],
            "params": {
                "model": model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_content}],
                "tools": [VERIFICATION_TOOL],
                "tool_choice": {"type": "tool", "name": "submit_verification"},
            },
        })

    batch_api_key = os.environ.get("ANTHROPIC_API_KEY_BATCH_API")
    assert batch_api_key, "ANTHROPIC_API_KEY_BATCH_API not set in environment"
    client = anthropic.Anthropic(api_key=batch_api_key)

    chunks = [requests[i:i+batch_chunk_size]
              for i in range(0, len(requests), batch_chunk_size)]
    print(f"  Split into {len(chunks)} batch chunks of up to {batch_chunk_size}")

    # Load or create batch state
    batch_ids = []
    if batch_state_path.exists():
        saved = json.loads(batch_state_path.read_text())
        batch_ids = saved.get("batch_ids", [])
        if batch_ids:
            print(f"  Resuming {len(batch_ids)} batches")

    # Submit any chunks that haven't been submitted yet
    while len(batch_ids) < len(chunks):
        chunk_idx = len(batch_ids)
        chunk = chunks[chunk_idx]
        batch = client.messages.batches.create(requests=chunk)
        batch_ids.append(batch.id)
        batch_state_path.write_text(json.dumps({
            "batch_ids": batch_ids,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "n_chunks": len(chunks),
            "chunk_size": batch_chunk_size,
            "total_requests": len(requests),
        }, indent=2))
        print(f"  Submitted chunk {chunk_idx+1}/{len(chunks)}: "
              f"{batch.id} ({len(chunk)} requests)")

    # Poll until all batches done
    while True:
        all_done = True
        for i, bid in enumerate(batch_ids):
            status = client.messages.batches.retrieve(bid)
            counts = status.request_counts
            if status.processing_status != "ended":
                all_done = False
            print(
                f"  [{i+1}/{len(batch_ids)}] {bid}: {status.processing_status} "
                f"(ok={counts.succeeded} err={counts.errored} "
                f"exp={counts.expired} pending={counts.processing})"
            )
        if all_done:
            break
        time.sleep(60)

    # Retrieve results
    results = []
    n_skipped = 0
    jsonl_file = open(jsonl_path, "w")

    for bid in batch_ids:
        for batch_result in client.messages.batches.results(bid):
            pid = batch_result.custom_id

            if batch_result.result.type != "succeeded":
                print(f"  SKIPPING {pid}: batch result {batch_result.result.type}")
                continue

            try:
                tool_input = extract_tool_input(batch_result.result.message)
                result = VerificationResult(
                    prompt_id=pid,
                    score=tool_input["score"],
                    experiment_assessments=tool_input.get("experiment_assessments", []),
                    explanation_assessment=tool_input["explanation_assessment"],
                    issues=tool_input.get("issues", []),
                    verified_at=datetime.now(timezone.utc).isoformat(),
                )
                results.append(result)
                jsonl_file.write(json.dumps(asdict(result)) + "\n")
            except (KeyError, TypeError, ValueError) as e:
                n_skipped += 1
                print(f"  SKIPPING {pid}: malformed verification result ({e})")

    jsonl_file.close()
    if n_skipped:
        print(f"  Skipped {n_skipped} malformed verification results")
    return results


def save_results(
    results: list[VerificationResult], output_path: Path, input_path: Path,
    model: str,
):
    """Save verification results as JSON."""
    output = {
        "metadata": {
            "verification_model": model,
            "input_file": str(input_path),
            "n_verified": len(results),
            "score_distribution": {
                str(s): sum(1 for r in results if r.score == s)
                for s in range(1, 11)
            },
            "mean_score": round(sum(r.score for r in results) / len(results), 2) if results else 0,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": [asdict(r) for r in results],
    }

    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(results)} verification results to {output_path}")

    # Summary
    print(f"\n--- Score Distribution ---")
    for s in range(10, 0, -1):
        count = sum(1 for r in results if r.score == s)
        if count > 0:
            print(f"  Score {s:2d}: {count}")
    if results:
        mean = sum(r.score for r in results) / len(results)
        print(f"  Mean: {mean:.2f}")



def main():
    parser = argparse.ArgumentParser(
        description="Verify investigation findings with independent LLM judge",
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to investigation results JSON",
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
        "--n", type=int, default=None,
        help="Limit to first N investigations",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Use batch API instead of async",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print constructed prompts without making API calls",
    )
    parser.add_argument(
        "--variant", type=str, default="default",
        choices=list(VERIFICATION_SYSTEM_PROMPTS.keys()),
        help="Which system prompt variant to use (default: default)",
    )
    parser.add_argument(
        "--max-baseline-completions", type=int, default=10,
        help="Max original completions to include in prompt (default 10)",
    )
    parser.add_argument(
        "--thinking-budget", type=int, default=None,
        help="Extended thinking budget (required for async mode)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help="Max concurrent API calls (required for async mode)",
    )
    parser.add_argument(
        "--batch-chunk-size", type=int, default=None,
        help="Requests per batch chunk (required for batch mode)",
    )
    args = parser.parse_args()

    investigations = load_investigations(args.input, n=args.n)

    if not investigations:
        print("No investigations with structured findings to verify")
        return

    if args.dry_run:
        for inv in investigations:
            pid = inv["prompt_id"]
            prompt = build_verification_prompt(
                inv, max_baseline_completions=args.max_baseline_completions,
            )
            print(f"\n{'='*80}")
            print(f"VERIFICATION PROMPT FOR {pid}")
            print(f"{'='*80}\n")
            print(prompt)
            print(f"\n[Prompt length: {len(prompt)} chars]")
        return

    # Output naming
    output_dir = args.input.parent
    suffix = f"_{args.variant}" if args.variant != "default" else ""
    if args.max_baseline_completions != 10:
        suffix += f"_base{args.max_baseline_completions}"
    output_path = output_dir / f"verification_{len(investigations)}{suffix}_results.json"

    if args.batch:
        assert args.batch_chunk_size is not None, "--batch-chunk-size is required for batch mode"
        results = verify_batch(
            investigations, output_path,
            model=args.model,
            max_tokens=args.max_tokens,
            batch_chunk_size=args.batch_chunk_size,
            variant=args.variant,
        )
    else:
        assert args.thinking_budget is not None, "--thinking-budget is required for async mode"
        assert args.concurrency is not None, "--concurrency is required for async mode"
        results = asyncio.run(verify_async(
            investigations, output_path,
            model=args.model,
            max_tokens=args.max_tokens,
            thinking_budget=args.thinking_budget,
            concurrency=args.concurrency,
            variant=args.variant,
            max_baseline_completions=args.max_baseline_completions,
        ))

    save_results(results, output_path, args.input, model=args.model)


if __name__ == "__main__":
    main()
