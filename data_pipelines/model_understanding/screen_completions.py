"""
Screen model completions for interesting/surprising behaviors.

Reads multi-completion data from generate_completions.py (multiple completions
per prompt), sends each prompt+completions to Sonnet for analysis, and outputs
scored results. Supports both normal async API and batch API.

Usage:
    source .env

    # Quick test on first 50 prompts
    .venv/bin/python data_pipelines/model_understanding/screen_completions.py \
        --input data_pipelines/model_understanding/Qwen3-14B/completions_1000p_10c.json \
        --n 50

    # Full run via batch API
    .venv/bin/python data_pipelines/model_understanding/screen_completions.py \
        --input data_pipelines/model_understanding/Qwen3-14B/completions_1000p_10c.json \
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
import random

from data_pipelines.pipeline_utils import async_api_call, extract_tool_input, run_concurrent

# Few-shot question pool — randomly sample 5 per screening call for diversity
FEW_SHOT_POOL_PATH = Path(__file__).resolve().parent / "few_shot_question_pool.json"
FEW_SHOT_POOL: list[dict] = []
if FEW_SHOT_POOL_PATH.exists():
    FEW_SHOT_POOL = json.loads(FEW_SHOT_POOL_PATH.read_text())


def _sample_few_shot_examples(n: int = 5) -> str:
    """Sample n random examples from the few-shot pool and format them."""
    if not FEW_SHOT_POOL:
        return ""
    samples = random.sample(FEW_SHOT_POOL, min(n, len(FEW_SHOT_POOL)))
    parts = ["\n## EXAMPLES OF GOOD QUESTIONS (randomly sampled)\n"]
    for i, s in enumerate(samples, 1):
        parts.append(
            f"Example {i}:\n"
            f"  Behavior: {s['behavior_summary']}\n"
            f"  Question: {s['question']}\n"
        )
    return "\n".join(parts)

SCREENING_SYSTEM_PROMPT = """\
You are screening outputs from an open-source language model to find \
cases where the model's behavior is surprising, coherent, and hard to \
explain — where there are MULTIPLE PLAUSIBLE HYPOTHESES for why the model \
did what it did.

You will see a user prompt and 10 independent completions from the model \
(all generated from the same prompt with temperature=1).

Your job: find cases where you can ask "Why did the model do X?" and the \
answer is genuinely not obvious. The best cases are ones where you can \
think of 2+ plausible explanations for the behavior, and a counterfactual \
experiment could distinguish between them.

## WHAT MAKES A GREAT QUESTION (score 4-5)

The strongest cases share two properties — both are necessary:

1. **The causal factor is non-obvious from reading the prompt and output.** \
   You should NOT be able to guess what drives the behavior just by \
   reading the text. The cause should live in the model's processing of \
   the input, not in the surface structure. If someone says "obviously \
   the model did X because of word Y," it's too easy.

2. **Counterfactuals produce clean, dramatic swings.** Not mushy 3/10 → \
   5/10 shifts that could be noise, but 0/10 → 10/10 or 1/10 → 9/10 \
   flips. The behavior should be binary enough that you can clearly say \
   "this factor flipped the behavior."

## WHAT WE'RE INTERESTED IN (diverse range)

We want a diverse range of question types. All of these are interesting:

**Non-obvious model judgments:** The model made a coherent choice that \
isn't obviously wrong — it's just non-obvious what drove it.

**Mistakes the model should be capable of avoiding:** The model gets \
something wrong that it should be able to handle.

**Safety boundary behavior:** The model complies in SOME completions \
but refuses in others — the completions are split (e.g., 4/10 comply, \
6/10 refuse). This means the prompt is right at the model's safety \
boundary, and we can investigate what specific prompt feature tips the \
balance. Note: if the model refuses 10/10 or complies 10/10, there is \
no boundary to investigate — that's just trained behavior (score 1).

**Assumption-driven framing:** The model makes an assumption about the \
user, domain, or task that shapes its response.

**Competing instructions:** Two parts of the prompt pull in different \
directions and the model must resolve the tension.

**Prior knowledge override:** The model's training knowledge overrides \
what's explicitly stated in the prompt.

## EXAMPLES OF STRONG CASES (score 4-5)

STRONG: `divide(9896, 23.6%)` causes the model to frame 9896 as "total \
revenue" (the whole) even though the math requires it to be a sub-component. \
Switching to infix notation `9896 / 0.236` flips correct framing from \
~1/10 to ~7/10. The causal factor (functional vs infix notation) is \
non-obvious, the swing is dramatic, and there's a second factor \
(percentage size) that interacts with it.

STRONG: A legitimate Canadian copyright notice gets flagged as a scam in \
7-8/10 completions. The cause is NOT the copyright content — it's the \
ISP's own consumer-protection disclaimer mentioning "settlements" and \
"financial information." Remove the disclaimer → 0/10 scam warnings. \
The causal factor is the opposite of what you'd guess.

STRONG: An unused `roundUp` parameter gets different semantics during \
refactoring. The parameter name must suggest functionality (roundUp → \
8/10 incorporation, _reserved → 0/10) AND the surrounding code context \
amplifies it dramatically (isolated function 1/10, full module 9/10). \
Two-factor interaction with clean swings on each axis.

STRONG: 3/10 completions misidentify a class-level variable as local. \
A nearby comment with imperative wording ("set X to 0") causes the \
model to think the variable is being initialized locally. Remove the \
comment → 0/10 errors. The comment's wording, not its presence, is the \
driver.

## EXAMPLES OF WEAK CASES (score 1-2)

WEAK: The model hallucinates a fact. Small model, expected.

WEAK: The model takes a single word literally. One obvious explanation.

WEAK: Token budget exhaustion causes truncation. Mechanical, not about \
the model's representations.

WEAK: The model always hedges/adds caveats. Generic RLHF behavior.

WEAK: Finding is "the model applies a generation template." If you \
could predict the behavior from the output structure alone without \
needing to look at the model's processing, it's not interesting enough.

## QUESTION FORMAT

Questions MUST be phrased as "Why did the model do X?" describing the \
observation only. Do NOT embed hypotheses in the question — those come \
from the investigation stage. The question should make someone curious \
about WHY, not just WHAT.

CRITICAL: The question will later be shown alongside a SINGLE completion \
(the first element of behavior_completion_indices), NOT all 10. Write \
the question ABOUT that specific completion. A reader will only see that \
one completion. Therefore:
- Do NOT reference the distribution across completions ("in 7/10 \
  completions", "in every completion", "in most completions").
- Do NOT quote phrases from other completions that don't appear in the \
  best exemplar.
- The question must be fully understandable and verifiable by reading \
  only the best exemplar completion.
(Use behavior_summary for distributional observations — that field IS \
allowed to discuss patterns across all 10 completions.)

- Good: "Why did the model name the valley 'Skyreach Valley' when \
  nothing in the prompt suggests that name?"
- Bad: "Why did 4 independent completions all converge on the name \
  'Skyreach Valley'?" (references cross-completion pattern)
- Good: "Why does the model add wearable applications when the \
  proposal is explicitly about communication electronics?"
- Bad: "Is the word 'flexible' triggering a biomedical association?"
  (This embeds the hypothesis — save it for the counterfactual field.)

## EXPLICITLY NOT INTERESTING — SCORE 1
- Hallucination, fabrication, factual errors (small model, expected)
- Uniform safety refusals (10/10 refuse) or uniform compliance (10/10 comply) \
  — these are trained behavior, not boundary cases. Only interesting if \
  the completions are SPLIT between refusal and compliance.
- Single obvious keyword triggers ("model did X because word Y")
- Generic sycophancy, politeness, hedging
- Capability limitations (hard math, obscure knowledge)
- Behavior easily explained by reading the prompt

## SCORING GUIDE
- 1: Routine, hallucination, obvious keyword trigger, or nothing to investigate
- 2: Slightly non-obvious but only one plausible explanation
- 3: Non-obvious choice with 2+ hypotheses, worth testing
- 4: Genuinely surprising behavior with multiple plausible explanations \
  and a clean counterfactual that could distinguish between them
- 5: Deeply puzzling coherent behavior with 3+ plausible explanations"""

SCREENING_TOOL = {
    "name": "score_completion",
    "description": "Score a model completion for interesting behavior",
    "input_schema": {
        "type": "object",
        "properties": {
            "interest_score": {
                "type": "integer",
                "description": "1-5 rating of how interesting the behavior is",
                "minimum": 1,
                "maximum": 5,
            },
            "behavior_summary": {
                "type": "string",
                "description": "1-2 sentence summary of what the model did across completions",
            },
            "question": {
                "type": "string",
                "description": (
                    "If score >= 3: a 'Why did the model do X?' question "
                    "describing the surprising behavior. Must be an observation, "
                    "NOT a hypothesis. The question must make sense when reading "
                    "ONLY the best exemplar completion (the first element of "
                    "behavior_completion_indices) — do not quote or reference "
                    "specific phrases from other completions, since a reader "
                    "will only see the exemplar. If score < 3: leave empty."
                ),
            },
            "second_person_question": {
                "type": "string",
                "description": (
                    "If score >= 3: the same question rephrased in second "
                    "person, as if asking the model directly. E.g., "
                    "'Why did you add wearable applications when the "
                    "proposal is about communication electronics?' "
                    "Same rule: must make sense when reading only the best "
                    "exemplar completion. If score < 3: leave empty."
                ),
            },
            "hypotheses": {
                "type": "string",
                "description": (
                    "If score >= 3: 2-4 plausible hypotheses for why the model "
                    "behaved this way, each predicting different outcomes under "
                    "different prompt changes. Format as numbered list. "
                    "If score < 3: leave empty."
                ),
            },
            "counterfactual": {
                "type": "string",
                "description": (
                    "If score >= 3: a concrete counterfactual experiment that "
                    "could distinguish between the hypotheses — what specific "
                    "change to make and which hypothesis it would confirm/deny. "
                    "If score < 3: leave empty."
                ),
            },
            "justification": {
                "type": "string",
                "description": (
                    "1-3 sentences explaining why you gave this score. "
                    "What makes this interesting or uninteresting? "
                    "Reference the scoring criteria."
                ),
            },
            "behavior_completion_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "Which completions (1-indexed) exhibit the described "
                    "behavior? The FIRST element must clearly and "
                    "unambiguously demonstrate the behavior (the question "
                    "will be shown alongside this completion only). "
                    "Remaining elements are other completions that also "
                    "exhibit it. E.g., if 7/10 completions refuse the "
                    "request: [5, 1, 2, 3, 6, 8, 9]."
                ),
            },
        },
        "required": [
            "interest_score",
            "behavior_summary",
            "question",
            "second_person_question",
            "hypotheses",
            "counterfactual",
            "justification",
            "behavior_completion_indices",
        ],
    },
}


@dataclass
class ScreeningResult:
    prompt_id: str
    interest_score: int
    behavior_summary: str
    question: str
    second_person_question: str
    hypotheses: str
    counterfactual: str
    justification: str
    behavior_completion_indices: list[int]
    user_message: str
    completions: list[str]
    messages: list[dict] | None = None
    n_turns: int = 1


def _build_user_message(
    user_msg: str, completions: list[str],
    messages: list[dict] | None = None,
) -> str:
    if messages and len(messages) > 1:
        # Multi-turn: show full conversation history
        parts = ["CONVERSATION HISTORY (prior turns from the original chat):\n"]
        for msg in messages[:-1]:
            role = msg["role"].upper()
            parts.append(f"[{role}]: {msg['content']}\n")
        parts.append(f"\nFINAL USER MESSAGE (completions below are responses to this):\n"
                      f"{messages[-1]['content']}\n")
    else:
        parts = [f"USER PROMPT:\n{user_msg}\n"]
    for i, comp in enumerate(completions, 1):
        parts.append(f"--- COMPLETION {i}/{len(completions)} ---\n{comp}\n")
    return "\n".join(parts)


def load_prompts(input_path: Path, n: int | None = None, offset: int | None = None,
                 top_k: int | None = None) -> list[dict]:
    data = json.loads(input_path.read_text())
    prompts = data["prompts"]

    # Filter out redacted prompts (NAME_1, NAME_2, etc. from lmsys anonymization)
    before = len(prompts)
    prompts = [p for p in prompts if "NAME_" not in p["user_message"]]
    if len(prompts) < before:
        print(f"Filtered {before - len(prompts)} redacted prompts (NAME_ placeholders)")

    # Filter for longer prompts with enough surface area for counterfactuals.
    # Use total_chars (all turns) if available, otherwise user_message length.
    before = len(prompts)
    prompts = [p for p in prompts
               if p.get("total_chars", len(p["user_message"])) >= 500]
    if len(prompts) < before:
        print(f"Filtered {before - len(prompts)} short prompts (<500 total chars)")

    if top_k is not None:
        prompts.sort(key=lambda p: len(p["user_message"]), reverse=True)
        prompts = prompts[:top_k]
        print(f"Took top {top_k} longest prompts (shortest: {len(prompts[-1]['user_message'])} chars)")
    if offset is not None:
        prompts = prompts[offset:]
    if n is not None:
        prompts = prompts[:n]
    n_completions = data["metadata"].get("n_completions_per_prompt", "?")
    print(f"Loaded {len(prompts)} prompts ({n_completions} completions each) from {input_path}")
    return prompts


async def screen_async(
    prompts: list[dict],
    output_path: Path,
    model: str,
    max_tokens: int,
    thinking_budget: int,
    concurrency: int,
):
    """Screen prompts using the normal async API."""
    print(f"\nScreening {len(prompts)} prompts with {model} "
          f"(concurrency={concurrency})...")

    client = anthropic.AsyncAnthropic()
    results: list[ScreeningResult] = []
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
                    results.append(ScreeningResult(**record))
        print(f"  Resuming: {len(completed_ids)} already completed")

    remaining = [p for p in prompts if p["id"] not in completed_ids]
    if not remaining:
        print("  All prompts already screened")
        return results

    jsonl_file = open(jsonl_path, "a")
    write_lock = asyncio.Lock()
    skipped = {"count": 0}

    async def screen_one(i: int, prompt: dict):
        try:
            user_content = _build_user_message(
                prompt["user_message"], prompt["completions"],
                messages=prompt.get("messages"),
            )
            # Append randomly sampled few-shot examples for diversity
            system = SCREENING_SYSTEM_PROMPT + _sample_few_shot_examples(5)
            resp = await async_api_call(
                client,
                model=model,
                max_tokens=max_tokens,
                thinking={"type": "enabled", "budget_tokens": thinking_budget},
                system=system,
                messages=[{"role": "user", "content": user_content}],
                tools=[SCREENING_TOOL],
            )
            tool_input = extract_tool_input(resp)

            result = ScreeningResult(
                prompt_id=prompt["id"],
                interest_score=tool_input["interest_score"],
                behavior_summary=tool_input["behavior_summary"],
                question=tool_input["question"],
                second_person_question=tool_input.get("second_person_question", ""),
                hypotheses=tool_input.get("hypotheses", ""),
                counterfactual=tool_input.get("counterfactual", ""),
                justification=tool_input.get("justification", ""),
                behavior_completion_indices=tool_input.get("behavior_completion_indices", []),
                user_message=prompt["user_message"],
                completions=prompt["completions"],
                messages=prompt.get("messages"),
                n_turns=prompt.get("n_turns", 1),
            )
            results.append(result)

            async with write_lock:
                jsonl_file.write(json.dumps(asdict(result)) + "\n")
                jsonl_file.flush()
        except (KeyError, AttributeError, TypeError) as e:
            skipped["count"] += 1
            print(f"  SKIPPING {prompt['id']}: malformed result ({e})")

    await run_concurrent(
        screen_one,
        remaining,
        concurrency=concurrency,
        label="Screening",
        progress_interval=10,
    )
    jsonl_file.close()

    if remaining:
        skip_rate = skipped["count"] / len(remaining)
        assert skip_rate < 0.001, (
            f"Screening skip rate {skip_rate:.1%} "
            f"({skipped['count']}/{len(remaining)}) exceeds 0.1% threshold"
        )

    return results


def screen_batch(
    prompts: list[dict],
    output_path: Path,
    model: str,
    max_tokens: int,
    thinking_budget: int,
    batch_chunk_size: int,
):
    """Screen prompts using the batch API."""
    print(f"\nScreening {len(prompts)} prompts with {model} (batch API)...")

    batch_state_path = output_path.with_suffix(".batch_state.json")
    jsonl_path = output_path.with_suffix(".jsonl")

    # Build batch requests
    requests = []
    for prompt in prompts:
        user_content = _build_user_message(
            prompt["user_message"], prompt["completions"],
            messages=prompt.get("messages"),
        )
        # Each batch request gets its own random few-shot examples
        system = SCREENING_SYSTEM_PROMPT + _sample_few_shot_examples(5)
        requests.append({
            "custom_id": prompt["id"],
            "params": {
                "model": model,
                "max_tokens": max_tokens,
                "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
                "system": system,
                "messages": [{"role": "user", "content": user_content}],
                "tools": [SCREENING_TOOL],
            },
        })

    batch_api_key = os.environ.get("ANTHROPIC_API_KEY_BATCH_API")
    assert batch_api_key, "ANTHROPIC_API_KEY_BATCH_API not set in environment"
    client = anthropic.Anthropic(api_key=batch_api_key)

    # Split requests into chunks to avoid 413 Payload Too Large
    chunks = [requests[i:i+batch_chunk_size] for i in range(0, len(requests), batch_chunk_size)]
    print(f"  Split into {len(chunks)} batch chunks of up to {batch_chunk_size}")

    # Load or create batch state (tracks multiple batch IDs)
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
        print(f"  Submitted chunk {chunk_idx+1}/{len(chunks)}: {batch.id} ({len(chunk)} requests)")

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

    # Retrieve results from all batches
    prompts_by_id = {p["id"]: p for p in prompts}
    results = []
    jsonl_file = open(jsonl_path, "w")
    n_skipped = 0

    for bid in batch_ids:
        for batch_result in client.messages.batches.results(bid):
            pid = batch_result.custom_id
            prompt = prompts_by_id[pid]

            if batch_result.result.type != "succeeded":
                n_skipped += 1
                print(f"  SKIPPING {pid}: batch result {batch_result.result.type}")
                continue

            try:
                tool_input = None
                for block in batch_result.result.message.content:
                    if block.type == "tool_use":
                        tool_input = block.input
                        break

                if tool_input is None:
                    print(f"  SKIPPING {pid}: no tool_use in response")
                    continue

                result = ScreeningResult(
                    prompt_id=pid,
                    interest_score=tool_input["interest_score"],
                    behavior_summary=tool_input["behavior_summary"],
                    question=tool_input["question"],
                    second_person_question=tool_input.get("second_person_question", ""),
                    hypotheses=tool_input.get("hypotheses", ""),
                    counterfactual=tool_input.get("counterfactual", ""),
                    justification=tool_input.get("justification", ""),
                    behavior_completion_indices=tool_input.get("behavior_completion_indices", []),
                    user_message=prompt["user_message"],
                    completions=prompt["completions"],
                    messages=prompt.get("messages"),
                    n_turns=prompt.get("n_turns", 1),
                )
                results.append(result)
                jsonl_file.write(json.dumps(asdict(result)) + "\n")
            except (KeyError, AttributeError, TypeError) as e:
                n_skipped += 1
                print(f"  SKIPPING {pid}: malformed result ({e})")
                continue

    jsonl_file.close()

    if prompts:
        skip_rate = n_skipped / len(prompts)
        assert skip_rate < 0.001, (
            f"Screening batch skip rate {skip_rate:.1%} "
            f"({n_skipped}/{len(prompts)}) exceeds 0.1% threshold"
        )

    return results


def save_results(results: list[ScreeningResult], output_path: Path, input_path: Path, model: str):
    """Save final scored results as JSON, sorted by interest score."""
    results.sort(key=lambda r: r.interest_score, reverse=True)

    output = {
        "metadata": {
            "screening_model": model,
            "input_file": str(input_path),
            "n_screened": len(results),
            "score_distribution": {
                str(s): sum(1 for r in results if r.interest_score == s)
                for s in range(1, 6)
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "system_prompt": SCREENING_SYSTEM_PROMPT,
            "tool_schema": SCREENING_TOOL,
            "few_shot_pool_size": len(FEW_SHOT_POOL),
            "few_shot_per_call": 5,
        },
        "results": [asdict(r) for r in results],
    }

    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(results)} results to {output_path}")

    # Print summary
    print(f"\n--- Score Distribution ---")
    for score in range(5, 0, -1):
        count = sum(1 for r in results if r.interest_score == score)
        print(f"  Score {score}: {count}")

    # Dump top results to readable text file
    top = [r for r in results if r.interest_score >= 4]
    if top:
        dump_path = output_path.with_name(
            output_path.stem.replace("screening", "top_results") + ".txt"
        )
        _dump_top_results(top, dump_path)


def _dump_top_results(results: list[ScreeningResult], dump_path: Path):
    """Dump top-scoring results to a human-readable text file."""
    lines = []
    lines.append(f"TOP SCREENING RESULTS (score >= 4)")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Total: {len(results)} results")
    lines.append("=" * 80)

    for r in results:
        lines.append("")
        lines.append(f"{'=' * 80}")
        lines.append(f"[{r.prompt_id}] SCORE: {r.interest_score}")
        lines.append(f"{'=' * 80}")
        lines.append("")
        lines.append(f"--- USER PROMPT ---")
        lines.append(r.user_message)
        lines.append("")
        for i, comp in enumerate(r.completions, 1):
            lines.append(f"--- COMPLETION {i}/{len(r.completions)} ---")
            lines.append(comp)
            lines.append("")
        lines.append(f"--- SONNET'S ANALYSIS ---")
        lines.append(f"Behavior: {r.behavior_summary}")
        lines.append("")
        if r.question:
            lines.append(f"Question: {r.question}")
            lines.append("")
        if r.hypotheses:
            lines.append(f"Hypotheses: {r.hypotheses}")
            lines.append("")
        if r.counterfactual:
            lines.append(f"Counterfactual: {r.counterfactual}")
            lines.append("")

    dump_path.write_text("\n".join(lines))
    print(f"Dumped {len(results)} top results to {dump_path}")


def main():
    parser = argparse.ArgumentParser(description="Screen completions for interesting behaviors")
    parser.add_argument("--input", type=Path, required=True, help="Path to completions JSON")
    parser.add_argument("--model", type=str, required=True, help="Anthropic model for screening")
    parser.add_argument("--max-tokens", type=int, required=True, help="Max response tokens")
    parser.add_argument("--thinking-budget", type=int, required=True, help="Extended thinking budget")
    parser.add_argument("--n", type=int, default=None, help="Limit to first N prompts (after offset)")
    parser.add_argument("--offset", type=int, default=None, help="Skip first N prompts (after filtering)")
    parser.add_argument("--top-k", type=int, default=None, help="Take the K longest prompts (by char count)")
    parser.add_argument("--batch", action="store_true", help="Use batch API instead of async")
    parser.add_argument("--concurrency", type=int, default=None, help="Max concurrent API calls (required for async mode)")
    parser.add_argument("--batch-chunk-size", type=int, default=None, help="Requests per batch chunk (required for batch mode)")
    args = parser.parse_args()

    prompts = load_prompts(args.input, n=args.n, offset=args.offset,
                           top_k=args.top_k)

    # Output naming: screening_<n_screened>_from_<input_stem>.json
    output_dir = args.input.parent
    n_label = len(prompts)
    output_path = output_dir / f"screening_{n_label}_from_{args.input.stem}.json"

    if args.batch:
        assert args.batch_chunk_size is not None, "--batch-chunk-size is required for batch mode"
        results = screen_batch(
            prompts, output_path,
            model=args.model,
            max_tokens=args.max_tokens,
            thinking_budget=args.thinking_budget,
            batch_chunk_size=args.batch_chunk_size,
        )
    else:
        assert args.concurrency is not None, "--concurrency is required for async mode"
        results = asyncio.run(screen_async(
            prompts, output_path,
            model=args.model,
            max_tokens=args.max_tokens,
            thinking_budget=args.thinking_budget,
            concurrency=args.concurrency,
        ))

    save_results(results, output_path, args.input, model=args.model)


if __name__ == "__main__":
    main()
