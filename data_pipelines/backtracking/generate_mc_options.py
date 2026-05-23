"""
Generate multiple-choice distractors for the backtracking eval dataset.

Takes the existing backtracking eval dataset and uses Opus to generate 3
plausible-but-wrong alternative uncertainty descriptions for each entry.
Distractors are about the *same problem* so the AO can't just pick the
most specific-sounding option.

Usage:
    source .env && .venv/bin/python data_pipelines/backtracking/generate_mc_options.py --model Qwen/Qwen3-8B
"""

import asyncio
import json
import random
from pathlib import Path
from typing import Any

import anthropic
from tqdm import tqdm

from data_pipelines.pipeline_utils import add_model_arg, model_dir_name

GENERATOR_MODEL = "claude-opus-4-6"

SYSTEM_PROMPT = """\
You are generating multiple-choice options for an evaluation about identifying what a \
reasoning model is uncertain about at a specific point in its chain-of-thought.

You will be given:
1. The problem the model is solving
2. The model's chain-of-thought PREFIX (up to the point of uncertainty)
3. The CORRECT description of what the model is actually uncertain about (this may include \
distribution percentages and mention multiple sub-reasons — ignore all of that)

Generate exactly 4 options as a JSON object with keys "correct" and "distractors":

"correct": A simplified rewrite of the correct uncertainty description. Focus ONLY on the \
single most dominant/likely reason. Remove all distribution percentages, continuation counts, \
and secondary reasons. Write it as a clean 1-2 sentence description.

"distractors": An array of 3 plausible-but-wrong alternative uncertainty descriptions. Each must:
- Be about a plausible uncertainty the model COULD have at this point in this specific problem
- Be wrong — it should NOT describe what the model is actually uncertain about
- Be similarly specific, detailed, and stylistically similar to the rewritten correct answer
- Reference concrete aspects of the problem or reasoning (not generic like "the model is confused")
- Be distinct from each other

Respond with ONLY a JSON object, e.g.:
{"correct": "rewritten correct description", "distractors": ["distractor 1", "distractor 2", "distractor 3"]}"""

USER_TEMPLATE = """\
PROBLEM:
{problem}

PREFIX (model's reasoning up to the uncertainty point):
{prefix}

CORRECT UNCERTAINTY DESCRIPTION:
{correct}"""


async def generate_mc_options_for_entry(
    client: anthropic.AsyncAnthropic,
    entry: dict,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Generate simplified correct option + 3 distractors for a single entry."""
    user_message = USER_TEMPLATE.format(
        problem=entry["problem"],
        # Truncate prefix to keep prompt reasonable — last 3000 chars
        # should capture the reasoning context near the backtracking point
        prefix=entry["prefix"][-3000:],
        correct=entry["uncertainty_description"],
    )

    async with semaphore:
        response = await client.messages.create(
            model=GENERATOR_MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

    text = response.content[0].text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
        text = text.strip()
    # Extract JSON object from anywhere in the response
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    text = text[start:end + 1]
    parsed = json.loads(text)
    assert isinstance(parsed, dict), f"Expected JSON object, got {type(parsed)}"
    assert "correct" in parsed and "distractors" in parsed
    assert isinstance(parsed["correct"], str)
    assert isinstance(parsed["distractors"], list) and len(parsed["distractors"]) == 3, (
        f"Expected 3 distractors, got {len(parsed.get('distractors', []))}"
    )
    assert all(isinstance(d, str) for d in parsed["distractors"])
    return parsed


def build_mc_entry(entry: dict, mc_result: dict[str, Any]) -> dict:
    """Add MC options to an entry with shuffled order."""
    correct = mc_result["correct"]
    options = [correct] + mc_result["distractors"]
    random.shuffle(options)
    correct_index = options.index(correct)

    return {
        **entry,
        "mc_options": options,
        "mc_correct_index": correct_index,
        "mc_correct_label": chr(ord("A") + correct_index),
    }


async def main(
    dataset_path: Path,
    max_entries: int | None = None,
    concurrency: int = 10,
    max_retries: int = 3,
):
    data = json.loads(dataset_path.read_text())
    all_entries = data["entries"]
    entries = all_entries if max_entries is None else all_entries[:max_entries]

    # Skip entries that already have MC options
    todo_indices = [i for i, e in enumerate(entries) if "mc_options" not in e]
    if not todo_indices:
        print("All entries already have MC options.")
        return

    print(f"Generating MC options for {len(todo_indices)}/{len(entries)} entries "
          f"({len(entries) - len(todo_indices)} already done) "
          f"using {GENERATOR_MODEL} (concurrency={concurrency})...")

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)

    remaining = list(todo_indices)
    for attempt in range(1, max_retries + 1):
        if not remaining:
            break

        pbar = tqdm(total=len(remaining), desc=f"Attempt {attempt}/{max_retries}")

        async def _generate(idx: int):
            result = await generate_mc_options_for_entry(client, entries[idx], semaphore)
            pbar.update(1)
            return idx, result

        tasks = [_generate(idx) for idx in remaining]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        pbar.close()

        failed = []
        for r in results:
            if isinstance(r, BaseException):
                # Can't recover the index from the exception, handle below
                continue
            idx, mc_result = r
            entries[idx] = build_mc_entry(entries[idx], mc_result)

        # Check which are still missing
        remaining = [i for i in remaining if "mc_options" not in entries[i]]
        if remaining:
            print(f"  {len(remaining)} failures, retrying...")

    if remaining:
        print(f"WARNING: {len(remaining)} entries still missing MC options after {max_retries} attempts")
        for i in remaining:
            print(f"  - index {i}: {entries[i].get('problem_id')}")

    done = sum(1 for e in entries if "mc_options" in e)
    print(f"\n{done}/{len(entries)} entries have MC options")

    # Print a few examples
    mc_done = [e for e in entries if "mc_options" in e]
    for i, mc_entry in enumerate(mc_done[:3]):
        print(f"\n{'='*60}")
        print(f"Entry: {mc_entry['problem_id']}")
        print(f"Problem: {mc_entry['problem'][:100]}...")
        print(f"Correct: {mc_entry['mc_correct_label']}")
        for j, opt in enumerate(mc_entry["mc_options"]):
            label = chr(ord("A") + j)
            marker = " <-- CORRECT" if j == mc_entry["mc_correct_index"] else ""
            print(f"  {label}. {opt}{marker}")

    # Write back in-place — update metadata, preserve all entries
    data["metadata"]["mc_generator_model"] = GENERATOR_MODEL
    data["metadata"]["mc_version"] = 1
    data["entries"] = entries
    dataset_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\nSaved to {dataset_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    add_model_arg(parser)
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    dataset_path = Path(f"data_pipelines/backtracking/{model_dir_name(args.model)}/backtracking_eval_dataset.json")
    asyncio.run(main(dataset_path=dataset_path, max_entries=args.max_entries, concurrency=args.concurrency))
