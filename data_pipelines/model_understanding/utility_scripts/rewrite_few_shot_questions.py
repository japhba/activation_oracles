"""Rewrite few-shot questions to remove distributional language.

Questions should be about a single completion, not about the distribution
across completions. Uses Opus to rewrite each question.
"""

import asyncio
import json
from pathlib import Path

from data_pipelines.pipeline_utils import load_dotenv

load_dotenv()

import anthropic

INPUT_PATH = Path(__file__).resolve().parent / "few_shot_question_pool.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "few_shot_question_pool_v2.json"

SYSTEM_PROMPT = """\
You are rewriting screening questions for a model behavior analysis pipeline.

The original questions were written to describe behavior across multiple \
completions (e.g., "Why does the model consistently do X in 10/10 completions?"). \
We need to rewrite them so they describe behavior visible in a SINGLE completion, \
because the question will later be shown alongside only one completion.

Rules:
- Remove all distributional language: "consistently", "in X/10 completions", \
  "in every completion", "in most completions", "across completions", \
  "all 10 completions", "different completions assign", etc.
- Keep the core behavioral observation intact — just reframe it as something \
  visible in one response.
- Keep the same level of specificity and detail.
- Do NOT add hedging like "in this completion" — just state what the model did.
- The question should still start with "Why does the model..." or similar.

Examples:
  BEFORE: "Why does the model consistently avoid the word 'richer' in all 10 \
  completions when simplifying to Level 1?"
  AFTER: "Why does the model avoid the word 'richer' when simplifying to \
  Level 1, even though 'richer' is a simple word?"

  BEFORE: "Why does the model translate the German comment string to English \
  in 10/10 completions when the task explicitly asks only to 'rename identifiers'?"
  AFTER: "Why does the model translate the German comment string to English \
  when the task explicitly asks only to 'rename identifiers' (and a comment \
  string is not an identifier)?"

  BEFORE: "Why do different completions assign incompatible semantics to the \
  roundUp parameter?"
  AFTER: "Why does the model reinterpret the unused roundUp parameter during \
  refactoring, assigning it semantics that weren't in the original code?"

Respond with ONLY the rewritten question, nothing else."""

MAX_CONCURRENCY = 5
MAX_RETRIES = 3


async def rewrite_one(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    entry: dict,
    index: int,
) -> dict:
    async with semaphore:
        question = entry["question"]
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=512,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": f"Rewrite this question:\n\n{question}"}],
                )
                break
            except anthropic.RateLimitError:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    raise
        new_question = response.content[0].text.strip()
        # Strip surrounding quotes if Opus added them
        if new_question.startswith('"') and new_question.endswith('"'):
            new_question = new_question[1:-1]
        result = dict(entry)
        result["question"] = new_question
        result["original_question"] = question
        if (index + 1) % 10 == 0:
            print(f"  Rewrote {index + 1} questions...")
        return result


async def main():
    pool = json.loads(INPUT_PATH.read_text())
    print(f"Loaded {len(pool)} questions from {INPUT_PATH}")

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    tasks = [
        rewrite_one(client, semaphore, entry, i)
        for i, entry in enumerate(pool)
    ]
    results = await asyncio.gather(*tasks)

    # Show some examples
    changed = 0
    for r in results:
        if r["question"] != r["original_question"]:
            changed += 1
    print(f"\n{changed}/{len(results)} questions were modified")

    print("\nSample rewrites:")
    for r in results[:5]:
        if r["question"] != r["original_question"]:
            print(f"  BEFORE: {r['original_question'][:120]}")
            print(f"  AFTER:  {r['question'][:120]}")
            print()

    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Wrote {len(results)} questions to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
