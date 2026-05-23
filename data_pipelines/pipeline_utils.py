"""
Shared utilities for dataset generation pipelines.

Provides:
- model_dir_name: Convert HF model ID to a directory-safe short name
- async_api_call: Anthropic API calls with retry + high-priority key fallback
- parse_json_response: Extract JSON from LLM responses (handles fences, trailing commas, etc.)
- extract_tool_input: Extract structured output from tool_use responses
- run_concurrent: Run async tasks with a concurrency limit and progress tracking
"""

import argparse
import asyncio
import functools
import json
import os
import re
from pathlib import Path

import anthropic
from anthropic._exceptions import APIStatusError, APIConnectionError, InternalServerError, OverloadedError, RateLimitError

# Force unbuffered output so we can monitor progress in background
print = functools.partial(print, flush=True)


def model_dir_name(model_id: str) -> str:
    """Convert a HuggingFace model ID to a directory-safe short name.

    Examples:
        "Qwen/Qwen3-8B" -> "Qwen3-8B"
        "meta-llama/Llama-3-8B" -> "Llama-3-8B"
        "Qwen3-8B" -> "Qwen3-8B"
    """
    return model_id.split("/")[-1]


def add_model_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add a required --model argument to an argparse parser."""
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g. Qwen/Qwen3-8B)",
    )
    return parser


def load_dotenv():
    """Load .env from repo root if API keys aren't already set."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    # Walk up from this file to find .env
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        key, _, value = line.partition("=")
        if key and value:
            # Strip inline comments (e.g. KEY=value # comment)
            if " #" in value:
                value = value[:value.index(" #")]
            os.environ.setdefault(key.strip(), value.strip())


load_dotenv()

async def async_api_call(client: anthropic.AsyncAnthropic, *, max_retries: int = 10, **kwargs):
    """Make an Anthropic API call with retry and backoff.

    On transient errors (429/529/500/connection), retries with linear backoff.
    Does NOT manage concurrency — wrap calls in a semaphore at the task level.
    """
    for attempt in range(max_retries):
        try:
            return await client.messages.create(**kwargs)
        except (OverloadedError, RateLimitError, InternalServerError, APIConnectionError) as e:
            error_code = getattr(e, 'status_code', type(e).__name__)
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                print(f"    [{error_code} retry {attempt+1}/{max_retries} → waiting {wait}s]")
                await asyncio.sleep(wait)
            else:
                raise


def parse_json_response(text: str):
    """Extract JSON from an LLM response, handling markdown fences and common issues."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
        text = text.strip()

    # Fix common LLM JSON issues
    cleaned = re.sub(r',\s*([}\]])', r'\1', text)  # trailing commas
    cleaned = re.sub(r':\s*\'([^\']*)\'\s*([,}\]])', r': "\1"\2', cleaned)  # single quotes to double
    # Fix unescaped backslashes (e.g. \boxed, \text) — escape any \ not followed by valid JSON escape chars
    cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Aggressive backslash escaping: replace ALL unescaped backslashes
    # (the regex above misses some edge cases with consecutive backslashes)
    aggressive = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu\\])', r'\\\\', cleaned)
    try:
        return json.loads(aggressive)
    except json.JSONDecodeError:
        pass
    # Scan for JSON array or object and use raw_decode to stop at the end
    decoder = json.JSONDecoder()
    for attempt_text in (aggressive, cleaned):
        for start_char in ("[", "{"):
            idx = attempt_text.find(start_char)
            if idx != -1:
                try:
                    obj, _ = decoder.raw_decode(attempt_text, idx)
                    return obj
                except json.JSONDecodeError:
                    continue
    raise json.JSONDecodeError("No JSON found", text, 0)


def extract_tool_input(resp) -> dict:
    """Extract tool use input from a response that used tool_choice.

    Use this when you force a tool call via tool_choice={"type": "tool", "name": "..."}.
    The response is guaranteed to be valid JSON (no parse_json_response needed).
    """
    for block in resp.content:
        if block.type == "tool_use":
            return block.input
    raise ValueError("No tool_use block in response")


async def run_concurrent(tasks_fn, items, *, concurrency, label="", progress_interval=10):
    """Run async tasks with a concurrency limit and progress tracking.

    Args:
        tasks_fn: async function(semaphore, i, item) -> None. Must handle its own errors.
        items: list of items to process
        concurrency: max concurrent tasks
        label: name for progress prints
        progress_interval: print progress every N completions
    """
    semaphore = asyncio.Semaphore(concurrency)
    progress = {"done": 0}
    total = len(items)

    async def wrapped(i, item):
        async with semaphore:
            await tasks_fn(i, item)
        progress["done"] += 1
        d = progress["done"]
        if d % progress_interval == 0 or d == total or d <= 3:
            print(f"  {label} progress: {d}/{total}")

    await asyncio.gather(*[wrapped(i, item) for i, item in enumerate(items)])
