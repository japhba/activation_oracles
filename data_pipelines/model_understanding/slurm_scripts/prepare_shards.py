"""
Prepare prompt shards for parallel completion generation.

Streams WildChat, applies all filters and tokenization, splits eligible
prompts into N shard files. Each shard contains pre-tokenized prompts
ready for GPU generation (no dataset streaming needed on the GPU node).

Supports both single-turn and multi-turn conversations, filtered by
total character count.

Usage:
    .venv/bin/python data_pipelines/model_understanding/slurm_scripts/prepare_shards.py \
        --model Qwen/Qwen3-14B --n-prompts 100 --n-shards 2 \
        --min-chars 500 --max-chars 5000 --max-turns 5 \
        --output-dir data_pipelines/model_understanding/runs/test_mixed/shards
"""

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from data_pipelines.pipeline_utils import model_dir_name

CHAT_DATASET = "allenai/WildChat-1M"
MAX_PROMPT_TOKENS = 3000
SEED = 42


def content_hash(messages: list[dict]) -> str:
    """Deterministic hash of a prompt's messages array."""
    content = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def load_exclude_hashes(exclude_files: list[str]) -> tuple[set[str], set[str]]:
    """Load conversation_hashes and content hashes to exclude.

    Returns (conversation_hashes, content_hashes). Content hashes catch
    WildChat duplicates that share identical messages but have different
    conversation_hash values.
    """
    conv_hashes = set()
    content_hashes = set()
    for path in exclude_files:
        data = json.loads(Path(path).read_text())
        for p in data["prompts"]:
            h = p.get("conversation_hash", "")
            if h:
                conv_hashes.add(h)
            msgs = p.get("messages")
            if msgs:
                content_hashes.add(content_hash(msgs))
    print(f"Loaded {len(conv_hashes)} conversation_hashes and "
          f"{len(content_hashes)} content hashes to exclude "
          f"from {len(exclude_files)} file(s)")
    return conv_hashes, content_hashes


def stream_prompts(
    min_chars: int,
    max_chars: int,
    max_turns: int,
    max_scan: int,
    exclude_conv_hashes: set[str] | None = None,
    exclude_content_hashes: set[str] | None = None,
):
    """Yield eligible prompts from WildChat (single + multi-turn)."""
    ds = iter(load_dataset(CHAT_DATASET, split="train", streaming=True))
    n_excluded_conv = 0
    n_excluded_content = 0

    for i, row in enumerate(ds):
        if i >= max_scan:
            break

        if row.get("language") != "English":
            continue
        if row.get("redacted", False):
            continue

        conv = row["conversation"]
        if not conv or conv[0]["role"] != "user":
            continue

        # Check conversation_hash exclusion early (before expensive processing)
        conv_hash = row.get("conversation_hash", "")
        if exclude_conv_hashes and conv_hash in exclude_conv_hashes:
            n_excluded_conv += 1
            continue

        # Build messages, ensure ends with user turn
        messages = [{"role": m["role"], "content": m["content"]} for m in conv]
        while messages and messages[-1]["role"] != "user":
            messages.pop()
        if not messages:
            continue

        # Filter redacted placeholders
        full_text = " ".join(m["content"] for m in messages)
        if "NAME_" in full_text:
            continue

        total_chars = sum(len(m["content"]) for m in messages)
        if total_chars < min_chars or total_chars > max_chars:
            continue

        n_user = sum(1 for m in messages if m["role"] == "user")
        if n_user > max_turns:
            continue

        # Check content hash exclusion (catches WildChat dupes with different
        # conversation_hash but identical messages)
        if exclude_content_hashes and content_hash(messages) in exclude_content_hashes:
            n_excluded_content += 1
            continue

        yield {
            "conversation_hash": conv_hash,
            "messages": messages,
            "user_message": messages[-1]["content"],
            "n_turns": n_user,
            "total_chars": total_chars,
        }

    if exclude_conv_hashes or exclude_content_hashes:
        print(f"Excluded {n_excluded_conv} by conversation_hash, "
              f"{n_excluded_content} by content hash")


def prepare_shards(
    model_name: str,
    n_prompts: int,
    n_shards: int,
    output_dir: str,
    min_chars: int,
    max_chars: int,
    max_turns: int,
    max_scan: int,
    exclude_files: list[str] | None = None,
):
    random.seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    supports_thinking = (
        hasattr(tokenizer, "chat_template")
        and tokenizer.chat_template is not None
        and "enable_thinking" in tokenizer.chat_template
    )

    template_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": None,
        "padding": False,
    }
    if supports_thinking:
        template_kwargs["enable_thinking"] = False

    # Load exclusion sets
    exclude_conv_hashes = None
    exclude_content_hashes = None
    if exclude_files:
        exclude_conv_hashes, exclude_content_hashes = load_exclude_hashes(exclude_files)

    print(f"Streaming {CHAT_DATASET} (scan up to {max_scan} rows)...")
    print(f"Filters: {min_chars}-{max_chars} total chars, <= {max_turns} turns")

    # Collect all eligible prompts
    all_candidates = []
    for prompt_data in tqdm(
        stream_prompts(
            min_chars, max_chars, max_turns, max_scan,
            exclude_conv_hashes, exclude_content_hashes,
        ),
        desc="Collecting prompts",
    ):
        prompt_ids = tokenizer.apply_chat_template(
            prompt_data["messages"], **template_kwargs,
        )

        if len(prompt_ids) > MAX_PROMPT_TOKENS or len(prompt_ids) == 0:
            continue

        all_candidates.append({
            **prompt_data,
            "prompt_ids": prompt_ids,
        })

    print(f"Found {len(all_candidates)} eligible prompts")

    # Sample from natural distribution
    n_sample = min(n_prompts, len(all_candidates))
    selected = random.sample(all_candidates, n_sample)

    # Assign IDs
    for i, c in enumerate(selected):
        c["id"] = f"prompt_{i:05d}"

    from collections import Counter
    turn_dist = Counter(c["n_turns"] for c in selected)
    print(f"Selected {n_sample} prompts:")
    for k in sorted(turn_dist):
        print(f"  {k}-turn: {turn_dist[k]}")

    # Split into shards
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    shard_size = len(selected) // n_shards
    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end = start + shard_size if shard_idx < n_shards - 1 else len(selected)
        shard_prompts = selected[start:end]

        shard_data = {
            "metadata": {
                "model": model_name,
                "dataset": CHAT_DATASET,
                "shard_index": shard_idx,
                "n_shards": n_shards,
                "n_prompts": len(shard_prompts),
                "total_prompts": len(selected),
                "max_prompt_tokens": MAX_PROMPT_TOKENS,
                "min_chars": min_chars,
                "max_chars": max_chars,
                "max_turns": max_turns,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "prompts": shard_prompts,
        }

        shard_path = out / f"shard_{shard_idx}.json"
        shard_path.write_text(json.dumps(shard_data, ensure_ascii=False))
        print(f"Shard {shard_idx}: {len(shard_prompts)} prompts -> {shard_path}")

    print(f"\nDone. {n_shards} shards in {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare prompt shards for parallel generation",
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n-prompts", type=int, required=True)
    parser.add_argument("--n-shards", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--min-chars", type=int, required=True)
    parser.add_argument("--max-chars", type=int, required=True)
    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--max-scan", type=int, default=50000)
    parser.add_argument(
        "--exclude", nargs="+", default=None,
        help="Completions JSON file(s) from previous runs to deduplicate against",
    )
    args = parser.parse_args()
    prepare_shards(
        args.model, args.n_prompts, args.n_shards, args.output_dir,
        args.min_chars, args.max_chars, args.max_turns, args.max_scan,
        exclude_files=args.exclude,
    )
