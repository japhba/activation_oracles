"""
One-off script to backfill conversation_hash into an existing completions.json.

Replays the exact same WildChat streaming + tokenization + random.sample that
prepare_shards.py used, recovering the conversation_hash for each prompt.
Verifies correctness by checking all user_messages match, then writes the
updated file in-place.

Usage:
    .venv/bin/python data_pipelines/model_understanding/backfill_conversation_hashes.py \
        --completions data_pipelines/model_understanding/runs/qwen3_32b_run/completions.json \
        --model Qwen/Qwen3-32B \
        --min-chars 500 --max-chars 5000 --max-turns 5 --max-scan 500000
"""

import argparse
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

from data_pipelines.model_understanding.slurm_scripts.prepare_shards import MAX_PROMPT_TOKENS, SEED, stream_prompts


def backfill(
    completions_path: str,
    model_name: str,
    min_chars: int,
    max_chars: int,
    max_turns: int,
    max_scan: int,
):
    # Load existing completions
    comp_path = Path(completions_path)
    data = json.loads(comp_path.read_text())
    prompts = data["prompts"]
    n_prompts = len(prompts)
    print(f"Loaded {n_prompts} prompts from {comp_path}")

    already = sum(1 for p in prompts if p.get("conversation_hash"))
    if already == n_prompts:
        print("All prompts already have conversation_hash, nothing to do.")
        return

    # Replay the exact same pipeline: stream → tokenize → filter → sample
    print(f"Loading tokenizer for {model_name}...")
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

    print(f"Streaming WildChat (max_scan={max_scan})...")
    all_candidates = []
    for prompt_data in tqdm(
        stream_prompts(min_chars, max_chars, max_turns, max_scan),
        desc="Collecting prompts",
    ):
        prompt_ids = tokenizer.apply_chat_template(
            prompt_data["messages"], **template_kwargs,
        )
        if len(prompt_ids) > MAX_PROMPT_TOKENS or len(prompt_ids) == 0:
            continue
        all_candidates.append(prompt_data)

    print(f"Found {len(all_candidates)} eligible candidates")

    # Same seed + sample as the original run
    random.seed(SEED)
    n_sample = min(n_prompts, len(all_candidates))
    selected = random.sample(all_candidates, n_sample)

    if n_sample != n_prompts:
        print(f"WARNING: selected {n_sample} but completions has {n_prompts}")

    # Assign IDs (same as prepare_shards)
    for i, c in enumerate(selected):
        c["id"] = f"prompt_{i:05d}"

    # Verify by comparing user_messages
    print("Verifying user_message match...")
    mismatches = 0
    for i, (sel, orig) in enumerate(zip(selected, prompts)):
        if sel["user_message"] != orig["user_message"]:
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH at {i}: replay={sel['user_message'][:80]!r}")
                print(f"               orig ={orig['user_message'][:80]!r}")

    if mismatches > 0:
        print(f"\nFATAL: {mismatches}/{n_prompts} user_messages don't match.")
        print("The WildChat dataset may have changed. Aborting.")
        sys.exit(1)

    print(f"All {n_prompts} user_messages match!")

    # Insert conversation_hash
    filled = 0
    for sel, orig in zip(selected, prompts):
        h = sel.get("conversation_hash", "")
        if h:
            orig["conversation_hash"] = h
            filled += 1

    print(f"Backfilled {filled}/{n_prompts} conversation_hashes")

    empty = sum(1 for p in prompts if not p.get("conversation_hash"))
    if empty:
        print(f"WARNING: {empty} prompts still have no conversation_hash (missing in WildChat)")

    # Write back
    comp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Saved updated completions to {comp_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill conversation_hash into existing completions.json",
    )
    parser.add_argument("--completions", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--min-chars", type=int, required=True)
    parser.add_argument("--max-chars", type=int, required=True)
    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--max-scan", type=int, required=True)
    args = parser.parse_args()
    backfill(
        args.completions, args.model,
        args.min_chars, args.max_chars, args.max_turns, args.max_scan,
    )
