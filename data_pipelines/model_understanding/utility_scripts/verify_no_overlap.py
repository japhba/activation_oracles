"""
One-off script to verify zero overlap between two sets of prompts.

Checks both conversation_hash and content hash to catch any leakage.

Usage:
    .venv/bin/python data_pipelines/model_understanding/verify_no_overlap.py \
        --run1 data_pipelines/model_understanding/runs/qwen3_32b_run/completions.json \
        --run2-shards data_pipelines/model_understanding/runs/qwen3_32b_run_2/shards/
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def content_hash(messages: list[dict]) -> str:
    """Deterministic hash of a prompt's messages array."""
    content = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def load_prompts_from_completions(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text())
    return data["prompts"]


def load_prompts_from_shards(shard_dir: str) -> list[dict]:
    prompts = []
    for shard_path in sorted(Path(shard_dir).glob("shard_*.json")):
        data = json.loads(shard_path.read_text())
        prompts.extend(data["prompts"])
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Verify no overlap between runs")
    parser.add_argument("--run1", type=str, required=True,
                        help="completions.json from first run")
    parser.add_argument("--run2-shards", type=str, default=None,
                        help="Shards directory from second run")
    parser.add_argument("--run2", type=str, default=None,
                        help="completions.json from second run")
    args = parser.parse_args()

    if not args.run2_shards and not args.run2:
        parser.error("Provide --run2-shards or --run2")

    # Load run 1
    print(f"Loading run 1: {args.run1}")
    r1 = load_prompts_from_completions(args.run1)
    print(f"  {len(r1)} prompts")

    # Load run 2
    if args.run2:
        print(f"Loading run 2: {args.run2}")
        r2 = load_prompts_from_completions(args.run2)
    else:
        print(f"Loading run 2 shards: {args.run2_shards}")
        r2 = load_prompts_from_shards(args.run2_shards)
    print(f"  {len(r2)} prompts")

    # Check conversation_hash overlap
    r1_conv_hashes = {p.get("conversation_hash", "") for p in r1} - {""}
    r2_conv_hashes = {p.get("conversation_hash", "") for p in r2} - {""}

    conv_overlap = r1_conv_hashes & r2_conv_hashes
    print(f"\nConversation hash overlap: {len(conv_overlap)}")
    if conv_overlap:
        print(f"  Overlapping hashes: {list(conv_overlap)[:10]}")

    # Check content hash overlap
    r1_content = {content_hash(p["messages"]) for p in r1 if p.get("messages")}
    r2_content = {content_hash(p["messages"]) for p in r2 if p.get("messages")}

    content_overlap = r1_content & r2_content
    print(f"Content hash overlap: {len(content_overlap)}")
    if content_overlap:
        print(f"  Overlapping content hashes: {list(content_overlap)[:10]}")

    # Check user_message overlap (less strict, just informational)
    r1_msgs = {p["user_message"] for p in r1}
    r2_msgs = {p["user_message"] for p in r2}
    msg_overlap = r1_msgs & r2_msgs
    print(f"User message overlap: {len(msg_overlap)}")
    if msg_overlap:
        examples = list(msg_overlap)[:3]
        for m in examples:
            print(f"  {m[:100]}...")

    # Verdict
    print(f"\n{'='*60}")
    if conv_overlap or content_overlap:
        print("FAIL: Overlap detected!")
        sys.exit(1)
    else:
        print("PASS: No overlap detected.")
        if msg_overlap:
            print(f"  Note: {len(msg_overlap)} shared user_messages, but different "
                  f"conversations (different message histories).")


if __name__ == "__main__":
    main()
