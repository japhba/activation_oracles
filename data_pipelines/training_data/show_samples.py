"""
Show a sample of training data entries for review.

Prints:
1. The system prompt templates (past internal-state, past factual, and future)
2. 10 sample entries (5 past, 5 future) with their:
   - Generator user prompt (what was sent to the LLM)
   - Resulting question/answer pair

Usage:
    .venv/bin/python data_pipelines/training_data/show_samples.py [seed] [data_file]
"""

import json
import random
import sys


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    path = sys.argv[2] if len(sys.argv) > 2 else "data_pipelines/training_data/artifacts/training_data_500.json"

    with open(path) as f:
        data = json.load(f)

    entries = data["entries"]

    # Collect one system prompt per (qa_type, response_format) combo
    seen_system_prompts = {}
    for e in entries:
        key = (e["qa_type"], e["response_format"])
        if key not in seen_system_prompts:
            seen_system_prompts[key] = e["generator_system_prompt"]

    # Print representative system prompts
    print("=" * 100)
    print("SYSTEM PROMPT: FUTURE (about what the model will do next)")
    print("=" * 100)
    print(seen_system_prompts.get(("future", "open_ended"), "NOT FOUND"))
    print()
    print("=" * 100)
    print("SYSTEM PROMPT: PAST (about what the model has already processed)")
    print("=" * 100)
    print(seen_system_prompts.get(("past", "open_ended"), "NOT FOUND"))
    print()

    # Sample 5 past and 5 future
    past = [e for e in entries if e["qa_type"] == "past"]
    future = [e for e in entries if e["qa_type"] == "future"]

    rng = random.Random(seed)
    sample = rng.sample(past, min(5, len(past))) + rng.sample(future, min(5, len(future)))
    rng.shuffle(sample)

    for i, entry in enumerate(sample, 1):
        past_mode = entry.get("past_mode", "")
        mode_str = f"  |  mode={past_mode}" if past_mode else ""
        print("=" * 100)
        print(f"ENTRY {i}/{len(sample)}  |  id={entry['id']}  |  type={entry['qa_type'].upper()}  |  format={entry['response_format']}{mode_str}")
        print("=" * 100)

        print()
        print("-" * 60)
        print("GENERATOR USER PROMPT (sent to the LLM):")
        print("-" * 60)
        print(entry["generator_user_prompt"])

        print()
        print("-" * 60)
        print("RESULTING Q/A PAIR:")
        print("-" * 60)
        print(f"Q: {entry['question']}")
        print()
        print(f"A: {entry['answer']}")

        if entry.get("qa_reasoning"):
            print()
            print(f"Reasoning: {entry['qa_reasoning']}")

        print()
        print()


if __name__ == "__main__":
    main()
