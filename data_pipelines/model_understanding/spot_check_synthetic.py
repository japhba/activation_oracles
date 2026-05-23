"""
Spot-check synthetic training data.

Shows the Sonnet prompt that was sent, the generated answer, and what the
final SFT training example would look like. Picks random examples or
a specific prompt_id.

Usage:
    # Random spot check (3 examples)
    .venv/bin/python data_pipelines/model_understanding/spot_check_synthetic.py \
        --input /tmp/synthetic_test.json

    # Specific prompt
    .venv/bin/python data_pipelines/model_understanding/spot_check_synthetic.py \
        --input /tmp/synthetic_test.json --id prompt_07059__bp

    # More examples
    .venv/bin/python data_pipelines/model_understanding/spot_check_synthetic.py \
        --input /tmp/synthetic_test.json --n 5

    # Only behavior predictions or only counterfactual
    .venv/bin/python data_pipelines/model_understanding/spot_check_synthetic.py \
        --input /tmp/synthetic_test.json --type bp
    .venv/bin/python data_pipelines/model_understanding/spot_check_synthetic.py \
        --input /tmp/synthetic_test.json --type cf
"""

import argparse
import json
import random
import textwrap
from pathlib import Path


DIVIDER = "=" * 80
THIN_DIVIDER = "-" * 60



def truncate(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [{len(text) - max_chars} more chars]"


def display_example(entry: dict, index: int, show_full: bool = False):
    trunc = (lambda t: t) if show_full else truncate

    print(f"\n{DIVIDER}")
    print(f"  EXAMPLE {index}: {entry['prompt_id']}  |  type={entry['example_type']}")
    print(f"  interest={entry['interest_score']}  verification={entry['verification_score']}"
          + (f"  experiment_index={entry['experiment_index']}" if entry.get('experiment_index') is not None else ""))
    print(DIVIDER)

    # 1. Sonnet prompt
    if entry.get("sonnet_system_prompt") or entry.get("sonnet_user_prompt"):
        print(f"\n{THIN_DIVIDER}")
        print("  SONNET SYSTEM PROMPT")
        print(THIN_DIVIDER)
        print(trunc(entry.get("sonnet_system_prompt", "(not recorded)")))

        print(f"\n{THIN_DIVIDER}")
        print("  SONNET USER PROMPT")
        print(THIN_DIVIDER)
        print(trunc(entry.get("sonnet_user_prompt", "(not recorded)")))
    else:
        print(f"\n  [Sonnet prompts not recorded — re-generate to capture them]")

    # 2. Sonnet output
    print(f"\n{THIN_DIVIDER}")
    print("  SONNET OUTPUT")
    print(THIN_DIVIDER)
    print(f"  Question: {entry['question']}")
    print(f"  Answer:   {entry['answer']}")

    # 3. SFT training example
    print(f"\n{THIN_DIVIDER}")
    print("  SFT TRAINING EXAMPLE (what the model sees during training)")
    print(THIN_DIVIDER)

    if entry["example_type"] == "behavior_prediction":
        # Format: original messages (with question appended to last user msg) + assistant answer
        # The question field contains separator + question text (varied per example)
        for msg in entry["messages"]:
            role = msg["role"].upper()
            content = msg["content"]
            if msg == entry["messages"][-1]:
                content = content + entry["question"]
            print(f"\n  [{role}]:")
            print(textwrap.indent(trunc(content), "    "))
        print(f"\n  [ASSISTANT] (target — loss computed here):")
        print(textwrap.indent(entry["answer"], "    "))

    elif entry["example_type"] == "counterfactual_prediction":
        # Format: original messages + completion + question + answer
        for msg in entry["messages"]:
            role = msg["role"].upper()
            print(f"\n  [{role}]:")
            print(textwrap.indent(trunc(msg["content"]), "    "))
        print(f"\n  [ASSISTANT] (completion):")
        print(textwrap.indent(trunc(entry["chosen_completion"]), "    "))
        print(f"\n  [USER] (question):")
        print(textwrap.indent(entry["question"], "    "))
        print(f"\n  [ASSISTANT] (target — loss computed here):")
        print(textwrap.indent(entry["answer"], "    "))

    print()


def main():
    parser = argparse.ArgumentParser(description="Spot-check synthetic training data")
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to synthetic_data.json")
    parser.add_argument("--n", type=int, default=3,
                        help="Number of random examples to show (default: 3)")
    parser.add_argument("--id", type=str, default=None,
                        help="Show a specific prompt_id (e.g., prompt_07059__bp)")
    parser.add_argument("--type", choices=["bp", "cf"], default=None,
                        help="Filter by type: bp=behavior_prediction, cf=counterfactual_prediction")
    parser.add_argument("--full", action="store_true",
                        help="Show full text without truncation")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible sampling")
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    results = data["results"]

    print(f"Loaded {len(results)} examples from {args.input}")
    bp_count = sum(1 for r in results if r["example_type"] == "behavior_prediction")
    cf_count = sum(1 for r in results if r["example_type"] == "counterfactual_prediction")
    print(f"  {bp_count} behavior predictions, {cf_count} counterfactual predictions")

    # Filter by type
    if args.type == "bp":
        results = [r for r in results if r["example_type"] == "behavior_prediction"]
        print(f"  Filtered to {len(results)} behavior predictions")
    elif args.type == "cf":
        results = [r for r in results if r["example_type"] == "counterfactual_prediction"]
        print(f"  Filtered to {len(results)} counterfactual predictions")

    # Select examples
    if args.id:
        # Match by full id (prompt_07059__bp) or base prompt_id (prompt_07059)
        selected = [r for r in results
                    if (f"{r['prompt_id']}__{r['example_type'][:2]}" == args.id
                        or r["prompt_id"] == args.id)]
        if not selected:
            # Try matching with experiment index
            selected = [r for r in results
                        if f"{r['prompt_id']}__cf_{r.get('experiment_index', '')}" == args.id]
        if not selected:
            print(f"\nNo examples found matching '{args.id}'")
            print("Available IDs (first 20):")
            for r in results[:20]:
                suffix = f"__cf_{r['experiment_index']}" if r.get("experiment_index") is not None else "__bp"
                print(f"  {r['prompt_id']}{suffix}")
            return
    else:
        rng = random.Random(args.seed)
        n = min(args.n, len(results))
        selected = rng.sample(results, n)

    for i, entry in enumerate(selected, 1):
        display_example(entry, i, show_full=args.full)

    # Summary
    if len(selected) < len(results):
        print(f"\nShowing {len(selected)}/{len(results)} examples. "
              f"Use --n to see more, --id to pick specific ones, "
              f"--full for untruncated text.")


if __name__ == "__main__":
    main()
