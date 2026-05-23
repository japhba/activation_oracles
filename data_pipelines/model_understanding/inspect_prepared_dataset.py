"""
Inspect a prepared .pt training dataset file.

Shows decoded training examples with the loss target highlighted,
plus summary statistics.

Usage:
    # Show 3 random examples
    .venv/bin/python data_pipelines/model_understanding/inspect_prepared_dataset.py \
        --pt text_sft_training_data/mu_qwen3_32b_s7_synthetic_test.pt

    # Show specific examples by index
    .venv/bin/python data_pipelines/model_understanding/inspect_prepared_dataset.py \
        --pt text_sft_training_data/mu_qwen3_32b_s7_synthetic_test.pt --indices 0,1,5

    # Show more examples
    .venv/bin/python data_pipelines/model_understanding/inspect_prepared_dataset.py \
        --pt text_sft_training_data/mu_qwen3_32b_s7_synthetic_test.pt --n 10

    # Only show synthetic examples (prompt_id contains __bp or __cf)
    .venv/bin/python data_pipelines/model_understanding/inspect_prepared_dataset.py \
        --pt text_sft_training_data/mu_qwen3_32b_s7_synthetic_test.pt --filter synthetic

    # Dump to markdown
    .venv/bin/python data_pipelines/model_understanding/inspect_prepared_dataset.py \
        --pt text_sft_training_data/mu_qwen3_32b_s7_synthetic_test.pt --n 5 --markdown out.md
"""

import argparse
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Inspect prepared .pt training dataset")
    parser.add_argument("--pt", type=Path, required=True, help="Path to .pt file")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name for tokenizer (auto-detected from .pt metadata)")
    parser.add_argument("--n", type=int, default=3, help="Number of examples to show")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated indices to show (e.g., 0,1,5)")
    parser.add_argument("--filter", choices=["synthetic", "investigation", "bp", "cf"],
                        default=None, help="Filter by example type")
    parser.add_argument("--markdown", type=Path, default=None,
                        help="Write output to markdown file instead of stdout")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    payload = torch.load(args.pt, map_location="cpu", weights_only=False)

    fingerprint = payload["fingerprint"]
    metadata = payload["metadata"]
    examples = payload["examples"]

    model_name = args.model or fingerprint.get("model_name", "Qwen/Qwen3-32B")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    lines = []

    def out(s=""):
        lines.append(s)

    out(f"# Prepared Dataset Inspection: {args.pt.name}\n")
    out(f"## Metadata\n")
    out(f"- Model: `{model_name}`")
    out(f"- Total examples: {metadata['num_examples']}")
    out(f"- Dropped (too long): {metadata['dropped_too_long']}")
    out(f"- Dropped (tokenization error): {metadata['dropped_tokenization_error']}")
    out(f"- Sequence length: min={metadata['min_seq_len']}, "
        f"median={metadata['median_seq_len']}, max={metadata['max_seq_len']}")
    out(f"- Screening results: {metadata['num_screening_results']}")
    out(f"- Investigations: {metadata['num_investigations']}")
    out(f"- Verifications: {metadata['num_verifications']}")
    out()

    # Count synthetic vs investigation
    synthetic_bp = [e for e in examples if "__bp" in e["prompt_id"]]
    synthetic_cf = [e for e in examples if "__cf_" in e["prompt_id"]]
    investigation = [e for e in examples if "__bp" not in e["prompt_id"] and "__cf_" not in e["prompt_id"]]
    out(f"## Example Counts\n")
    out(f"- Investigation examples: {len(investigation)}")
    out(f"- Synthetic behavior predictions: {len(synthetic_bp)}")
    out(f"- Synthetic counterfactual predictions: {len(synthetic_cf)}")
    out(f"- Total: {len(examples)}")
    out()

    # Filter
    if args.filter == "synthetic":
        examples = synthetic_bp + synthetic_cf
    elif args.filter == "investigation":
        examples = investigation
    elif args.filter == "bp":
        examples = synthetic_bp
    elif args.filter == "cf":
        examples = synthetic_cf

    if not examples:
        out("No examples match the filter.")
        _output(lines, args.markdown)
        return

    # Select examples
    if args.indices:
        indices = [int(x) for x in args.indices.split(",")]
        selected = [examples[i] for i in indices]
    else:
        rng = random.Random(args.seed)
        n = min(args.n, len(examples))
        selected = rng.sample(examples, n)

    out(f"## Examples ({len(selected)} shown)\n")

    for i, ex in enumerate(selected, 1):
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        loss_start = ex["loss_span"]["start"]
        loss_end = ex["loss_span"]["end"]

        # Decode context (everything before loss span)
        context_ids = input_ids[:loss_start]
        target_ids = input_ids[loss_start:loss_end]

        context_text = tokenizer.decode(context_ids, skip_special_tokens=False)
        target_text = tokenizer.decode(target_ids, skip_special_tokens=False)

        out(f"---\n")
        out(f"### Example {i}: `{ex['prompt_id']}`\n")
        out(f"- Sequence length: {ex['sequence_length']}")
        out(f"- Target tokens: {ex['target_token_count']}")
        out(f"- Loss span: [{loss_start}, {loss_end})")
        out(f"- Chosen completion index: {ex['chosen_completion_index']}")
        out()

        out(f"**Context (input, no loss):**\n")
        out(f"```\n{context_text}\n```\n")

        out(f"**Target (loss computed here):**\n")
        out(f"```\n{target_text}\n```\n")

    _output(lines, args.markdown)


def _output(lines, markdown_path):
    text = "\n".join(lines)
    if markdown_path:
        markdown_path.write_text(text)
        print(f"Wrote {markdown_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
