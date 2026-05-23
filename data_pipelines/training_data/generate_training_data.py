"""
Synthetic training data pipeline for activation oracles.

Subcommands:
- prep: create a run-scoped prompt set
- stage12: run stages 1-2 locally or for a specific shard
- merge-stage12: merge shard outputs into a canonical merged file
- stage3: generate QA pairs from a merged stage12 output
- all: convenience wrapper for local end-to-end runs

Usage:
    source .env
    .venv/bin/python data_pipelines/training_data/generate_training_data.py prep --run my_run --n-prompts 1000
    .venv/bin/python data_pipelines/training_data/generate_training_data.py stage12 --run my_run
    .venv/bin/python data_pipelines/training_data/generate_training_data.py stage3 --run my_run
"""

import argparse
import asyncio
import functools
import json
from datetime import datetime, timezone
from pathlib import Path

# Force unbuffered output
print = functools.partial(print, flush=True)

from data_pipelines.training_data.stage12_vllm import (
    N_PROMPTS,
    QA_MODEL,
    SEED,
    get_run_dir,
    get_stage12_merged_path,
    load_run_manifest,
    merge_stage12,
    prep_prompts,
    run_stage12,
)
from data_pipelines.training_data.stage3_qa import stage_3_generate_qa, stage_3_generate_qa_batch


def _get_stage3_paths(run):
    run_dir = get_run_dir(run)
    stage3_dir = run_dir / "stage3"
    return {
        "stage3_dir": stage3_dir,
        "qa_json": stage3_dir / "qa.json",
        "qa_jsonl": stage3_dir / "qa.jsonl",
        "batch_state": stage3_dir / "batch_state.json",
        "dataset": run_dir / "training_data.json",
    }


def assemble_dataset(entries, manifest=None):
    """Assemble final dataset."""
    print("\n=== Assembling dataset ===")

    output_entries = []
    for entry in entries:
        out = {
            "id": entry["id"],
            "source": entry.get("source", "wildchat"),
            "qa_type": entry["qa_type"],
            "enable_thinking": entry["enable_thinking"],
            "response_format": entry["response_format"],
            "past_mode": entry.get("past_mode", ""),
            "question": entry["question"],
            "answer": entry["answer"],
            "prompt_messages": entry["prompt_messages"],
            "prefix_text": entry.get("prefix_text", ""),
            "selected_text": entry.get("selected_text", ""),
            "window_start": entry["window_start"],
            "window_end": entry["window_end"],
            "window_desc": entry["window_desc"],
            "prompt_len": entry.get("prompt_len", 0),
            "response_text": (
                entry["response_text"] if entry["qa_type"] == "past" else entry["prefix"]
            ),
            "qa_reasoning": entry.get("qa_reasoning", ""),
            "generator_system_prompt": entry.get("generator_system_prompt", ""),
            "generator_user_prompt": entry.get("generator_user_prompt", ""),
        }
        if entry["qa_type"] == "future":
            out["truncation_pos"] = entry["truncation_pos"]
            out["original_continuation"] = entry["original_continuation"]
            out["continuations"] = entry["continuations"]
        output_entries.append(out)

    metadata = {
        "model": manifest.get("target_model", "unknown") if manifest else "unknown",
        "qa_model": QA_MODEL,
        "total_entries": len(output_entries),
        "past_entries": sum(1 for e in output_entries if e["qa_type"] == "past"),
        "future_entries": sum(1 for e in output_entries if e["qa_type"] == "future"),
        "response_formats": dict(
            (fmt, sum(1 for e in output_entries if e["response_format"] == fmt))
            for fmt in set(e["response_format"] for e in output_entries)
        ),
        "seed": SEED,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if manifest:
        metadata["run"] = manifest.get("run", "")
        metadata["n_prompts"] = manifest.get("n_prompts", 0)

    return {
        "metadata": metadata,
        "entries": output_entries,
    }


def _print_dataset_summary(dataset, output_path):
    print(f"\n{'=' * 60}")
    print(f"DONE! {len(dataset['entries'])} entries saved to {output_path}")
    print(f"\nBreakdown:")
    print(f"  Past-context: {dataset['metadata']['past_entries']}")
    print(f"  Future-continuation: {dataset['metadata']['future_entries']}")
    print(f"  Response formats: {dataset['metadata']['response_formats']}")

    sample_size = min(20, len(dataset["entries"]))
    print(f"\n{'=' * 60}")
    print(f"SAMPLE ENTRIES ({sample_size}/{len(dataset['entries'])})")
    print("=" * 60)
    for entry in dataset["entries"][:sample_size]:
        print(f"\n--- {entry['id']} ({entry['qa_type']}/{entry['response_format']}) ---")
        print(f"Q: {entry['question']}")
        print(f"A: {entry['answer']}")
        print(
            f"Window: {entry['window_desc']} "
            f"(chars {entry['window_start']}-{entry['window_end']})"
        )
        print()


async def run_stage3(run, use_batch=False):
    """Run stage 3 from a merged stage12 artifact and assemble the final dataset."""
    manifest = load_run_manifest(run)
    merged_path = get_stage12_merged_path(run)
    assert merged_path.exists(), (
        f"Merged stage12 output not found: {merged_path}\n"
        f"Run `stage12 --run {run}` (and `merge-stage12` for sharded runs) first."
    )

    stage3_paths = _get_stage3_paths(run)
    stage3_paths["stage3_dir"].mkdir(parents=True, exist_ok=True)

    target_model = manifest["target_model"]
    print(f"Target model: {target_model}")

    if stage3_paths["qa_json"].exists():
        print("Loading cached stage 3 results...")
        selected = json.loads(stage3_paths["qa_json"].read_text())
    else:
        selected = json.loads(merged_path.read_text())
        if use_batch:
            selected = stage_3_generate_qa_batch(
                selected,
                target_model,
                jsonl_path=stage3_paths["qa_jsonl"],
                batch_state_path=stage3_paths["batch_state"],
            )
        else:
            selected = await stage_3_generate_qa(
                selected,
                target_model,
                jsonl_path=stage3_paths["qa_jsonl"],
            )
        stage3_paths["qa_json"].write_text(json.dumps(selected, indent=2, ensure_ascii=False))

    dataset = assemble_dataset(selected, manifest=manifest)
    stage3_paths["dataset"].write_text(json.dumps(dataset, indent=2, ensure_ascii=False))
    _print_dataset_summary(dataset, stage3_paths["dataset"])
    return dataset


async def run_all(run, n_prompts, target_model, use_batch=False):
    """Convenience wrapper for local end-to-end runs."""
    prep_prompts(run, n_prompts, target_model)
    run_stage12(run, shard=0, num_shards=1)
    merge_stage12(run, num_shards=1)
    await run_stage3(run, use_batch=use_batch)


def build_parser():
    parser = argparse.ArgumentParser(description="Generate training data for activation oracles.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep_parser = subparsers.add_parser(
        "prep",
        help="Create a run-scoped prompt set from WildChat.",
    )
    prep_parser.add_argument("--run", required=True, help="Run name under artifacts/.")
    prep_parser.add_argument("--model", required=True, help="Target model (e.g. Qwen/Qwen3-14B).")
    prep_parser.add_argument(
        "--n-prompts",
        type=int,
        default=N_PROMPTS,
        help="Number of prompts to sample.",
    )

    stage12_parser = subparsers.add_parser(
        "stage12",
        help="Run stages 1-2 locally or for one shard.",
    )
    stage12_parser.add_argument("--run", required=True, help="Run name under artifacts/.")
    stage12_parser.add_argument("--shard", type=int, default=0, help="Shard index (0-based).")
    stage12_parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total shard count. Omit for local mode.",
    )
    stage12_parser.add_argument(
        "--stop-after-stage",
        type=int,
        default=None,
        help="Optional early stop (e.g. 1 to skip stage 2).",
    )

    merge_parser = subparsers.add_parser(
        "merge-stage12",
        help="Merge stage12 shard outputs into a canonical merged file.",
    )
    merge_parser.add_argument("--run", required=True, help="Run name under artifacts/.")
    merge_parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total shard count to merge.",
    )

    stage3_parser = subparsers.add_parser(
        "stage3",
        help="Generate QA pairs from a merged stage12 artifact.",
    )
    stage3_parser.add_argument("--run", required=True, help="Run name under artifacts/.")
    stage3_parser.add_argument(
        "--batch",
        action="store_true",
        help="Use Anthropic Batch API for stage 3.",
    )

    all_parser = subparsers.add_parser(
        "all",
        help="Convenience wrapper for local end-to-end runs.",
    )
    all_parser.add_argument("--run", required=True, help="Run name under artifacts/.")
    all_parser.add_argument("--model", required=True, help="Target model (e.g. Qwen/Qwen3-14B).")
    all_parser.add_argument(
        "--n-prompts",
        type=int,
        default=N_PROMPTS,
        help="Number of prompts to sample.",
    )
    all_parser.add_argument(
        "--batch",
        action="store_true",
        help="Use Anthropic Batch API for stage 3.",
    )

    return parser


async def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prep":
        prep_prompts(args.run, args.n_prompts, args.model)
        return

    if args.command == "stage12":
        run_stage12(
            args.run,
            shard=args.shard,
            num_shards=args.num_shards,
            stop_after_stage=args.stop_after_stage,
        )
        return

    if args.command == "merge-stage12":
        merge_stage12(args.run, num_shards=args.num_shards)
        return

    if args.command == "stage3":
        await run_stage3(args.run, use_batch=args.batch)
        return

    if args.command == "all":
        await run_all(args.run, args.n_prompts, args.model, use_batch=args.batch)
        return

    raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    asyncio.run(main())
