"""
Regenerate eval datasets for a given model.

Runs each dataset generation script as a subprocess with --model.
Scripts are run sequentially (only one vLLM instance can fit in GPU memory at a time).

Usage:
    # Regenerate all eval datasets:
    source .env && .venv/bin/python data_pipelines/regenerate_all.py --model Qwen/Qwen3-14B

    # Regenerate specific datasets:
    source .env && .venv/bin/python data_pipelines/regenerate_all.py --model Qwen/Qwen3-14B --datasets backtracking sycophancy

    # Dry run (show what would be run):
    .venv/bin/python data_pipelines/regenerate_all.py --model Qwen/Qwen3-14B --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Each entry: (dataset_name, script_path, extra_args, needs_env)
# Scripts are run in this order. Multi-step pipelines list their steps.
DATASET_SCRIPTS: list[tuple[str, str, list[str], bool]] = [
    (
        "backtracking",
        "data_pipelines/backtracking/generate_backtracking_rollouts.py",
        [],
        False,
    ),
    # NOTE: backtracking also needs assemble_backtracking_dataset.py and
    # generate_mc_options.py, but those are complex multi-step pipelines
    # that need manual intervention. generate_backtracking_rollouts.py
    # produces rollouts; assembling the eval dataset is a separate step.
    (
        "chat_regularization",
        "data_pipelines/chat_regularization/generate_dataset.py",
        [],
        False,
    ),
    (
        "mmlu_prediction",
        "data_pipelines/mmlu_prediction/generate_dataset.py",
        [],
        False,
    ),
    (
        "number_prediction",
        "data_pipelines/number_prediction/generate_dataset.py",
        [],
        False,
    ),
    (
        "sycophancy",
        "data_pipelines/sycophancy/generate_dataset.py",
        [],
        True,  # needs ANTHROPIC_API_KEY for Haiku judging
    ),
    # NOTE: missing_info and system_prompt_qa are model-agnostic (shared across
    # all target models). They live at flat paths like data_pipelines/missing_info/
    # without model subdirectories. Regenerate them manually if needed:
    #   .venv/bin/python data_pipelines/missing_info/generate_dataset.py --model Qwen/Qwen3-8B
    #   .venv/bin/python data_pipelines/system_prompt_qa/generate_handcrafted_dataset.py --model Qwen/Qwen3-8B
    #   .venv/bin/python data_pipelines/system_prompt_qa/generate_latentqa_dataset.py --model Qwen/Qwen3-8B
    (
        "latentqa_responses",
        "data_pipelines/latentqa_datasets/generate_responses.py",
        [],
        False,
    ),
    (
        "latentqa_qa",
        "data_pipelines/latentqa_datasets/generate_qa.py",
        ["--all"],
        True,  # needs ANTHROPIC_API_KEY for Claude Sonnet QA generation
    ),
]

# Canonical dataset names that map to one or more script entries
DATASET_GROUPS: dict[str, list[str]] = {
    "backtracking": ["backtracking"],
    "chat_regularization": ["chat_regularization"],
    "mmlu_prediction": ["mmlu_prediction"],
    "number_prediction": ["number_prediction"],
    "sycophancy": ["sycophancy"],
    "latentqa": ["latentqa_responses", "latentqa_qa"],
}

ALL_DATASET_NAMES = list(DATASET_GROUPS.keys())


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate eval datasets for a given model",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g. Qwen/Qwen3-14B)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASET_NAMES,
        help=f"Which datasets to regenerate (default: all). Choices: {ALL_DATASET_NAMES}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    selected_datasets = args.datasets or ALL_DATASET_NAMES

    # Resolve which script entries to run
    selected_entries: list[str] = []
    for ds_name in selected_datasets:
        selected_entries.extend(DATASET_GROUPS[ds_name])

    # Filter to selected entries, preserving order
    scripts_to_run = [
        (name, script, extra_args, needs_env)
        for name, script, extra_args, needs_env in DATASET_SCRIPTS
        if name in selected_entries
    ]

    print(f"Model: {args.model}")
    print(f"Datasets: {selected_datasets}")
    print(f"Scripts to run: {len(scripts_to_run)}")
    print()

    python = sys.executable  # Use the same Python that's running this script

    for i, (name, script, extra_args, needs_env) in enumerate(scripts_to_run, 1):
        cmd = [python, script, "--model", args.model] + extra_args
        print(f"[{i}/{len(scripts_to_run)}] {name}")
        print(f"  cmd: {' '.join(cmd)}")

        if needs_env:
            import os
            if not os.environ.get("ANTHROPIC_API_KEY"):
                print(f"  WARNING: {name} needs ANTHROPIC_API_KEY but it's not set. "
                      f"Run `source .env` first or this step will fail.")

        if args.dry_run:
            print("  (dry run — skipping)")
            print()
            continue

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n  FAILED with exit code {result.returncode}")
            print(f"  Stopping. Fix the error and re-run with: --datasets {name}")
            sys.exit(result.returncode)

        print(f"  Done.")
        print()

    print("All done!" if not args.dry_run else "Dry run complete.")


if __name__ == "__main__":
    main()
