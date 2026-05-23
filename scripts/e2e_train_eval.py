"""End-to-end: dataset generation → training → HF push → open-ended evals.

Usage (quick debugging run):
    python scripts/e2e_train_eval.py --quick --hf-repo-id adamkarvonen/ao-e2e-test

Usage (full training with config JSON):
    python scripts/e2e_train_eval.py --config training_configs/my_config.json --hf-repo-id adamkarvonen/my-lora

Options:
    --quick               Use training_configs/quick.json (small dataset, few steps)
    --config PATH         Use a saved config JSON
    --hf-repo-id ID       HuggingFace repo ID for the trained LoRA (required)
    --skip-gen            Skip dataset generation step
    --skip-eval           Skip open-ended evals after training
    --num-gpus N          Number of GPUs for torchrun (default: 1)
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def run(cmd: list[str], description: str) -> None:
    print(f"\n{'=' * 40}")
    print(f"  {description}")
    print(f"{'=' * 40}\n")
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"\nFailed: {description} (exit code {result.returncode})")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E train + eval pipeline")
    parser.add_argument("--quick", action="store_true", help="Use training_configs/quick.json")
    parser.add_argument("--config", type=str, default=None, help="Path to training config JSON")
    parser.add_argument("--hf-repo-id", type=str, required=True, help="HuggingFace repo ID for the trained LoRA")
    parser.add_argument("--skip-gen", action="store_true", help="Skip dataset generation step")
    parser.add_argument("--skip-eval", action="store_true", help="Skip open-ended evals after training")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for torchrun (default: 1)")
    args = parser.parse_args()

    if args.quick and args.config:
        parser.error("--quick and --config are mutually exclusive")
    if not args.quick and not args.config:
        parser.error("must specify either --quick or --config PATH")

    config = args.config if args.config else "training_configs/quick.json"

    print("=" * 40)
    print("  E2E Train + Eval Pipeline")
    print(f"  Config: {config}")
    print(f"  HF repo: {args.hf_repo_id}")
    print(f"  GPUs: {args.num_gpus}")
    print("=" * 40)

    # Step 1: Dataset generation
    if not args.skip_gen:
        run(
            [PYTHON, "nl_probes/sft.py", "--config", config, "--hf-repo-id", args.hf_repo_id, "--gen-only"],
            "Step 1: Dataset Generation",
        )
    else:
        print("\nSkipping dataset generation (--skip-gen)")

    # Step 2: Training (+ HF push)
    run(
        ["torchrun", f"--nproc_per_node={args.num_gpus}", "nl_probes/sft.py", "--config", config, "--hf-repo-id", args.hf_repo_id],
        "Step 2: Training",
    )

    # Step 3: Open-ended evals
    if not args.skip_eval:
        run(
            [PYTHON, "-m", "experiments.run_all_open_ended_evals", "--verbalizer-lora", args.hf_repo_id],
            "Step 3: Open-ended Evals",
        )
    else:
        print("\nSkipping evals (--skip-eval)")

    print(f"\n{'=' * 40}")
    print("  Pipeline complete!")
    print(f"  Model pushed to: https://huggingface.co/{args.hf_repo_id}")
    print("=" * 40)


if __name__ == "__main__":
    main()
