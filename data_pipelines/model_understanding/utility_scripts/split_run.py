"""Split a pipeline run into train and test sets.

Samples prompt_ids from verification score buckets for the test set.
Everything else goes to train. The original run directory is only read,
never modified.

Usage:
    .venv/bin/python data_pipelines/model_understanding/utility_scripts/split_run.py \
        --run-dir data_pipelines/model_understanding/runs/qwen3_32b_run_2 \
        --scores 6 7 8 9 10 \
        --samples-per-score 100 \
        --seed 42
"""

import argparse
import json
import random
from pathlib import Path


FILES_TO_SPLIT = ["verification.json", "investigations.json", "screening.json", "synthetic_data.json"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    print(f"  Wrote {path} ({len(data.get('results', []))} results)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a run into train/test sets")
    parser.add_argument("--run-dir", type=Path, required=True, help="Source run directory")
    parser.add_argument("--scores", type=int, nargs="+", required=True, help="Score buckets to sample from")
    parser.add_argument("--samples-per-score", type=int, required=True, help="Number of samples per score bucket")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
    args = parser.parse_args()

    src_dir = args.run_dir
    train_dir = src_dir.parent / f"{src_dir.name}_train"
    test_dir = src_dir.parent / f"{src_dir.name}_test"

    # --- Load verification to get score buckets ---
    verification = load_json(src_dir / "verification.json")

    by_score: dict[int, list[str]] = {}
    for result in verification["results"]:
        score = result["score"]
        by_score.setdefault(score, []).append(result["prompt_id"])

    # --- Sample test set prompt_ids ---
    rng = random.Random(args.seed)
    test_prompt_ids: set[str] = set()

    print("Test set sampling:")
    for score in args.scores:
        available = by_score.get(score, [])
        assert len(available) >= args.samples_per_score, (
            f"Score {score}: only {len(available)} available, need {args.samples_per_score}"
        )
        sampled = rng.sample(available, args.samples_per_score)
        test_prompt_ids.update(sampled)
        print(f"  Score {score}: sampled {len(sampled)} from {len(available)} available")

    print(f"  Total test prompt_ids: {len(test_prompt_ids)}")

    # --- Split each JSON file ---
    for filename in FILES_TO_SPLIT:
        filepath = src_dir / filename
        if not filepath.exists():
            print(f"\n{filename}: not found, skipping")
            continue

        data = load_json(filepath)
        train_results = [r for r in data["results"] if r["prompt_id"] not in test_prompt_ids]
        test_results = [r for r in data["results"] if r["prompt_id"] in test_prompt_ids]

        print(f"\n{filename}:")
        print(f"  Original: {len(data['results'])}")
        print(f"  Train:    {len(train_results)}")
        print(f"  Test:     {len(test_results)}")

        train_data = {"metadata": data["metadata"], "results": train_results}
        test_data = {"metadata": data["metadata"], "results": test_results}

        save_json(train_dir / filename, train_data)
        save_json(test_dir / filename, test_data)

    # --- Save the test manifest ---
    manifest = {
        "seed": args.seed,
        "scores_sampled": args.scores,
        "samples_per_score": args.samples_per_score,
        "prompt_ids": sorted(test_prompt_ids),
    }
    save_json(test_dir / "test_manifest.json", manifest)
    print(f"\nSaved test manifest with {len(test_prompt_ids)} prompt_ids")

    print(f"\nDone.")
    print(f"  Train: {train_dir}")
    print(f"  Test:  {test_dir}")
    print(f"  Original {src_dir.name}/ was not modified.")


if __name__ == "__main__":
    main()
