"""Filter synthetic data to exclude test set prompt_ids.

One-off script for when synthetic data was generated before the
train/test split. Reads the test manifest to get test prompt_ids,
then writes a filtered copy of the synthetic data to the train dir.

Usage:
    .venv/bin/python data_pipelines/model_understanding/utility_scripts/filter_synthetic_data.py \
        --synthetic-data data_pipelines/model_understanding/runs/qwen3_32b_run_2/synthetic_data_qwen3_32b_run2.json \
        --test-manifest data_pipelines/model_understanding/runs/qwen3_32b_run_2_test/test_manifest.json \
        --output data_pipelines/model_understanding/runs/qwen3_32b_run_2_train/synthetic_data_qwen3_32b_run2.json
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter synthetic data to exclude test prompt_ids")
    parser.add_argument("--synthetic-data", type=Path, required=True, help="Source synthetic data JSON")
    parser.add_argument("--test-manifest", type=Path, required=True, help="Test manifest with prompt_ids to exclude")
    parser.add_argument("--output", type=Path, required=True, help="Output path for filtered synthetic data")
    args = parser.parse_args()

    manifest = json.loads(args.test_manifest.read_text())
    test_prompt_ids = set(manifest["prompt_ids"])
    print(f"Loaded {len(test_prompt_ids)} test prompt_ids from {args.test_manifest}")

    data = json.loads(args.synthetic_data.read_text())
    original_count = len(data["results"])

    filtered = [r for r in data["results"] if r["prompt_id"] not in test_prompt_ids]
    removed = original_count - len(filtered)

    print(f"Original: {original_count}")
    print(f"Removed:  {removed} (matched test prompt_ids)")
    print(f"Kept:     {len(filtered)}")

    data["results"] = filtered
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
