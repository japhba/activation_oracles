"""
Combine multiple completion part files into a single dataset.

Usage:
    .venv/bin/python data_pipelines/model_understanding/combine_completions.py \
        --inputs part1.json part2.json part3.json part4.json \
        --output combined.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def combine(input_files: list[str], output_file: str):
    all_prompts = []
    metadata = None

    for f in input_files:
        print(f"Loading {f}...")
        data = json.loads(Path(f).read_text())
        if metadata is None:
            metadata = data["metadata"]
        all_prompts.extend(data["prompts"])
        print(f"  {len(data['prompts'])} prompts loaded")

    # Sort by ID for consistent ordering
    all_prompts.sort(key=lambda p: p["id"])

    # Check for duplicate IDs
    ids = [p["id"] for p in all_prompts]
    unique_ids = set(ids)
    if len(ids) != len(unique_ids):
        dupes = [id_ for id_ in unique_ids if ids.count(id_) > 1]
        raise ValueError(f"Duplicate prompt IDs found: {dupes}")

    # Update metadata
    metadata["n_prompts"] = len(all_prompts)
    metadata["total_completions"] = sum(len(p["completions"]) for p in all_prompts)
    metadata["combined_from"] = [str(f) for f in input_files]
    metadata["combined_at"] = datetime.now(timezone.utc).isoformat()
    metadata.pop("offset", None)

    combined = {"metadata": metadata, "prompts": all_prompts}

    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(all_prompts)} prompts to {out}...")
    out.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    print(f"Done. {len(all_prompts)} prompts, {metadata['total_completions']} total completions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine completion part files")
    parser.add_argument("--inputs", nargs="+", required=True, help="Part files to combine")
    parser.add_argument("--output", type=str, required=True, help="Output combined file path")
    args = parser.parse_args()
    combine(args.inputs, args.output)
