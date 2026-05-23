"""
Cross-pair validation for hidden bias "inactive" training data.

Takes an existing hidden bias run, cross-pairs system prompts with unrelated
user prompts (adding "only follow if relevant" scoping), and regenerates 50
responses with vLLM to verify the model ignores irrelevant instructions.

Compares regenerated responses to baselines to confirm they're similar.

Usage (GPU node or Slurm):
    python data_pipelines/training_data/validate_cross_pair.py \
        --run hidden_bias_v2_qwen3_8b \
        --model Qwen/Qwen3-8B \
        --num-validate 50
"""

import argparse
import json
import random
from pathlib import Path

from training_data.stage12_vllm import SEED

ARTIFACTS_DIR = Path("data_pipelines/training_data/artifacts/system_prompt_qa")

SCOPE_SUFFIX = (
    " Only follow this instruction if it directly applies to the current "
    "conversation. Otherwise, respond normally and ignore this instruction entirely."
)


def cross_pair(entries: list[dict], seed: int = SEED) -> list[dict]:
    """Cross-pair system prompts with unrelated user prompts.

    Shuffles system prompts so each entry gets a system prompt from a different
    entry. Uses baseline_response as the expected "correct" response (since the
    system prompt is irrelevant).
    """
    rng = random.Random(seed + 100)

    # Separate system prompts and shuffle
    system_prompts = [e["system_prompt"] for e in entries]
    categories = [e["category"] for e in entries]
    ground_truths = [e["ground_truth"] for e in entries]

    # Shift by a large offset to ensure no self-pairing
    n = len(entries)
    offset = n // 3
    shuffled_indices = list(range(n))
    rng.shuffle(shuffled_indices)
    # Rotate to avoid any entry getting its own system prompt
    rotated = shuffled_indices[offset:] + shuffled_indices[:offset]

    paired = []
    for i, entry in enumerate(entries):
        donor_idx = rotated[i]
        paired.append({
            "id": f"xpair_{i:05d}",
            "original_id": entry["id"],
            "donor_id": entries[donor_idx]["id"],
            "system_prompt": system_prompts[donor_idx] + SCOPE_SUFFIX,
            "system_prompt_raw": system_prompts[donor_idx],
            "category": categories[donor_idx],
            "ground_truth": ground_truths[donor_idx],
            "user_prompt": entry["user_prompt"],
            "baseline_response": entry["baseline_response"],
        })

    return paired


def validate_with_vllm(
    paired: list[dict], model_name: str, num_validate: int, seed: int = SEED,
):
    """Regenerate a subset with vLLM and compare to baselines."""
    from generate_hidden_bias_data import generate_vllm_responses

    rng = random.Random(seed + 200)
    sample = rng.sample(paired, min(num_validate, len(paired)))

    print(f"\n{'='*60}")
    print(f"Validating {len(sample)} cross-paired entries with vLLM")
    print(f"{'='*60}")

    prompts = [
        {"system": e["system_prompt"], "user": e["user_prompt"]}
        for e in sample
    ]
    responses, kept_indices = generate_vllm_responses(model_name, prompts)
    sample = [sample[i] for i in kept_indices]

    # Compare
    results = []
    for entry, regenerated in zip(sample, responses):
        baseline = entry["baseline_response"]

        # Simple similarity: character-level overlap ratio
        baseline_words = set(baseline.lower().split())
        regen_words = set(regenerated.lower().split())
        if baseline_words or regen_words:
            jaccard = len(baseline_words & regen_words) / len(baseline_words | regen_words)
        else:
            jaccard = 1.0

        len_ratio = len(regenerated) / len(baseline) if baseline else 999

        results.append({
            **entry,
            "regenerated_response": regenerated,
            "jaccard_similarity": round(jaccard, 3),
            "length_ratio": round(len_ratio, 2),
        })

    return results


def print_report(results: list[dict]):
    """Print comparison report."""
    jaccards = [r["jaccard_similarity"] for r in results]
    len_ratios = [r["length_ratio"] for r in results]

    print(f"\n{'='*60}")
    print(f"CROSS-PAIR VALIDATION REPORT ({len(results)} entries)")
    print(f"{'='*60}")
    print(f"\nJaccard word similarity (baseline vs regenerated):")
    print(f"  Mean: {sum(jaccards)/len(jaccards):.3f}")
    print(f"  Min:  {min(jaccards):.3f}")
    print(f"  Max:  {max(jaccards):.3f}")
    print(f"\nLength ratio (regenerated / baseline):")
    print(f"  Mean: {sum(len_ratios)/len(len_ratios):.2f}")
    print(f"  Min:  {min(len_ratios):.2f}")
    print(f"  Max:  {max(len_ratios):.2f}")

    # Show some examples
    print(f"\n{'='*60}")
    print("SAMPLE ENTRIES (sorted by similarity, lowest first)")
    print(f"{'='*60}")

    sorted_results = sorted(results, key=lambda r: r["jaccard_similarity"])
    for r in sorted_results[:10]:
        print(f"\n--- {r['id']} (jaccard={r['jaccard_similarity']:.3f}, len_ratio={r['length_ratio']:.2f}) ---")
        print(f"  Irrelevant system prompt: {r['system_prompt_raw'][:120]}...")
        print(f"  Category: {r['category']}")
        print(f"  User prompt: {r['user_prompt'][:120]}...")
        print(f"  Baseline (first 200): {r['baseline_response'][:200]}...")
        print(f"  Regenerated (first 200): {r['regenerated_response'][:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Cross-pair validation for inactive hidden bias data")
    parser.add_argument("--run", type=str, required=True, help="Source run name to cross-pair from")
    parser.add_argument("--model", type=str, required=True, help="Model name for vLLM regeneration")
    parser.add_argument("--num-validate", type=int, default=50, help="Number of entries to regenerate for validation")
    parser.add_argument("--output", type=str, default=None, help="Output path for results (default: artifacts/<run>/cross_pair_validation.json)")
    args = parser.parse_args()

    run_dir = ARTIFACTS_DIR / args.run
    stage3_path = run_dir / "stage3_hidden_bias.json"
    assert stage3_path.exists(), f"Stage 3 output not found: {stage3_path}"

    with open(stage3_path) as f:
        stage3_data = json.load(f)

    entries = stage3_data["entries"]
    print(f"Loaded {len(entries)} entries from {stage3_path}")

    # Cross-pair
    paired = cross_pair(entries)
    print(f"Cross-paired {len(paired)} entries")

    # Show a few examples of the pairing
    print(f"\nSample cross-pairings:")
    for p in paired[:3]:
        print(f"  {p['id']}: prompt='{p['user_prompt'][:80]}...' ← sys='{p['system_prompt_raw'][:80]}...'")

    # Validate with vLLM
    results = validate_with_vllm(paired, args.model, args.num_validate)

    # Report
    print_report(results)

    # Save
    output_path = args.output or str(run_dir / "cross_pair_validation.json")
    with open(output_path, "w") as f:
        json.dump({
            "metadata": {
                "source_run": args.run,
                "model": args.model,
                "num_validated": len(results),
                "num_total_paired": len(paired),
                "scope_suffix": SCOPE_SUFFIX,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
