"""
Verify backtracking consistency: for each identified backtracking point,
feed the prefix to vLLM and generate 10 continuations to see if the model
consistently backtracks at that point.
"""

import json
import re
from pathlib import Path

import vllm
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-8B"
NUM_CONTINUATIONS = 10
MAX_CONTINUATION_TOKENS = 100

VERIFICATION_PATH = Path("data_pipelines/backtracking/verification_points.json")
OUTPUT_PATH = Path("data_pipelines/backtracking/verification_results.json")


def main():
    with open(VERIFICATION_PATH) as f:
        points = json.load(f)

    print(f"Loaded {len(points)} backtracking points to verify")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    llm = vllm.LLM(
        model=MODEL_NAME,
        max_model_len=4000,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )

    sampling_params = vllm.SamplingParams(
        temperature=1.0,
        top_p=0.95,
        max_tokens=MAX_CONTINUATION_TOKENS,
    )

    # Build prompts: original question + thinking prefix
    # Format: <chat template up to assistant turn><think>\n{prefix}
    all_prompts = []
    prompt_map = []  # (point_idx, continuation_idx)

    for pt_idx, point in enumerate(points):
        messages = [{"role": "user", "content": point["problem"]}]
        chat_prefix = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        # Append <think> + the thinking prefix up to the backtrack point
        full_prefix = chat_prefix + "<think>\n" + point["prefix"]

        for cont_idx in range(NUM_CONTINUATIONS):
            all_prompts.append(full_prefix)
            prompt_map.append((pt_idx, cont_idx))

    print(f"Generating {len(all_prompts)} continuations ({len(points)} points x {NUM_CONTINUATIONS} each)")

    outputs = llm.generate(all_prompts, sampling_params)

    # Analyze results
    backtrack_patterns = [
        r'\b[Ww]ait\b',
        r'\b[Aa]ctually\b',
        r'\b[Nn]o,\s',
        r'\b[Hh]old on\b',
        r'\b[Hh]mm\b',
        r'\blet me re',
        r'\bmistake\b',
        r'\bthat\'s (wrong|incorrect|not right)',
        r'\bthat (doesn\'t|can\'t|won\'t)',
        r'\bcontradiction\b',
        r'\bscratch that\b',
        r'\bcorrection\b',
    ]

    results = []
    for pt_idx, point in enumerate(points):
        continuations = []
        backtrack_count = 0

        for cont_idx in range(NUM_CONTINUATIONS):
            flat_idx = pt_idx * NUM_CONTINUATIONS + cont_idx
            text = outputs[flat_idx].outputs[0].text

            has_backtrack = any(
                re.search(pat, text[:80], re.IGNORECASE)
                for pat in backtrack_patterns
            )
            if has_backtrack:
                backtrack_count += 1

            continuations.append({
                "text": text,
                "has_backtrack": has_backtrack,
            })

        results.append({
            "problem_id": point["problem_id"],
            "problem": point["problem"],
            "rollout_idx": point["rollout_idx"],
            "backtrack_char_pos": point["backtrack_char_pos"],
            "prefix_tail": point["prefix"][-200:],
            "original_suffix": point["suffix"][:200],
            "backtrack_rate": backtrack_count / NUM_CONTINUATIONS,
            "continuations": continuations,
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Summary stats
    rates = [r["backtrack_rate"] for r in results]
    print(f"\nDone! Saved to {OUTPUT_PATH}")
    print(f"\nBacktrack consistency rates across {len(results)} points:")
    print(f"  Mean: {sum(rates)/len(rates):.2f}")
    print(f"  Always backtracks (10/10): {sum(1 for r in rates if r == 1.0)}")
    print(f"  Usually backtracks (>=7/10): {sum(1 for r in rates if r >= 0.7)}")
    print(f"  Sometimes (3-6/10): {sum(1 for r in rates if 0.3 <= r < 0.7)}")
    print(f"  Rarely (<3/10): {sum(1 for r in rates if r < 0.3)}")

    # Show a few examples
    sorted_results = sorted(results, key=lambda x: -x["backtrack_rate"])
    print(f"\n--- Most consistent (top 5) ---")
    for r in sorted_results[:5]:
        print(f"  {r['problem_id']:30s} rate={r['backtrack_rate']:.1f} | prefix: ...{r['prefix_tail'][-80:]}")

    print(f"\n--- Least consistent (bottom 5) ---")
    for r in sorted_results[-5:]:
        print(f"  {r['problem_id']:30s} rate={r['backtrack_rate']:.1f} | prefix: ...{r['prefix_tail'][-80:]}")


if __name__ == "__main__":
    main()
