"""
Generate target model responses for all persona+goal latentQA entries.

Loads entries from system.json (system prompt + user message), generates
responses with vLLM (thinking disabled), and saves to a JSON file.

Usage:
    source .env
    .venv/bin/python data_pipelines/latentqa_datasets/generate_responses.py --model Qwen/Qwen3-8B

Output: data_pipelines/latentqa_datasets/artifacts/<model_short_name>/responses.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from data_pipelines.pipeline_utils import model_dir_name, add_model_arg

MAX_RESPONSE_TOKENS = 400
DATASET_DIR = Path(__file__).parent / "train"


def load_entries():
    """Load persona + goal entries from system.json."""
    with open(DATASET_DIR / "system.json") as f:
        entries = json.load(f)

    # system.json and stimulus_completion.json have identical entries —
    # system.json already has the system prompt structure we want.
    result = []
    for e in entries:
        result.append({
            "label": e["label"],
            "system_prompt": e["system"],
            "user_message": e["stimulus_user"],
        })

    persona = sum(1 for e in result if e["label"].startswith("persona"))
    goal = sum(1 for e in result if e["label"].startswith("goal"))
    print(f"Loaded {len(result)} entries ({persona} persona, {goal} goal)")
    return result


def generate_responses(entries, model_name):
    """Generate target model responses via vLLM."""
    import vllm
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = vllm.LLM(
        model=model_name,
        max_model_len=4096,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
    )

    # Build prompts and measure prompt lengths
    formatted = []
    for entry in entries:
        messages = [
            {"role": "system", "content": entry["system_prompt"]},
            {"role": "user", "content": entry["user_message"]},
        ]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        entry["prompt_token_count"] = len(token_ids)
        formatted.append({"prompt_token_ids": token_ids})

    # Cap max_tokens so prompt + response stays roughly within 512 tokens.
    # Each entry gets its own max_tokens based on prompt length.
    sampling_params_by_max = {}
    requests = []
    for entry, fmt in zip(entries, formatted):
        max_tokens = max(64, min(MAX_RESPONSE_TOKENS, 512 - entry["prompt_token_count"]))
        if max_tokens not in sampling_params_by_max:
            sampling_params_by_max[max_tokens] = vllm.SamplingParams(
                temperature=0.7,
                max_tokens=max_tokens,
            )
        requests.append((fmt, sampling_params_by_max[max_tokens]))

    # vLLM generate doesn't support per-request sampling params directly,
    # so we batch by max_tokens value.
    print(f"Generating responses (max_tokens distribution: "
          f"min={min(sampling_params_by_max)}, max={max(sampling_params_by_max)})...")

    # Group by max_tokens for batched generation
    groups = {}
    for i, (fmt, sp) in enumerate(requests):
        key = sp.max_tokens
        if key not in groups:
            groups[key] = []
        groups[key].append((i, fmt))

    results = [None] * len(entries)
    for max_tok, group in sorted(groups.items()):
        indices, fmts = zip(*group)
        sp = sampling_params_by_max[max_tok]
        print(f"  Batch max_tokens={max_tok}: {len(fmts)} entries")
        outputs = llm.generate(list(fmts), sp)
        for idx, output in zip(indices, outputs):
            results[idx] = output.outputs[0].text

    for entry, response in zip(entries, results):
        entry["response"] = response

    return entries


def save_results(entries, output_dir, model_name):
    """Save to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "responses.json"

    output = {
        "metadata": {
            "model": model_name,
            "max_response_tokens": MAX_RESPONSE_TOKENS,
            "total_entries": len(entries),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "entries": entries,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(entries)} entries to {output_path}")


def main(model_name):
    output_dir = Path(__file__).parent / "artifacts" / model_dir_name(model_name)

    entries = load_entries()
    entries = generate_responses(entries, model_name)

    # Print stats
    response_lens = [len(e["response"].split()) for e in entries]
    print(f"\nResponse word count: min={min(response_lens)}, max={max(response_lens)}, "
          f"mean={sum(response_lens)/len(response_lens):.0f}")

    save_results(entries, output_dir, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_arg(parser)
    args = parser.parse_args()
    main(args.model)
