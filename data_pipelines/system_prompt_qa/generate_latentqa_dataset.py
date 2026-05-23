"""
Generate the LatentQA system prompt detection eval dataset.

Same experiment as system_prompt_qa, but using LatentQA's stimulus eval set.
Maps LatentQA fields to the standard schema:
  - control_user -> system_prompt
  - stimulus_user -> user_prompt
  - label -> category, id
  - ground_truth = the control_user text (what the AO should detect)
  - assistant_response = generated fresh via vLLM

Usage:
    source .env && .venv/bin/python datasets/latentqa_system_prompt/generate_dataset.py --model Qwen/Qwen3-8B
"""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import vllm
from transformers import AutoTokenizer

from data_pipelines.pipeline_utils import add_model_arg, model_dir_name

SOURCE_PATH = Path("data_pipelines/latentqa_datasets/eval/stimulus.json")
NUM_ENTRIES = 100
SEED = 42


def main(model_name: str):
    output_path = Path("data_pipelines/system_prompt_qa/latentqa_eval_dataset.json")

    random.seed(SEED)

    source_data = json.loads(SOURCE_PATH.read_text())
    print(f"Loaded {len(source_data)} LatentQA stimulus entries")

    # Sample entries
    sampled = random.sample(source_data, min(NUM_ENTRIES, len(source_data)))
    print(f"Sampled {len(sampled)} entries")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm = vllm.LLM(
        model=model_name,
        max_model_len=4096,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
    )

    sampling_params = vllm.SamplingParams(
        temperature=0,
        max_tokens=1024,
    )

    # Build prompts: control_user as system prompt, stimulus_user as user prompt
    all_prompts = []
    for entry in sampled:
        messages = [
            {"role": "system", "content": entry["control_user"]},
            {"role": "user", "content": entry["stimulus_user"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        all_prompts.append(prompt)

    print(f"Generating responses for {len(all_prompts)} prompts...")
    outputs = llm.generate(all_prompts, sampling_params)

    # Build dataset entries in standard schema
    dataset_entries = []
    for i, (entry, output) in enumerate(zip(sampled, outputs)):
        response_text = output.outputs[0].text
        response_token_count = len(tokenizer.encode(response_text, add_special_tokens=False))

        dataset_entries.append({
            "id": f"latentqa_{i}",
            "category": entry["label"],
            "system_prompt": entry["control_user"],
            "user_prompt": entry["stimulus_user"],
            "assistant_response": response_text,
            "response_token_count": response_token_count,
            "ground_truth": (
                f"The model has the following system prompt instruction: "
                f"\"{entry['control_user']}\""
            ),
        })

        print(f"  {i}: [{entry['label']}] {response_token_count} tokens")

    # Free GPU
    del llm

    dataset = {
        "metadata": {
            "model": model_name,
            "total_entries": len(dataset_entries),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "data_pipelines/latentqa_datasets/eval/stimulus.json",
            "num_sampled": len(sampled),
            "seed": SEED,
            "description": (
                "System prompt detection eval using LatentQA stimulus data. "
                "control_user is the system prompt, stimulus_user is the user prompt. "
                "Same schema as system_prompt_qa eval."
            ),
        },
        "entries": dataset_entries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nSaved {len(dataset_entries)} entries to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LatentQA system prompt eval dataset")
    add_model_arg(parser)
    args = parser.parse_args()
    main(args.model)
