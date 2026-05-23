# %%
"""
Interactive spot-check notebook for the system prompt QA eval.

Run cells top-to-bottom. Pick an entry, then run the verification or AO cells
to spot-check that the pipeline reproduces roughly the same results.

Single model load — Qwen3-8B is loaded once and used for both answer
verification (base weights, greedy) and AO inference (LoRA adapter).
"""

import json
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(REPO_ROOT / ".env")

# Switch between datasets:
# DATASET_PATH = REPO_ROOT / "data_pipelines/system_prompt_qa/hidden_instruction_eval_dataset.json"
DATASET_PATH = REPO_ROOT / "data_pipelines/system_prompt_qa/latentqa_eval_dataset.json"


EVAL_RESULTS_DIR = REPO_ROOT / "experiments/system_prompt_qa_eval_results"

MODEL_NAME = "Qwen/Qwen3-8B"

# %%
# === Load model + AO adapter ===

import torch
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.open_ended_eval.system_prompt_qa import (
    build_prompt_infos_for_mode,
    VERBALIZER_PROMPTS_HIDDEN_INSTRUCTION,
    VERBALIZER_PROMPTS_SYSTEM_PROMPT_QA,
    ACTIVATION_MODES,
    GENERATION_KWARGS,
    judge_single_response,
    JUDGE_MODEL,
)
from nl_probes.open_ended_eval.eval_runner import (
    build_verbalizer_eval_config,
    ensure_default_adapter,
)
import nl_probes.base_experiment as base_experiment

# Pick the matching verbalizer prompt for the selected dataset
if "hidden_instruction" in str(DATASET_PATH):
    VERBALIZER_PROMPTS = VERBALIZER_PROMPTS_HIDDEN_INSTRUCTION
else:
    VERBALIZER_PROMPTS = VERBALIZER_PROMPTS_SYSTEM_PROMPT_QA

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = load_tokenizer(MODEL_NAME)
model = load_model(MODEL_NAME, torch.bfloat16)
model.eval()

ensure_default_adapter(model)

verbalizer_lora = "adamkarvonen/checkpoints_latentqa_cls_on_policy_Qwen3-8B"
adapter_name, training_config = base_experiment.load_oracle_adapter(model, verbalizer_lora)

eval_config = build_verbalizer_eval_config(
    model_name=MODEL_NAME,
    training_config=training_config,
    eval_batch_size=32,
    generation_kwargs=GENERATION_KWARGS,
)
print(f"Model + AO loaded: adapter={adapter_name}")

# %%
# === Load dataset and eval results ===

data = json.loads(DATASET_PATH.read_text())
entries = data["entries"]

print(f"Loaded {len(entries)} entries")
print(f"Categories: {sorted(set(e['category'] for e in entries))}")

# Try to find eval results
eval_data = {}
if EVAL_RESULTS_DIR.exists():
    for result_file in EVAL_RESULTS_DIR.rglob("*.json"):
        result = json.loads(result_file.read_text())
        mode = result.get("mode", result_file.parent.name)
        eval_data[mode] = result
        scored = result.get("scored_results", [])
        print(f"Loaded eval results for mode={mode}: {len(scored)} scored entries")

if not eval_data:
    print("No eval results found")

# %%
# === Pick an entry ===
# Change this index to explore different entries.

IDX = 0  # <-- change this

entry = entries[IDX]
print(f"Entry {IDX}  |  id={entry['id']}  |  category={entry['category']}")
print(f"\nSystem prompt:\n  {entry['system_prompt']}")
print(f"\nUser prompt:\n  {entry['user_prompt']}")
print(f"\nGround truth:\n  {entry['ground_truth']}")
print(f"\nAssistant response ({entry['response_token_count']} tokens):")
print(entry["assistant_response"])

# %%
# === Show saved eval results (if available) ===

if eval_data:
    for mode, result in eval_data.items():
        scored = result.get("scored_results", [])
        matching = [sr for sr in scored if sr.get("entry_id") == entry["id"]]
        if matching:
            print(f"--- Mode: {mode} ---")
            for sr in matching:
                spec = sr.get("specificity", "?")
                corr = sr.get("correctness", "?")
                print(f"  spec={spec}  corr={corr}")
                print(f"  reasoning: {sr.get('reasoning', '')}")
                print(f"  AO response: {sr.get('ao_response', '')}")
        else:
            print(f"--- Mode: {mode}: no result for entry {entry['id']} ---")
else:
    print("No eval results loaded")

# %%
# === Verify model response ===
# Runs the model at temperature 0 with the system prompt to confirm the stored response.

model.set_adapter("default")

messages = [
    {"role": "system", "content": entry["system_prompt"]},
    {"role": "user", "content": entry["user_prompt"]},
]
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
input_ids = tokenizer.encode(formatted, return_tensors="pt", add_special_tokens=False).to(device)

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=1024,
        do_sample=False,
    )

new_tokens = output[0][input_ids.shape[1] :]
response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

print(f"Entry: {entry['id']}")
print("\nStored response:")
print(entry["assistant_response"])
print("\nFresh response:")
print(response_text)
print(f"\nExact match: {response_text.strip() == entry['assistant_response'].strip()}")

# %%
# === Visualize selected tokens for ALL activation modes ===
# Shows which tokens the AO reads activations from for the current entry,
# printed for every mode so you can scroll through and compare.

for mode in ACTIVATION_MODES:
    prompt_infos_viz, _ = build_prompt_infos_for_mode([entry], mode, tokenizer, verbalizer_prompts=VERBALIZER_PROMPTS)
    info = prompt_infos_viz[0]
    positions_set = set(info.positions)

    print(f"\n{'=' * 70}")
    print(f"MODE: {mode}  |  entry: {entry['id']}")
    print(f"Total tokens: {len(info.context_token_ids)}  |  Selected: {len(info.positions)} tokens")
    print("-" * 70)
    for i, token_id in enumerate(info.context_token_ids):
        token_str = tokenizer.decode([token_id]).replace("\n", "\\n").replace("\r", "\\r")
        if i in positions_set:
            print(f"  [{i:3d}] >>> {token_str}")
        else:
            print(f"  [{i:3d}]     {token_str}")
    print("-" * 70)

    segment_token_ids = [info.context_token_ids[p] for p in info.positions]
    print(f"\n--- AO segment decoded ({len(info.positions)} tokens) ---")
    print(tokenizer.decode(segment_token_ids))
    print(f"\n--- Full prompt decoded ({len(info.context_token_ids)} tokens) ---")
    print(tokenizer.decode(info.context_token_ids))

# %%
# === Run AO across all activation modes for current entry ===
# Builds all prompt_infos across modes in one batch, runs verbalizer once.

all_prompt_infos = []
all_metadata = []
for mode in ACTIVATION_MODES:
    pis, metas = build_prompt_infos_for_mode([entry], mode, tokenizer, verbalizer_prompts=VERBALIZER_PROMPTS)
    all_prompt_infos.extend(pis)
    all_metadata.extend(metas)

all_results = base_experiment.run_verbalizer(
    model=model,
    tokenizer=tokenizer,
    verbalizer_prompt_infos=all_prompt_infos,
    verbalizer_lora_path=adapter_name,
    target_lora_path=None,
    config=eval_config,
    device=device,
)

print(f"Entry: {entry['id']}  |  Category: {entry['category']}")
print(f"Ground truth: {entry['ground_truth']}")
print()
for result, meta in zip(all_results, all_metadata):
    ao_resp = result.responses[0] if result.responses else "(no response)"
    print(f"  [{meta['mode']}]")
    print(f"    {ao_resp}")

# %%
# === Batch spot-check: verify stored responses for several entries ===

import random

random.seed(42)

num_to_check = min(1, len(entries))
sample_indices = random.sample(range(len(entries)), num_to_check)

model.set_adapter("default")

mismatches = 0
for idx in sample_indices:
    e = entries[idx]
    messages = [
        {"role": "system", "content": e["system_prompt"]},
        {"role": "user", "content": e["user_prompt"]},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    input_ids = tokenizer.encode(formatted, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=1024, do_sample=False)

    new_tokens = output[0][input_ids.shape[1] :]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    match = response_text.strip() == e["assistant_response"].strip()
    status = "✓" if match else "✗ MISMATCH"
    if not match:
        mismatches += 1
    print(f"  {status}  [{e['id']}] {e['category']}")

print(f"\n{mismatches}/{num_to_check} mismatches")

# %%
# === Batch spot-check: run AO + LLM judge on several entries ===
# All AO inference is batched into a single run_verbalizer call.
# LLM judge scores are run async after.

import asyncio
import random

import anthropic
import nest_asyncio

nest_asyncio.apply()

random.seed(123)

BATCH_MODE = "system_only"  # <-- change this
BATCH_MODE = "assistant_only"
BATCH_MODE = "user_only"
num_to_check = min(10, len(entries))
sample_indices = random.sample(range(len(entries)), num_to_check)
sample_entries = [entries[idx] for idx in sample_indices]

# Build all prompt_infos in one batch
all_prompt_infos = []
all_metadata = []
for e in sample_entries:
    pis, metas = build_prompt_infos_for_mode([e], BATCH_MODE, tokenizer, verbalizer_prompts=VERBALIZER_PROMPTS)
    all_prompt_infos.extend(pis)
    all_metadata.extend(metas)

# Single batched verbalizer call
all_results = base_experiment.run_verbalizer(
    model=model,
    tokenizer=tokenizer,
    verbalizer_prompt_infos=all_prompt_infos,
    verbalizer_lora_path=adapter_name,
    target_lora_path=None,
    config=eval_config,
    device=device,
)

# Collect AO responses
ao_responses = []
for result in all_results:
    ao_responses.append(result.responses[0] if result.responses else "(no response)")

# Run LLM judge on all responses async
client = anthropic.AsyncAnthropic()
semaphore = asyncio.Semaphore(20)


async def judge_all():
    tasks = [
        judge_single_response(
            client=client,
            ground_truth=e["ground_truth"],
            ao_response=ao_resp,
            semaphore=semaphore,
        )
        for e, ao_resp in zip(sample_entries, ao_responses)
    ]
    return await asyncio.gather(*tasks)


judge_results = asyncio.run(judge_all())

# Print everything
for idx, e, ao_resp, judge_result in zip(sample_indices, sample_entries, ao_responses, judge_results):
    print(f"\n{'=' * 60}")
    print(f"Entry {idx}  |  {e['id']}  |  {e['category']}  |  mode={BATCH_MODE}")
    print(f"Ground truth: {e['ground_truth']}")
    print(f"AO response: {ao_resp}")
    print(
        f"Judge ({JUDGE_MODEL}): specificity={judge_result['specificity']}  correctness={judge_result['correctness']}"
    )
    print(f"Judge reasoning: {judge_result['reasoning']}")

# %%
