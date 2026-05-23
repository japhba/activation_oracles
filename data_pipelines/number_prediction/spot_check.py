# %%
"""
Interactive spot-check notebook for the number prediction eval.

Run cells top-to-bottom. Pick an entry, then run the verification or AO cells
to spot-check that the pipeline reproduces roughly the same results.

Single model load — Qwen3-8B is loaded once and used for both answer
verification (base weights, greedy) and AO inference (LoRA adapter).
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_PATH = REPO_ROOT / "data_pipelines/number_prediction/Qwen3-8B/number_prediction_eval_dataset.json"
EVAL_RESULTS_PATH = (
    REPO_ROOT
    / "experiments/number_prediction_eval_results/Qwen3-8B/number_prediction_checkpoints_latentqa_cls_on_policy_Qwen3-8B.json"
)

MODEL_NAME = "Qwen/Qwen3-8B"


def extract_number(text):
    match = re.search(r"-?\d+", text.strip())
    return int(match.group()) if match else None


# %%
# === Load model + AO adapter ===

import torch
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.open_ended_eval.number_prediction import (
    build_number_prediction_text_baseline_inputs,
    build_number_prediction_verbalizer_prompt_infos,
    extract_number_from_response,
    VERBALIZER_PROMPTS,
)
from nl_probes.open_ended_eval.eval_runner import (
    build_verbalizer_eval_config,
    ensure_default_adapter,
)
import nl_probes.base_experiment as base_experiment

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
    generation_kwargs={"do_sample": False, "temperature": 0.0, "max_new_tokens": 150},
)
print(f"Model + AO loaded: adapter={adapter_name}")

# %%
# Load dataset and eval results

data = json.loads(DATASET_PATH.read_text())
entries = data["entries"]

print(f"Loaded {len(entries)} entries")

eval_data = None
if EVAL_RESULTS_PATH.exists():
    eval_data = json.loads(EVAL_RESULTS_PATH.read_text())
    scored = eval_data.get("scored_results", [])
    print(f"Loaded eval results: {len(scored)} scored entries")
else:
    print("No eval results found")

# %%
# === Pick an entry ===
# Change this index to explore different entries.

IDX = 5  # <-- change this

entry = entries[IDX]
print(f"Entry {IDX}  |  id={entry['id']}  |  category={entry['category']}")
print(f"Expression: {entry['expression']}")
print(f"True answer: {entry['true_answer']}  ({entry['true_answer_num_tokens']} tokens)")
print(f"Model answer: {entry['model_answer']}  (correct: {entry['model_correct']})")
print(f"Model raw response: {entry['model_raw_response']}")
print(f"Single-token answer: {entry['is_single_token_answer']}")

# %%
# === Show saved eval result (if available) ===

if eval_data is not None:
    scored = eval_data.get("scored_results", [])
    matching = [sr for sr in scored if sr.get("id") == entry["id"]]
    if matching:
        print(f"--- Saved Eval Results for {entry['id']} ({len(matching)} prompt variants) ---")
        for sr in matching:
            status = "✓" if sr["matches_model_answer"] else "✗"
            print(
                f"  {status} prompt={sr['prompt_name']:<10}  "
                f"predicted_number={sr['predicted_number']!s:<8}  model={sr['model_answer']!s:<8}  "
                f"response_text: {sr['response_text']}"
            )
    else:
        print(f"No saved eval result for entry {entry['id']}")
else:
    print("No eval results loaded")

# %%
# === Verify model answer ===
# Runs the model at temperature 0 on the expression to confirm the stored answer.
# Re-run the "Pick an entry" cell above to change which entry is tested.

model.set_adapter("default")

messages = [{"role": "user", "content": f"What is {entry['expression']}? Answer with just the number, nothing else."}]
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
        max_new_tokens=50,
        temperature=1.0,  # ignored with do_sample=False
        do_sample=False,
    )

new_tokens = output[0][input_ids.shape[1] :]
response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
fresh_answer = extract_number(response_text)

print(f"Expression: {entry['expression']}")
print(f"True answer: {entry['true_answer']}")
print(f"Stored model answer: {entry['model_answer']}")
print(f"Fresh model answer: {fresh_answer}")
print(f"Fresh raw response: {response_text}")
print(f"Stored matches fresh: {fresh_answer == entry['model_answer']}")

# %%
# === Build prompt infos for current entry ===
# Built once and reused for both visualization and AO inference below,
# so we're inspecting the exact same data that run_verbalizer sees.

prompt_infos, metadata = build_number_prediction_verbalizer_prompt_infos([entry], VERBALIZER_PROMPTS, tokenizer)

# %%
# === Inspect text baseline prompts ===
# Prints the exact rendered prompt text sent to the baseline model.

baseline_inputs, baseline_metadata = build_number_prediction_text_baseline_inputs(
    [entry],
    VERBALIZER_PROMPTS,
    tokenizer,
)

for baseline_input, meta in zip(baseline_inputs, baseline_metadata):
    print(f"--- prompt={meta['prompt_name']}  variant={meta['baseline_variant']} ---")
    print(baseline_input.prompt_text)
    print()

# %%
# === Visualize selected tokens ===
# Shows which tokens the AO reads activations from (the segment window).
# Tokens marked with >>> are the ones whose activations are fed to the AO.

info = prompt_infos[0]  # all prompt variants share the same token_ids/positions
positions_set = set(info.positions)

print(f"Token selection visualization for: {entry['expression']}")
print(
    f"Total tokens: {len(info.context_token_ids)}  |  Segment: {len(info.positions)} tokens"
)
print("-" * 60)
for i, token_id in enumerate(info.context_token_ids):
    token_str = tokenizer.decode([token_id]).replace("\n", "\\n").replace("\r", "\\r")
    if i in positions_set:
        print(f"  [{i:3d}] >>> {token_str}")
    else:
        print(f"  [{i:3d}]     {token_str}")
print("-" * 60)

segment_token_ids = [info.context_token_ids[p] for p in info.positions]
print(f"\n--- AO segment decoded ({len(info.positions)} tokens) ---")
print(tokenizer.decode(segment_token_ids))
print(f"\n--- Full prompt decoded ({len(info.context_token_ids)} tokens) ---")
print(tokenizer.decode(info.context_token_ids))

# %%
# === Run Activation Oracle on current entry ===
# Uses the same prompt_infos built above — no re-tokenization.

results = base_experiment.run_verbalizer(
    model=model,
    tokenizer=tokenizer,
    verbalizer_prompt_infos=prompt_infos,
    verbalizer_lora_path=adapter_name,
    target_lora_path=None,
    config=eval_config,
    device=device,
)

print(
    f"\nExpression: {entry['expression']}  |  Model answer: {entry['model_answer']}  |  True answer: {entry['true_answer']}"
)
for result, meta in zip(results, metadata):
    ao_resp = result.responses[0] if result.responses else "(no response)"
    ao_num = extract_number_from_response(ao_resp)
    status = "✓" if ao_num == entry["model_answer"] else "✗"
    print(f"  {status} [{meta['prompt_name']}] AO says: {ao_resp}")

# %%
# === Batch spot-check: verify stored model answers for several entries ===

import random

random.seed(42)

num_to_check = 10
sample_indices = random.sample(range(len(entries)), num_to_check)

model.set_adapter("default")

mismatches = 0
for idx in sample_indices:
    e = entries[idx]
    messages = [{"role": "user", "content": f"What is {e['expression']}? Answer with just the number, nothing else."}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    input_ids = tokenizer.encode(formatted, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=50, do_sample=False)

    new_tokens = output[0][input_ids.shape[1] :]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    fresh_answer = extract_number(response_text)

    match = fresh_answer == e["model_answer"]
    status = "✓" if match else "✗ MISMATCH"
    if not match:
        mismatches += 1
    print(f"  {status}  [{e['id']}] {e['expression']:<35} stored={e['model_answer']!s:<10} fresh={fresh_answer!s:<10}")

print(f"\n{mismatches}/{num_to_check} mismatches")

# %%
# === Batch spot-check: run AO on several random entries ===

random.seed(123)

num_to_check = 5
sample_indices = random.sample(range(len(entries)), num_to_check)

for idx in sample_indices:
    e = entries[idx]
    prompt_infos, metadata = build_number_prediction_verbalizer_prompt_infos([e], VERBALIZER_PROMPTS, tokenizer)

    results = base_experiment.run_verbalizer(
        model=model,
        tokenizer=tokenizer,
        verbalizer_prompt_infos=prompt_infos,
        verbalizer_lora_path=adapter_name,
        target_lora_path=None,
        config=eval_config,
        device=device,
    )

    print(f"\n{'=' * 60}")
    print(
        f"Entry {idx}  |  {e['expression']}  |  model={e['model_answer']}  true={e['true_answer']}  |  {e['category']}"
    )
    for result, meta in zip(results, metadata):
        ao_resp = result.responses[0] if result.responses else "(no response)"
        ao_num = extract_number_from_response(ao_resp)
        status = "✓" if ao_num == e["model_answer"] else "✗"
        print(f"  {status} [{meta['prompt_name']}] ao={ao_num!s:<8} | {ao_resp}")

# %%
