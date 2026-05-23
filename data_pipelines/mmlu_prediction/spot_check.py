# %%
"""
Interactive spot-check notebook for the MMLU prediction eval.

Run cells top-to-bottom. Pick an entry, then run the verification or AO cells
to spot-check that the pipeline reproduces roughly the same results.

Single model load — Qwen3-8B is loaded once and used for both answer
verification (base weights, greedy) and AO inference (LoRA adapter).
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_PATH = REPO_ROOT / "data_pipelines/mmlu_prediction/Qwen3-8B/mmlu_prediction_eval_dataset.json"
EVAL_RESULTS_PATHS = {
    "pre_answer": (
        REPO_ROOT
        / "experiments/mmlu_prediction_pre_answer_seg10_eval_results/Qwen3-8B/"
        "mmlu_prediction_binary_checkpoints_latentqa_cls_on_policy_Qwen3-8B.json"
    ),
    "post_answer": (
        REPO_ROOT
        / "experiments/mmlu_prediction_post_answer_seg10_eval_results/Qwen3-8B/"
        "mmlu_prediction_binary_checkpoints_latentqa_cls_on_policy_Qwen3-8B.json"
    ),
}

MODEL_NAME = "Qwen/Qwen3-8B"

ANSWER_LETTERS = ["A", "B", "C", "D"]

# %%
# === Load model + AO adapter ===

import torch
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.open_ended_eval.mmlu_prediction import (
    AO_VERBALIZER_PROMPTS,
    build_mmlu_prediction_verbalizer_prompt_infos,
)
from nl_probes.open_ended_eval.eval_runner import (
    build_verbalizer_eval_config,
    ensure_default_adapter,
    extract_yes_no,
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


def format_mmlu_question(question: str, choices: list[str]) -> str:
    lines = [question, ""]
    for i, choice in enumerate(choices):
        lines.append(f"{ANSWER_LETTERS[i]}. {choice}")
    lines.append("")
    lines.append("Answer with just the letter (A, B, C, or D), nothing else.")
    return "\n".join(lines)


def extract_answer_letter(text: str) -> str | None:
    text = text.strip()
    if text and text[0] in "ABCD":
        return text[0]
    match = re.search(r"\b([A-D])\b", text)
    if match:
        return match.group(1)
    return None


# %%
# Load dataset and eval results

data = json.loads(DATASET_PATH.read_text())
entries = data["entries"]

print(f"Loaded {len(entries)} entries")
print(f"Correct: {sum(1 for e in entries if e['model_correct'])}")
print(f"Incorrect: {sum(1 for e in entries if not e['model_correct'])}")

eval_data_by_mode = {}
for mode_name, path in EVAL_RESULTS_PATHS.items():
    if path.exists():
        eval_data_by_mode[mode_name] = json.loads(path.read_text())
        scored = eval_data_by_mode[mode_name].get("scored_results", [])
        print(f"Loaded {mode_name} eval results: {len(scored)} scored entries")
    else:
        print(f"No {mode_name} eval results found at {path}")

# %%
# === Pick an entry ===
# Change this index to explore different entries.

IDX = 0  # <-- change this

entry = entries[IDX]
print(f"Entry {IDX}  |  id={entry['id']}  |  subject={entry['subject']}")
print(f"Question: {entry['question']}")
print(f"Choices: {entry['choices']}")
print(f"Correct answer: {entry['correct_answer_letter']}")
print(f"Model answer: {entry['model_answer_letter']}  (correct: {entry['model_correct']})")
print(f"Model raw response: {entry['model_raw_response']}")

# %%
# === Show saved eval result (if available) ===

if eval_data_by_mode:
    any_matching = False
    for mode_name, eval_data in eval_data_by_mode.items():
        scored = eval_data["binary_scored_results"]
        matching = [sr for sr in scored if sr["id"] == entry["id"]]
        if not matching:
            continue
        any_matching = True
        print(f"--- Saved {mode_name} Eval Results for {entry['id']} ({len(matching)} prompt variants) ---")
        for sr in matching:
            status = "✓" if sr["is_correct"] else "✗"
            print(
                f"  {status} prompt={sr['prompt_name']:<20}  "
                f"predicted={sr['predicted_answer']!s:<8}  model_correct={sr['model_correct']!s:<6}  "
                f"yes={sr['yes_score']:.3f}  no={sr['no_score']:.3f}  margin={sr['margin_yes_minus_no']:.3f}"
            )
    if not any_matching:
        print(f"No saved eval result for entry {entry['id']}")
else:
    print("No eval results loaded")

# %%
# === Verify model answer ===
# Runs the model at temperature 0 on the MMLU question to confirm the stored answer.

model.set_adapter("default")

question_text = format_mmlu_question(entry["question"], entry["choices"])
messages = [{"role": "user", "content": question_text}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
input_ids = tokenizer.encode(formatted, return_tensors="pt", add_special_tokens=False).to(device)

with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=10, do_sample=False)

new_tokens = output[0][input_ids.shape[1] :]
response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
fresh_answer = extract_answer_letter(response_text)

print(f"Question: {entry['question'][:120]}")
print(f"Correct answer: {entry['correct_answer_letter']}")
print(f"Stored model answer: {entry['model_answer_letter']}")
print(f"Fresh model answer: {fresh_answer}")
print(f"Fresh raw response: {response_text}")
print(f"Stored matches fresh: {fresh_answer == entry['model_answer_letter']}")

# %%
# === Show exact verbalizer inputs (prompt_infos + metadata) ===
# This is exactly what goes into run_verbalizer(). Displays:
#   1. Each prompt variant's verbalizer_prompt, ground_truth, and context_prompt
#   2. The full decoded target prompt (context_prompt through chat template)
#   3. Token-by-token visualization with >>> on the AO segment

prompt_infos, metadata = build_mmlu_prediction_verbalizer_prompt_infos([entry], AO_VERBALIZER_PROMPTS, tokenizer)

print(f"Entry {IDX}  |  id={entry['id']}  |  {len(prompt_infos)} prompt_infos")
print("=" * 80)

for pi, meta in zip(prompt_infos, metadata):
    print(f"\n  --- prompt_name: {meta['prompt_name']} ---")
    print(f"  ground_truth: {pi.ground_truth}")
    print(f"  model_correct: {meta['model_correct']}")
    print(f"  verbalizer_prompt: {pi.verbalizer_prompt}")

# Show full decoded target prompt + token visualization for each mode
for info, meta in zip(prompt_infos, metadata):
    mode_label = meta["prompt_name"]
    decoded = tokenizer.decode(info.context_token_ids, skip_special_tokens=False)
    num_tokens = len(info.context_token_ids)
    positions_set = set(info.positions)

    print(f"\n{'=' * 40} {mode_label} {'=' * 40}")
    print(f"--- Full decoded target prompt ({num_tokens} tokens) ---")
    print(decoded)

    # Token visualization with AO segment markers
    print(f"\n--- Token visualization (segment: {len(info.positions)} tokens) ---")
    for i, token_id in enumerate(info.context_token_ids):
        token_str = tokenizer.decode([token_id]).replace("\n", "\\n").replace("\r", "\\r")
        if i in positions_set:
            print(f"  [{i:3d}] >>> {token_str}")
        else:
            print(f"  [{i:3d}]     {token_str}")

    segment_token_ids = [info.context_token_ids[p] for p in info.positions]
    print(f"\n--- AO segment decoded ({len(info.positions)} tokens) ---")
    print(tokenizer.decode(segment_token_ids))

# %%
# === Run Activation Oracle on current entry ===

prompt_infos, metadata = build_mmlu_prediction_verbalizer_prompt_infos([entry], AO_VERBALIZER_PROMPTS, tokenizer)

results = base_experiment.run_verbalizer(
    model=model,
    tokenizer=tokenizer,
    verbalizer_prompt_infos=prompt_infos,
    verbalizer_lora_path=adapter_name,
    target_lora_path=None,
    config=eval_config,
    device=device,
)

print(f"\nQuestion: {entry['question']}")
print(
    f"Model answer: {entry['model_answer_letter']}  |  Correct: {entry['correct_answer_letter']}  |  Model correct: {entry['model_correct']}"
)
for result, meta in zip(results, metadata):
    ao_resp = result.responses[0] if result.responses else "(no response)"
    ao_pred = extract_yes_no(ao_resp)
    expected = "yes" if entry["model_correct"] else "no"
    status = "✓" if ao_pred == expected else "✗"
    print(f"  {status} [{meta['prompt_name']}] AO says: {ao_resp}")

# %%
# === Batch spot-check: verify stored model answers ===

import random

random.seed(42)

num_to_check = 10
sample_indices = random.sample(range(len(entries)), num_to_check)

model.set_adapter("default")

mismatches = 0
for idx in sample_indices:
    e = entries[idx]
    question_text = format_mmlu_question(e["question"], e["choices"])
    messages = [{"role": "user", "content": question_text}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    input_ids = tokenizer.encode(formatted, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=10, do_sample=False)

    new_tokens = output[0][input_ids.shape[1] :]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    fresh_answer = extract_answer_letter(response_text)

    match = fresh_answer == e["model_answer_letter"]
    status = "✓" if match else "✗ MISMATCH"
    if not match:
        mismatches += 1
    print(
        f"  {status}  [{e['id']}] {e['subject']:<30} stored={e['model_answer_letter']!s:<4} fresh={fresh_answer!s:<4}"
    )

print(f"\n{mismatches}/{num_to_check} mismatches")

# %%
# === Batch spot-check: run AO on several random entries ===

random.seed(123)

num_to_check = 5
sample_indices = random.sample(range(len(entries)), num_to_check)

for idx in sample_indices:
    e = entries[idx]
    prompt_infos, metadata = build_mmlu_prediction_verbalizer_prompt_infos([e], AO_VERBALIZER_PROMPTS, tokenizer)

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
        f"Entry {idx}  |  {e['subject']}  |  model={e['model_answer_letter']}  "
        f"correct={e['correct_answer_letter']}  model_correct={e['model_correct']}"
    )
    print(f"Q: {e['question']}")
    for result, meta in zip(results, metadata):
        ao_resp = result.responses[0] if result.responses else "(no response)"
        print(f"  [{meta['prompt_name']}] {ao_resp[:80]}")

# %%
