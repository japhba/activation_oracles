# %%
"""
Interactive spot-check notebook for the backtracking eval (v2 dataset).

Run cells top-to-bottom. Pick an entry, then run the verification or AO cells
to spot-check that the pipeline reproduces roughly the same results.

Single model load — Qwen3-8B is loaded once and used for both continuation
verification (base weights) and AO inference (LoRA adapter).
"""

import json
import re
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(REPO_ROOT / ".env")

DATASET_PATH = REPO_ROOT / "data_pipelines/backtracking/Qwen3-8B/backtracking_eval_dataset.json"
EVAL_RESULTS_PATH = (
    REPO_ROOT
    / "experiments/backtracking_eval_results/Qwen3-8B/backtracking_checkpoints_latentqa_cls_on_policy_Qwen3-8B.json"
)

MODEL_NAME = "Qwen/Qwen3-8B"
NUM_CONTINUATIONS = 10
MAX_CONTINUATION_TOKENS = 100

BACKTRACK_PATTERNS = [
    r"\b[Ww]ait\b",
    r"\b[Aa]ctually\b",
    r"\b[Nn]o,\s",
    r"\b[Hh]old on\b",
    r"\b[Hh]mm\b",
    r"\blet me re",
    r"\bmistake\b",
    r"\bthat\'s (wrong|incorrect|not right)",
    r"\bthat (doesn\'t|can\'t|won\'t)",
    r"\bcontradiction\b",
    r"\bscratch that\b",
    r"\bcorrection\b",
]


def check_backtrack(text):
    return any(re.search(pat, text[:80], re.IGNORECASE) for pat in BACKTRACK_PATTERNS)


# %%
# === Load model + AO adapter ===
# Loads Qwen3-8B once with the LatentQA AO LoRA adapter.
# The same model is used for both continuation verification and AO inference.

import torch
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.open_ended_eval.backtracking import (
    build_backtracking_text_baseline_inputs,
    build_backtracking_verbalizer_prompt_infos,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    JUDGE_MODEL,
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
    generation_kwargs={
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": 150,
    },
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
    print(f"Loaded eval results: {len(eval_data.get('scored_results', []))} scored entries")
else:
    print("No eval results found")


# %%
# === Pick an entry ===
# Change this index to explore different entries.

IDX = 160  # <-- change this

entry = entries[IDX]
print(f"Entry {IDX}  |  backtrack_rate={entry['backtrack_rate']}  |  domain={entry.get('domain', 'N/A')}")
print(f"Problem ID: {entry.get('problem_id', 'N/A')}")
print(f"\nPROBLEM:\n{entry['problem']}")
print(f"\nPREFIX:\n{entry['prefix']}")
print(f"\nORIGINAL CONTINUATION:\n{entry['original_continuation']}")
print(f"\nUNCERTAINTY DESCRIPTION:\n{entry['uncertainty_description']}")
print(f"\nFILTER PREDICTION:\n{entry.get('filter_prediction', 'N/A')}")
print(f"FILTER CONFIDENCE: {entry.get('filter_confidence', 'N/A')}")
print(f"WHY NOT OBVIOUS: {entry.get('why_not_obvious', 'N/A')}")

# %%
# === Show saved eval result (if available) ===

if eval_data is not None:
    scored = eval_data.get("scored_results", [])
    # Find by result_index matching IDX
    matching = [sr for sr in scored if sr.get("result_index") == IDX]
    if matching:
        sr = matching[0]
        print(f"--- Saved Eval Result (entry {IDX}) ---")
        print(f"  Response: {sr['response_text']}")
        print(f"  Ground Truth: {sr['ground_truth']}")
        print(f"  Specificity: {sr['specificity']}/5")
        print(f"  Correctness: {sr['correctness']}/5")
        print(f"  Judge Reasoning: {sr['reasoning']}")
    else:
        print(f"No saved eval result for entry {IDX}")
else:
    print("No eval results loaded")

# %%
# === Show all stored continuations ===

stored = entry.get("continuations", [])
print(f"All {len(stored)} stored continuations:")
for j, c in enumerate(stored):
    text = c if isinstance(c, str) else c.get("text", str(c))
    has_bt = check_backtrack(text)
    bt_marker = " [BACKTRACK]" if has_bt else ""
    print(f"\n  [{j + 1}]{bt_marker}:")
    print(f"    {text}")

stored_bt_count = sum(1 for c in stored if check_backtrack(c if isinstance(c, str) else c.get("text", str(c))))
print(f"\nStored backtrack rate (regex): {stored_bt_count}/{len(stored)} = {stored_bt_count / len(stored):.1f}")
print(f"Dataset backtrack rate (LLM-judged): {entry['backtrack_rate']}")

# %%
# === Verify backtracking consistency ===
# Generates fresh continuations from the prefix and checks backtrack rate.
# Uses the base model weights (default adapter) for generation.
# Re-run the "Pick an entry" cell above to change which entry is tested.

model.set_adapter("default")

from nl_probes.base_experiment import tokenize_chat_messages

messages = [
    {"role": "user", "content": entry["problem"]},
    {"role": "assistant", "content": entry["prefix"]},
]
token_ids = tokenize_chat_messages(tokenizer, messages, add_generation_prompt=False, continue_thinking=True)
input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
attn_mask = torch.ones_like(input_ids)

# Batch: repeat the same input NUM_CONTINUATIONS times
batch_input_ids = input_ids.repeat(NUM_CONTINUATIONS, 1)
batch_attn_mask = attn_mask.repeat(NUM_CONTINUATIONS, 1)
print(f"Generating {NUM_CONTINUATIONS} continuations in one batch ({input_ids.shape[1]} input tokens each)...")

with torch.no_grad():
    batch_output = model.generate(
        batch_input_ids,
        attention_mask=batch_attn_mask,
        max_new_tokens=MAX_CONTINUATION_TOKENS,
        temperature=1.0,
        top_p=0.95,
        do_sample=True,
    )

continuations = []
backtrack_count = 0
prefix_len = input_ids.shape[1]

for i in range(NUM_CONTINUATIONS):
    new_tokens = batch_output[i][prefix_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    has_bt = check_backtrack(text)
    if has_bt:
        backtrack_count += 1
    continuations.append(text)
    bt_marker = " [BACKTRACK]" if has_bt else ""
    print(f"  [{i + 1}/{NUM_CONTINUATIONS}]{bt_marker}: {text}")

new_rate = backtrack_count / NUM_CONTINUATIONS
print(f"\nFresh backtrack rate: {new_rate:.1f}  (dataset: {entry['backtrack_rate']:.1f})")

# %%
# === Inspect text baseline prompts ===
# Prints the exact rendered prompt text sent to the baseline model.

baseline_inputs, baseline_metadata = build_backtracking_text_baseline_inputs(
    [entry],
    tokenizer=tokenizer,
)

for baseline_input, meta in zip(baseline_inputs, baseline_metadata):
    print(f"--- prompt={meta['prompt_name']}  variant={meta['baseline_variant']} ---")
    print(baseline_input.prompt_text)
    print()

# %%
# === Visualize selected tokens ===
# Shows which tokens the AO reads activations from (the segment window).
# Tokens marked with >>> are the ones whose activations are fed to the AO.

prompt_infos, _ = build_backtracking_verbalizer_prompt_infos([entry], tokenizer=tokenizer)
info = prompt_infos[0]
positions_set = set(info.positions)

print(f"Token selection visualization for entry {IDX}")
print(f"Total tokens: {len(info.context_token_ids)}  |  Segment: {len(info.positions)} tokens")
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
# Re-run the "Pick an entry" cell above to change which entry is tested.

prompt_infos, _ = build_backtracking_verbalizer_prompt_infos([entry], tokenizer=tokenizer)

results = base_experiment.run_verbalizer(
    model=model,
    tokenizer=tokenizer,
    verbalizer_prompt_infos=prompt_infos,
    verbalizer_lora_path=adapter_name,
    target_lora_path=None,
    config=eval_config,
    device=device,
)

result = results[0]
ao_response = result.responses[0] if result.responses else "(no response)"

print(f"AO RESPONSE:\n  {ao_response}")
print(f"\nGROUND TRUTH:\n  {entry['uncertainty_description']}")

# %%
# === Judge the AO response ===
# Calls the Haiku judge to score specificity and correctness.

import anthropic

client = anthropic.Anthropic()

judge_message = JUDGE_USER_TEMPLATE.format(
    prefix=entry["prefix"][-1500:],
    ground_truth=entry["uncertainty_description"],
    ao_response=ao_response,
)

judge_response = client.messages.create(
    model=JUDGE_MODEL,
    max_tokens=300,
    system=JUDGE_SYSTEM_PROMPT,
    messages=[{"role": "user", "content": judge_message}],
)

judge_text = judge_response.content[0].text.strip()
if judge_text.startswith("```"):
    judge_text = judge_text.split("\n", 1)[1]
    judge_text = judge_text.rsplit("```", 1)[0].strip()
judge_result = json.loads(judge_text)

print(f"JUDGE SCORES:")
print(f"  Specificity: {judge_result['specificity']}/5")
print(f"  Correctness: {judge_result['correctness']}/5")
print(f"  Reasoning: {judge_result['reasoning']}")

# %%
# === Batch spot-check: run AO + judge on several random entries ===
# Useful for quickly scanning a handful of entries.

import asyncio
import random

from nl_probes.open_ended_eval.backtracking import judge_single_response

random.seed(42)

num_to_check = 5
sample_indices = random.sample(range(len(entries)), num_to_check)

for idx in sample_indices:
    e = entries[idx]
    prompt_infos, _ = build_backtracking_verbalizer_prompt_infos([e], tokenizer=tokenizer)

    results = base_experiment.run_verbalizer(
        model=model,
        tokenizer=tokenizer,
        verbalizer_prompt_infos=prompt_infos,
        verbalizer_lora_path=adapter_name,
        target_lora_path=None,
        config=eval_config,
        device=device,
    )

    result = results[0]
    ao_resp = result.responses[0] if result.responses else "(no response)"

    # Judge
    semaphore = asyncio.Semaphore(1)

    async def _judge():
        return await judge_single_response(
            client=anthropic.AsyncAnthropic(),
            prefix=e["prefix"],
            ground_truth=e["uncertainty_description"],
            ao_response=ao_resp,
            semaphore=semaphore,
        )

    judge_result = asyncio.run(_judge())

    print(f"\n{'=' * 60}")
    print(f"Entry {idx}  |  rate={e['backtrack_rate']}  |  domain={e.get('domain', 'N/A')}")
    print(f"GT:   {e['uncertainty_description']}")
    print(f"AO:   {ao_resp}")
    print(f"Score: specificity={judge_result['specificity']}/5  correctness={judge_result['correctness']}/5")
    print(f"Judge: {judge_result['reasoning']}")

# %%
