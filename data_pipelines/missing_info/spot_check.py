# %%
"""
Interactive spot-check notebook for the missing information eval.

Run cells top-to-bottom. Pick a problem, then compare A vs B vs C conditions.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_PATH = REPO_ROOT / "data_pipelines/missing_info/missing_info_eval_dataset.json"
EVAL_RESULTS_PATH = (
    REPO_ROOT
    / "experiments/missing_info_eval_results/Qwen3-8B/missing_info_checkpoints_latentqa_cls_on_policy_Qwen3-8B.json"
)

MODEL_NAME = "Qwen/Qwen3-8B"

# %%
# === Load model + AO adapter ===

import torch
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.open_ended_eval.missing_info import (
    build_missing_info_verbalizer_prompt_infos,
    VERBALIZER_PROMPTS,
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

# %%
# Load dataset and eval results

data = json.loads(DATASET_PATH.read_text())
entries = data["entries"]
problem_ids = sorted(set(e["problem_id"] for e in entries))

print(f"Loaded {len(entries)} entries across {len(problem_ids)} problems")
print(f"Problems: {problem_ids}")

eval_data = None
if EVAL_RESULTS_PATH.exists():
    eval_data = json.loads(EVAL_RESULTS_PATH.read_text())
    scored = eval_data.get("scored_results", [])
    print(f"Loaded eval results: {len(scored)} scored entries")
else:
    print("No eval results found")

# %%
# === Pick a problem ===

IDX = 1  # <-- change this (0-9)
pid = problem_ids[IDX]

problem_entries = [e for e in entries if e["problem_id"] == pid]
a_entry = [e for e in problem_entries if e["condition"] == "A_complete"][0]
b_entry = [e for e in problem_entries if e["condition"] == "B_incomplete"][0]
c_entry = [e for e in problem_entries if e["condition"] == "C_forced"][0]

print(f"Problem: {pid}")
print(f"\n--- Complete prompt (A) ---")
print(a_entry["problem_text"])
print(f"\n--- Incomplete prompt (B/C) ---")
print(b_entry["problem_text"])
print(f"\n--- Missing info ---")
print(a_entry["missing_info_description"])
print(f"\n--- Neutral segment ({a_entry['neutral_segment_token_count']} tokens) ---")
print(a_entry["neutral_segment"])
print(f"\n--- A: full_reasoning ({len(a_entry['full_reasoning'])} chars) ---")
print(a_entry["full_reasoning"][:500] + "..." if len(a_entry["full_reasoning"]) > 500 else a_entry["full_reasoning"])
print(f"\n--- B: full_reasoning ({len(b_entry['full_reasoning'])} chars) ---")
print(b_entry["full_reasoning"][:500] + "..." if len(b_entry["full_reasoning"]) > 500 else b_entry["full_reasoning"])
print(f"\n--- Teacher-forced segment (same for A and C) ---")
print(a_entry["teacher_forced_segment"])

# %%
# === Show exact verbalizer inputs (prompt_infos + metadata) for each condition ===
# This is exactly what goes into run_verbalizer(). For each condition we build
# the prompt_infos and metadata, then display:
#   1. The full decoded target prompt (context_prompt through chat template)
#   2. The verbalizer prompt and ground truth
#   3. Token-by-token visualization with >>> on the AO segment


for condition_label, cond_entry in [("A", a_entry), ("B", b_entry), ("C", c_entry)]:
    prompt_infos, metadata = build_missing_info_verbalizer_prompt_infos([cond_entry], VERBALIZER_PROMPTS, tokenizer)

    print(f"\n{'=' * 80}")
    print(f"CONDITION {condition_label} ({cond_entry['condition']})")
    print(f"  {len(prompt_infos)} prompt_infos (one per verbalizer prompt variant)")
    print("=" * 80)

    for pi, meta in zip(prompt_infos, metadata):
        print(f"\n  --- prompt_name: {meta['prompt_name']} ---")
        print(f"  ground_truth: {pi.ground_truth}")
        print(f"  ground_truth_missing_info: {meta['ground_truth_missing_info']}")
        print(f"  verbalizer_prompt: {pi.verbalizer_prompt}")

    # Token visualization using pre-tokenized data
    info = prompt_infos[0]
    num_tokens = len(info.context_token_ids)
    positions_set = set(info.positions)

    print(f"\n  --- Encoded target prompt: {num_tokens} tokens ---")
    print(f"\n  --- Token visualization (segment: {len(info.positions)} tokens) ---")
    for i, token_id in enumerate(info.context_token_ids):
        token_str = tokenizer.decode([token_id]).replace("\n", "\\n").replace("\r", "\\r")
        if i in positions_set:
            print(f"  [{i:3d}] >>> {token_str}")
        else:
            print(f"  [{i:3d}]     {token_str}")

    segment_token_ids = [info.context_token_ids[p] for p in info.positions]
    print(f"\n  --- AO segment decoded ({len(info.positions)} tokens) ---")
    print(tokenizer.decode(segment_token_ids))
    print(f"\n  --- Full prompt decoded ({num_tokens} tokens) ---")
    print(tokenizer.decode(info.context_token_ids))

# %%
# === Show eval results for this problem ===

if eval_data is not None:
    scored = eval_data.get("scored_results", [])
    matching = [sr for sr in scored if sr.get("problem_id") == pid]
    if matching:
        print(f"--- Eval Results for {pid} ---")
        for sr in matching:
            status = "✓" if sr["is_correct"] else "✗"
            print(
                f"  {status} {sr['condition']:<15} [{sr['prompt_name']:<15}] "
                f"ao={sr['ao_prediction']!s:<6} expected={sr['expected']!s:<6} "
                f"response: {sr['ao_response'][:60]}"
            )

        # Highlight A vs C
        for prompt in set(sr["prompt_name"] for sr in matching):
            a_result = [sr for sr in matching if sr["condition"] == "A_complete" and sr["prompt_name"] == prompt]
            c_result = [sr for sr in matching if sr["condition"] == "C_forced" and sr["prompt_name"] == prompt]
            if a_result and c_result:
                same = a_result[0]["ao_prediction"] == c_result[0]["ao_prediction"]
                print(f"\n  [{prompt}] A vs C: {'SAME (token-based)' if same else 'DIFFERENT (activation-based)'}")
    else:
        print(f"No eval results for {pid}")

# %%
# === Run AO on all 3 conditions for current problem ===

for condition_label, entry in [("A", a_entry), ("B", b_entry), ("C", c_entry)]:
    prompt_infos, metadata = build_missing_info_verbalizer_prompt_infos([entry], VERBALIZER_PROMPTS, tokenizer)

    results = base_experiment.run_verbalizer(
        model=model,
        tokenizer=tokenizer,
        verbalizer_prompt_infos=prompt_infos,
        verbalizer_lora_path=adapter_name,
        target_lora_path=None,
        config=eval_config,
        device=device,
    )

    print(f"\n--- Condition {condition_label} ({entry['condition']}) ---")
    print(f"Problem: {entry['problem_text'][:100]}...")
    teacher = entry.get("teacher_forced_segment", "")
    print(f"Reasoning: {entry['full_reasoning'][:80]}...")
    print(f"Teacher-forced: {teacher[:80]}..." if teacher else "Teacher-forced: (none)")
    for result, meta in zip(results, metadata):
        ao_resp = result.responses[0] if result.responses else "(no response)"
        print(f"  [{meta['prompt_name']}] AO: {ao_resp}")

# %%
# === Visualize token overlap between A and C ===

# Use pre-tokenized data from prompt_infos
a_infos, _ = build_missing_info_verbalizer_prompt_infos([a_entry], VERBALIZER_PROMPTS, tokenizer)
c_infos, _ = build_missing_info_verbalizer_prompt_infos([c_entry], VERBALIZER_PROMPTS, tokenizer)

a_tokens = a_infos[0].context_token_ids
c_tokens = c_infos[0].context_token_ids
a_positions = a_infos[0].positions
c_positions = c_infos[0].positions

# Compare segment tokens (what the AO actually reads)
a_seg = [a_tokens[p] for p in a_positions]
c_seg = [c_tokens[p] for p in c_positions]

matching = sum(1 for a, c in zip(a_seg, c_seg) if a == c)
print(f"A total tokens: {len(a_tokens)}")
print(f"C total tokens: {len(c_tokens)}")
print(f"Segment tokens: A={len(a_seg)}, C={len(c_seg)}")
print(f"Matching: {matching}/{min(len(a_seg), len(c_seg))}")
print()

print("A segment tokens (last 20):")
for i, tid in enumerate(a_seg[-20:]):
    print(f"  [{i}] {tokenizer.decode([tid])!r}")
print("\nC segment tokens (last 20):")
for i, tid in enumerate(c_seg[-20:]):
    print(f"  [{i}] {tokenizer.decode([tid])!r}")

# %%
