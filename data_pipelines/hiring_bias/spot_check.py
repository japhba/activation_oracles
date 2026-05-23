# %%
"""
Interactive spot-check notebook for the hiring bias eval dataset.

Run cells top-to-bottom. Inspect the exact tokenization and segment
positions that get fed to the AO. All outputs printed in full.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_NAME = "Qwen/Qwen3-8B"
DATASET_PATH = REPO_ROOT / "data_pipelines/hiring_bias/Qwen3-8B/hiring_bias_eval_dataset.json"

# %%
# === Load dataset ===

data = json.loads(DATASET_PATH.read_text())
entries = data["entries"]
print(f"Total entries: {len(entries)}")
print(f"Metadata:\n{json.dumps(data['metadata'], indent=2)}")

# %%
# === Overview: spread distribution and source breakdown ===

from collections import defaultdict
import numpy as np

by_group = defaultdict(list)
for e in entries:
    by_group[e["group_key"]].append(e)

spreads = {k: v[0]["group_spread"] for k, v in by_group.items()}
spreads_arr = np.array(list(spreads.values()))

print(f"Total groups: {len(by_group)}")
print(f"Spread: mean={spreads_arr.mean():.4f}, median={np.median(spreads_arr):.4f}, max={spreads_arr.max():.4f}")
for thresh in [0.05, 0.10, 0.15, 0.20]:
    print(f"  spread > {thresh}: {(spreads_arr > thresh).sum()}/{len(spreads_arr)}")

# Per-source
for source in ["resume", "anthropic_explicit", "anthropic_implicit"]:
    source_groups = {k: v for k, v in spreads.items() if source.replace("anthropic_", "") in k or k.startswith(source.split("_")[0])}
    # More precise filtering
    if source == "resume":
        source_groups = {k: v for k, v in spreads.items() if k.startswith("resume")}
    elif source == "anthropic_explicit":
        source_groups = {k: v for k, v in spreads.items() if "explicit" in k}
    elif source == "anthropic_implicit":
        source_groups = {k: v for k, v in spreads.items() if "implicit" in k}
    n = len(source_groups)
    n_high = sum(1 for v in source_groups.values() if v > 0.10)
    source_entries = [e for e in entries if e["source"] == source]
    print(f"\n  {source}: {n} groups, {n_high} with spread > 0.10, {len(source_entries)} entries")

# %%
# === Load eval module and tokenizer ===

import torch
from nl_probes.utils.common import load_tokenizer
from nl_probes.open_ended_eval.hiring_bias import (
    load_hiring_bias_dataset,
    build_hiring_bias_verbalizer_prompt_infos,
    VERBALIZER_PROMPTS,
)
from nl_probes.base_experiment import tokenize_chat_messages, compute_segment_positions

tokenizer = load_tokenizer(MODEL_NAME)

# %%
# === Load eval entries (the filtered, balanced set) ===

eval_entries = load_hiring_bias_dataset(MODEL_NAME, min_wm_bf_diff=0.10)

print(f"\nFirst 10 eval entries:")
for i, e in enumerate(eval_entries[:10]):
    print(
        f"  [{i:3d}] {e['source']:20s} {e['race']:>6s} {e['gender']:>7s}  "
        f"P(Yes)={e['yes_prob']:.4f}  spread={e['group_spread']:.4f}  "
        f"biased={e['biased']}  answer={e['assistant_text']}"
    )

# %%
# === Pick an entry and inspect the raw data ===

IDX = 0  # <-- change this

entry = eval_entries[IDX]
print(f"Entry {IDX}")
print(f"  id:             {entry['id']}")
print(f"  source:         {entry['source']}")
print(f"  group_key:      {entry['group_key']}")
print(f"  group_spread:   {entry['group_spread']:.4f}")
print(f"  race:           {entry['race']}")
print(f"  gender:         {entry['gender']}")
print(f"  name:           {entry['name']}")
print(f"  biased:         {entry['biased']}")
print(f"  yes_prob:       {entry['yes_prob']:.6f}")
print(f"  no_prob:        {entry['no_prob']:.6f}")
print(f"  generated_token:{entry['generated_token']}")
print(f"  assistant_text: {entry['assistant_text']}")

print(f"\n--- USER PROMPT (full) ---")
print(entry["user_prompt"])

# %%
# === Show all demographic variants for this group ===

group_entries = [e for e in entries if e["group_key"] == entry["group_key"]]
print(f"\nAll variants for group {entry['group_key']}:")
for e in sorted(group_entries, key=lambda x: x["yes_prob"], reverse=True):
    marker = " <-- THIS ENTRY" if e["id"] == entry["id"] else ""
    print(
        f"  {e['race']:>6s} {e['gender']:>7s}  "
        f"P(Yes)={e['yes_prob']:.6f}  gen={e['generated_token']}{marker}"
    )

# %%
# === Inspect the exact tokenization fed to the AO ===
# This is Pattern C from docs/open_ended_eval_tokenization_strategies.md:
# user prompt + assistant text, continue_final_message=True, enable_thinking=False

messages = [
    {"role": "user", "content": entry["user_prompt"]},
    {"role": "assistant", "content": entry["assistant_text"]},
]

token_ids = tokenize_chat_messages(
    tokenizer,
    messages,
    add_generation_prompt=False,
    continue_final_message=True,
    enable_thinking=False,
)

positions = compute_segment_positions(len(token_ids), start_idx=0)
positions_set = set(positions)

print(f"Total tokens: {len(token_ids)}")
print(f"Segment positions: {len(positions)} tokens (start_idx=0 → all tokens)")
print(f"Segment range: [{min(positions)}, {max(positions)}]")

# %%
# === Token-by-token visualization ===
# Shows each token with >>> marking segment positions.

print(f"\nToken visualization for entry {IDX}")
print("-" * 60)
for i, token_id in enumerate(token_ids):
    token_str = tokenizer.decode([token_id]).replace("\n", "\\n").replace("\r", "\\r")
    marker = ">>>" if i in positions_set else "   "
    print(f"  [{i:4d}] {marker} {token_str!r}")
print("-" * 60)

# %%
# === Decoded views: full prompt and segment ===

print(f"\n--- Full context decoded ({len(token_ids)} tokens) ---")
print(tokenizer.decode(token_ids))

segment_token_ids = [token_ids[p] for p in positions]
print(f"\n--- Segment decoded ({len(positions)} tokens) ---")
print(tokenizer.decode(segment_token_ids))

# %%
# === Verify this matches Pattern C from the tokenization doc ===
# Expected:
# <|im_start|>user
# {user_prompt}<|im_end|>
# <|im_start|>assistant
# <think>
#
# </think>
#
# {assistant_text}

decoded = tokenizer.decode(token_ids)
print("\n--- Tokenization structure check ---")

expected_parts = [
    "<|im_start|>user",
    "<|im_end|>",
    "<|im_start|>assistant",
    "<think>",
    "</think>",
]

for part in expected_parts:
    if part in decoded:
        print(f"  FOUND: {part}")
    else:
        print(f"  MISSING: {part}")

# Check that assistant_text appears at the end
if decoded.rstrip().endswith(entry["assistant_text"]):
    print(f"  OK: assistant_text '{entry['assistant_text']}' is at the end")
else:
    print(f"  WARNING: assistant_text '{entry['assistant_text']}' NOT at the end")
    print(f"  Last 50 chars of decoded: {decoded[-50:]!r}")

# Check there's no trailing <|im_end|> (should be open, not closed)
if decoded.rstrip().endswith("<|im_end|>"):
    print("  WARNING: context ends with <|im_end|> — turn is closed, should be open")
else:
    print("  OK: context does not end with <|im_end|> — turn is open (continue_final_message=True)")

# %%
# === Build VerbalizerInputInfos and inspect ===

prompt_infos, entry_metadata = build_hiring_bias_verbalizer_prompt_infos(
    [entry],
    VERBALIZER_PROMPTS,
    tokenizer,
)

print(f"Generated {len(prompt_infos)} VerbalizerInputInfos for 1 entry × {len(VERBALIZER_PROMPTS)} prompts")
for i, (info, meta) in enumerate(zip(prompt_infos, entry_metadata)):
    print(f"\n  [{i}] prompt_name={meta['prompt_name']}")
    print(f"      ground_truth={info.ground_truth}")
    print(f"      context_token_ids length={len(info.context_token_ids)}")
    print(f"      positions length={len(info.positions)}")
    print(f"      verbalizer_prompt={info.verbalizer_prompt}")

# %%
# === Run AO inference on this entry (optional — requires GPU + model load) ===
# Uncomment below to actually run AO inference.

# from nl_probes.utils.common import load_model
# from nl_probes.open_ended_eval.eval_runner import (
#     build_verbalizer_eval_config,
#     ensure_default_adapter,
#     build_yes_no_candidate_token_groups,
# )
# import nl_probes.base_experiment as base_experiment
#
# torch.set_grad_enabled(False)
# device = torch.device("cuda")
# model = load_model(MODEL_NAME, torch.bfloat16)
# model.eval()
# ensure_default_adapter(model)
#
# verbalizer_lora = "adamkarvonen/checkpoints_latentqa_cls_on_policy_Qwen3-8B"
# adapter_name, training_config = base_experiment.load_oracle_adapter(model, verbalizer_lora)
# eval_config = build_verbalizer_eval_config(
#     model_name=MODEL_NAME,
#     training_config=training_config,
#     eval_batch_size=32,
#     generation_kwargs={"do_sample": False, "max_new_tokens": 1},
# )
#
# candidate_token_groups = build_yes_no_candidate_token_groups(tokenizer)
# binary_results = base_experiment.run_verbalizer_binary_score(
#     model=model,
#     tokenizer=tokenizer,
#     verbalizer_prompt_infos=prompt_infos,
#     verbalizer_lora_path=adapter_name,
#     target_lora_path=None,
#     config=eval_config,
#     device=device,
#     candidate_token_groups=candidate_token_groups,
# )
#
# import math
# for i, result in enumerate(binary_results):
#     yes_score = result.candidate_scores["yes"]
#     no_score = result.candidate_scores["no"]
#     margin = yes_score - no_score
#     prediction = "yes (biased)" if margin >= 0 else "no (not biased)"
#     print(f"  [{i}] yes_score={yes_score:.4f}  no_score={no_score:.4f}  margin={margin:.4f}  → {prediction}")
#     print(f"       ground_truth={prompt_infos[i].ground_truth}")
