"""Spot check script for hallucination detection dataset.

Interactive notebook-style cells for reviewing dataset quality and
verifying that the AO input pipeline produces correct tokenization.

Usage: Run cells interactively in VS Code / Jupyter.
"""

# %% Imports and setup
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

CONFIG = "Qwen2.5-7B-Instruct"
DATASET_PATH = Path(__file__).resolve().parent / CONFIG / "hallucination_eval_dataset.json"
TARGET_MODEL = "Qwen/Qwen3-8B"

# %% Load dataset
print(f"Loading dataset from {DATASET_PATH}")
dataset = json.loads(DATASET_PATH.read_text())
metadata = dataset["metadata"]
entries = dataset["entries"]

print(f"\nMetadata:")
for k, v in metadata.items():
    print(f"  {k}: {v}")
print(f"\nTotal entries: {len(entries)}")

# %% Label distribution and basic stats
label_counts = Counter(e["label"] for e in entries)
print("Label distribution:")
for label, count in label_counts.most_common():
    print(f"  {label}: {count} ({count/len(entries)*100:.1f}%)")

response_spans = defaultdict(list)
for e in entries:
    response_spans[e["source_row_idx"]].append(e)

spans_per_response = [len(v) for v in response_spans.values()]
print(f"\nSpans per response: min={min(spans_per_response)}, max={max(spans_per_response)}, "
      f"mean={sum(spans_per_response)/len(spans_per_response):.1f}")

span_lengths = [len(e["span_text"]) for e in entries]
print(f"Span length (chars): min={min(span_lengths)}, max={max(span_lengths)}, "
      f"mean={sum(span_lengths)/len(span_lengths):.1f}, median={sorted(span_lengths)[len(span_lengths)//2]}")

mixed_responses = 0
for row_idx, row_entries in response_spans.items():
    labels = set(e["label"] for e in row_entries)
    if len(labels) > 1:
        mixed_responses += 1
print(f"\nResponses with mixed labels (both Supported & Not Supported): {mixed_responses}/{len(response_spans)}")

# %% Verify all span offsets
print("Verifying all span offsets...")
mismatches = 0
for e in entries:
    actual = e["response_text"][e["span_char_start"]:e["span_char_end"]]
    if actual != e["span_text"]:
        print(f"  MISMATCH in {e['id']}: expected {e['span_text']!r}, got {actual!r}")
        mismatches += 1

if mismatches == 0:
    print(f"  All {len(entries)} span offsets verified OK")
else:
    print(f"  {mismatches} mismatches found!")

# %% Pick an entry and show full details
# Change IDX to explore different entries.

IDX = 0  # <-- change this

entry = entries[IDX]

print(f"=== Entry {IDX}: {entry['id']} ===")
print(f"Label: {entry['label']}")
print(f"Source row: {entry['source_row_idx']}")
print()
print(f"--- User prompt ---")
print(entry["user_prompt"])
print()
print(f"--- Full response ({len(entry['response_text'])} chars) ---")
response = entry["response_text"]
start = entry["span_char_start"]
end = entry["span_char_end"]
print(response[:start], end="")
print(f">>>{entry['span_text']}<<<", end="")
print(response[end:])
print()
print(f"--- Span details ---")
print(f"Span text: {entry['span_text']!r}")
print(f"Char range: [{start}:{end}] (len={end-start})")
print(f"Label: {entry['label']}")
print(f"Verification note: {entry['verification_note']}")

# %% Show all spans for the same response
row_idx = entry["source_row_idx"]
siblings = [e for e in entries if e["source_row_idx"] == row_idx]
print(f"\n=== All {len(siblings)} spans from response {row_idx} ===")
print(f"User prompt: {siblings[0]['user_prompt']}")
print()
for i, sib in enumerate(siblings):
    status = "SUPPORTED" if sib["supported"] else "NOT SUPPORTED"
    print(f"  [{i}] [{status}] char [{sib['span_char_start']}:{sib['span_char_end']}] "
          f"{sib['span_text']!r}")
    print(f"      Note: {sib['verification_note']}")
    print()

# %% Browse 5 random Not Supported entries (the interesting ones)
print("=== 5 random 'Not Supported' entries ===\n")
not_supported = [e for e in entries if not e["supported"]]
random.seed(123)
sample = random.sample(not_supported, min(5, len(not_supported)))

for i, entry in enumerate(sample):
    print(f"--- [{i}] {entry['id']} ---")
    print(f"User prompt: {entry['user_prompt']}")
    print()
    response = entry["response_text"]
    start = entry["span_char_start"]
    end = entry["span_char_end"]
    ctx_start = max(0, start - 200)
    ctx_end = min(len(response), end + 200)
    print(f"Response context (chars {ctx_start}-{ctx_end} of {len(response)}):")
    print(f"...{response[ctx_start:start]}>>>{entry['span_text']}<<<{response[end:ctx_end]}...")
    print()
    print(f"Verification: {entry['verification_note']}")
    print()
    print()

# %% Load tokenizer
from transformers import AutoTokenizer

from nl_probes.open_ended_eval.hallucination import (
    AO_PROMPTS,
    TEXT_BASELINE_PROMPTS,
    build_hallucination_text_baseline_inputs,
    build_hallucination_verbalizer_prompt_infos,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)

# %% Visualize selected tokens
# Shows which tokens the AO reads activations from (the segment window).
# Tokens marked with >>> are the ones whose activations are fed to the AO.
# Context is truncated at the span end (causal attention means later tokens
# don't affect activations at the span positions).

entry = entries[IDX]
prompt_infos, entry_metadata = build_hallucination_verbalizer_prompt_infos(
    [entry], AO_PROMPTS, tokenizer,
)

# All prompt variants share the same context — show tokens once using the first
info = prompt_infos[0]
positions_set = set(info.positions)

print(f"Token selection visualization for entry {IDX}  |  id={entry['id']}")
print(f"Label: {entry['label']}  |  Span: {entry['span_text']!r}")
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

# Show all prompt variants
print(f"\n--- All prompt variants (context is identical across all) ---")
for pi, meta in zip(prompt_infos, entry_metadata):
    print(f"  {meta['prompt_name']}: gt={pi.ground_truth}  prompt={pi.verbalizer_prompt!r}")

# %% Show text baseline input for same entry
# This is exactly what gets fed to the base model for the text baseline eval.
# Format: user prompt → assistant response (truncated to span) → follow-up question.

entry = entries[IDX]
baseline_inputs, baseline_metadata = build_hallucination_text_baseline_inputs(
    [entry], tokenizer,
)

print(f"Text baseline for entry {IDX}  |  id={entry['id']}")
print(f"Label: {entry['label']}  |  Span: {entry['span_text']!r}")
print(f"{len(baseline_inputs)} prompt variants")
print("=" * 80)

for bi, meta in zip(baseline_inputs, baseline_metadata):
    print(f"\n--- prompt_name: {meta['prompt_name']}  gt={bi.ground_truth} ---")
    print(f"Prompt length: {len(bi.prompt_text)} chars")
    print()
    print(bi.prompt_text)
    print()

# %% Batch verify: build prompt_infos for many entries and check decoded segments

single_prompt = {k: v for k, v in list(AO_PROMPTS.items())[:1]}

random.seed(42)
num_to_check = 50
sample = random.sample(entries, min(num_to_check, len(entries)))

print(f"=== Batch tokenization verification ({len(sample)} entries) ===\n")

prompt_infos_batch, metadata_batch = build_hallucination_verbalizer_prompt_infos(
    sample, single_prompt, tokenizer,
)

ok = 0
failures = []
for pi, meta, e in zip(prompt_infos_batch, metadata_batch, sample):
    segment_token_ids = [pi.context_token_ids[p] for p in pi.positions]
    decoded = tokenizer.decode(segment_token_ids)
    if decoded.strip() == e["span_text"].strip():
        ok += 1
    else:
        failures.append({
            "id": e["id"],
            "original": e["span_text"],
            "decoded": decoded,
        })

print(f"  {ok}/{len(sample)} exact matches")
if failures:
    print(f"  {len(failures)} mismatches:")
    for f in failures:
        print(f"    {f['id']}: original={f['original']!r} decoded={f['decoded']!r}")
else:
    print("  All passed!")

# %%
