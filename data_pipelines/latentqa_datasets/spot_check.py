# %%
"""
Spot-check latentQA v2 training data.

For each entry, shows:
1. Raw fields (prefix_text, selected_text, question, answer)
2. Processed TrainingDataPoint (decoded prompt, target, context, selected positions)

Run as interactive cells or as a script.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

from nl_probes.dataset_classes.synthetic_qa_dataset import (
    compute_context_and_positions,
    create_synthetic_qa_datapoint,
)
from nl_probes.utils.common import layer_percent_to_layer, load_tokenizer
from nl_probes.utils.dataset_utils import SPECIAL_TOKEN

# %%
# === Configuration ===

DATA_PATH = Path(__file__).parent / "artifacts" / "training_data_20.json"
MODEL_NAME = "Qwen/Qwen3-8B"
LAYER_PERCENTS = [25, 50, 75]

# %%
# === Load data and tokenizer ===

tokenizer = load_tokenizer(MODEL_NAME)
act_layers = [layer_percent_to_layer(MODEL_NAME, lp) for lp in LAYER_PERCENTS]

with open(DATA_PATH) as f:
    raw = json.load(f)

entries = raw["entries"]
print(f"Loaded {len(entries)} entries from {DATA_PATH}")
print(f"Layers: {act_layers}")
print(f"Metadata: {json.dumps(raw['metadata'], indent=2)}")


# %%
# === Inspect a single entry ===


def inspect_entry(idx: int) -> None:
    entry = entries[idx]

    print(f"\n{'#' * 100}")
    print(f"# ENTRY {idx}: {entry['id']}")
    print(f"{'#' * 100}")

    # ── Part 1: Raw fields ──
    print(f"\n{'=' * 80}")
    print("RAW DATA")
    print(f"{'=' * 80}")

    print(f"\n  qa_type:  {entry['qa_type']}")
    print(f"  label:    {entry.get('label', '')}")
    print(f"  window:   chars {entry.get('window_start', '?')}-{entry.get('window_end', '?')} ({entry.get('window_desc', '')})")

    print(f"\n--- Prefix Text ---")
    print(entry["prefix_text"])

    print(f"\n--- Selected Text ---")
    print(entry["selected_text"])

    print(f"\n--- Question ---")
    print(entry["question"])

    print(f"\n--- Answer ---")
    print(entry["answer"])

    # ── Part 2: Processed TrainingDataPoint ──
    print(f"\n{'=' * 80}")
    print("PROCESSED TRAINING DATAPOINT")
    print(f"{'=' * 80}")

    dp = create_synthetic_qa_datapoint(entry, tokenizer, act_layers)
    if dp is None:
        print("  SKIPPED: empty question or answer")
        return

    # Decode prompt and target from the training input
    first_target_idx = next((i for i, lab in enumerate(dp.labels) if lab != -100), len(dp.labels))
    prompt_ids = dp.input_ids[:first_target_idx]
    target_ids = dp.input_ids[first_target_idx:]

    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    target_text = tokenizer.decode(target_ids, skip_special_tokens=False)

    print(f"\n  datapoint_type:    {dp.datapoint_type}")
    print(f"  layers:            {dp.layers}")
    print(f"  input_len:         {len(dp.input_ids)} tokens")
    print(f"  prompt_len:        {len(prompt_ids)} tokens")
    print(f"  target_len:        {len(target_ids)} tokens")
    print(f"  context_len:       {len(dp.context_input_ids)} tokens")
    print(
        f"  context_positions: {len(dp.context_positions)} tokens "
        f"(indices {dp.context_positions[0]}-{dp.context_positions[-1]})"
    )

    print(f"\n--- AO Prompt (what the oracle sees as input) ---")
    print(prompt_text)

    print(f"\n--- AO Target (what the oracle is trained to output) ---")
    print(target_text)

    print(f"\n--- Context (full tokenized context for activation extraction) ---")
    print(tokenizer.decode(dp.context_input_ids, skip_special_tokens=False))

    # Decode selected tokens
    print(f"\n--- Selected Tokens (activation positions decoded) ---")
    selected_token_ids = [dp.context_input_ids[p] for p in dp.context_positions]
    selected_text_decoded = tokenizer.decode(selected_token_ids, skip_special_tokens=False)
    print(f"  Decoded: {selected_text_decoded!r}")

    # Verify it matches the original selected_text
    original = entry["selected_text"]
    if selected_text_decoded.strip() != original.strip():
        print(f"  WARNING: selected text mismatch!")
        print(f"    original: {original!r}")
        print(f"    decoded:  {selected_text_decoded!r}")
    else:
        print(f"  MATCH OK")

    # Show context around the selected window
    ctx_start = max(0, dp.context_positions[0] - 5)
    ctx_end = min(len(dp.context_input_ids), dp.context_positions[-1] + 6)
    print(f"\n--- Context Window (positions {ctx_start}-{ctx_end - 1}) ---")
    for i in range(ctx_start, ctx_end):
        tok_id = dp.context_input_ids[i]
        tok_str = tokenizer.decode([tok_id], skip_special_tokens=False).replace("\n", "\\n")
        marker = " <SEL>" if i in dp.context_positions else "      "
        print(f"  {marker} pos={i:4d}  id={tok_id:7d}  token={tok_str!r}")

    # Verify placeholder positions
    special_token_ids = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)
    if len(special_token_ids) == 1:
        special_token_id = special_token_ids[0]
        found_positions = [idx for idx, tok in enumerate(dp.input_ids) if tok == special_token_id]
        if found_positions != dp.positions:
            print(f"\n  WARNING: placeholder positions mismatch!")
            print(f"    expected: {dp.positions}")
            print(f"    found:    {found_positions}")
        else:
            print(f"\n  Placeholder positions verified OK ({len(dp.positions)} positions)")


# %%
# === Inspect entries ===

inspect_entry(0)

# %%

inspect_entry(1)

# %%

inspect_entry(2)

# %%

inspect_entry(3)

# %%
# === Batch inspect all entries ===

for i in range(len(entries)):
    inspect_entry(i)

# %%
