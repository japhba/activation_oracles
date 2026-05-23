# %%
"""
Spot-check hidden bias training data (stage 4 output).

Shows the key fields for each entry:
- Prefix text (context before the activation window)
- Selected text (the activation window)
- Question (what the AO is asked)
- Answer / ground truth (what the AO should output)
- AO prompt (the full formatted prompt the AO sees during training)

Run as interactive cells or as a script.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

from nl_probes.dataset_classes.synthetic_qa_dataset import create_synthetic_qa_datapoint
from nl_probes.utils.common import layer_percent_to_layer, load_tokenizer

# %%
# === Configuration ===

DATA_PATH = Path(__file__).parent / "artifacts" / "system_prompt_qa" / "harmful_test_100" / "training_data.json"
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


# %%
# === Helper to inspect a single entry ===


def inspect_entry(idx: int) -> None:
    entry = entries[idx]

    print(f"\n{'#' * 80}")
    print(f"# ENTRY {idx}: {entry['id']}  |  qa_type: {entry['qa_type']}  |  window: {entry['window_desc']}")
    print(f"{'#' * 80}")

    print(f"\n--- Prefix Text (context before activation window) ---")
    print(entry["prefix_text"])

    print(f"\n--- Selected Text (activation window) ---")
    print(entry["selected_text"])

    print(f"\n--- Question ---")
    print(entry["question"])

    print(f"\n--- Answer (ground truth) ---")
    print(entry["answer"])

    # Build the AO prompt
    dp = create_synthetic_qa_datapoint(entry, tokenizer, act_layers)
    if dp is None:
        print("\n  SKIPPED: could not create datapoint")
        return

    first_target_idx = next((i for i, lab in enumerate(dp.labels) if lab != -100), len(dp.labels))
    prompt_ids = dp.input_ids[:first_target_idx]
    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)

    print(f"\n--- AO Prompt (what the oracle sees as input) ---")
    print(prompt_text)


# %%
# === Inspect entries one by one ===

inspect_entry(0)

# %%

inspect_entry(1)

# %%

inspect_entry(5)

# %%

inspect_entry(17)

# %%

inspect_entry(42)

# %%

inspect_entry(80)

# %%
