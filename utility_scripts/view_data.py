# %%
"""
Detailed viewer for SFT training data (.pt files).

Shows prompt/target split via labels, placeholder positions, per-token context windows.
Point DATA_PATH at any .pt file in sft_training_data/.

Known hidden bias 14B mappings:
  - hidden_bias_v2_qwen3_14b (active):   synthetic_qa_model_Qwen3-14B_n_49952_save_acts_False_train_ee628ad684f9.pt
  - hidden_bias_inactive_qwen3_14b:       synthetic_qa_model_Qwen3-14B_n_49950_save_acts_False_train_4fc6530a92a8.pt
"""

from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
SFT_DIR = REPO_ROOT / "sft_training_data"

# %%
# === Configuration — change DATA_PATH to inspect a different file ===

# hidden_bias_v2_qwen3_14b (active bias)
# DATA_PATH = SFT_DIR / "synthetic_qa_model_Qwen3-14B_n_49952_save_acts_False_train_ee628ad684f9.pt"
# hidden_bias_inactive_qwen3_14b (inactive / cross-pair bias)
DATA_PATH = SFT_DIR / "synthetic_qa_model_Qwen3-14B_n_49950_save_acts_False_train_4fc6530a92a8.pt"


# %%
# === Helpers ===


SEP = "=" * 100


def inspect_datapoint(item: dict, datapoint_idx: int) -> None:
    input_ids = item["input_ids"]
    labels = item["labels"]
    layers = item["layers"]
    context_input_ids = item["context_input_ids"]
    context_positions = item["context_positions"]

    first_target_idx = next((i for i, lab in enumerate(labels) if lab != -100), len(labels))
    prompt_ids = input_ids[:first_target_idx]
    target_ids = input_ids[first_target_idx:]

    print(f"\n{'#' * 100}")
    print(f"# Datapoint {datapoint_idx}  |  type: {item['datapoint_type']}  |  layers: {layers}")
    if item.get("meta_info"):
        print(f"# meta: {item['meta_info']}")
    print(f"{'#' * 100}")

    # -- 1. Full context prompt (what the target model was processing) --
    print(f"\n{SEP}")
    print("1. FULL CONTEXT PROMPT  (context_input_ids decoded, no truncation)")
    print(SEP)
    print(tokenizer.decode(context_input_ids, skip_special_tokens=False))

    # -- 2. Selected segment (the activation window, indexed from context_positions) --
    print(f"\n{SEP}")
    print(f"2. SELECTED SEGMENT  (context_input_ids[context_positions], {len(context_positions)} tokens)")
    print(SEP)
    selected_token_ids = [context_input_ids[p] for p in context_positions]
    print(tokenizer.decode(selected_token_ids, skip_special_tokens=False))

    # -- 3. Verbalizer prompt (the AO input prompt) --
    print(f"\n{SEP}")
    print(f"3. VERBALIZER PROMPT  (AO input, {len(prompt_ids)} tokens)")
    print(SEP)
    print(tokenizer.decode(prompt_ids, skip_special_tokens=False))

    # -- 4. Verbalizer answer (training target) --
    print(f"\n{SEP}")
    print(f"4. VERBALIZER ANSWER  (AO training target, {len(target_ids)} tokens)")
    print(SEP)
    print(tokenizer.decode(target_ids, skip_special_tokens=False))


# %%
# === Load data ===

data = torch.load(DATA_PATH, weights_only=False, map_location="cpu")
config = data["config"]
items = data["data"]

print(f"File: {DATA_PATH.name}")
print(f"Config: {config}")
print(f"Total datapoints: {len(items)}")

# %%
# === Load tokenizer (inferred from config) ===

model_name = config["model_name"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Tokenizer: {model_name}")

# %%
# === Inspect individual datapoints ===

inspect_datapoint(items[0], 0)

# %%

inspect_datapoint(items[1], 1)

# %%

inspect_datapoint(items[5], 5)

# %%

inspect_datapoint(items[42], 42)

# %%

inspect_datapoint(items[100], 100)

# %%
