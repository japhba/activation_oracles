# %%
"""
Spot-check tokenization for text SFT datasets.

For each example, shows:
1. Token table — every token (detokenized) with train/mask status
2. Context text — full detokenized text of all masked tokens (label=-100)
3. Target text — full detokenized text of the tokens being trained on

Run as interactive cells or as a script.
"""

import json
from pathlib import Path

from nl_probes.text_sft.train import (
    ChatMessage,
    TokenizedExample,
    TextSFTConfig,
    load_conversations,
    tokenize_conversations,
)
from nl_probes.utils.common import load_tokenizer

# %%
# === Configuration ===

MODEL_NAME = "Qwen/Qwen3-8B"
MAX_SEQ_LEN = 4096

# Point this at your conversations JSON, or set to None to use built-in examples.
DATASET_PATH: str | None = None

# Path to a model understanding run dir (screening.json, investigations.json, verification.json).
# Set to None to skip model understanding examples.
MU_RUN_DIR: str | None = "data_pipelines/model_understanding/runs/test_mixed_v2"

# %%
# === Built-in examples ===

BUILTIN_EXAMPLES = [
    {"messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2 + 2 equals 4."},
    ]},
    {"messages": [
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "What is the Pythagorean theorem?"},
        {"role": "assistant", "content": "The Pythagorean theorem states that in a right triangle, a squared plus b squared equals c squared."},
    ]},
    {"messages": [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language."},
        {"role": "user", "content": "What are its main uses?"},
        {"role": "assistant", "content": "Python is widely used in web development, data science, machine learning, automation, and scripting."},
    ]},
]

# %%
# === Load tokenizer ===

tokenizer = load_tokenizer(MODEL_NAME)

# %%
# === Helper to load model understanding examples as conversations ===


def load_mu_conversations(run_dir: str, max_examples: int = 3) -> list[tuple[ChatMessage, ...]]:
    """Load model understanding pipeline data and convert to conversations."""
    import random

    run_path = Path(run_dir)
    screening = json.loads((run_path / "screening.json").read_text())["results"]
    investigations = json.loads((run_path / "investigations.json").read_text())["results"]
    verifications = json.loads((run_path / "verification.json").read_text())["results"]

    screening_by_id = {r["prompt_id"]: r for r in screening}
    inv_by_id = {r["prompt_id"]: r for r in investigations}
    ver_by_id = {r["prompt_id"]: r for r in verifications}

    conversations = []
    for prompt_id in sorted(inv_by_id):
        if prompt_id not in ver_by_id or ver_by_id[prompt_id]["score"] < 7:
            continue
        if prompt_id not in screening_by_id or screening_by_id[prompt_id]["interest_score"] < 3:
            continue

        s = screening_by_id[prompt_id]
        inv = inv_by_id[prompt_id]

        if s["messages"] is None:
            msgs = [ChatMessage(role="user", content=s["user_message"])]
        else:
            msgs = [ChatMessage(role=m["role"], content=m["content"]) for m in s["messages"]]

        # Pick first behavior completion
        if s.get("behavior_completion_indices"):
            comp_idx = s["behavior_completion_indices"][0] - 1
        else:
            comp_idx = 0
        completion = s["completions"][comp_idx]
        question = s.get("second_person_question") or s["question"]
        answer = inv["structured_findings"]["first_person_answer"]

        msgs.append(ChatMessage(role="assistant", content=completion))
        msgs.append(ChatMessage(role="user", content=question))
        msgs.append(ChatMessage(role="assistant", content=answer))
        conversations.append(tuple(msgs))

    # Sort by total character count and return the shortest
    conversations.sort(key=lambda conv: sum(len(m.content) for m in conv))
    return conversations[:max_examples]


# %%
# === Helper to inspect a single example ===


def inspect_example(example: TokenizedExample, label: str = "") -> None:
    print(f"\n{'#' * 80}")
    print(f"# {label}  ({example.sequence_length} tokens, {example.target_token_count} target)")
    print(f"{'#' * 80}")

    # ── Token table ──
    # HuggingFace convention: labels[i] = input_ids[i] for trained tokens, -100 for masked.
    # The model internally shifts (logits[i] trained against labels[i+1]).
    print(f"\n{'=' * 70}")
    print("TOKEN TABLE")
    print(f"{'=' * 70}")
    print(f"{'idx':>5} | {'label':>20} | token")
    print(f"{'-' * 5}-+-{'-' * 20}-+{'-' * 40}")
    for idx, (token_id, label_val) in enumerate(zip(example.input_ids, example.labels)):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        if label_val == -100:
            label_str = "-100"
        else:
            label_str = tokenizer.decode([label_val], skip_special_tokens=False)
        print(f"{idx:5d} | {label_str!r:>20} | {token_text!r}")

    # ── Context (label=-100) ──
    context_ids = [tid for tid, lab in zip(example.input_ids, example.labels) if lab == -100]
    context_text = tokenizer.decode(context_ids, skip_special_tokens=False)
    print(f"\n{'=' * 60}")
    print(f"CONTEXT (masked, {len(context_ids)} tokens)")
    print(f"{'=' * 60}")
    print(context_text)

    # ── Target (trained on) ──
    target_ids = [tid for tid, lab in zip(example.input_ids, example.labels) if lab != -100]
    target_text = tokenizer.decode(target_ids, skip_special_tokens=False)
    print(f"\n{'=' * 60}")
    print(f"TARGET (trained on, {len(target_ids)} tokens)")
    print(f"{'=' * 60}")
    print(target_text)


# %%
# === Tokenize built-in / dataset examples ===

if DATASET_PATH is not None:
    conversations = load_conversations(DATASET_PATH)
    print(f"Loaded {len(conversations)} conversations from {DATASET_PATH}")
else:
    conversations = [
        tuple(ChatMessage(role=m["role"], content=m["content"]) for m in entry["messages"])
        for entry in BUILTIN_EXAMPLES
    ]
    print(f"Using {len(conversations)} built-in examples")

cfg = TextSFTConfig(
    model_name=MODEL_NAME,
    dataset_path=DATASET_PATH or "__builtin__",
    save_dir="/tmp/text_sft_spot_check",
    wandb_project="spot_check",
    wandb_run_name="spot_check",
    global_train_batch_size=1,
    num_epochs=1,
    lr=1e-4,
    max_seq_len=MAX_SEQ_LEN,
)

examples = tokenize_conversations(conversations, tokenizer, cfg, write_debug=False)
examples_by_length = sorted(examples, key=lambda e: e.sequence_length)

print(f"Tokenized {len(examples)} examples")
print(f"Sequence lengths: {[e.sequence_length for e in examples_by_length]}")

# %%
inspect_example(examples_by_length[0], "SHORTEST")

# %%
inspect_example(examples_by_length[1], "SECOND SHORTEST")

# %%
inspect_example(examples_by_length[2], "THIRD SHORTEST")

# %%
# === Model understanding examples ===

if MU_RUN_DIR is not None and Path(MU_RUN_DIR).exists():
    mu_conversations = load_mu_conversations(MU_RUN_DIR, max_examples=2)
    print(f"Loaded {len(mu_conversations)} model understanding conversations from {MU_RUN_DIR}")

    mu_examples = tokenize_conversations(mu_conversations, tokenizer, cfg, write_debug=False)
    mu_examples_by_length = sorted(mu_examples, key=lambda e: e.sequence_length)

    print(f"Tokenized {len(mu_examples)} model understanding examples")
    print(f"Sequence lengths: {[e.sequence_length for e in mu_examples_by_length]}")
else:
    mu_examples_by_length = []
    print(f"Skipping model understanding examples (MU_RUN_DIR={MU_RUN_DIR!r})")

# %%

if mu_examples_by_length:
    inspect_example(mu_examples_by_length[0], "MODEL UNDERSTANDING - SHORTEST")

# %%

if len(mu_examples_by_length) > 1:
    inspect_example(mu_examples_by_length[1], "MODEL UNDERSTANDING - SECOND SHORTEST")

# %%
