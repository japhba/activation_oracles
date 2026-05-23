# %%
"""
Collect prompts from the WildChat dataset for training data generation.

Downloads WildChat from HuggingFace, samples ~100 conversations (single-turn
and multi-turn), and saves them to a JSON file for inspection.
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

# %%
# === Configuration ===

N_PROMPTS = 100
SEED = 42
OUTPUT_PATH = Path(__file__).parent / "wildchat_prompts.json"

print("Loading WildChat dataset (this may take a while on first run)...")
ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

# %%
# === Sample prompts ===
# We collect more than we need, then sample, to get a good mix.
# Filter for English conversations only.

random.seed(SEED)

# Collect a pool of candidates
POOL_SIZE = 2000
pool = []
single_turn_count = 0
multi_turn_count = 0

print(f"Collecting pool of {POOL_SIZE} English conversations...")
for i, example in enumerate(ds):
    # Filter for English
    if example.get("language") != "English":
        continue

    conversation = example["conversation"]

    # Basic sanity: must have at least one user message
    if not conversation or conversation[0]["role"] != "user":
        continue

    # Skip very short user messages (< 10 chars) for single-turn
    if len(conversation) == 2 and len(conversation[0]["content"].strip()) < 10:
        continue

    # Determine turn count (count user messages)
    n_user_turns = sum(1 for msg in conversation if msg["role"] == "user")
    is_multi_turn = n_user_turns > 1

    entry = {
        "conversation": [{"role": msg["role"], "content": msg["content"]} for msg in conversation],
        "n_user_turns": n_user_turns,
        "is_multi_turn": is_multi_turn,
    }

    pool.append(entry)
    if is_multi_turn:
        multi_turn_count += 1
    else:
        single_turn_count += 1

    if len(pool) >= POOL_SIZE:
        break

print(f"Pool collected: {len(pool)} conversations ({single_turn_count} single-turn, {multi_turn_count} multi-turn)")

# %%
# === Sample a balanced mix ===

single_turn = [e for e in pool if not e["is_multi_turn"]]
multi_turn = [e for e in pool if e["is_multi_turn"]]

random.shuffle(single_turn)
random.shuffle(multi_turn)

# Aim for roughly 60/40 single/multi split, but flexible
n_single = min(60, len(single_turn))
n_multi = min(40, len(multi_turn))

# If one category is short, fill from the other
remaining = N_PROMPTS - n_single - n_multi
if remaining > 0:
    if len(single_turn) > n_single:
        extra = min(remaining, len(single_turn) - n_single)
        n_single += extra
        remaining -= extra
    if remaining > 0 and len(multi_turn) > n_multi:
        extra = min(remaining, len(multi_turn) - n_multi)
        n_multi += extra

selected = single_turn[:n_single] + multi_turn[:n_multi]
random.shuffle(selected)

print(f"\nSelected {len(selected)} prompts: {n_single} single-turn, {n_multi} multi-turn")

# %%
# === Build output entries ===

entries = []
for i, item in enumerate(selected):
    # For the output, we store the conversation as the user would send it
    # (i.e., the messages list suitable for chat template).
    # For multi-turn, we include the full conversation history.
    # For single-turn, it's just one user message.

    # Strip assistant responses — we only want the user side as prompts
    # For multi-turn: keep user messages + assistant responses as context
    # (the model needs prior turns to respond coherently)
    conversation = item["conversation"]

    # We want to use these as prompts, so we keep the full conversation
    # but mark which turn we'd generate from.
    # Convention: the conversation up to (and including) the last user message
    # is the "prompt". We drop any trailing assistant message.
    if conversation[-1]["role"] == "assistant":
        prompt_messages = conversation[:-1]
    else:
        prompt_messages = conversation

    entry = {
        "id": f"wildchat_{i:04d}",
        "prompt_messages": prompt_messages,
        "full_conversation": conversation,
        "n_user_turns": item["n_user_turns"],
        "is_multi_turn": item["is_multi_turn"],
    }
    entries.append(entry)

# %%
# === Save to JSON ===

output = {
    "metadata": {
        "source": "allenai/WildChat-1M",
        "total_entries": len(entries),
        "single_turn": sum(1 for e in entries if not e["is_multi_turn"]),
        "multi_turn": sum(1 for e in entries if e["is_multi_turn"]),
        "seed": SEED,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    },
    "entries": entries,
}

OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
print(f"\nSaved {len(entries)} prompts to {OUTPUT_PATH}")

# %%
# === Quick stats ===

turn_counts = [e["n_user_turns"] for e in entries]
print(f"\nTurn count distribution:")
for t in sorted(set(turn_counts)):
    count = turn_counts.count(t)
    print(f"  {t} turn(s): {count}")

# Show a few examples
print("\n" + "=" * 80)
print("SAMPLE ENTRIES")
print("=" * 80)

for entry in entries[:5]:
    print(f"\n--- {entry['id']} (multi_turn={entry['is_multi_turn']}, turns={entry['n_user_turns']}) ---")
    for msg in entry["prompt_messages"]:
        role = msg["role"].upper()
        content = msg["content"]
        print(f"  [{role}]: {content}")
    print()

# %%
