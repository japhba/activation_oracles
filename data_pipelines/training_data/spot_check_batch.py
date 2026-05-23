# %%
"""
Spot-check batch API QA results.

Reads a JSONL file from the batch test and displays each entry's
question, answer, reasoning, and metadata in a readable format.

Usage: run as interactive cells in VS Code.
"""

import json
from pathlib import Path

# %%
# === Configuration ===

JSONL_PATH = Path(__file__).parent / "artifacts" / "batch_test_100.jsonl"

# %%
# === Load results ===

with open(JSONL_PATH) as f:
    results = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(results)} results from {JSONL_PATH.name}")

by_format = {}
for r in results:
    fmt = r["response_format"]
    by_format[fmt] = by_format.get(fmt, 0) + 1
print(f"Formats: {by_format}")

by_mode = {}
for r in results:
    mode = r.get("past_mode", "") or r.get("qa_type", "future")
    by_mode[mode] = by_mode.get(mode, 0) + 1
print(f"Modes: {by_mode}")

# %%
# === Display entries ===

for i, r in enumerate(results):
    past_mode = r.get("past_mode", "")
    mode_label = f"{past_mode}" if past_mode else "future"

    print(f"\n{'#' * 100}")
    print(f"# [{i}] {r['id']}  |  format={r['response_format']}  |  mode={mode_label}")
    print(f"{'#' * 100}")

    print(f"\n--- Conversation (as the LLM sees it, with chat tokens) ---")
    print(r.get("prefix_text", "(not stored)"))

    print(f"\n--- Selected Text ({r.get('window_desc', '')}) ---")
    print(r.get("selected_text", "(not stored)"))

    print(f"\n--- Generator Prompt (sent to Sonnet) ---")
    print(r.get("generator_user_prompt", "(not stored)"))

    print(f"\n--- Question ---")
    print(r["question"])

    print(f"\n--- Answer ---")
    print(r["answer"])

    if r.get("qa_reasoning"):
        print(f"\n--- Reasoning ---")
        print(r["qa_reasoning"])

    print()
