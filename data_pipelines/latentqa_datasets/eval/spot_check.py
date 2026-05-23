# %%
"""
Interactive spot-check for LatentQA eval data.

Shows QA pairs side-by-side with the control (system prompt) so you can
judge whether the QA ground-truth answers are actually derivable from
the system prompt instruction.

Uses stimulus.json (which has control_user + stimulus_user + label)
joined with qa.json (which has QA pairs keyed by label).
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

stimulus = json.loads((DATA_DIR / "stimulus.json").read_text())
qa = json.loads((DATA_DIR / "qa.json").read_text())

print(f"Loaded {len(stimulus)} stimulus entries, {len(qa)} QA labels")

# %%
# === Browse entries ===
# Change NUM_ENTRIES or START_IDX to explore different ranges.

START_IDX = 0
NUM_ENTRIES = 100

subset = stimulus[START_IDX : START_IDX + NUM_ENTRIES]

for i, entry in enumerate(subset):
    idx = START_IDX + i
    label = entry["label"]
    qa_pairs = qa.get(label, [])

    print(f"\n{'=' * 70}")
    print(f"[{idx}] {label}")
    print(f"{'=' * 70}")
    print(f"\nSYSTEM PROMPT (control_user):")
    print(f"  {entry['control_user']}")
    print(f"\nUSER PROMPT (stimulus_user):")
    print(f"  {entry['stimulus_user']}")

    if qa_pairs:
        print(f"\nQA PAIRS ({len(qa_pairs)}):")
        for j, (question, answer) in enumerate(qa_pairs):
            print(f"  Q{j+1}: {question}")
            print(f"  A{j+1}: {answer}")
            print()
    else:
        print(f"\n  (no QA pairs for this label)")

# %%
# === Single entry deep-dive ===
# Change IDX to inspect one entry in detail.

IDX = 0  # <-- change this

entry = stimulus[IDX]
label = entry["label"]
qa_pairs = qa.get(label, [])

print(f"{'=' * 70}")
print(f"[{IDX}] {label}")
print(f"{'=' * 70}")

print(f"\nSYSTEM PROMPT (control_user):")
print(entry["control_user"])

if "control_thought" in entry:
    print(f"\nCONTROL THOUGHT:")
    print(entry["control_thought"])

if "control_model" in entry:
    print(f"\nCONTROL MODEL RESPONSE:")
    print(entry["control_model"])

print(f"\nUSER PROMPT (stimulus_user):")
print(entry["stimulus_user"])

print(f"\nQA PAIRS ({len(qa_pairs)}):")
for j, (question, answer) in enumerate(qa_pairs):
    print(f"\n  Q{j+1}: {question}")
    print(f"  A{j+1}: {answer}")

# %%
