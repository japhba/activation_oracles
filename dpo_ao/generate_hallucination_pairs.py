#!/usr/bin/env python3
"""Generate DPO pairs SPECIFICALLY targeting hallucination behavior.

For each context:
  - Take a snippet from cot-v5
  - Build a generic AO prompt: "Describe what is being discussed in the model's recent context."
  - chosen: Sonnet-generated 1-sentence faithful summary
  - rejected: Sonnet-generated 1-sentence confidently-wrong summary
    (mentions things that are plausibly LM-like but ARE NOT in the context)

Output schema matches dpo_v1.jsonl so train_dpo.py works unmodified.

Usage:
  export ANTHROPIC_API_KEY=...
  python generate_hallucination_pairs.py --n 250 --out data/hallucination_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from datasets import load_dataset
import anthropic

SYSTEM = """You are helping build a preference dataset for an activation oracle — a model that produces short text descriptions of what another LLM is "thinking about" at a particular point in its context. You'll be given an excerpt from a model's chain-of-thought. Produce TWO one-sentence descriptions of what the model is discussing:

1. FAITHFUL: a faithful, grounded description of the actual content/topic of the excerpt.
2. HALLUCINATED: a fluent, confident, but FACTUALLY WRONG description. It should sound plausible (mention concepts the model COULD plausibly discuss) but assert things that are NOT in the excerpt.

Format your response as exactly two lines:
FAITHFUL: <one sentence>
HALLUCINATED: <one sentence>

Do not add anything else."""

USER_TEMPLATE = """Excerpt from model's chain-of-thought (last ~150 words):

\"\"\"
{ctx}
\"\"\"

Produce the two descriptions."""


def call_sonnet(client, ctx: str, retries: int = 3) -> tuple[str, str] | None:
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                system=SYSTEM,
                messages=[{"role": "user", "content": USER_TEMPLATE.format(ctx=ctx)}],
            )
            text = resp.content[0].text.strip()
            f_line = next((l for l in text.splitlines() if l.startswith("FAITHFUL:")), None)
            h_line = next((l for l in text.splitlines() if l.startswith("HALLUCINATED:")), None)
            if not f_line or not h_line:
                continue
            faithful = f_line.removeprefix("FAITHFUL:").strip()
            hallucinated = h_line.removeprefix("HALLUCINATED:").strip()
            if faithful and hallucinated and faithful != hallucinated:
                return faithful, hallucinated
        except Exception as e:
            print(f"  retry {attempt}: {type(e).__name__}: {e}")
            time.sleep(2 ** attempt)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="ceselder/cot-oracle-corpus-v5")
    ap.add_argument("--text-field", default="cot_response")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--min-ctx-words", type=int, default=30)
    ap.add_argument("--max-ctx-chars", type=int, default=1500)
    ap.add_argument("--last-words", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY (use the low-prio key by default).")
    client = anthropic.Anthropic()

    print(f"Loading {args.corpus} ...")
    ds = load_dataset(args.corpus, split="train", streaming=False)
    n = len(ds)
    print(f"  {n} rows")
    rng = random.Random(args.seed)
    shuffled = list(range(n))
    rng.shuffle(shuffled)

    pairs = []
    cursor = 0
    while len(pairs) < args.n and cursor < len(shuffled):
        idx = shuffled[cursor]
        cursor += 1
        text = ds[idx].get(args.text_field, "") or ""
        ctx = text[-args.max_ctx_chars:]
        words = ctx.split()
        if len(words) < args.min_ctx_words:
            continue
        # Use only the last ~150 words for the Sonnet prompt to keep cost down
        sonnet_input = " ".join(words[-args.last_words:])
        result = call_sonnet(client, sonnet_input)
        if result is None:
            print(f"  skipped idx={idx} (sonnet failed)")
            continue
        faithful, hallucinated = result
        pairs.append({
            "template": "hallucination_summary",
            "source_idx": idx,
            "context": ctx,
            "prompt": "Describe what is being discussed in the model's recent context.",
            "chosen": faithful,
            "rejected": hallucinated,
            "ground_truth": {"sonnet_generated": True},
        })
        if len(pairs) % 25 == 0:
            print(f"  generated {len(pairs)}/{args.n}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng.shuffle(pairs)
    with out_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(pairs)} pairs to {out_path}")

    # Show 3 random pairs
    print("\nSample pairs:")
    for p in rng.sample(pairs, min(3, len(pairs))):
        print(f"--- src={p['source_idx']} ---")
        print(f"  ctx tail: {p['context'][-100:]!r}")
        print(f"  chosen:   {p['chosen']}")
        print(f"  rejected: {p['rejected']}")


if __name__ == "__main__":
    main()
