#!/usr/bin/env python3
"""Generate DPO pairs for AO from cot-v5 corpus snippets + the 20 templated prompts.

Output JSONL schema (one record per pair):
  {
    "template": "<name>",
    "source_idx": <int>,        # row index in cot-v5
    "context": "<cot prefix str>",  # the activation comes from the last token of this
    "prompt": "<question shown alongside the activation>",
    "chosen": "<correct answer>",
    "rejected": "<wrong answer>",
    "ground_truth": {...},
  }

Activations are computed at training time from `context` (we don't pre-compute them
here — the AO's hook handles it during the DPO forward).

Usage:
  python generate_dpo_pairs.py --n-per-template 13 --seed 42 --out data/dpo_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from prompts import TEMPLATES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="ceselder/cot-oracle-corpus-v5")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="cot_response",
                    help="JSON field on each cot-v5 row containing the prefix text")
    ap.add_argument("--n-per-template", type=int, default=13,
                    help="Examples per template (20 templates -> 260 pairs at default)")
    ap.add_argument("--min-ctx-words", type=int, default=20,
                    help="Reject contexts shorter than this many words")
    ap.add_argument("--max-ctx-chars", type=int, default=2000,
                    help="Truncate context to last N characters before processing")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print 2 pairs per template and exit without writing the jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    random.seed(args.seed)

    print(f"Loading {args.corpus} ({args.split}) ...")
    ds = load_dataset(args.corpus, split=args.split, streaming=False)
    n = len(ds)
    print(f"  {n} rows available")

    # Pre-shuffle indices so each template draws independent random rows
    shuffled = list(range(n))
    rng.shuffle(shuffled)

    pairs = []
    cursor = 0

    def next_context() -> tuple[int, str] | None:
        """Return (idx, ctx) with ctx long enough; advances cursor."""
        nonlocal cursor
        while cursor < len(shuffled):
            idx = shuffled[cursor]
            cursor += 1
            text = ds[idx].get(args.text_field, "") or ""
            ctx = text[-args.max_ctx_chars:]
            if len(ctx.split()) >= args.min_ctx_words:
                return idx, ctx
        return None

    for t in TEMPLATES:
        attempts = 0
        kept = 0
        while kept < args.n_per_template and attempts < args.n_per_template * 10:
            attempts += 1
            got = next_context()
            if got is None:
                print(f"  corpus exhausted at template={t.name} kept={kept}")
                break
            idx, ctx = got
            gt = t.ground_truth(ctx)
            if not gt.get("valid"):
                continue
            ch = t.chosen(ctx, gt)
            rj = t.rejected(ctx, gt)
            if ch.strip() == rj.strip():
                continue
            pairs.append({
                "template": t.name,
                "source_idx": idx,
                "context": ctx,
                "prompt": t.prompt,
                "chosen": ch,
                "rejected": rj,
                "ground_truth": {k: v for k, v in gt.items() if k != "valid"},
            })
            kept += 1
        print(f"  {t.name}: kept {kept}/{args.n_per_template} ({attempts} attempts)")

    if args.dry_run:
        # Print 2 per template
        from itertools import groupby
        pairs_sorted = sorted(pairs, key=lambda p: p["template"])
        for tname, grp in groupby(pairs_sorted, key=lambda p: p["template"]):
            grp = list(grp)[:2]
            print(f"\n--- {tname} ---")
            for p in grp:
                print(f"  ctx tail: {p['context'][-120:]!r}")
                print(f"  chosen:   {p['chosen']}")
                print(f"  rejected: {p['rejected']}")
        return

    rng.shuffle(pairs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(pairs)} DPO pairs to {out_path}")
    counts = {}
    for p in pairs:
        counts[p["template"]] = counts.get(p["template"], 0) + 1
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
