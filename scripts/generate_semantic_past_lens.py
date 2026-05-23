#!/usr/bin/env python3
"""Generate NLA-style summary targets for past_lens training.

For each cot-v5 CoT, pick 3 positions (25%, 50%, 75% of n_sentences). For each
(cot, position) pair, ask Sonnet 4.6 to summarize the prefix up to that position
in the NLA paper's style — short paragraph with bolded topic headings.

Output is intended as a new past_lens target distribution: instead of "predict
the literal next/prev k tokens", train AO to predict a semantic annotation of
what the model has attended to.

Usage:
    python scripts/generate_nla_past_lens.py --round r1 --max-cots 200 \\
        --out-prefix /tmp/nla_test
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

NLA_SYSTEM = """You are annotating what a language model has attended to in a piece of text. Given a text excerpt (the prefix of a longer document), write a concise NLA-style annotation describing the content as a model would encode it in its activations at this point.

Format your annotation as a short paragraph with bolded topic headings, in the style:

**Topic**: one-line description of the main subject
**Key claims**: 1-3 key facts, deductions, or statements made so far
**Reasoning state**: what the reasoner is currently doing (exploring, verifying, concluding, etc.)
**Tone/style**: notable register, formality, or rhetorical features (only if salient)

Be specific. Reference entities, numbers, or technical terms when present. Total output should be 60-150 tokens. Do NOT speculate about what comes next; only describe what's in the prefix."""


def build_nla_prompt(orig_question: str, prefix_text: str) -> str:
    return (
        f"## Original question/task\n{orig_question}\n\n"
        f"## Text excerpt (prefix only — annotate this)\n{prefix_text}\n\n"
        f"Now write the NLA-style annotation."
    )


# --- Anthropic batch API helpers (same as chunked-convqa) ---

API_BASE = "https://api.anthropic.com/v1/messages/batches"
ANTHROPIC_VERSION = "2023-06-01"


def _headers(api_key: str) -> dict:
    return {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def submit_batch(requests_list: list, api_key: str) -> str:
    body = {"requests": requests_list}
    r = requests.post(API_BASE, headers=_headers(api_key), data=json.dumps(body))
    if r.status_code >= 300:
        print(f"submit error {r.status_code}: {r.text[:500]}", file=sys.stderr)
        r.raise_for_status()
    return r.json()["id"]


def submit_chunked(requests_list: list, api_key: str, chunk_size: int = 4000) -> list[str]:
    batch_ids = []
    for i in range(0, len(requests_list), chunk_size):
        sub = requests_list[i : i + chunk_size]
        bid = submit_batch(sub, api_key)
        print(f"  chunk {i//chunk_size}: {len(sub)} -> {bid}")
        batch_ids.append(bid)
    return batch_ids


def make_request(custom_id: str, model: str, system: str, user: str, max_tokens: int = 250) -> dict:
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": 0.0,
        },
    }


# --- Position-picking ---

def positions_for_cot(n_sentences: int, num_positions: int = 3) -> list[int]:
    """Pick num_positions sentence indices at fixed fractions of the CoT."""
    if n_sentences < 8:
        return []
    fractions = [0.25, 0.5, 0.75] if num_positions == 3 else [
        (i + 1) / (num_positions + 1) for i in range(num_positions)
    ]
    positions = [max(2, min(n_sentences - 3, int(n_sentences * f))) for f in fractions]
    return positions


# --- Main ---

def load_corpus(max_cots: int | None) -> list[dict]:
    df0 = pd.read_parquet("hf://datasets/ceselder/cot-oracle-corpus-v5/data/train-00000-of-00002.parquet")
    df1 = pd.read_parquet("hf://datasets/ceselder/cot-oracle-corpus-v5/data/train-00001-of-00002.parquet")
    df = pd.concat([df0, df1]).reset_index(drop=True)
    if max_cots is not None:
        df = df.head(max_cots)
    rows = []
    for _, r in df.iterrows():
        n_sentences = r.get("n_sentences") or len(r.get("sentences") or [])
        if n_sentences < 8:
            continue
        rows.append({
            "id": r["id"],
            "source": r["source"],
            "question": r["question"],
            "sentences": list(r["sentences"]),
            "n_sentences": int(n_sentences),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", choices=["r1", "poll"], required=True)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--max-cots", type=int, default=None)
    ap.add_argument("--num-positions", type=int, default=3)
    ap.add_argument("--out-prefix", default="/tmp/nla_full")
    ap.add_argument("--chunk-size", type=int, default=4000)
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY_BATCH") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("set ANTHROPIC_API_KEY_BATCH", file=sys.stderr)
        sys.exit(2)

    if args.round == "r1":
        corpus = load_corpus(args.max_cots)
        print(f"loaded {len(corpus)} CoTs from cot-v5")
        reqs = []
        position_records = []  # for later: which custom_id maps to which (cot_id, position)
        for r in corpus:
            positions = positions_for_cot(r["n_sentences"], args.num_positions)
            for i, p in enumerate(positions):
                custom_id = f"{r['id']}__p{p}__c{i}"
                prefix_text = " ".join(r["sentences"][: p + 1])
                user = build_nla_prompt(r["question"], prefix_text)
                reqs.append(make_request(custom_id, args.model, NLA_SYSTEM, user, max_tokens=250))
                position_records.append({
                    "custom_id": custom_id,
                    "cot_id": r["id"],
                    "source": r["source"],
                    "split_index": p,
                    "n_sentences": r["n_sentences"],
                })
        print(f"prepared {len(reqs)} R1 requests across {len(corpus)} CoTs")
        t0 = time.time()
        batch_ids = submit_chunked(reqs, api_key, chunk_size=args.chunk_size)
        print(f"submitted {len(batch_ids)} batches in {time.time()-t0:.1f}s")
        Path(f"{args.out_prefix}_r1_manifest.json").write_text(json.dumps({
            "batch_ids": batch_ids,
            "model": args.model,
            "round": "r1",
            "submitted_at_unix": t0,
            "corpus_size": len(corpus),
            "num_positions": args.num_positions,
            "num_requests": len(reqs),
        }, indent=2))
        # Save position records for assembly later
        Path(f"{args.out_prefix}_position_records.jsonl").write_text(
            "\n".join(json.dumps(r) for r in position_records)
        )
        # Save corpus snapshot
        Path(f"{args.out_prefix}_corpus.jsonl").write_text(
            "\n".join(json.dumps({"id": r["id"], "source": r["source"], "question": r["question"], "sentences": r["sentences"]}) for r in corpus)
        )
        print(f"manifest -> {args.out_prefix}_r1_manifest.json")
        print(f"positions -> {args.out_prefix}_position_records.jsonl")
        print(f"corpus -> {args.out_prefix}_corpus.jsonl")

    elif args.round == "poll":
        manifest = json.loads(Path(f"{args.out_prefix}_r1_manifest.json").read_text())
        while True:
            ended = succ = total = 0
            for bid in manifest["batch_ids"]:
                d = requests.get(f"{API_BASE}/{bid}", headers=_headers(api_key)).json()
                c = d.get("request_counts", {})
                total += sum(c.values())
                succ += c.get("succeeded", 0)
                if d.get("processing_status") == "ended":
                    ended += 1
            print(f"[{time.strftime('%H:%M:%S')}] {ended}/{len(manifest['batch_ids'])} batches done, {succ}/{total} succeeded")
            if ended == len(manifest["batch_ids"]):
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
