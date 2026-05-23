#!/usr/bin/env python3
"""Generate chunked ConvQA via Anthropic batched API (Sonnet 4.6).

3-round pipeline:
  R1: split-point + question (sees full CoT sentences)
  R2: BB baseline answer (sees only prefix)
  R3: GT answer (sees only suffix) — this is the training target_response

Operates on cot-v5 corpus (ceselder/cot-oracle-corpus-v5). Each CoT can spawn
K rows via different boundary positions, configurable via --chunks-per-cot.

Usage:
    python scripts/generate_chunked_convqa_anthropic.py \
        --round r1 --max-cots 200 --chunks-per-cot 1 \
        --out-prefix /tmp/convqa_test
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# --- Prompts (copied from generate_chunked_convqa.py) ---

R1_SYSTEM = """\
You are analyzing a chain-of-thought (CoT) reasoning trace. Your task:

1. Find a natural splitting point — a sentence index where the reasoning shifts direction, introduces a new insight, makes a key deduction, or changes approach. The suffix (everything after the split) should contain something that is NOT obvious from reading the prefix alone.

2. Generate an open-ended question about the suffix content that:
   - Cannot be confidently answered from the prefix text alone
   - Has a concrete, specific answer derivable from the suffix
   - Is about the reasoning process, conclusions, or approach in the suffix

Return ONLY a JSON object with keys "split_index" (int, 0-based sentence index — last sentence of the prefix) and "question" (str).

Constraints:
- split_index must leave at least 3 sentences in both prefix and suffix
- The question should be answerable in 1-3 sentences"""

R2_SYSTEM = """\
You can only see the beginning of a chain-of-thought reasoning trace (the prefix). \
A question is asked about what comes later in the reasoning. \
Answer as best you can from the prefix alone. If you truly cannot answer, say so honestly. \
Be concise: 1-3 sentences."""

R3_SYSTEM = """\
You are given a portion of a chain-of-thought reasoning trace (the suffix, after a split point) \
and a question about it. Answer the question based on the suffix content. \
Be concise: 1-3 sentences."""


def build_r1_user(question: str, sentences: list) -> str:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
    return f"## Original Problem\n{question}\n\n## CoT (numbered sentences)\n{numbered}"


def build_r2_user(orig_question: str, prefix_text: str, generated_question: str) -> str:
    return (
        f"## Original Problem\n{orig_question}\n\n"
        f"## Reasoning so far (prefix)\n{prefix_text}\n\n"
        f"## Question\n{generated_question}"
    )


def build_r3_user(orig_question: str, suffix_text: str, generated_question: str) -> str:
    return (
        f"## Original Problem\n{orig_question}\n\n"
        f"## Reasoning continuation (suffix)\n{suffix_text}\n\n"
        f"## Question\n{generated_question}"
    )


# --- Anthropic batch API ---

API_BASE = "https://api.anthropic.com/v1/messages/batches"
ANTHROPIC_VERSION = "2023-06-01"


def _headers(api_key: str) -> dict:
    return {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def submit_batch(requests_list: list, api_key: str) -> str:
    """Submit a batch. Returns batch_id."""
    body = {"requests": requests_list}
    r = requests.post(API_BASE, headers=_headers(api_key), data=json.dumps(body))
    if r.status_code >= 300:
        print(f"submit error {r.status_code}: {r.text[:500]}", file=sys.stderr)
        r.raise_for_status()
    obj = r.json()
    return obj["id"]


def submit_batches_chunked(requests_list: list, api_key: str, chunk_size: int = 15000) -> list[str]:
    """Split a list of requests into chunks and submit each as a separate batch."""
    batch_ids = []
    for i in range(0, len(requests_list), chunk_size):
        sub = requests_list[i : i + chunk_size]
        bid = submit_batch(sub, api_key)
        print(f"  chunk {i//chunk_size}: {len(sub)} reqs -> {bid}")
        batch_ids.append(bid)
    return batch_ids


def get_batch(batch_id: str, api_key: str) -> dict:
    r = requests.get(f"{API_BASE}/{batch_id}", headers=_headers(api_key))
    r.raise_for_status()
    return r.json()


def fetch_results(results_url: str, api_key: str) -> list:
    """Stream JSONL results from a finished batch."""
    r = requests.get(results_url, headers=_headers(api_key), stream=True)
    r.raise_for_status()
    out = []
    for line in r.iter_lines():
        if not line:
            continue
        out.append(json.loads(line))
    return out


def make_request(custom_id: str, model: str, system: str, user: str, max_tokens: int = 800) -> dict:
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


# --- Pipeline ---

def load_cot_corpus(max_cots: int | None) -> list[dict]:
    """Load cot-v5; return list of dicts with id, question, sentences, source."""
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


def round1(args, api_key: str, model: str):
    corpus = load_cot_corpus(args.max_cots)
    print(f"loaded {len(corpus)} CoTs from cot-v5 (after min-sentence filter)")
    reqs = []
    for r in corpus:
        custom_id = f"{r['id']}__c0"  # single chunk for v1
        reqs.append(make_request(custom_id, model, R1_SYSTEM, build_r1_user(r["question"], r["sentences"]), max_tokens=400))
    print(f"submitting R1 batches (chunk_size={args.chunk_size}) of {len(reqs)} total requests to {model}...")
    t0 = time.time()
    batch_ids = submit_batches_chunked(reqs, api_key, chunk_size=args.chunk_size)
    print(f"submitted {len(batch_ids)} batches in {time.time()-t0:.1f}s")
    # Save manifest for downstream
    manifest = {
        "batch_ids": batch_ids,
        "model": model,
        "round": "r1",
        "submitted_at_unix": t0,
        "corpus_size": len(corpus),
    }
    Path(f"{args.out_prefix}_r1_manifest.json").write_text(json.dumps(manifest, indent=2))
    # Also save corpus snapshot so we can correlate R2/R3 later
    Path(f"{args.out_prefix}_r1_corpus.jsonl").write_text(
        "\n".join(json.dumps({"id": r["id"], "source": r["source"], "question": r["question"], "sentences": r["sentences"]}) for r in corpus)
    )
    print(f"manifest -> {args.out_prefix}_r1_manifest.json")


def _parse_r1_json(text: str) -> dict | None:
    import re
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text.strip())
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
        if "split_index" not in obj or "question" not in obj:
            return None
        if not isinstance(obj["split_index"], int) or not isinstance(obj["question"], str):
            return None
        return obj
    except json.JSONDecodeError:
        return None


def round_two_three(args, api_key: str, model: str, which: str):
    """Submit R2 (BB-from-prefix) or R3 (GT-from-suffix). Requires R1 outputs."""
    assert which in ("r2", "r3")
    # Load R1 manifest + corpus snapshot + R1 results
    manifest = json.loads(Path(f"{args.out_prefix}_r1_manifest.json").read_text())
    corpus = [json.loads(l) for l in Path(f"{args.out_prefix}_r1_corpus.jsonl").read_text().splitlines() if l]
    by_id = {f"{r['id']}__c0": r for r in corpus}

    print(f"fetching R1 results from {manifest['batch_id']}...")
    info = get_batch(manifest["batch_id"], api_key)
    if info.get("processing_status") != "ended":
        raise RuntimeError(f"R1 batch not ended yet: {info.get('processing_status')}")
    r1_results = fetch_results(info["results_url"], api_key)
    print(f"R1 results: {len(r1_results)}")

    reqs = []
    for entry in r1_results:
        if entry["result"]["type"] != "succeeded":
            continue
        cid = entry["custom_id"]
        if cid not in by_id:
            continue
        text = entry["result"]["message"]["content"][0]["text"]
        parsed = _parse_r1_json(text)
        if parsed is None:
            continue
        split_idx = parsed["split_index"]
        question = parsed["question"]
        c = by_id[cid]
        sents = c["sentences"]
        if not (3 <= split_idx + 1 <= len(sents) - 3):
            continue  # not enough sentences on each side
        if which == "r2":
            prefix_text = " ".join(sents[: split_idx + 1])
            user = build_r2_user(c["question"], prefix_text, question)
            sys_p = R2_SYSTEM
        else:
            suffix_text = " ".join(sents[split_idx + 1 :])
            user = build_r3_user(c["question"], suffix_text, question)
            sys_p = R3_SYSTEM
        reqs.append(make_request(cid, model, sys_p, user, max_tokens=400))

    print(f"submitting {which.upper()} batch of {len(reqs)} requests to {model}...")
    t0 = time.time()
    batch_id = submit_batch(reqs, api_key)
    print(f"submitted batch_id={batch_id} in {time.time()-t0:.1f}s")
    Path(f"{args.out_prefix}_{which}_manifest.json").write_text(json.dumps({
        "batch_id": batch_id, "model": model, "round": which,
        "submitted_at_unix": t0, "num_requests": len(reqs),
    }, indent=2))
    print(f"manifest -> {args.out_prefix}_{which}_manifest.json")


def assemble(args, api_key: str):
    """Combine R1+R2+R3 into final dataset rows."""
    corpus = [json.loads(l) for l in Path(f"{args.out_prefix}_r1_corpus.jsonl").read_text().splitlines() if l]
    by_id = {f"{r['id']}__c0": r for r in corpus}

    def fetch_round(round_name):
        m = json.loads(Path(f"{args.out_prefix}_{round_name}_manifest.json").read_text())
        info = get_batch(m["batch_id"], api_key)
        if info.get("processing_status") != "ended":
            raise RuntimeError(f"{round_name} batch not ended yet: {info.get('processing_status')}")
        return {e["custom_id"]: e for e in fetch_results(info["results_url"], api_key)}

    r1 = fetch_round("r1")
    r2 = fetch_round("r2")
    r3 = fetch_round("r3")
    print(f"R1: {len(r1)}, R2: {len(r2)}, R3: {len(r3)}")

    rows = []
    for cid, entry1 in r1.items():
        if entry1["result"]["type"] != "succeeded":
            continue
        text1 = entry1["result"]["message"]["content"][0]["text"]
        p1 = _parse_r1_json(text1)
        if p1 is None:
            continue
        c = by_id.get(cid)
        if c is None:
            continue
        sents = c["sentences"]
        s = p1["split_index"]
        if not (3 <= s + 1 <= len(sents) - 3):
            continue
        q = p1["question"]
        e2 = r2.get(cid)
        e3 = r3.get(cid)
        if not (e2 and e3 and e2["result"]["type"] == "succeeded" and e3["result"]["type"] == "succeeded"):
            continue
        prefix = " ".join(sents[: s + 1])
        suffix = " ".join(sents[s + 1 :])
        bb_resp = e2["result"]["message"]["content"][0]["text"].strip()
        gt_resp = e3["result"]["message"]["content"][0]["text"].strip()
        rows.append({
            "cot_id": c["id"], "source": c["source"], "question": c["question"],
            "cot_text": " ".join(sents),
            "cot_prefix": prefix, "cot_suffix": suffix,
            "split_index": s, "n_sentences": len(sents),
            "prompt": q, "target_response": gt_resp, "bb_response": bb_resp,
            "generation_prompt": "",  # could populate if needed
        })
    print(f"final rows: {len(rows)}  ({len(rows)/len(r1)*100:.1f}% yield from R1)")
    out_parquet = f"{args.out_prefix}_final.parquet"
    pd.DataFrame(rows).to_parquet(out_parquet)
    print(f"wrote {out_parquet}")
    # quick token stats
    if rows:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", token=os.environ.get("HF_TOKEN", ""))
        sample_size = min(500, len(rows))
        sample = rows[:sample_size]
        avg = sum(len(tok.encode(r["target_response"])) for r in sample) / sample_size
        print(f"avg target_response tokens (Qwen3 tokenizer, sample {sample_size}): {avg:.1f}")
        print(f"estimated total target tokens: {int(avg * len(rows)):,} ({avg * len(rows)/1e6:.2f}M)")


def poll(args, api_key: str):
    bid = args.batch_id
    t0 = time.time()
    while True:
        info = get_batch(bid, api_key)
        st = info.get("processing_status", "?")
        counts = info.get("request_counts", {})
        elapsed = time.time() - t0
        print(f"[{elapsed:6.0f}s] status={st} counts={counts}")
        if st == "ended":
            print(f"results_url={info.get('results_url')}")
            return info
        time.sleep(args.poll_seconds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", choices=["r1", "r2", "r3", "poll", "assemble"], required=True)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--max-cots", type=int, default=None)
    ap.add_argument("--chunks-per-cot", type=int, default=1)
    ap.add_argument("--out-prefix", default="/tmp/convqa_run")
    ap.add_argument("--batch-id")
    ap.add_argument("--poll-seconds", type=int, default=20)
    ap.add_argument("--chunk-size", type=int, default=15000)
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY_BATCH") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("set ANTHROPIC_API_KEY_BATCH (or ANTHROPIC_API_KEY)", file=sys.stderr)
        sys.exit(2)

    if args.round == "r1":
        round1(args, api_key, args.model)
    elif args.round in ("r2", "r3"):
        round_two_three(args, api_key, args.model, args.round)
    elif args.round == "assemble":
        assemble(args, api_key)
    elif args.round == "poll":
        if not args.batch_id:
            print("--batch-id required for poll", file=sys.stderr)
            sys.exit(2)
        poll(args, api_key)


if __name__ == "__main__":
    main()
