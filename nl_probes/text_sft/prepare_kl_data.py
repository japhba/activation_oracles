"""Prepare tokenized WildChat conversations for KL regularization during SFT.

Streams allenai/WildChat-1M, tokenizes full conversations (user + assistant turns),
and saves as a .pt file. Used by train.py when kl_loss_weight > 0.

Usage:
    .venv/bin/python nl_probes/text_sft/prepare_kl_data.py \
        --model Qwen/Qwen3-8B --n-conversations 20000
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

CHAT_DATASET = "allenai/WildChat-1M"


def stream_conversations(max_scan: int = 500_000):
    ds = iter(load_dataset(CHAT_DATASET, split="train", streaming=True))
    for i, row in enumerate(ds):
        if i >= max_scan:
            break
        if row.get("language") != "English":
            continue
        if row.get("redacted", False):
            continue
        conv = row["conversation"]
        if not conv or conv[0]["role"] != "user":
            continue
        messages = [{"role": m["role"], "content": m["content"]} for m in conv]
        if any("NAME_" in m["content"] for m in messages):
            continue
        total_chars = sum(len(m["content"]) for m in messages)
        if total_chars < 200 or total_chars > 5000:
            continue
        yield messages


def main():
    parser = argparse.ArgumentParser(description="Prepare KL regularization data from WildChat")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n-conversations", type=int, required=True)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-scan", type=int, default=500_000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    supports_thinking = (
        hasattr(tokenizer, "chat_template")
        and tokenizer.chat_template is not None
        and "enable_thinking" in tokenizer.chat_template
    )

    entries = []
    for messages in tqdm(
        stream_conversations(args.max_scan),
        desc="Tokenizing",
        total=args.n_conversations,
    ):
        kwargs = {"tokenize": True, "add_generation_prompt": False, "return_tensors": None}
        if supports_thinking:
            kwargs["enable_thinking"] = False

        token_ids = tokenizer.apply_chat_template(messages, **kwargs)
        if len(token_ids) > args.max_seq_len:
            token_ids = token_ids[: args.max_seq_len]
        if len(token_ids) < 50:
            continue

        entries.append({"input_ids": token_ids, "sequence_length": len(token_ids)})
        if len(entries) >= args.n_conversations:
            break

    output_dir = Path("text_sft_training_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model.split("/")[-1].lower()
    output_path = output_dir / f"kl_wildchat_{model_short}_{len(entries)}.pt"

    torch.save(
        {
            "metadata": {
                "model": args.model,
                "num_entries": len(entries),
                "max_seq_len": args.max_seq_len,
                "source": CHAT_DATASET,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "entries": entries,
        },
        output_path,
    )

    lengths = [e["sequence_length"] for e in entries]
    print(f"Saved {len(entries)} entries to {output_path}")
    print(f"Sequence lengths: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.0f}")


if __name__ == "__main__":
    main()
