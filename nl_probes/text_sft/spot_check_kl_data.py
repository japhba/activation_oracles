"""Spot check KL regularization data: show exactly which tokens get KL loss.

Usage:
    .venv/bin/python nl_probes/text_sft/spot_check_kl_data.py \
        --model Qwen/Qwen3-8B \
        --kl-data text_sft_training_data/kl_wildchat_qwen3-8b_20000.pt \
        --n-examples 3 \
        --output kl_spot_check.md
"""

import argparse
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nl_probes.text_sft.train import KLEntry, _load_kl_data, construct_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--kl-data", type=str, required=True)
    parser.add_argument("--n-examples", type=int, default=3)
    parser.add_argument("--output", type=str, default="kl_spot_check.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    entries = _load_kl_data(args.kl_data)

    rng = random.Random(args.seed)
    # Pick examples with varying lengths to show padding behavior
    entries_sorted = sorted(entries, key=lambda e: e.sequence_length)
    picks = [
        entries_sorted[len(entries_sorted) // 4],       # short
        entries_sorted[len(entries_sorted) // 2],       # medium
        entries_sorted[3 * len(entries_sorted) // 4],   # long
    ]
    if args.n_examples > 3:
        extras = rng.sample(entries, args.n_examples - 3)
        picks.extend(extras)
    picks = picks[: args.n_examples]

    device = torch.device("cpu")
    batch = construct_batch(picks, tokenizer.pad_token_id, device)
    kl_mask = batch.attention_mask

    lines = ["# KL Data Spot Check\n"]
    lines.append(f"- Data: `{args.kl_data}`")
    lines.append(f"- Model: `{args.model}`")
    lines.append(f"- Batch size: {len(picks)}")
    lines.append(f"- Padded sequence length: {batch.input_ids.shape[1]}")
    lines.append("")

    for ex_idx in range(len(picks)):
        seq_len = picks[ex_idx].sequence_length
        padded_len = batch.input_ids.shape[1]
        pad_count = padded_len - seq_len
        input_ids = batch.input_ids[ex_idx].tolist()
        attn = batch.attention_mask[ex_idx].tolist()
        mask_row = kl_mask[ex_idx].tolist()

        lines.append(f"## Example {ex_idx + 1} (original length={seq_len}, padded to {padded_len})\n")

        # Full decoded text (non-padding only)
        real_ids = input_ids[pad_count:]
        full_text = tokenizer.decode(real_ids, skip_special_tokens=False)
        lines.append("### Full decoded text\n")
        lines.append("```")
        lines.append(full_text)
        lines.append("```\n")

        # Token-by-token breakdown
        lines.append("### Token-by-token breakdown\n")
        lines.append("Each line: `position | attention_mask | KL_loss | decoded_token`\n")
        lines.append("KL is over the full next-token distribution (not a specific target token).")
        lines.append("Computed at every non-padding position.\n")
        lines.append("```")

        for pos in range(padded_len):
            token_id = input_ids[pos]
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
            decoded = decoded.replace("\n", "\\n").replace("\t", "\\t")
            attn_val = attn[pos]

            kl_active = "YES" if mask_row[pos] else "NO "

            if not attn_val:
                label = f"pos {pos:4d} | attn=0 | KL={kl_active} | [PAD]"
            else:
                label = f"pos {pos:4d} | attn=1 | KL={kl_active} | {decoded}"

            lines.append(label)

        lines.append("```\n")

        # Summary
        if pad_count > 0:
            lines.append(f"- Padding tokens: {pad_count} (positions 0-{pad_count - 1})")
        else:
            lines.append("- Padding tokens: 0")
        n_kl = sum(mask_row)
        lines.append(f"- Positions with KL loss: {n_kl}")
        lines.append(f"- Positions without KL loss: {padded_len - n_kl} (padding)")
        lines.append("")

    output_path = Path(args.output)
    output_path.write_text("\n".join(lines))
    print(f"Wrote {len(lines)} lines to {output_path}")


if __name__ == "__main__":
    main()
