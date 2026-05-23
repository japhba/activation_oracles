"""
Compare chat regularization loss and KL divergence across checkpoints.

Loads pre-generated validation data, then measures:
  - Standard LM loss for each model
  - KL divergence from base model: D_KL(base || checkpoint)

Models compared:
  1. Base Qwen3-8B (no LoRA)
  2. Non-chatreg checkpoint (final)
  3. Chatreg checkpoint (final)

Usage:
    .venv/bin/python experiments/chat_reg_loss_comparison.py
"""

import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-8B"
# If OOM at batch_size=1, refactor to use a single base model with LoRA toggle
# (PEFT's model.disable_adapter()) instead of loading two full model copies.
BATCH_SIZE = 8
FIRST_N_TOKENS = 5  # for early-response KL analysis

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_NO_CHATREG = str(REPO_ROOT / "checkpoints/500k_pl_31k_spqav2_199k_sqav3_126k_cls/final")
CHECKPOINT_CHATREG = str(REPO_ROOT / "checkpoints/500k_pl_31k_spqav2_199k_sqav3_126k_cls_100k_chatreg/final")

VAL_DATA_PATH = Path("data_pipelines/chat_regularization/Qwen3-8B/t1_first_user_think50_1k_val.json")


def load_val_data():
    data = json.loads(VAL_DATA_PATH.read_text())
    entries = data["entries"]
    print(f"Loaded {len(entries)} validation entries from {VAL_DATA_PATH}")
    return entries


def build_batch(entries, tokenizer, device, max_response_tokens=None):
    """Build a padded batch. Returns input_ids, labels, attention_mask, response_mask.

    If max_response_tokens is set, only the first N response tokens are marked
    in response_mask (for early-token KL analysis). input_ids/labels/attn_mask
    are unchanged — the full sequence is still fed through the model.
    """
    batch_input_ids = []
    batch_labels = []
    for e in entries:
        input_ids = e["prompt_token_ids"] + e["response_token_ids"]
        labels = [-100] * e["prompt_len"] + e["response_token_ids"]
        batch_input_ids.append(input_ids)
        batch_labels.append(labels)

    max_len = max(len(ids) for ids in batch_input_ids)
    pad_id = tokenizer.pad_token_id

    padded_ids = []
    padded_labels = []
    attn_masks = []
    response_masks = []
    for ids, labs in zip(batch_input_ids, batch_labels):
        pad_len = max_len - len(ids)
        padded_ids.append([pad_id] * pad_len + ids)
        padded_labels.append([-100] * pad_len + labs)
        attn_masks.append([0] * pad_len + [1] * len(ids))
        # Build response mask, optionally limited to first N response tokens
        raw_mask = [0 if l == -100 else 1 for l in labs]
        if max_response_tokens is not None:
            count = 0
            for j in range(len(raw_mask)):
                if raw_mask[j] == 1:
                    count += 1
                    if count > max_response_tokens:
                        raw_mask[j] = 0
        response_masks.append([0] * pad_len + raw_mask)

    return (
        torch.tensor(padded_ids, dtype=torch.long, device=device),
        torch.tensor(padded_labels, dtype=torch.long, device=device),
        torch.tensor(attn_masks, dtype=torch.bool, device=device),
        torch.tensor(response_masks, dtype=torch.bool, device=device),
    )


def compute_loss(model, tokenizer, entries, device, batch_size=BATCH_SIZE):
    """Compute mean LM loss over entries (prompt tokens masked)."""
    model.eval()
    total_loss = 0.0
    total_response_tokens = 0

    for i in tqdm(range(0, len(entries), batch_size), desc="Computing loss"):
        batch_entries = entries[i : i + batch_size]
        input_ids, labels, attn_mask, _ = build_batch(batch_entries, tokenizer, device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)

        n_response_tokens = sum(e["response_len"] for e in batch_entries)
        total_loss += out.loss.item() * n_response_tokens
        total_response_tokens += n_response_tokens

    return total_loss / total_response_tokens


def compute_loss_and_kl(ckpt_model, base_model, tokenizer, entries, device, batch_size=BATCH_SIZE):
    """Compute checkpoint loss AND D_KL(base || checkpoint) over response positions.

    KL is computed per-position over the full vocabulary using shifted logits
    (position i predicts token i+1) to match HF's loss computation.

    Returns (mean_loss, mean_kl_full, mean_kl_first_n) where first_n uses FIRST_N_TOKENS.
    """
    ckpt_model.eval()
    base_model.eval()

    total_loss = 0.0
    total_kl = 0.0
    total_kl_first_n = 0.0
    total_response_tokens = 0
    total_first_n_tokens = 0

    for i in tqdm(range(0, len(entries), batch_size), desc="Computing loss + KL"):
        batch_entries = entries[i : i + batch_size]
        input_ids, labels, attn_mask, response_mask = build_batch(
            batch_entries, tokenizer, device
        )
        # Build first-N mask from full response_mask: keep only first FIRST_N_TOKENS 1s per row
        response_mask_first_n = torch.zeros_like(response_mask)
        for row_idx in range(response_mask.size(0)):
            count = 0
            for col_idx in range(response_mask.size(1)):
                if response_mask[row_idx, col_idx]:
                    count += 1
                    if count <= FIRST_N_TOKENS:
                        response_mask_first_n[row_idx, col_idx] = True
                    else:
                        break

        with torch.no_grad():
            ckpt_out = ckpt_model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            base_out = base_model(input_ids=input_ids, attention_mask=attn_mask)

        # Loss
        n_response_tokens = sum(e["response_len"] for e in batch_entries)
        total_loss += ckpt_out.loss.item() * n_response_tokens

        # KL divergence: shift logits to align with labels
        ckpt_logits = ckpt_out.logits[:, :-1, :].float()
        base_logits = base_out.logits[:, :-1, :].float()
        shifted_response_mask = response_mask[:, 1:]
        shifted_response_mask_first_n = response_mask_first_n[:, 1:]

        base_log_probs = F.log_softmax(base_logits, dim=-1)
        ckpt_log_probs = F.log_softmax(ckpt_logits, dim=-1)

        # D_KL(base || ckpt) = sum_v p_base(v) * (log p_base(v) - log p_ckpt(v))
        base_probs = base_log_probs.exp()
        kl_per_position = (base_probs * (base_log_probs - ckpt_log_probs)).sum(dim=-1)

        masked_kl = (kl_per_position * shifted_response_mask).sum().item()
        total_kl += masked_kl
        total_response_tokens += n_response_tokens

        # First-N KL
        masked_kl_first_n = (kl_per_position * shifted_response_mask_first_n).sum().item()
        total_kl_first_n += masked_kl_first_n
        total_first_n_tokens += shifted_response_mask_first_n.sum().item()

        del ckpt_logits, base_logits, base_log_probs, ckpt_log_probs, base_probs, kl_per_position

    mean_loss = total_loss / total_response_tokens
    mean_kl = total_kl / total_response_tokens
    mean_kl_first_n = total_kl_first_n / total_first_n_tokens if total_first_n_tokens > 0 else 0.0
    return mean_loss, mean_kl, mean_kl_first_n


def main():
    entries = load_val_data()

    prompt_lens = [e["prompt_len"] for e in entries]
    response_lens = [e["response_len"] for e in entries]
    print(f"Prompt lengths: min={min(prompt_lens)}, max={max(prompt_lens)}, mean={sum(prompt_lens)/len(prompt_lens):.0f}")
    print(f"Response lengths: min={min(response_lens)}, max={max(response_lens)}, mean={sum(response_lens)/len(response_lens):.0f}")

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Base model loss
    print(f"\nLoading base model {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=device
    )

    print("\n--- Base model (no LoRA) ---")
    base_loss = compute_loss(base_model, tokenizer, entries, device)
    print(f"Loss: {base_loss:.4f}")

    # Non-chatreg checkpoint
    print(f"\n--- Loading LoRA: {CHECKPOINT_NO_CHATREG} ---")
    ckpt_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=device
    )
    ckpt_model = PeftModel.from_pretrained(ckpt_model, CHECKPOINT_NO_CHATREG)
    ckpt_model = ckpt_model.merge_and_unload()

    no_chatreg_loss, no_chatreg_kl, no_chatreg_kl_first_n = compute_loss_and_kl(
        ckpt_model, base_model, tokenizer, entries, device
    )
    print(f"Loss: {no_chatreg_loss:.4f}")
    print(f"KL(base || this): {no_chatreg_kl:.6f}")
    print(f"KL first {FIRST_N_TOKENS} tokens: {no_chatreg_kl_first_n:.6f}")

    del ckpt_model
    gc.collect()
    torch.cuda.empty_cache()

    # Chatreg checkpoint
    print(f"\n--- Loading LoRA: {CHECKPOINT_CHATREG} ---")
    ckpt_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=device
    )
    ckpt_model = PeftModel.from_pretrained(ckpt_model, CHECKPOINT_CHATREG)
    ckpt_model = ckpt_model.merge_and_unload()

    chatreg_loss, chatreg_kl, chatreg_kl_first_n = compute_loss_and_kl(
        ckpt_model, base_model, tokenizer, entries, device
    )
    print(f"Loss: {chatreg_loss:.4f}")
    print(f"KL(base || this): {chatreg_kl:.6f}")
    print(f"KL first {FIRST_N_TOKENS} tokens: {chatreg_kl_first_n:.6f}")

    del ckpt_model, base_model
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 85)
    print("SUMMARY")
    print("=" * 85)
    print(f"{'Model':<25} {'Loss':>8} {'dLoss':>8} {'KL(full)':>10} {'KL(first ' + str(FIRST_N_TOKENS) + ')':>12}")
    print("-" * 85)
    print(f"{'Base model':<25} {base_loss:>8.4f} {'---':>8} {'---':>10} {'---':>12}")
    print(f"{'No chat-reg LoRA':<25} {no_chatreg_loss:>8.4f} {no_chatreg_loss - base_loss:>+8.4f} {no_chatreg_kl:>10.6f} {no_chatreg_kl_first_n:>12.6f}")
    print(f"{'Chat-reg LoRA':<25} {chatreg_loss:>8.4f} {chatreg_loss - base_loss:>+8.4f} {chatreg_kl:>10.6f} {chatreg_kl_first_n:>12.6f}")
    print("-" * 85)
    print(f"Chat-reg effect on loss:        {chatreg_loss - no_chatreg_loss:+.4f}")
    print(f"Chat-reg effect on KL (full):   {chatreg_kl - no_chatreg_kl:+.6f}")
    print(f"Chat-reg effect on KL (first {FIRST_N_TOKENS}): {chatreg_kl_first_n - no_chatreg_kl_first_n:+.6f}")


if __name__ == "__main__":
    main()
