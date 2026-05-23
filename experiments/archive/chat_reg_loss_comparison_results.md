# Chat Regularization Loss Comparison

Comparing two checkpoints trained with the same AO data mix, differing only in whether chat regularization was applied during training.

## Setup

- **Validation data**: 1k fresh entries from lmsys-chat-1m (seed=9999, skipping first 200k English entries to avoid overlap with training data). Saved at `data_pipelines/chat_regularization/Qwen3-8B/t1_first_user_think50_1k_val.json`.
- **Loss**: Standard causal LM loss on response tokens (prompt tokens masked with -100).
- **KL**: D_KL(base || checkpoint), computed per-position over the full vocabulary at response token positions, then averaged.
- **Checkpoints compared**:
  - Base: `Qwen/Qwen3-8B` (no LoRA)
  - No chat-reg: `checkpoints/500k_pl_31k_spqav2_199k_sqav3_126k_cls/final` (lora_r=64)
  - Chat-reg: `checkpoints/500k_pl_31k_spqav2_199k_sqav3_126k_cls_100k_chatreg/final` (lora_r=64, chat_reg_every_9_ao_updates, weight=1.0)

## Results

| Model | Loss | dLoss vs base | KL(full) | KL(first 5 tokens) |
|---|---|---|---|---|
| Base model | 0.4808 | — | — | — |
| No chat-reg LoRA | 0.6034 | +0.1226 | 0.1221 | 0.9593 |
| Chat-reg LoRA | 0.4895 | +0.0087 | 0.0088 | 0.0155 |

Chat-reg effect: -0.1138 loss, -0.1134 KL (full), -0.9438 KL (first 5). Roughly 14x less drift from base on full-response KL, 62x less on early tokens.

## Interpretation

- The full-response KL (0.12) significantly understates the no-chatreg model's divergence. KL on the first 5 response tokens is **0.96 nats** — nearly 8x the full-response average. The model diverges most at the start of the response, then snaps back toward base behavior as it settles into the transcript. This means the full-response KL is diluted by hundreds of later tokens where the model has already "recovered."
- 0.12 KL (full) is in the range of a moderate RLHF fine-tune — noticeable but not catastrophic. But 0.96 KL on the opening tokens is substantial — this is where the model decides its initial direction, and large divergence here compounds in free generation.
- KL is a per-token metric conditioned on the same prefix. In free generation, small per-token differences compound as the model's own samples shift the context. The actual behavioral divergence during generation is likely larger than even the early-token KL suggests, since in free generation the divergent early tokens shift the context for everything that follows.
- The loss metric (measured on the base model's own rollouts) may be more practically meaningful: the no-chat-reg model is measurably worse at continuing text the base model would write (0.48 -> 0.60), suggesting noticeably different generation behavior.
- The chat-reg model is nearly indistinguishable from base on all metrics (0.48 -> 0.49 loss, 0.015 early KL).

## Scripts

- `experiments/chat_reg_loss_comparison.py` — main comparison script
- `experiments/gen_chat_reg_val.py` — one-off script to generate the 1k validation set
