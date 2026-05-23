# DPO for Activation Oracles — simple proof of concept

**Goal**: take an existing AO checkpoint (e.g. `multi5_sonnet_lr3em5` at +0.404 AObench, or
`multi5_sonnet_lr1em5` at +0.399/+0.302 morg) and run a small DPO pass to push
its outputs toward more factually-correct descriptions of what an activation
encodes.

**Why this might help**: the AO is SFT'd to predict literal next-k tokens
(`past_lens`) and chunked-convQA target responses. It is not directly trained
on the binary correctness of its answers. DPO with verifiable preferences
should shift it toward "say the true thing about the activation" instead of
"say a fluent thing that happens to match the training distribution".

## Setup

Held constant from the SFT recipe:
- Backbone: Qwen3-8B, rsLoRA r=128 α=16
- Hook layer 1, norm-matched injection
- Activation source: `ceselder/cot-oracle-corpus-v5` (Qwen3-8B's own CoT rollouts)
- Activation layer: matches the source SFT checkpoint (L21 single OR [21..25] multi)

DPO-specific:
- 20 prompt templates with programmatic ground truth from the source context
- ~12-13 datapoints per template → ~250 DPO pairs
- Chosen = correct templated answer + Sonnet 4.6 polish for fluency
- Rejected = plausible-but-wrong templated answer + Sonnet 4.6 polish
- DPO via TRL's `DPOTrainer`, β=0.1, lr=5e-7, ~200 steps

## Files

- `prompts.py` — 20 prompt templates (question + ground-truth fn + chosen/rejected fns)
- `generate_dpo_pairs.py` — sample corpus, compute GT, polish with Sonnet, write jsonl
- `train_dpo.py` — load AO ckpt, run DPOTrainer, save
- `data/` — generated DPO datasets
- `checkpoints/` — output

## Order of operations

1. `python prompts.py --dry-run` — verify ground-truth extractors on 50 sample contexts
2. `python generate_dpo_pairs.py --n-per-template 13 --out data/dpo_v1.jsonl`
3. inspect `data/dpo_v1.jsonl` manually for ~10 pairs
4. `python train_dpo.py --data data/dpo_v1.jsonl --init multi5_sonnet_lr1em5 --out checkpoints/dpo_v1`
5. AObench the result, compare to init

## Caveats

- Templated chosen/rejected = somewhat stilted prose, may not generalize beyond template phrasing
- DPO at 250 pairs is small; expect noise. The user explicitly asked for "very simple DPO" first
- For a real paper run, we'd want either (a) AO-sampled candidates ranked by Sonnet, or (b) 2-5k pairs
