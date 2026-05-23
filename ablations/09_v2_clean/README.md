# 09 — V2 clean ablations (current paper-grade ablations)

## Default recipe
- Qwen3-8B base
- rsLoRA `r=128, α=16`, dropout=0, target=all-linear
- lr=3e-4 (rsLoRA-equivalent of vanilla 2e-4 baseline)
- batch_size=16
- **Single layer [22]** (the median of the 60-65% depth plateau)
- **10M target-token strict budget** (training stops when cumulative loss-contributing tokens ≥ 10M; checkpoint saved at that moment)
- Mix (token-balanced 45/45/10):
  - 4.5M tokens from chunked-convqa-haiku (~44k examples)
  - 4.5M tokens from past_lens on cot-v5 (~177k examples; true past+future, no vLLM, `future_corpus_only=True`)
  - 1.0M tokens from classification (10 datasets × 2 entries, 16,667 per entry — oversamples 1,588 unique rows ~10× with different position/window samplings)

## Ablations — each changes ONE knob from the default

| ablation | changed knob |
|---|---|
| layer_count_single_L21 | act_layers = [21] |
| layer_count_single_L23 | act_layers = [23] |
| layer_count_3layer | act_layers = [21, 22, 23] |
| layer_count_5layer | act_layers = [21, 22, 23, 24, 25] |
| layer_count_7layer | act_layers = [21..27] |
| direction_past_only | past_lens directions = ["past"] |
| direction_future_only | past_lens directions = ["future"] |
| direction_past_future | past_lens directions = ["past", "future"] (= default) |
| corpus_cotv5 | past_lens corpus = cot-v5 (= default) |
| corpus_finefineweb | past_lens corpus = m-a-p/FineFineWeb |
| corpus_fineweb | past_lens corpus = HuggingFaceFW/fineweb |
| no_convqa_haiku | drop chunked-convqa-haiku from mix |
| plus_latentqa | add 60k LatentQA entries |

## Token-budget patch
`nl_probes/sft.py` was patched to add `cfg.max_target_tokens` (set to 10_000_000 in all V2 configs). After every training batch the trainer accumulates target-token count (loss-contributing tokens, i.e. labels != -100) and when the threshold is crossed, it (a) immediately saves the checkpoint to `<save_dir>/token_budget_10M/`, (b) sets `token_budget_hit=True`, and (c) breaks out of all training loops. The standard `<save_dir>/final/` checkpoint also gets saved by the existing post-train code path, so each run leaves both — `token_budget_10M/` (atomic save at the 10M mark) and `final/` (overwritten same weights from the main loop exit).

## Why 10M tokens
- The original AO paper trained on ~65M tokens — for ablations we don't need full convergence, just enough signal to compare recipes.
- 10M tokens ≈ 45 min per run on a B200 (single-layer). 14 ablations × 45 min / 4 GPUs ≈ 3h wall-time.
- Once the winning V2 recipe is identified from ablations, a final "headline" run can be scaled to 30M+ tokens.
