# 08 — Corpus mix + dataset swap (Phase 7+, in progress)

| tag | swap | status |
|---|---|---|
| AA_5layer_480k | 5-layer + 480k examples (push scale further) | running |
| BB_5layer_320k_lqa30k | W + Y combined (5-layer + 320k + 30k LatentQA) | running |
| CC_6layer_320k | 6-layer [21..26] + 320k | running |
| DD_5layer_320k_doubleconvqa | + non-haiku ConvQA (cds-jb/cot-oracle-convqa-chunked) | gen running |
| EE_5layer_320k_cotv5_ffw_mix | past_lens corpus = cot-v5 50% + FineFineWeb 50% | queued |
| FF (planned) | past_lens corpus = cot-v5 50% + fineweb 50% | queued |
| GG (planned, token-matched) | chunked-convqa-haiku only (no LatentQA), strict token-match to HH | queued |
| HH (planned, token-matched) | LatentQA only (no chunked-convqa-haiku), strict token-match to GG | queued |
| II (planned) | single-layer L21 baseline for "multi vs single" plot | queued |
| JJ (planned) | single-layer L22 baseline for "multi vs single" plot | queued |
