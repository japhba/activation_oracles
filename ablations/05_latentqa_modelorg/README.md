# 05 — LatentQA model-org ablations (Phase 4, O-Q)

Tests whether adding LatentQA training data lifts taboo/personaqa specifically.

| tag | swap | result |
|---|---|---|
| O_with_latentqa_60k | + 60k LatentQA on multi[21,22,23] | +0.278 |
| P_with_latentqa_120k_budget | + 60k LatentQA, 120k total budget | +0.318 |
| Q_no_classification_latentqa_heavy | drop classification, heavy LatentQA | +0.297 |

Key finding: LatentQA lifts personaqa modestly (+0.10 → +0.16) but
hurts overall if you don't also bump the budget. Best balance is P.
