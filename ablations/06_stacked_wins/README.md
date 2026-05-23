# 06 — Stack the wins (Phase 5, R-U)

Combines Phase 3 best (5-layer wider, 2× train) with Phase 4 wins (LatentQA).

| tag | swap | result |
|---|---|---|
| R_5layer_160k | 5-layer + 160k examples | +0.357 (NEW best at the time) |
| S_5layer_latentqa | 5-layer + LatentQA 60k | +0.296 |
| T_5layer_latentqa_160k | 5-layer + LatentQA 60k + 160k examples | +0.328 |
| U_5layer_lr5em5 | 5-layer + lr=5e-5 (between 3e-5 and 1e-4) | +0.305 |

Key finding: R = 5-layer + 160k wins overall. LatentQA HURTS the 5-layer + 160k combo.
