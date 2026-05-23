# 07 — Push past +0.357 (Phase 6, V-Z)

Tries wider layers, more data, more epochs, smaller LatentQA, batch-size effect.

| tag | swap | result |
|---|---|---|
| V_7layer_160k | 7-layer [21..27] + 160k | +0.327 (best taboo +0.495) |
| W_5layer_320k | 5-layer + 320k examples | +0.360 (NEW best overall) |
| X_5layer_160k_2ep | 5-layer + 160k + 2 epochs | +0.311 |
| Y_5layer_160k_latentqa30k | 5-layer + 160k + small (30k) LatentQA | +0.334 (best model-org balance) |
| Z_5layer_160k_bs32 | 5-layer + 160k + bs=32 (vs R's bs=16) | +0.324 |

Key finding: W (5-layer + 320k) beats R — more data > more epochs (W beats X). V (7-layer) hurts overall but best taboo + personaqa.
