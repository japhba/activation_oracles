# 04 — Layer-selection + scaling ablations (Phase 3, I-N)

All on the best Phase 1+2 recipe (rsLoRA + lr=3e-5 + cot-v5 past_lens + on-policy w/ inject).

| tag | swap | result |
|---|---|---|
| I_5layer_21_25 | act_layer = [21,22,23,24,25] | +0.307 (was Phase-3 best) |
| J_4layer_19_21_23_25 | act_layer = [19,21,23,25] (interleaved) | +0.282 |
| K_single_L23 | act_layer = [23] (single) | +0.278 |
| L_more_train_160k | max_train_examples = 160k | +0.302 |
| M_pastlens_80k | past_lens num_train = 80k | +0.286 |
| N_lora_r128 | lora_r = 128 | +0.264 (HURT — overfit) |

Key finding: wider 5-layer + 2× train both help. More LoRA capacity hurts.
