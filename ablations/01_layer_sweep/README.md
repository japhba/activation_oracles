# 01 — Single-layer sweep (initial)

5 single-layer AOs at depths 19, 22, 25, 28, 31 of Qwen3-8B.
Vanilla LoRA, lr=2e-4, no past_lens. Data: cot_oracle_convqa-haiku + classification.

Builder: scripts/_layer_sweep_template.py
Configs: configs/layer{19,22,25,28,31}.json

Key result: L22 won (+0.141), confirmed multi-layer plateau idea later.
