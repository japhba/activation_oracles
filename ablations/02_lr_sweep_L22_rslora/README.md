# 02 — LR sweep on L22 with rsLoRA + past_lens(CoT v5)

Tests rsLoRA-equivalent learning rates {1e-5, 3e-5, 1e-4, 3e-4} for the rsLoRA scaling.
Single-layer L22, multi-layer baseline recipe. Past_lens corpus is CoT v5 (cot-oracle-corpus-v5).

Result: lr=3e-5 and 1e-4 tied at top (+0.28). lr=3e-4 collapsed (rsLoRA over-scaling).
