# Activation Oracle Ablations — Qwen3-8B

Each subfolder is one focused set of ablations. Configs live in `configs/`, the
Python builders / shell launchers live in `scripts/`. Run with the unified
launchers (`scripts/run_phase3.sh`, etc.) — they pass each config through
`nl_probes/sft.py` with the appropriate torchrun setup.

| folder | summary | best result |
|---|---|---|
| 01_layer_sweep | initial single-layer sweep at 5 depths | L22 +0.141 |
| 02_lr_sweep_L22_rslora | LR sweep with rsLoRA on single-layer L22 | lr=3e-5 +0.280 |
| 03_past_lens_modes | past_lens corpus + direction + on/off-policy ablations | baseline +0.292 |
| 04_layer_scale_ablations | layer-combo + scaling sweep (Phase 3) | I_5layer +0.307 |
| 05_latentqa_modelorg | add LatentQA for model-org evals | P +0.318 |
| 06_stacked_wins | combine Phase-3 & Phase-4 winners | R +0.357 |
| 07_push_past_036 | more layers / data / epochs | W_5layer_320k +0.360 |
| 08_corpus_mix_and_swap | corpus diversification & strict-token-match swaps | in progress |

The unified rsLoRA + 5-layer + cot-v5 past_lens + on-policy recipe + 320k examples
is the current winner. Phase 7+ refines around this.

Results land in `/workspace/repo/.../AObench/eval_results/` on the GPU box and
get rsync'd into `/home/celeste/shared/{phase3,phase4,phase5,phase6}_results/`.
Final plot: `/home/celeste/shared/full_aobench_heatmap.png`.
