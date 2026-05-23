# Held-Out Loss Diagnostic: 5k Subset

## Setup

- Checkpoint: `checkpoints/500k_pl_31k_spqav2_199k_sqav3_126k_cls/final`
- Subset size: `5000`
- Batch size: `16`
- Length trim: `p=0.999`, threshold `1587`
- Allocation mode: `heldout_current_mix`
- Scoring modes: `real`, `random_direction`, `zero_vector`, `batch_row_sample`, `batch_repeat_single_donor`, `batch_repeat_single_donor_shared_offset`
- Held-out `past_lens` seed: `43`
- Held-out synthetic QA config: `checkpoints/Qwen3-8B_synthetic_qa_v2_only/final/ao_config.json`
- Classification rows are collapsed into one `classification` group in the summary below.

Subset composition:

| Dataset group | n |
| --- | ---: |
| `past_lens` | 2,500 |
| `synthetic_qa` | 1,198 |
| `classification` | 1,302 |

Held-out source allocation:

| Dataset | Loader variant | Source name | Split | Post-trim available | Sampled |
| --- | ---: | ---: | ---: | ---: | ---: |
| `classification_geometry_of_truth` | `classification_geometry_of_truth_qa_2_window_1_1_heldout_test` | `heldout_classification_test` | `test` | 500 | 107 |
| `classification_geometry_of_truth` | `classification_geometry_of_truth_qa_1_window_1_50_heldout_test` | `heldout_classification_test` | `test` | 250 | 79 |
| `classification_md_gender` | `classification_md_gender_qa_2_window_1_1_heldout_test` | `heldout_classification_test` | `test` | 500 | 107 |
| `classification_md_gender` | `classification_md_gender_qa_1_window_1_50_heldout_test` | `heldout_classification_test` | `test` | 250 | 79 |
| `classification_ner` | `classification_ner_qa_2_window_1_1_heldout_test` | `heldout_classification_test` | `test` | 500 | 107 |
| `classification_ner` | `classification_ner_qa_1_window_1_50_heldout_test` | `heldout_classification_test` | `test` | 250 | 79 |
| `classification_relations` | `classification_relations_qa_2_window_1_1_heldout_test` | `heldout_classification_test` | `test` | 500 | 107 |
| `classification_relations` | `classification_relations_qa_1_window_1_50_heldout_test` | `heldout_classification_test` | `test` | 250 | 79 |
| `classification_snli` | `classification_snli_qa_2_window_1_1_heldout_test` | `heldout_classification_test` | `test` | 500 | 107 |
| `classification_snli` | `classification_snli_qa_1_window_1_50_heldout_test` | `heldout_classification_test` | `test` | 250 | 79 |
| `classification_sst2` | `classification_sst2_qa_2_window_1_1_heldout_test` | `heldout_classification_test` | `test` | 500 | 107 |
| `classification_sst2` | `classification_sst2_qa_1_window_1_50_heldout_test` | `heldout_classification_test` | `test` | 250 | 79 |
| `classification_tense` | `classification_tense_qa_2_window_1_1_heldout_test` | `heldout_classification_test` | `test` | 500 | 107 |
| `classification_tense` | `classification_tense_qa_1_window_1_50_heldout_test` | `heldout_classification_test` | `test` | 250 | 79 |
| `past_lens` | `past_lens_kacts_1_1_ktokens_1_50_heldout_seed_43` | `heldout_past_lens_seed_43` | `train` | 1,314 | 1,250 |
| `past_lens` | `past_lens_kacts_1_50_ktokens_1_50_heldout_seed_43` | `heldout_past_lens_seed_43` | `train` | 1,314 | 1,250 |
| `synthetic_qa` | `synthetic_qa_artifacts_training_data_400000_heldout_v2` | `heldout_synthetic_qa_v2` | `train` | 274,282 | 1,198 |

## Overall Mean Losses And Win Rates

| Baseline | Baseline mean NLL | Mean delta | Median delta | Real better | Real not better |
| --- | ---: | ---: | ---: | ---: | ---: |
| `random_direction` | 2.7430 | -1.3261 | -0.7244 | 94.18% | 5.82% |
| `zero_vector` | 2.8722 | -1.4553 | -0.8141 | 94.82% | 5.18% |
| `batch_row_sample` | 2.9764 | -1.5595 | -0.9353 | 93.80% | 6.20% |
| `batch_repeat_single_donor` | 3.2158 | -1.7989 | -1.0993 | 92.94% | 7.06% |
| `batch_repeat_single_donor_shared_offset` | 3.2404 | -1.8235 | -1.0867 | 93.04% | 6.96% |

Real mean NLL on this subset: `1.4169`

## By Dataset Group

### Mean NLLs

| Dataset | Real | Random Direction | Zero Vector | Batch Row Sample | Batch Repeat Single Donor | Batch Repeat Single Donor Shared Offset |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `past_lens` | 2.0095 | 4.4403 | 4.6856 | 4.8370 | 5.2746 | 5.3283 |
| `synthetic_qa` | 1.6169 | 1.8681 | 1.8888 | 1.9057 | 1.9285 | 1.9286 |
| `classification` | 0.0951 | 0.2890 | 0.2951 | 0.3889 | 0.4473 | 0.4385 |

### Real Better Rate

| Dataset | Vs random direction | Vs zero vector | Vs batch row sample | Vs batch repeat single donor | Vs batch repeat single donor shared offset |
| --- | ---: | ---: | ---: | ---: | ---: |
| `past_lens` | 99.96% | 99.88% | 99.84% | 100.00% | 100.00% |
| `synthetic_qa` | 93.07% | 94.41% | 96.58% | 95.58% | 95.66% |
| `classification` | 84.10% | 85.48% | 79.65% | 76.96% | 77.27% |

## Delta Quantiles

### `delta_vs_random = real - random`

| Dataset | q01 | q10 | q25 | q50 | q75 | q90 | q95 | q99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `overall` | -8.7344 | -3.3238 | -1.7893 | -0.7244 | -0.1771 | -0.0339 | 0.0104 | 0.3034 |
| `past_lens` | -11.6407 | -4.6162 | -2.8140 | -1.7891 | -1.2347 | -0.8795 | -0.7085 | -0.4659 |
| `synthetic_qa` | -1.2077 | -0.5711 | -0.3445 | -0.1923 | -0.0784 | -0.0149 | 0.0094 | 0.0713 |
| `classification` | -1.1341 | -0.5625 | -0.3239 | -0.1634 | -0.0322 | 0.0675 | 0.2151 | 0.8043 |

### `delta_vs_zero_vector = real - zero_vector`

| Dataset | q01 | q10 | q25 | q50 | q75 | q90 | q95 | q99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `overall` | -8.9043 | -3.5803 | -2.0341 | -0.8141 | -0.1846 | -0.0416 | 0.0019 | 0.3038 |
| `past_lens` | -11.8998 | -4.9700 | -3.1862 | -2.0335 | -1.4288 | -1.0911 | -0.8915 | -0.5819 |
| `synthetic_qa` | -1.1725 | -0.5984 | -0.3785 | -0.2082 | -0.0951 | -0.0263 | 0.0028 | 0.0628 |
| `classification` | -1.0427 | -0.5904 | -0.3759 | -0.1421 | -0.0355 | 0.0581 | 0.2083 | 0.6619 |

## Token-Level Aggregates

These are token-level, not example-level.

| Dataset | Mean token delta vs random | Token real-better rate vs random | Mean token delta vs zero | Token real-better rate vs zero |
| --- | ---: | ---: | ---: | ---: |
| `past_lens` | -1.6174 | 80.47% | -1.8309 | 83.49% |
| `synthetic_qa` | -0.2489 | 65.78% | -0.2698 | 65.40% |
| `classification` | -0.1939 | 65.23% | -0.2000 | 68.33% |

## Quick Interpretation

- This run uses held-out data sources rather than the checkpoint's original train shards.
- Real activations are compared against wrong-activation baselines on the same held-out examples.
- More negative deltas mean real activations lowered loss relative to the baseline.
- The combined random-baseline histogram is the main summary figure for fast comparison across dataset groups.

## Plots

- `png_plots/past_lens_delta_vs_random_hist.png`
- `png_plots/synthetic_qa_delta_vs_random_hist.png`
- `png_plots/classification_delta_vs_random_hist.png`
- `png_plots/combined_delta_vs_random_histograms.png`
- `png_plots/past_lens_delta_vs_zero_vector_hist.png`
- `png_plots/synthetic_qa_delta_vs_zero_vector_hist.png`
- `png_plots/classification_delta_vs_zero_vector_hist.png`
- `png_plots/combined_delta_vs_zero_vector_histograms.png`
