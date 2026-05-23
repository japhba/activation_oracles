# Training Data Loss Diagnostic: 5k Subset

## Setup

- Checkpoint: `checkpoints/500k_pl_31k_spqav2_199k_sqav3_126k_cls/final`
- Subset size: `5000`
- Batch size: `16`
- Length trim: `p=0.999`, threshold `1587`
- Source of truth: checkpoint-embedded `ao_config.json`
- Classification rows are collapsed into one `classification` group in the summary below.

Subset composition:

| Dataset group | n |
|---|---:|
| `past_lens` | 2500 |
| `synthetic_qa` | 1198 |
| `classification` | 1302 |

## Baselines Tried

- `real`: real materialized steering vectors from each example's own context.
- `random_direction`: Gaussian random steering vectors with the same shape as the real steering tensor.
- `zero_vector`: all-zero steering vectors. On a 32-example smoke run, this matched `no_hook` exactly on summary metrics.
- `batch_row_sample`: per layer, sample rows from other examples in the same batch and stitch them together.
- `batch_repeat_single_donor`: choose one donor example from the same batch, then tile/truncate its layer blocks to the recipient length.
- `batch_repeat_single_donor_shared_offset`: same as above, but use one shared circular offset across all layers.

Conventions:

- `delta = real_mean_nll - baseline_mean_nll`
- More negative is better for real activations.
- `real_better_rate` means fraction of examples with `delta < 0`.

## Overall Mean Losses And Win Rates

| Baseline | Baseline mean NLL | Mean delta | Median delta | Real better | Real not better |
|---|---:|---:|---:|---:|---:|
| `random_direction` | 2.6665 | -1.4101 | -0.8716 | 95.56% | 4.44% |
| `zero_vector` | 2.8050 | -1.5487 | -1.0076 | 95.84% | 4.16% |
| `batch_row_sample` | 2.9282 | -1.6718 | -1.0702 | 94.30% | 5.70% |
| `batch_repeat_single_donor` | 3.1685 | -1.9121 | -1.2235 | 93.66% | 6.34% |
| `batch_repeat_single_donor_shared_offset` | 3.1786 | -1.9222 | -1.2406 | 93.76% | 6.24% |

Real mean NLL on this 5k subset: `1.2564`

## By Dataset Group

### Mean NLLs

| Dataset | Real | Random | Zero | Row sample | Repeat donor | Repeat donor shared |
|---|---:|---:|---:|---:|---:|---:|
| `past_lens` | 1.9033 | 4.3613 | 4.6068 | 4.7847 | 5.2126 | 5.2271 |
| `synthetic_qa` | 1.1784 | 1.7135 | 1.7735 | 1.7931 | 1.8586 | 1.8589 |
| `classification` | 0.0859 | 0.2892 | 0.2946 | 0.4077 | 0.4489 | 0.4596 |

### Real Better Rate

| Dataset | Vs random | Vs zero | Vs row sample | Vs repeat donor | Vs repeat donor shared |
|---|---:|---:|---:|---:|---:|
| `past_lens` | 99.92% | 99.92% | 99.92% | 99.96% | 99.96% |
| `synthetic_qa` | 98.50% | 98.91% | 99.00% | 99.42% | 99.17% |
| `classification` | 84.49% | 85.18% | 79.19% | 76.27% | 76.88% |

## Delta Quantiles

### `delta_vs_random = real - random`

| Dataset | q01 | q10 | q25 | q50 | q75 | q90 | q95 | q99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `overall` | -9.1214 | -3.2893 | -1.8165 | -0.8716 | -0.2793 | -0.0737 | -0.0044 | 0.2575 |
| `past_lens` | -11.4434 | -4.7794 | -2.9273 | -1.8113 | -1.2227 | -0.8873 | -0.7282 | -0.4410 |
| `synthetic_qa` | -1.7353 | -1.0426 | -0.7153 | -0.4628 | -0.2603 | -0.1403 | -0.0800 | 0.0208 |
| `classification` | -1.1355 | -0.5968 | -0.3215 | -0.1581 | -0.0365 | 0.0492 | 0.1929 | 0.6770 |

### `delta_vs_zero_vector = real - zero_vector`

| Dataset | q01 | q10 | q25 | q50 | q75 | q90 | q95 | q99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `overall` | -9.2593 | -3.6279 | -2.0519 | -1.0076 | -0.3471 | -0.0813 | -0.0099 | 0.2500 |
| `past_lens` | -10.8552 | -5.0835 | -3.2352 | -2.0440 | -1.4228 | -1.0732 | -0.9203 | -0.6409 |
| `synthetic_qa` | -1.7875 | -1.1237 | -0.7905 | -0.5318 | -0.3107 | -0.1664 | -0.1046 | 0.0032 |
| `classification` | -1.0547 | -0.5679 | -0.3778 | -0.1521 | -0.0371 | 0.0569 | 0.1930 | 0.6921 |

## Token-Level Aggregates

These are token-level, not example-level.

| Dataset | Mean token delta vs random | Token real-better rate vs random | Mean token delta vs zero | Token real-better rate vs zero |
|---|---:|---:|---:|---:|
| `past_lens` | -1.6254 | 81.33% | -1.8462 | 84.04% |
| `synthetic_qa` | -0.5292 | 74.04% | -0.5870 | 73.85% |
| `classification` | -0.2034 | 66.05% | -0.2087 | 68.23% |

## Quick Interpretation

- Real activations are clearly useful relative to `random_direction` and `zero_vector`.
- `zero_vector` is slightly harsher than `random_direction`.
- All of the structured wrong-activation baselines are harsher than `random_direction`, and usually harsher than `zero_vector`.
- `past_lens` is the clearest case: almost every example prefers real activations, and the whole distribution is shifted strongly negative.
- `synthetic_qa` also clearly benefits from real activations, but with a smaller effect size.
- Classification still benefits on average, but has a noticeably larger tail where real activations do not help.

## Plots

5k PNG histograms:

- `png_plots/past_lens_delta_vs_random_hist.png`
- `png_plots/synthetic_qa_delta_vs_random_hist.png`
- `png_plots/classification_delta_vs_random_hist.png`
- `png_plots/combined_delta_vs_random_histograms.png`
- `png_plots/past_lens_delta_vs_zero_vector_hist.png`
- `png_plots/synthetic_qa_delta_vs_zero_vector_hist.png`
- `png_plots/classification_delta_vs_zero_vector_hist.png`
- `png_plots/combined_delta_vs_zero_vector_histograms.png`
