# Open-Ended Linear Probe Report

Date: 2026-03-30 UTC

## Summary

I ran linear-probe ceiling experiments for the four open-ended AO binary tasks:

- `sycophancy_no_cot`
- `sycophancy_cot`
- `mmlu_pre_answer`
- `mmlu_post_answer`

Main result: the best linear probe beat the current AO ROC-AUC on all four tasks. The best representation in this run was `mean_all` for every task.

## Final Ceiling Results

| Task | AO ROC-AUC | Best Probe ROC-AUC | Delta | Best Pooling | Best Hyperparameters |
| --- | ---: | ---: | ---: | --- | --- |
| `sycophancy_no_cot` | 0.4737 | 0.9306 | +0.4569 | `mean_all` | `epochs=100`, `lr=3e-4`, `wd=0.0` |
| `sycophancy_cot` | 0.84535 | 0.9544 | +0.10905 | `mean_all` | `epochs=200`, `lr=1e-4`, `wd=0.0` |
| `mmlu_pre_answer` | 0.6372 | 0.8796 | +0.2424 | `mean_all` | `epochs=25`, `lr=3e-4`, `wd=0.0` |
| `mmlu_post_answer` | 0.6250 | 0.8668 | +0.2418 | `mean_all` | `epochs=25`, `lr=3e-4`, `wd=0.0` |

Best-test balanced accuracy / accuracy at those probe settings:

- `sycophancy_no_cot`: `0.7750 / 0.7750`
- `sycophancy_cot`: `0.7950 / 0.7950`
- `mmlu_pre_answer`: `0.8000 / 0.8000`
- `mmlu_post_answer`: `0.7900 / 0.7900`

## What I Ran

### Feature collection

I materialized probe feature caches for both pooling choices:

- `mean_concat_layers`
- `mean_all`

using the AO-matched layer combo `[25, 50, 75]`.

### Probe sweep

Full grid for each task/pooling sweep:

- `epochs`: `25, 50, 100, 200, 400`
- `learning_rate`: `1e-4, 3e-4, 1e-3, 3e-3, 1e-2`
- `weight_decay`: `0.0, 1e-5, 1e-4, 1e-3, 1e-2`
- `batch_size`: `1024`
- seed: `42`

This was a single-seed sweep, optimizing directly on the eval test set because the goal here was a ceiling estimate rather than a clean model-selection protocol.

## Dataset / Split Details

### Sycophancy

Used the exact script logic in `experiments/open_ended_linear_probe.py`:

- test set: balanced `100` positive / `100` negative
- seed: `42`
- `min_neutral_consistency=0.8`
- group holdout enabled at the source-entry level

Train sizes after reconstruction:

- `sycophancy_no_cot`: `1008` train (`328` positive, `680` negative)
- `sycophancy_cot`: `742` train (`86` positive, `656` negative)

### MMLU

Test set:

- exact AO eval JSON: `data_pipelines/mmlu_prediction/mmlu_prediction_eval_dataset.json`
- `100` examples (`50` correct, `50` incorrect)

Train set:

- generated tonight from `cais/mmlu` `validation`
- greedy Qwen answers at temperature `0`
- parseable validation total: `1434`
- full validation accuracy: `0.7364`
- balanced correctness subset saved to:
  `data_pipelines/mmlu_prediction/mmlu_prediction_probe_train_validation_balanced.json`
- final train size: `756` (`378` correct, `378` incorrect)

## Pooling Comparison

Best `mean_concat_layers` results:

- `sycophancy_no_cot`: `0.9266`
- `sycophancy_cot`: `0.9489`
- `mmlu_pre_answer`: `0.8608`
- `mmlu_post_answer`: `0.8580`

Best `mean_all` results:

- `sycophancy_no_cot`: `0.9306`
- `sycophancy_cot`: `0.9544`
- `mmlu_pre_answer`: `0.8796`
- `mmlu_post_answer`: `0.8668`

In this run, `mean_all` beat `mean_concat_layers` on all four tasks.

## Main Caveats

- This is a test-set ceiling run, not a clean validation-selected estimate.
- MMLU train data is from `validation`, not a larger `auxiliary_train` build. The ceiling may move upward with more train data.
- `sycophancy_cot` train is heavily imbalanced (`86` positive vs `656` negative). I kept the script-faithful split rather than introducing class weighting.
- Probe training was single-seed only.

## Artifacts

Report:

- `experiments/open_ended_linear_probe_results/linear_probe_report.md`

Generated MMLU train dataset:

- `data_pipelines/mmlu_prediction/mmlu_prediction_probe_train_validation_balanced.json`

Best-result sweep JSONs:

- `experiments/open_ended_linear_probe_results/sweeps/sycophancy_no_cot_mean_all_grid.json`
- `experiments/open_ended_linear_probe_results/sweeps/sycophancy_cot_mean_all_grid.json`
- `experiments/open_ended_linear_probe_results/sweeps/mmlu_pre_answer_mean_all_grid.json`
- `experiments/open_ended_linear_probe_results/sweeps/mmlu_post_answer_mean_all_grid.json`

Comparison sweep JSONs:

- `experiments/open_ended_linear_probe_results/sweeps/sycophancy_no_cot_mean_concat_layers_grid.json`
- `experiments/open_ended_linear_probe_results/sweeps/sycophancy_cot_mean_concat_layers_grid.json`
- `experiments/open_ended_linear_probe_results/sweeps/mmlu_pre_answer_mean_concat_layers_grid.json`
- `experiments/open_ended_linear_probe_results/sweeps/mmlu_post_answer_mean_concat_layers_grid.json`

Logs:

- `experiments/open_ended_linear_probe_results/logs/`

## Short Take

The open-ended AO tasks are very far from saturating the linear information available in these pooled activations. The biggest gap is `sycophancy_no_cot` (`0.4737` AO vs `0.9306` probe). Even the strongest AO here, `sycophancy_cot`, still trails the best linear probe by about `0.109` AUC.
