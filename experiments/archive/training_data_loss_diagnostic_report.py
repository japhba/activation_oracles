import argparse
import json
from pathlib import Path

import pandas as pd


DATASET_ORDER = ["past_lens", "synthetic_qa", "classification"]
BASELINE_ORDER = [
    "random_direction",
    "zero_vector",
    "batch_row_sample",
    "batch_repeat_single_donor",
    "batch_repeat_single_donor_shared_offset",
    "no_hook",
]
BASELINE_COLUMN_MAP = {
    "random_direction": {
        "mean": "random_mean_nll",
        "delta": "delta_mean_nll",
        "first1_delta": "delta_first1_mean_nll",
        "first3_delta": "delta_first3_mean_nll",
        "first5_delta": "delta_first5_mean_nll",
    },
    "zero_vector": {
        "mean": "zero_vector_mean_nll",
        "delta": "delta_vs_zero_vector_mean_nll",
        "first1_delta": "delta_vs_zero_vector_first1_mean_nll",
        "first3_delta": "delta_vs_zero_vector_first3_mean_nll",
        "first5_delta": "delta_vs_zero_vector_first5_mean_nll",
    },
    "batch_row_sample": {
        "mean": "batch_row_sample_mean_nll",
        "delta": "delta_vs_batch_row_sample_mean_nll",
        "first1_delta": "delta_vs_batch_row_sample_first1_mean_nll",
        "first3_delta": "delta_vs_batch_row_sample_first3_mean_nll",
        "first5_delta": "delta_vs_batch_row_sample_first5_mean_nll",
    },
    "batch_repeat_single_donor": {
        "mean": "batch_repeat_single_donor_mean_nll",
        "delta": "delta_vs_batch_repeat_single_donor_mean_nll",
        "first1_delta": "delta_vs_batch_repeat_single_donor_first1_mean_nll",
        "first3_delta": "delta_vs_batch_repeat_single_donor_first3_mean_nll",
        "first5_delta": "delta_vs_batch_repeat_single_donor_first5_mean_nll",
    },
    "batch_repeat_single_donor_shared_offset": {
        "mean": "batch_repeat_single_donor_shared_offset_mean_nll",
        "delta": "delta_vs_batch_repeat_single_donor_shared_offset_mean_nll",
        "first1_delta": "delta_vs_batch_repeat_single_donor_shared_offset_first1_mean_nll",
        "first3_delta": "delta_vs_batch_repeat_single_donor_shared_offset_first3_mean_nll",
        "first5_delta": "delta_vs_batch_repeat_single_donor_shared_offset_first5_mean_nll",
    },
    "no_hook": {
        "mean": "no_hook_mean_nll",
        "delta": "delta_vs_no_hook_mean_nll",
    },
}


def collapse_dataset_group(dataset_name: str) -> str:
    if dataset_name.startswith("classification_"):
        return "classification"
    return dataset_name


def fmt_pct(value: float) -> str:
    return f"{value:.2%}"


def fmt_float(value: float) -> str:
    return f"{value:.4f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def baseline_rows(per_example_df: pd.DataFrame) -> list[str]:
    rows = []
    for baseline_name in BASELINE_ORDER:
        spec = BASELINE_COLUMN_MAP[baseline_name]
        if spec["mean"] not in per_example_df.columns:
            continue

        delta = per_example_df[spec["delta"]]
        rows.append(
            [
                f"`{baseline_name}`",
                fmt_float(float(per_example_df[spec["mean"]].mean())),
                fmt_float(float(delta.mean())),
                fmt_float(float(delta.median())),
                fmt_pct(float((delta < 0.0).mean())),
                fmt_pct(float((delta >= 0.0).mean())),
            ]
        )
    return rows


def grouped_mean_rows(per_example_df: pd.DataFrame) -> list[list[str]]:
    rows = []
    for dataset_group in DATASET_ORDER:
        dataset_df = per_example_df[per_example_df["dataset_group"] == dataset_group]
        row = [f"`{dataset_group}`", fmt_float(float(dataset_df["real_mean_nll"].mean()))]
        for baseline_name in BASELINE_ORDER:
            spec = BASELINE_COLUMN_MAP[baseline_name]
            if spec["mean"] in per_example_df.columns:
                row.append(fmt_float(float(dataset_df[spec["mean"]].mean())))
        rows.append(row)
    return rows


def grouped_win_rate_rows(per_example_df: pd.DataFrame) -> list[list[str]]:
    rows = []
    for dataset_group in DATASET_ORDER:
        dataset_df = per_example_df[per_example_df["dataset_group"] == dataset_group]
        row = [f"`{dataset_group}`"]
        for baseline_name in BASELINE_ORDER:
            spec = BASELINE_COLUMN_MAP[baseline_name]
            if spec["mean"] in per_example_df.columns:
                row.append(fmt_pct(float((dataset_df[spec["delta"]] < 0.0).mean())))
        rows.append(row)
    return rows


def quantile_rows(per_example_df: pd.DataFrame, delta_column: str) -> list[list[str]]:
    quantiles = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    rows = []
    for dataset_group in ["overall"] + DATASET_ORDER:
        dataset_df = per_example_df if dataset_group == "overall" else per_example_df[per_example_df["dataset_group"] == dataset_group]
        series = dataset_df[delta_column]
        row = [f"`{dataset_group}`"]
        for quantile in quantiles:
            row.append(fmt_float(float(series.quantile(quantile))))
        rows.append(row)
    return rows


def token_rows(per_token_df: pd.DataFrame) -> list[list[str]]:
    rows = []
    for dataset_group in DATASET_ORDER:
        dataset_df = per_token_df[per_token_df["dataset_group"] == dataset_group]
        row = [
            f"`{dataset_group}`",
            fmt_float(float(dataset_df["delta_nll"].mean())),
            fmt_pct(float((dataset_df["delta_nll"] < 0.0).mean())),
        ]
        if "delta_vs_zero_vector_nll" in per_token_df.columns:
            row.append(fmt_float(float(dataset_df["delta_vs_zero_vector_nll"].mean())))
            row.append(fmt_pct(float((dataset_df["delta_vs_zero_vector_nll"] < 0.0).mean())))
        rows.append(row)
    return rows


def source_rows(allocation_df: pd.DataFrame) -> list[list[str]]:
    rows = []
    for _, row in allocation_df.sort_values(["dataset_name", "loader_index"]).iterrows():
        rows.append(
            [
                f"`{row['dataset_name']}`",
                f"`{row['loader_variant']}`",
                f"`{row['source_name']}`",
                f"`{row['split']}`",
                f"{int(row['kept_count']):,}",
                f"{int(row['allocation']):,}",
            ]
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    assert result_dir.exists(), f"Missing result dir: {result_dir}"

    per_example_df = pd.read_parquet(result_dir / "per_example.parquet")
    per_token_df = pd.read_parquet(result_dir / "per_token.parquet")
    summary = json.loads((result_dir / "summary.json").read_text())
    run_config = json.loads((result_dir / "run_config.json").read_text())
    allocation_df = pd.read_csv(result_dir / "subset_allocation.csv")

    per_example_df["dataset_group"] = per_example_df["dataset_name"].map(collapse_dataset_group)
    per_token_df["dataset_group"] = per_token_df["dataset_name"].map(collapse_dataset_group)

    dataset_group_counts = (
        per_example_df.groupby("dataset_group", dropna=False)["example_id"].count().reindex(DATASET_ORDER)
    )
    scoring_modes = ", ".join(f"`{mode}`" for mode in run_config["scoring_modes"])

    lines = []
    title = "Held-Out Loss Diagnostic: 5k Subset" if run_config["heldout_eval"] else "Training Data Loss Diagnostic: 5k Subset"
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Checkpoint: `{run_config['checkpoint_dir']}`")
    lines.append(f"- Subset size: `{run_config['subset_size']}`")
    lines.append(f"- Batch size: `{run_config['batch_size']}`")
    lines.append(f"- Length trim: `p={run_config['length_percentile']}`, threshold `{run_config['length_threshold']}`")
    lines.append(f"- Allocation mode: `{run_config['allocation_mode']}`")
    lines.append(f"- Scoring modes: {scoring_modes}")
    if run_config["heldout_eval"]:
        lines.append(f"- Held-out `past_lens` seed: `{run_config['heldout_past_lens_seed']}`")
        lines.append(f"- Held-out synthetic QA config: `{run_config['heldout_synthetic_qa_v2_config']}`")
    lines.append("- Classification rows are collapsed into one `classification` group in the summary below.")
    lines.append("")
    lines.append("Subset composition:")
    lines.append("")
    composition_rows = [[f"`{dataset_group}`", f"{int(dataset_group_counts[dataset_group]):,}"] for dataset_group in DATASET_ORDER]
    lines.append(markdown_table(["Dataset group", "n"], composition_rows))
    lines.append("")
    lines.append("Held-out source allocation:")
    lines.append("")
    lines.append(
        markdown_table(
            ["Dataset", "Loader variant", "Source name", "Split", "Post-trim available", "Sampled"],
            source_rows(allocation_df),
        )
    )
    lines.append("")
    lines.append("## Overall Mean Losses And Win Rates")
    lines.append("")
    lines.append(
        markdown_table(
            ["Baseline", "Baseline mean NLL", "Mean delta", "Median delta", "Real better", "Real not better"],
            baseline_rows(per_example_df),
        )
    )
    lines.append("")
    lines.append(f"Real mean NLL on this subset: `{fmt_float(summary['overall']['real_mean_nll'])}`")
    lines.append("")
    lines.append("## By Dataset Group")
    lines.append("")
    mean_headers = ["Dataset", "Real"] + [
        baseline_name.replace("_", " ").title()
        for baseline_name in BASELINE_ORDER
        if BASELINE_COLUMN_MAP[baseline_name]["mean"] in per_example_df.columns
    ]
    lines.append("### Mean NLLs")
    lines.append("")
    lines.append(markdown_table(mean_headers, grouped_mean_rows(per_example_df)))
    lines.append("")
    win_headers = ["Dataset"] + [
        f"Vs {baseline_name.replace('_', ' ')}"
        for baseline_name in BASELINE_ORDER
        if BASELINE_COLUMN_MAP[baseline_name]["mean"] in per_example_df.columns
    ]
    lines.append("### Real Better Rate")
    lines.append("")
    lines.append(markdown_table(win_headers, grouped_win_rate_rows(per_example_df)))
    lines.append("")

    delta_sections = [("delta_mean_nll", "`delta_vs_random = real - random`")]
    if "delta_vs_zero_vector_mean_nll" in per_example_df.columns:
        delta_sections.append(("delta_vs_zero_vector_mean_nll", "`delta_vs_zero_vector = real - zero_vector`"))

    lines.append("## Delta Quantiles")
    lines.append("")
    quantile_headers = ["Dataset", "q01", "q10", "q25", "q50", "q75", "q90", "q95", "q99"]
    for delta_column, title_text in delta_sections:
        lines.append(f"### {title_text}")
        lines.append("")
        lines.append(markdown_table(quantile_headers, quantile_rows(per_example_df, delta_column)))
        lines.append("")

    lines.append("## Token-Level Aggregates")
    lines.append("")
    lines.append("These are token-level, not example-level.")
    lines.append("")
    token_headers = [
        "Dataset",
        "Mean token delta vs random",
        "Token real-better rate vs random",
    ]
    if "delta_vs_zero_vector_nll" in per_token_df.columns:
        token_headers += [
            "Mean token delta vs zero",
            "Token real-better rate vs zero",
        ]
    lines.append(markdown_table(token_headers, token_rows(per_token_df)))
    lines.append("")

    lines.append("## Quick Interpretation")
    lines.append("")
    if run_config["heldout_eval"]:
        lines.append("- This run uses held-out data sources rather than the checkpoint's original train shards.")
    lines.append("- Real activations are compared against wrong-activation baselines on the same held-out examples.")
    lines.append("- More negative deltas mean real activations lowered loss relative to the baseline.")
    lines.append("- The combined random-baseline histogram is the main summary figure for fast comparison across dataset groups.")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append("- `png_plots/past_lens_delta_vs_random_hist.png`")
    lines.append("- `png_plots/synthetic_qa_delta_vs_random_hist.png`")
    lines.append("- `png_plots/classification_delta_vs_random_hist.png`")
    lines.append("- `png_plots/combined_delta_vs_random_histograms.png`")
    if "delta_vs_zero_vector_mean_nll" in per_example_df.columns:
        lines.append("- `png_plots/past_lens_delta_vs_zero_vector_hist.png`")
        lines.append("- `png_plots/synthetic_qa_delta_vs_zero_vector_hist.png`")
        lines.append("- `png_plots/classification_delta_vs_zero_vector_hist.png`")
        lines.append("- `png_plots/combined_delta_vs_zero_vector_histograms.png`")
    lines.append("")

    report_path = result_dir / "REPORT.md"
    report_path.write_text("\n".join(lines))
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
