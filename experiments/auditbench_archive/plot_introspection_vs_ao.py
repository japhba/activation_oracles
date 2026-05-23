"""Plot saved greedy AO baselines against the prompt-only introspection baseline."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_PATH = (
    Path(__file__).resolve().parent
    / "introspection_adapter_baseline_prompt_only"
    / "introspection_baseline_targets_transcripts-synth_docs_nbeh_14_nprompts_4.json"
)

METHOD_ORDER = [
    "Original AO",
    "MLAO",
    "AO v2",
    "AO v2 + HB",
    "Target only",
    "Target + MetaLoRA",
]

METHOD_STYLE = {
    "Original AO": {"color": "#f58231", "hatch": None},
    "MLAO": {"color": "#e6194b", "hatch": None},
    "AO v2": {"color": "#3cb44b", "hatch": None},
    "AO v2 + HB": {"color": "#4363d8", "hatch": None},
    "Target only": {"color": "#cccccc", "hatch": "//"},
    "Target + MetaLoRA": {"color": "#8c564b", "hatch": "\\\\"},
}

TARGET_ORDER = ["transcripts", "synth_docs"]
TARGET_TITLES = {
    "transcripts": "Transcripts + KTO",
    "synth_docs": "Synth-Docs + SFT",
}


def load_rows() -> list[dict]:
    with open(RESULTS_PATH) as f:
        payload = json.load(f)
    return payload["aligned_summary_rows"]


def rows_by_target(rows: list[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault(row["target_id"], {})[row["method"]] = row
    return grouped


def _values_for_target(
    grouped_rows: dict[str, dict[str, dict]],
    *,
    target_id: str,
    key: str,
) -> list[float]:
    return [grouped_rows[target_id][method][key] for method in METHOD_ORDER]


def plot_metric(
    *,
    grouped_rows: dict[str, dict[str, dict]],
    key: str,
    ylabel: str,
    title: str,
    subtitle: str,
    output_name: str,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    x = np.arange(len(METHOD_ORDER))

    for ax, target_id in zip(axes, TARGET_ORDER, strict=True):
        values = _values_for_target(grouped_rows, target_id=target_id, key=key)
        bars = []
        for idx, method in enumerate(METHOD_ORDER):
            style = METHOD_STYLE[method]
            bar = ax.bar(
                x[idx],
                values[idx],
                color=style["color"],
                edgecolor="black",
                linewidth=1.0,
                hatch=style["hatch"],
                width=0.72,
            )
            bars.append(bar[0])

        ax.set_title(TARGET_TITLES[target_id], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)

        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.03,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.03)
    fig.text(0.5, 0.98, subtitle, ha="center", fontsize=10, color="#555555")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = RESULTS_PATH.parent / output_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    rows = load_rows()
    grouped = rows_by_target(rows)

    correctness_out = plot_metric(
        grouped_rows=grouped,
        key="plotted_correctness_0_4",
        ylabel="Plotted Correctness (mean_correctness - 1.0)",
        title="AuditBench: Greedy AO vs Prompt-Only Introspection Baseline",
        subtitle="Metric shown: plotted correctness on 0-4 scale",
        output_name="introspection_vs_ao_correctness.png",
    )
    avg_out = plot_metric(
        grouped_rows=grouped,
        key="avg_corr_spec",
        ylabel="Avg(Correctness, Specificity)",
        title="AuditBench: Greedy AO vs Prompt-Only Introspection Baseline",
        subtitle="Metric shown: average of mean correctness and mean specificity",
        output_name="introspection_vs_ao_avg_corr_spec.png",
    )

    print(f"Saved: {correctness_out}")
    print(f"Saved: {avg_out}")


if __name__ == "__main__":
    main()
