"""
Plot comparison of old vs new AO model across all open-ended evals.

Reads per-eval summary JSONs from experiments/all_eval_summaries/
and produces comparison bar charts.

Usage:
    .venv/bin/python experiments/plotting/plot_all_eval_comparison.py
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SUMMARY_DIR = "experiments/all_eval_summaries"
PLOT_DIR = "experiments/all_eval_summaries/plots"

# Friendly names for verbalizer LoRAs
VERBALIZER_DISPLAY_NAMES = {
    "checkpoints_latentqa_cls_on_policy_Qwen3-8B": "Multi-layer (new)",
    "checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B": "Single-layer (old)",
}


def get_display_name(verb_key: str) -> str:
    for suffix, name in VERBALIZER_DISPLAY_NAMES.items():
        if suffix in verb_key:
            return name
    return verb_key


def load_summary(eval_name: str) -> dict | None:
    path = Path(SUMMARY_DIR) / f"{eval_name}_summary.json"
    if not path.exists():
        print(f"Warning: {path} not found, skipping")
        return None
    return json.loads(path.read_text())


def extract_primary_metric(eval_name: str, metrics: dict) -> tuple[str, float]:
    """Extract the primary metric name and value for each eval type."""
    if eval_name == "taboo":
        return "segment_accuracy", metrics.get("segment_accuracy", 0)
    elif eval_name == "personaqa":
        return "segment_accuracy", metrics.get("segment_accuracy", 0)
    elif eval_name == "number_prediction":
        return "model_match_rate", metrics.get("matches_model_answer_rate", 0)
    elif eval_name == "backtracking":
        return "mean_specificity", metrics.get("mean_specificity", 0)
    elif eval_name == "missing_info":
        return "accuracy", metrics.get("accuracy", 0)
    elif eval_name == "mmlu_prediction":
        return "accuracy", metrics.get("accuracy", 0)
    return "accuracy", metrics.get("accuracy", 0)


def plot_primary_metric_comparison():
    """Bar chart comparing old vs new model on primary metric for each eval."""
    eval_names = ["taboo", "personaqa", "number_prediction", "mmlu_prediction", "backtracking", "missing_info"]
    display_names = ["Taboo", "PersonaQA", "Number\nPrediction", "MMLU\nPrediction", "Backtracking", "Missing\nInfo"]

    new_values = []
    old_values = []
    metric_names = []

    for eval_name in eval_names:
        summary = load_summary(eval_name)
        if summary is None:
            new_values.append(0)
            old_values.append(0)
            metric_names.append("N/A")
            continue

        mbv = summary.get("metrics_by_verbalizer", {})

        new_val = 0
        old_val = 0
        metric_name = "N/A"

        for verb_key, metrics in mbv.items():
            display = get_display_name(verb_key)
            m_name, value = extract_primary_metric(eval_name, metrics)
            metric_name = m_name

            if "new" in display.lower() or "multi" in display.lower():
                new_val = value
            elif "old" in display.lower() or "single" in display.lower():
                old_val = value

        new_values.append(new_val)
        old_values.append(old_val)
        metric_names.append(metric_name)

    x = np.arange(len(eval_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width / 2, new_values, width, label="Multi-layer (new)", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, old_values, width, label="Single-layer (old)", color="#FF9800", alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_title("AO Model Comparison Across Open-Ended Evals")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.legend()

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # Add metric name annotations below x-axis
    for i, mname in enumerate(metric_names):
        ax.annotate(
            f"({mname})",
            xy=(i, 0),
            xytext=(0, -25),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7,
            color="gray",
        )

    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, "all_evals_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_taboo_detail():
    """Taboo: accuracy by position mode."""
    summary = load_summary("taboo")
    if not summary:
        return

    mbv = summary.get("metrics_by_verbalizer", {})
    modes = ["segment", "full_seq", "single_token"]
    mode_labels = ["Segment", "Full Seq", "Single Token"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(modes))
    width = 0.35
    i = 0

    for verb_key, metrics in mbv.items():
        display = get_display_name(verb_key)
        values = [metrics.get(f"{m}_accuracy", 0) for m in modes]
        offset = -width / 2 + i * width
        bars = ax.bar(x + offset, values, width, label=display, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        i += 1

    ax.set_ylabel("Accuracy")
    ax.set_title("Taboo: Accuracy by Position Mode")
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "taboo_detail.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_personaqa_detail():
    """PersonaQA: accuracy by position mode."""
    summary = load_summary("personaqa")
    if not summary:
        return

    mbv = summary.get("metrics_by_verbalizer", {})
    modes = ["segment", "full_seq", "single_token"]
    mode_labels = ["Segment", "Full Seq", "Single Token"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(modes))
    width = 0.35
    i = 0

    for verb_key, metrics in mbv.items():
        display = get_display_name(verb_key)
        values = [metrics.get(f"{m}_accuracy", 0) for m in modes]
        offset = -width / 2 + i * width
        bars = ax.bar(x + offset, values, width, label=display, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        i += 1

    ax.set_ylabel("Accuracy")
    ax.set_title("PersonaQA: Accuracy by Position Mode")
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "personaqa_detail.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_mmlu_detail():
    """MMLU: accuracy by prompt type for each mode (pre/post answer)."""
    summary = load_summary("mmlu_prediction")
    if not summary:
        return

    mode_results = summary.get("mode_results", {})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (mode_name, mode_data) in zip(axes, mode_results.items()):
        mbv = mode_data.get("metrics_by_verbalizer", {})
        if not mbv:
            continue

        # Collect prompt-level metrics
        prompt_keys = set()
        for metrics in mbv.values():
            for k in metrics:
                if k.startswith("prompt_") and k.endswith("_accuracy"):
                    prompt_keys.add(k)

        prompt_keys = sorted(prompt_keys)
        prompt_labels = [k.replace("prompt_", "").replace("_accuracy", "") for k in prompt_keys]

        x = np.arange(len(prompt_keys))
        width = 0.35
        i = 0

        for verb_key, metrics in mbv.items():
            display = get_display_name(verb_key)
            values = [metrics.get(k, 0) for k in prompt_keys]
            offset = -width / 2 + i * width
            bars = ax.bar(x + offset, values, width, label=display, alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
            i += 1

        ax.set_ylabel("Accuracy")
        ax.set_title(f"MMLU Prediction: {mode_name}")
        ax.set_xticks(x)
        ax.set_xticklabels(prompt_labels, rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "mmlu_prediction_detail.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_number_prediction_detail():
    """Number prediction: match rate by category."""
    summary = load_summary("number_prediction")
    if not summary:
        return

    mbv = summary.get("metrics_by_verbalizer", {})

    # Find category keys
    cat_keys = set()
    for metrics in mbv.values():
        for k in metrics:
            if k.startswith("cat_") and k.endswith("_model_match_rate"):
                cat_keys.add(k)

    cat_keys = sorted(cat_keys)
    cat_labels = [k.replace("cat_", "").replace("_model_match_rate", "") for k in cat_keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cat_keys))
    width = 0.35
    i = 0

    for verb_key, metrics in mbv.items():
        display = get_display_name(verb_key)
        values = [metrics.get(k, 0) for k in cat_keys]
        offset = -width / 2 + i * width
        bars = ax.bar(x + offset, values, width, label=display, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        i += 1

    ax.set_ylabel("Model Match Rate")
    ax.set_title("Number Prediction: Match Rate by Category")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, max(0.15, max(metrics.get(k, 0) for k in cat_keys for metrics in mbv.values()) * 1.3))
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "number_prediction_detail.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_backtracking_detail():
    """Backtracking: specificity and correctness side by side."""
    summary = load_summary("backtracking")
    if not summary:
        return

    mbv = summary.get("metrics_by_verbalizer", {})
    metric_pairs = [
        ("mean_specificity", "Mean Specificity"),
        ("mean_correctness", "Mean Correctness"),
        ("specificity_>=3_rate", "Specificity >= 3"),
        ("correctness_>=3_rate", "Correctness >= 3"),
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(metric_pairs))
    width = 0.35
    i = 0

    for verb_key, metrics in mbv.items():
        display = get_display_name(verb_key)
        values = [metrics.get(k, 0) for k, _ in metric_pairs]
        offset = -width / 2 + i * width
        bars = ax.bar(x + offset, values, width, label=display, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        i += 1

    ax.set_ylabel("Score")
    ax.set_title("Backtracking: LLM Judge Scores")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metric_pairs])
    ax.legend()
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "backtracking_detail.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_missing_info_detail():
    """Missing info: accuracy by condition + A vs C agreement."""
    summary = load_summary("missing_info")
    if not summary:
        return

    mbv = summary.get("metrics_by_verbalizer", {})
    conditions = ["A_complete", "B_incomplete", "C_forced"]
    metric_keys = [f"{c}_accuracy" for c in conditions] + ["A_vs_C_agreement_rate", "accuracy"]
    metric_labels = ["A (complete)", "B (incomplete)", "C (forced)", "A vs C\nagreement", "Overall\naccuracy"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(metric_keys))
    width = 0.35
    i = 0

    for verb_key, metrics in mbv.items():
        display = get_display_name(verb_key)
        values = [metrics.get(k, 0) for k in metric_keys]
        offset = -width / 2 + i * width
        bars = ax.bar(x + offset, values, width, label=display, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        i += 1

    ax.set_ylabel("Score")
    ax.set_title("Missing Info: Accuracy by Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "missing_info_detail.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Overview comparison
    plot_primary_metric_comparison()

    # Per-eval detail plots
    plot_taboo_detail()
    plot_personaqa_detail()
    plot_mmlu_detail()
    plot_number_prediction_detail()
    plot_backtracking_detail()
    plot_missing_info_detail()

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
