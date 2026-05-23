"""Generate grouped eval comparison plots.

Produces:
1. A 4-subplot bar chart with selected LoRAs (presentation-ready).
2. A line chart with all LoRAs, best one bold, others muted.
3. Binary detail: accuracy vs AUC side by side.
4. Open-ended detail: specificity vs correctness side by side.

All outputs go to experiments/eval_plots/.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from experiments.plot_eval_summaries import (
    ALL_LORAS,
    PRESETS,
    RESULT_DIR_PREFIX,
    find_eval_jsons,
    extract_lora_name,
    load_all_metrics,
    build_label_map_from_dir_labels,
    shorten_lora,
)

OUTPUT_DIR = "experiments/eval_plots"

# ---------------------------------------------------------------------------
# Metric group definitions
# ---------------------------------------------------------------------------

GROUPS = [
    {
        "key": "binary",
        "title": "Binary Classification (ROC AUC)",
        "metrics": [
            ("mmlu_prediction/pre_answer", "roc_auc", "MMLU\nPre-Answer", "0-1"),
            ("mmlu_prediction/post_answer", "roc_auc", "MMLU\nPost-Answer", "0-1"),
            ("sycophancy/no_cot", "roc_auc", "Sycophancy\nNo CoT", "0-1"),
            ("sycophancy/cot", "roc_auc", "Sycophancy\nCoT", "0-1"),
            ("missing_info", "A_vs_C_roc_auc", "Missing Info\nA vs C", "0-1"),
        ],
    },
    {
        "key": "accuracy",
        "title": "Multi-class Accuracy",
        "metrics": [
            ("taboo", "full_seq_accuracy", "Taboo\nFull Seq", "0-1"),
            ("taboo", "single_token_accuracy", "Taboo\nSingle Tok", "0-1"),
            ("personaqa", "full_seq_accuracy", "PersonaQA\nFull Seq", "0-1"),
            ("personaqa", "single_token_accuracy", "PersonaQA\nSingle Tok", "0-1"),
        ],
    },
    {
        "key": "open_ended",
        "title": "Open-ended (LLM Judge, Correctness)",
        "metrics": [
            ("backtracking", "mean_correctness", "Backtracking", "1-5"),
            ("system_prompt_qa_hidden/user_and_assistant", "mean_correctness", "SysPrompt\nHidden", "1-5"),
            ("system_prompt_qa_latentqa/user_and_assistant", "mean_correctness", "SysPrompt\nLatentQA", "1-5"),
        ],
    },
    {
        "key": "prediction",
        "title": "Prediction (Exact Match)",
        "metrics": [
            ("mmlu_prediction/pre_answer/letter_prediction", "matches_model_rate", "MMLU Letter\nPrediction", "0-1"),
            ("number_prediction", "matches_model_answer_rate", "Number Pred\nMatch Rate", "0-1"),
        ],
    },
]

# The headline LoRA — the latest/best one. Gets a bold black outline on bar
# charts and a bold line on line charts. Change this as new models are trained.
HEADLINE_LORA_KEY = "full_mix_500k_past_lens_plus_synthetic_qa_v3"

BAR_COLORS = ["#4363d8", "#e6194b", "#3cb44b", "#f58231", "#911eb4", "#42d4f4", "#f032e6"]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p", "d"]
LINE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff",
]

# Font sizes
FONT_SUPTITLE = 20
FONT_TITLE = 16
FONT_AXIS_LABEL = 14
FONT_TICK = 13
FONT_BAR_LABEL = 9
FONT_LEGEND = 11
FONT_LEGEND_ALL = 9  # for the all-LoRA plot (more items)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Confidence interval computation
# ---------------------------------------------------------------------------

N_BOOTSTRAP = 2000
RNG = np.random.RandomState(42)


def bootstrap_ci(values: np.ndarray, stat_fn=np.mean, alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap 95% CI for a statistic (vectorized)."""
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    # Generate all bootstrap indices at once: (N_BOOTSTRAP, n)
    idx = RNG.randint(0, n, size=(N_BOOTSTRAP, n))
    samples = values[idx]  # (N_BOOTSTRAP, n)
    if stat_fn is np.mean:
        boot_stats = samples.mean(axis=1)
    else:
        boot_stats = np.array([stat_fn(s) for s in samples])
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return (float(lo), float(hi))


def _fast_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUC without sklearn overhead (assumes binary labels 0/1)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Wilcoxon-Mann-Whitney statistic
    # For each positive, count how many negatives it beats
    # Vectorized via broadcasting for small arrays
    if n_pos * n_neg <= 1_000_000:
        wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    else:
        # Fallback for very large arrays
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(labels, scores)
    return float(wins / (n_pos * n_neg))


def bootstrap_auc_ci(labels: np.ndarray, scores: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap 95% CI for ROC AUC (vectorized)."""
    n = len(labels)
    if n < 2 or len(np.unique(labels)) < 2:
        return (float("nan"), float("nan"))
    # Generate all bootstrap indices at once
    idx = RNG.randint(0, n, size=(N_BOOTSTRAP, n))
    boot_labels = labels[idx]  # (N_BOOTSTRAP, n)
    boot_scores = scores[idx]
    boot_stats = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        bl, bs = boot_labels[i], boot_scores[i]
        if len(np.unique(bl)) < 2:
            boot_stats[i] = float("nan")
        else:
            boot_stats[i] = _fast_auc(bl, bs)
    boot_stats = boot_stats[~np.isnan(boot_stats)]
    if len(boot_stats) == 0:
        return (float("nan"), float("nan"))
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return (float(lo), float(hi))


def compute_ci_for_result(data: dict, metric_key: str) -> tuple[float, float] | None:
    """Compute 95% CI for a given metric from a result JSON's per-entry data."""

    # --- Binary evals: AUC and accuracy ---
    if "binary_scored_results" in data:
        entries = data["binary_scored_results"]

        # Handle A_vs_C subset for missing_info
        if metric_key == "A_vs_C_roc_auc":
            entries = [e for e in entries if e.get("condition") in ("A_complete", "C_forced")]
            if not entries:
                return None
            # A_complete = has all info (ground_truth "no" for missing info)
            # C_forced = forced to act on missing info (ground_truth "yes")
            labels = np.array([0 if e.get("condition") == "A_complete" else 1 for e in entries])
            scores = np.array([e.get("margin_yes_minus_no", 0) for e in entries])
            return bootstrap_auc_ci(labels, scores)

        if metric_key == "roc_auc":
            labels = np.array([1 if e.get("ground_truth") == "yes" or e.get("binary_label") == 1 else 0 for e in entries])
            scores = np.array([e.get("margin_yes_minus_no", 0) for e in entries])
            return bootstrap_auc_ci(labels, scores)

        if metric_key == "accuracy_at_zero":
            correct = np.array([1.0 if e.get("is_correct") else 0.0 for e in entries])
            return bootstrap_ci(correct)

    # --- Open-ended evals: per-entry specificity/correctness scores ---
    if "scored_results" in data:
        entries = data["scored_results"]

        if metric_key == "mean_correctness" and entries and "correctness" in entries[0]:
            vals = np.array([e["correctness"] for e in entries], dtype=float)
            return bootstrap_ci(vals)

        if metric_key == "mean_specificity" and entries and "specificity" in entries[0]:
            vals = np.array([e["specificity"] for e in entries], dtype=float)
            return bootstrap_ci(vals)

        # Multi-class accuracy (e.g. backtracking MC)
        if metric_key == "accuracy" and entries and "is_correct" in entries[0]:
            vals = np.array([1.0 if e["is_correct"] else 0.0 for e in entries])
            return bootstrap_ci(vals)

        # Letter/number prediction match rates
        if metric_key == "matches_model_rate" and entries and "matches_model" in entries[0]:
            vals = np.array([1.0 if e["matches_model"] else 0.0 for e in entries])
            return bootstrap_ci(vals)

        if metric_key == "matches_model_answer_rate":
            # number_prediction uses a different structure
            for match_key in ("matches_model_answer", "matches_model", "model_correct"):
                if entries and match_key in entries[0]:
                    vals = np.array([1.0 if e[match_key] else 0.0 for e in entries])
                    return bootstrap_ci(vals)

    # --- Taboo / PersonaQA: per-entry results with ground_truth + responses ---
    if "results" in data and isinstance(data.get("results"), list):
        entries = data["results"]
        metadata = data.get("entry_metadata", [])
        if not entries or "ground_truth" not in entries[0]:
            return None

        # Map metric to position_mode used in the eval
        mode_map = {
            "full_seq_accuracy": "full_seq",
            "single_token_accuracy": "single_token",
            "segment_accuracy": "segment",
        }
        target_mode = mode_map.get(metric_key)

        if metric_key in mode_map:
            correct = []
            for i, e in enumerate(entries):
                # Filter by position_mode if metadata is available
                if metadata and i < len(metadata):
                    if metadata[i].get("position_mode") != target_mode:
                        continue

                gt = _normalize_answer(e.get("ground_truth", ""))
                responses = e.get("responses", [])
                if not responses:
                    continue
                resp = responses[0] if isinstance(responses[0], str) else str(responses[0])
                resp = _normalize_answer(resp)
                # Match logic from taboo/personaqa: substring containment
                correct.append(1.0 if gt in resp else 0.0)
            if correct:
                return bootstrap_ci(np.array(correct))

    return None


def _normalize_answer(answer: str) -> str:
    """Normalize answer for matching — mirrors taboo.py normalize_answer."""
    return answer.rstrip(".!?,;:").strip().lower()


def load_all_cis(result_dirs: list[str]) -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    """Load per-entry CIs from all result dirs.

    Returns: {eval_display_name: {lora_name: {metric: (ci_lo, ci_hi)}}}
    """
    all_cis: dict[str, dict[str, dict[str, tuple[float, float]]]] = defaultdict(dict)

    # Collect all (eval_pattern, metric_key) pairs we need CIs for
    needed_metrics: dict[str, set[str]] = defaultdict(set)
    for group in GROUPS:
        for eval_pattern, metric_key, _, _ in group["metrics"]:
            needed_metrics[eval_pattern].add(metric_key)
    # Also add detail plot metrics
    for eval_pattern, _, overrides in BINARY_EVALS:
        needed_metrics[eval_pattern].add("roc_auc")
        needed_metrics[eval_pattern].add("accuracy_at_zero")
        for actual_key in overrides.values():
            needed_metrics[eval_pattern].add(actual_key)
    for eval_pattern, _ in OPEN_ENDED_EVALS:
        needed_metrics[eval_pattern].add("mean_specificity")
        needed_metrics[eval_pattern].add("mean_correctness")
    # Backtracking comparison plot
    for eval_pattern, metric_key, _, _ in BACKTRACKING_COMPARISON_GROUP["metrics"]:
        needed_metrics[eval_pattern].add(metric_key)

    for result_dir in result_dirs:
        eval_jsons = find_eval_jsons(result_dir)
        for eval_name, json_files in eval_jsons.items():
            for json_file in json_files:
                with open(json_file) as f:
                    data = json.load(f)

                lora_name = extract_lora_name(data)

                rel = json_file.relative_to(Path(result_dir) / eval_name)
                if len(rel.parts) > 1:
                    sub = "/".join(rel.parts[:-1])
                    display_name = f"{eval_name}/{sub}"
                else:
                    display_name = eval_name

                if display_name not in needed_metrics:
                    continue

                ci_dict = {}
                for metric_key in needed_metrics[display_name]:
                    ci = compute_ci_for_result(data, metric_key)
                    if ci is not None:
                        ci_dict[metric_key] = ci

                if ci_dict:
                    all_cis[display_name][lora_name] = ci_dict

    return dict(all_cis)


def gather_group_cis(
    group: dict,
    all_cis: dict,
    lora_names: list[str],
) -> dict[str, list[tuple[float, float] | None]]:
    """Return {lora: [(ci_lo, ci_hi) or None]} matching gather_group_data order."""
    cis: dict[str, list[tuple[float, float] | None]] = {l: [] for l in lora_names}

    for eval_pattern, metric_key, label, scale in group["metrics"]:
        if eval_pattern not in all_cis:
            # Still need to append None to keep alignment
            for lora in lora_names:
                cis[lora].append(None)
            continue
        ci_data = all_cis[eval_pattern]
        for lora in lora_names:
            ci = ci_data.get(lora, {}).get(metric_key)
            cis[lora].append(ci)

    return cis


def resolve_selected_loras() -> list[str]:
    """Get actual lora_name strings for the 'selected' preset."""
    selected_loras = []
    for key in PRESETS["selected"]:
        result_dir = RESULT_DIR_PREFIX + key
        eval_jsons = find_eval_jsons(result_dir)
        for json_files in eval_jsons.values():
            for json_file in json_files:
                with open(json_file) as f:
                    data = json.load(f)
                lora_name = extract_lora_name(data)
                if lora_name not in selected_loras:
                    selected_loras.append(lora_name)
                break
            break
    return selected_loras


def resolve_headline_lora() -> str | None:
    """Get actual lora_name string for the headline LoRA."""
    result_dir = RESULT_DIR_PREFIX + HEADLINE_LORA_KEY
    eval_jsons = find_eval_jsons(result_dir)
    for json_files in eval_jsons.values():
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)
            return extract_lora_name(data)
    return None


def gather_group_data(
    group: dict,
    all_metrics: dict,
    lora_names: list[str],
) -> tuple[list[str], dict[str, list[float | None]]]:
    """Return (x_labels, {lora: [values]}) for a group."""
    x_labels = []
    values: dict[str, list[float | None]] = {l: [] for l in lora_names}

    for eval_pattern, metric_key, label, scale in group["metrics"]:
        if eval_pattern not in all_metrics:
            continue
        eval_data = all_metrics[eval_pattern]
        has_any = any(metric_key in eval_data.get(l, {}) for l in lora_names)
        if not has_any:
            continue
        x_labels.append(label)
        for lora in lora_names:
            values[lora].append(eval_data.get(lora, {}).get(metric_key))

    return x_labels, values


def is_1_5_group(group: dict) -> bool:
    return any(s == "1-5" for _, _, _, s in group["metrics"])


def draw_bars(ax, x, lora_vals, bar_width, offset, color, label, is_headline,
              ci_bounds=None):
    """Draw a set of bars, with black outline if headline and optional error bars.

    ci_bounds: list of (lo, hi) or None per bar. Error bars are drawn as
    distance from the bar height to the CI bound.
    """
    edgecolor = "black" if is_headline else "white"
    linewidth = 5.0 if is_headline else 0.5

    # Compute yerr if we have CI bounds
    yerr = None
    if ci_bounds is not None:
        yerr_lo = []
        yerr_hi = []
        for val, ci in zip(lora_vals, ci_bounds):
            if ci is not None and not (np.isnan(ci[0]) or np.isnan(ci[1])):
                yerr_lo.append(max(0, val - ci[0]))
                yerr_hi.append(max(0, ci[1] - val))
            else:
                yerr_lo.append(0)
                yerr_hi.append(0)
        yerr = [yerr_lo, yerr_hi]

    bars = ax.bar(x + offset, lora_vals, bar_width * 0.9,
                  label=label, color=color,
                  alpha=0.85, edgecolor=edgecolor, linewidth=linewidth,
                  yerr=yerr, capsize=3, error_kw={"elinewidth": 1.2, "capthick": 1.2})
    for bar, val in zip(bars, lora_vals):
        if val > 0:
            fmt = f"{val:.2f}" if val < 10 else f"{val:.1f}"
            # Place label above error bar if present
            top = bar.get_height()
            if ci_bounds is not None:
                idx = list(bars).index(bar)
                ci = ci_bounds[idx]
                if ci is not None and not np.isnan(ci[1]):
                    top = ci[1]
            ax.text(bar.get_x() + bar.get_width() / 2, top + 0.01,
                    fmt, ha="center", va="bottom", fontsize=FONT_BAR_LABEL)
    return bars


# ---------------------------------------------------------------------------
# Plot 1: 4-subplot bar chart (selected LoRAs)
# ---------------------------------------------------------------------------


def plot_selected_bars(all_metrics: dict, label_map: dict[str, str],
                       all_cis: dict | None = None):
    """4-subplot bar chart with selected LoRAs."""
    selected_loras = resolve_selected_loras()
    headline_lora = resolve_headline_lora()

    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    axes = axes.flatten()

    for idx, group in enumerate(GROUPS):
        ax = axes[idx]
        x_labels, values = gather_group_data(group, all_metrics, selected_loras)
        group_cis = gather_group_cis(group, all_cis, selected_loras) if all_cis else None
        if not x_labels:
            ax.set_title(group["title"], fontsize=FONT_TITLE)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        n_metrics = len(x_labels)
        n_loras = len(selected_loras)
        bar_width = 0.8 / n_loras
        x = np.arange(n_metrics)

        for i, lora in enumerate(selected_loras):
            lora_vals = [v if v is not None else 0 for v in values[lora]]
            ci_bounds = group_cis[lora] if group_cis else None
            offset = (i - n_loras / 2 + 0.5) * bar_width
            label = label_map.get(lora, shorten_lora(lora))
            is_headline = (lora == headline_lora)
            draw_bars(ax, x, lora_vals, bar_width, offset,
                      BAR_COLORS[i % len(BAR_COLORS)], label, is_headline,
                      ci_bounds=ci_bounds)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=FONT_TICK, ha="center")
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.set_title(group["title"], fontsize=FONT_TITLE, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        if is_1_5_group(group):
            ax.set_ylim(0, 5.0)
            ax.set_ylabel("Score (1-5)", fontsize=FONT_AXIS_LABEL)
        else:
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Score", fontsize=FONT_AXIS_LABEL)


    # Shared legend at the bottom
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=len(selected_loras),
               fontsize=FONT_SUPTITLE, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle("Eval Comparison — Selected LoRAs", fontsize=FONT_SUPTITLE,
                 fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    out = os.path.join(OUTPUT_DIR, "eval_selected_bars.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def resolve_preset_loras(preset_name: str) -> list[str]:
    """Get actual lora_name strings for any named preset."""
    loras = []
    for key in PRESETS[preset_name]:
        result_dir = RESULT_DIR_PREFIX + key
        eval_jsons = find_eval_jsons(result_dir)
        for json_files in eval_jsons.values():
            for json_file in json_files:
                with open(json_file) as f:
                    data = json.load(f)
                lora_name = extract_lora_name(data)
                if lora_name not in loras:
                    loras.append(lora_name)
                break
            break
    return loras


def plot_selected_bars_2(all_metrics: dict, label_map: dict[str, str],
                          all_cis: dict | None = None):
    """4-subplot bar chart with selected_2 LoRAs (original 3 + 4 new)."""
    selected_loras = resolve_preset_loras("selected_2")
    headline_lora = resolve_headline_lora()

    fig, axes = plt.subplots(2, 2, figsize=(28, 16))
    axes = axes.flatten()

    for idx, group in enumerate(GROUPS):
        ax = axes[idx]
        x_labels, values = gather_group_data(group, all_metrics, selected_loras)
        group_cis = gather_group_cis(group, all_cis, selected_loras) if all_cis else None
        if not x_labels:
            ax.set_title(group["title"], fontsize=FONT_TITLE)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        n_metrics = len(x_labels)
        n_loras = len(selected_loras)
        bar_width = 0.8 / n_loras
        x = np.arange(n_metrics)

        for i, lora in enumerate(selected_loras):
            lora_vals = [v if v is not None else 0 for v in values[lora]]
            ci_bounds = group_cis[lora] if group_cis else None
            offset = (i - n_loras / 2 + 0.5) * bar_width
            label = label_map.get(lora, shorten_lora(lora))
            is_headline = (lora == headline_lora)
            draw_bars(ax, x, lora_vals, bar_width, offset,
                      BAR_COLORS[i % len(BAR_COLORS)], label, is_headline,
                      ci_bounds=ci_bounds)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=FONT_TICK, ha="center")
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.set_title(group["title"], fontsize=FONT_TITLE, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        if is_1_5_group(group):
            ax.set_ylim(0, 5.0)
            ax.set_ylabel("Score (1-5)", fontsize=FONT_AXIS_LABEL)
        else:
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Score", fontsize=FONT_AXIS_LABEL)

    # Shared legend at the bottom
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=min(len(selected_loras), 4),
               fontsize=FONT_LEGEND, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Eval Comparison — Selected LoRAs + New Models", fontsize=FONT_SUPTITLE,
                 fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    out = os.path.join(OUTPUT_DIR, "eval_selected_bars_2.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 2: line chart with all LoRAs, headline bold
# ---------------------------------------------------------------------------


def plot_all_lines(all_metrics: dict, label_map: dict[str, str]):
    """4-subplot line chart, all LoRAs, headline one bold."""
    all_loras = sorted(set(
        lora for eval_metrics in all_metrics.values() for lora in eval_metrics
    ))
    headline_lora = resolve_headline_lora()

    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    axes = axes.flatten()

    for idx, group in enumerate(GROUPS):
        ax = axes[idx]
        x_labels, values = gather_group_data(group, all_metrics, all_loras)
        if not x_labels:
            ax.set_title(group["title"], fontsize=FONT_TITLE)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        n_points = len(x_labels)
        x = np.arange(n_points)

        # Plot non-headline loras first (muted), then headline on top
        for i, lora in enumerate(all_loras):
            if lora == headline_lora:
                continue
            color = LINE_COLORS[i % len(LINE_COLORS)]
            marker = MARKERS[i % len(MARKERS)]
            vals = values[lora]
            xs = [x[j] for j in range(n_points) if vals[j] is not None]
            ys = [vals[j] for j in range(n_points) if vals[j] is not None]
            label = label_map.get(lora, shorten_lora(lora))
            ax.plot(xs, ys, color=color, marker=marker, markersize=5,
                    linewidth=1.0, label=label, alpha=0.4)

        # Plot headline bold on top
        if headline_lora and headline_lora in values:
            vals = values[headline_lora]
            xs = [x[j] for j in range(n_points) if vals[j] is not None]
            ys = [vals[j] for j in range(n_points) if vals[j] is not None]
            label = label_map.get(headline_lora, shorten_lora(headline_lora))
            ax.plot(xs, ys, color="#e6194b", marker="o", markersize=9,
                    linewidth=3.0, label=f"{label} (best)", alpha=1.0, zorder=10)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=FONT_TICK, ha="center")
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.set_title(group["title"], fontsize=FONT_TITLE, fontweight="bold")
        ax.grid(axis="both", alpha=0.3)

        if is_1_5_group(group):
            ax.set_ylim(1.0, 5.0)
            ax.set_ylabel("Score (1-5)", fontsize=FONT_AXIS_LABEL)
        else:
            all_vals = [v for vs in values.values() for v in vs if v is not None]
            y_min = max(0, min(all_vals) - 0.05) if all_vals else 0
            ax.set_ylim(y_min, 1.0)
            ax.set_ylabel("Score", fontsize=FONT_AXIS_LABEL)

        ax.set_xlim(-0.5, n_points - 0.5)

    # Shared legend at the bottom
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=4,
               fontsize=FONT_LEGEND_ALL, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("All LoRAs — 500k PL + SynQA v3 Highlighted", fontsize=FONT_SUPTITLE,
                 fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    out = os.path.join(OUTPUT_DIR, "eval_all_loras_highlighted.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 3: Binary classification detail — Accuracy vs AUC side by side
# ---------------------------------------------------------------------------

# Each entry: (eval_pattern, label, {panel_metric_key: actual_metric_key_override})
# If no override, the panel's default metric key is used.
BINARY_EVALS = [
    ("mmlu_prediction/pre_answer", "MMLU Pre", {}),
    ("mmlu_prediction/post_answer", "MMLU Post", {}),
    ("sycophancy/no_cot", "Syco No-CoT", {}),
    ("sycophancy/cot", "Syco CoT", {}),
    ("missing_info", "Missing Info\nA vs C", {
        "roc_auc": "A_vs_C_roc_auc",
    }),
]


def plot_binary_detail(all_metrics: dict, label_map: dict[str, str],
                       all_cis: dict | None = None):
    """2-subplot bar chart: accuracy (left) and AUC (right) for binary evals."""
    selected_loras = resolve_selected_loras()
    headline_lora = resolve_headline_lora()

    panels = [
        ("Accuracy", "accuracy_at_zero"),
        ("ROC AUC", "roc_auc"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, (panel_title, metric_key) in zip(axes, panels):
        x_labels = []
        lora_values: dict[str, list[float | None]] = {l: [] for l in selected_loras}
        lora_cis: dict[str, list[tuple[float, float] | None]] = {l: [] for l in selected_loras}

        for eval_pattern, eval_label, overrides in BINARY_EVALS:
            if eval_pattern not in all_metrics:
                continue
            eval_data = all_metrics[eval_pattern]
            actual_key = overrides.get(metric_key, metric_key)
            has_any = any(actual_key in eval_data.get(l, {}) for l in selected_loras)
            if not has_any:
                continue
            x_labels.append(eval_label)
            ci_data = all_cis.get(eval_pattern, {}) if all_cis else {}
            for lora in selected_loras:
                lora_values[lora].append(eval_data.get(lora, {}).get(actual_key))
                lora_cis[lora].append(ci_data.get(lora, {}).get(actual_key))

        if not x_labels:
            ax.set_title(panel_title, fontsize=FONT_TITLE)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        n_metrics = len(x_labels)
        n_loras = len(selected_loras)
        bar_width = 0.8 / n_loras
        x = np.arange(n_metrics)

        for i, lora in enumerate(selected_loras):
            vals = [v if v is not None else 0 for v in lora_values[lora]]
            ci_bounds = lora_cis[lora] if all_cis else None
            offset = (i - n_loras / 2 + 0.5) * bar_width
            label = label_map.get(lora, shorten_lora(lora))
            is_headline = (lora == headline_lora)
            draw_bars(ax, x, vals, bar_width, offset,
                      BAR_COLORS[i % len(BAR_COLORS)], label, is_headline,
                      ci_bounds=ci_bounds)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=FONT_TICK, ha="center")
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.set_title(panel_title, fontsize=FONT_TITLE, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Score", fontsize=FONT_AXIS_LABEL)
        if metric_key == "accuracy_at_zero":
            ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)

    # Shared legend at the bottom
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=len(selected_loras),
               fontsize=FONT_LEGEND, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle("Binary Classification Detail — Accuracy vs ROC AUC",
                 fontsize=FONT_SUPTITLE, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    out = os.path.join(OUTPUT_DIR, "eval_binary_detail.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 4: Open-ended detail — Specificity vs Correctness side by side
# ---------------------------------------------------------------------------

OPEN_ENDED_EVALS = [
    ("backtracking", "Backtracking"),
    ("system_prompt_qa_hidden/user_and_assistant", "SysPrompt\nHidden"),
    ("system_prompt_qa_latentqa/user_and_assistant", "SysPrompt\nLatentQA"),
]


def plot_open_ended_detail(all_metrics: dict, label_map: dict[str, str],
                           all_cis: dict | None = None):
    """2-subplot bar chart: specificity (left) and correctness (right)."""
    selected_loras = resolve_selected_loras()
    headline_lora = resolve_headline_lora()

    panels = [
        ("Specificity", "mean_specificity"),
        ("Correctness", "mean_correctness"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, (panel_title, metric_key) in zip(axes, panels):
        x_labels = []
        lora_values: dict[str, list[float | None]] = {l: [] for l in selected_loras}
        lora_cis: dict[str, list[tuple[float, float] | None]] = {l: [] for l in selected_loras}

        for eval_pattern, eval_label in OPEN_ENDED_EVALS:
            if eval_pattern not in all_metrics:
                continue
            eval_data = all_metrics[eval_pattern]
            has_any = any(metric_key in eval_data.get(l, {}) for l in selected_loras)
            if not has_any:
                continue
            x_labels.append(eval_label)
            ci_data = all_cis.get(eval_pattern, {}) if all_cis else {}
            for lora in selected_loras:
                lora_values[lora].append(eval_data.get(lora, {}).get(metric_key))
                lora_cis[lora].append(ci_data.get(lora, {}).get(metric_key))

        if not x_labels:
            ax.set_title(panel_title, fontsize=FONT_TITLE)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        n_metrics = len(x_labels)
        n_loras = len(selected_loras)
        bar_width = 0.8 / n_loras
        x = np.arange(n_metrics)

        for i, lora in enumerate(selected_loras):
            vals = [v if v is not None else 0 for v in lora_values[lora]]
            ci_bounds = lora_cis[lora] if all_cis else None
            offset = (i - n_loras / 2 + 0.5) * bar_width
            label = label_map.get(lora, shorten_lora(lora))
            is_headline = (lora == headline_lora)
            draw_bars(ax, x, vals, bar_width, offset,
                      BAR_COLORS[i % len(BAR_COLORS)], label, is_headline,
                      ci_bounds=ci_bounds)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=FONT_TICK, ha="center")
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.set_title(panel_title, fontsize=FONT_TITLE, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 5.0)
        ax.set_ylabel("Score (1-5)", fontsize=FONT_AXIS_LABEL)

    # Shared legend at the bottom
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=len(selected_loras),
               fontsize=FONT_LEGEND, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle("Open-ended Eval Detail — Specificity vs Correctness",
                 fontsize=FONT_SUPTITLE, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    out = os.path.join(OUTPUT_DIR, "eval_open_ended_detail.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 5: Prediction (Exact Match) standalone
# ---------------------------------------------------------------------------

PREDICTION_GROUP = {
    "key": "prediction",
    "title": "Prediction (Exact Match)",
    "metrics": [
        ("mmlu_prediction/pre_answer/letter_prediction", "matches_model_rate", "MMLU Letter\nPrediction", "0-1"),
        ("number_prediction", "matches_model_answer_rate", "Number Pred\nMatch Rate", "0-1"),
    ],
}


def plot_prediction_standalone(all_metrics: dict, label_map: dict[str, str],
                                all_cis: dict | None = None):
    """Standalone bar chart for prediction (exact match) evals."""
    selected_loras = resolve_selected_loras()
    headline_lora = resolve_headline_lora()

    x_labels, values = gather_group_data(PREDICTION_GROUP, all_metrics, selected_loras)
    group_cis = gather_group_cis(PREDICTION_GROUP, all_cis, selected_loras) if all_cis else None

    if not x_labels:
        print("No prediction data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    n_metrics = len(x_labels)
    n_loras = len(selected_loras)
    bar_width = 0.8 / n_loras
    x = np.arange(n_metrics)

    for i, lora in enumerate(selected_loras):
        lora_vals = [v if v is not None else 0 for v in values[lora]]
        ci_bounds = group_cis[lora] if group_cis else None
        offset = (i - n_loras / 2 + 0.5) * bar_width
        label = label_map.get(lora, shorten_lora(lora))
        is_headline = (lora == headline_lora)
        draw_bars(ax, x, lora_vals, bar_width, offset,
                  BAR_COLORS[i % len(BAR_COLORS)], label, is_headline,
                  ci_bounds=ci_bounds)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=FONT_TICK, ha="center")
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_title("Prediction (Exact Match)", fontsize=FONT_TITLE, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score", fontsize=FONT_AXIS_LABEL)
    ax.legend(fontsize=FONT_LEGEND, loc="best")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "eval_prediction_standalone.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 6: Backtracking open-ended correctness vs MC accuracy (selected only)
# ---------------------------------------------------------------------------

BACKTRACKING_COMPARISON_GROUP = {
    "key": "backtracking_comparison",
    "title": "Backtracking: Open-ended Correctness vs MC Accuracy",
    "metrics": [
        ("backtracking", "mean_correctness", "Open-ended\nCorrectness", "1-5"),
        ("backtracking_mc", "accuracy", "MC\nAccuracy", "0-1"),
    ],
}


def plot_backtracking_comparison(all_metrics: dict, label_map: dict[str, str],
                                  all_cis: dict | None = None):
    """Side-by-side bar chart: backtracking open-ended correctness vs MC accuracy."""
    selected_loras = resolve_selected_loras()
    headline_lora = resolve_headline_lora()

    n_loras = len(selected_loras)

    # Build data manually since the two metrics have different scales
    metrics_spec = BACKTRACKING_COMPARISON_GROUP["metrics"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, (eval_pattern, metric_key, label, scale) in zip(axes, metrics_spec):
        eval_data = all_metrics.get(eval_pattern, {})
        ci_data = all_cis.get(eval_pattern, {}) if all_cis else {}

        bar_width = 0.8 / n_loras
        x = np.arange(1)

        for i, lora in enumerate(selected_loras):
            val = eval_data.get(lora, {}).get(metric_key)
            lora_val = [val if val is not None else 0]
            ci = ci_data.get(lora, {}).get(metric_key)
            ci_bounds = [ci]
            offset = (i - n_loras / 2 + 0.5) * bar_width
            lbl = label_map.get(lora, shorten_lora(lora))
            is_headline = (lora == headline_lora)
            draw_bars(ax, x, lora_val, bar_width, offset,
                      BAR_COLORS[i % len(BAR_COLORS)], lbl, is_headline,
                      ci_bounds=ci_bounds)

        ax.set_xticks(x)
        ax.set_xticklabels([label], fontsize=FONT_TICK, ha="center")
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.grid(axis="y", alpha=0.3)

        if scale == "1-5":
            ax.set_ylim(0, 5.0)
            ax.set_ylabel("Score (1-5)", fontsize=FONT_AXIS_LABEL)
        else:
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Score", fontsize=FONT_AXIS_LABEL)

    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=n_loras,
               fontsize=FONT_LEGEND, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle("Backtracking: Open-ended vs Multiple Choice",
                 fontsize=FONT_SUPTITLE, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    out = os.path.join(OUTPUT_DIR, "eval_backtracking_comparison.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 7: Baseline comparison — AO vs Linear Probe vs Text-Only
# ---------------------------------------------------------------------------

# Paths to baseline result files
BASELINE_LINEAR_PROBE_PATHS = {
    "sycophancy/no_cot": "experiments/open_ended_linear_probe_results/val_sweeps/sycophancy_no_cot_mean_all_val_sweep.json",
    "sycophancy/cot": "experiments/open_ended_linear_probe_results/val_sweeps/sycophancy_cot_mean_all_val_sweep.json",
    "mmlu_prediction/pre_answer": "experiments/open_ended_linear_probe_results/val_sweeps/mmlu_pre_answer_mean_all_val_sweep.json",
    "mmlu_prediction/post_answer": "experiments/open_ended_linear_probe_results/val_sweeps/mmlu_post_answer_mean_all_val_sweep.json",
}

BASELINE_TEXT_ONLY_PATHS = {
    "sycophancy/no_cot": "experiments/sycophancy_text_baseline_no_cot_results/Qwen3-8B/sycophancy_no_cot_text_baseline_hf_binary_yes_no.json",
    "sycophancy/cot": "experiments/sycophancy_text_baseline_cot_results/Qwen3-8B/sycophancy_cot_text_baseline_hf_binary_yes_no.json",
    "mmlu_prediction/pre_answer": "experiments/mmlu_prediction_text_baseline_pre_answer_seg10_results/Qwen3-8B/mmlu_prediction_text_baseline_hf_binary_yes_no.json",
    "mmlu_prediction/post_answer": "experiments/mmlu_prediction_text_baseline_post_answer_seg10_results/Qwen3-8B/mmlu_prediction_text_baseline_hf_binary_yes_no.json",
}

BASELINE_TEXT_ONLY_BACKTRACKING_PATH = "experiments/backtracking_text_baseline_seg20_results/Qwen3-8B/backtracking_text_baseline_vllm_rollout.json"

BASELINE_TASK_LABELS = {
    "mmlu_prediction/pre_answer": "MMLU\nPre-Answer",
    "mmlu_prediction/post_answer": "MMLU\nPost-Answer",
    "sycophancy/no_cot": "Sycophancy\nNo CoT",
    "sycophancy/cot": "Sycophancy\nCoT",
}

# Task ordering for the plot
BASELINE_TASK_ORDER = [
    "mmlu_prediction/pre_answer",
    "mmlu_prediction/post_answer",
    "sycophancy/no_cot",
    "sycophancy/cot",
]


def load_baseline_values() -> dict[str, dict[str, float]]:
    """Load ROC AUC values for baselines.

    Returns: {"linear_probe": {task: roc_auc}, "text_only": {task: roc_auc}}

    Text-only baseline uses the full_context variant roc_auc (model sees the
    full conversation text).
    """
    baselines: dict[str, dict[str, float]] = {"linear_probe": {}, "text_only": {}}

    for task, path in BASELINE_LINEAR_PROBE_PATHS.items():
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            baselines["linear_probe"][task] = data["test_retrained_on_full_train"]["roc_auc"]

    for task, path in BASELINE_TEXT_ONLY_PATHS.items():
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            baselines["text_only"][task] = data["metrics"]["variant_full_context_roc_auc"]

    return baselines


def load_text_only_cis() -> dict[str, tuple[float, float]]:
    """Bootstrap AUC CIs for text-only baselines (full_context variant only)."""
    cis: dict[str, tuple[float, float]] = {}
    for task, path in BASELINE_TEXT_ONLY_PATHS.items():
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        entries = [sr for sr in data["scored_results"]
                   if sr.get("baseline_variant") == "full_context"]
        if len(entries) < 2:
            continue
        labels = np.array([1 if e["ground_truth"] == "yes" else 0 for e in entries])
        scores = np.array([e["margin_yes_minus_no"] for e in entries])
        if len(np.unique(labels)) < 2:
            continue
        cis[task] = bootstrap_auc_ci(labels, scores)
    return cis


def load_linear_probe_cis() -> dict[str, tuple[float, float]]:
    """Load per-entry predictions from linear probe val sweeps and bootstrap AUC CIs."""
    cis: dict[str, tuple[float, float]] = {}
    for task, path in BASELINE_LINEAR_PROBE_PATHS.items():
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        entries = data.get("per_entry_test")
        if not entries or len(entries) < 2:
            continue
        labels = np.array([e["ground_truth"] for e in entries])
        scores = np.array([e["logit"] for e in entries])
        if len(np.unique(labels)) < 2:
            continue
        cis[task] = bootstrap_auc_ci(labels, scores)
    return cis


SHOW_BEST_AO = True  # Toggle: show best-across-all-runs AO marker per task


def _draw_baseline_panel_binary(ax, all_metrics, all_cis, selected_loras,
                                headline_lora, label_map, baselines, text_cis, probe_cis):
    """Left panel: binary classification ROC AUC tasks."""
    tasks = [t for t in BASELINE_TASK_ORDER if t in all_metrics]
    if not tasks:
        return

    x_labels = [BASELINE_TASK_LABELS[t] for t in tasks]
    n_tasks = len(tasks)

    n_ao = len(selected_loras)
    n_extra = 2 + (1 if SHOW_BEST_AO else 0)
    n_total = n_ao + n_extra
    bar_width = 0.8 / n_total
    x = np.arange(n_tasks)

    bar_idx = 0

    for i, lora in enumerate(selected_loras):
        vals = []
        ci_bounds_list = []
        for task in tasks:
            eval_data = all_metrics.get(task, {})
            vals.append(eval_data.get(lora, {}).get("roc_auc"))
            ci = all_cis.get(task, {}).get(lora, {}).get("roc_auc") if all_cis else None
            ci_bounds_list.append(ci)

        lora_vals = [v if v is not None else 0 for v in vals]
        offset = (bar_idx - n_total / 2 + 0.5) * bar_width
        label = label_map.get(lora, shorten_lora(lora))
        is_headline = (lora == headline_lora)
        draw_bars(ax, x, lora_vals, bar_width, offset,
                  BAR_COLORS[i % len(BAR_COLORS)], label, is_headline,
                  ci_bounds=ci_bounds_list)
        bar_idx += 1

    if SHOW_BEST_AO:
        best_ao_vals = []
        best_ao_ci_list = []
        for task in tasks:
            eval_data = all_metrics.get(task, {})
            best_auc = 0
            best_lora = None
            for lora, metrics in eval_data.items():
                auc = metrics.get("roc_auc")
                if auc is not None and auc > best_auc:
                    best_auc = auc
                    best_lora = lora
            best_ao_vals.append(best_auc)
            ci = all_cis.get(task, {}).get(best_lora, {}).get("roc_auc") if all_cis and best_lora else None
            best_ao_ci_list.append(ci)

        offset = (bar_idx - n_total / 2 + 0.5) * bar_width
        draw_bars(ax, x, best_ao_vals, bar_width, offset,
                  "#e6194b", "Best AO (any run)", False,
                  ci_bounds=best_ao_ci_list)
        bar_idx += 1

    text_vals = [baselines["text_only"].get(t, 0) for t in tasks]
    text_ci_list = [text_cis.get(t) for t in tasks]
    text_yerr = None
    if any(ci is not None for ci in text_ci_list):
        yerr_lo = []
        yerr_hi = []
        for val, ci in zip(text_vals, text_ci_list):
            if ci is not None and not (np.isnan(ci[0]) or np.isnan(ci[1])):
                yerr_lo.append(max(0, val - ci[0]))
                yerr_hi.append(max(0, ci[1] - val))
            else:
                yerr_lo.append(0)
                yerr_hi.append(0)
        text_yerr = [yerr_lo, yerr_hi]

    offset = (bar_idx - n_total / 2 + 0.5) * bar_width
    bars = ax.bar(x + offset, text_vals, bar_width * 0.9,
                  label="Text-Only Baseline", color="#cccccc",
                  alpha=0.85, edgecolor="black", linewidth=0.8,
                  hatch="//", yerr=text_yerr, capsize=3,
                  error_kw={"elinewidth": 1.2, "capthick": 1.2})
    for bar, val, ci in zip(bars, text_vals, text_ci_list):
        if val > 0:
            top = ci[1] if ci and not np.isnan(ci[1]) else bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, top + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=FONT_BAR_LABEL)
    bar_idx += 1

    probe_vals = [baselines["linear_probe"].get(t, 0) for t in tasks]
    probe_ci_list = [probe_cis.get(t) for t in tasks]
    probe_yerr = None
    if any(ci is not None for ci in probe_ci_list):
        yerr_lo = []
        yerr_hi = []
        for val, ci in zip(probe_vals, probe_ci_list):
            if ci is not None and not (np.isnan(ci[0]) or np.isnan(ci[1])):
                yerr_lo.append(max(0, val - ci[0]))
                yerr_hi.append(max(0, ci[1] - val))
            else:
                yerr_lo.append(0)
                yerr_hi.append(0)
        probe_yerr = [yerr_lo, yerr_hi]

    offset = (bar_idx - n_total / 2 + 0.5) * bar_width
    bars = ax.bar(x + offset, probe_vals, bar_width * 0.9,
                  label="Linear Probe", color="#888888",
                  alpha=0.85, edgecolor="black", linewidth=0.8,
                  hatch="xx", yerr=probe_yerr, capsize=3,
                  error_kw={"elinewidth": 1.2, "capthick": 1.2})
    for bar, val, ci in zip(bars, probe_vals, probe_ci_list):
        if val > 0:
            top = ci[1] if ci and not np.isnan(ci[1]) else bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, top + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=FONT_BAR_LABEL)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=FONT_TICK, ha="center")
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_title("Binary Classification (ROC AUC)", fontsize=FONT_TITLE, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC AUC", fontsize=FONT_AXIS_LABEL)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)


def _draw_baseline_panel_backtracking(ax, all_metrics, selected_loras,
                                       headline_lora, label_map):
    """Right panel: backtracking correctness (1-5 scale)."""
    bt_data = all_metrics.get("backtracking", {})
    if not bt_data:
        return

    # Load text-only baseline
    text_baseline_correctness = None
    if os.path.exists(BASELINE_TEXT_ONLY_BACKTRACKING_PATH):
        with open(BASELINE_TEXT_ONLY_BACKTRACKING_PATH) as f:
            data = json.load(f)
        fc_results = [r for r in data["scored_results"] if r.get("baseline_variant") == "full_context"]
        if fc_results:
            text_baseline_correctness = sum(r["correctness"] for r in fc_results) / len(fc_results)

    # Bars: selected AO loras + best AO + text-only baseline
    n_ao = len(selected_loras)
    n_extra = 1 + (1 if SHOW_BEST_AO else 0)  # text-only, (best AO)
    n_total = n_ao + n_extra
    bar_width = 0.8 / n_total
    x = np.arange(1)  # single group

    bar_idx = 0

    for i, lora in enumerate(selected_loras):
        val = bt_data.get(lora, {}).get("mean_correctness", 0)
        offset = (bar_idx - n_total / 2 + 0.5) * bar_width
        label = label_map.get(lora, shorten_lora(lora))
        is_headline = (lora == headline_lora)
        draw_bars(ax, x, [val], bar_width, offset,
                  BAR_COLORS[i % len(BAR_COLORS)], label, is_headline)
        bar_idx += 1

    if SHOW_BEST_AO:
        best_val = 0
        for lora, metrics in bt_data.items():
            corr = metrics.get("mean_correctness", 0)
            if corr > best_val:
                best_val = corr
        offset = (bar_idx - n_total / 2 + 0.5) * bar_width
        draw_bars(ax, x, [best_val], bar_width, offset,
                  "#e6194b", "Best AO (any run)", False)
        bar_idx += 1

    if text_baseline_correctness is not None:
        offset = (bar_idx - n_total / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, [text_baseline_correctness], bar_width * 0.9,
                      label="Text-Only Baseline", color="#cccccc",
                      alpha=0.85, edgecolor="black", linewidth=0.8,
                      hatch="//")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"{text_baseline_correctness:.2f}", ha="center", va="bottom",
                    fontsize=FONT_BAR_LABEL)

    ax.set_xticks(x)
    ax.set_xticklabels(["Backtracking"], fontsize=FONT_TICK, ha="center")
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_title("Backtracking (Correctness)", fontsize=FONT_TITLE, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 5.2)
    ax.set_ylabel("Mean Correctness (1-5)", fontsize=FONT_AXIS_LABEL)


def plot_baseline_comparison(all_metrics: dict, label_map: dict[str, str],
                              all_cis: dict | None = None):
    """Two-panel bar chart: binary AUC tasks (left) + backtracking correctness (right)."""
    selected_loras = resolve_selected_loras()
    headline_lora = resolve_headline_lora()
    baselines = load_baseline_values()
    text_cis = load_text_only_cis()
    probe_cis = load_linear_probe_cis()

    has_binary = any(t in all_metrics for t in BASELINE_TASK_ORDER)
    has_backtracking = "backtracking" in all_metrics

    if not has_binary and not has_backtracking:
        print("No baseline comparison data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                              gridspec_kw={"width_ratios": [4, 1.2]})

    _draw_baseline_panel_binary(axes[0], all_metrics, all_cis, selected_loras,
                                headline_lora, label_map, baselines, text_cis, probe_cis)
    _draw_baseline_panel_backtracking(axes[1], all_metrics, selected_loras,
                                       headline_lora, label_map)

    # Unified legend from both panels
    handles, leg_labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in leg_labels:
                handles.append(hi)
                leg_labels.append(li)

    fig.legend(handles, leg_labels, loc="lower center", ncol=min(len(handles), 5),
               fontsize=FONT_LEGEND, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("AO vs Baselines", fontsize=FONT_SUPTITLE, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    out = os.path.join(OUTPUT_DIR, "eval_baseline_comparison.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lora_keys = list(ALL_LORAS.keys())
    result_dirs = [RESULT_DIR_PREFIX + k for k in lora_keys]
    result_dirs = [d for d in result_dirs if Path(d).exists()]
    dir_labels = {k: ALL_LORAS[k] for k in lora_keys}

    print(f"Loading from {len(result_dirs)} result dirs...")
    all_metrics = load_all_metrics(result_dirs)
    label_map = build_label_map_from_dir_labels(all_metrics, result_dirs, dir_labels)

    print("Computing confidence intervals (bootstrap)...")
    all_cis = load_all_cis(result_dirs)

    print("Generating selected LoRA bar chart...")
    plot_selected_bars(all_metrics, label_map, all_cis=all_cis)

    print("Generating selected LoRA bar chart 2 (with new models)...")
    plot_selected_bars_2(all_metrics, label_map, all_cis=all_cis)

    print("Generating all LoRA highlighted line chart...")
    plot_all_lines(all_metrics, label_map)

    print("Generating binary classification detail...")
    plot_binary_detail(all_metrics, label_map, all_cis=all_cis)

    print("Generating open-ended detail...")
    plot_open_ended_detail(all_metrics, label_map, all_cis=all_cis)

    print("Generating prediction standalone...")
    plot_prediction_standalone(all_metrics, label_map, all_cis=all_cis)

    print("Generating backtracking comparison...")
    plot_backtracking_comparison(all_metrics, label_map, all_cis=all_cis)

    print("Generating baseline comparison...")
    plot_baseline_comparison(all_metrics, label_map, all_cis=all_cis)

    print("\nDone! All plots in experiments/eval_plots/")


if __name__ == "__main__":
    main()
