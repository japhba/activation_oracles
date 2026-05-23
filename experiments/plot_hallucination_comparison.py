"""Hallucination detection bar charts: span-only and last-100 segment strategies.

Usage:
    .venv/bin/python experiments/plot_hallucination_comparison.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.plot_eval_grouped import (
    _fast_auc,
    FONT_TITLE,
    FONT_AXIS_LABEL,
    FONT_TICK,
    FONT_BAR_LABEL,
    FONT_LEGEND,
    N_BOOTSTRAP,
    RNG,
)

OUTPUT_DIR = "experiments/eval_plots"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ── Colors ──
COLOR_ORIGINAL = "#f58231"   # Original AO (past_lens)
COLOR_MLAO = "#e6194b"       # MLAO (on_policy)
COLOR_AO_V25 = "#4363d8"    # AO v2.3 (500k_pl)
COLOR_TEXT = "#cccccc"       # text baseline
COLOR_LINEAR = "#888888"     # linear probe

# ── Bar order: Original AO → MLAO → AO v2.3 → Base Model → Linear Probe ──
AO_BAR_ORDER = ["Original AO", "MLAO", "AO v2.3"]

AO_RESULTS = {
    "span": {
        "Original AO": "experiments/hallucination_eval_results/Qwen3-8B/hallucination_binary_checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B.json",
        "MLAO": "experiments/hallucination_eval_results/Qwen3-8B/hallucination_binary_checkpoints_latentqa_cls_on_policy_Qwen3-8B.json",
        "AO v2.3": "experiments/hallucination_eval_results/chatreg_variants/hallu_chatreg_span_binary_final.json",
    },
    "last100": {
        "Original AO": "experiments/hallucination_eval_results/last100_segment/hallucination_last100_binary_checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B.json",
        "MLAO": "experiments/hallucination_eval_results/last100_segment/hallucination_last100_binary_checkpoints_latentqa_cls_on_policy_Qwen3-8B.json",
        "AO v2.3": "experiments/hallucination_eval_results/chatreg_variants/hallu_chatreg_last100_binary_final.json",
    },
}

TEXT_BASELINE_PATH = "experiments/hallucination_text_baseline_results/Qwen3-8B/hallucination_text_baseline_hf_binary_yes_no.json"
LINEAR_PROBE_PATH = "experiments/hallucination_eval_results/linear_probe_results.json"

AO_COLORS = {
    "Original AO": COLOR_ORIGINAL,
    "MLAO": COLOR_MLAO,
    "AO v2.3": COLOR_AO_V25,
}


def _split_by_prompt(entries: list[dict], label_key: str = "ground_truth",
                     label_pos: str = "yes") -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split entries by prompt_name, return {prompt: (labels, scores)}."""
    by_prompt: dict[str, tuple[list, list]] = {}
    for e in entries:
        p = e["prompt_name"]
        if p not in by_prompt:
            by_prompt[p] = ([], [])
        lab = 1 if e.get(label_key) == label_pos or e.get("binary_label") == 1 else 0
        by_prompt[p][0].append(lab)
        by_prompt[p][1].append(e.get("margin_yes_minus_no", 0))
    return {p: (np.array(ls), np.array(ss)) for p, (ls, ss) in by_prompt.items()}


def _bootstrap_mean_prompt_auc(prompt_data: dict[str, tuple[np.ndarray, np.ndarray]],
                               alpha: float = 0.05) -> tuple[float, tuple[float, float]]:
    """Compute mean-of-per-prompt AUC and bootstrap CI by resampling entries within each prompt."""
    # Point estimate
    prompt_aucs = []
    for labels, scores in prompt_data.values():
        prompt_aucs.append(_fast_auc(labels, scores))
    mean_auc = float(np.mean(prompt_aucs))

    # Bootstrap: resample entries (by index) within each prompt, compute per-prompt AUC, average
    # Use the same resampled indices across prompts (they all have the same entries, just different scores)
    n = len(next(iter(prompt_data.values()))[0])
    idx = RNG.randint(0, n, size=(N_BOOTSTRAP, n))
    boot_stats = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        per_prompt = []
        for labels, scores in prompt_data.values():
            bl, bs = labels[idx[i]], scores[idx[i]]
            if len(np.unique(bl)) < 2:
                per_prompt.append(float("nan"))
            else:
                per_prompt.append(_fast_auc(bl, bs))
        boot_stats[i] = np.nanmean(per_prompt)
    boot_stats = boot_stats[~np.isnan(boot_stats)]
    if len(boot_stats) == 0:
        return mean_auc, (float("nan"), float("nan"))
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return mean_auc, (lo, hi)


def compute_ao_auc_and_ci(data: dict) -> tuple[float, tuple[float, float]]:
    """Mean-of-per-prompt AUC with bootstrapped CI for AO results."""
    entries = data["binary_scored_results"]
    prompt_data = _split_by_prompt(entries)
    return _bootstrap_mean_prompt_auc(prompt_data)


def compute_text_baseline_auc_and_ci(data: dict) -> tuple[float, tuple[float, float]]:
    """Mean-of-per-prompt AUC with bootstrapped CI for text baseline."""
    entries = data["scored_results"]
    prompt_data = _split_by_prompt(entries, label_key="ground_truth", label_pos="yes")
    return _bootstrap_mean_prompt_auc(prompt_data)


def draw_bar(ax, x_pos, val, width, color, label, ci=None, hatch=None,
             edgecolor="white", linewidth=0.5, bold=False):
    """Draw a single bar with optional CI error bar."""
    if bold:
        edgecolor = "black"
        linewidth = 2.5

    yerr = None
    if ci is not None and not (np.isnan(ci[0]) or np.isnan(ci[1])):
        yerr = [[max(0, val - ci[0])], [max(0, ci[1] - val)]]

    ax.bar(x_pos, val, width * 0.9, label=label, color=color,
           alpha=0.85, edgecolor=edgecolor, linewidth=linewidth,
           hatch=hatch, yerr=yerr, capsize=4,
           error_kw={"elinewidth": 1.2, "capthick": 1.2})
    top = ci[1] if ci and not np.isnan(ci[1]) else val
    ax.text(x_pos, top + 0.01, f"{val:.2f}", ha="center", va="bottom",
            fontsize=FONT_BAR_LABEL)


def hanley_mcneil_ci(auc: float, n_pos: int, n_neg: int,
                     alpha: float = 0.05) -> tuple[float, float]:
    """Hanley-McNeil approximation for AUC 95% CI (no per-entry data needed)."""
    import scipy.stats as st
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    se = np.sqrt((auc*(1-auc) + (n_pos-1)*(q1-auc**2) + (n_neg-1)*(q2-auc**2))
                 / (n_pos * n_neg))
    z = st.norm.ppf(1 - alpha / 2)
    return (auc - z * se, auc + z * se)


def plot_hallucination_segment(segment: str, title_suffix: str, output_name: str,
                               include_linear_probe: bool = True):
    """Generate a single hallucination comparison bar chart for a given segment strategy."""
    ao_paths = AO_RESULTS[segment]
    ao_names = AO_BAR_ORDER

    # Load AO data
    ao_data = {}
    for name, path in ao_paths.items():
        with open(path) as f:
            ao_data[name] = json.load(f)

    # Load baselines
    with open(TEXT_BASELINE_PATH) as f:
        text_data = json.load(f)

    # Bar order
    bar_labels = ao_names + ["Base Model"]
    if include_linear_probe:
        with open(LINEAR_PROBE_PATH) as f:
            linear_data = json.load(f)
        bar_labels.append("Linear Probe")

    n_bars = len(bar_labels)
    bar_width = 0.8 / n_bars
    x = np.array([0.0])  # single group

    fig, ax = plt.subplots(figsize=(10, 7))

    bar_idx = 0
    for name in ao_names:
        data = ao_data[name]
        auc, ci = compute_ao_auc_and_ci(data)
        offset = (bar_idx - n_bars / 2 + 0.5) * bar_width
        draw_bar(ax, x[0] + offset, auc, bar_width, AO_COLORS[name],
                 name, ci=ci, bold=(name == "AO v2.3"))
        bar_idx += 1

    # Text baseline
    text_auc, text_ci = compute_text_baseline_auc_and_ci(text_data)
    offset = (bar_idx - n_bars / 2 + 0.5) * bar_width
    draw_bar(ax, x[0] + offset, text_auc, bar_width, COLOR_TEXT,
             "Base Model", ci=text_ci, hatch="//", edgecolor="black", linewidth=0.8)
    bar_idx += 1

    # Linear probe (Hanley-McNeil CI since no per-entry data saved)
    if include_linear_probe:
        linear_auc = linear_data["test_roc_auc"]
        # n_pos/n_neg from the test split (971 supported, 1030 not supported)
        linear_ci = hanley_mcneil_ci(linear_auc, n_pos=971, n_neg=1030)
        offset = (bar_idx - n_bars / 2 + 0.5) * bar_width
        draw_bar(ax, x[0] + offset, linear_auc, bar_width, COLOR_LINEAR,
                 "Linear Probe", ci=linear_ci, hatch="xx", edgecolor="black", linewidth=0.8)

    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_title(f"Hallucination Detection — {title_suffix}\n(ROC AUC, mean of 3 prompts)",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("ROC AUC", fontsize=FONT_AXIS_LABEL)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    handles, leg_labels = ax.get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=len(handles),
               fontsize=FONT_LEGEND, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1.0])
    out = str(Path(OUTPUT_DIR) / output_name)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    plot_hallucination_segment("span", "Span Tokens", "hallucination_span_comparison.png",
                              include_linear_probe=True)
    plot_hallucination_segment("last100", "Last 100 Tokens", "hallucination_last100_comparison.png",
                              include_linear_probe=False)
