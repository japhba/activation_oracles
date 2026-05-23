"""Clean plots for Slack update.

Produces three plots:
1. Binary classification (ROC AUC): AO v3 vs text-only baseline vs linear probe.
2. Backtracking correctness: AO v3 at different context lengths + text-only baseline.
3. 4-subplot overview: AO v3 + Best AO (any run) across all eval groups.

AO v3 = 500k_pl_31k_spqav2_199k_sqav3_126k_cls_100k_chatreg_2ep

Usage:
    .venv/bin/python experiments/plot_clean_update.py
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.plot_eval_grouped import (
    bootstrap_auc_ci,
    bootstrap_ci,
    compute_ci_for_result,
    FONT_SUPTITLE,
    FONT_TITLE,
    FONT_AXIS_LABEL,
    FONT_TICK,
    FONT_BAR_LABEL,
    FONT_LEGEND,
    GROUPS,
    is_1_5_group,
)
from experiments.plot_eval_summaries import (
    ALL_LORAS,
    RESULT_DIR_PREFIX,
    find_eval_jsons,
    load_all_metrics,
    extract_lora_name,
    shorten_lora,
)

OUTPUT_DIR = "experiments/eval_plots"

# ── AO v3 config ──
AO_V3_KEY = "500k_pl_31k_spqav2_199k_sqav3_126k_cls_100k_chatreg_2ep"
AO_V3_RESULT_DIR = f"experiments/eval_results_{AO_V3_KEY}"
AO_V3_LABEL = "AO v2.3"

# ── Baseline paths (same as plot_eval_grouped.py) ──
BASELINE_TEXT_ONLY_PATHS = {
    "sycophancy/no_cot": "experiments/sycophancy_text_baseline_no_cot_results/Qwen3-8B/sycophancy_no_cot_text_baseline_hf_binary_yes_no.json",
    "sycophancy/cot": "experiments/sycophancy_text_baseline_cot_results/Qwen3-8B/sycophancy_cot_text_baseline_hf_binary_yes_no.json",
    "mmlu_prediction/pre_answer": "experiments/mmlu_prediction_text_baseline_pre_answer_seg10_results/Qwen3-8B/mmlu_prediction_text_baseline_hf_binary_yes_no.json",
    "mmlu_prediction/post_answer": "experiments/mmlu_prediction_text_baseline_post_answer_seg10_results/Qwen3-8B/mmlu_prediction_text_baseline_hf_binary_yes_no.json",
}

BASELINE_LINEAR_PROBE_PATHS = {
    "sycophancy/no_cot": "experiments/open_ended_linear_probe_results/val_sweeps/sycophancy_no_cot_mean_all_val_sweep.json",
    "sycophancy/cot": "experiments/open_ended_linear_probe_results/val_sweeps/sycophancy_cot_mean_all_val_sweep.json",
    "mmlu_prediction/pre_answer": "experiments/open_ended_linear_probe_results/val_sweeps/mmlu_pre_answer_mean_all_val_sweep.json",
    "mmlu_prediction/post_answer": "experiments/open_ended_linear_probe_results/val_sweeps/mmlu_post_answer_mean_all_val_sweep.json",
}

BASELINE_TEXT_ONLY_BACKTRACKING_PATH = "experiments/backtracking_text_baseline_seg20_results/Qwen3-8B/backtracking_text_baseline_vllm_rollout.json"

# ── Context length sweep paths ──
CONTEXT_SWEEP_DIR = "experiments/context_length_sweep_results/backtracking"
AO_BEST_OF_10_PATH = "experiments/backtracking_ao_best_of_10_full_context/backtracking_best_of_n_final.json"
AO_CONSENSUS_10_PATH = "experiments/backtracking_ao_consensus_10_full_context/backtracking_consensus_final.json"
MMLU_CONTEXT_SWEEP_DIR = "experiments/context_length_sweep_results/mmlu_prediction"

# Map from binary task to full-context sweep JSON path (only tasks where default != full)
MMLU_FULL_CONTEXT_PATHS = {
    "mmlu_prediction/pre_answer": f"{MMLU_CONTEXT_SWEEP_DIR}/seg_full/pre_answer/mmlu_prediction_binary_final.json",
    "mmlu_prediction/post_answer": f"{MMLU_CONTEXT_SWEEP_DIR}/seg_full/post_answer/mmlu_prediction_binary_final.json",
}

BINARY_TASKS = [
    ("mmlu_prediction/pre_answer", "MMLU\nPre-Answer"),
    ("mmlu_prediction/post_answer", "MMLU\nPost-Answer"),
    ("sycophancy/no_cot", "Sycophancy\nNo CoT"),
    ("sycophancy/cot", "Sycophancy\nCoT"),
]

# Colors
COLOR_AO = "#4363d8"
COLOR_TEXT_BASELINE = "#cccccc"
COLOR_LINEAR_PROBE = "#888888"
COLOR_SEG50 = "#3cb44b"
COLOR_SEG_FULL = "#f58231"
COLOR_BEST_AO = "#e6194b"
COLOR_AO_V2 = "#3cb44b"
COLOR_ORIGINAL_AO = "#f58231"
COLOR_BEST_OF_N = "#911eb4"
COLOR_CONSENSUS = "#42d4f4"

# ── Additional AO configs ──
AO_V2_KEY = "full_mix_500k_past_lens_plus_synthetic_qa_v3"
AO_V2_RESULT_DIR = f"experiments/eval_results_{AO_V2_KEY}"
AO_V2_LABEL = "AO v2"

ORIGINAL_AO_KEY = "past_lens_addition_Qwen3-8B"
ORIGINAL_AO_RESULT_DIR = f"experiments/eval_results_{ORIGINAL_AO_KEY}"
ORIGINAL_AO_LABEL = "Original AO"


# ── Data loading helpers ──

def load_ao_binary_results(result_dir: str) -> dict[str, dict]:
    """Load AO binary eval results. Returns {task: data_dict}."""
    results = {}
    for task, _ in BINARY_TASKS:
        parts = task.split("/")
        # e.g. mmlu_prediction/pre_answer -> result_dir/mmlu_prediction/pre_answer/*.json
        search_dir = Path(result_dir) / parts[0]
        if len(parts) > 1:
            search_dir = search_dir / parts[1]
        if not search_dir.exists():
            continue
        for json_file in sorted(search_dir.glob("*.json")):
            if "_summary" in json_file.name:
                continue
            with open(json_file) as f:
                data = json.load(f)
            if "binary_score_metrics" in data:
                results[task] = data
                break
    return results


def extract_auc_and_ci(data: dict) -> tuple[float, tuple[float, float]]:
    """Extract ROC AUC and bootstrap CI from binary eval data."""
    auc = data["binary_score_metrics"]["roc_auc"]
    entries = data["binary_scored_results"]
    labels = np.array([1 if e.get("ground_truth") == "yes" or e.get("binary_label") == 1 else 0 for e in entries])
    scores = np.array([e.get("margin_yes_minus_no", 0) for e in entries])
    ci = bootstrap_auc_ci(labels, scores)
    return auc, ci


def load_text_baseline_auc(task: str) -> tuple[float | None, tuple[float, float] | None]:
    """Load text-only baseline AUC (full_context variant) and CI."""
    path = BASELINE_TEXT_ONLY_PATHS.get(task)
    if not path or not os.path.exists(path):
        return None, None
    with open(path) as f:
        data = json.load(f)
    auc = data["metrics"]["variant_full_context_roc_auc"]
    entries = [sr for sr in data["scored_results"] if sr.get("baseline_variant") == "full_context"]
    if len(entries) < 2:
        return auc, None
    labels = np.array([1 if e["ground_truth"] == "yes" else 0 for e in entries])
    scores = np.array([e["margin_yes_minus_no"] for e in entries])
    ci = bootstrap_auc_ci(labels, scores)
    return auc, ci


def load_linear_probe_auc(task: str) -> tuple[float | None, tuple[float, float] | None]:
    """Load linear probe AUC and CI."""
    path = BASELINE_LINEAR_PROBE_PATHS.get(task)
    if not path or not os.path.exists(path):
        return None, None
    with open(path) as f:
        data = json.load(f)
    auc = data["test_retrained_on_full_train"]["roc_auc"]
    entries = data.get("per_entry_test")
    if not entries or len(entries) < 2:
        return auc, None
    labels = np.array([e["ground_truth"] for e in entries])
    scores = np.array([e["logit"] for e in entries])
    ci = bootstrap_auc_ci(labels, scores)
    return auc, ci


def load_backtracking_correctness_and_ci(path: str) -> tuple[float, tuple[float, float]]:
    """Load mean correctness and bootstrap CI from backtracking result JSON."""
    with open(path) as f:
        data = json.load(f)
    entries = data["scored_results"]
    vals = np.array([e["correctness"] for e in entries], dtype=float)
    mean = float(vals.mean())
    ci = bootstrap_ci(vals)
    return mean, ci


# ── Drawing helpers ──

def draw_bar(ax, x_pos, val, width, color, label, ci=None, hatch=None,
             edgecolor="white", linewidth=0.5, bold=False):
    """Draw a single bar with optional CI error bar."""
    if bold:
        edgecolor = "black"
        linewidth = 2.5

    yerr = None
    if ci is not None and not (np.isnan(ci[0]) or np.isnan(ci[1])):
        yerr = [[max(0, val - ci[0])], [max(0, ci[1] - val)]]

    bar = ax.bar(x_pos, val, width * 0.9, label=label, color=color,
                 alpha=0.85, edgecolor=edgecolor, linewidth=linewidth,
                 hatch=hatch, yerr=yerr, capsize=4,
                 error_kw={"elinewidth": 1.2, "capthick": 1.2})
    top = ci[1] if ci and not np.isnan(ci[1]) else val
    ax.text(x_pos, top + 0.01, f"{val:.2f}", ha="center", va="bottom",
            fontsize=FONT_BAR_LABEL)
    return bar


# ── Plot 1: Binary classification ──

def load_full_context_auc(task: str) -> tuple[float | None, tuple[float, float] | None]:
    """Load full-context sweep AUC and CI for an MMLU task."""
    path = MMLU_FULL_CONTEXT_PATHS.get(task)
    if not path or not os.path.exists(path):
        return None, None
    with open(path) as f:
        data = json.load(f)
    auc = data["binary_score_metrics"]["roc_auc"]
    entries = data["binary_scored_results"]
    labels = np.array([1 if e.get("ground_truth") == "yes" or e.get("binary_label") == 1 else 0 for e in entries])
    scores = np.array([e.get("margin_yes_minus_no", 0) for e in entries])
    ci = bootstrap_auc_ci(labels, scores)
    return auc, ci


COLOR_FULL_CTX = "#911eb4"


def _find_best_ao_for_task(task: str, metric_key: str = "roc_auc") -> tuple[float | None, tuple[float, float] | None]:
    """Find the best AO ROC AUC across all runs for a binary task, with CI."""
    best_auc = None
    best_data = None
    for key in ALL_LORAS:
        result_dir = RESULT_DIR_PREFIX + key
        results = load_ao_binary_results(result_dir)
        if task not in results:
            continue
        data = results[task]
        auc = data["binary_score_metrics"].get(metric_key)
        if auc is not None and (best_auc is None or auc > best_auc):
            best_auc = auc
            best_data = data
    if best_data is None:
        return None, None
    _, ci = extract_auc_and_ci(best_data)
    return best_auc, ci


def plot_binary_clean():
    """Bar chart: AO v2.5 vs best AO vs text-only vs linear probe on binary ROC AUC tasks."""
    ao_results = load_ao_binary_results(AO_V3_RESULT_DIR)

    tasks = [t for t, _ in BINARY_TASKS if t in ao_results]
    x_labels = [lbl for t, lbl in BINARY_TASKS if t in ao_results]

    if not tasks:
        print("No binary AO data found.")
        return

    n_tasks = len(tasks)
    n_bars = 4  # AO, best AO, text-only, linear probe
    bar_width = 0.8 / n_bars
    x = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(14, 7))

    for j, task in enumerate(tasks):
        bar_idx = 0

        # AO v2.5
        auc, ci = extract_auc_and_ci(ao_results[task])
        offset = (bar_idx - n_bars / 2 + 0.5) * bar_width
        draw_bar(ax, x[j] + offset, auc, bar_width, COLOR_AO,
                 AO_V3_LABEL if j == 0 else None, ci=ci, bold=True)
        bar_idx += 1

        # Best AO (any run)
        best_auc, best_ci = _find_best_ao_for_task(task)
        offset = (bar_idx - n_bars / 2 + 0.5) * bar_width
        if best_auc is not None:
            draw_bar(ax, x[j] + offset, best_auc, bar_width, COLOR_BEST_AO,
                     "Best AO (any run)" if j == 0 else None, ci=best_ci)
        bar_idx += 1

        # Text-only baseline
        text_auc, text_ci = load_text_baseline_auc(task)
        offset = (bar_idx - n_bars / 2 + 0.5) * bar_width
        if text_auc is not None:
            draw_bar(ax, x[j] + offset, text_auc, bar_width, COLOR_TEXT_BASELINE,
                     "Base Model" if j == 0 else None, ci=text_ci,
                     hatch="//", edgecolor="black", linewidth=0.8)
        bar_idx += 1

        # Linear probe
        probe_auc, probe_ci = load_linear_probe_auc(task)
        offset = (bar_idx - n_bars / 2 + 0.5) * bar_width
        if probe_auc is not None:
            draw_bar(ax, x[j] + offset, probe_auc, bar_width, COLOR_LINEAR_PROBE,
                     "Linear Probe" if j == 0 else None, ci=probe_ci,
                     hatch="xx", edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=FONT_TICK, ha="center")
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_title("Binary Classification (ROC AUC)", fontsize=FONT_TITLE, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC AUC", fontsize=FONT_AXIS_LABEL)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    handles, leg_labels = ax.get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=len(handles),
               fontsize=FONT_LEGEND, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1.0])
    out = os.path.join(OUTPUT_DIR, "clean_binary_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ── Plot 2: Backtracking with context length variants ──

def plot_backtracking_clean():
    """Bar chart: AO v3 at different context lengths + text-only baseline."""
    # AO v3 default (seg_20 from main eval results)
    ao_default_path = Path(AO_V3_RESULT_DIR) / "backtracking"
    ao_default_json = None
    if ao_default_path.exists():
        for f in sorted(ao_default_path.glob("*.json")):
            if "_summary" not in f.name:
                ao_default_json = f
                break

    # Context sweep variants
    seg_50_path = Path(CONTEXT_SWEEP_DIR) / "seg_50" / "backtracking_final.json"
    seg_full_path = Path(CONTEXT_SWEEP_DIR) / "seg_full" / "backtracking_final.json"

    # Text-only baseline
    text_baseline_path = Path(BASELINE_TEXT_ONLY_BACKTRACKING_PATH)

    # Collect bars: (label, mean, ci, color, hatch, bold)
    bars_data = []

    if ao_default_json and ao_default_json.exists():
        mean, ci = load_backtracking_correctness_and_ci(str(ao_default_json))
        bars_data.append((f"{AO_V3_LABEL}\n(20 tokens)", mean, ci, COLOR_AO, None, True))

    if seg_50_path.exists():
        mean, ci = load_backtracking_correctness_and_ci(str(seg_50_path))
        bars_data.append((f"{AO_V3_LABEL}\n(50 tokens)", mean, ci, COLOR_SEG50, None, False))

    if seg_full_path.exists():
        mean, ci = load_backtracking_correctness_and_ci(str(seg_full_path))
        bars_data.append((f"{AO_V3_LABEL}\n(full context)", mean, ci, COLOR_SEG_FULL, None, False))

    if os.path.exists(AO_CONSENSUS_10_PATH):
        with open(AO_CONSENSUS_10_PATH) as f:
            data = json.load(f)
        vals = np.array([r["correctness"] for r in data["scored_results"]], dtype=float)
        mean = float(vals.mean())
        ci = bootstrap_ci(vals)
        bars_data.append((f"{AO_V3_LABEL}\n(consensus-10,\nfull context)", mean, ci, COLOR_CONSENSUS, None, False))

    if os.path.exists(AO_BEST_OF_10_PATH):
        with open(AO_BEST_OF_10_PATH) as f:
            data = json.load(f)
        vals = np.array([r["correctness"] for r in data["scored_results"]], dtype=float)
        mean = float(vals.mean())
        ci = bootstrap_ci(vals)
        bars_data.append((f"{AO_V3_LABEL}\n(best-of-10,\nfull context)", mean, ci, COLOR_BEST_OF_N, None, False))

    if text_baseline_path.exists():
        with open(text_baseline_path) as f:
            data = json.load(f)
        fc_results = [r for r in data["scored_results"] if r.get("baseline_variant") == "full_context"]
        if fc_results:
            vals = np.array([r["correctness"] for r in fc_results], dtype=float)
            mean = float(vals.mean())
            ci = bootstrap_ci(vals)
            bars_data.append(("Base\nModel", mean, ci, COLOR_TEXT_BASELINE, "//", False))

    if not bars_data:
        print("No backtracking data found.")
        return

    n_bars = len(bars_data)
    x = np.arange(n_bars)
    bar_width = 0.6

    fig, ax = plt.subplots(figsize=(8, 7))

    for j, (label, mean, ci, color, hatch, bold) in enumerate(bars_data):
        draw_bar(ax, x[j], mean, bar_width, color, label, ci=ci, hatch=hatch,
                 edgecolor="black" if (hatch or bold) else "white",
                 linewidth=2.5 if bold else 0.8, bold=False)  # bold handled via params

    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars_data], fontsize=FONT_TICK, ha="center")
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_title("Backtracking — Mean Correctness", fontsize=FONT_TITLE, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 5.0)
    ax.set_ylabel("Mean Correctness (1-5)", fontsize=FONT_AXIS_LABEL)
    fig.set_size_inches(max(12, n_bars * 2), 7)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "clean_backtracking_context.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ── Plot 3: Number prediction ──

NUMBER_PRED_TEXT_BASELINE_PATH = "experiments/number_prediction_text_baseline_seg10_results/Qwen3-8B/number_prediction_text_baseline_vllm_rollout.json"
NUMBER_PRED_SEG10_PATH = "experiments/context_length_sweep_results/number_prediction/seg_10/number_prediction_final.json"
NUMBER_PRED_SEG_FULL_PATH = "experiments/context_length_sweep_results/number_prediction/seg_full/number_prediction_final.json"


def _number_pred_match_rate_ci(entries: list[dict]) -> tuple[float, tuple[float, float]]:
    """Compute match rate and bootstrap CI from number prediction entries."""
    vals = np.array([1.0 if e["matches_model_answer"] else 0.0 for e in entries])
    return float(vals.mean()), bootstrap_ci(vals)


def plot_number_prediction_clean():
    """Bar chart: AO v2.5 (10 tokens vs full context) + base model on number prediction."""
    bars_data = []  # (label, mean, ci, color, hatch, bold)

    # AO v2.5 (10 tokens)
    if os.path.exists(NUMBER_PRED_SEG10_PATH):
        with open(NUMBER_PRED_SEG10_PATH) as f:
            data = json.load(f)
        mean, ci = _number_pred_match_rate_ci(data["scored_results"])
        bars_data.append((f"{AO_V3_LABEL}\n(assistant control tokens)", mean, ci, COLOR_AO, None, True))

    # AO v2.5 (full context)
    if os.path.exists(NUMBER_PRED_SEG_FULL_PATH):
        with open(NUMBER_PRED_SEG_FULL_PATH) as f:
            data = json.load(f)
        mean, ci = _number_pred_match_rate_ci(data["scored_results"])
        bars_data.append((f"{AO_V3_LABEL}\n(full context)", mean, ci, COLOR_SEG_FULL, None, False))

    # Base model (full context)
    if os.path.exists(NUMBER_PRED_TEXT_BASELINE_PATH):
        with open(NUMBER_PRED_TEXT_BASELINE_PATH) as f:
            data = json.load(f)
        fc_entries = [r for r in data["scored_results"] if r.get("baseline_variant") == "full_context"]
        if fc_entries:
            mean, ci = _number_pred_match_rate_ci(fc_entries)
            bars_data.append(("Base\nModel", mean, ci, COLOR_TEXT_BASELINE, "//", False))

    if not bars_data:
        print("No number prediction data found.")
        return

    n_bars = len(bars_data)
    x = np.arange(n_bars)
    bar_width = 0.6

    fig, ax = plt.subplots(figsize=(8, 7))

    for j, (label, mean, ci, color, hatch, bold) in enumerate(bars_data):
        draw_bar(ax, x[j], mean, bar_width, color, label, ci=ci, hatch=hatch,
                 edgecolor="black" if (hatch or bold) else "white",
                 linewidth=2.5 if bold else 0.8, bold=False)

    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars_data], fontsize=FONT_TICK, ha="center")
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_title("Number Prediction — Match Rate", fontsize=FONT_TITLE, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Match Rate", fontsize=FONT_AXIS_LABEL)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "clean_number_prediction.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ── Plot 4: 4-subplot overview (AO v3 + Best AO) ──

def plot_overview_4panel():
    """4-subplot bar chart: AO v3 + Best AO (any run) across all eval groups."""
    # Load metrics from all result dirs
    lora_keys = list(ALL_LORAS.keys())
    result_dirs = [RESULT_DIR_PREFIX + k for k in lora_keys]
    result_dirs = [d for d in result_dirs if Path(d).exists()]

    all_metrics = load_all_metrics(result_dirs)

    # Resolve lora names from result dirs
    ao_configs = [
        (ORIGINAL_AO_KEY, ORIGINAL_AO_LABEL, COLOR_ORIGINAL_AO),
        (AO_V2_KEY, AO_V2_LABEL, COLOR_AO_V2),
        (AO_V3_KEY, AO_V3_LABEL, COLOR_AO),
    ]

    resolved_aos = []  # [(lora_name, label, color, cis_dict)]
    for key, label, color in ao_configs:
        result_dir = RESULT_DIR_PREFIX + key
        lora_name = _resolve_lora_name(result_dir)
        if lora_name is None:
            print(f"Warning: could not resolve lora name for {key}")
            continue
        cis = _load_cis_for_lora(result_dir)
        resolved_aos.append((lora_name, label, color, cis))

    if not resolved_aos:
        print("No AO loras resolved.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(22, 14))
    axes = axes.flatten()

    for idx, group in enumerate(GROUPS):
        ax = axes[idx]
        metrics_spec = group["metrics"]

        x_labels = []
        # Per-AO data: list of (label, color, vals, ci_list)
        ao_data = [(label, color, [], []) for _, label, color, _ in resolved_aos]
        best_ao_vals = []
        best_ao_ci_list = []

        for eval_pattern, metric_key, label, scale in metrics_spec:
            if eval_pattern not in all_metrics:
                continue
            eval_data = all_metrics[eval_pattern]

            # Check if any AO or any run has data
            has_any = False
            for lora_name, _, _, _ in resolved_aos:
                if eval_data.get(lora_name, {}).get(metric_key) is not None:
                    has_any = True
                    break
            # Also check best across all
            best_val = None
            best_lora = None
            for lora, metrics in eval_data.items():
                v = metrics.get(metric_key)
                if v is not None and (best_val is None or v > best_val):
                    best_val = v
                    best_lora = lora

            if not has_any and best_val is None:
                continue

            x_labels.append(label)

            for i, (lora_name, _, _, cis) in enumerate(resolved_aos):
                val = eval_data.get(lora_name, {}).get(metric_key)
                ao_data[i][2].append(val if val is not None else 0)
                ao_data[i][3].append(cis.get(eval_pattern, {}).get(metric_key))

            best_ao_vals.append(best_val if best_val is not None else 0)
            # CI for best lora
            best_lora_is_known = any(ln == best_lora for ln, _, _, _ in resolved_aos)
            if best_lora_is_known:
                for ln, _, _, cis in resolved_aos:
                    if ln == best_lora:
                        best_ao_ci_list.append(cis.get(eval_pattern, {}).get(metric_key))
                        break
            elif best_lora:
                best_ci = _load_ci_for_metric(all_metrics, eval_pattern, metric_key, best_lora)
                best_ao_ci_list.append(best_ci)
            else:
                best_ao_ci_list.append(None)

        if not x_labels:
            ax.set_title(group["title"], fontsize=FONT_TITLE)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        n_metrics = len(x_labels)
        n_bars = len(resolved_aos) + 1  # AOs + best AO
        bar_width = 0.8 / n_bars
        x = np.arange(n_metrics)

        # AO bars
        for i, (_, ao_label, color, _) in enumerate(resolved_aos):
            vals = ao_data[i][2]
            ci_list = ao_data[i][3]
            is_headline = (ao_label == AO_V3_LABEL)
            for j in range(n_metrics):
                offset = (i - n_bars / 2 + 0.5) * bar_width
                draw_bar(ax, x[j] + offset, vals[j], bar_width, color,
                         ao_label if j == 0 else None, ci=ci_list[j],
                         bold=is_headline)

        # Best AO bars
        for j in range(n_metrics):
            offset = (len(resolved_aos) - n_bars / 2 + 0.5) * bar_width
            draw_bar(ax, x[j] + offset, best_ao_vals[j], bar_width, COLOR_BEST_AO,
                     "Best AO (any run)" if j == 0 else None, ci=best_ao_ci_list[j])

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=FONT_TICK, ha="center")
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.set_title(group["title"], fontsize=FONT_TITLE, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        if is_1_5_group(group):
            ax.set_ylim(0, 5.0)
            ax.set_ylabel("Score (1-5)", fontsize=FONT_AXIS_LABEL)
        else:
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score", fontsize=FONT_AXIS_LABEL)

        ax.legend(fontsize=FONT_LEGEND, loc="upper left")

    plt.suptitle(f"{AO_V3_LABEL} vs Best AO (Any Run)", fontsize=FONT_SUPTITLE,
                 fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUTPUT_DIR, "clean_overview_4panel.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def _resolve_lora_name(result_dir: str) -> str | None:
    """Get the lora_name string from a result dir's JSON files."""
    eval_jsons = find_eval_jsons(result_dir)
    for json_files in eval_jsons.values():
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)
            return extract_lora_name(data)
    return None


def _load_cis_for_lora(result_dir: str) -> dict[str, dict[str, tuple[float, float]]]:
    """Load CIs for all metrics from a single result dir.

    Returns: {eval_display_name: {metric_key: (ci_lo, ci_hi)}}
    """
    cis = {}
    needed = {}
    for group in GROUPS:
        for eval_pattern, metric_key, _, _ in group["metrics"]:
            needed.setdefault(eval_pattern, set()).add(metric_key)

    eval_jsons = find_eval_jsons(result_dir)
    for eval_name, json_files in eval_jsons.items():
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)

            rel = json_file.relative_to(Path(result_dir) / eval_name)
            if len(rel.parts) > 1:
                display_name = f"{eval_name}/{'/'.join(rel.parts[:-1])}"
            else:
                display_name = eval_name

            if display_name not in needed:
                continue

            ci_dict = {}
            for metric_key in needed[display_name]:
                ci = compute_ci_for_result(data, metric_key)
                if ci is not None:
                    ci_dict[metric_key] = ci
            if ci_dict:
                cis[display_name] = ci_dict

    return cis


def _load_ci_for_metric(all_metrics, eval_pattern, metric_key, target_lora):
    """Load CI for a specific lora/metric by finding its result file."""
    # Find which result dir contains this lora
    for key in ALL_LORAS:
        result_dir = RESULT_DIR_PREFIX + key
        if not Path(result_dir).exists():
            continue
        eval_jsons = find_eval_jsons(result_dir)
        for eval_name, json_files in eval_jsons.items():
            for json_file in json_files:
                with open(json_file) as f:
                    data = json.load(f)
                lora_name = extract_lora_name(data)
                if lora_name != target_lora:
                    continue

                rel = json_file.relative_to(Path(result_dir) / eval_name)
                if len(rel.parts) > 1:
                    display_name = f"{eval_name}/{'/'.join(rel.parts[:-1])}"
                else:
                    display_name = eval_name

                if display_name == eval_pattern:
                    return compute_ci_for_result(data, metric_key)
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(42)

    print("Generating clean binary comparison plot...")
    plot_binary_clean()

    print("Generating clean backtracking context length plot...")
    plot_backtracking_clean()

    print("Generating clean number prediction plot...")
    plot_number_prediction_clean()

    print("Generating clean 4-panel overview plot...")
    plot_overview_4panel()

    print("\nDone!")


if __name__ == "__main__":
    main()
