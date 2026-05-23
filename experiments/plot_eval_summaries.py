"""Plot comparison of open-ended eval results across verbalizer LoRAs.

Reads individual per-eval JSON files from result directories. Each directory
should contain subdirectories per eval (e.g. mmlu_prediction/, sycophancy/).

Usage:
    python experiments/plot_eval_summaries.py \
        --result-dirs experiments/eval_results_* \
        -o experiments/eval_comparison.png
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Result loading — works on individual JSON files, not all_summaries.json
# ---------------------------------------------------------------------------


def find_eval_jsons(result_dir: str) -> dict[str, list[Path]]:
    """Find all per-eval JSON files grouped by eval name."""
    result_dir = Path(result_dir)
    evals: dict[str, list[Path]] = defaultdict(list)
    if not result_dir.exists():
        return evals
    for json_file in sorted(result_dir.rglob("*.json")):
        if json_file.name in ("all_summaries.json",) or "_summary" in json_file.name:
            continue
        # Eval name is the first subdirectory under result_dir
        rel = json_file.relative_to(result_dir)
        eval_name = rel.parts[0]
        evals[eval_name].append(json_file)
    return evals


def extract_lora_name(data: dict) -> str:
    """Extract short LoRA name from a result JSON."""
    verbalizer = data.get("verbalizer") or data.get("verbalizer_lora_path") or "unknown"
    return verbalizer.split("/")[-1]


# ---------------------------------------------------------------------------
# Metric extraction per eval type
# ---------------------------------------------------------------------------


def extract_binary_metrics(data: dict) -> dict[str, float]:
    """Extract metrics from binary eval result JSON."""
    m = data.get("binary_score_metrics", {})
    return {k: v for k, v in m.items() if isinstance(v, (int, float))}


def extract_generation_metrics(data: dict) -> dict[str, float]:
    """Extract metrics from generation eval result JSON."""
    m = data.get("metrics", {})
    return {k: v for k, v in m.items() if isinstance(v, (int, float))}


def load_all_metrics(result_dirs: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    """Load metrics from all result dirs.

    Returns: {eval_display_name: {lora_name: {metric: value}}}
    """
    all_metrics: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)

    for result_dir in result_dirs:
        eval_jsons = find_eval_jsons(result_dir)
        for eval_name, json_files in eval_jsons.items():
            for json_file in json_files:
                with open(json_file) as f:
                    data = json.load(f)

                lora_name = extract_lora_name(data)

                # Determine eval type and extract metrics
                if "binary_score_metrics" in data:
                    metrics = extract_binary_metrics(data)
                elif "metrics" in data and data["metrics"]:
                    metrics = {k: v for k, v in data["metrics"].items() if isinstance(v, (int, float))}
                else:
                    continue

                # Build display name including subdirectory context
                rel = json_file.relative_to(Path(result_dir) / eval_name)
                if len(rel.parts) > 1:
                    sub = "/".join(rel.parts[:-1])
                    display_name = f"{eval_name}/{sub}"
                else:
                    display_name = eval_name

                all_metrics[display_name][lora_name] = metrics

    return dict(all_metrics)


# ---------------------------------------------------------------------------
# What to plot for each eval
# ---------------------------------------------------------------------------

# Each entry: (eval_display_name_pattern, metric_key, bar_label, scale)
# scale: "0-1" means already in [0, 1], "1-5" means LLM judge score mapped via (x-1)/4
EVAL_METRICS_TO_PLOT = [
    ("taboo", "full_seq_accuracy", "Taboo\nFull Seq", "0-1"),
    ("taboo", "single_token_accuracy", "Taboo\nSingle Token", "0-1"),
    ("personaqa", "full_seq_accuracy", "PersonaQA\nFull Seq", "0-1"),
    ("personaqa", "single_token_accuracy", "PersonaQA\nSingle Token", "0-1"),
    ("mmlu_prediction/pre_answer", "roc_auc", "MMLU Pre\nROC AUC", "0-1"),
    ("mmlu_prediction/post_answer", "roc_auc", "MMLU Post\nROC AUC", "0-1"),
    ("mmlu_prediction/pre_answer/letter_prediction", "matches_model_rate", "MMLU Letter\nPrediction", "0-1"),
    ("sycophancy/no_cot", "roc_auc", "Syco No-CoT\nROC AUC", "0-1"),
    ("sycophancy/cot", "roc_auc", "Syco CoT\nROC AUC", "0-1"),
    ("missing_info", "A_vs_B_roc_auc", "MissingInfo\nA vs B AUC", "0-1"),
    ("missing_info", "A_vs_C_roc_auc", "MissingInfo\nA vs C AUC", "0-1"),
    ("backtracking", "mean_specificity", "Backtrack\nSpecificity", "1-5"),
    ("backtracking", "mean_correctness", "Backtrack\nCorrectness", "1-5"),
    ("system_prompt_qa_hidden/user_and_assistant", "mean_specificity", "SysPrompt Hid\nSpecificity", "1-5"),
    ("system_prompt_qa_hidden/user_and_assistant", "mean_correctness", "SysPrompt Hid\nCorrectness", "1-5"),
    ("system_prompt_qa_latentqa/user_and_assistant", "mean_specificity", "SysPrompt LQA\nSpecificity", "1-5"),
    ("system_prompt_qa_latentqa/user_and_assistant", "mean_correctness", "SysPrompt LQA\nCorrectness", "1-5"),
    ("number_prediction", "matches_model_answer_rate", "Number Pred\nMatch Rate", "0-1"),
    ("backtracking_mc", "accuracy", "Backtrack MC\nAccuracy", "0-1"),
]


def normalize_value(val: float, scale: str) -> float:
    """Normalize a metric value to [0, 1]."""
    if scale == "1-5":
        return (val - 1.0) / 4.0
    return val


def shorten_lora(name: str) -> str:
    """Create a short display label from a LoRA name."""
    name = name.replace("checkpoints_latentqa_cls_", "").replace("_Qwen3-8B", "")
    return name


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p", "d"]
LINE_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9A6324",
    "#800000",
    "#aaffc3",
    "#808000",
]


def plot_summary_lines(
    all_metrics: dict[str, dict[str, dict[str, float]]],
    labels: dict[str, str] | None = None,
    output_path: str = "experiments/eval_comparison.png",
):
    """Line graph with all metrics on a 0-1 scale.

    Each LoRA is a line with a distinct color + marker.
    X-axis = eval metric, Y-axis = score (0-1).
    Metrics on a 1-5 LLM judge scale are mapped to 0-1 via (x-1)/4.
    """

    # Collect all LoRA names across all evals
    all_loras = sorted(set(lora for eval_metrics in all_metrics.values() for lora in eval_metrics))
    if labels is None:
        labels = {l: shorten_lora(l) for l in all_loras}

    # Build data
    x_labels = []
    plot_values: dict[str, list[float | None]] = {l: [] for l in all_loras}

    for eval_pattern, metric_key, bar_label, scale in EVAL_METRICS_TO_PLOT:
        if eval_pattern not in all_metrics:
            continue
        eval_data = all_metrics[eval_pattern]
        has_any = any(metric_key in eval_data.get(l, {}) for l in all_loras)
        if not has_any:
            continue
        x_labels.append(bar_label)
        for lora in all_loras:
            raw = eval_data.get(lora, {}).get(metric_key)
            plot_values[lora].append(normalize_value(raw, scale) if raw is not None else None)

    if not x_labels:
        print("No metrics found to plot.")
        return

    n_points = len(x_labels)
    x = np.arange(n_points)

    fig, ax = plt.subplots(figsize=(max(16, n_points * 1.3), 8))

    for i, lora in enumerate(all_loras):
        color = LINE_COLORS[i % len(LINE_COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        values = plot_values[lora]

        # Only plot non-None points
        xs = [x[j] for j in range(n_points) if values[j] is not None]
        ys = [values[j] for j in range(n_points) if values[j] is not None]

        ax.plot(
            xs, ys, color=color, marker=marker, markersize=7, linewidth=1.5, label=labels.get(lora, lora), alpha=0.85
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9, ha="center")
    ax.set_ylabel("Score (0-1 scale)", fontsize=12)
    ax.set_title("Open-Ended Eval Comparison", fontsize=14)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    ax.grid(axis="both", alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, n_points - 0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

RESULT_DIR_PREFIX = "experiments/eval_results_"

# Map from result dir suffix -> short label for the legend.
# Add new LoRAs here as you run evals.
ALL_LORAS = {
    "on_policy_Qwen3-8B": "ML AO",
    "past_lens_addition_Qwen3-8B": "Original AO",
    "past_lens_only_500k": "Past Lens Only 500k",
    "300k_latentqa_only": "300k LQA Only",
    "300k_full_mix": "300k Full Mix",
    "full_mix_plus_synthetic_qa_v3": "Full Mix + SynQA v3",
    "full_mix_synthetic_qa_v3_replace_lqa": "Full Mix SynQA v3 Replace LQA",
    "full_mix_500k_past_lens_plus_synthetic_qa_v3": "AO v2",
    "full_mix_500k_past_lens_synthetic_qa_v3_replace_lqa": "500k PL SynQA v3 Replace LQA",
    "full_mix_250k_past_lens_plus_synthetic_qa_v3": "250k PL + SynQA v3",
    "full_mix_250k_past_lens_synthetic_qa_v3_replace_lqa": "250k PL SynQA v3 Replace LQA",
    "134k_pl_31k_spqav2_199k_sqav3_126k_cls": "134k PL + SQA Mix",
    "500k_pl_31k_spqav2_199k_sqav3_126k_cls": "500k PL + SQA Mix",
    "codex_old_past_lens_300k_full_mix": "Codex Old 300k Mix",
    "codex_old_past_lens_300k_full_mix_plus_synthetic_qa_v3": "Codex Old 300k + SynQA v3",
    "134k_pl_31k_spqav2_199k_sqav3_126k_cls_100k_chatreg": "134k PL + SQA + ChatReg",
    "134k_pl_31k_spqav2_199k_sqav3_126k_cls_100k_chatreg_2ep": "134k PL + SQA + ChatReg 2ep",
    "500k_pl_31k_spqav2_199k_sqav3_126k_cls_100k_chatreg": "500k PL + SQA + ChatReg",
    "500k_pl_31k_spqav2_199k_sqav3_126k_cls_100k_chatreg_2ep": "500k PL + SQA + ChatReg 2ep",
}

# Named subsets for focused comparisons. Each value is a list of keys from ALL_LORAS.
PRESETS = {
    "all": list(ALL_LORAS.keys()),
    "selected": [
        "on_policy_Qwen3-8B",
        "past_lens_addition_Qwen3-8B",
        "full_mix_500k_past_lens_plus_synthetic_qa_v3",
        "134k_pl_31k_spqav2_199k_sqav3_126k_cls_100k_chatreg",
        "134k_pl_31k_spqav2_199k_sqav3_126k_cls_100k_chatreg_2ep",
        "500k_pl_31k_spqav2_199k_sqav3_126k_cls_100k_chatreg",
        "500k_pl_31k_spqav2_199k_sqav3_126k_cls_100k_chatreg_2ep",
    ],
    "selected_2": [
        "on_policy_Qwen3-8B",
        "past_lens_addition_Qwen3-8B",
        "full_mix_500k_past_lens_plus_synthetic_qa_v3",
        "134k_pl_31k_spqav2_199k_sqav3_126k_cls",
        "500k_pl_31k_spqav2_199k_sqav3_126k_cls",
        "codex_old_past_lens_300k_full_mix",
        "codex_old_past_lens_300k_full_mix_plus_synthetic_qa_v3",
    ],
}


def resolve_preset(preset_name: str) -> tuple[list[str], dict[str, str]]:
    """Return (result_dirs, label_map) for a named preset."""
    keys = PRESETS[preset_name]
    result_dirs = [RESULT_DIR_PREFIX + k for k in keys]
    # Build label map: lora_name (as it appears in the JSON) -> short label
    # We don't know the exact lora_name yet, so we return dir->label and
    # let the caller resolve after loading.
    dir_labels = {k: ALL_LORAS[k] for k in keys}
    return result_dirs, dir_labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_label_map_from_dir_labels(
    all_metrics: dict[str, dict[str, dict[str, float]]],
    result_dirs: list[str],
    dir_labels: dict[str, str],
) -> dict[str, str]:
    """Build lora_name -> label map by matching which LoRA came from which dir."""
    # Each result dir has exactly one LoRA. Map dir suffix -> lora name.
    label_map = {}
    for result_dir in result_dirs:
        suffix = result_dir.removeprefix(RESULT_DIR_PREFIX)
        if suffix not in dir_labels:
            continue
        # Find which lora_name appears in results from this dir
        eval_jsons = find_eval_jsons(result_dir)
        for json_files in eval_jsons.values():
            for json_file in json_files:
                with open(json_file) as f:
                    data = json.load(f)
                lora_name = extract_lora_name(data)
                if lora_name not in label_map:
                    label_map[lora_name] = dir_labels[suffix]
                break
            break
    return label_map


def print_metrics_summary(all_metrics: dict[str, dict[str, dict[str, float]]]) -> None:
    print(f"{'=' * 60}")
    print("LOADED METRICS")
    print(f"{'=' * 60}")
    for eval_name, lora_metrics in sorted(all_metrics.items()):
        print(f"\n{eval_name}:")
        for lora, metrics in sorted(lora_metrics.items()):
            key_metrics = {
                k: f"{v:.4f}"
                for k, v in metrics.items()
                if isinstance(v, float)
                and not k.startswith("prompt_")
                and not k.startswith("cat_")
                and "total" not in k
                and "num_" not in k
            }
            print(f"  {shorten_lora(lora)}: {key_metrics}")


def main():
    parser = argparse.ArgumentParser(description="Plot eval comparison from individual JSON files")
    parser.add_argument(
        "--result-dirs", nargs="+", default=None, help="Directories with eval results (e.g. experiments/eval_results_*)"
    )
    parser.add_argument(
        "--preset", choices=list(PRESETS.keys()), default=None, help="Use a named preset instead of --result-dirs"
    )
    parser.add_argument("--labels", nargs="+", default=None, help="Custom labels for each LoRA (in order they appear)")
    parser.add_argument(
        "-o", "--output", default=None, help="Output path (defaults to experiments/eval_comparison_{preset}.png)"
    )
    parser.add_argument("--all-presets", action="store_true", help="Generate plots for all presets")
    args = parser.parse_args()

    if args.all_presets:
        for preset_name in PRESETS:
            result_dirs, dir_labels = resolve_preset(preset_name)
            # Skip dirs that don't exist yet
            result_dirs = [d for d in result_dirs if Path(d).exists()]
            if not result_dirs:
                print(f"Skipping preset '{preset_name}': no result dirs found")
                continue
            all_metrics = load_all_metrics(result_dirs)
            label_map = build_label_map_from_dir_labels(all_metrics, result_dirs, dir_labels)
            output = f"experiments/eval_comparison_{preset_name}.png"
            plot_summary_lines(all_metrics, labels=label_map, output_path=output)
        return

    if args.preset:
        result_dirs, dir_labels = resolve_preset(args.preset)
        result_dirs = [d for d in result_dirs if Path(d).exists()]
        output = args.output or f"experiments/eval_comparison_{args.preset}.png"
    elif args.result_dirs:
        result_dirs = args.result_dirs
        dir_labels = {}
        output = args.output or "experiments/eval_comparison.png"
    else:
        parser.error("Provide --result-dirs, --preset, or --all-presets")
        return

    all_metrics = load_all_metrics(result_dirs)
    print_metrics_summary(all_metrics)

    # Build label map
    if args.labels:
        all_loras = sorted(set(lora for eval_metrics in all_metrics.values() for lora in eval_metrics))
        if len(args.labels) == len(all_loras):
            label_map = dict(zip(all_loras, args.labels))
        else:
            label_map = None
    elif dir_labels:
        label_map = build_label_map_from_dir_labels(all_metrics, result_dirs, dir_labels)
    else:
        label_map = None

    plot_summary_lines(all_metrics, labels=label_map, output_path=output)


if __name__ == "__main__":
    main()
