import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Configuration
OUTPUT_JSON_DIR = "taboo_eval_results"
OUTPUT_JSON_DIR = "taboo_eval_results_open_ended_Qwen3-8B"
OUTPUT_JSON_DIR = "taboo_eval_results_yes_no_Qwen3-8B"
OUTPUT_JSON_DIR = "experiments/taboo_eval_results/Qwen3-8B_open_ended_direct"
# OUTPUT_JSON_DIR = "experiments/taboo_eval_results/Qwen3-8B_open_ended_direct_50_mix"
OUTPUT_JSON_DIR = "experiments/taboo_eval_results/Qwen3-8B_open_ended_all_direct_test"
# OUTPUT_JSON_DIR = "experiments/taboo_eval_results/Qwen3-32B_open_ended_direct"
# OUTPUT_JSON_DIR = "experiments/taboo_eval_results/Qwen3-32B_yes_no_direct"
# OUTPUT_JSON_DIR = "experiments/taboo_eval_results/gemma-2-9b-it_open_ended_all_direct"
# OUTPUT_JSON_DIR = "experiments/taboo_eval_results/Qwen3-8B_open_ended_all_direct"

DATA_DIR = OUTPUT_JSON_DIR.split("/")[-1]

IMAGE_FOLDER = "images"
CLS_IMAGE_FOLDER = f"{IMAGE_FOLDER}/taboo"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(CLS_IMAGE_FOLDER, exist_ok=True)


SEQUENCE = False
# SEQUENCE = True

sequence_str = "sequence" if SEQUENCE else "token"

if "Qwen3-8B" in DATA_DIR:
    model_name = "Qwen3-8B"
elif "Qwen3-32B" in DATA_DIR:
    model_name = "Qwen3-32B"
elif "gemma-2-9b-it" in DATA_DIR:
    model_name = "Gemma-2-9B-IT"

if "open_ended" in DATA_DIR:
    task_type = "Open Ended"
elif "yes_no" in DATA_DIR:
    task_type = "Yes / No"

if "50_mix" in DATA_DIR:
    misc = " (50% Mix of Taboo and Chat Data)"
else:
    misc = ""

TITLE = f"Taboo Results{misc}: {task_type} Response with {sequence_str.capitalize()}-Level Inputs for {model_name}"


OUTPUT_PATH = f"{CLS_IMAGE_FOLDER}/taboo_results_{DATA_DIR}_{sequence_str}.png"


# Filter filenames - skip files containing any of these strings
FILTER_FILENAMES = []

# Define your custom labels here (fill in the empty strings with your labels)
CUSTOM_LABELS = {
    # gemma 2 9b
    "checkpoints_cls_latentqa_only_addition_gemma-2-9b-it": "LatentQA + Classification",
    "checkpoints_latentqa_only_addition_gemma-2-9b-it": "LatentQA",
    "checkpoints_cls_only_addition_gemma-2-9b-it": "Classification",
    "checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it": "Past Lens + LatentQA + Classification",
    # qwen3 8b
    "checkpoints_cls_latentqa_only_addition_Qwen3-8B": "LatentQA + Classification",
    "checkpoints_latentqa_only_addition_Qwen3-8B": "LatentQA",
    "checkpoints_cls_only_addition_Qwen3-8B": "Classification",
    "checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B": "Past Lens + Classification + LatentQA",
    "checkpoints_cls_latentqa_sae_addition_Qwen3-8B": "SAE + Classification + LatentQA",
}


def calculate_accuracy(record: dict, investigator_lora: str) -> float:
    if "gemma" in investigator_lora:
        idx = -3
    elif "Qwen3" in investigator_lora:
        idx = -7
    else:
        raise ValueError(f"Unknown model in investigator_lora: {investigator_lora}")

    if SEQUENCE:
        ground_truth = record["ground_truth"].lower()
        full_seq_responses = record["full_sequence_responses"]
        # full_seq_responses = record["segment_responses"]

        num_correct = sum(1 for resp in full_seq_responses if ground_truth in resp.lower())
        total = len(full_seq_responses)

        return num_correct / total if total > 0 else 0
    else:
        ground_truth = record["ground_truth"].lower()
        responses = record["token_responses"][idx : idx + 1]
        # responses = record["token_responses"][-9:-6]

        num_correct = sum(1 for resp in responses if ground_truth in resp.lower())
        total = len(responses)

        return num_correct / total if total > 0 else 0


def load_results(json_dir: str, required_verbalizer_prompt: str | None = None):
    """Load all JSON files from the directory."""
    results_by_lora = defaultdict(list)
    results_by_lora_word = defaultdict(lambda: defaultdict(list))

    json_dir = Path(json_dir)
    if not json_dir.exists():
        print(f"Directory {json_dir} does not exist!")
        return results_by_lora, results_by_lora_word

    json_files = list(json_dir.glob("*.json"))

    # Apply filename filter
    if FILTER_FILENAMES:
        filtered_files = []
        for json_file in json_files:
            if not any(filter_str in json_file.name for filter_str in FILTER_FILENAMES):
                filtered_files.append(json_file)
            else:
                print(f"Skipping filtered file: {json_file.name}")
        json_files = filtered_files

    print(f"Found {len(json_files)} JSON files (after filtering)")

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        investigator_lora = data["verbalizer_lora_path"]

        # Calculate accuracy for each record
        for record in data["results"]:
            if required_verbalizer_prompt and record["verbalizer_prompt"] != required_verbalizer_prompt:
                continue
            accuracy = calculate_accuracy(record, investigator_lora)
            word = record["verbalizer_prompt"]

            results_by_lora[investigator_lora].append(accuracy)
            results_by_lora_word[investigator_lora][word].append(accuracy)

    return results_by_lora, results_by_lora_word


def calculate_confidence_interval(accuracies, confidence=0.95):
    """Calculate 95% confidence interval for accuracy data."""
    n = len(accuracies)
    if n == 0:
        return 0, 0

    mean = np.mean(accuracies)
    std_err = np.std(accuracies, ddof=1) / np.sqrt(n)

    # For 95% CI, use z-score of 1.96
    margin = 1.96 * std_err

    return margin


def plot_results(results_by_lora, highlight_keyword, highlight_color="#FDB813", highlight_hatch="////"):
    """Create a bar chart of average accuracy by investigator LoRA, highlighting exactly one LoRA."""
    if not results_by_lora:
        print("No results to plot!")
        return

    # Calculate mean accuracy and confidence intervals for each LoRA
    lora_names = []
    mean_accuracies = []
    error_bars = []

    for lora_path, accuracies in results_by_lora.items():
        lora_name = lora_path.split("/")[-1]
        lora_names.append(lora_name)
        mean_acc = sum(accuracies) / len(accuracies)
        mean_accuracies.append(mean_acc)
        ci_margin = calculate_confidence_interval(accuracies)
        error_bars.append(ci_margin)
        print(f"{lora_name}: {mean_acc:.3f} ± {ci_margin:.3f} (n={len(accuracies)} records)")

    # Try to find a match and move it to index 0, otherwise just use the first one
    matches = [i for i, name in enumerate(lora_names) if highlight_keyword in name]
    if len(matches) == 0:
        print(f"Warning: Keyword '{highlight_keyword}' matched 0 LoRA names. Using first LoRA.")
        m = 0
    elif len(matches) > 1:
        print(
            f"Warning: Keyword '{highlight_keyword}' matched {len(matches)} LoRA names: {[lora_names[i] for i in matches]}. Using the first match."
        )
        m = matches[0]
    else:
        m = matches[0]
    order = [m] + [i for i in range(len(lora_names)) if i != m]
    lora_names = [lora_names[i] for i in order]
    mean_accuracies = [mean_accuracies[i] for i in order]
    error_bars = [error_bars[i] for i in order]

    # Print dictionary template for labels (for LoRA entries)
    print("\n" + "=" * 60)
    print("Copy this dictionary and fill in your custom labels:")
    print("=" * 60)
    print("CUSTOM_LABELS = {")
    for name in lora_names:
        print(f'    "{name}": "",')
    print("}")
    print("=" * 60 + "\n")

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = list(plt.cm.tab10(np.linspace(0, 1, len(lora_names))))
    colors[0] = highlight_color  # highlighted bar color
    bars = ax.bar(
        range(len(lora_names)), mean_accuracies, color=colors, yerr=error_bars, capsize=5, error_kw={"linewidth": 2}
    )

    # Distinctive styling for the highlighted bar
    bars[0].set_hatch(highlight_hatch)
    bars[0].set_edgecolor("black")
    bars[0].set_linewidth(2.0)

    ax.set_xlabel("Investigator LoRA", fontsize=12)
    ax.set_ylabel("Average Accuracy", fontsize=12)
    ax.set_title(TITLE, fontsize=14)
    ax.set_xticks(range(len(lora_names)))
    ax.set_xticklabels([])  # use legend instead
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    # Value labels on bars
    for bar, acc, err in zip(bars, mean_accuracies, error_bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + err + 0.02,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Legend uses CUSTOM_LABELS when available
    legend_labels = []
    for name in lora_names:
        if CUSTOM_LABELS and name in CUSTOM_LABELS and CUSTOM_LABELS[name]:
            legend_labels.append(CUSTOM_LABELS[name])
        else:
            legend_labels.append(name)

    ax.legend(bars, legend_labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=10, ncol=2, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved as '{OUTPUT_PATH}'")
    # # plt.show()


def plot_by_keyword_with_extras(
    results_by_lora, required_keyword, extra_bars, output_path=None, highlight_color="#FDB813", highlight_hatch="////"
):
    """
    Plot a LoRA (selected by required_keyword in its name) plus extra bars.
    If no LoRA matches the keyword, only the extra bars are plotted.
    If multiple LoRAs match, the first one is used.
    """
    entries = []
    for lora_path, accuracies in results_by_lora.items():
        lora_name = lora_path.split("/")[-1]
        entries.append((lora_name, accuracies))

    matches = [(name, accs) for name, accs in entries if required_keyword in name]

    if len(matches) == 0:
        print(f"Warning: Keyword '{required_keyword}' matched 0 LoRA names. Plotting only extra bars.")
        selected_name = None
        mean_acc = None
        ci = None
    elif len(matches) > 1:
        print(
            f"Warning: Keyword '{required_keyword}' matched {len(matches)} LoRA names: {[m[0] for m in matches]}. Using the first one."
        )
        selected_name, selected_accs = matches[0]
        mean_acc = sum(selected_accs) / len(selected_accs)
        ci = calculate_confidence_interval(selected_accs)
        print(f"Selected LoRA: {selected_name} -> {mean_acc:.3f} ± {ci:.3f} (n={len(selected_accs)})")
    else:
        selected_name, selected_accs = matches[0]
        mean_acc = sum(selected_accs) / len(selected_accs)
        ci = calculate_confidence_interval(selected_accs)
        print(f"Selected LoRA: {selected_name} -> {mean_acc:.3f} ± {ci:.3f} (n={len(selected_accs)})")

    assert isinstance(extra_bars, list) and len(extra_bars) > 0, "extra_bars must be a non-empty list"
    for b in extra_bars:
        assert "label" in b and "value" in b and "error" in b, f"extra_bars entries must have label, value, error: {b}"

    if selected_name is not None:
        labels = [selected_name] + [b["label"] for b in extra_bars]
        values = [mean_acc] + [b["value"] for b in extra_bars]
        errors = [ci] + [b["error"] for b in extra_bars]
    else:
        labels = [b["label"] for b in extra_bars]
        values = [b["value"] for b in extra_bars]
        errors = [b["error"] for b in extra_bars]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = list(plt.cm.tab10(np.linspace(0, 1, len(labels))))
    if selected_name is not None:
        colors[0] = highlight_color
    bars = ax.bar(range(len(labels)), values, color=colors, yerr=errors, capsize=5, error_kw={"linewidth": 2})

    # Distinctive styling for the highlighted bar (if there is one)
    if selected_name is not None:
        bars[0].set_hatch(highlight_hatch)
        bars[0].set_edgecolor("black")
        bars[0].set_linewidth(2.0)

    ax.set_xlabel("Selected LoRA + Extras", fontsize=12)
    ax.set_ylabel("Average Accuracy", fontsize=12)
    ax.set_title(TITLE + " (Selected + Extras)", fontsize=14)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([])  # legend carries names
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    for bar, acc, err in zip(bars, values, errors):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + err + 0.02,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    legend_labels = []
    if selected_name is not None:
        if CUSTOM_LABELS and selected_name in CUSTOM_LABELS and CUSTOM_LABELS[selected_name]:
            legend_labels.append(CUSTOM_LABELS[selected_name])
        else:
            legend_labels.append(selected_name)
    legend_labels.extend([b["label"] for b in extra_bars])

    ax.legend(bars, legend_labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=10, ncol=2, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    path = (
        OUTPUT_PATH.replace(".png", f"_{required_keyword}_selected_with_extras.png")
        if output_path is None
        else output_path
    )
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved as '{path}'")
    # plt.show()


def plot_per_word_accuracy(results_by_lora_word):
    """Create separate plots for each investigator showing per-word accuracy."""
    if not results_by_lora_word:
        print("No per-word results to plot!")
        return

    for lora_path, word_accuracies in results_by_lora_word.items():
        lora_name = lora_path.split("/")[-1]

        # Calculate mean accuracy and CI per word
        words = sorted(word_accuracies.keys())
        mean_accs = [sum(word_accuracies[w]) / len(word_accuracies[w]) for w in words]
        error_bars = [calculate_confidence_interval(word_accuracies[w]) for w in words]

        for w, accs in word_accuracies.items():
            mean_acc = sum(accs) / len(accs)
            ci = calculate_confidence_interval(accs)
            print(f"{lora_name} - Word '{w}': {mean_acc:.3f} ± {ci:.3f} (n={len(accs)})")

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = plt.cm.tab20(np.linspace(0, 1, len(words)))
        bars = ax.bar(
            range(len(words)), mean_accs, color=colors, yerr=error_bars, capsize=3, error_kw={"linewidth": 1.5}
        )

        ax.set_xlabel("Word", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"Per-Word Accuracy: {lora_name}", fontsize=14)
        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

        # Add horizontal line for overall mean
        overall_mean = sum(mean_accs) / len(mean_accs)
        ax.axhline(y=overall_mean, color="red", linestyle="--", label=f"Overall mean: {overall_mean:.3f}", linewidth=2)
        ax.legend()

        plt.tight_layout()
        safe_lora_name = lora_name.replace("/", "_").replace(" ", "_")
        filename = f"per_word_{safe_lora_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved per-word plot: {filename}")
        plt.close()


def main():
    extra_bars = [
        {"label": "Best Interp Method (SAEs)", "value": 0.0413, "error": 0.0038},
        {"label": "Best Black Box Method (Prefill)", "value": 0.0717, "error": 0.0055},
    ]

    # Load results from all JSON files (no prompt filter - use all results)
    results_by_lora, results_by_lora_word = load_results(OUTPUT_JSON_DIR, required_verbalizer_prompt=None)

    # Plot 1: Overall accuracy by investigator
    # Use the first available LoRA name as the keyword, or just use a generic keyword
    if results_by_lora:
        # Get the first LoRA name and use part of it as keyword
        first_lora = list(results_by_lora.keys())[0]
        first_lora_name = first_lora.split("/")[-1]
        # Try to find a reasonable keyword - use "latentqa" if present, otherwise use first part of name
        if "latentqa" in first_lora_name.lower():
            keyword = "latentqa"
        else:
            # Just use the first word or a reasonable substring
            keyword = first_lora_name.split("_")[0] if "_" in first_lora_name else first_lora_name[:10]
        print(f"Using keyword '{keyword}' for highlighting (from LoRA: {first_lora_name})")
    else:
        keyword = "latentqa_cls_past_lens"  # fallback

    plot_results(results_by_lora, highlight_keyword=keyword)
    plot_by_keyword_with_extras(results_by_lora, required_keyword=keyword, extra_bars=extra_bars)

    # Plot 2: Per-word accuracy for each investigator
    # plot_per_word_accuracy(results_by_lora_word)


if __name__ == "__main__":
    main()
