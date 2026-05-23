"""
Visualize per-token probability distributions for the number prediction dataset.

Produces:
1. Per-problem heatmap: each row is a problem, columns are output token positions,
   color = probability of the chosen token
2. Bar chart of top-5 token distribution for the hardest problems (lowest min-token prob)
3. Entropy distribution histogram

Usage:
    .venv/bin/python data_pipelines/number_prediction/plot_token_probs.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DATASET_PATH = Path("data_pipelines/number_prediction/Qwen3-8B/number_prediction_eval_dataset.json")
OUTPUT_DIR = Path("data_pipelines/number_prediction/plots")


def main():
    data = json.loads(DATASET_PATH.read_text())
    entries = data["entries"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: Heatmap of chosen-token probability per position ──
    # Only look at the answer tokens (exclude trailing <|im_end|> etc)
    # Find max answer length across entries
    max_answer_tokens = max(e["model_answer_num_tokens"] for e in entries if e["model_answer"] is not None)

    fig, ax = plt.subplots(figsize=(10, 16))

    # Sort entries by category then by expression for visual grouping
    cat_order = ["simple_2op", "large_numbers", "divmod", "medium_3_4op", "nested"]
    sorted_entries = sorted(entries, key=lambda e: (cat_order.index(e["category"]), e["id"]))

    heatmap_data = []
    labels = []
    for e in sorted_entries:
        if e["model_answer"] is None:
            continue
        n_answer_tokens = e["model_answer_num_tokens"]
        row = []
        for i in range(max_answer_tokens):
            if i < n_answer_tokens and i < len(e["token_logprobs"]):
                row.append(e["token_logprobs"][i]["chosen_prob"])
            else:
                row.append(np.nan)
        heatmap_data.append(row)
        correct_marker = "✓" if e["model_correct"] else "✗"
        labels.append(f"{correct_marker} {e['expression'][:30]:<30} → {e['model_answer']}")

    heatmap_arr = np.array(heatmap_data)

    # Use a diverging colormap where 1.0 = green, low = red
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="white")

    im = ax.imshow(heatmap_arr, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6, fontfamily="monospace")
    ax.set_xticks(range(max_answer_tokens))
    ax.set_xticklabels([f"tok {i}" for i in range(max_answer_tokens)], fontsize=8)
    ax.set_xlabel("Output token position")
    ax.set_title("Chosen token probability per position (green=1.0, red=0.0)")

    # Add text annotations
    for i in range(len(heatmap_data)):
        for j in range(max_answer_tokens):
            val = heatmap_arr[i, j]
            if not np.isnan(val):
                token_str = sorted_entries[i]["token_logprobs"][j]["chosen_token"]
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{token_str}\n{val:.2f}", ha="center", va="center", fontsize=5, color=color)

    # Draw category separators
    cat_boundaries = []
    prev_cat = None
    for idx, e in enumerate(sorted_entries):
        if e["category"] != prev_cat and prev_cat is not None:
            cat_boundaries.append(idx - 0.5)
        prev_cat = e["category"]

    for boundary in cat_boundaries:
        ax.axhline(y=boundary, color="blue", linewidth=1.5, linestyle="--")

    plt.colorbar(im, ax=ax, shrink=0.5, label="P(chosen token)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "token_prob_heatmap.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'token_prob_heatmap.png'}")
    plt.close()

    # ── Plot 2: Top-5 token distributions for hardest problems ──
    # Pick the 15 problems with lowest minimum chosen-token prob
    problems_with_min_prob = []
    for e in entries:
        if e["model_answer"] is None:
            continue
        n_tok = e["model_answer_num_tokens"]
        answer_probs = [e["token_logprobs"][i]["chosen_prob"] for i in range(min(n_tok, len(e["token_logprobs"])))]
        problems_with_min_prob.append((min(answer_probs), e))

    problems_with_min_prob.sort(key=lambda x: x[0])
    hardest = problems_with_min_prob[:15]

    fig, axes = plt.subplots(5, 3, figsize=(18, 20))
    axes = axes.flatten()

    for ax_idx, (min_p, e) in enumerate(hardest):
        ax = axes[ax_idx]
        n_tok = e["model_answer_num_tokens"]
        n_positions = min(n_tok, len(e["token_logprobs"]))

        # For each token position, show top-5 as stacked bars
        positions = list(range(n_positions))
        bar_width = 0.7
        colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]

        bottom = np.zeros(n_positions)
        for rank in range(5):
            probs = []
            token_labels = []
            for pos in range(n_positions):
                top5 = e["token_logprobs"][pos]["top5"]
                if rank < len(top5):
                    probs.append(top5[rank]["prob"])
                    token_labels.append(top5[rank]["token"])
                else:
                    probs.append(0)
                    token_labels.append("")
            bars = ax.bar(positions, probs, bar_width, bottom=bottom, color=colors[rank], alpha=0.85)

            # Label each bar segment
            for bar, label, p, b in zip(bars, token_labels, probs, bottom):
                if p > 0.05:
                    display = repr(label).strip("'")
                    if len(display) > 6:
                        display = display[:5] + "…"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, b + p / 2,
                        f"{display}\n{p:.2f}", ha="center", va="center", fontsize=6,
                    )
            bottom += probs

        correct_marker = "✓" if e["model_correct"] else "✗"
        ax.set_title(f"{correct_marker} {e['expression'][:35]}\nmodel={e['model_answer']}  true={e['true_answer']}", fontsize=8)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"pos {i}" for i in positions], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Probability", fontsize=7)

    legend_patches = [mpatches.Patch(color=colors[i], label=f"Rank {i+1}") for i in range(5)]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=9)
    fig.suptitle("Top-5 token distributions for 15 hardest problems (by lowest token prob)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hardest_problems_top5.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'hardest_problems_top5.png'}")
    plt.close()

    # ── Plot 3: Entropy distribution by category ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram of first-token entropy
    first_tok_entropy = []
    cats = []
    for e in entries:
        if e["token_logprobs"]:
            first_tok_entropy.append(e["token_logprobs"][0]["entropy_bits"])
            cats.append(e["category"])

    cat_colors = {"simple_2op": "#2ecc71", "large_numbers": "#3498db", "divmod": "#f39c12", "medium_3_4op": "#e74c3c", "nested": "#9b59b6"}

    for cat in cat_order:
        cat_vals = [h for h, c in zip(first_tok_entropy, cats) if c == cat]
        axes[0].hist(cat_vals, bins=15, alpha=0.6, label=cat, color=cat_colors[cat])
    axes[0].set_xlabel("First token entropy (bits)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("First token entropy distribution by category")
    axes[0].legend(fontsize=8)

    # Right: scatter of min-token-prob vs number of answer tokens
    min_probs = []
    n_toks = []
    scatter_cats = []
    for e in entries:
        if e["model_answer"] is None:
            continue
        n_tok = e["model_answer_num_tokens"]
        answer_probs = [e["token_logprobs"][i]["chosen_prob"] for i in range(min(n_tok, len(e["token_logprobs"])))]
        min_probs.append(min(answer_probs))
        n_toks.append(n_tok)
        scatter_cats.append(e["category"])

    for cat in cat_order:
        idxs = [i for i, c in enumerate(scatter_cats) if c == cat]
        axes[1].scatter(
            [n_toks[i] for i in idxs], [min_probs[i] for i in idxs],
            label=cat, color=cat_colors[cat], alpha=0.7, s=50,
        )
    axes[1].set_xlabel("Number of answer tokens")
    axes[1].set_ylabel("Min chosen-token probability")
    axes[1].set_title("Answer length vs model confidence")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "entropy_overview.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'entropy_overview.png'}")
    plt.close()


if __name__ == "__main__":
    main()
