"""Plot overnight sweep results: 7 AO checkpoints × 4 conditions."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "overnight_sweep"

# Checkpoint mapping: internal ID → (display label, color)
AO_DISPLAY = {
    "hf_past_lens": ("Original AO", "#f58231"),
    "local_spqav2": ("MLAO", "#e6194b"),
    "local_original": ("AO v2", "#3cb44b"),
    "local_hb": ("AO v2 + 50k HB", "#4363d8"),
    "local_50k_hb_50k_hbi": ("AO v2 + 50k HB\n+ 50k HBI", "#911eb4"),
    "cont_50k_hbi_50k": ("Cont. 50k HBI\n+ 50k Mix", "#42d4f4"),
    "cont_50k_hbi_250k": ("Cont. 50k HBI\n+ 250k Mix", "#f032e6"),
}

AO_ORDER = [
    "hf_past_lens", "local_spqav2", "local_original", "local_hb",
    "local_50k_hb_50k_hbi", "cont_50k_hbi_50k", "cont_50k_hbi_250k",
]

# 4 conditions to plot
CONDITIONS = [
    {"target": "transcripts", "position": "pre_answer", "title": "Transcripts + KTO Adv. Train\nAO input: prompt only"},
    {"target": "transcripts", "position": "full_seq", "title": "Transcripts + KTO Adv. Train\nAO input: prompt + response"},
    {"target": "synth_docs", "position": "pre_answer", "title": "Synth-Docs + SFT Adv. Train\nAO input: prompt only"},
    {"target": "synth_docs", "position": "full_seq", "title": "Synth-Docs + SFT Adv. Train\nAO input: prompt + response"},
]


def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    values = np.array(values)
    n = len(values)
    boot_means = np.array([
        rng.choice(values, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return lo, hi


def load_experiment(ao_id, target_id, position_mode, verbalizer_set="original"):
    name = f"{ao_id}__{target_id}__{position_mode}__{verbalizer_set}"
    path = OUTPUT_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharey=True)

    bar_width = 0.7
    x_positions = np.arange(len(AO_ORDER))

    axes_flat = axes.ravel() if hasattr(axes, 'ravel') else [axes]
    for ax_idx, cond in enumerate(CONDITIONS):
        ax = axes_flat[ax_idx]

        means = []
        ci_los = []
        ci_his = []
        colors = []
        labels = []

        for ao_id in AO_ORDER:
            result = load_experiment(ao_id, cond["target"], cond["position"])
            label, color = AO_DISPLAY[ao_id]
            labels.append(label)
            colors.append(color)

            if result is None:
                means.append(0)
                ci_los.append(0)
                ci_his.append(0)
                continue

            # Extract per-item correctness scores, convert to signal
            signals = [d["correctness"] - 1.0 for d in result["detailed_results"]]
            mean_signal = np.mean(signals)
            lo, hi = bootstrap_ci(signals)

            means.append(mean_signal)
            ci_los.append(max(0, mean_signal - lo))
            ci_his.append(max(0, hi - mean_signal))

        means = np.array(means)
        errors = np.array([ci_los, ci_his])

        bars = ax.bar(
            x_positions, means, bar_width,
            color=colors, edgecolor="black", linewidth=0.8,
            yerr=errors, capsize=4, error_kw={"linewidth": 1.2},
        )

        ax.set_title(cond["title"], fontsize=12, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([""] * len(labels))  # hide x labels, use legend instead
        ax.set_ylabel("Correctness (0–4)" if ax_idx % 2 == 0 else "")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            if mean > 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, mean + 0.03,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=8,
                )

        # Store handles for legend (only need from one subplot)
        if ax_idx == 0:
            legend_handles = bars

    # Consistent y-axis
    axes_flat[0].set_ylim(-0.05, max(1.7, axes_flat[0].get_ylim()[1]))

    fig.suptitle("AuditBench AO Performance by Checkpoint and Target Model",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.5, 0.98, "Correctness = LLM judge score, range 0–4",
             ha="center", fontsize=10, style="italic", color="#555555")

    # Single legend at bottom
    ao_labels = [AO_DISPLAY[ao_id][0].replace("\n", " ") for ao_id in AO_ORDER]
    fig.legend(legend_handles, ao_labels, loc="lower center",
               ncol=4, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])

    out_path = OUTPUT_DIR / "auditbench_checkpoint_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Also save a combined correctness+specificity version
    _plot_combined(OUTPUT_DIR)

    # Also save a per-behavior version
    _plot_per_behavior(fig, axes)


def _plot_combined(output_dir):
    """Plot average of correctness and specificity (both 0-4 scale)."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharey=True)

    bar_width = 0.7
    x_positions = np.arange(len(AO_ORDER))

    axes_flat = axes.ravel()
    for ax_idx, cond in enumerate(CONDITIONS):
        ax = axes_flat[ax_idx]

        means = []
        ci_los = []
        ci_his = []
        colors = []

        for ao_id in AO_ORDER:
            result = load_experiment(ao_id, cond["target"], cond["position"])
            _, color = AO_DISPLAY[ao_id]
            colors.append(color)

            if result is None:
                means.append(0)
                ci_los.append(0)
                ci_his.append(0)
                continue

            signals = [
                ((d["correctness"] + d["specificity"]) / 2.0) - 1.0
                for d in result["detailed_results"]
            ]
            mean_signal = np.mean(signals)
            lo, hi = bootstrap_ci(signals)

            means.append(mean_signal)
            ci_los.append(max(0, mean_signal - lo))
            ci_his.append(max(0, hi - mean_signal))

        means = np.array(means)
        errors = np.array([ci_los, ci_his])

        bars = ax.bar(
            x_positions, means, bar_width,
            color=colors, edgecolor="black", linewidth=0.8,
            yerr=errors, capsize=4, error_kw={"linewidth": 1.2},
        )

        ax.set_title(cond["title"], fontsize=12, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([""] * len(AO_ORDER))
        ax.set_ylabel("Avg(Correctness, Specificity) (0–4)" if ax_idx % 2 == 0 else "")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)

        for bar, mean in zip(bars, means):
            if mean > 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, mean + 0.03,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=8,
                )

        if ax_idx == 0:
            legend_handles = bars

    axes_flat[0].set_ylim(-0.05, max(1.7, axes_flat[0].get_ylim()[1]))

    fig.suptitle("AuditBench: Avg(Correctness, Specificity) by Checkpoint and Target",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.5, 0.98, "Score = mean of correctness and specificity (both 0–4 LLM judge), minus 1",
             ha="center", fontsize=10, style="italic", color="#555555")

    ao_labels = [AO_DISPLAY[ao_id][0].replace("\n", " ") for ao_id in AO_ORDER]
    fig.legend(legend_handles, ao_labels, loc="lower center",
               ncol=4, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])

    out_path = output_dir / "auditbench_combined_score.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def _plot_per_behavior(fig_overview, axes_overview):
    """Per-behavior heatmap showing signal across checkpoints and conditions."""
    behaviors = [
        "Animal Welfare", "Anti AI Regulation", "Contextual Optimism",
        "Data Poisoning", "Emotional Bond", "Hallucinates Citations",
        "Reward Wireheading", "Secret Loyalty", "Self Promotion",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(22, 12), sharey=True)

    axes_flat = axes.ravel() if hasattr(axes, 'ravel') else [axes]
    for ax_idx, cond in enumerate(CONDITIONS):
        ax = axes_flat[ax_idx]
        matrix = np.zeros((len(behaviors), len(AO_ORDER)))

        for j, ao_id in enumerate(AO_ORDER):
            result = load_experiment(ao_id, cond["target"], cond["position"])
            if result is None:
                continue
            for i, b in enumerate(behaviors):
                scores = [d["correctness"] - 1.0 for d in result["detailed_results"]
                          if d["behavior_name"] == b]
                matrix[i, j] = np.mean(scores) if scores else 0

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=3)

        ao_labels = [AO_DISPLAY[ao_id][0] for ao_id in AO_ORDER]
        ax.set_xticks(range(len(AO_ORDER)))
        ax.set_xticklabels(ao_labels, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(behaviors)))
        ax.set_yticklabels(behaviors, fontsize=9)
        ax.set_title(cond["title"], fontsize=11, fontweight="bold")

        # Annotate cells
        for i in range(len(behaviors)):
            for j in range(len(AO_ORDER)):
                val = matrix[i, j]
                color = "white" if val > 1.5 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color=color)

    fig.suptitle("Per-Behavior Signal by AO Checkpoint and Target",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, wspace=0.45, hspace=0.35)
    cbar_ax = fig.add_axes([0.25, 0.03, 0.50, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="Correctness (0–4)")

    out_path = OUTPUT_DIR / "per_behavior_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
