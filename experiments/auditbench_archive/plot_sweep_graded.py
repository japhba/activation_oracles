"""Plot overnight sweep v2 results: 4 AO checkpoints × 4 conditions."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "overnight_sweep_v2"

# Checkpoint mapping: internal ID → (display label, color)
# Colors from docs/plotting_guidelines.md
AO_DISPLAY = {
    "hf_past_lens": ("Original AO", "#f58231"),
    "mlao": ("MLAO", "#e6194b"),
    "local_original": ("AO v2", "#3cb44b"),
    "local_hb": ("AO v2 + HB", "#4363d8"),
}

AO_ORDER = ["hf_past_lens", "mlao", "local_original", "local_hb"]

# 4 conditions to plot
CONDITIONS = [
    {"target": "transcripts", "position": "pre_answer", "title": "Transcripts + KTO Adv. Train\nAO input: prompt only"},
    {"target": "transcripts", "position": "full_seq", "title": "Transcripts + KTO Adv. Train\nAO input: prompt + response"},
    {"target": "synth_docs", "position": "pre_answer", "title": "Synth-Docs + SFT Adv. Train\nAO input: prompt only"},
    {"target": "synth_docs", "position": "full_seq", "title": "Synth-Docs + SFT Adv. Train\nAO input: prompt + response"},
]


def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=42):
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


def _plot_bars(fig, axes, metric_fn, metric_label, suptitle):
    bar_width = 0.7
    x_positions = np.arange(len(AO_ORDER))

    axes_flat = axes.ravel()
    legend_handles = None

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

            signals = [metric_fn(d) for d in result["detailed_results"]]
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
        ax.set_ylabel(metric_label if ax_idx % 2 == 0 else "")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)

        for bar, mean in zip(bars, means):
            if mean > 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, mean + 0.03,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=9,
                )

        if ax_idx == 0:
            legend_handles = bars

    axes_flat[0].set_ylim(-0.05, max(1.7, axes_flat[0].get_ylim()[1]))

    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)

    ao_labels = [AO_DISPLAY[ao_id][0] for ao_id in AO_ORDER]
    fig.legend(legend_handles, ao_labels, loc="lower center",
               ncol=len(AO_ORDER), fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])


def main():
    # --- Correctness bar chart ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    _plot_bars(
        fig, axes,
        metric_fn=lambda d: d["correctness"] - 1.0,
        metric_label="Correctness (0-4)",
        suptitle="AuditBench: Correctness by AO Checkpoint",
    )
    out = OUTPUT_DIR / "auditbench_correctness.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

    # --- Combined (correctness + specificity) / 2 bar chart ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    _plot_bars(
        fig, axes,
        metric_fn=lambda d: ((d["correctness"] + d["specificity"]) / 2.0) - 1.0,
        metric_label="Avg(Correctness, Specificity) (0-4)",
        suptitle="AuditBench: Avg(Correctness, Specificity) by AO Checkpoint",
    )
    out = OUTPUT_DIR / "auditbench_combined_score.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

    # --- Per-behavior heatmap ---
    _plot_per_behavior()


def _plot_per_behavior():
    behaviors = sorted({
        d["behavior_name"]
        for ao_id in AO_ORDER
        for cond in CONDITIONS
        if (r := load_experiment(ao_id, cond["target"], cond["position"])) is not None
        for d in r["detailed_results"]
    })

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
    axes_flat = axes.ravel()

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

        for i in range(len(behaviors)):
            for j in range(len(AO_ORDER)):
                val = matrix[i, j]
                color = "white" if val > 1.5 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=9, color=color)

    fig.suptitle("Per-Behavior Correctness by AO Checkpoint",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.10, wspace=0.45, hspace=0.35)
    cbar_ax = fig.add_axes([0.25, 0.03, 0.50, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="Correctness (0-4)")

    out = OUTPUT_DIR / "per_behavior_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
