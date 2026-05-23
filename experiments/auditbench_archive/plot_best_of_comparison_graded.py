"""Plot best-of-10 vs greedy (best-of-1) correctness comparison."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

GREEDY_DIR = Path(__file__).resolve().parent / "overnight_sweep_v2"
BEST_OF_DIR = Path(__file__).resolve().parent / "overnight_sweep_v2_best_of_10"

# Colors from docs/plotting_guidelines.md
AO_DISPLAY = {
    "hf_past_lens": ("Original AO", "#f58231"),
    "mlao": ("MLAO", "#e6194b"),
    "local_original": ("AO v2", "#3cb44b"),
    "local_hb": ("AO v2 + HB", "#4363d8"),
}

AO_ORDER = ["hf_past_lens", "mlao", "local_original", "local_hb"]

CONDITIONS = [
    {"target": "transcripts", "position": "pre_answer", "title": "Transcripts + KTO\nprompt only"},
    {"target": "transcripts", "position": "full_seq", "title": "Transcripts + KTO\nprompt + response"},
    {"target": "synth_docs", "position": "pre_answer", "title": "Synth-Docs + SFT\nprompt only"},
    {"target": "synth_docs", "position": "full_seq", "title": "Synth-Docs + SFT\nprompt + response"},
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
    return np.percentile(boot_means, alpha * 100), np.percentile(boot_means, (1 - alpha) * 100)


def load_experiment(output_dir, ao_id, target_id, position_mode, verbalizer_set="original"):
    name = f"{ao_id}__{target_id}__{position_mode}__{verbalizer_set}"
    path = output_dir / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_correctness_signal(result):
    """Extract per-item correctness - 1.0 (so baseline = 0)."""
    if result is None:
        return []
    return [d["correctness"] - 1.0 for d in result["detailed_results"]]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    axes_flat = axes.ravel()

    bar_width = 0.35
    x_positions = np.arange(len(AO_ORDER))

    for ax_idx, cond in enumerate(CONDITIONS):
        ax = axes_flat[ax_idx]

        greedy_means = []
        greedy_ci_los = []
        greedy_ci_his = []
        best_of_means = []
        best_of_ci_los = []
        best_of_ci_his = []
        colors = []

        for ao_id in AO_ORDER:
            _, color = AO_DISPLAY[ao_id]
            colors.append(color)

            # Greedy
            greedy_result = load_experiment(GREEDY_DIR, ao_id, cond["target"], cond["position"])
            greedy_signals = get_correctness_signal(greedy_result)
            if greedy_signals:
                gm = np.mean(greedy_signals)
                glo, ghi = bootstrap_ci(greedy_signals)
                greedy_means.append(gm)
                greedy_ci_los.append(max(0, gm - glo))
                greedy_ci_his.append(max(0, ghi - gm))
            else:
                greedy_means.append(0)
                greedy_ci_los.append(0)
                greedy_ci_his.append(0)

            # Best-of-10
            bo_result = load_experiment(BEST_OF_DIR, ao_id, cond["target"], cond["position"])
            bo_signals = get_correctness_signal(bo_result)
            if bo_signals:
                bm = np.mean(bo_signals)
                blo, bhi = bootstrap_ci(bo_signals)
                best_of_means.append(bm)
                best_of_ci_los.append(max(0, bm - blo))
                best_of_ci_his.append(max(0, bhi - bm))
            else:
                best_of_means.append(0)
                best_of_ci_los.append(0)
                best_of_ci_his.append(0)

        greedy_means = np.array(greedy_means)
        best_of_means = np.array(best_of_means)
        greedy_errors = np.array([greedy_ci_los, greedy_ci_his])
        best_of_errors = np.array([best_of_ci_los, best_of_ci_his])

        # Greedy bars (left)
        bars_g = ax.bar(
            x_positions - bar_width / 2, greedy_means, bar_width,
            color=colors, edgecolor="black", linewidth=0.8, alpha=0.6,
            yerr=greedy_errors, capsize=3, error_kw={"linewidth": 1.0},
            label="Greedy" if ax_idx == 0 else None,
        )

        # Best-of-10 bars (right)
        bars_b = ax.bar(
            x_positions + bar_width / 2, best_of_means, bar_width,
            color=colors, edgecolor="black", linewidth=0.8,
            yerr=best_of_errors, capsize=3, error_kw={"linewidth": 1.0},
            hatch="//",
            label="Best-of-10" if ax_idx == 0 else None,
        )

        ax.set_title(cond["title"], fontsize=11, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([AO_DISPLAY[ao_id][0] for ao_id in AO_ORDER],
                           fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Correctness (0-4)" if ax_idx % 2 == 0 else "")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)

        # Value labels
        for bar, mean in zip(bars_g, greedy_means):
            if mean > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.03,
                        f"{mean:.2f}", ha="center", va="bottom", fontsize=7, color="#666")
        for bar, mean in zip(bars_b, best_of_means):
            if mean > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.03,
                        f"{mean:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    axes_flat[0].set_ylim(-0.05, max(2.5, axes_flat[0].get_ylim()[1]))

    fig.suptitle("AuditBench: Greedy vs Best-of-10 Correctness",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, 0.98, "Faded = greedy (T=0),  Hatched = best-of-10 (T=1.0)",
             ha="center", fontsize=10, style="italic", color="#555555")

    # Legend for greedy vs best-of
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#999999", alpha=0.6, edgecolor="black", label="Greedy (T=0)"),
        Patch(facecolor="#999999", edgecolor="black", hatch="//", label="Best-of-10 (T=1.0)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    out = BEST_OF_DIR / "greedy_vs_best_of_10.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

    # --- Per-behavior heatmap: best-of-10 ---
    _plot_per_behavior_best_of()

    # --- Print numeric summary ---
    _print_summary()


def _plot_per_behavior_best_of():
    behaviors = sorted({
        d["behavior_name"]
        for ao_id in AO_ORDER
        for cond in CONDITIONS
        if (r := load_experiment(BEST_OF_DIR, ao_id, cond["target"], cond["position"])) is not None
        for d in r["detailed_results"]
    })

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
    axes_flat = axes.ravel()

    for ax_idx, cond in enumerate(CONDITIONS):
        ax = axes_flat[ax_idx]
        matrix = np.zeros((len(behaviors), len(AO_ORDER)))

        for j, ao_id in enumerate(AO_ORDER):
            result = load_experiment(BEST_OF_DIR, ao_id, cond["target"], cond["position"])
            if result is None:
                continue
            for i, b in enumerate(behaviors):
                scores = [d["correctness"] - 1.0 for d in result["detailed_results"]
                          if d["behavior_name"] == b]
                matrix[i, j] = np.mean(scores) if scores else 0

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=4)

        ao_labels = [AO_DISPLAY[ao_id][0] for ao_id in AO_ORDER]
        ax.set_xticks(range(len(AO_ORDER)))
        ax.set_xticklabels(ao_labels, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(behaviors)))
        ax.set_yticklabels(behaviors, fontsize=9)
        ax.set_title(cond["title"], fontsize=11, fontweight="bold")

        for i in range(len(behaviors)):
            for j in range(len(AO_ORDER)):
                val = matrix[i, j]
                color = "white" if val > 2.0 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=9, color=color)

    fig.suptitle("Per-Behavior Best-of-10 Correctness",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.10, wspace=0.45, hspace=0.35)
    cbar_ax = fig.add_axes([0.25, 0.03, 0.50, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="Correctness (0-4)")

    out = BEST_OF_DIR / "per_behavior_best_of_10.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def _print_summary():
    print("\n" + "=" * 80)
    print("GREEDY vs BEST-OF-10 SUMMARY")
    print("=" * 80)
    print(f"{'AO':<16} {'Condition':<30} {'Greedy':>8} {'Best10':>8} {'Δ':>8}")
    print("-" * 80)

    for ao_id in AO_ORDER:
        label, _ = AO_DISPLAY[ao_id]
        for cond in CONDITIONS:
            g = load_experiment(GREEDY_DIR, ao_id, cond["target"], cond["position"])
            b = load_experiment(BEST_OF_DIR, ao_id, cond["target"], cond["position"])

            g_corr = g["overall_metrics"]["mean_correctness"] if g else 0
            b_corr = b["overall_metrics"]["mean_correctness"] if b else 0
            delta = b_corr - g_corr

            cond_label = f"{cond['target']}/{cond['position']}"
            print(f"{label:<16} {cond_label:<30} {g_corr:>8.3f} {b_corr:>8.3f} {delta:>+8.3f}")
        print()


if __name__ == "__main__":
    main()
