"""Replicate the AuditBench 2x2 greedy plot layout with introspection baselines."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GREEDY_DIR = Path(__file__).resolve().parent.parent / "auditbench_archive" / "overnight_sweep_v2"
INTROSPECTION_RESULTS = (
    Path(__file__).resolve().parent
    / "introspection_adapter_baseline_prompt_only"
    / "introspection_baseline_targets_transcripts-synth_docs_nbeh_14_nprompts_4.json"
)

AO_DISPLAY = {
    "hf_past_lens": ("Original AO", "#f58231"),
    "mlao": ("MLAO", "#e6194b"),
    "local_original": ("AO v2", "#3cb44b"),
    "local_hb": ("AO v2 + HB", "#4363d8"),
}

AO_ORDER = ["hf_past_lens", "mlao", "local_original", "local_hb"]

INTROSPECTION_DISPLAY = {
    "direct_target_plus_meta": ("IA", "#8c564b", "\\\\"),
}

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
    boot_means = np.array([rng.choice(values, size=n, replace=True).mean() for _ in range(n_bootstrap)])
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
    if result is None:
        return []
    return [d["correctness"] - 1.0 for d in result["detailed_results"]]


def load_introspection_payload():
    with open(INTROSPECTION_RESULTS) as f:
        return json.load(f)


def get_introspection_signal(payload, target_id: str, evaluation_mode: str) -> list[float]:
    return [
        row["correctness"] - 1.0
        for row in payload["scored_results"]
        if row["target_id"] == target_id and row["evaluation_mode"] == evaluation_mode
    ]


def _mean_and_errors(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(values))
    lo, hi = bootstrap_ci(values)
    return mean, max(0.0, mean - lo), max(0.0, hi - mean)


def main() -> None:
    introspection_payload = load_introspection_payload()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    axes_flat = axes.ravel()

    for ax_idx, cond in enumerate(CONDITIONS):
        ax = axes_flat[ax_idx]

        labels = []
        means = []
        ci_los = []
        ci_his = []
        colors = []
        hatches = []

        for ao_id in AO_ORDER:
            label, color = AO_DISPLAY[ao_id]
            signals = get_correctness_signal(load_experiment(GREEDY_DIR, ao_id, cond["target"], cond["position"]))
            mean, lo, hi = _mean_and_errors(signals)
            labels.append(label)
            means.append(mean)
            ci_los.append(lo)
            ci_his.append(hi)
            colors.append(color)
            hatches.append(None)

        if cond["position"] == "pre_answer":
            for evaluation_mode in ("direct_target_plus_meta",):
                label, color, hatch = INTROSPECTION_DISPLAY[evaluation_mode]
                signals = get_introspection_signal(introspection_payload, cond["target"], evaluation_mode)
                mean, lo, hi = _mean_and_errors(signals)
                labels.append(label)
                means.append(mean)
                ci_los.append(lo)
                ci_his.append(hi)
                colors.append(color)
                hatches.append(hatch)

        x_positions = np.arange(len(labels))
        errors = np.array([ci_los, ci_his])

        bars = []
        for idx, (label, mean, color, hatch) in enumerate(zip(labels, means, colors, hatches, strict=True)):
            bar = ax.bar(
                x_positions[idx],
                mean,
                0.72,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                hatch=hatch,
                yerr=np.array([[errors[0][idx]], [errors[1][idx]]]),
                capsize=3,
                error_kw={"linewidth": 1.0},
            )
            bars.append(bar[0])

        ax.set_title(cond["title"], fontsize=11, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Correctness (0-4)" if ax_idx % 2 == 0 else "")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)

        if cond["position"] == "full_seq":
            ax.text(
                0.98,
                0.94,
                "No prompt+response\nintrospection run",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#555555",
            )

        for bar, mean in zip(bars, means, strict=True):
            if mean > 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean + 0.03,
                    f"{mean:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold" if "MetaLoRA" in bar.get_label() else None,
                )

    axes_flat[0].set_ylim(-0.05, max(2.5, axes_flat[0].get_ylim()[1]))

    fig.suptitle("AuditBench: Greedy AO vs Introspection Adapter",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.text(
        0.5,
        0.98,
        "Metric = mean correctness - 1.0. Error bars = bootstrap 95% CI.",
        ha="center",
        fontsize=10,
        style="italic",
        color="#555555",
    )

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#f58231", edgecolor="black", label="Original AO"),
        Patch(facecolor="#e6194b", edgecolor="black", label="MLAO"),
        Patch(facecolor="#3cb44b", edgecolor="black", label="AO v2"),
        Patch(facecolor="#4363d8", edgecolor="black", label="AO v2 + HB"),
        Patch(facecolor="#8c564b", edgecolor="black", hatch="\\\\", label="IA"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])

    out = INTROSPECTION_RESULTS.parent / "introspection_vs_ao_2x2_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
