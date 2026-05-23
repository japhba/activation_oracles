"""Plot binary judge accuracy for AO checkpoint sweep + introspection adapter baselines."""

import asyncio
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

AUDITBENCH_DIR = Path(__file__).resolve().parents[1] / "auditbench"
SWEEP_DIR = AUDITBENCH_DIR / "sweep_binary"
INTROSPECTION_DIR = AUDITBENCH_DIR / "introspection_adapter_baseline_prompt_only"

AO_DISPLAY = {
    "hf_past_lens": ("Original AO", "#f58231"),
    "mlao": ("MLAO", "#e6194b"),
    "local_original": ("AO v2", "#3cb44b"),
    "local_hb": ("AO v2 + HB", "#4363d8"),
    "local_50k_hb_50k_hbi": ("AO v2 + 50k HB\n+ 50k HBI", "#911eb4"),
    "local_300k_hb": ("AO v2 + 300k HB", "#469990"),
    "local_300k_hb_300k_hbi": ("AO v2 + 300k HB\n+ 300k HBIv2", "#42d4f4"),
}

AO_ORDER = ["hf_past_lens", "mlao", "local_original", "local_hb", "local_50k_hb_50k_hbi", "local_300k_hb", "local_300k_hb_300k_hbi"]

# Baselines (not AOs — direct text generation)
BASELINE_DISPLAY = {
    "direct_target_only": ("Base Model", "#cccccc"),
    "direct_target_plus_meta": ("IA", "#8c564b"),
}
BASELINE_ORDER = ["direct_target_only", "direct_target_plus_meta"]

CONDITIONS = [
    {"target": "transcripts", "position": "pre_answer", "title": "Transcripts + KTO\nAO input: prompt only"},
    {"target": "transcripts", "position": "full_seq", "title": "Transcripts + KTO\nAO input: prompt + assistant response"},
    {"target": "synth_docs", "position": "pre_answer", "title": "Synth-Docs + SFT\nAO input: prompt only"},
    {"target": "synth_docs", "position": "full_seq", "title": "Synth-Docs + SFT\nAO input: prompt + assistant response"},
]


def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    values = np.array(values, dtype=float)
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


def get_correct_values(result):
    if result is None:
        return []
    return [float(d["correct"]) for d in result["detailed_results"]]


def load_introspection_binary_scores(target_id: str) -> dict[str, list[float]]:
    """Load introspection adapter results and re-judge with binary scorer.

    Returns {evaluation_mode: [correct_values]}.
    """
    cache_path = INTROSPECTION_DIR / "binary_scores_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        result = {}
        for r in cached:
            if r["target_id"] == target_id:
                result.setdefault(r["evaluation_mode"], []).append(float(r["correct"]))
        if result:
            return result

    # Need to re-judge — import and run binary judge on existing responses
    from nl_probes.base_experiment import VerbalizerResults
    from nl_probes.open_ended_eval.auditbench import (
        judge_auditbench_responses,
        get_first_ao_response,
    )

    introspection_files = sorted(INTROSPECTION_DIR.glob("introspection_baseline_*.json"))
    assert introspection_files, f"No introspection results found in {INTROSPECTION_DIR}"

    all_scored = []
    for fpath in introspection_files:
        with open(fpath) as f:
            data = json.load(f)

        results = []
        metadata = []
        for sr in data["scored_results"]:
            results.append(VerbalizerResults(
                verbalizer_lora_path=None, target_lora_path="",
                context_token_ids=[], act_key="",
                verbalizer_prompt=sr["verbalizer_prompt"],
                ground_truth=sr["behavior_name"],
                num_tokens=0, responses=[sr["ao_response"]],
            ))
            metadata.append({
                "behavior_name": sr["behavior_name"],
                "behavior_description": sr["behavior_description"],
                "context_prompt": sr["context_prompt"],
                "context_prompt_key": sr["context_prompt_key"],
                "verbalizer_prompt": sr["verbalizer_prompt"],
                "verbalizer_prompt_key": sr["verbalizer_prompt_key"],
                "position_mode": sr["position_mode"],
            })

        binary_scored = asyncio.run(judge_auditbench_responses(
            results=results, metadata=metadata,
            judge_model="claude-haiku-4-5-20251001", judge_concurrency=20,
            judge_mode="binary",
        ))

        for sr, bs in zip(data["scored_results"], binary_scored):
            all_scored.append({
                "target_id": sr["target_id"],
                "evaluation_mode": sr["evaluation_mode"],
                "behavior_name": sr["behavior_name"],
                "correct": bs["correct"],
            })

    with open(cache_path, "w") as f:
        json.dump(all_scored, f, indent=2)
    print(f"Cached binary scores to {cache_path}")

    result = {}
    for r in all_scored:
        if r["target_id"] == target_id:
            result.setdefault(r["evaluation_mode"], []).append(float(r["correct"]))
    return result


def _compute_bar_data(vals):
    """Return (mean, ci_lo, ci_hi) for a list of values."""
    if vals:
        m = np.mean(vals)
        lo, hi = bootstrap_ci(vals)
        return m, max(0, m - lo), max(0, hi - m)
    return 0, 0, 0


def main():
    # --- Bar chart: accuracy by AO and condition, with baselines ---
    # Baselines only shown for pre_answer (prompt-only) conditions
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharey=True)
    axes_flat = axes.ravel()

    bar_width = 0.7

    for ax_idx, cond in enumerate(CONDITIONS):
        ax = axes_flat[ax_idx]
        include_baselines = cond["position"] == "pre_answer"

        bar_ids = AO_ORDER + (BASELINE_ORDER if include_baselines else [])
        n_bars = len(bar_ids)
        x_positions = np.arange(n_bars)

        means = []
        ci_los = []
        ci_his = []
        colors = []
        labels = []
        hatches = []

        # AO bars
        for ao_id in AO_ORDER:
            label, color = AO_DISPLAY[ao_id]
            labels.append(label)
            colors.append(color)
            hatches.append("")

            result = load_experiment(SWEEP_DIR, ao_id, cond["target"], cond["position"])
            m, clo, chi = _compute_bar_data(get_correct_values(result))
            means.append(m)
            ci_los.append(clo)
            ci_his.append(chi)

        # Baseline bars (only for prompt-only conditions)
        if include_baselines:
            introspection_scores = load_introspection_binary_scores(cond["target"])
            for bl_id in BASELINE_ORDER:
                label, color = BASELINE_DISPLAY[bl_id]
                labels.append(label)
                colors.append(color)
                hatches.append("//" if bl_id == "direct_target_only" else "")

                m, clo, chi = _compute_bar_data(introspection_scores.get(bl_id, []))
                means.append(m)
                ci_los.append(clo)
                ci_his.append(chi)

        means = np.array(means)
        errors = np.array([ci_los, ci_his])

        bars = ax.bar(
            x_positions, means, bar_width,
            color=colors, edgecolor="black", linewidth=0.8,
            yerr=errors, capsize=4, error_kw={"linewidth": 1.0},
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        ax.set_title(cond["title"], fontsize=15, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([""] * len(bar_ids))  # hide x labels, use legend
        ax.set_ylabel("Accuracy" if ax_idx % 2 == 0 else "", fontsize=14)
        ax.set_ylim(0, 0.5)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=13)

        for bar, mean in zip(bars, means):
            if mean >= 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.008,
                        f"{mean:.0%}", ha="center", va="bottom", fontsize=12,
                        fontweight="bold")

        # Save handles from first subplot for legend
        if ax_idx == 0:
            legend_bars = bars
            legend_labels = labels

    fig.suptitle("AuditBench: Binary Accuracy",
                 fontsize=17, fontweight="bold", y=1.01)

    # Single legend at bottom
    fig.legend(legend_bars, [l.replace("\n", " ") for l in legend_labels],
               loc="lower center", ncol=4, fontsize=12, frameon=True,
               bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])

    out = SWEEP_DIR / "auditbench_binary_accuracy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

    # --- Per-behavior heatmap ---
    _plot_per_behavior()

    # --- Greedy vs best-of-10 ---
    _plot_best_of_10()

    # --- Print numeric summary ---
    _print_summary()


def _plot_per_behavior():
    behaviors = sorted({
        d["behavior_name"]
        for ao_id in AO_ORDER
        for cond in CONDITIONS
        if (r := load_experiment(SWEEP_DIR, ao_id, cond["target"], cond["position"])) is not None
        for d in r["detailed_results"]
    })

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=True)
    axes_flat = axes.ravel()

    for ax_idx, cond in enumerate(CONDITIONS):
        ax = axes_flat[ax_idx]
        matrix = np.zeros((len(behaviors), len(AO_ORDER)))

        for j, ao_id in enumerate(AO_ORDER):
            result = load_experiment(SWEEP_DIR, ao_id, cond["target"], cond["position"])
            if result is None:
                continue
            for i, b in enumerate(behaviors):
                scores = [float(d["correct"]) for d in result["detailed_results"]
                          if d["behavior_name"] == b]
                matrix[i, j] = np.mean(scores) if scores else 0

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1.0)

        ao_labels = [AO_DISPLAY[ao_id][0] for ao_id in AO_ORDER]
        ax.set_xticks(range(len(AO_ORDER)))
        ax.set_xticklabels(ao_labels, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(behaviors)))
        ax.set_yticklabels(behaviors, fontsize=9)
        ax.set_title(cond["title"], fontsize=11, fontweight="bold")

        for i in range(len(behaviors)):
            for j in range(len(AO_ORDER)):
                val = matrix[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=8, color=color)

    fig.suptitle("Per-Behavior Binary Accuracy by AO Checkpoint",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.10, wspace=0.45, hspace=0.35)
    cbar_ax = fig.add_axes([0.25, 0.03, 0.50, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="Accuracy")

    out = SWEEP_DIR / "per_behavior_binary_accuracy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def _print_summary():
    print("\n" + "=" * 70)
    print("BINARY ACCURACY SUMMARY")
    print("=" * 70)
    print(f"{'AO':<22} {'Condition':<28} {'Accuracy':>8} {'N':>5}")
    print("-" * 70)

    for ao_id in AO_ORDER:
        label, _ = AO_DISPLAY[ao_id]
        for cond in CONDITIONS:
            result = load_experiment(SWEEP_DIR, ao_id, cond["target"], cond["position"])
            if result:
                acc = result["overall_metrics"]["accuracy"]
                n = int(result["overall_metrics"]["num_scored"])
            else:
                acc = 0
                n = 0
            cond_label = f"{cond['target']}/{cond['position']}"
            print(f"{label:<22} {cond_label:<28} {acc:>8.3f} {n:>5}")
        print()


def _plot_best_of_10():
    """Plot greedy vs best-of-10 comparison."""
    BEST_OF_DIR = AUDITBENCH_DIR / "sweep_binary_best_of_10"
    BEST_OF_IA_DIR = AUDITBENCH_DIR / "introspection_adapter_baseline_prompt_only_best_of_10"

    fig, axes = plt.subplots(2, 2, figsize=(20, 11), sharey=True)
    axes_flat = axes.ravel()

    bar_width = 0.35

    for ax_idx, cond in enumerate(CONDITIONS):
        ax = axes_flat[ax_idx]
        include_baselines = cond["position"] == "pre_answer"

        greedy_means = []
        greedy_ci_los = []
        greedy_ci_his = []
        bo_means = []
        bo_ci_los = []
        bo_ci_his = []
        colors = []
        labels = []

        # AO bars
        for ao_id in AO_ORDER:
            label, color = AO_DISPLAY[ao_id]
            labels.append(label)
            colors.append(color)

            # Greedy
            gm, gclo, gchi = _compute_bar_data(get_correct_values(
                load_experiment(SWEEP_DIR, ao_id, cond["target"], cond["position"])))
            greedy_means.append(gm)
            greedy_ci_los.append(gclo)
            greedy_ci_his.append(gchi)

            # Best-of-10
            bm, bclo, bchi = _compute_bar_data(get_correct_values(
                load_experiment(BEST_OF_DIR, ao_id, cond["target"], cond["position"])))
            bo_means.append(bm)
            bo_ci_los.append(bclo)
            bo_ci_his.append(bchi)

        # Baseline bars (only for prompt-only conditions)
        if include_baselines:
            introspection_greedy = load_introspection_binary_scores(cond["target"])

            # Load best-of-10 IA scores if available
            bo_ia_cache = BEST_OF_IA_DIR / "binary_scores_cache.json"
            bo_ia_scores: dict[str, list[float]] = {}
            if bo_ia_cache.exists():
                with open(bo_ia_cache) as f:
                    for r in json.load(f):
                        if r["target_id"] == cond["target"]:
                            bo_ia_scores.setdefault(r["evaluation_mode"], []).append(float(r["correct"]))
            else:
                bo_ia_files = sorted(BEST_OF_IA_DIR.glob("introspection_baseline_*.json")) if BEST_OF_IA_DIR.exists() else []
                for fpath in bo_ia_files:
                    with open(fpath) as f:
                        data = json.load(f)
                    for sr in data.get("scored_results", []):
                        if sr.get("target_id") == cond["target"]:
                            bo_ia_scores.setdefault(sr["evaluation_mode"], []).append(
                                float(sr.get("correct", 0)))

            # Hardcoded fallback from printed logs (job crashed before saving JSON).
            # Remove once real JSON exists.
            _IA_BEST_OF_10_FALLBACK = {
                "transcripts": {"direct_target_only": 0.036, "direct_target_plus_meta": 0.071},
                "synth_docs": {"direct_target_only": 0.143, "direct_target_plus_meta": 0.518},
            }

            for bl_id in BASELINE_ORDER:
                label, color = BASELINE_DISPLAY[bl_id]
                labels.append(label)
                colors.append(color)

                gm, gclo, gchi = _compute_bar_data(introspection_greedy.get(bl_id, []))
                greedy_means.append(gm)
                greedy_ci_los.append(gclo)
                greedy_ci_his.append(gchi)

                b_vals = bo_ia_scores.get(bl_id, [])
                if b_vals:
                    bm, bclo, bchi = _compute_bar_data(b_vals)
                else:
                    # Use fallback — no CI since it's from printed logs
                    bm = _IA_BEST_OF_10_FALLBACK.get(cond["target"], {}).get(bl_id, 0)
                    bclo, bchi = 0, 0
                bo_means.append(bm)
                bo_ci_los.append(bclo)
                bo_ci_his.append(bchi)

        n_bars = len(greedy_means)
        x_positions = np.arange(n_bars)

        greedy_means = np.array(greedy_means)
        bo_means = np.array(bo_means)
        greedy_errors = np.array([greedy_ci_los, greedy_ci_his])
        bo_errors = np.array([bo_ci_los, bo_ci_his])

        bars_g = ax.bar(
            x_positions - bar_width / 2, greedy_means, bar_width,
            color=colors, edgecolor="black", linewidth=0.8, alpha=0.5,
            yerr=greedy_errors, capsize=3, error_kw={"linewidth": 1.0},
        )
        bars_b = ax.bar(
            x_positions + bar_width / 2, bo_means, bar_width,
            color=colors, edgecolor="black", linewidth=0.8,
            yerr=bo_errors, capsize=3, error_kw={"linewidth": 1.0},
            hatch="//",
        )

        ax.set_title(cond["title"], fontsize=14, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([""] * n_bars)
        ax.set_ylabel("Accuracy" if ax_idx % 2 == 0 else "", fontsize=13)
        ax.set_ylim(0, 0.7)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=12)

        for bar, mean in zip(bars_g, greedy_means):
            if mean >= 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.008,
                        f"{mean:.0%}", ha="center", va="bottom", fontsize=9,
                        color="#666")
        for bar, mean in zip(bars_b, bo_means):
            if mean >= 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.008,
                        f"{mean:.0%}", ha="center", va="bottom", fontsize=9,
                        fontweight="bold")

        if ax_idx == 0:
            legend_bars_g = bars_g
            legend_bars_b = bars_b
            legend_labels = labels

    fig.suptitle("AuditBench: Greedy vs Best-of-10 Binary Accuracy",
                 fontsize=16, fontweight="bold", y=1.02)

    from matplotlib.patches import Patch
    method_legend = [
        Patch(facecolor="#999999", alpha=0.5, edgecolor="black", label="Greedy (T=0)"),
        Patch(facecolor="#999999", edgecolor="black", hatch="//", label="Best-of-10 (T=1.0)"),
    ]
    # Color legend
    color_handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="black")
                     for _, c in [AO_DISPLAY[ao] for ao in AO_ORDER] + [BASELINE_DISPLAY[bl] for bl in BASELINE_ORDER]]
    color_labels = [l.replace("\n", " ") for l in legend_labels]

    leg1 = fig.legend(handles=method_legend, loc="lower left",
                      fontsize=11, frameon=True, bbox_to_anchor=(0.02, -0.06))
    fig.legend(handles=color_handles, labels=color_labels,
               loc="lower center", ncol=4, fontsize=11, frameon=True,
               bbox_to_anchor=(0.55, -0.06))
    fig.add_artist(leg1)

    plt.tight_layout(rect=[0, 0.07, 1, 0.97])

    out = SWEEP_DIR / "auditbench_binary_greedy_vs_best10.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
