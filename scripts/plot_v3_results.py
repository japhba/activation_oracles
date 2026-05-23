"""V3 results plots — runs after AObench on all 17 ablations completes.

Generates 4 plots requested by user + 1 master leaderboard:
  Plot 1: corpus comparison (cot-v5 vs fineweb vs finefineweb)
  Plot 2: layer sweep (L18..L26)
  Plot 3: multi-layer vs single-layer
  Plot 4: conversational (default haiku vs latentqa)
  Plot 5: data quality (Gemini-haiku vs Sonnet-haiku vs combined)
  Plot 6: recipe (rsLoRA vs vanilla LoRA)
  Plot 7: master heatmap of all V3 runs vs all evals

Usage: python scripts/plot_v3_results.py [/path/to/v3_results_dir]
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Per-eval order (chance-adj normalized)
EVAL_ORDER = [
    "number_prediction", "mmlu_prediction", "missing_info", "sycophancy",
    "backtracking", "vagueness", "domain_confusion", "activation_sensitivity",
    "hallucination", "taboo", "personaqa",
    "system_prompt_qa_hidden", "system_prompt_qa_latentqa",
]
MORG_EVALS = ["taboo", "personaqa", "system_prompt_qa_hidden", "system_prompt_qa_latentqa"]


def load_results(results_dir: Path) -> dict[str, dict]:
    """Return {tag: aggregate_scores_final_dict}."""
    out = {}
    for sub in sorted(results_dir.iterdir()):
        if not sub.is_dir():
            continue
        # tag is the directory name with v3_ stripped
        tag = sub.name.replace("v3_", "")
        p = sub / "report" / "aggregate_scores.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        out[tag] = d.get("final", {})
    return out


def bar_plot(results: dict, tags: list[str], out_path: Path, title: str, subset_evals: list[str] | None = None):
    """Side-by-side bars: overall + morg mean, plus optional per-eval heatmap row."""
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(tags))))
    rows = []
    for tag in tags:
        if tag not in results:
            rows.append((tag, np.nan, np.nan))
            continue
        f = results[tag]
        overall = f.get("mean_normalized_score", np.nan)
        per = f.get("per_eval_normalized", {})
        morg = np.nanmean([per.get(e, np.nan) for e in MORG_EVALS]) if any(e in per for e in MORG_EVALS) else np.nan
        rows.append((tag, overall, morg))
    y = np.arange(len(rows))
    overall_vals = np.array([r[1] for r in rows])
    morg_vals = np.array([r[2] for r in rows])
    ax.barh(y - 0.2, overall_vals, height=0.4, label="Overall AObench", color="#3a78c2")
    ax.barh(y + 0.2, morg_vals, height=0.4, label="Model-org mean", color="#d44b4b")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Chance-adj score")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def master_heatmap(results: dict, out_path: Path):
    """All V3 runs as rows, evals as cols. Sorted by overall."""
    tags = sorted(results.keys(), key=lambda t: -results[t].get("mean_normalized_score", -1))
    evals_present = [e for e in EVAL_ORDER if any(e in results[t].get("per_eval_normalized", {}) for t in tags)]
    mat = np.array([[results[t].get("per_eval_normalized", {}).get(e, np.nan) for e in evals_present] for t in tags])

    fig, (ax_b, ax_h) = plt.subplots(1, 2, figsize=(max(14, 0.9 * len(evals_present) + 7), max(7, 0.4 * len(tags))),
                                       gridspec_kw={"width_ratios": [1, max(5, 0.6 * len(evals_present))]})
    means = np.array([results[t].get("mean_normalized_score", np.nan) for t in tags])
    ax_b.barh(range(len(tags)), means, color="#3a78c2", edgecolor="black", linewidth=0.3)
    for i, m in enumerate(means):
        ax_b.text(m + 0.005, i, f"{m:+.3f}", va="center", fontsize=8,
                  fontweight="bold" if m == np.nanmax(means) else "normal")
    ax_b.set_yticks(range(len(tags)))
    ax_b.set_yticklabels(tags, fontsize=9)
    ax_b.invert_yaxis()
    ax_b.axvline(0, color="gray", ls="--", lw=0.7)
    ax_b.set_xlabel("Chance-adj overall AObench mean")
    ax_b.set_title("V3 leaderboard", fontweight="bold")
    ax_b.grid(axis="x", alpha=0.3)

    absmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)))
    im = ax_h.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-absmax, vmax=absmax)
    ax_h.set_xticks(range(len(evals_present)))
    ax_h.set_xticklabels([e.replace("_", "\n") for e in evals_present], fontsize=8)
    ax_h.set_yticks(range(len(tags)))
    ax_h.set_yticklabels([])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax_h.text(j, i, f"{v:+.2f}", ha="center", va="center",
                      fontsize=7, color="white" if abs(v) > 0.35 else "black")
    # highlight morg evals
    for k in MORG_EVALS:
        if k in evals_present:
            j = evals_present.index(k)
            ax_h.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, len(tags), fill=False, edgecolor="#1565c0", linewidth=1.5))
    ax_h.set_title("Per-eval normalized (blue boxes = model-organism evals)", fontweight="bold")
    plt.colorbar(im, ax=ax_h, shrink=0.7, label="normalized score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def main(results_dir: str | None = None):
    if results_dir is None:
        results_dir = "/home/celeste/shared/v3_results"
    results_dir = Path(results_dir)
    results = load_results(results_dir)
    if not results:
        print(f"no results found under {results_dir}")
        return
    print(f"loaded {len(results)} runs: {list(results.keys())}")

    out_dir = Path("/home/celeste/shared/ao_paper_ablations/v3")
    out_dir.mkdir(parents=True, exist_ok=True)

    bar_plot(results,
             ["default_L21", "corpus_fineweb", "corpus_finefineweb"],
             out_dir / "01_corpus_comparison.png",
             "Plot 1: past_lens corpus (cot-v5 vs fineweb vs finefineweb)")

    bar_plot(results,
             [f"layer_L{L}" for L in [18, 19, 20]] + ["default_L21"] + [f"layer_L{L}" for L in [22, 23, 24, 25, 26]],
             out_dir / "02_layer_sweep_L18_L26.png",
             "Plot 2: layer sweep (L18 → L26)")

    bar_plot(results,
             ["default_L21", "multi_L20_22_24", "multi_L21_22_23_24_25"],
             out_dir / "03_multi_vs_single.png",
             "Plot 3: multi-layer vs single-layer")

    bar_plot(results,
             ["default_L21", "haiku_to_latentqa"],
             out_dir / "04_conv_haiku_vs_latentqa.png",
             "Plot 4: conversational data (haiku vs LatentQA, token-matched)")

    bar_plot(results,
             ["default_L21", "sonnet_haiku", "combined_haiku"],
             out_dir / "05_data_quality_sonnet.png",
             "Plot 5: haiku data quality (Gemini vs Sonnet vs combined)")

    bar_plot(results,
             ["default_L21", "vanilla_lora"],
             out_dir / "06_rslora_vs_vanilla.png",
             "Plot 6: rsLoRA r=128/α=16 vs vanilla LoRA r=64/α=128")

    master_heatmap(results, out_dir / "07_master_leaderboard.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
