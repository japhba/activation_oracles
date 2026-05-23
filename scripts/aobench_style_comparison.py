"""Reproduce AObench's standard comparison.png layout (multi-panel grid:
Overall + one panel per eval, with one bar per checkpoint per panel) for the
5-layer sweep.  Uses the per_eval_normalized scores already in the aggregate
JSONs.
"""

import json
import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/celeste/shared/ao_layer_sweep")
LAYERS = [19, 22, 25, 28, 31]
DEPTH_PCT = {19: 53, 22: 62, 25: 70, 28: 78, 31: 87}
DISPLAY_NAME = {L: f"L{L} ({DEPTH_PCT[L]}%)" for L in LAYERS}

EVAL_TITLES = {
    "number_prediction": "Number Prediction",
    "mmlu_prediction": "MMLU Prediction",
    "backtracking": "Backtracking",
    "missing_info": "Detect Missing Info",
    "sycophancy": "Sycophancy",
    "vagueness": "Response Specificity\n(↑ = less vague)",
    "domain_confusion": "Domain Confusion\n(↑ = better)",
    "activation_sensitivity": "Activation Sensitivity",
    "hallucination": "Not Obviously Wrong\n(↑ = less hallucination)",
}

# Load per-layer aggregate scores
data: dict[int, dict] = {}
for L in LAYERS:
    p = ROOT / f"aggregate_L{L}.json"
    if not p.exists():
        continue
    data[L] = json.loads(p.read_text())["final"]

eval_names = sorted({e for d in data.values() for e in d["per_eval_normalized"]})
panel_specs = [("overall", "Overall Score")] + [
    (e, EVAL_TITLES.get(e, e.replace("_", " ").title())) for e in eval_names
]

n_panels = len(panel_specs)
n_rows = 3
n_cols = math.ceil(n_panels / n_rows)
fig_width = max(16.0, 3.4 * n_cols + 2.4)
fig_height = 13.0
fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
axes_flat = axes.flatten()

verb_names = LAYERS
display_labels = [DISPLAY_NAME[L] for L in LAYERS]
bar_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(LAYERS)))
x = np.arange(len(LAYERS), dtype=float)

# Overall panel y range
overall_lo = min(data[L]["ci_lo"] for L in LAYERS)
overall_hi = max(data[L]["ci_hi"] for L in LAYERS)
overall_ymin = min(-0.08, overall_lo - 0.05)
overall_ymax = max(0.30, overall_hi + 0.10)

# Per-eval panel y range (across all layers, all evals)
per_eval_min = min(data[L]["per_eval_normalized"].get(e, 0.0) for L in LAYERS for e in eval_names)
per_eval_max = max(data[L]["per_eval_normalized"].get(e, 0.0) for L in LAYERS for e in eval_names)
eval_ymin = min(-0.05, per_eval_min - 0.05)
eval_ymax = max(0.35, per_eval_max + 0.05)

for panel_idx, (panel_name, panel_title) in enumerate(panel_specs):
    ax = axes_flat[panel_idx]
    if panel_name == "overall":
        values = [data[L]["mean_normalized_score"] for L in LAYERS]
        lowers = [max(0.0, v - data[L]["ci_lo"]) for v, L in zip(values, LAYERS)]
        uppers = [max(0.0, data[L]["ci_hi"] - v) for v, L in zip(values, LAYERS)]
        ax.set_ylim(overall_ymin, overall_ymax)
        ax.set_ylabel("Normalized score")
        yerr = np.array([lowers, uppers])
    else:
        values = [data[L]["per_eval_normalized"].get(panel_name, 0.0) for L in LAYERS]
        ax.set_ylim(eval_ymin, eval_ymax)
        if panel_idx % n_cols == 0:
            ax.set_ylabel("Normalized score")
        yerr = None

    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.85)
    ax.bar(
        x,
        values,
        width=0.78,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
        yerr=yerr,
        capsize=3,
    )
    # Highlight best
    best_idx = int(np.argmax(values))
    for i, v in enumerate(values):
        ax.text(i, v + (0.01 if v >= 0 else -0.025), f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8.5, fontweight="bold" if i == best_idx else "normal",
                color="#1a4d1a" if i == best_idx else "#444444")
    ax.set_title(panel_title, fontsize=11, pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{L}" for L in LAYERS], fontsize=9)
    ax.set_xlim(-0.6, len(LAYERS) - 0.4)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)

for extra in axes_flat[n_panels:]:
    extra.axis("off")

legend_handles = [
    mpatches.Patch(facecolor=bar_colors[i], edgecolor="white", linewidth=0.5, label=display_labels[i])
    for i in range(len(LAYERS))
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.10),
    ncol=len(LAYERS),
    fontsize=12,
    title="Checkpoint (trained at single layer)",
    frameon=False,
    columnspacing=1.6,
    handlelength=2.0,
)

fig.text(
    0.5,
    0.022,
    "Higher is better for all panels (sign-flipped on lower-is-better metrics).\n"
    "Overall = mean chance-adjusted normalized score across evals; error bars are 95% bootstrap CI on the overall mean.",
    ha="center",
    va="bottom",
    fontsize=10,
    color="#333333",
)

fig.suptitle(
    "Qwen3-8B Activation Oracle — AObench layer sweep (L19/L22/L25/L28/L31)",
    fontsize=17,
    y=0.97,
)
fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.18, hspace=0.45, wspace=0.22)

out_path = ROOT / "comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"wrote {out_path}")
