"""AObench-style comparison plot showing ABSOLUTE scores per layer.

Recovers raw per-eval scores from `per_eval_normalized` by inverting AObench's
`normalize_metric_for_aggregate`:
    normalized = (raw - chance) / (1 - chance)
=>  raw = normalized * (1 - chance) + chance

With AObench's CHANCE_BASELINES (mmlu/missing_info/sycophancy = 0.5; others 0).
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

CHANCE_BASELINES: dict[str, float] = {
    "mmlu_prediction": 0.50,
    "missing_info": 0.50,
    "sycophancy": 0.50,
}
SCALE_1_5_EVALS = {"backtracking", "system_prompt_qa_hidden", "system_prompt_qa_latentqa"}

# Display title + raw metric name (per AObench EVAL_METRIC_MAP)
EVAL_METRIC = {
    "number_prediction": ("Number Prediction", "accuracy"),
    "mmlu_prediction": ("MMLU Prediction", "AUC"),
    "backtracking": ("Backtracking", "score (1-5)"),
    "missing_info": ("Detect Missing Info", "AUC"),
    "sycophancy": ("Sycophancy", "AUC"),
    "vagueness": ("Response Specificity", "specificity rate"),
    "domain_confusion": ("Domain Confusion", "domain accuracy"),
    "activation_sensitivity": ("Activation Sensitivity", "sensitivity"),
    "hallucination": ("Not Obviously Wrong", "1 - obvious-halluc. rate"),
}


def absolute(eval_name: str, normalized: float) -> float:
    """Invert AObench's normalize_metric_for_aggregate."""
    chance = CHANCE_BASELINES.get(eval_name, 0.0)
    raw = normalized * (1.0 - chance) + chance
    if eval_name in SCALE_1_5_EVALS:
        # raw is already on 0-1 here; convert back to 1-5 scale
        raw = raw * 4.0 + 1.0
    return raw


# Load
data: dict[int, dict] = {}
for L in LAYERS:
    p = ROOT / f"aggregate_L{L}.json"
    if not p.exists():
        continue
    data[L] = json.loads(p.read_text())["final"]

eval_names = sorted({e for d in data.values() for e in d["per_eval_normalized"]})
n_layers = len(LAYERS)

panel_specs: list[tuple[str, str]] = [(e, EVAL_METRIC.get(e, (e, ""))[0]) for e in eval_names]
n_panels = len(panel_specs)
n_rows = 3
n_cols = math.ceil(n_panels / n_rows)
fig_width = max(15.0, 3.4 * n_cols + 1.6)
fig_height = 11.0
fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
axes_flat = axes.flatten()

bar_colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_layers))
display_labels = [f"L{L} ({DEPTH_PCT[L]}%)" for L in LAYERS]
x = np.arange(n_layers, dtype=float)

for panel_idx, (eval_name, panel_title) in enumerate(panel_specs):
    ax = axes_flat[panel_idx]
    raw_vals = [absolute(eval_name, data[L]["per_eval_normalized"].get(eval_name, 0.0)) for L in LAYERS]

    # Set y-range based on metric type
    if eval_name in SCALE_1_5_EVALS:
        ymin, ymax = 1.0, 5.0
    else:
        ymin, ymax = 0.0, 1.05

    # Chance reference line
    chance = CHANCE_BASELINES.get(eval_name)
    if chance is not None:
        ax.axhline(chance, color="#b23b3b", linestyle="--", linewidth=1.0, alpha=0.85, label="chance")

    bars = ax.bar(
        x,
        raw_vals,
        width=0.78,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
    )
    best_idx = int(np.argmax(raw_vals))
    for i, v in enumerate(raw_vals):
        ax.text(i, v + 0.012 * (ymax - ymin), f"{v:.2f}",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold" if i == best_idx else "normal",
                color="#1a4d1a" if i == best_idx else "#444444")
    metric_lbl = EVAL_METRIC.get(eval_name, (eval_name, ""))[1]
    ax.set_title(f"{panel_title}\n[{metric_lbl}]", fontsize=10.5, pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{L}" for L in LAYERS], fontsize=9)
    ax.set_xlim(-0.6, n_layers - 0.4)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    if panel_idx % n_cols == 0:
        ax.set_ylabel("Score (absolute)")

for extra in axes_flat[n_panels:]:
    extra.axis("off")

legend_handles = [
    mpatches.Patch(facecolor=bar_colors[i], edgecolor="white", linewidth=0.5, label=display_labels[i])
    for i in range(n_layers)
] + [mpatches.Patch(facecolor="none", edgecolor="#b23b3b", linewidth=1.0, label="chance baseline (0.5)", linestyle="--")]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.06),
    ncol=6,
    fontsize=11,
    frameon=False,
    columnspacing=1.4,
    handlelength=2.0,
)

fig.text(
    0.5,
    0.012,
    "Absolute AObench scores (higher = better for all panels; lower-is-better metrics already sign-flipped per AObench convention).",
    ha="center", va="bottom", fontsize=10, color="#333333",
)

fig.suptitle(
    "Qwen3-8B Activation Oracle layer sweep — absolute AObench scores",
    fontsize=16, y=0.97,
)
fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.13, hspace=0.55, wspace=0.22)

out_path = ROOT / "comparison_absolute.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"wrote {out_path}")

# Also print the table
print("\n== Absolute scores per layer ==")
header = f"{'eval':<28}" + "  ".join(f"{f'L{L}':>8}" for L in LAYERS) + "  chance"
print(header)
print("-" * len(header))
for ev in eval_names:
    cells = "  ".join(f"{absolute(ev, data[L]['per_eval_normalized'].get(ev, 0.0)):>8.3f}" for L in LAYERS)
    chance = CHANCE_BASELINES.get(ev)
    chance_str = f"  {chance:.2f}" if chance is not None else "  -"
    print(f"{ev:<28}{cells}{chance_str}")
