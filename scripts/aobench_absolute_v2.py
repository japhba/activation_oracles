"""Per-layer absolute AObench scores plot for the L20/L21/L22/L23 sweep.

Reads each layer's report/eval_intervals.json — that file contains the primary
metric value + 95% bootstrap CI per eval in the absolute scale, exactly the
data AObench's own report uses for its bar charts.
"""
import json
import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/celeste/shared/ao_layer_sweep_v2")
LAYERS = [20, 21, 22, 23]
DEPTH_PCT = {20: 56, 21: 59, 22: 62, 23: 64}

EVAL_TITLES = {
    "number_prediction": ("Number Prediction", "match-true rate"),
    "mmlu_prediction": ("MMLU Prediction", "AUC"),
    "backtracking": ("Backtracking", "score (1-5)"),
    "missing_info": ("Detect Missing Info", "AUC"),
    "sycophancy": ("Sycophancy", "AUC"),
    "vagueness": ("Response Specificity", "1 - vagueness"),
    "domain_confusion": ("Domain Confusion", "domain accuracy"),
    "activation_sensitivity": ("Activation Sensitivity", "sensitivity"),
    "hallucination": ("Not Obviously Wrong", "1 - obvious-halluc."),
}

CHANCE_BASELINES = {
    "mmlu_prediction": 0.50,
    "missing_info": 0.50,
    "sycophancy": 0.50,
}
SCALE_1_5_EVALS = {"backtracking", "system_prompt_qa_hidden", "system_prompt_qa_latentqa"}

# Load
data: dict[int, dict[str, dict]] = {}
for L in LAYERS:
    p = ROOT / f"L{L}" / "report" / "eval_intervals.json"
    if not p.exists():
        print(f"[skip] L{L}: no eval_intervals"); continue
    intervals = json.loads(p.read_text())
    data[L] = {e: v.get("final", {}) for e, v in intervals.items()}

eval_names = sorted({e for d in data.values() for e in d})
panels = [(e, *EVAL_TITLES.get(e, (e, ""))) for e in eval_names]

# -- Figure layout: top row = "Overall Score" panel (chance-adjusted), then per-eval grid --
agg = {}
for L in LAYERS:
    p = ROOT / f"L{L}" / "report" / "aggregate_scores.json"
    if p.exists():
        agg[L] = json.loads(p.read_text())["final"]

n_panels = len(panels) + 1  # +1 for overall
n_rows = 3
n_cols = math.ceil(n_panels / n_rows)
fig_width = max(15.0, 3.4 * n_cols + 1.6)
fig_height = 12.0
fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
axes_flat = axes.flatten()

bar_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(LAYERS)))
display_labels = [f"L{L} ({DEPTH_PCT[L]}%)" for L in LAYERS]
x = np.arange(len(LAYERS), dtype=float)

# Panel 0: Overall (chance-adjusted normalized score with CI from aggregate_scores)
ax = axes_flat[0]
vals = [agg[L]["mean_normalized_score"] for L in LAYERS]
lo = [max(0.0, v - agg[L]["ci_lo"]) for v, L in zip(vals, LAYERS)]
hi = [max(0.0, agg[L]["ci_hi"] - v) for v, L in zip(vals, LAYERS)]
ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.85)
ax.bar(x, vals, width=0.78, color=bar_colors, edgecolor="white", linewidth=0.5,
       yerr=np.array([lo, hi]), capsize=3)
best_idx = int(np.argmax(vals))
for i, v in enumerate(vals):
    ax.text(i, v + 0.012, f"{v:+.3f}",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold" if i == best_idx else "normal",
            color="#1a4d1a" if i == best_idx else "#444444")
ax.set_title("Overall Score\n[chance-adjusted mean]", fontsize=11, pad=6, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([f"L{L}" for L in LAYERS], fontsize=9)
ax.set_ylabel("Normalized score", fontsize=10)
overall_lo = min(agg[L]["ci_lo"] for L in LAYERS)
overall_hi = max(agg[L]["ci_hi"] for L in LAYERS)
ax.set_ylim(min(-0.05, overall_lo - 0.02), max(0.30, overall_hi + 0.05))
ax.set_xlim(-0.6, len(LAYERS) - 0.4)
ax.yaxis.grid(True, color="#dddddd", linewidth=0.8, alpha=0.9)
ax.set_axisbelow(True)

# Per-eval panels
for panel_idx, (eval_name, title, metric_lbl) in enumerate(panels, start=1):
    ax = axes_flat[panel_idx]
    raw_vals, lows, highs = [], [], []
    for L in LAYERS:
        info = data.get(L, {}).get(eval_name, {})
        v = info.get("mean", 0.0)
        raw_vals.append(v)
        lows.append(max(0.0, v - info.get("lo", v)))
        highs.append(max(0.0, info.get("hi", v) - v))
    ymin, ymax = (1.0, 5.0) if eval_name in SCALE_1_5_EVALS else (0.0, 1.05)
    chance = CHANCE_BASELINES.get(eval_name)
    if chance is not None:
        ax.axhline(chance, color="#b23b3b", linestyle="--", linewidth=1.0, alpha=0.85)
    ax.bar(x, raw_vals, width=0.78, color=bar_colors, edgecolor="white", linewidth=0.5,
           yerr=np.array([lows, highs]), capsize=3)
    best_idx = int(np.argmax(raw_vals))
    for i, v in enumerate(raw_vals):
        ax.text(i, v + 0.012 * (ymax - ymin), f"{v:.3f}",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold" if i == best_idx else "normal",
                color="#1a4d1a" if i == best_idx else "#444444")
    ax.set_title(f"{title}\n[{metric_lbl}]", fontsize=10.5, pad=6)
    ax.set_xticks(x); ax.set_xticklabels([f"L{L}" for L in LAYERS], fontsize=9)
    ax.set_xlim(-0.6, len(LAYERS) - 0.4)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    if (panel_idx) % n_cols == 0:
        ax.set_ylabel("Score (absolute)", fontsize=10)

for extra in axes_flat[n_panels:]:
    extra.axis("off")

legend_handles = [
    mpatches.Patch(facecolor=bar_colors[i], edgecolor="white", linewidth=0.5, label=display_labels[i])
    for i in range(len(LAYERS))
] + [
    mpatches.Patch(facecolor="none", edgecolor="#b23b3b", linewidth=1.0, label="chance (0.5)", linestyle="--"),
    mpatches.Patch(facecolor="none", edgecolor="#666666", linewidth=1.0, label="baseline (0)", linestyle="--"),
]
fig.legend(
    handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.06),
    ncol=6, fontsize=11, frameon=False, columnspacing=1.4, handlelength=2.0,
)
fig.text(
    0.5, 0.012,
    "Absolute AObench scores. Higher is better for every panel.\n"
    "Error bars: 95% bootstrap CI from AObench's eval_intervals.json. Overall panel uses chance-adjusted normalized aggregate.",
    ha="center", va="bottom", fontsize=9.5, color="#333333",
)
fig.suptitle(
    "Qwen3-8B Activation Oracle layer sweep — absolute AObench scores (L20/L21/L22/L23)",
    fontsize=15, y=0.97,
)
fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.14, hspace=0.55, wspace=0.22)

out_path = ROOT / "comparison_absolute.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"wrote {out_path}")

# Tabular printout
print("\n== Absolute primary-metric scores per layer ==")
header = f"{'eval':<28}" + "  ".join(f"{f'L{L}':>10}" for L in LAYERS) + "  chance"
print(header)
print("-" * len(header))
for ev in eval_names:
    cells = []
    for L in LAYERS:
        v = data.get(L, {}).get(ev, {}).get("mean")
        cells.append(f"{v:>10.3f}" if v is not None else f"{'n/a':>10}")
    chance = CHANCE_BASELINES.get(ev)
    chance_str = f"  {chance:.2f}" if chance is not None else "    -"
    print(f"{ev:<28}" + "  ".join(cells) + chance_str)

print("\n== Chance-adjusted overall mean ==")
for L in LAYERS:
    a = agg[L]
    print(f"  L{L} ({DEPTH_PCT[L]}%): mean={a['mean_normalized_score']:+.4f}  CI=[{a['ci_lo']:+.4f}, {a['ci_hi']:+.4f}]")
