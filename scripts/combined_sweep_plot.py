"""Combined plot of both sweeps across all 8 unique layers.

v1 sweep (only normalized aggregate JSONs locally, no per-eval bootstrap CIs):
    /home/celeste/shared/ao_layer_sweep/aggregate_L{19,22,25,28,31}.json
v2 sweep (full per-layer eval dirs with eval_intervals.json):
    /home/celeste/shared/ao_layer_sweep_v2/L{20,21,22,23}/
"""
import json
import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

V1 = Path("/home/celeste/shared/ao_layer_sweep")
V2 = Path("/home/celeste/shared/ao_layer_sweep_v2")

# (layer, source_label) -> normalized aggregate dict
points: list[tuple[int, str, dict]] = []
for L in [19, 25, 28, 31]:
    p = V1 / f"aggregate_L{L}.json"
    if p.exists():
        points.append((L, "v1", json.loads(p.read_text())["final"]))
# L22 has two runs — show both
for L in [22]:
    p1 = V1 / f"aggregate_L{L}.json"
    if p1.exists():
        points.append((L, "v1", json.loads(p1.read_text())["final"]))
for L in [20, 21, 22, 23]:
    p = V2 / f"L{L}" / "report" / "aggregate_scores.json"
    if p.exists():
        points.append((L, "v2", json.loads(p.read_text())["final"]))

points.sort(key=lambda t: (t[0], t[1]))

# x positions: actual depth percent (one per layer; L22 gets two markers slightly offset)
DEPTH_PCT = {19: 53, 20: 56, 21: 59, 22: 62, 23: 64, 25: 70, 28: 78, 31: 87}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [1, 1.4]})

# Top: chance-adjusted overall mean vs depth, with CI
v1_pts = [(p[0], p[2]) for p in points if p[1] == "v1"]
v2_pts = [(p[0], p[2]) for p in points if p[1] == "v2"]


def scatter_with_ci(ax, pts, color, label, marker, x_offset=0.0):
    if not pts:
        return
    xs = [DEPTH_PCT[L] + x_offset for L, _ in pts]
    ms = [d["mean_normalized_score"] for _, d in pts]
    lo = [m - d["ci_lo"] for (_, d), m in zip(pts, ms)]
    hi = [d["ci_hi"] - m for (_, d), m in zip(pts, ms)]
    ax.errorbar(xs, ms, yerr=[lo, hi], fmt=marker, color=color,
                capsize=5, ms=10, lw=2, label=label, mfc=color, mec="black", mew=0.6)
    for L, d in pts:
        ax.annotate(f"L{L}", (DEPTH_PCT[L] + x_offset, d["mean_normalized_score"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=10, fontweight="bold")


scatter_with_ci(ax1, v1_pts, "#4878d0", "v1 sweep (L19/22/25/28/31)", "o", x_offset=-0.4)
scatter_with_ci(ax1, v2_pts, "#ee854a", "v2 sweep (L20/21/22/23)",    "s", x_offset=+0.4)
ax1.axhline(0, color="gray", ls="--", lw=1, alpha=0.7)
ax1.set_xlabel("Layer depth (% of 36)")
ax1.set_ylabel("Mean normalized score (chance-adj.)")
ax1.set_title("Chance-adjusted overall score across both sweeps", fontweight="bold")
ax1.grid(alpha=0.3)
ax1.legend(loc="upper right")
all_pcts = sorted(set(DEPTH_PCT[p[0]] for p in points))
ax1.set_xticks(all_pcts)
ax1.set_xticklabels([f"{p}%\n(L{L})" for p in all_pcts for L in [k for k, v in DEPTH_PCT.items() if v == p][:1]])

# Bottom: per-eval grouped bars (using per_eval_normalized so v1 and v2 are comparable)
all_layers_sorted = sorted(set(L for L, _, _ in points), key=lambda L: DEPTH_PCT[L])
# For L22, prefer v2 (more recent) but mark with hatch
chosen: dict[int, tuple[str, dict]] = {}
for L, src, d in points:
    if L not in chosen or src == "v2":
        chosen[L] = (src, d)

eval_names = sorted({e for _, d in chosen.values() for e in d["per_eval_normalized"]})
n_layers = len(all_layers_sorted)
bar_w = 0.10
x = np.arange(len(eval_names))
colors = plt.cm.viridis(np.linspace(0.10, 0.92, n_layers))
for i, L in enumerate(all_layers_sorted):
    src, d = chosen[L]
    vals = [d["per_eval_normalized"].get(e, 0.0) for e in eval_names]
    offset = (i - (n_layers - 1) / 2) * bar_w
    hatch = "//" if src == "v1" else None
    ax2.bar(x + offset, vals, bar_w, color=colors[i], hatch=hatch,
            edgecolor="black", linewidth=0.4,
            label=f"L{L} ({DEPTH_PCT[L]}%, {src})")
ax2.axhline(0, color="black", lw=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels([e.replace("_", "\n") for e in eval_names], fontsize=9)
ax2.set_ylabel("Normalized score (chance-adj.)")
ax2.set_title("Per-eval normalized score across all layers (v1 = hatched, v2 = solid)", fontweight="bold")
ax2.legend(ncol=8, loc="upper left", fontsize=8.5, frameon=False, columnspacing=1.0)
ax2.grid(axis="y", alpha=0.3)

plt.suptitle("Qwen3-8B Activation Oracle — combined layer sweep (L19/L20/L21/L22/L23/L25/L28/L31)",
             fontsize=15, y=1.0)
plt.tight_layout()

out = Path("/home/celeste/shared/combined_layer_sweep.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")

# Print a combined table
print("\n== Combined chance-adjusted overall (mean_normalized_score) ==")
print(f"{'Layer':<8}{'depth':>8}{'sweep':>8}{'mean':>10}{'CI':>22}")
for L, src, d in points:
    print(f"L{L:<7}{DEPTH_PCT[L]}%{'':<3}{src:>8}{d['mean_normalized_score']:>+10.4f}  [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]")
