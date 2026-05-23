"""Final cross-experiment summary: chance-adjusted overall scores for every
trained AO config so far.

  - L19/L25/L28/L31 single-layer, vanilla LoRA, no past_lens (v1 sweep)
  - L20/L21/L22/L23 single-layer, vanilla LoRA, no past_lens (v2 sweep)
  - L22 single-layer, rsLoRA + past_lens, lr ∈ {1e-5, 3e-5, 1e-4, 3e-4}
  - L21/L22/L23 multi-layer, rsLoRA + past_lens, lr=3e-5

All on Qwen3-8B, 80k train examples, 1 epoch.
"""
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

points: list[dict] = []

# v1 sweep
for L in [19, 25, 28, 31]:
    p = Path(f"/home/celeste/shared/ao_layer_sweep/aggregate_L{L}.json")
    if p.exists():
        d = json.loads(p.read_text())["final"]
        points.append({"label": f"L{L}", "group": "single (vanilla LoRA, lr=2e-4, no past_lens)", **d})
# v1 L22
p = Path("/home/celeste/shared/ao_layer_sweep/aggregate_L22.json")
if p.exists():
    d = json.loads(p.read_text())["final"]
    points.append({"label": "L22 (v1)", "group": "single (vanilla LoRA, lr=2e-4, no past_lens)", **d})

# v2 sweep
for L in [20, 21, 22, 23]:
    p = Path(f"/home/celeste/shared/ao_layer_sweep_v2/L{L}/report/aggregate_scores.json")
    if p.exists():
        d = json.loads(p.read_text())["final"]
        suffix = " (v2)" if L == 22 else ""
        points.append({"label": f"L{L}{suffix}", "group": "single (vanilla LoRA, lr=2e-4, no past_lens)", **d})

# LR sweep on L22 with rsLoRA + past_lens
for tag, lr in [("1em5", 1e-5), ("3em5", 3e-5), ("1em4", 1e-4), ("3em4", 3e-4)]:
    p = Path(f"/home/celeste/shared/lr_sweep_L22_rslora/lr{tag}/report/aggregate_scores.json")
    if p.exists():
        d = json.loads(p.read_text())["final"]
        points.append({"label": f"L22 lr={lr:g}", "group": "single L22 (rsLoRA + past_lens)", **d})

# Multi-layer
p = Path("/home/celeste/shared/multi_layer_L21_22_23/report/aggregate_scores.json")
if p.exists():
    d = json.loads(p.read_text())["final"]
    points.append({"label": "L21·L22·L23", "group": "multi-layer (rsLoRA + past_lens, lr=3e-5)", **d})

# Sort: best first within group; group order fixed
group_order = [
    "single (vanilla LoRA, lr=2e-4, no past_lens)",
    "single L22 (rsLoRA + past_lens)",
    "multi-layer (rsLoRA + past_lens, lr=3e-5)",
]
group_color = {
    "single (vanilla LoRA, lr=2e-4, no past_lens)": "#7e9bd6",
    "single L22 (rsLoRA + past_lens)": "#f5a96a",
    "multi-layer (rsLoRA + past_lens, lr=3e-5)": "#5fbf7f",
}

# Order points: group, then by mean (ascending so best is at top of horizontal bars)
points.sort(key=lambda p: (group_order.index(p["group"]), p["mean_normalized_score"]))

fig, ax = plt.subplots(figsize=(13, 0.42 * len(points) + 2.5))
labels = [p["label"] for p in points]
means = [p["mean_normalized_score"] for p in points]
lo = [p["mean_normalized_score"] - p["ci_lo"] for p in points]
hi = [p["ci_hi"] - p["mean_normalized_score"] for p in points]
colors = [group_color[p["group"]] for p in points]
y = np.arange(len(points))

ax.barh(y, means, xerr=[lo, hi], color=colors, edgecolor="black", linewidth=0.4, capsize=4, error_kw={"elinewidth": 1.0})
for i, (m, p) in enumerate(zip(means, points)):
    ax.text(m + 0.005, i, f"{m:+.3f}", va="center", fontsize=9,
            fontweight="bold" if "Multi" in p["label"] or "21·22·23" in p["label"] else "normal")
ax.axvline(0, color="gray", ls="--", lw=1, alpha=0.7)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Mean normalized score (chance-adjusted, higher = better)")
ax.set_title("Qwen3-8B Activation Oracle — every config trained, ranked", fontweight="bold")
ax.grid(axis="x", alpha=0.3)
ax.set_xlim(min(means) - 0.05, max(means) + 0.08)

handles = [mpatches.Patch(facecolor=group_color[g], edgecolor="black", linewidth=0.4, label=g)
           for g in group_order]
ax.legend(handles=handles, loc="lower right", fontsize=9, frameon=True)

plt.tight_layout()
out = Path("/home/celeste/shared/all_runs_summary.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
print()
print("== Final ranking (chance-adj. mean) ==")
for p in sorted(points, key=lambda p: -p["mean_normalized_score"])[:10]:
    print(f"  {p['mean_normalized_score']:+.4f}  CI=[{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}]  {p['label']:<20}  ({p['group']})")
