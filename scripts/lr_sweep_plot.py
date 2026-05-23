"""LR sweep plot for L22 + rsLoRA + past_lens.

Top: overall chance-adjusted mean vs LR (log-x), with 95% bootstrap CI.
Bottom: per-eval primary metric across the 4 LR points.
"""
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/celeste/shared/lr_sweep_L22_rslora")
LRS = [("1em5", 1e-5), ("3em5", 3e-5), ("1em4", 1e-4), ("3em4", 3e-4)]

# Load
data = {}
for tag, lr in LRS:
    rep = ROOT / f"lr{tag}" / "report"
    agg = json.loads((rep / "aggregate_scores.json").read_text())["final"]
    intervals = json.loads((rep / "eval_intervals.json").read_text())
    data[lr] = {"agg": agg, "intervals": {e: v.get("final", {}) for e, v in intervals.items()}}

# Old reference points (L22 from previous sweeps, vanilla LoRA, lr=2e-4, no past_lens)
ref_points = {}
for path, label in [
    (Path("/home/celeste/shared/ao_layer_sweep/aggregate_L22.json"), "L22 v1"),
    (Path("/home/celeste/shared/ao_layer_sweep_v2/L22/report/aggregate_scores.json"), "L22 v2"),
]:
    if path.exists():
        ref_points[label] = json.loads(path.read_text())["final"]

# --- Figure ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [1, 1.4]})

# Top: overall chance-adj mean vs LR (log scale)
xs = [lr for _, lr in LRS]
means = [data[lr]["agg"]["mean_normalized_score"] for lr in xs]
lo = [data[lr]["agg"]["mean_normalized_score"] - data[lr]["agg"]["ci_lo"] for lr in xs]
hi = [data[lr]["agg"]["ci_hi"] - data[lr]["agg"]["mean_normalized_score"] for lr in xs]
ax1.errorbar(xs, means, yerr=[lo, hi], fmt="o-", color="C0", capsize=6, ms=11, lw=2, label="rsLoRA + past_lens")

# Annotate each point with LR + score
for x, m in zip(xs, means):
    ax1.annotate(f"{m:+.3f}", (x, m), textcoords="offset points", xytext=(8, 10), fontsize=10, fontweight="bold")

# Reference: L22 vanilla LoRA at lr=2e-4 (the matched-equivalent x position)
matched_lr = 2e-4 / 8  # rsLoRA-equivalent of vanilla 2e-4
for label, ref in ref_points.items():
    color = "#888888" if "v1" in label else "#444444"
    marker = "s" if "v1" in label else "D"
    ax1.errorbar(
        [2e-4], [ref["mean_normalized_score"]],
        yerr=[[ref["mean_normalized_score"] - ref["ci_lo"]], [ref["ci_hi"] - ref["mean_normalized_score"]]],
        fmt=marker, color=color, capsize=5, ms=10, label=f"{label} (vanilla LoRA, no past_lens)",
        mfc=color, mec="black", mew=0.6,
    )

ax1.axhline(0, color="gray", ls="--", lw=1, alpha=0.7)
ax1.set_xscale("log")
ax1.set_xlabel("learning rate (log scale)")
ax1.set_ylabel("Mean normalized score (chance-adj.)")
ax1.set_title("L22 + rsLoRA + past_lens(CoT v5) — chance-adjusted overall vs lr", fontweight="bold")
ax1.grid(alpha=0.3, which="both")
ax1.legend(loc="lower left")
ax1.set_xticks([1e-5, 3e-5, 1e-4, 3e-4, 2e-4])
ax1.set_xticklabels(["1e-5", "3e-5", "1e-4", "3e-4", "2e-4\n(vanilla)"])

# Bottom: per-eval primary metric across 4 LR points
all_evals = sorted({e for d in data.values() for e in d["intervals"]})
n_lrs = len(LRS)
bar_w = 0.18
colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_lrs))
xidx = np.arange(len(all_evals))
for i, (tag, lr) in enumerate(LRS):
    vals = [data[lr]["intervals"].get(e, {}).get("mean", 0.0) for e in all_evals]
    ax2.bar(
        xidx + (i - (n_lrs - 1) / 2) * bar_w,
        vals,
        bar_w,
        color=colors[i],
        edgecolor="black",
        linewidth=0.4,
        label=f"lr={lr:g}",
    )

ax2.axhline(0, color="black", lw=0.7)
ax2.axhline(0.5, color="#b23b3b", ls="--", lw=0.8, alpha=0.6, label="chance baseline (AUC 0.5)")
ax2.set_xticks(xidx)
ax2.set_xticklabels([e.replace("_", "\n") for e in all_evals], fontsize=8.5)
ax2.set_ylabel("Score (absolute, primary metric)")
ax2.set_title("Per-eval absolute scores", fontweight="bold")
ax2.legend(ncol=5, fontsize=9, loc="upper left", frameon=False)
ax2.grid(axis="y", alpha=0.3)
ax2.set_ylim(0, 5.1)  # 1-5 scale evals (backtracking, sysprompt_qa) go up to 5

plt.suptitle("Qwen3-8B Activation Oracle — LR sweep on L22 with rsLoRA + past_lens(CoT v5)",
             fontsize=14, y=1.0)
plt.tight_layout()

out = Path("/home/celeste/shared/lr_sweep_L22_rslora/lr_sweep.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")

# Print table
print()
print(f"{'lr':<8}{'mean':>10}{'CI':>22}  {'best eval':>30}")
print("-" * 75)
for tag, lr in LRS:
    a = data[lr]["agg"]
    per = {e: data[lr]["intervals"].get(e, {}).get("mean", 0.0) for e in all_evals}
    best = max(per.items(), key=lambda kv: kv[1])
    print(f"{lr:<8.0e}{a['mean_normalized_score']:>+10.4f}  [{a['ci_lo']:+.4f}, {a['ci_hi']:+.4f}]  {best[0]} ({best[1]:.3f})")

print("\nReference (no past_lens, vanilla LoRA, lr=2e-4):")
for label, ref in ref_points.items():
    print(f"  {label}: mean={ref['mean_normalized_score']:+.4f}  CI=[{ref['ci_lo']:+.4f}, {ref['ci_hi']:+.4f}]")
