"""Render the full leaderboard across all phases."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SHARED = Path("/home/celeste/shared")

# (label, path-glob-style, category)
sources = []

def add(label, path, group):
    sources.append((label, Path(path), group))

# Baseline / earlier sweeps
add("baseline multi L21/22/23", SHARED/"multi_layer_L21_22_23/report/aggregate_scores.json", "baseline")

# Phase 1+2 (A-H, single-mode ablations)
for tag in ["A_offpolicy","B_pastonly","C_finefineweb","D_fineweb",
            "E_past_vllm","F_future_corpus","G_future_vllm_noinject","H_future_vllm_inject"]:
    add(tag, SHARED/f"abl_results/{tag}/report/aggregate_scores.json", "p1+2 modes")

# Phase 3 (I-N: layer/scale)
for tag in ["I_5layer_21_25","J_4layer_19_21_23_25","K_single_L23",
            "L_more_train_160k","M_pastlens_80k","N_lora_r128"]:
    add(tag, SHARED/f"phase3_results/{tag}/report/aggregate_scores.json", "p3 layer/scale")

# Phase 4 (O-Q: latentqa)
for tag in ["O_with_latentqa_60k","P_with_latentqa_120k_budget","Q_no_classification_latentqa_heavy"]:
    add(tag, SHARED/f"phase4_results/{tag}/report/aggregate_scores.json", "p4 latentqa")

# Phase 5 (R-U: stacking + lr)
for tag in ["R_5layer_160k","S_5layer_latentqa","T_5layer_latentqa_160k","U_5layer_lr5em5"]:
    add(tag, SHARED/f"phase5_results/{tag}/report/aggregate_scores.json", "p5 stacked")

# Phase 6 (V-Z: more layers/data/bs)
for tag in ["V_7layer_160k","W_5layer_320k","X_5layer_160k_2ep","Y_5layer_160k_latentqa30k","Z_5layer_160k_bs32"]:
    add(tag, SHARED/f"phase6_results/{tag}/report/aggregate_scores.json", "p6 push")

rows = []
for label, p, grp in sources:
    if not p.exists():
        continue
    a = json.loads(p.read_text())["final"]
    per = a.get("per_eval_normalized", {})
    rows.append({
        "label": label, "group": grp,
        "mean": a["mean_normalized_score"], "lo": a["ci_lo"], "hi": a["ci_hi"],
        "taboo": per.get("taboo", 0), "personaqa": per.get("personaqa", 0),
    })
rows.sort(key=lambda r: -r["mean"])

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(17, max(7, len(rows) * 0.27)), gridspec_kw={"width_ratios": [3, 1.1, 1.1]})

group_colors = {
    "baseline": "#444444",
    "p1+2 modes": "#7e9bd6",
    "p3 layer/scale": "#5fbf7f",
    "p4 latentqa": "#e0a87f",
    "p5 stacked": "#bf6f99",
    "p6 push": "#d44b4b",
}

y = np.arange(len(rows))
labels = [r["label"] for r in rows]
means = [r["mean"] for r in rows]
lo = [r["mean"] - r["lo"] for r in rows]
hi = [r["hi"] - r["mean"] for r in rows]
colors = [group_colors.get(r["group"], "#888") for r in rows]

# panel 1: overall mean with CI
ax = axes[0]
ax.barh(y, means, xerr=[lo, hi], color=colors, edgecolor="black", linewidth=0.3, capsize=3, error_kw={"elinewidth": 0.8})
for i, r in enumerate(rows):
    weight = "bold" if r["mean"] == max(rr["mean"] for rr in rows) else "normal"
    ax.text(r["mean"] + 0.005, i, f"{r['mean']:+.3f}", va="center", fontsize=7.5, fontweight=weight)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.axvline(0, color="gray", ls="--", lw=0.8)
ax.set_xlabel("Chance-adjusted overall (higher = better)")
ax.set_title("Overall AObench score (mean ± 95% CI)", fontweight="bold")
ax.grid(axis="x", alpha=0.3)
ax.set_xlim(-0.05, 0.42)

# panel 2: taboo
ax = axes[1]
taboo = [r["taboo"] for r in rows]
ax.barh(y, taboo, color=colors, edgecolor="black", linewidth=0.3)
for i, v in enumerate(taboo):
    weight = "bold" if v == max(taboo) else "normal"
    ax.text(v + 0.005, i, f"{v:+.3f}", va="center", fontsize=7.5, fontweight=weight)
ax.set_yticks(y); ax.set_yticklabels([])
ax.invert_yaxis()
ax.axvline(0, color="gray", ls="--", lw=0.8)
ax.set_xlabel("taboo")
ax.set_title("Taboo (model-org eval)", fontweight="bold")
ax.grid(axis="x", alpha=0.3)
ax.set_xlim(0, 0.55)

# panel 3: personaqa
ax = axes[2]
pq = [r["personaqa"] for r in rows]
ax.barh(y, pq, color=colors, edgecolor="black", linewidth=0.3)
for i, v in enumerate(pq):
    weight = "bold" if v == max(pq) else "normal"
    ax.text(v + 0.003, i, f"{v:+.3f}", va="center", fontsize=7.5, fontweight=weight)
ax.set_yticks(y); ax.set_yticklabels([])
ax.invert_yaxis()
ax.axvline(0, color="gray", ls="--", lw=0.8)
ax.set_xlabel("personaqa")
ax.set_title("PersonaQA (model-org eval)", fontweight="bold")
ax.grid(axis="x", alpha=0.3)
ax.set_xlim(0, 0.20)

handles = [mpatches.Patch(facecolor=c, edgecolor="black", linewidth=0.3, label=g) for g, c in group_colors.items()]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(group_colors), fontsize=10, frameon=False)
fig.suptitle("Qwen3-8B Activation Oracle — all runs through Phase 6", fontsize=13, fontweight="bold", y=1.005)
plt.tight_layout()

out = Path("/home/celeste/shared/full_leaderboard_through_p6.png")
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out} ({len(rows)} runs)")
