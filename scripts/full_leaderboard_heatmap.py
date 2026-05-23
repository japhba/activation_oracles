"""Render the full AObench leaderboard across ALL evals as a heatmap."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SHARED = Path("/home/celeste/shared")

sources = []
def add(label, path, group):
    sources.append((label, Path(path), group))

add("baseline multi L21/22/23", SHARED/"multi_layer_L21_22_23/report/aggregate_scores.json", "baseline")
for tag in ["A_offpolicy","B_pastonly","C_finefineweb","D_fineweb",
            "E_past_vllm","F_future_corpus","G_future_vllm_noinject","H_future_vllm_inject"]:
    add(tag, SHARED/f"abl_results/{tag}/report/aggregate_scores.json", "p1+2 modes")
for tag in ["I_5layer_21_25","J_4layer_19_21_23_25","K_single_L23",
            "L_more_train_160k","M_pastlens_80k","N_lora_r128"]:
    add(tag, SHARED/f"phase3_results/{tag}/report/aggregate_scores.json", "p3 layer/scale")
for tag in ["O_with_latentqa_60k","P_with_latentqa_120k_budget","Q_no_classification_latentqa_heavy"]:
    add(tag, SHARED/f"phase4_results/{tag}/report/aggregate_scores.json", "p4 latentqa")
for tag in ["R_5layer_160k","S_5layer_latentqa","T_5layer_latentqa_160k","U_5layer_lr5em5"]:
    add(tag, SHARED/f"phase5_results/{tag}/report/aggregate_scores.json", "p5 stacked")
for tag in ["V_7layer_160k","W_5layer_320k","X_5layer_160k_2ep","Y_5layer_160k_latentqa30k","Z_5layer_160k_bs32"]:
    add(tag, SHARED/f"phase6_results/{tag}/report/aggregate_scores.json", "p6 push")

rows = []
all_evals = set()
for label, p, grp in sources:
    if not p.exists(): continue
    a = json.loads(p.read_text())["final"]
    per = a.get("per_eval_normalized", {})
    rows.append({"label": label, "group": grp, "mean": a["mean_normalized_score"],
                 "lo": a["ci_lo"], "hi": a["ci_hi"], "per": per})
    all_evals.update(per.keys())

# Stable eval ordering — group judge/binary/etc together
EVAL_ORDER = [
    "number_prediction",
    "mmlu_prediction",
    "missing_info",
    "sycophancy",
    "backtracking",
    "vagueness",
    "domain_confusion",
    "activation_sensitivity",
    "hallucination",
    "taboo",
    "personaqa",
    "system_prompt_qa_hidden",
    "system_prompt_qa_latentqa",
]
evals = [e for e in EVAL_ORDER if e in all_evals]

rows.sort(key=lambda r: -r["mean"])
labels = [r["label"] for r in rows]
means = np.array([r["mean"] for r in rows])
mat = np.array([[r["per"].get(e, np.nan) for e in evals] for r in rows])  # rows x evals

# Layout: heatmap + side overall-score bar
fig, (ax_mean, ax_heat) = plt.subplots(1, 2, figsize=(max(11, 0.8 * len(evals) + 3.5), max(7, 0.32 * len(rows))),
                                       gridspec_kw={"width_ratios": [1.0, max(5, 0.55 * len(evals))]})

# Group color strip on left of mean bars
group_colors = {
    "baseline": "#444444",
    "p1+2 modes": "#7e9bd6",
    "p3 layer/scale": "#5fbf7f",
    "p4 latentqa": "#e0a87f",
    "p5 stacked": "#bf6f99",
    "p6 push": "#d44b4b",
}
colors = [group_colors.get(r["group"], "#888") for r in rows]
y = np.arange(len(rows))
lo = means - np.array([r["lo"] for r in rows])
hi = np.array([r["hi"] for r in rows]) - means
ax_mean.barh(y, means, xerr=[lo, hi], color=colors, edgecolor="black", linewidth=0.3, capsize=2, error_kw={"elinewidth": 0.7})
for i, m in enumerate(means):
    weight = "bold" if m == means.max() else "normal"
    ax_mean.text(m + 0.005, i, f"{m:+.3f}", va="center", fontsize=8, fontweight=weight)
ax_mean.set_yticks(y); ax_mean.set_yticklabels(labels, fontsize=8)
ax_mean.invert_yaxis()
ax_mean.axvline(0, color="gray", ls="--", lw=0.8)
ax_mean.set_xlabel("Chance-adj mean")
ax_mean.set_title("Overall AObench score", fontweight="bold", fontsize=10)
ax_mean.grid(axis="x", alpha=0.3)
ax_mean.set_xlim(-0.05, 0.45)

# Heatmap of per-eval scores
vmin = float(np.nanmin(mat))
vmax = float(np.nanmax(mat))
absmax = max(abs(vmin), abs(vmax))
im = ax_heat.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-absmax, vmax=absmax)
ax_heat.set_xticks(np.arange(len(evals)))
ax_heat.set_xticklabels([e.replace("_", "\n") for e in evals], fontsize=8, rotation=0)
ax_heat.set_yticks(y); ax_heat.set_yticklabels([])
# annotate cells
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        if np.isnan(v): continue
        # white text on dark cells, black otherwise
        bright = abs(v) > 0.35
        ax_heat.text(j, i, f"{v:+.2f}", ha="center", va="center",
                     fontsize=6.5, color=("white" if bright else "black"))
# emphasize model-org cols
for k in ("taboo", "personaqa", "system_prompt_qa_hidden", "system_prompt_qa_latentqa"):
    if k in evals:
        j = evals.index(k)
        ax_heat.add_patch(plt.Rectangle((j-0.5, -0.5), 1, len(rows), fill=False, edgecolor="#1565c0", linewidth=1.4))
ax_heat.set_title("Per-eval normalized score (RdYlGn) — blue boxes = model-org evals", fontweight="bold", fontsize=10)
plt.colorbar(im, ax=ax_heat, shrink=0.6, label="normalized score")

handles = [mpatches.Patch(facecolor=c, edgecolor="black", linewidth=0.3, label=g) for g, c in group_colors.items()]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=len(group_colors), fontsize=9, frameon=False)
fig.suptitle("Qwen3-8B Activation Oracle — full AObench leaderboard (all 11 evals)", fontsize=12, fontweight="bold", y=1.005)
plt.tight_layout()

out = SHARED / "full_aobench_heatmap.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}  rows={len(rows)} evals={len(evals)}")
