"""Big leaderboard heatmap with descriptive names (no A/B/C codenames)."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SHARED = Path("/home/celeste/shared")

# (descriptive_label, path, group)
sources = []
def add(label, path, group):
    sources.append((label, Path(path), group))

# Baseline
add("3-layer [21,22,23] · 80k · cot-v5 past_lens · on-policy w/ inject",
    SHARED/"multi_layer_L21_22_23/report/aggregate_scores.json", "baseline")

# Phase 1+2: past_lens mode ablations on 3-layer base
add("3-layer · no sys-prompt inject (sp=0)",                 SHARED/"abl_results/A_offpolicy/report/aggregate_scores.json", "past_lens modes")
add("3-layer · past direction only (off-policy corpus)",     SHARED/"abl_results/B_pastonly/report/aggregate_scores.json", "past_lens modes")
add("3-layer · past_lens corpus = FineFineWeb",              SHARED/"abl_results/C_finefineweb/report/aggregate_scores.json", "past_lens modes")
add("3-layer · past_lens corpus = fineweb (raw web)",        SHARED/"abl_results/D_fineweb/report/aggregate_scores.json", "past_lens modes")
add("3-layer · past direction only · on-policy vLLM",        SHARED/"abl_results/E_past_vllm/report/aggregate_scores.json", "past_lens modes")
add("3-layer · future direction only · raw corpus (no vLLM)",SHARED/"abl_results/F_future_corpus/report/aggregate_scores.json", "past_lens modes")
add("3-layer · future direction only · on-policy no inject", SHARED/"abl_results/G_future_vllm_noinject/report/aggregate_scores.json", "past_lens modes")
add("3-layer · future direction only · on-policy + inject",  SHARED/"abl_results/H_future_vllm_inject/report/aggregate_scores.json", "past_lens modes")

# Phase 3: layer/scale
add("5-layer [21..25] · 80k",                                SHARED/"phase3_results/I_5layer_21_25/report/aggregate_scores.json", "layer/scale")
add("4-layer [19,21,23,25] interleaved · 80k",               SHARED/"phase3_results/J_4layer_19_21_23_25/report/aggregate_scores.json", "layer/scale")
add("1-layer [23] · 80k",                                    SHARED/"phase3_results/K_single_L23/report/aggregate_scores.json", "layer/scale")
add("3-layer · 160k examples (2× budget)",                   SHARED/"phase3_results/L_more_train_160k/report/aggregate_scores.json", "layer/scale")
add("3-layer · 80k past_lens entries (2× past_lens)",        SHARED/"phase3_results/M_pastlens_80k/report/aggregate_scores.json", "layer/scale")
add("3-layer · lora_r=128 (2× LoRA capacity)",               SHARED/"phase3_results/N_lora_r128/report/aggregate_scores.json", "layer/scale")

# Phase 4: latentqa
add("3-layer · + 60k LatentQA · 80k budget",                 SHARED/"phase4_results/O_with_latentqa_60k/report/aggregate_scores.json", "+ LatentQA")
add("3-layer · + 60k LatentQA · 120k budget",                SHARED/"phase4_results/P_with_latentqa_120k_budget/report/aggregate_scores.json", "+ LatentQA")
add("3-layer · + 80k LatentQA · no classification",          SHARED/"phase4_results/Q_no_classification_latentqa_heavy/report/aggregate_scores.json", "+ LatentQA")

# Phase 5: stacking
add("5-layer · 160k examples · bs=16",                       SHARED/"phase5_results/R_5layer_160k/report/aggregate_scores.json", "stacked")
add("5-layer · 60k LatentQA · 80k budget",                   SHARED/"phase5_results/S_5layer_latentqa/report/aggregate_scores.json", "stacked")
add("5-layer · 60k LatentQA · 160k budget",                  SHARED/"phase5_results/T_5layer_latentqa_160k/report/aggregate_scores.json", "stacked")
add("5-layer · lr=5e-5 (vs default 3e-5)",                   SHARED/"phase5_results/U_5layer_lr5em5/report/aggregate_scores.json", "stacked")

# Phase 6: push
add("7-layer [21..27] · 160k",                               SHARED/"phase6_results/V_7layer_160k/report/aggregate_scores.json", "push")
add("5-layer · 320k examples",                               SHARED/"phase6_results/W_5layer_320k/report/aggregate_scores.json", "push")
add("5-layer · 160k × 2 epochs",                             SHARED/"phase6_results/X_5layer_160k_2ep/report/aggregate_scores.json", "push")
add("5-layer · 160k + 30k LatentQA",                         SHARED/"phase6_results/Y_5layer_160k_latentqa30k/report/aggregate_scores.json", "push")
add("5-layer · 160k · bs=32 (vs bs=16)",                     SHARED/"phase6_results/Z_5layer_160k_bs32/report/aggregate_scores.json", "push")

# Phase 7: scale/combine
add("5-layer · 480k examples",                               SHARED/"phase7_results/AA_5layer_480k/report/aggregate_scores.json", "scale/combine")
add("5-layer · 320k + 30k LatentQA",                         SHARED/"phase7_results/BB_5layer_320k_lqa30k/report/aggregate_scores.json", "scale/combine")
add("6-layer [21..26] · 320k",                               SHARED/"phase7_results/CC_6layer_320k/report/aggregate_scores.json", "scale/combine")

rows = []
all_evals = set()
for label, p, grp in sources:
    if not p.exists(): continue
    a = json.loads(p.read_text())["final"]
    per = a.get("per_eval_normalized", {})
    rows.append({"label": label, "group": grp, "mean": a["mean_normalized_score"],
                 "lo": a["ci_lo"], "hi": a["ci_hi"], "per": per})
    all_evals.update(per.keys())

EVAL_ORDER = ["number_prediction","mmlu_prediction","missing_info","sycophancy",
              "backtracking","vagueness","domain_confusion","activation_sensitivity",
              "hallucination","taboo","personaqa","system_prompt_qa_hidden","system_prompt_qa_latentqa"]
evals = [e for e in EVAL_ORDER if e in all_evals]

rows.sort(key=lambda r: -r["mean"])
labels = [r["label"] for r in rows]
means = np.array([r["mean"] for r in rows])
mat = np.array([[r["per"].get(e, np.nan) for e in evals] for r in rows])

fig, (ax_mean, ax_heat) = plt.subplots(
    1, 2, figsize=(max(15, 1.0 * len(evals) + 8), max(8, 0.34 * len(rows))),
    gridspec_kw={"width_ratios": [1.0, max(5, 0.55 * len(evals))]},
)

group_colors = {
    "baseline":          "#444444",
    "past_lens modes":   "#7e9bd6",
    "layer/scale":       "#5fbf7f",
    "+ LatentQA":        "#e0a87f",
    "stacked":           "#bf6f99",
    "push":              "#d44b4b",
    "scale/combine":     "#9b6dc4",
}
colors = [group_colors.get(r["group"], "#888") for r in rows]
y = np.arange(len(rows))
lo = means - np.array([r["lo"] for r in rows])
hi = np.array([r["hi"] for r in rows]) - means
ax_mean.barh(y, means, xerr=[lo, hi], color=colors, edgecolor="black",
             linewidth=0.3, capsize=2, error_kw={"elinewidth": 0.7})
for i, m in enumerate(means):
    weight = "bold" if m == means.max() else "normal"
    ax_mean.text(m + 0.005, i, f"{m:+.3f}", va="center", fontsize=8, fontweight=weight)
ax_mean.set_yticks(y); ax_mean.set_yticklabels(labels, fontsize=8.5)
ax_mean.invert_yaxis()
ax_mean.axvline(0, color="gray", ls="--", lw=0.8)
ax_mean.set_xlabel("Chance-adj mean")
ax_mean.set_title("Overall AObench score", fontweight="bold", fontsize=11)
ax_mean.grid(axis="x", alpha=0.3)
ax_mean.set_xlim(-0.05, 0.45)

absmax = float(max(abs(np.nanmin(mat)), abs(np.nanmax(mat))))
im = ax_heat.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-absmax, vmax=absmax)
ax_heat.set_xticks(np.arange(len(evals)))
ax_heat.set_xticklabels([e.replace("_", "\n") for e in evals], fontsize=9, rotation=0)
ax_heat.set_yticks(y); ax_heat.set_yticklabels([])
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        if np.isnan(v): continue
        bright = abs(v) > 0.35
        ax_heat.text(j, i, f"{v:+.2f}", ha="center", va="center",
                     fontsize=7, color=("white" if bright else "black"))
for k in ("taboo", "personaqa", "system_prompt_qa_hidden", "system_prompt_qa_latentqa"):
    if k in evals:
        j = evals.index(k)
        ax_heat.add_patch(plt.Rectangle((j-0.5, -0.5), 1, len(rows),
                                         fill=False, edgecolor="#1565c0", linewidth=1.5))
ax_heat.set_title("Per-eval normalized score (RdYlGn) — blue boxes = model-organism evals",
                   fontweight="bold", fontsize=11)
plt.colorbar(im, ax=ax_heat, shrink=0.6, label="normalized score")

handles = [mpatches.Patch(facecolor=c, edgecolor="black", linewidth=0.3, label=g)
           for g, c in group_colors.items() if any(r["group"] == g for r in rows)]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01),
           ncol=len(handles), fontsize=10, frameon=False)
fig.suptitle("Qwen3-8B Activation Oracle — all runs through Phase 7", fontsize=13, fontweight="bold", y=1.008)
plt.tight_layout()

out = SHARED / "ao_paper_ablations" / "07_full_leaderboard" / "full_aobench_heatmap_descriptive.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}  rows={len(rows)} evals={len(evals)}")
