"""All the comparison plots requested for the paper / writeup.

Each plot is one specific claim. Bars compare matched conditions.
Legends show exact example counts (and target tokens where known)."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

SHARED = Path("/home/celeste/shared")
OUT = SHARED / "ao_paper_ablations"
OUT.mkdir(parents=True, exist_ok=True)

# Approximate chance baselines for per-eval dotted lines (per AObench audit)
# Where the eval is already chance-corrected (mmlu/missing_info/sycophancy), baseline is 0.
# For 1-5 rescaled evals (backtracking, sys_prompt_qa_*), baseline is 0 (=score of 1).
# For others we approximate the random-judge / random-guess level.
CHANCE_LEVEL = {
    "mmlu_prediction": 0.0,  # AUC=0.5 → 0 after chance correction
    "missing_info": 0.0,
    "sycophancy": 0.0,
    "backtracking": 0.0,
    "system_prompt_qa_hidden": 0.0,
    "system_prompt_qa_latentqa": 0.0,
    "number_prediction": 0.0,  # random integer guess almost never matches
    "vagueness": 0.25,  # ~4 judge buckets
    "domain_confusion": 0.25,
    "activation_sensitivity": 0.33,  # 3 judge categories
    "hallucination": 0.05,  # most random outputs are obviously wrong
    "taboo": 0.10,  # ~10 taboo words to guess from
    "personaqa": 0.05,
}


def load(label_paths):
    """label_paths: list of (label, json_path) → return list of dicts."""
    rows = []
    for label, path in label_paths:
        p = Path(path)
        if not p.exists():
            print(f"[skip] {label}: missing {p}")
            continue
        a = json.loads(p.read_text())["final"]
        rows.append({
            "label": label,
            "mean": a["mean_normalized_score"],
            "lo": a["ci_lo"], "hi": a["ci_hi"],
            "per": a.get("per_eval_normalized", {}),
        })
    return rows


def annotated_bars(rows, ax, key="mean", label_fmt="{:+.3f}", color_func=None):
    n = len(rows)
    x = np.arange(n)
    vals = [r[key] for r in rows]
    if key == "mean":
        lo = [r["mean"] - r["lo"] for r in rows]
        hi = [r["hi"] - r["mean"] for r in rows]
        yerr = np.array([lo, hi])
    else:
        yerr = None
    colors = [color_func(i) for i in range(n)] if color_func else None
    ax.bar(x, vals, yerr=yerr, capsize=5, color=colors, edgecolor="black", linewidth=0.5,
           error_kw={"elinewidth": 1.0})
    mx = max(vals)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.005 if v >= 0 else -0.012),
                label_fmt.format(v), ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=10, fontweight="bold" if v == mx else "normal")
    ax.set_xticks(x)
    ax.axhline(0, color="gray", ls="--", lw=0.8)


# ------------------------------------------------------------
# Plot 1: past+future > past > future (direction ablation)
# ------------------------------------------------------------
rows = load([
    ("past + future\n(both directions, on-policy)\n40k past_lens entries", SHARED/"multi_layer_L21_22_23/report/aggregate_scores.json"),
    ("past only\n(off-policy corpus text)\n40k past_lens entries",         SHARED/"abl_results/B_pastonly/report/aggregate_scores.json"),
    ("future only\n(off-policy corpus next-k)\n40k past_lens entries",     SHARED/"abl_results/F_future_corpus/report/aggregate_scores.json"),
])
fig, ax = plt.subplots(figsize=(8, 5.5))
annotated_bars(rows, ax, color_func=lambda i: ["#4a72b8", "#7fa84a", "#b8784a"][i])
ax.set_xticklabels([r["label"] for r in rows], fontsize=9.5)
ax.set_ylabel("Chance-adjusted overall AObench score")
ax.set_title("Direction ablation (token-matched at 40k past_lens entries)\n"
             "multi-layer [21,22,23] · rsLoRA · lr=3e-5 · cot-v5 corpus", fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(r["hi"] for r in rows) + 0.03)
plt.tight_layout()
plt.savefig(OUT / "01_direction" / "p1_direction_ablation.png", dpi=140, bbox_inches="tight")
print("wrote p1_direction_ablation.png")
plt.close()


# ------------------------------------------------------------
# Plot 2: corpus ablation cot-v5 > FineFineWeb > fineweb
# ------------------------------------------------------------
rows = load([
    ("cot-oracle-corpus-v5\n(CoT teacher outputs)\n40k entries · ~2k ctx", SHARED/"multi_layer_L21_22_23/report/aggregate_scores.json"),
    ("FineFineWeb\n(m-a-p curated web)\n40k entries · ~2k ctx",            SHARED/"abl_results/C_finefineweb/report/aggregate_scores.json"),
    ("fineweb\n(HuggingFaceFW raw web)\n40k entries · ~2k ctx",            SHARED/"abl_results/D_fineweb/report/aggregate_scores.json"),
])
fig, ax = plt.subplots(figsize=(8, 5.5))
annotated_bars(rows, ax, color_func=lambda i: ["#4a72b8", "#aa7ab8", "#b8784a"][i])
ax.set_xticklabels([r["label"] for r in rows], fontsize=9.5)
ax.set_ylabel("Chance-adjusted overall AObench score")
ax.set_title("past_lens corpus ablation (token-matched, only `pretrain_dataset` differs)\n"
             "multi-layer [21,22,23] · rsLoRA · lr=3e-5 · on-policy w/ 50% sys-prompt inject", fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(r["hi"] for r in rows) + 0.03)
plt.tight_layout()
plt.savefig(OUT / "02_corpus" / "p2_corpus_ablation.png", dpi=140, bbox_inches="tight")
print("wrote p2_corpus_ablation.png")
plt.close()


# ------------------------------------------------------------
# Plot 3: # layers vs overall + taboo + personaqa (the key model-org claim)
# ------------------------------------------------------------
rows = load([
    ("1 layer [23]\n(80k examples)",                  SHARED/"phase3_results/K_single_L23/report/aggregate_scores.json"),
    ("3 layers [21,22,23]\n(80k examples)",           SHARED/"multi_layer_L21_22_23/report/aggregate_scores.json"),
    ("5 layers [21..25]\n(80k examples)",             SHARED/"phase3_results/I_5layer_21_25/report/aggregate_scores.json"),
    ("7 layers [21..27]\n(160k examples)",            SHARED/"phase6_results/V_7layer_160k/report/aggregate_scores.json"),
])
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
metrics = [("mean", "Overall mean", lambda r: r["mean"], 0.0),
           ("taboo", "Taboo (model-org)", lambda r: r["per"].get("taboo", 0), CHANCE_LEVEL["taboo"]),
           ("personaqa", "PersonaQA (model-org)", lambda r: r["per"].get("personaqa", 0), CHANCE_LEVEL["personaqa"])]
n_layers = [1, 3, 5, 7]
for ax, (key, title, getter, chance) in zip(axes, metrics):
    vals = [getter(r) for r in rows]
    ax.bar(np.arange(len(rows)), vals,
           color=["#7e9bd6", "#5fbf7f", "#e0a87f", "#d44b4b"],
           edgecolor="black", linewidth=0.5)
    mx = max(vals)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:+.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold" if v == mx else "normal")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([f"{n}L" for n in n_layers], fontsize=10)
    ax.set_xlabel("# activation layers fed to AO")
    ax.set_title(title, fontweight="bold")
    if chance > 0:
        ax.axhline(chance, color="#b23b3b", ls=":", lw=1.4, alpha=0.85,
                   label=f"chance ≈ {chance:.2f}")
        ax.legend(loc="upper left", fontsize=8.5)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("Chance-adjusted score")
fig.suptitle("More layers help everything but disproportionately help model-org evals\n"
             "Same recipe (rsLoRA + lr=3e-5 + cot-v5 past_lens + 80k examples — except 7L is 160k for compute reasons)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "03_layer_count" / "p3_layer_count_modelorg.png", dpi=140, bbox_inches="tight")
print("wrote p3_layer_count_modelorg.png")
plt.close()


# ------------------------------------------------------------
# Plot 4: chunked-convqa > LatentQA (best existing evidence; strict swap pending)
# ------------------------------------------------------------
rows = load([
    ("no LatentQA\n(W_5layer_320k)\n320k total, 80k haiku, 40k past_lens",   SHARED/"phase6_results/W_5layer_320k/report/aggregate_scores.json"),
    ("+ 30k LatentQA\n(Y_5layer_160k_latentqa30k)\n160k total, 30k LatentQA", SHARED/"phase6_results/Y_5layer_160k_latentqa30k/report/aggregate_scores.json"),
    ("+ 60k LatentQA\n(T_5layer_latentqa_160k)\n160k total, 60k LatentQA",    SHARED/"phase5_results/T_5layer_latentqa_160k/report/aggregate_scores.json"),
])
fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
for ax, (key, title, getter, chance) in zip(axes, metrics):
    vals = [getter(r) for r in rows]
    ax.bar(np.arange(len(rows)), vals, color=["#4a72b8", "#e0a87f", "#d4634b"],
           edgecolor="black", linewidth=0.5)
    mx = max(vals)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:+.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold" if v == mx else "normal")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([r["label"] for r in rows], fontsize=8.5)
    ax.set_title(title, fontweight="bold")
    if chance > 0:
        ax.axhline(chance, color="#b23b3b", ls=":", lw=1.4, alpha=0.85,
                   label=f"chance ≈ {chance:.2f}")
        ax.legend(loc="upper left", fontsize=8.5)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("Chance-adjusted score")
fig.suptitle("Adding LatentQA hurts overall, helps personaqa slightly\n"
             "Caveat: NOT strictly token-matched (strict swap GG vs HH pending)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "04_latentqa_vs_chunked" / "p4_latentqa_vs_chunked.png", dpi=140, bbox_inches="tight")
print("wrote p4_latentqa_vs_chunked.png")
plt.close()


# ------------------------------------------------------------
# Plot 5: scaling with training data (5-layer recipe, varying max_train_examples)
# ------------------------------------------------------------
rows_data = [
    ("80k\n(I_5layer_21_25)",  SHARED/"phase3_results/I_5layer_21_25/report/aggregate_scores.json",  80_000),
    ("160k\n(R_5layer_160k)",  SHARED/"phase5_results/R_5layer_160k/report/aggregate_scores.json",   160_000),
    ("320k\n(W_5layer_320k)",  SHARED/"phase6_results/W_5layer_320k/report/aggregate_scores.json",   320_000),
]
rows = []
for label, path, n in rows_data:
    p = Path(path)
    if not p.exists():
        continue
    a = json.loads(p.read_text())["final"]
    rows.append({"label": label, "n": n, "mean": a["mean_normalized_score"],
                 "lo": a["ci_lo"], "hi": a["ci_hi"], "per": a.get("per_eval_normalized", {})})

fig, ax = plt.subplots(figsize=(8, 5.5))
xs = [r["n"] for r in rows]
means = [r["mean"] for r in rows]
lo = [r["mean"] - r["lo"] for r in rows]
hi = [r["hi"] - r["mean"] for r in rows]
ax.errorbar(xs, means, yerr=[lo, hi], fmt="o-", capsize=6, lw=2, ms=11, color="#4a72b8", label="overall mean")
# also plot taboo + personaqa
ax.plot(xs, [r["per"].get("taboo",0) for r in rows], "s--", color="#bf6f99", label="taboo")
ax.plot(xs, [r["per"].get("personaqa",0) for r in rows], "^--", color="#5fbf7f", label="personaqa")
for r in rows:
    ax.annotate(f"{r['mean']:+.3f}", (r["n"], r["mean"]), textcoords="offset points", xytext=(10, 8), fontsize=10, fontweight="bold")
ax.set_xscale("log")
ax.set_xticks(xs)
ax.set_xticklabels([f"{n//1000}k" for n in xs])
ax.set_xlabel("max_train_examples (5-layer recipe, log scale)")
ax.set_ylabel("Chance-adjusted score")
ax.set_title("Scaling with training data — 5-layer recipe, all else equal\n"
             "rsLoRA · lr=3e-5 · cot-v5 past_lens · multi-layer [21..25]", fontweight="bold")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "05_scaling" / "p5_scaling.png", dpi=140, bbox_inches="tight")
print("wrote p5_scaling.png")
plt.close()

print()
print(f"All plots saved in: {OUT}")
