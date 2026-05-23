"""One-off: 4-quadrant eval comparison of AO v2 vs AO v2 + SPQA v2."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.plot_eval_summaries import (
    RESULT_DIR_PREFIX,
    find_eval_jsons,
    extract_lora_name,
    load_all_metrics,
)
from experiments.plot_eval_grouped import (
    GROUPS,
    compute_ci_for_result,
    is_1_5_group,
    FONT_SUPTITLE, FONT_TITLE, FONT_AXIS_LABEL, FONT_TICK, FONT_BAR_LABEL, FONT_LEGEND,
)

MODELS = {
    "full_mix_500k_past_lens_plus_synthetic_qa_v3": "AO v2",
    "500k_pl_31k_spqav2_199k_sqav3_126k_cls": "AO v2 + SPQA v2",
}

BAR_COLORS = ["#4363d8", "#e6194b"]


def main():
    result_dirs = [RESULT_DIR_PREFIX + k for k in MODELS]
    result_dirs = [d for d in result_dirs if Path(d).exists()]

    all_metrics = load_all_metrics(result_dirs)

    # Resolve lora names
    label_map = {}
    lora_order = []
    for key, label in MODELS.items():
        rd = RESULT_DIR_PREFIX + key
        eval_jsons = find_eval_jsons(rd)
        for json_files in eval_jsons.values():
            for jf in json_files:
                with open(jf) as f:
                    data = json.load(f)
                lora_name = extract_lora_name(data)
                if lora_name not in label_map:
                    label_map[lora_name] = label
                    lora_order.append(lora_name)
                break
            break

    # Compute CIs
    all_cis = {}
    for rd in result_dirs:
        eval_jsons = find_eval_jsons(rd)
        for eval_name, json_files in eval_jsons.items():
            for jf in json_files:
                with open(jf) as f:
                    data = json.load(f)
                lora_name = extract_lora_name(data)
                rel = jf.relative_to(Path(rd) / eval_name)
                display_name = f"{eval_name}/{'/'.join(rel.parts[:-1])}" if len(rel.parts) > 1 else eval_name

                needed = set()
                for group in GROUPS:
                    for eval_pattern, metric_key, _, _ in group["metrics"]:
                        if eval_pattern == display_name:
                            needed.add(metric_key)
                if not needed:
                    continue

                ci_dict = {}
                for mk in needed:
                    ci = compute_ci_for_result(data, mk)
                    if ci is not None:
                        ci_dict[mk] = ci
                if ci_dict:
                    if display_name not in all_cis:
                        all_cis[display_name] = {}
                    all_cis[display_name][lora_name] = ci_dict

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()

    for idx, group in enumerate(GROUPS):
        ax = axes[idx]

        x_labels = []
        values = {l: [] for l in lora_order}
        cis = {l: [] for l in lora_order}

        for eval_pattern, metric_key, label, scale in group["metrics"]:
            if eval_pattern not in all_metrics:
                continue
            eval_data = all_metrics[eval_pattern]
            has_any = any(metric_key in eval_data.get(l, {}) for l in lora_order)
            if not has_any:
                continue
            x_labels.append(label)
            ci_data = all_cis.get(eval_pattern, {})
            for lora in lora_order:
                values[lora].append(eval_data.get(lora, {}).get(metric_key))
                cis[lora].append(ci_data.get(lora, {}).get(metric_key))

        if not x_labels:
            ax.set_title(group["title"], fontsize=FONT_TITLE)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        n_metrics = len(x_labels)
        n_loras = len(lora_order)
        bar_width = 0.8 / n_loras
        x = np.arange(n_metrics)

        for i, lora in enumerate(lora_order):
            lora_vals = [v if v is not None else 0 for v in values[lora]]
            ci_bounds = cis[lora]
            offset = (i - n_loras / 2 + 0.5) * bar_width

            yerr = None
            if any(c is not None for c in ci_bounds):
                yerr_lo, yerr_hi = [], []
                for val, ci in zip(lora_vals, ci_bounds):
                    if ci and not (np.isnan(ci[0]) or np.isnan(ci[1])):
                        yerr_lo.append(max(0, val - ci[0]))
                        yerr_hi.append(max(0, ci[1] - val))
                    else:
                        yerr_lo.append(0)
                        yerr_hi.append(0)
                yerr = [yerr_lo, yerr_hi]

            bars = ax.bar(x + offset, lora_vals, bar_width * 0.9,
                          label=label_map[lora], color=BAR_COLORS[i],
                          alpha=0.85, edgecolor="white", linewidth=0.5,
                          yerr=yerr, capsize=3,
                          error_kw={"elinewidth": 1.2, "capthick": 1.2})
            for bar, val, ci in zip(bars, lora_vals, ci_bounds):
                if val > 0:
                    top = bar.get_height()
                    if ci and not np.isnan(ci[1]):
                        top = ci[1]
                    fmt = f"{val:.2f}" if val < 10 else f"{val:.1f}"
                    ax.text(bar.get_x() + bar.get_width() / 2, top + 0.01,
                            fmt, ha="center", va="bottom", fontsize=FONT_BAR_LABEL)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=FONT_TICK, ha="center")
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.set_title(group["title"], fontsize=FONT_TITLE, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        if is_1_5_group(group):
            ax.set_ylim(0, 5.0)
            ax.set_ylabel("Score (1-5)", fontsize=FONT_AXIS_LABEL)
        else:
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Score", fontsize=FONT_AXIS_LABEL)

    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=len(lora_order),
               fontsize=FONT_SUPTITLE, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle("AO v2 vs AO v2 + SPQA v2",
                 fontsize=FONT_SUPTITLE, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    os.makedirs("experiments/eval_plots", exist_ok=True)
    out = "experiments/eval_plots/spqav2_comparison.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
