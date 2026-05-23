# %%
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import re
import json
import numpy as np
import matplotlib.pyplot as plt

# Text sizes for plots (matching plot_classification_eval.py)
FONT_SIZE_Y_AXIS_LABEL = 16  # Y-axis labels (e.g., "Grade (1..5)")
FONT_SIZE_Y_AXIS_TICK = 16  # Y-axis tick labels (numbers on y-axis)
FONT_SIZE_X_AXIS_LABEL = 16  # X-axis labels (organism types)
FONT_SIZE_BAR_VALUE = 16  # Numbers above each bar
FONT_SIZE_LEGEND = 14  # Legend text size

CONFIG_PATH = "configs/config.yaml"

AGENT_LLM_FILTER: Optional[str] = "openai/gpt-5"  # os.environ.get("AGENT_LLM_FILTER", None)


VARIANTS: List[Tuple[str, str]] = [
    ("agent_mi0", "ADL$^{i=0}$"),
    ("agent_mi5", "ADL$^{i=5}$"),
    ("talkative_mi0", "Talkative$^{i=0}$"),
    ("talkative_mi5", "Talkative$^{i=5}$"),
    ("baseline_mi0", "Blackbox$^{i=0}$"),
    ("baseline_mi5", "Blackbox$^{i=5}$"),
    ("baseline_mi50", "Blackbox$^{i=50}$"),
]

VARIANT_COLORS: Dict[str, str] = {
    "agent_mi5": "#0569ad",
    "agent_mi0": "#59afea",
    "talkative_mi0": "#ff7f0e",
    "talkative_mi5": "#ff9933",
    "baseline_mi0": "#c7c7c7",
    "baseline_mi5": "#8f8f8f",
    "baseline_mi50": "#525252",
}

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "qwen3_1_7B": "Qwen3 1.7B",
    "qwen3_32B": "Qwen3 32B",
    "qwen3_8B": "Qwen3 8B",
    "qwen25_7B_Instruct": "Qwen2.5 7B",
    "gemma2_9B_it": "Gemma2 9B",
    "gemma3_1B": "Gemma3 1B",
    "llama31_8B_Instruct": "Llama3.1 8B",
    "llama32_1B_Instruct": "Llama3.2 1B",
    "qwen25_VL_3B_Instruct": "Qwen2.5 VL 3B",
}


def _model_display_name(model: str) -> str:
    name = MODEL_DISPLAY_NAMES.get(model, None)
    assert isinstance(name, str), f"Missing display name mapping for model: {model}"
    return name


def _results_root_for_agent_type(cfg, agent_type: str) -> Path:
    assert isinstance(agent_type, str) and len(agent_type) > 0
    method_dir = "activation_difference_lens" if agent_type in {"ADL", "ADLBlackbox"} else "talkative_probe"
    root = Path(cfg.diffing.results_dir) / method_dir
    assert root.exists() and root.is_dir(), f"Results root not found: {root}"
    return root


def _normalize_llm_id(llm_id: str) -> str:
    return str(llm_id).replace("/", "_")


def _agent_llm_filter_norm() -> Optional[str]:
    if AGENT_LLM_FILTER is None:
        return None
    val = _normalize_llm_id(AGENT_LLM_FILTER)
    assert isinstance(val, str) and len(val) > 0
    return val


def _parse_entry(entry: Tuple) -> Tuple[str, str, str, Tuple[str, str, str]]:
    assert isinstance(entry, tuple) and len(entry) >= 3
    model = str(entry[0])
    organism = str(entry[1])
    organism_type = str(entry[2])

    if len(entry) == 3:
        return (model, organism, organism_type, ("", "", ""))
    elif len(entry) == 4:
        ids = entry[3]
        assert isinstance(ids, tuple) and len(ids) == 3
        id_adl = str(ids[0]) if ids[0] is not None else ""
        id_baseline = str(ids[1]) if ids[1] is not None else ""
        id_talkative = str(ids[2]) if ids[2] is not None else ""
        return (model, organism, organism_type, (id_adl, id_baseline, id_talkative))
    else:
        raise AssertionError(f"Entry must have 3 or 4 elements, got {len(entry)}")


def _find_all_grade_paths_by_kind_and_mi(
    agent_root: Path,
    organism: str,
    model: str,
    *,
    mi: int,
    is_baseline: bool,
    llm_id_filter: Optional[str] = None,
    position: Optional[int] = None,
    agent_type: Optional[str] = None,
    run_identifier: Optional[str] = None,
) -> List[Path]:
    assert agent_root.exists() and agent_root.is_dir(), f"Agent root not found: {agent_root}"
    assert isinstance(mi, int) and mi >= 0
    agent_type_eff = str(agent_type) if agent_type is not None else ("ADLBlackbox" if is_baseline else "ADL")
    assert agent_type_eff in {"ADL", "ADLBlackbox", "TalkativeProbe"}

    def _name_matches(name: str) -> bool:
        if not isinstance(name, str) or len(name) == 0:
            return False
        base = name
        expected_prefix = f"{agent_type_eff}_"
        if agent_type_eff == "ADLBlackbox":
            expected_prefix = "Blackbox_"
        if not base.startswith(expected_prefix):
            return False
        if llm_id_filter is not None and f"_{llm_id_filter}_" not in base:
            return False
        if position is not None:
            if re.search(rf"_pos{re.escape(str(position))}(?:_|$)", base) is None:
                return False
        if re.search(rf"mi{re.escape(str(mi))}(?:_|$)", base) is None:
            return False

        if run_identifier is not None and len(run_identifier) > 0:
            if run_identifier not in base:
                return False
        else:
            if "hints" in base.lower():
                return False
        if re.search(r"__?run\d+$", base) is None:
            return False
        return True

    out: List[Path] = []
    for child in agent_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not _name_matches(name):
            continue
        grade_path = child / "hypothesis_grade.json"
        if grade_path.exists() and grade_path.is_file():
            out.append(grade_path)
    assert len(out) >= 1, (
        f"No graded outputs found in {agent_root} for organism={organism} model={model} "
        f"agent_type={agent_type_eff} mi={mi} position={position}"
    )
    return out


def _load_grade_score(json_path: Path) -> float:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert isinstance(payload, dict) and "score" in payload
    s = int(payload["score"])
    assert 1 <= s <= 6
    return float(s)


def export_full_runs_for_entries(
    entries: List[Tuple],
    *,
    export_dir: str = "export",
    config_path: str = CONFIG_PATH,
    baseline_mi_values: Optional[List[int]] = None,
    show_adl: bool = True,
    show_talkative: bool = True,
    show_blackbox: bool = True,
) -> None:
    from src.utils.interactive import load_hydra_config

    assert isinstance(entries, list) and len(entries) > 0
    assert isinstance(show_adl, bool) and isinstance(show_talkative, bool) and isinstance(show_blackbox, bool)
    assert show_adl or show_talkative or show_blackbox, "At least one group must be enabled"
    if baseline_mi_values is None:
        baseline_mi_values = [0, 5, 50]
    assert isinstance(baseline_mi_values, list)

    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    out_file = export_path / "runs.jsonl"

    all_variant_keys = [k for k, _ in VARIANTS]
    variant_keys: List[str] = [
        k
        for k in all_variant_keys
        if not k.startswith("baseline_") or any(f"baseline_mi{mi_val}" == k for mi_val in baseline_mi_values)
    ]
    if not show_adl:
        variant_keys = [k for k in variant_keys if not k.startswith("agent_")]
    if not show_talkative:
        variant_keys = [k for k in variant_keys if not k.startswith("talkative_")]
    if not show_blackbox:
        variant_keys = [k for k in variant_keys if not k.startswith("baseline_")]
    assert len(variant_keys) >= 1, "No variants remaining after filtering"

    def _variant_params(v_key: str) -> Tuple[str, int, bool]:
        if v_key == "agent_mi0":
            return "ADL", 0, False
        if v_key == "agent_mi5":
            return "ADL", 5, False
        if v_key == "talkative_mi0":
            return "TalkativeProbe", 0, False
        if v_key == "talkative_mi5":
            return "TalkativeProbe", 5, False
        if v_key.startswith("baseline_mi"):
            mi = int(v_key.replace("baseline_mi", ""))
            return "ADLBlackbox", mi, True
        raise AssertionError(f"Unknown variant key: {v_key}")

    llm_filter_norm = _agent_llm_filter_norm()

    with open(out_file, "w", encoding="utf-8") as f_out:
        for entry in entries:
            model, organism, organism_type, (id_adl, id_baseline, id_talkative) = _parse_entry(entry)
            cfg = load_hydra_config(
                config_path,
                f"organism={organism}",
                f"model={model}",
                "infrastructure=mats_cluster_paper",
            )
            for v_key in variant_keys:
                agent_type, mi, is_baseline = _variant_params(v_key)
                agent_root = _results_root_for_agent_type(cfg, agent_type) / "agent"
                assert agent_root.exists() and agent_root.is_dir(), f"Agent root not found: {agent_root}"
                run_id = None
                if agent_type == "ADL" and len(id_adl) > 0:
                    run_id = id_adl
                elif agent_type == "ADLBlackbox" and len(id_baseline) > 0:
                    run_id = id_baseline
                elif agent_type == "TalkativeProbe" and len(id_talkative) > 0:
                    run_id = id_talkative
                grade_paths = _find_all_grade_paths_by_kind_and_mi(
                    agent_root,
                    organism,
                    model,
                    mi=mi,
                    is_baseline=is_baseline,
                    llm_id_filter=llm_filter_norm,
                    agent_type=agent_type,
                    run_identifier=run_id,
                )
                for grade_path in grade_paths:
                    run_dir = grade_path.parent
                    assert run_dir.exists() and run_dir.is_dir()
                    m = re.search(r"__?run(\d+)$", run_dir.name)
                    assert m is not None, f"Cannot parse run index from: {run_dir.name}"
                    run_idx = int(m.group(1))

                    with open(grade_path, "r", encoding="utf-8") as f:
                        grade_payload = json.load(f)
                    assert isinstance(grade_payload, dict) and "score" in grade_payload
                    desc_fp = run_dir / "description.txt"
                    msg_fp = run_dir / "messages.json"
                    stats_fp = run_dir / "stats.json"
                    assert desc_fp.exists() and desc_fp.is_file()
                    assert msg_fp.exists() and msg_fp.is_file()
                    assert stats_fp.exists() and stats_fp.is_file()
                    description_text = desc_fp.read_text(encoding="utf-8")
                    with open(msg_fp, "r", encoding="utf-8") as fm:
                        messages = json.load(fm)
                    assert isinstance(messages, list) and len(messages) >= 1
                    with open(stats_fp, "r", encoding="utf-8") as fs:
                        stats_obj = json.load(fs)
                    assert isinstance(stats_obj, dict)

                    record: Dict[str, Any] = {
                        "model": model,
                        "model_display": _model_display_name(model),
                        "organism": organism,
                        "organism_type": organism_type,
                        "variant": v_key,
                        "agent_type": agent_type,
                        "mi": int(mi),
                        "run_idx": int(run_idx),
                        "score": int(grade_payload["score"]),
                        "grade": grade_payload,
                        "description": description_text,
                        "messages": messages,
                        "stats": stats_obj,
                        "run_dir": str(run_dir),
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_export_records(export_dir: str) -> List[Dict[str, Any]]:
    fp = Path(export_dir) / "runs.jsonl"
    assert fp.exists() and fp.is_file(), f"Export file not found: {fp}"
    records: List[Dict[str, Any]] = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            assert isinstance(rec, dict)
            records.append(rec)
    assert len(records) >= 1
    return records


def _collect_scores_for_entry_from_export(
    export_dir: str,
    model: str,
    organism: str,
    *,
    aggregation: str,
    baseline_mi_values: Optional[List[int]],
    show_adl: bool = True,
    show_talkative: bool = True,
    show_blackbox: bool = True,
) -> Dict[str, float]:
    records = _load_export_records(export_dir)
    all_variant_keys = [k for k, _ in VARIANTS]
    if baseline_mi_values is None:
        baseline_mi_values = [0, 5, 50]
    assert isinstance(baseline_mi_values, list)
    variant_keys: List[str] = [
        k
        for k in all_variant_keys
        if not k.startswith("baseline_") or any(f"baseline_mi{mi_val}" == k for mi_val in baseline_mi_values)
    ]
    if not show_adl:
        variant_keys = [k for k in variant_keys if not k.startswith("agent_")]
    if not show_talkative:
        variant_keys = [k for k in variant_keys if not k.startswith("talkative_")]
    if not show_blackbox:
        variant_keys = [k for k in variant_keys if not k.startswith("baseline_")]

    values: Dict[str, List[float]] = {k: [] for k in variant_keys}
    for rec in records:
        if rec.get("model") != model or rec.get("organism") != organism:
            continue
        v_key = rec.get("variant")
        if v_key not in values:
            continue
        s = float(rec.get("score"))
        values[v_key].append(s)

    def _agg(vals: List[float]) -> float:
        assert len(vals) >= 1
        arr = np.asarray(vals, dtype=np.float32)
        if aggregation == "max":
            return float(np.max(arr))
        elif aggregation == "min":
            return float(np.min(arr))
        elif aggregation == "median":
            return float(np.median(arr))
        elif aggregation == "mean":
            return float(np.mean(arr))
        else:
            raise ValueError(f"Invalid aggregation: {aggregation}")

    out: Dict[str, float] = {}
    for k in variant_keys:
        if len(values[k]) >= 1:
            out[k] = _agg(values[k])
    return out


def visualize_grades_by_type_average(
    entries: List[Tuple[str, str, str]],
    *,
    config_path: str = CONFIG_PATH,
    export_dir: str = "export",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5.5),
    columnspacing: float = 0.6,
    labelspacing: float = 0.6,
    aggregation: str = "mean",
    font_size: int = 20,
    x_label_pad: float = 70,
    n_cols: int = 7,
    baseline_mi_values: Optional[List[int]] = None,
    show_adl: bool = True,
    show_talkative: bool = True,
    show_blackbox: bool = True,
    inter_type_gap: float = 1.0,
    bar_width: float = 0.1,
    legend_loc: str = "lower center",
) -> None:
    assert isinstance(entries, list) and len(entries) > 0
    assert isinstance(show_adl, bool) and isinstance(show_talkative, bool) and isinstance(show_blackbox, bool)
    assert show_adl or show_talkative or show_blackbox, "At least one group must be enabled"

    if baseline_mi_values is None:
        baseline_mi_values = [0, 5, 50]

    per_variant_type_scores: Dict[str, Dict[str, List[float]]] = {k: {} for k, _ in VARIANTS}

    for entry in entries:
        model, organism, organism_type, _ = _parse_entry(entry)
        scores = _collect_scores_for_entry_from_export(
            export_dir,
            model,
            organism,
            aggregation=aggregation,
            baseline_mi_values=baseline_mi_values,
            show_adl=show_adl,
            show_talkative=show_talkative,
            show_blackbox=show_blackbox,
        )
        for v_key, score in scores.items():
            per_variant_type_scores.setdefault(v_key, {}).setdefault(organism_type, []).append(float(score))

    all_variant_keys = [k for k, _ in VARIANTS]
    variant_keys = [
        k
        for k in all_variant_keys
        if not k.startswith("baseline_") or any(f"baseline_mi{mi_val}" == k for mi_val in baseline_mi_values)
    ]
    if not show_adl:
        variant_keys = [k for k in variant_keys if not k.startswith("agent_")]
    if not show_talkative:
        variant_keys = [k for k in variant_keys if not k.startswith("talkative_")]
    if not show_blackbox:
        variant_keys = [k for k in variant_keys if not k.startswith("baseline_")]
    assert len(variant_keys) >= 1, "No variants remaining after filtering"
    variant_labels = [lbl for k, lbl in VARIANTS if k in variant_keys]
    colors_for_variant: Dict[str, str] = {
        "agent_mi0": "#ffd973",
        "agent_mi5": "#FDB813",
        "talkative_mi0": "#ffd973",
        "talkative_mi5": "#FDB813",
        "baseline_mi0": "#9ecae1",
        "baseline_mi5": "#3182bd",
        "baseline_mi50": "#08519c",
    }
    local_hatch_for_variant: Dict[str, str] = {
        "agent_mi0": "",
        "agent_mi5": "",
        "talkative_mi0": "////",
        "talkative_mi5": "////",
        "baseline_mi0": "",
        "baseline_mi5": "",
        "baseline_mi50": "",
    }

    types = sorted({_parse_entry(entry)[2] for entry in entries})
    fig, ax = plt.subplots(figsize=figsize)
    inner_spacing = bar_width * 0.25
    minor_group_gap = bar_width * 0.8
    adl_keys = [k for k in variant_keys if k.startswith("agent_")]
    talk_keys = [k for k in variant_keys if k.startswith("talkative_")]
    blackbox_keys = [k for k in variant_keys if k.startswith("baseline_")]

    enabled_groups: List[Tuple[str, List[str]]] = []
    if show_talkative and len(talk_keys) > 0:
        enabled_groups.append(("talk", talk_keys))
    if show_adl and len(adl_keys) > 0:
        enabled_groups.append(("adl", adl_keys))
    if show_blackbox and len(blackbox_keys) > 0:
        enabled_groups.append(("blackbox", blackbox_keys))

    assert len(enabled_groups) >= 1, "At least one group must have variants"

    variant_keys_ordered: List[str] = []
    variant_labels_ordered: List[str] = []
    variant_labels_dict = {k: lbl for k, lbl in VARIANTS if k in variant_keys}
    for _group_name, group_keys in enabled_groups:
        for k in group_keys:
            variant_keys_ordered.append(k)
            variant_labels_ordered.append(variant_labels_dict[k])
    variant_keys = variant_keys_ordered
    variant_labels = variant_labels_ordered

    group_widths: List[float] = []
    for _group_name, group_keys in enabled_groups:
        group_width = len(group_keys) * bar_width + (len(group_keys) - 1) * inner_spacing
        group_widths.append(group_width)

    total_group_width = sum(group_widths) + (len(enabled_groups) - 1) * minor_group_gap
    assert total_group_width < 0.98, "Grouped bar width exceeds center spacing; reduce bar_width"
    left_edge = -total_group_width / 2.0
    offsets_map: Dict[str, float] = {}
    cursor = left_edge + bar_width / 2.0
    for i, (_group_name, group_keys) in enumerate(enabled_groups):
        for k in group_keys:
            offsets_map[k] = cursor
            cursor += bar_width + inner_spacing
        if i < len(enabled_groups) - 1:
            cursor += minor_group_gap
    offsets = [offsets_map[k] for k in variant_keys]

    centers = np.arange(len(types), dtype=float) * inter_type_gap

    for i, v_key in enumerate(variant_keys):
        means: List[float] = []
        stds: List[float] = []
        for t in types:
            vals = per_variant_type_scores.get(v_key, {}).get(t, [])
            assert len(vals) >= 1, f"No grades for variant={v_key} type={t}"
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        xs = centers + offsets[i]
        is_talkative = v_key.startswith("talkative_")
        bars = ax.bar(
            xs,
            np.asarray(means, dtype=np.float32),
            width=bar_width,
            yerr=np.asarray(stds, dtype=np.float32),
            label=variant_labels[i],
            color=colors_for_variant[variant_keys[i]],
            hatch=local_hatch_for_variant[variant_keys[i]],
            alpha=0.9,
            ecolor="black",
            capsize=5,
            error_kw={"linewidth": 2, "alpha": 0.3},
            edgecolor="black" if is_talkative else None,
            linewidth=2.0 if is_talkative else 0.0,
        )
        for rect, m, s in zip(bars, means, stds):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height() + s + 0.05,
                f"{m:.2f}",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE_BAR_VALUE,
            )
        color = colors_for_variant[variant_keys[i]]
        for idx, t in enumerate(types):
            vals = per_variant_type_scores.get(v_key, {}).get(t, [])
            if len(vals) == 0:
                continue
            x_center = centers[idx] + offsets[i]
            n = len(vals)
            if n == 1:
                xs_pts = np.array([x_center], dtype=np.float32)
            else:
                spread = bar_width * 0.35
                xs_pts = x_center + (np.linspace(-0.5, 0.5, n, dtype=np.float32) * spread)
            ax.scatter(
                xs_pts,
                np.asarray(vals, dtype=np.float32),
                color=color,
                s=18,
                alpha=1.0,
                edgecolors="black",
                linewidths=0.2,
                zorder=3,
            )
    ax.set_xticks(centers)
    ax.set_xticklabels(types, fontsize=FONT_SIZE_X_AXIS_LABEL)
    ax.set_ylabel("Grade (1..5)", fontsize=FONT_SIZE_Y_AXIS_LABEL)
    ax.set_ylim(1.0, 5.0)
    ax.grid(True, linestyle=":", alpha=0.3, axis="y")
    ax.tick_params(axis="x", pad=x_label_pad, labelsize=FONT_SIZE_X_AXIS_LABEL)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_Y_AXIS_TICK)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=labels,
        loc=legend_loc,
        ncol=n_cols,
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
    )

    plt.tight_layout()
    if save_path is not None:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path_obj), dpi=300, bbox_inches="tight")
    # plt.show()


# %%
entries_grouped = [
    ("gemma2_9B_it", "em_risky_financial_advice_mix1-1p0", "EM"),
    ("gemma2_9B_it", "em_bad_medical_advice_mix1-1p0", "EM"),
    ("gemma2_9B_it", "em_extreme_sports_mix1-1p0", "EM"),
    ("qwen3_8B", "em_risky_financial_advice_mix1-1p0", "EM"),
    ("qwen3_8B", "em_bad_medical_advice_mix1-1p0", "EM"),
    ("qwen3_8B", "em_extreme_sports_mix1-1p0", "EM"),
]
# %%
# export_full_runs_for_entries(
#     entries_grouped,
#     export_dir="export",
#     config_path="configs/config.yaml",
#     baseline_mi_values=[0, 5, 50],
#     show_adl=True,
#     show_talkative=True,
#     show_blackbox=True,
# )
# %%
visualize_grades_by_type_average(
    entries_grouped,
    config_path="configs/config.yaml",
    save_path=str(Path(__file__).parent.parent.parent / "images" / "em_audit" / "em_audit_results.pdf"),
    export_dir=str(Path(__file__).parent.parent / "em_runs"),
    inter_type_gap=1.2,
    figsize=(12, 5.5),
    columnspacing=0.02,
    labelspacing=0.02,
    baseline_mi_values=[0, 5, 50],
    show_adl=True,
    show_talkative=True,
    show_blackbox=True,
    x_label_pad=15,
    bar_width=0.1,
    n_cols=4,
    legend_loc=(0.08, 0.77),
)
# %%
