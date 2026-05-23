"""Aggregate AObench results across the 5 layers (19/22/25/28/31).

Reads each layer's `paper_collection_aobench_*/all_summaries.json` (or
falls back to per-eval `*_summary.json`) and emits a tidy CSV plus a
markdown table to stdout.

Usage:
    python scripts/aggregate_layer_sweep.py \
        --eval-results-root /workspace/repo/activation_oracles_dev/third_party/cot-oracle/AObench/eval_results \
        --output /workspace/logs/aobench_summary.csv
"""

import argparse
import csv
import json
import re
from pathlib import Path

LAYERS = [19, 22, 25, 28, 31]


def find_layer_summary(root: Path, layer: int) -> dict | None:
    """Find the most recent paper_collection_aobench_* dir for a given layer
    and return its all_summaries.json contents."""
    candidates = sorted(
        root.glob("paper_collection_aobench_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    pattern = re.compile(rf"L{layer}\b|layer{layer}\b|_layer{layer}_|_L{layer}_")
    for d in candidates:
        cfg = d / "run_config.json"
        if not cfg.exists():
            continue
        try:
            run_cfg = json.loads(cfg.read_text())
        except Exception:
            continue
        loras = run_cfg.get("verbalizer_loras") or run_cfg.get("verbalizer_lora") or []
        if isinstance(loras, str):
            loras = [loras]
        if any(pattern.search(str(x)) for x in loras):
            sums = d / "all_summaries.json"
            if sums.exists():
                return json.loads(sums.read_text())
    return None


def flatten(summary: dict, layer: int) -> list[dict]:
    rows = []
    for eval_name, payload in summary.items():
        if isinstance(payload, dict):
            for verbalizer, metrics in payload.items():
                if not isinstance(metrics, dict):
                    continue
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        rows.append({
                            "layer": layer,
                            "eval": eval_name,
                            "verbalizer": verbalizer,
                            "metric": metric_name,
                            "value": value,
                        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-results-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    rows = []
    for layer in LAYERS:
        summary = find_layer_summary(args.eval_results_root, layer)
        if summary is None:
            print(f"[warn] no summary found for layer {layer}")
            continue
        rows.extend(flatten(summary, layer))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "eval", "verbalizer", "metric", "value"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")

    # markdown pivot: layer x eval (primary metric where possible)
    primary = {}  # (layer, eval) -> value
    for r in rows:
        if r["metric"] in {"accuracy", "auc", "f1", "score", "judge_score", "mean", "specificity"}:
            primary[(r["layer"], r["eval"])] = r["value"]
    evals = sorted({r["eval"] for r in rows})
    print()
    print("| Layer | " + " | ".join(evals) + " |")
    print("|" + "---|" * (len(evals) + 1))
    for layer in LAYERS:
        cells = [f"{primary.get((layer, e), 'n/a'):.3f}" if isinstance(primary.get((layer, e)), (int, float)) else "n/a" for e in evals]
        print(f"| {layer} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
