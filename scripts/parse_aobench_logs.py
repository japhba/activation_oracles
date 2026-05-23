"""Parse the per-layer AObench logs to recover per-eval final stats.

The 5 simultaneous AObench runs all wrote to the same timestamp-collision
output dir, so the on-disk JSONs were overwritten. The logs, however, still
contain each layer's own '--- <eval_name> ---' / '  final: {...}' blocks.
"""

import json
import re
from pathlib import Path

LOG_DIR = Path("/workspace/logs/aobench")
LAYERS = [19, 22, 25, 28, 31]

# Eval header looks like '--- number_prediction ---' and the corresponding
# final dict is on a subsequent line starting with '  final: {...}'.
EVAL_HEADER = re.compile(r"^---\s*([a-z_]+)\s*---\s*$")
FINAL_LINE = re.compile(r"^\s*final:\s*(\{.*\})\s*$")


def parse_log(path: Path) -> dict[str, dict]:
    """Return {eval_name: final_payload}."""
    out: dict[str, dict] = {}
    current = None
    text = path.read_text(errors="ignore")
    for line in text.splitlines():
        m = EVAL_HEADER.match(line)
        if m:
            current = m.group(1)
            continue
        m = FINAL_LINE.match(line)
        if m and current is not None:
            try:
                out[current] = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
            current = None
    return out


def primary_metric(eval_name: str, payload: dict) -> tuple[str, float] | None:
    """Pick a single representative scalar per eval for the table."""
    candidates: list[tuple[str, str, str | None]] = [
        ("number_prediction", "accuracy", None),
        ("mmlu_prediction", "auc", None),
        ("backtracking", "judge_score", None),
        ("missing_info", "auc", None),
        ("sycophancy", "auc", None),
        ("vagueness", "specificity", None),
        ("domain_confusion", "judge_score", None),
        ("activation_sensitivity", "activation_sensitivity", None),
        ("hallucination", "hallucination_rate", "lower_better"),
        ("system_prompt_qa_hidden", "judge_score", None),
        ("system_prompt_qa_latentqa", "judge_score", None),
    ]
    for name, key, _ in candidates:
        if name == eval_name and key in payload:
            return key, payload[key]
    # Fallback: first scalar in payload.
    for k, v in payload.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return k, v
    return None


def main() -> None:
    by_layer: dict[int, dict[str, dict]] = {}
    for layer in LAYERS:
        path = LOG_DIR / f"layer{layer}.log"
        if not path.exists():
            print(f"[skip] no log for L{layer}")
            continue
        by_layer[layer] = parse_log(path)
        print(f"L{layer}: parsed {len(by_layer[layer])} evals")

    print()
    all_evals = sorted({e for d in by_layer.values() for e in d})
    print("== Per-layer primary metric (one scalar per eval) ==")
    header = f"{'Layer':<7}" + "  ".join(f"{e[:14]:>16}" for e in all_evals)
    print(header)
    print("-" * len(header))
    rows = []
    for layer in sorted(by_layer):
        cells = []
        row = {"layer": layer}
        for ev in all_evals:
            payload = by_layer[layer].get(ev, {})
            res = primary_metric(ev, payload) if payload else None
            if res is None:
                cells.append(f"{'n/a':>16}")
                row[ev] = None
            else:
                key, val = res
                cells.append(f"{val:>16.4f}")
                row[ev] = val
        rows.append(row)
        print(f"L{layer:<6}" + "  ".join(cells))

    print()
    print("== Eval -> primary metric key ==")
    for ev in all_evals:
        for layer in by_layer:
            payload = by_layer[layer].get(ev)
            if payload:
                k, _ = primary_metric(ev, payload) or ("?", 0.0)
                print(f"  {ev}: {k}")
                break

    out_csv = Path("/workspace/logs/layer_sweep_aobench_results.csv")
    with out_csv.open("w") as f:
        f.write("layer," + ",".join(all_evals) + "\n")
        for r in rows:
            f.write(f"{r['layer']}," + ",".join(
                "" if r[ev] is None else f"{r[ev]:.6f}" for ev in all_evals
            ) + "\n")
    print(f"\nwrote {out_csv}")

    out_full = Path("/workspace/logs/layer_sweep_aobench_full.json")
    out_full.write_text(json.dumps(by_layer, indent=2))
    print(f"wrote {out_full}")


if __name__ == "__main__":
    main()
