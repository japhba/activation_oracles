"""Aggregate AObench results across the 5 layer sweep checkpoints."""
import json
import re
from pathlib import Path

ROOT = Path("/workspace/repo/activation_oracles_dev/third_party/cot-oracle/AObench/eval_results")

rows = {}
for layer in [19, 22, 25, 28, 31]:
    d = ROOT / f"layer_sweep_L{layer}"
    agg_path = d / "report" / "aggregate_scores.json"
    if not agg_path.exists():
        print(f"[skip] L{layer}: no aggregate_scores.json at {d}")
        continue
    agg = json.loads(agg_path.read_text())
    final = agg.get("final", {})
    rows[layer] = (final, str(d))

print(f"Found {len(rows)} layer results: {sorted(rows)}\n")
if not rows:
    raise SystemExit("no results found")

all_evals = sorted({e for f, _ in rows.values() for e in f.get("per_eval_normalized", {})})
header = f"{'Layer':<7}{'mean':>10}{'CI_lo':>10}{'CI_hi':>10}  " + "  ".join(f"{e[:14]:>15}" for e in all_evals)
print(header)
print("-" * len(header))
nan = float("nan")
for layer in sorted(rows):
    final, _ = rows[layer]
    mean = final.get("mean_normalized_score", nan)
    cilo = final.get("ci_lo", nan)
    cihi = final.get("ci_hi", nan)
    per = final.get("per_eval_normalized", {})
    cells = "  ".join(f"{per.get(e, nan):>15.4f}" for e in all_evals)
    print(f"L{layer:<6}{mean:>10.4f}{cilo:>10.4f}{cihi:>10.4f}  {cells}")

print()
print("Per-layer best / worst eval:")
for layer in sorted(rows):
    per = rows[layer][0].get("per_eval_normalized", {})
    if not per:
        continue
    best = max(per.items(), key=lambda kv: kv[1])
    worst = min(per.items(), key=lambda kv: kv[1])
    print(f"  L{layer}: best={best[0]} ({best[1]:+.4f}), worst={worst[0]} ({worst[1]:+.4f})")

print()
print("Best layer overall (mean_normalized_score):")
best_layer = max(rows, key=lambda L: rows[L][0].get("mean_normalized_score", -float("inf")))
print(f"  L{best_layer} = {rows[best_layer][0].get('mean_normalized_score'):+.4f}")
print(f"  source: {rows[best_layer][1]}")
