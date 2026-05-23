"""
One-off script: sweep context length (segment_tokens) across MMLU, number prediction,
and backtracking evals for a single checkpoint.

Loads the model once, runs all combos, collects results, writes a markdown report.

Usage:
    source .env && .venv/bin/python experiments/context_length_sweep.py \
        --checkpoint-dir checkpoints/500k_pl_31k_spqav2_199k_sqav3_126k_cls_100k_chatreg/final
"""

import argparse
import json
import os
import random
import time
from typing import Any

import torch

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from nl_probes.open_ended_eval.backtracking import (
    GENERATION_KWARGS as BACKTRACKING_GEN_KWARGS,
    run_backtracking_open_ended_eval,
)
from nl_probes.open_ended_eval.mmlu_prediction import (
    AO_PRE_ANSWER_PROMPTS,
    AO_POST_ANSWER_PROMPTS,
    run_mmlu_prediction_open_ended_eval,
)
from nl_probes.open_ended_eval.number_prediction import (
    run_number_prediction_open_ended_eval,
)
from nl_probes.utils.common import load_model, load_tokenizer

MODEL_NAME = "Qwen/Qwen3-8B"

# Use 99999 for "full context" — compute_segment_positions clamps to available tokens
FULL = 99999

SWEEP_CONFIG = {
    "mmlu_prediction": {
        "segment_tokens_values": [10, 50, FULL],
    },
    "backtracking": {
        "segment_tokens_values": [20, 50, 200, FULL],
    },
    "number_prediction": {
        "segment_tokens_values": [10, FULL],
    },
}


def label_for_seg(seg: int) -> str:
    return "full" if seg == FULL else str(seg)


def run_mmlu(model, tokenizer, device, checkpoint_dir, segment_tokens, output_dir):
    """Run MMLU in both pre_answer and post_answer modes, return combined metrics."""
    results = {}
    for mode_name, prompts in [("pre_answer", AO_PRE_ANSWER_PROMPTS), ("post_answer", AO_POST_ANSWER_PROMPTS)]:
        mode_dir = os.path.join(output_dir, mode_name)
        os.makedirs(mode_dir, exist_ok=True)
        summary = run_mmlu_prediction_open_ended_eval(
            model_name=MODEL_NAME,
            model=model,
            tokenizer=tokenizer,
            device=device,
            output_dir=mode_dir,
            verbalizer_prompts=prompts,
            verbalizer_lora_paths=[checkpoint_dir],
            segment_tokens=segment_tokens,
        )
        results[mode_name] = summary
    return results


def run_number(model, tokenizer, device, checkpoint_dir, segment_tokens, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return run_number_prediction_open_ended_eval(
        model_name=MODEL_NAME,
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=output_dir,
        verbalizer_lora_paths=[checkpoint_dir],
        segment_tokens=segment_tokens,
    )


def run_backtrack(model, tokenizer, device, checkpoint_dir, segment_tokens, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return run_backtracking_open_ended_eval(
        model_name=MODEL_NAME,
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=output_dir,
        generation_kwargs=BACKTRACKING_GEN_KWARGS,
        verbalizer_lora_paths=[checkpoint_dir],
        segment_tokens=segment_tokens,
    )


def extract_mmlu_metrics(results: dict) -> dict[str, float]:
    """Extract key metrics from MMLU results (both modes)."""
    metrics = {}
    for mode_name, summary in results.items():
        mbv = summary.get("metrics_by_verbalizer", {})
        # There's typically one verbalizer — grab the first
        for _vkey, m in mbv.items():
            if "roc_auc" in m:
                metrics[f"{mode_name}_roc_auc"] = m["roc_auc"]
            if "accuracy_at_zero" in m:
                metrics[f"{mode_name}_accuracy"] = m["accuracy_at_zero"]
            break
    return metrics


def extract_number_metrics(summary: dict) -> dict[str, float]:
    mbv = summary.get("metrics_by_verbalizer", {})
    for _vkey, m in mbv.items():
        return {k: v for k, v in m.items() if isinstance(v, (int, float))}
    return {}


def extract_backtracking_metrics(summary: dict) -> dict[str, float]:
    mbv = summary.get("metrics_by_verbalizer", {})
    for _vkey, m in mbv.items():
        return {k: v for k, v in m.items() if isinstance(v, (int, float))}
    return {}


def generate_report(all_results: dict[str, dict[int, dict]], output_path: str):
    """Generate a markdown report from collected results."""
    lines = [
        "# Context Length Sweep Results",
        "",
        f"**Checkpoint**: `500k_pl_31k_spqav2_199k_sqav3_126k_cls_100k_chatreg/final`",
        f"**Model**: {MODEL_NAME}",
        "",
    ]

    # --- MMLU ---
    lines.append("## MMLU Prediction")
    lines.append("")
    mmlu_results = all_results.get("mmlu_prediction", {})
    if mmlu_results:
        # Build table
        seg_vals = sorted(mmlu_results.keys(), key=lambda x: x if x != FULL else 999999)
        headers = ["segment_tokens", "pre_answer ROC AUC", "pre_answer accuracy", "post_answer ROC AUC", "post_answer accuracy", "time (s)"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for seg in seg_vals:
            data = mmlu_results[seg]
            m = data["metrics"]
            row = [
                label_for_seg(seg),
                f"{m.get('pre_answer_roc_auc', '-'):.4f}" if isinstance(m.get('pre_answer_roc_auc'), float) else "-",
                f"{m.get('pre_answer_accuracy', '-'):.4f}" if isinstance(m.get('pre_answer_accuracy'), float) else "-",
                f"{m.get('post_answer_roc_auc', '-'):.4f}" if isinstance(m.get('post_answer_roc_auc'), float) else "-",
                f"{m.get('post_answer_accuracy', '-'):.4f}" if isinstance(m.get('post_answer_accuracy'), float) else "-",
                f"{data['elapsed']:.0f}",
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # --- Number Prediction ---
    lines.append("## Number Prediction")
    lines.append("")
    num_results = all_results.get("number_prediction", {})
    if num_results:
        seg_vals = sorted(num_results.keys(), key=lambda x: x if x != FULL else 999999)
        # Get all metric keys
        all_metric_keys = set()
        for data in num_results.values():
            all_metric_keys.update(data["metrics"].keys())
        metric_keys = sorted(all_metric_keys)
        headers = ["segment_tokens"] + metric_keys + ["time (s)"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for seg in seg_vals:
            data = num_results[seg]
            m = data["metrics"]
            row = [label_for_seg(seg)]
            for k in metric_keys:
                v = m.get(k)
                if isinstance(v, float):
                    row.append(f"{v:.4f}")
                elif v is not None:
                    row.append(str(v))
                else:
                    row.append("-")
            row.append(f"{data['elapsed']:.0f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # --- Backtracking ---
    lines.append("## Backtracking")
    lines.append("")
    bt_results = all_results.get("backtracking", {})
    if bt_results:
        seg_vals = sorted(bt_results.keys(), key=lambda x: x if x != FULL else 999999)
        all_metric_keys = set()
        for data in bt_results.values():
            all_metric_keys.update(data["metrics"].keys())
        metric_keys = sorted(all_metric_keys)
        headers = ["segment_tokens"] + metric_keys + ["time (s)"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for seg in seg_vals:
            data = bt_results[seg]
            m = data["metrics"]
            row = [label_for_seg(seg)]
            for k in metric_keys:
                v = m.get(k)
                if isinstance(v, float):
                    row.append(f"{v:.4f}")
                elif v is not None:
                    row.append(str(v))
                else:
                    row.append("-")
            row.append(f"{data['elapsed']:.0f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport written to {output_path}")
    print(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to checkpoint directory with adapter files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/context_length_sweep_results",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="experiments/context_length_sweep_report.md",
    )
    parser.add_argument(
        "--evals",
        nargs="*",
        default=None,
        help="Which evals to run (default: all). Options: mmlu_prediction, number_prediction, backtracking",
    )
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = load_tokenizer(MODEL_NAME)
    print(f"Loading model: {MODEL_NAME} on {device} with dtype={dtype}")
    model = load_model(MODEL_NAME, dtype)
    model.eval()

    checkpoint_dir = args.checkpoint_dir
    evals_to_run = args.evals or list(SWEEP_CONFIG.keys())
    all_results: dict[str, dict[int, dict]] = {}

    for eval_name in evals_to_run:
        if eval_name not in SWEEP_CONFIG:
            print(f"Unknown eval: {eval_name}, skipping")
            continue

        config = SWEEP_CONFIG[eval_name]
        all_results[eval_name] = {}

        for seg_tokens in config["segment_tokens_values"]:
            seg_label = label_for_seg(seg_tokens)
            print(f"\n{'='*70}")
            print(f"  {eval_name} | segment_tokens={seg_label}")
            print(f"{'='*70}\n")

            out_dir = os.path.join(args.output_dir, eval_name, f"seg_{seg_label}")
            os.makedirs(out_dir, exist_ok=True)

            start = time.perf_counter()
            try:
                if eval_name == "mmlu_prediction":
                    raw = run_mmlu(model, tokenizer, device, checkpoint_dir, seg_tokens, out_dir)
                    metrics = extract_mmlu_metrics(raw)
                elif eval_name == "number_prediction":
                    raw = run_number(model, tokenizer, device, checkpoint_dir, seg_tokens, out_dir)
                    metrics = extract_number_metrics(raw)
                elif eval_name == "backtracking":
                    raw = run_backtrack(model, tokenizer, device, checkpoint_dir, seg_tokens, out_dir)
                    metrics = extract_backtracking_metrics(raw)
                else:
                    continue
            except Exception as e:
                print(f"ERROR in {eval_name} seg={seg_label}: {e}")
                import traceback
                traceback.print_exc()
                metrics = {"error": str(e)}

            elapsed = time.perf_counter() - start
            all_results[eval_name][seg_tokens] = {
                "metrics": metrics,
                "elapsed": elapsed,
            }
            print(f"\n  {eval_name} seg={seg_label} done in {elapsed:.0f}s | metrics={metrics}")

            # Save incremental results
            incremental_path = os.path.join(args.output_dir, "all_results.json")
            # Convert int keys to strings for JSON
            json_safe = {
                ename: {str(k): v for k, v in seg_dict.items()}
                for ename, seg_dict in all_results.items()
            }
            with open(incremental_path, "w") as f:
                json.dump(json_safe, f, indent=2)

    generate_report(all_results, args.report_path)


if __name__ == "__main__":
    main()
