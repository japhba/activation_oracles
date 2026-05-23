"""
Run the eval-awareness comparison requested for Qwen3-14B:

- HF text baseline (full-context and selected-span variants)
- Three AO verbalizer checkpoints

Usage:
    source .env && .venv/bin/python experiments/eval_awareness_compare.py
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import torch

from nl_probes.open_ended_eval.eval_awareness import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_SEGMENT_TOKENS,
    GENERATION_KWARGS,
    TEXT_BASELINE_VARIANTS,
    VERBALIZER_PROMPTS,
    run_eval_awareness_open_ended_eval,
    run_eval_awareness_text_baseline_eval,
)
from nl_probes.utils.common import load_model, load_tokenizer

DEFAULT_MODEL = "Qwen/Qwen3-14B"
DEFAULT_VERBALIZER_LORAS = [
    "adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-14B",
    "checkpoints/500k_pl_31k_spqav2_199k_sqav3_126k_cls_qwen3_14b/final",
    "checkpoints/500k_pl_31k_spqav2_199k_sqav3_50k_hb_126k_cls_qwen3_14b/final",
]


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "total",
        "accuracy_at_zero",
        "roc_auc",
        "mean_margin_when_yes",
        "mean_margin_when_no",
        "alignment_vs_real_roc_auc",
        "capability_vs_real_roc_auc",
        "variant_full_context_roc_auc",
        "variant_selected_span_roc_auc",
        "variant_full_context_accuracy_at_zero",
        "variant_selected_span_accuracy_at_zero",
        "prompt_evaluation_or_benchmark_roc_auc",
        "prompt_model_being_evaluated_roc_auc",
    ]
    return {key: metrics[key] for key in keys if key in metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--verbalizer-lora",
        type=str,
        action="append",
        dest="verbalizer_loras",
        default=None,
        help="Repeat to override the default three 14B checkpoints.",
    )
    parser.add_argument("--output-dir", type=str, default="experiments/eval_awareness_comparison_qwen3_14b")
    parser.add_argument("--max-context-tokens", type=int, default=DEFAULT_MAX_CONTEXT_TOKENS)
    parser.add_argument("--segment-tokens", type=int, default=DEFAULT_SEGMENT_TOKENS)
    parser.add_argument("--max-entries-per-class", type=int, default=None)
    parser.add_argument("--dataset-json-path", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    verbalizer_loras = args.verbalizer_loras or DEFAULT_VERBALIZER_LORAS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Loading tokenizer: {args.model}")
    tokenizer = load_tokenizer(args.model)

    print(f"Loading base model for text baseline: {args.model} on {device} with dtype={dtype}")
    hf_model = load_model(args.model, dtype)
    hf_model.eval()

    text_output_dir = output_dir / "text_baseline"
    text_summary = run_eval_awareness_text_baseline_eval(
        model_name=args.model,
        tokenizer=tokenizer,
        backend="hf",
        hf_model=hf_model,
        device=device,
        output_dir=str(text_output_dir),
        eval_batch_size=64,
        generation_kwargs=GENERATION_KWARGS,
        max_entries_per_class=args.max_entries_per_class,
        max_context_tokens=args.max_context_tokens,
        segment_tokens=args.segment_tokens,
        verbalizer_prompts=VERBALIZER_PROMPTS,
        variants=TEXT_BASELINE_VARIANTS,
        dataset_json_path=args.dataset_json_path,
        cache_dir=args.cache_dir,
    )

    del hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Loading base model for AO eval: {args.model} on {device} with dtype={dtype}")
    ao_model = load_model(args.model, dtype)
    ao_model.eval()

    ao_output_dir = output_dir / "ao"
    ao_summary = run_eval_awareness_open_ended_eval(
        model_name=args.model,
        model=ao_model,
        tokenizer=tokenizer,
        device=device,
        output_dir=str(ao_output_dir),
        eval_batch_size=64,
        generation_kwargs=GENERATION_KWARGS,
        verbalizer_lora_paths=verbalizer_loras,
        max_entries_per_class=args.max_entries_per_class,
        max_context_tokens=args.max_context_tokens,
        segment_tokens=args.segment_tokens,
        verbalizer_prompts=VERBALIZER_PROMPTS,
        dataset_json_path=args.dataset_json_path,
        cache_dir=args.cache_dir,
    )

    comparison_summary = {
        "model_name": args.model,
        "verbalizer_loras": verbalizer_loras,
        "dataset_filters": {
            "max_context_tokens": args.max_context_tokens,
            "segment_tokens": args.segment_tokens,
            "max_entries_per_class": args.max_entries_per_class,
            "chat_only": True,
            "final_user_turn_only": True,
            "balanced_classes": True,
        },
        "text_baseline_summary": text_summary,
        "text_baseline_compact_metrics": _compact_metrics(text_summary["overall_metrics"]),
        "ao_summary": ao_summary,
        "ao_compact_metrics_by_verbalizer": {
            key: _compact_metrics(metrics)
            for key, metrics in ao_summary["metrics_by_verbalizer"].items()
        },
    }

    summary_path = output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(comparison_summary, indent=2))
    print(f"Saved comparison summary to {summary_path}")

    print("\n=== Text Baseline ===")
    print(json.dumps(comparison_summary["text_baseline_compact_metrics"], indent=2))

    print("\n=== AO Verbalizers ===")
    for verbalizer_key, metrics in comparison_summary["ao_compact_metrics_by_verbalizer"].items():
        print(f"\n--- {verbalizer_key} ---")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
