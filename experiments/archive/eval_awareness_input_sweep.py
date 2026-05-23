"""
Sweep AO input-position variants for eval-awareness on Qwen3-14B.

This keeps the chat-only, final-user-turn dataset slice fixed and varies which
token positions are exposed to the AO:

- final_user_turn
- last_20 tokens
- last_50 tokens
- last_100 tokens
- last_500 tokens

Usage:
    source .env && .venv/bin/python experiments/eval_awareness_input_sweep.py
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
    GENERATION_KWARGS,
    VERBALIZER_PROMPTS,
    run_eval_awareness_open_ended_eval,
)
from nl_probes.utils.common import load_model, load_tokenizer

DEFAULT_MODEL = "Qwen/Qwen3-14B"
DEFAULT_VERBALIZER_LORAS = [
    "adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-14B",
    "checkpoints/500k_pl_31k_spqav2_199k_sqav3_126k_cls_qwen3_14b/final",
    "checkpoints/500k_pl_31k_spqav2_199k_sqav3_50k_hb_126k_cls_qwen3_14b/final",
]
DEFAULT_VARIANTS = [
    {"label": "final_user_turn", "position_strategy": "final_user_turn", "segment_tokens": 0},
    {"label": "last_20", "position_strategy": "last_tokens", "segment_tokens": 20},
    {"label": "last_50", "position_strategy": "last_tokens", "segment_tokens": 50},
    {"label": "last_100", "position_strategy": "last_tokens", "segment_tokens": 100},
    {"label": "last_500", "position_strategy": "last_tokens", "segment_tokens": 500},
]


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "total",
        "accuracy_at_zero",
        "roc_auc",
        "mean_margin_when_yes",
        "mean_margin_when_no",
        "mean_segment_token_count",
        "alignment_vs_real_roc_auc",
        "capability_vs_real_roc_auc",
        "prompt_evaluation_or_benchmark_roc_auc",
        "prompt_model_being_evaluated_roc_auc",
    ]
    compact = {key: metrics[key] for key in keys if key in metrics}
    prompt_auc_keys = [key for key in compact if key.startswith("prompt_") and key.endswith("_roc_auc")]
    if prompt_auc_keys:
        compact["mean_prompt_roc_auc"] = sum(compact[key] for key in prompt_auc_keys) / len(prompt_auc_keys)
    return compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--verbalizer-lora",
        type=str,
        action="append",
        dest="verbalizer_loras",
        default=None,
    )
    parser.add_argument("--output-dir", type=str, default="experiments/eval_awareness_input_sweep_qwen3_14b")
    parser.add_argument("--max-context-tokens", type=int, default=DEFAULT_MAX_CONTEXT_TOKENS)
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

    print(f"Loading base model for AO sweep: {args.model} on {device} with dtype={dtype}")
    model = load_model(args.model, dtype)
    model.eval()

    per_variant_summary: dict[str, Any] = {}

    for variant in DEFAULT_VARIANTS:
        label = variant["label"]
        print(f"\n=== Running variant: {label} ===")
        variant_output_dir = output_dir / label
        summary = run_eval_awareness_open_ended_eval(
            model_name=args.model,
            model=model,
            tokenizer=tokenizer,
            device=device,
            output_dir=str(variant_output_dir),
            eval_batch_size=64,
            generation_kwargs=GENERATION_KWARGS,
            verbalizer_lora_paths=verbalizer_loras,
            max_entries_per_class=args.max_entries_per_class,
            max_context_tokens=args.max_context_tokens,
            position_strategy=variant["position_strategy"],
            segment_tokens=variant["segment_tokens"],
            verbalizer_prompts=VERBALIZER_PROMPTS,
            dataset_json_path=args.dataset_json_path,
            cache_dir=args.cache_dir,
        )
        per_variant_summary[label] = {
            "position_strategy": variant["position_strategy"],
            "segment_tokens": variant["segment_tokens"],
            "metrics_by_verbalizer": {
                key: _compact_metrics(metrics)
                for key, metrics in summary["metrics_by_verbalizer"].items()
            },
        }

    summary = {
        "model_name": args.model,
        "verbalizer_loras": verbalizer_loras,
        "dataset_filters": {
            "max_context_tokens": args.max_context_tokens,
            "max_entries_per_class": args.max_entries_per_class,
            "chat_only": True,
            "final_user_turn_only": True,
            "balanced_classes": True,
        },
        "variants": per_variant_summary,
    }

    summary_path = output_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved sweep summary to {summary_path}")

    for label, variant_summary in per_variant_summary.items():
        print(f"\n=== {label} ===")
        for verbalizer_key, metrics in variant_summary["metrics_by_verbalizer"].items():
            print(f"\n--- {verbalizer_key} ---")
            print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
