"""
Run all open-ended evals.

Standalone usage:
    source .env && .venv/bin/python -m nl_probes.open_ended_eval.run_all
    source .env && .venv/bin/python -m nl_probes.open_ended_eval.run_all --verbalizer-lora my-org/my-lora
    source .env && .venv/bin/python -m nl_probes.open_ended_eval.run_all --include number_prediction mmlu_prediction

Also used by nl_probes/sft.py for during-training eval (which saves the current
LoRA to disk and passes the path here — same codepath as standalone).
"""

import argparse
import json
import os
import random
import time
from typing import Any


# ---------------------------------------------------------------------------
# Eval wrappers — each knows its own generation_kwargs and batch_size
# ---------------------------------------------------------------------------


def _average_numeric_metrics_by_verbalizer(mode_summaries: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    merged_mbv: dict[str, dict[str, list[float]]] = {}
    for summary in mode_summaries.values():
        for verb_key, metrics in summary.items():
            if verb_key not in merged_mbv:
                merged_mbv[verb_key] = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    merged_mbv[verb_key].setdefault(k, []).append(float(v))

    return {
        verb_key: {metric_name: sum(values) / len(values) for metric_name, values in metric_lists.items()}
        for verb_key, metric_lists in merged_mbv.items()
    }


def _average_numeric_overall_metrics(metrics_by_verbalizer: dict[str, dict[str, float]]) -> dict[str, float]:
    overall_metrics: dict[str, float] = {}
    if not metrics_by_verbalizer:
        return overall_metrics

    all_metrics = list(metrics_by_verbalizer.values())
    for k in all_metrics[0]:
        if isinstance(all_metrics[0][k], (int, float)):
            values = [float(m[k]) for m in all_metrics if k in m]
            if values:
                overall_metrics[k] = sum(values) / len(values)
    return overall_metrics


def run_number_prediction(
    model,
    tokenizer,
    device,
    output_dir: str,
    model_name: str,
    verbalizer_lora_paths: list[str],
    max_entries: int | None = None,
) -> dict[str, Any]:
    from nl_probes.open_ended_eval.number_prediction import run_number_prediction_open_ended_eval

    os.makedirs(output_dir, exist_ok=True)
    return run_number_prediction_open_ended_eval(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=output_dir,
        verbalizer_lora_paths=verbalizer_lora_paths,
        max_entries=max_entries,
    )


def run_mmlu_prediction(
    model,
    tokenizer,
    device,
    output_dir: str,
    model_name: str,
    verbalizer_lora_paths: list[str],
    max_entries: int | None = None,
) -> dict[str, Any]:
    from nl_probes.open_ended_eval.mmlu_prediction import (
        AO_PRE_ANSWER_PROMPTS,
        AO_POST_ANSWER_PROMPTS,
        run_mmlu_prediction_open_ended_eval,
    )

    all_summaries = {}
    for mode_name, prompts in [("pre_answer", AO_PRE_ANSWER_PROMPTS), ("post_answer", AO_POST_ANSWER_PROMPTS)]:
        mode_output_dir = os.path.join(output_dir, mode_name)
        os.makedirs(mode_output_dir, exist_ok=True)

        summary = run_mmlu_prediction_open_ended_eval(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            device=device,
            output_dir=mode_output_dir,
            verbalizer_prompts=prompts,
            verbalizer_lora_paths=verbalizer_lora_paths,
            run_letter_prediction_eval=(mode_name == "pre_answer"),
            max_entries=max_entries,
        )
        all_summaries[mode_name] = summary

    # Merge per-mode binary results into a single summary
    metrics_by_verbalizer: dict[str, dict[str, Any]] = {}
    for mode_name, summary in all_summaries.items():
        for verb_key, metrics in summary.get("metrics_by_verbalizer", {}).items():
            metrics_by_verbalizer[f"{mode_name}/{verb_key}"] = metrics

    return {
        "mode_results": all_summaries,
        "metrics_by_verbalizer": metrics_by_verbalizer,
        "overall_metrics": _average_numeric_overall_metrics(metrics_by_verbalizer),
    }


def run_backtracking(
    model,
    tokenizer,
    device,
    output_dir: str,
    model_name: str,
    verbalizer_lora_paths: list[str],
    max_entries: int | None = None,
) -> dict[str, Any]:
    from nl_probes.open_ended_eval.backtracking import (
        GENERATION_KWARGS,
        run_backtracking_open_ended_eval,
    )

    os.makedirs(output_dir, exist_ok=True)
    return run_backtracking_open_ended_eval(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=output_dir,
        eval_batch_size=32,
        generation_kwargs=GENERATION_KWARGS,
        verbalizer_lora_paths=verbalizer_lora_paths,
    )


def run_backtracking_mc(
    model,
    tokenizer,
    device,
    output_dir: str,
    model_name: str,
    verbalizer_lora_paths: list[str],
    max_entries: int | None = None,
) -> dict[str, Any]:
    from nl_probes.open_ended_eval.backtracking import run_backtracking_mc_eval

    os.makedirs(output_dir, exist_ok=True)
    return run_backtracking_mc_eval(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=output_dir,
        eval_batch_size=64,
        verbalizer_lora_paths=verbalizer_lora_paths,
    )


def run_missing_info(
    model,
    tokenizer,
    device,
    output_dir: str,
    model_name: str,
    verbalizer_lora_paths: list[str],
    max_entries: int | None = None,
) -> dict[str, Any]:
    from nl_probes.open_ended_eval.missing_info import run_missing_info_open_ended_eval

    os.makedirs(output_dir, exist_ok=True)
    return run_missing_info_open_ended_eval(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=output_dir,
        eval_batch_size=32,
        verbalizer_lora_paths=verbalizer_lora_paths,
    )


def _run_sycophancy_with_dataset_dir(
    model,
    tokenizer,
    device,
    output_dir: str,
    model_name: str,
    verbalizer_lora_paths: list[str],
    dataset_dir: str,
    max_entries: int | None = None,
) -> dict[str, Any]:
    """Run sycophancy eval on both cot and no_cot modes, average metrics."""
    from nl_probes.open_ended_eval.sycophancy import run_sycophancy_open_ended_eval

    all_mode_summaries: dict[str, Any] = {}
    for mode in ("no_cot", "cot"):
        mode_output_dir = os.path.join(output_dir, mode)
        os.makedirs(mode_output_dir, exist_ok=True)

        summary = run_sycophancy_open_ended_eval(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            device=device,
            output_dir=mode_output_dir,
            mode=mode,
            verbalizer_lora_paths=verbalizer_lora_paths,
            dataset_dir=dataset_dir,
        )
        all_mode_summaries[mode] = summary

    averaged_mbv = _average_numeric_metrics_by_verbalizer(
        {mode_name: summary.get("metrics_by_verbalizer", {}) for mode_name, summary in all_mode_summaries.items()}
    )

    return {
        "mode_results": all_mode_summaries,
        "metrics_by_verbalizer": averaged_mbv,
        "overall_metrics": _average_numeric_overall_metrics(averaged_mbv),
    }


def run_sycophancy(
    model,
    tokenizer,
    device,
    output_dir: str,
    model_name: str,
    verbalizer_lora_paths: list[str],
    max_entries: int | None = None,
) -> dict[str, Any]:
    """Run original sycophancy eval (A/B dilemmas)."""
    from nl_probes.open_ended_eval.sycophancy import DATASET_DIR
    return _run_sycophancy_with_dataset_dir(
        model, tokenizer, device, output_dir, model_name, verbalizer_lora_paths,
        dataset_dir=DATASET_DIR,
    )


def run_sycophancy_aita(
    model,
    tokenizer,
    device,
    output_dir: str,
    model_name: str,
    verbalizer_lora_paths: list[str],
    max_entries: int | None = None,
) -> dict[str, Any]:
    """Run AITA sycophancy eval (Reddit AITA posts).

    WIP: This eval is in progress and has not been verified by Adam.
    """
    from nl_probes.open_ended_eval.sycophancy import DATASET_DIR_AITA
    return _run_sycophancy_with_dataset_dir(
        model, tokenizer, device, output_dir, model_name, verbalizer_lora_paths,
        dataset_dir=DATASET_DIR_AITA,
    )


def run_eval_awareness(
    model,
    tokenizer,
    device,
    output_dir: str,
    model_name: str,
    verbalizer_lora_paths: list[str],
    max_entries: int | None = None,
) -> dict[str, Any]:
    from nl_probes.open_ended_eval.eval_awareness import run_eval_awareness_open_ended_eval

    os.makedirs(output_dir, exist_ok=True)
    return run_eval_awareness_open_ended_eval(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=output_dir,
        verbalizer_lora_paths=verbalizer_lora_paths,
    )


def _flatten_system_prompt_qa_result(result: dict[str, Any]) -> dict[str, Any]:
    """Add top-level metrics_by_verbalizer from per-mode results for the comparison table."""
    merged_mbv: dict[str, dict[str, Any]] = {}
    for mode_name, mode_result in result.get("mode_results", {}).items():
        for verb_key, metrics in mode_result.get("metrics_by_verbalizer", {}).items():
            merged_mbv[f"{mode_name}/{verb_key}"] = metrics
    result["metrics_by_verbalizer"] = merged_mbv
    return result


def run_system_prompt_qa_hidden(
    model,
    tokenizer,
    device,
    output_dir: str,
    model_name: str,
    verbalizer_lora_paths: list[str],
    max_entries: int | None = None,
) -> dict[str, Any]:
    from nl_probes.open_ended_eval.system_prompt_qa import (
        VERBALIZER_PROMPTS_HIDDEN_INSTRUCTION,
        run_system_prompt_qa_open_ended_eval,
    )

    os.makedirs(output_dir, exist_ok=True)
    result = run_system_prompt_qa_open_ended_eval(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=output_dir,
        dataset_path="data_pipelines/system_prompt_qa/hidden_instruction_eval_dataset.json",
        verbalizer_prompts=VERBALIZER_PROMPTS_HIDDEN_INSTRUCTION,
        verbalizer_lora_paths=verbalizer_lora_paths,
        modes=("user_and_assistant",),
    )
    return _flatten_system_prompt_qa_result(result)


def run_system_prompt_qa_latentqa(
    model,
    tokenizer,
    device,
    output_dir: str,
    model_name: str,
    verbalizer_lora_paths: list[str],
    max_entries: int | None = None,
) -> dict[str, Any]:
    from nl_probes.open_ended_eval.system_prompt_qa import (
        VERBALIZER_PROMPTS_SYSTEM_PROMPT_QA,
        run_system_prompt_qa_open_ended_eval,
    )

    os.makedirs(output_dir, exist_ok=True)
    result = run_system_prompt_qa_open_ended_eval(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=output_dir,
        dataset_path="data_pipelines/system_prompt_qa/latentqa_eval_dataset.json",
        verbalizer_prompts=VERBALIZER_PROMPTS_SYSTEM_PROMPT_QA,
        verbalizer_lora_paths=verbalizer_lora_paths,
        modes=("user_and_assistant",),
    )
    return _flatten_system_prompt_qa_result(result)


def run_system_prompt_qa_hidden_bias(
    model,
    tokenizer,
    device,
    output_dir: str,
    model_name: str,
    verbalizer_lora_paths: list[str],
    max_entries: int | None = None,
) -> dict[str, Any]:
    from nl_probes.open_ended_eval.system_prompt_qa import (
        VERBALIZER_PROMPTS_HIDDEN_BIAS,
        hidden_bias_dataset_path,
        run_system_prompt_qa_open_ended_eval,
    )

    os.makedirs(output_dir, exist_ok=True)
    result = run_system_prompt_qa_open_ended_eval(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=output_dir,
        dataset_path=hidden_bias_dataset_path(model_name),
        verbalizer_prompts=VERBALIZER_PROMPTS_HIDDEN_BIAS,
        verbalizer_lora_paths=verbalizer_lora_paths,
        modes=("user_and_assistant",),
    )
    return _flatten_system_prompt_qa_result(result)


def run_taboo(
    model,
    tokenizer,
    device,
    output_dir: str,
    model_name: str,
    verbalizer_lora_paths: list[str],
    max_entries: int | None = None,
) -> dict[str, Any]:
    from nl_probes.open_ended_eval.taboo import (
        get_default_taboo_model_settings,
        run_taboo_open_ended_eval,
    )

    settings = get_default_taboo_model_settings(model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_json_template = os.path.join(output_dir, "taboo_results_open_{lora}.json")

    return run_taboo_open_ended_eval(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        target_lora_suffixes=settings["target_lora_suffixes"],
        target_lora_path_template=settings["target_lora_path_template"],
        verbalizer_lora_paths=verbalizer_lora_paths,
        output_json_template=output_json_template,
        prompt_type="all_direct",
        dataset_type="test",
        truncated=False,
        segment_start=settings["segment_start"],
        preferred_token_position=settings["preferred_token_position"],
    )


def run_personaqa(
    model,
    tokenizer,
    device,
    output_dir: str,
    model_name: str,
    verbalizer_lora_paths: list[str],
    max_entries: int | None = None,
) -> dict[str, Any]:
    from nl_probes.open_ended_eval.personaqa import (
        get_default_personaqa_model_settings,
        run_personaqa_open_ended_eval,
    )

    settings = get_default_personaqa_model_settings(model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_json_template = os.path.join(output_dir, "personaqa_open_{lora}.json")

    return run_personaqa_open_ended_eval(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        target_lora_suffixes=settings["target_lora_suffixes"],
        target_lora_path_template=settings["target_lora_path_template"],
        verbalizer_lora_paths=verbalizer_lora_paths,
        output_json_template=output_json_template,
        segment_start=settings["segment_start"],
        preferred_token_position=settings["preferred_token_position"],
    )


# ---------------------------------------------------------------------------
# Eval registry
# ---------------------------------------------------------------------------

EVALS = [
    ("taboo", run_taboo),
    ("personaqa", run_personaqa),
    ("number_prediction", run_number_prediction),
    ("mmlu_prediction", run_mmlu_prediction),
    ("backtracking", run_backtracking),
    ("backtracking_mc", run_backtracking_mc),
    ("missing_info", run_missing_info),
    ("sycophancy", run_sycophancy),
    ("sycophancy_aita", run_sycophancy_aita),
    ("eval_awareness", run_eval_awareness),
    ("system_prompt_qa_hidden", run_system_prompt_qa_hidden),
    ("system_prompt_qa_latentqa", run_system_prompt_qa_latentqa),
    ("system_prompt_qa_hidden_bias", run_system_prompt_qa_hidden_bias),
]

DEFAULT_INCLUDE = [name for name, _ in EVALS if name not in {"taboo", "personaqa", "sycophancy_aita"}]


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_all_evals(
    *,
    model,
    tokenizer,
    device,
    model_name: str,
    output_dir: str,
    verbalizer_lora_paths: list[str],
    include: list[str] | None = None,
    max_entries: int | None = None,
) -> dict[str, Any]:
    """Run open-ended evals sequentially, returning a dict of summaries.

    Args:
        include: Which evals to run. Defaults to DEFAULT_INCLUDE (all except taboo/personaqa).
    """
    if include is None:
        include = DEFAULT_INCLUDE
    eval_names_to_run = set(include)

    all_summaries: dict[str, Any] = {}

    for eval_name, eval_fn in EVALS:
        if eval_name not in eval_names_to_run:
            continue

        print(f"\n{'=' * 70}")
        print(f"  RUNNING EVAL: {eval_name}")
        print(f"{'=' * 70}\n")

        eval_output_dir = os.path.join(output_dir, eval_name)
        start = time.perf_counter()
        summary = eval_fn(
            model,
            tokenizer,
            device,
            eval_output_dir,
            model_name=model_name,
            verbalizer_lora_paths=verbalizer_lora_paths,
            max_entries=max_entries,
        )
        elapsed = time.perf_counter() - start

        summary["elapsed_seconds"] = elapsed
        all_summaries[eval_name] = summary

        # Save per-eval summary
        os.makedirs(output_dir, exist_ok=True)
        eval_summary_path = os.path.join(output_dir, f"{eval_name}_summary.json")
        with open(eval_summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  {eval_name} done in {elapsed:.0f}s")

    return all_summaries


def print_results_comparison(all_summaries: dict[str, Any]) -> None:
    """Print a comparison table of results across evals."""
    print(f"\n{'=' * 70}")
    print("RESULTS COMPARISON")
    print(f"{'=' * 70}")
    for eval_name, summary in all_summaries.items():
        print(f"\n--- {eval_name} ---")
        mbv = summary.get("metrics_by_verbalizer", {})
        for verb_key, metrics in mbv.items():
            if "roc_auc" in metrics:
                print(f"  {verb_key}: roc_auc={metrics['roc_auc']:.3f}, accuracy={metrics.get('accuracy_at_zero', 0):.3f}")
            elif "mean_specificity" in metrics:
                print(
                    f"  {verb_key}: specificity={metrics['mean_specificity']:.2f}, correctness={metrics['mean_correctness']:.2f}"
                )
            elif "matches_model_answer_rate" in metrics:
                print(f"  {verb_key}: model_match={metrics['matches_model_answer_rate']:.3f}")
            elif "accuracy" in metrics:
                print(f"  {verb_key}: accuracy={metrics['accuracy']:.3f}")
            else:
                print(
                    f"  {verb_key}: {json.dumps({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, indent=None)}"
                )


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

OUTPUT_DIR = "experiments/all_eval_summaries"


if __name__ == "__main__":
    import torch

    from nl_probes.open_ended_eval.eval_runner import STANDARD_VERBALIZER_LORAS
    from nl_probes.utils.common import load_model, load_tokenizer

    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g. Qwen/Qwen3-8B, Qwen/Qwen3-14B)",
    )
    parser.add_argument(
        "--verbalizer-lora",
        type=str,
        action="append",
        default=None,
        help="Verbalizer LoRA path(s) to evaluate. Can be specified multiple times. "
        "Defaults to STANDARD_VERBALIZER_LORAS if not provided.",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="*",
        default=None,
        help=f"Eval names to run. Defaults to {DEFAULT_INCLUDE}.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory for results. Defaults to {OUTPUT_DIR}.",
    )
    args = parser.parse_args()

    model_name = args.model
    lora_paths = args.verbalizer_lora or list(STANDARD_VERBALIZER_LORAS)
    include = args.include if args.include is not None else None
    output_dir = args.output_dir if args.output_dir is not None else OUTPUT_DIR

    random.seed(42)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading tokenizer: {model_name}")
    tokenizer = load_tokenizer(model_name)
    print(f"Loading model: {model_name} on {device} with dtype={dtype}")
    model = load_model(model_name, dtype)
    model.eval()
    print(f"Verbalizer LoRA paths: {lora_paths}")

    all_summaries = run_all_evals(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=model_name,
        output_dir=output_dir,
        verbalizer_lora_paths=lora_paths,
        include=include,
    )

    combined_path = os.path.join(output_dir, "all_summaries.json")
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n{'=' * 70}")
    print(f"All eval summaries saved to {combined_path}")

    print_results_comparison(all_summaries)
