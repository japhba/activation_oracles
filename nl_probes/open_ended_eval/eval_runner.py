"""
Shared infrastructure for open-ended AO evals.

Each eval provides:
  - Dataset loading + prompt building (eval-specific)
  - score_fn(results, metadata) -> scored_results
  - metrics_fn(scored_results) -> metrics dict

This module handles the boilerplate: adapter loading, run_verbalizer loop,
result saving, cleanup, and the __main__ block.
"""

import asyncio
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal

import anthropic
import matplotlib
import numpy as np
import torch
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nl_probes.base_experiment as base_experiment
from nl_probes.base_experiment import (
    VerbalizerInputInfo,
    VerbalizerResults,
)
from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.dataset_utils import BinaryFeatureResult

# Standard verbalizer LoRAs for running evals. Not used as a default —
# callers must pass verbalizer_lora_paths explicitly.
STANDARD_VERBALIZER_LORAS = [
    "adamkarvonen/checkpoints_latentqa_cls_on_policy_Qwen3-8B",
    "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
]

# Type aliases for the eval-specific callables
ScoreFn = Callable[[list[VerbalizerResults], list[dict[str, Any]]], list[dict[str, Any]]]
MetricsFn = Callable[[list[dict[str, Any]]], dict[str, Any]]
BinaryMetricsFn = Callable[[list[dict[str, Any]]], dict[str, Any]]
PrintSampleFn = Callable[[list[dict[str, Any]]], None] | None
BaselineBackend = Literal["hf", "vllm"]
BaselineInferenceMode = Literal["rollout", "binary_yes_no"]
BaselinePromptVariant = Literal["full_context", "selected_span"]


@dataclass
class TextBaselineInput:
    prompt_text: str
    ground_truth: str
    prompt_name: str
    variant: BaselinePromptVariant


@dataclass
class TextBaselineGenerationResult:
    prompt_text: str
    response_text: str


@dataclass
class ClaudeBaselineInput:
    """Input for a Claude API baseline eval entry."""
    messages: list[dict[str, str]]  # Full message list (few-shot + actual question)
    ground_truth: str
    prompt_name: str


@dataclass
class ClaudeBaselineResult:
    """Result from a single Claude API baseline call."""
    messages: list[dict[str, str]]  # What was sent
    response_text: str  # What Claude returned


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def ensure_default_adapter(model: AutoModelForCausalLM) -> None:
    """Ensure model has a 'default' adapter (needed for adapter switching)."""
    if not hasattr(model, "peft_config") or "default" not in model.peft_config:
        dummy_config = LoraConfig()
        model.add_adapter(dummy_config, adapter_name="default")


def extract_yes_no(response: str) -> str | None:
    """Extract yes/no from AO response (first word)."""
    text = response.strip().lower()
    first_word = text.split()[0] if text.split() else ""
    first_word = re.sub(r"[^a-z]", "", first_word)
    if first_word in ("yes", "no"):
        return first_word
    return None


def decode_contiguous_span_text(
    context_token_ids: list[int],
    positions: list[int],
    tokenizer: AutoTokenizer,
) -> str:
    assert len(positions) > 0, "positions must be non-empty"
    assert positions == list(range(positions[0], positions[-1] + 1)), (
        f"selected_span requires contiguous positions, got {positions}"
    )
    span_token_ids = context_token_ids[positions[0] : positions[-1] + 1]
    assert len(span_token_ids) == len(positions), (
        f"Decoded span length {len(span_token_ids)} != positions length {len(positions)}"
    )
    return tokenizer.decode(span_token_ids, skip_special_tokens=False)


def render_baseline_chat_prompt(
    *,
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
    enable_thinking: bool,
    continue_final_message: bool = False,
) -> str:
    assert not (add_generation_prompt and continue_final_message), (
        "add_generation_prompt and continue_final_message are mutually exclusive"
    )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        enable_thinking=enable_thinking,
    )


def default_layer_combination(training_config: SelfInterpTrainingConfig) -> list[int]:
    """Select which layer combination to use from a training config.

    Prefers multi-layer [25, 50, 75] if available, falls back to single-layer [50].
    """
    combos = training_config.layer_combinations

    # Multi-layer models trained on 25/50/75% layers (current standard)
    if [25, 50, 75] in combos:
        return [25, 50, 75]

    # Single-layer models trained on 50% layer only (older checkpoints, e.g. past_lens_addition)
    if [50] in combos:
        return [50]

    raise ValueError(
        f"No recognized layer combination found in {combos}. "
        f"Expected [25, 50, 75] (multi-layer) or [50] (single-layer)."
    )


def build_verbalizer_eval_config(
    model_name: str,
    training_config: SelfInterpTrainingConfig,
    eval_batch_size: int,
    generation_kwargs: dict[str, Any],
    selected_layer_combination: list[int] | None = None,
) -> base_experiment.VerbalizerEvalConfig:
    """Build a VerbalizerEvalConfig from an AO training config.

    training_config is required — layer_combinations and selected_layer_combination
    are read from it to ensure the eval uses the same layers the AO was trained on.
    """
    layer_combinations = training_config.layer_combinations
    if selected_layer_combination is None:
        selected_layer_combination = default_layer_combination(training_config)
    assert selected_layer_combination in layer_combinations, (
        f"selected_layer_combination {selected_layer_combination} not in "
        f"training config layer_combinations {layer_combinations}"
    )

    return base_experiment.VerbalizerEvalConfig(
        model_name=model_name,
        activation_input_types=["lora"],
        eval_batch_size=eval_batch_size,
        verbalizer_generation_kwargs=generation_kwargs,
        layer_combinations=layer_combinations,
        selected_layer_combination=selected_layer_combination,
    )


def _print_metrics(metrics: dict[str, Any]) -> None:
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.3f}")
        else:
            print(f"    {k}: {v}")


def _verbalizer_output_identifiers(verbalizer_entry: str) -> tuple[str, str]:
    """Return stable identifiers for metrics keys and output filenames.

    Using only the final path component causes collisions for common local paths
    like multiple checkpoints ending in `/final`.
    """
    verbalizer_key = verbalizer_entry
    file_stem = base_experiment.sanitize_lora_name(verbalizer_entry).replace("/", "_")
    return verbalizer_key, file_stem


def _load_adapter_and_build_config(
    model: AutoModelForCausalLM,
    verbalizer_entry: str,
    model_name: str,
    eval_batch_size: int,
    generation_kwargs: dict[str, Any],
) -> tuple[str, base_experiment.VerbalizerEvalConfig]:
    """Load a verbalizer adapter and build its eval config."""
    sanitized_name, training_config = base_experiment.load_oracle_adapter(model, verbalizer_entry)
    config = build_verbalizer_eval_config(
        model_name=model_name,
        training_config=training_config,
        eval_batch_size=eval_batch_size,
        generation_kwargs=generation_kwargs,
    )
    base_experiment.assert_training_config_matches_verbalizer_eval_config(config, training_config)
    return sanitized_name, config


def get_first_ao_response(result: VerbalizerResults) -> str | None:
    """Extract the first AO response from a VerbalizerResults."""
    if not result.responses:
        return None
    return result.responses[0]


def run_default_eval(
    *,
    eval_name: str,
    run_eval_fn: Callable[..., dict[str, Any]],
    run_eval_kwargs: dict[str, Any],
    model_name: str,
) -> None:
    """
    Standard __main__ boilerplate: set seeds, load model, run eval, print summary.

    run_eval_fn is called with (model_name, model, tokenizer, device, **run_eval_kwargs).
    It should accept those as keyword arguments and return the summary dict.

    Callers must provide verbalizer_lora_paths in run_eval_kwargs — no silent
    defaults are applied.
    """
    model_name_str = model_name.split("/")[-1].replace(".", "_")

    random.seed(42)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # Set default output_dir if not provided
    if "output_dir" not in run_eval_kwargs:
        output_dir = f"experiments/{eval_name}_eval_results/{model_name_str}"
        os.makedirs(output_dir, exist_ok=True)
        run_eval_kwargs["output_dir"] = output_dir

    print(f"Loading tokenizer: {model_name}")
    tokenizer = load_tokenizer(model_name)
    print(f"Loading model: {model_name} on {device} with dtype={dtype}")
    model = load_model(model_name, dtype)
    model.eval()

    summary = run_eval_fn(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        **run_eval_kwargs,
    )

    print(f"\n=== {eval_name.replace('_', ' ').title()} Eval Summary ===")
    print(json.dumps(summary, indent=2))


def run_default_text_baseline_eval(
    *,
    eval_name: str,
    run_eval_fn: Callable[..., dict[str, Any]],
    run_eval_kwargs: dict[str, Any],
    model_name: str,
    backend: BaselineBackend,
) -> None:
    """
    Standard entrypoint for text-baseline evals.

    Loads the tokenizer in all cases. For HF backends, also loads the base model.
    For vLLM rollout, avoids loading a parallel HF model onto the same GPU.
    """
    model_name_str = model_name.split("/")[-1].replace(".", "_")

    random.seed(42)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    if "output_dir" not in run_eval_kwargs:
        output_dir = f"experiments/{eval_name}_results/{model_name_str}"
        os.makedirs(output_dir, exist_ok=True)
        run_eval_kwargs["output_dir"] = output_dir

    print(f"Loading tokenizer: {model_name}")
    tokenizer = load_tokenizer(model_name)

    hf_model = None
    device = None
    if backend == "hf":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16
        print(f"Loading model: {model_name} on {device} with dtype={dtype}")
        hf_model = load_model(model_name, dtype)
        hf_model.eval()

    summary = run_eval_fn(
        model_name=model_name,
        tokenizer=tokenizer,
        hf_model=hf_model,
        device=device,
        backend=backend,
        **run_eval_kwargs,
    )

    print(f"\n=== {eval_name.replace('_', ' ').title()} Summary ===")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Generation eval loop
# ---------------------------------------------------------------------------


def run_verbalizer_generation_eval_loop(
    *,
    eval_name: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    model_name: str,
    eval_batch_size: int,
    generation_kwargs: dict[str, Any],
    prompt_infos: list[VerbalizerInputInfo],
    entry_metadata: list[dict[str, Any]],
    score_fn: ScoreFn,
    metrics_fn: MetricsFn,
    num_entries: int,
    verbalizer_lora_paths: list[str],
    output_dir: str | None = None,
    print_sample_fn: PrintSampleFn = None,
    extra_output_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run generation-based eval loop.

    Iterates verbalizer LoRAs, runs run_verbalizer (text generation),
    scores with score_fn, computes metrics, saves results.
    """
    ensure_default_adapter(model)
    model.eval()

    all_scored_results: list[dict[str, Any]] = []
    metrics_by_verbalizer: dict[str, dict[str, Any]] = {}

    for verbalizer_entry in verbalizer_lora_paths:
        sanitized_verbalizer_name, loop_config = _load_adapter_and_build_config(
            model, verbalizer_entry, model_name, eval_batch_size, generation_kwargs,
        )

        print(f"Running {eval_name} eval with verbalizer: {verbalizer_entry}")
        verbalizer_key, lora_name = _verbalizer_output_identifiers(verbalizer_entry)

        results = base_experiment.run_verbalizer(
            model=model,
            tokenizer=tokenizer,
            verbalizer_prompt_infos=prompt_infos,
            verbalizer_lora_path=sanitized_verbalizer_name,
            target_lora_path=None,
            config=loop_config,
            device=device,
        )

        scored_results = score_fn(results, entry_metadata)
        metrics: dict[str, Any] | None = None
        if scored_results:
            metrics = metrics_fn(scored_results)
            metrics_by_verbalizer[verbalizer_key] = metrics
            print(f"\n  Metrics for {verbalizer_key}:")
            _print_metrics(metrics)

            if print_sample_fn is not None:
                print_sample_fn(scored_results)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{eval_name}_{lora_name}.json")
            output_data: dict[str, Any] = {
                "config": asdict(loop_config),
                "verbalizer": verbalizer_entry,
                "num_entries": num_entries,
                "scored_results": scored_results,
                "metrics": metrics if scored_results else None,
                "verbalizer_results": [asdict(r) for r in results],
            }
            if extra_output_data:
                output_data.update(extra_output_data)
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"  Saved results to {output_path}")

        all_scored_results.extend(scored_results)

        if sanitized_verbalizer_name is not None:
            if sanitized_verbalizer_name in model.peft_config:
                model.delete_adapter(sanitized_verbalizer_name)

    overall_metrics = metrics_fn(all_scored_results) if all_scored_results else {}
    return {
        "overall_metrics": overall_metrics,
        "metrics_by_verbalizer": metrics_by_verbalizer,
        "num_entries": num_entries,
        "num_scored": len(all_scored_results),
    }


# ---------------------------------------------------------------------------
# Binary (logit) scoring eval loop
# ---------------------------------------------------------------------------

YES_NO_CANDIDATE_VARIANTS: dict[str, list[str]] = {
    "yes": ["yes", " yes", "Yes", " Yes", "YES", " YES", "\nyes", "\nYes", "\nYES"],
    "no": ["no", " no", "No", " No", "NO", " NO", "\nno", "\nNo", "\nNO"],
}


def build_yes_no_candidate_token_groups(tokenizer: AutoTokenizer) -> dict[str, list[int]]:
    """Collect single-token yes/no variants for first-token AO scoring."""
    token_groups: dict[str, list[int]] = {}
    for label, variants in YES_NO_CANDIDATE_VARIANTS.items():
        token_ids: list[int] = []
        for text in variants:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in token_ids:
                token_ids.append(int(ids[0]))
        if not token_ids:
            raise ValueError(f"Tokenizer had no single-token variants for label '{label}'")
        token_groups[label] = token_ids
    return token_groups


def describe_candidate_token_groups(
    tokenizer: AutoTokenizer,
    candidate_token_groups: dict[str, list[int]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        label: [
            {
                "token_id": int(token_id),
                "token_text": tokenizer.decode([token_id], skip_special_tokens=False),
            }
            for token_id in token_ids
        ]
        for label, token_ids in candidate_token_groups.items()
    }


def compute_roc_curve_data(
    labels: list[int],
    scores: list[float],
) -> dict[str, Any] | None:
    if not labels:
        return None

    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(scores, dtype=np.float64)
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]

    distinct_indices = np.where(np.diff(y_score))[0]
    threshold_indices = np.r_[distinct_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_indices]
    fps = 1 + threshold_indices - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[np.inf, y_score[threshold_indices]]

    tpr = tps / positives
    fpr = fps / negatives
    auc = float(np.trapz(tpr, fpr))

    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": auc,
        "positives": positives,
        "negatives": negatives,
    }


def score_binary_yes_no_results(
    results: list[BinaryFeatureResult],
    entry_metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []

    for result, meta in zip(results, entry_metadata, strict=True):
        ground_truth = result.meta_info["ground_truth"].strip().lower()
        if ground_truth not in ("yes", "no"):
            continue

        yes_score = float(result.candidate_scores["yes"])
        no_score = float(result.candidate_scores["no"])
        margin = yes_score - no_score
        predicted_answer = "yes" if margin >= 0 else "no"

        scored.append({
            **meta,
            "act_key": result.meta_info["act_key"],
            "ground_truth": ground_truth,
            "binary_label": 1 if ground_truth == "yes" else 0,
            "yes_score": yes_score,
            "no_score": no_score,
            "margin_yes_minus_no": margin,
            "predicted_answer": predicted_answer,
            "is_correct": predicted_answer == ground_truth,
            "argmax_token_id": result.argmax_token_id,
            "argmax_token_text": result.argmax_token_text,
            "argmax_logit": result.argmax_logit,
        })

    return scored


def _compute_binary_metric_block(scored_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(scored_results)
    metrics: dict[str, Any] = {
        "total": total,
        "correct_at_zero": sum(1 for r in scored_results if r["is_correct"]),
        "accuracy_at_zero": (
            sum(1 for r in scored_results if r["is_correct"]) / total if total > 0 else 0.0
        ),
    }

    if total > 0:
        positive_rows = [r for r in scored_results if r["binary_label"] == 1]
        negative_rows = [r for r in scored_results if r["binary_label"] == 0]
        metrics["positive_rate"] = len(positive_rows) / total
        metrics["mean_margin_when_yes"] = (
            sum(r["margin_yes_minus_no"] for r in positive_rows) / len(positive_rows)
            if positive_rows else 0.0
        )
        metrics["mean_margin_when_no"] = (
            sum(r["margin_yes_minus_no"] for r in negative_rows) / len(negative_rows)
            if negative_rows else 0.0
        )

    roc_data = compute_roc_curve_data(
        labels=[int(r["binary_label"]) for r in scored_results],
        scores=[float(r["margin_yes_minus_no"]) for r in scored_results],
    )
    if roc_data is not None:
        metrics["roc_auc"] = float(roc_data["auc"])
        metrics["num_positive"] = int(roc_data["positives"])
        metrics["num_negative"] = int(roc_data["negatives"])

    return metrics


MAX_GROUP_VALUES = 20


def _append_group_metrics(
    metrics: dict[str, Any],
    scored_results: list[dict[str, Any]],
    field_name: str,
    prefix: str,
) -> None:
    field_values = sorted({r[field_name] for r in scored_results if field_name in r and r[field_name] is not None})
    if not field_values or len(field_values) > MAX_GROUP_VALUES:
        return

    for field_value in field_values:
        subset = [r for r in scored_results if r.get(field_name) == field_value]
        subset_metrics = _compute_binary_metric_block(subset)
        for key, value in subset_metrics.items():
            metrics[f"{prefix}_{field_value}_{key}"] = value


def compute_binary_yes_no_metrics(scored_results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = _compute_binary_metric_block(scored_results)
    _append_group_metrics(metrics, scored_results, "prompt_name", "prompt")
    _append_group_metrics(metrics, scored_results, "condition", "cond")
    _append_group_metrics(metrics, scored_results, "baseline_variant", "variant")
    return metrics


def save_binary_yes_no_roc_plot(
    scored_results: list[dict[str, Any]],
    output_path: str,
    title: str,
) -> str | None:
    overall_curve = compute_roc_curve_data(
        labels=[int(r["binary_label"]) for r in scored_results],
        scores=[float(r["margin_yes_minus_no"]) for r in scored_results],
    )
    if overall_curve is None:
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.plot(
        overall_curve["fpr"],
        overall_curve["tpr"],
        label=f"overall (AUC={overall_curve['auc']:.3f})",
        linewidth=2.5,
        color="black",
    )

    prompt_names = sorted({r["prompt_name"] for r in scored_results if "prompt_name" in r})
    if len(prompt_names) > 1:
        for prompt_name in prompt_names:
            subset = [r for r in scored_results if r.get("prompt_name") == prompt_name]
            prompt_curve = compute_roc_curve_data(
                labels=[int(r["binary_label"]) for r in subset],
                scores=[float(r["margin_yes_minus_no"]) for r in subset],
            )
            if prompt_curve is None:
                continue
            plt.plot(
                prompt_curve["fpr"],
                prompt_curve["tpr"],
                label=f"{prompt_name} (AUC={prompt_curve['auc']:.3f})",
                linewidth=1.8,
            )

    plt.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1.5, label="chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def run_verbalizer_binary_eval_loop(
    *,
    eval_name: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    model_name: str,
    eval_batch_size: int,
    generation_kwargs: dict[str, Any],
    prompt_infos: list[VerbalizerInputInfo],
    entry_metadata: list[dict[str, Any]],
    num_entries: int,
    verbalizer_lora_paths: list[str],
    output_dir: str | None = None,
    binary_metrics_fn: BinaryMetricsFn | None = None,
) -> dict[str, Any]:
    """Run binary (logit) scoring eval loop.

    Iterates verbalizer LoRAs, runs run_verbalizer_binary_score,
    scores with score_binary_yes_no_results, computes metrics, saves ROC plots.
    """
    ensure_default_adapter(model)
    model.eval()

    candidate_token_groups = build_yes_no_candidate_token_groups(tokenizer)
    candidate_group_info = describe_candidate_token_groups(tokenizer, candidate_token_groups)

    all_scored_results: list[dict[str, Any]] = []
    metrics_by_verbalizer: dict[str, dict[str, Any]] = {}
    plot_paths_by_verbalizer: dict[str, str] = {}

    for verbalizer_entry in verbalizer_lora_paths:
        sanitized_verbalizer_name, loop_config = _load_adapter_and_build_config(
            model, verbalizer_entry, model_name, eval_batch_size, generation_kwargs,
        )

        assert len(loop_config.activation_input_types) == 1, (
            f"Binary eval requires exactly one activation_input_type, "
            f"got {loop_config.activation_input_types}. Multiple act types produce "
            f"multiple results per entry, breaking the 1:1 mapping with entry_metadata."
        )

        print(f"Running {eval_name} binary eval with verbalizer: {verbalizer_entry}")
        verbalizer_key, lora_name = _verbalizer_output_identifiers(verbalizer_entry)

        binary_results = base_experiment.run_verbalizer_binary_score(
            model=model,
            tokenizer=tokenizer,
            verbalizer_prompt_infos=prompt_infos,
            verbalizer_lora_path=sanitized_verbalizer_name,
            target_lora_path=None,
            config=loop_config,
            device=device,
            candidate_token_groups=candidate_token_groups,
        )

        scored_results = score_binary_yes_no_results(binary_results, entry_metadata)
        metrics: dict[str, Any] | None = None
        plot_path: str | None = None

        if scored_results:
            _metrics_fn = binary_metrics_fn or compute_binary_yes_no_metrics
            metrics = _metrics_fn(scored_results)
            metrics_by_verbalizer[verbalizer_key] = metrics
            print(f"\n  Binary score metrics for {verbalizer_key}:")
            _print_metrics(metrics)

            if output_dir is not None:
                plot_path = save_binary_yes_no_roc_plot(
                    scored_results,
                    output_path=os.path.join(output_dir, f"{eval_name}_{lora_name}_roc_auc.png"),
                    title=f"{eval_name} - {verbalizer_key}",
                )
                if plot_path is not None:
                    plot_paths_by_verbalizer[verbalizer_key] = plot_path
                    print(f"  Saved ROC curve to {plot_path}")

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{eval_name}_binary_{lora_name}.json")
            output_data: dict[str, Any] = {
                "config": asdict(loop_config),
                "verbalizer": verbalizer_entry,
                "num_entries": num_entries,
                "binary_score_candidate_groups": candidate_group_info,
                "binary_scored_results": scored_results,
                "binary_score_metrics": metrics if scored_results else None,
                "binary_roc_plot_path": plot_path,
            }
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"  Saved binary results to {output_path}")

        all_scored_results.extend(scored_results)

        if sanitized_verbalizer_name is not None:
            if sanitized_verbalizer_name in model.peft_config:
                model.delete_adapter(sanitized_verbalizer_name)

    _overall_metrics_fn = binary_metrics_fn or compute_binary_yes_no_metrics
    overall_metrics = _overall_metrics_fn(all_scored_results) if all_scored_results else {}
    return {
        "overall_metrics": overall_metrics,
        "metrics_by_verbalizer": metrics_by_verbalizer,
        "num_scored": len(all_scored_results),
        "candidate_token_groups": candidate_group_info,
        "plot_paths_by_verbalizer": plot_paths_by_verbalizer,
    }


def _serialize_text_baseline_results(raw_results: list[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for result in raw_results:
        if isinstance(result, TextBaselineGenerationResult):
            serialized.append(asdict(result))
            continue
        if isinstance(result, BinaryFeatureResult):
            serialized.append(result.model_dump())
            continue
        raise TypeError(f"Unsupported text baseline result type: {type(result)}")
    return serialized


@torch.no_grad()
def _run_hf_text_baseline_rollout(
    *,
    baseline_inputs: list[TextBaselineInput],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    eval_batch_size: int,
    generation_kwargs: dict[str, Any],
) -> list[TextBaselineGenerationResult]:
    assert not hasattr(model, "peft_config"), (
        "Text baseline HF path expects a base model without PEFT adapters"
    )
    model.eval()

    results: list[TextBaselineGenerationResult] = []
    for start in tqdm(
        range(0, len(baseline_inputs), eval_batch_size),
        desc="HF text baseline rollout",
    ):
        batch = baseline_inputs[start : start + eval_batch_size]
        prompt_texts = [item.prompt_text for item in batch]
        tokenized = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        output_ids = model.generate(**tokenized, **generation_kwargs)
        generated_token_ids = output_ids[:, tokenized["input_ids"].shape[1] :]
        decoded_outputs = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)

        for prompt_text, response_text in zip(prompt_texts, decoded_outputs, strict=True):
            results.append(
                TextBaselineGenerationResult(
                    prompt_text=prompt_text,
                    response_text=response_text,
                )
            )

    return results


def _build_vllm_sampling_params(generation_kwargs: dict[str, Any]) -> Any:
    import vllm

    supported_keys = {"do_sample", "temperature", "max_new_tokens", "top_p"}
    unsupported_keys = set(generation_kwargs) - supported_keys
    assert not unsupported_keys, (
        f"Unsupported vLLM rollout generation kwargs: {sorted(unsupported_keys)}"
    )
    assert "temperature" in generation_kwargs, "generation_kwargs must specify temperature"
    assert "max_new_tokens" in generation_kwargs, "generation_kwargs must specify max_new_tokens"

    if "do_sample" in generation_kwargs:
        is_sampling = float(generation_kwargs["temperature"]) > 0.0
        assert bool(generation_kwargs["do_sample"]) == is_sampling, (
            f"do_sample={generation_kwargs['do_sample']} is inconsistent with "
            f"temperature={generation_kwargs['temperature']}"
        )

    sampling_kwargs: dict[str, Any] = {
        "temperature": float(generation_kwargs["temperature"]),
        "max_tokens": int(generation_kwargs["max_new_tokens"]),
    }
    if "top_p" in generation_kwargs:
        sampling_kwargs["top_p"] = float(generation_kwargs["top_p"])
    return vllm.SamplingParams(**sampling_kwargs)


def _run_vllm_text_baseline_rollout(
    *,
    baseline_inputs: list[TextBaselineInput],
    model_name: str,
    generation_kwargs: dict[str, Any],
    vllm_lora_path: str | None = None,
    enforce_eager: bool = True,
) -> list[TextBaselineGenerationResult]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    import vllm

    sampling_params = _build_vllm_sampling_params(generation_kwargs)
    prompt_texts = [item.prompt_text for item in baseline_inputs]

    lora_request = None
    if vllm_lora_path is not None:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("text_sft_lora", 1, lora_path=vllm_lora_path)

    llm = vllm.LLM(
        model=model_name,
        tensor_parallel_size=1,
        enforce_eager=enforce_eager,
        enable_lora=vllm_lora_path is not None,
        max_lora_rank=256,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
    )

    outputs = llm.generate(
        prompt_texts,
        sampling_params=sampling_params,
        lora_request=lora_request,
        use_tqdm=True,
    )

    results = [
        TextBaselineGenerationResult(
            prompt_text=prompt_text,
            response_text=output.outputs[0].text,
        )
        for prompt_text, output in zip(prompt_texts, outputs, strict=True)
    ]

    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def _logsumexp_list(values: list[float]) -> float:
    assert len(values) > 0, "values must be non-empty"
    max_value = max(values)
    return float(max_value + np.log(sum(np.exp(value - max_value) for value in values)))


def _run_vllm_text_baseline_binary_yes_no(
    *,
    baseline_inputs: list[TextBaselineInput],
    model_name: str,
    tokenizer: AutoTokenizer,
    vllm_lora_path: str | None = None,
    enforce_eager: bool = True,
) -> tuple[list[BinaryFeatureResult], dict[str, list[dict[str, Any]]]]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    import vllm

    candidate_token_groups = build_yes_no_candidate_token_groups(tokenizer)
    candidate_group_info = describe_candidate_token_groups(tokenizer, candidate_token_groups)
    candidate_token_id_sets = {
        label: set(token_ids) for label, token_ids in candidate_token_groups.items()
    }

    lora_request = None
    if vllm_lora_path is not None:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("text_sft_lora", 1, lora_path=vllm_lora_path)

    llm = vllm.LLM(
        model=model_name,
        tensor_parallel_size=1,
        enforce_eager=enforce_eager,
        enable_lora=vllm_lora_path is not None,
        max_lora_rank=256,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
    )
    sampling_params = vllm.SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=20,
    )

    outputs = llm.generate(
        [item.prompt_text for item in baseline_inputs],
        sampling_params=sampling_params,
        lora_request=lora_request,
        use_tqdm=True,
    )

    results: list[BinaryFeatureResult] = []
    for idx, (item, output) in enumerate(zip(baseline_inputs, outputs, strict=True)):
        assert len(output.outputs) > 0, "vLLM returned no outputs for a binary text-baseline prompt"
        token_logprobs = output.outputs[0].logprobs
        assert token_logprobs is not None and len(token_logprobs) > 0, (
            "vLLM returned no first-token logprobs for a binary text-baseline prompt"
        )
        logprobs_dict = token_logprobs[0]

        candidate_token_scores: dict[str, list[dict[str, Any]]] = {"yes": [], "no": []}
        candidate_scores: dict[str, float] = {}

        for label, token_id_set in candidate_token_id_sets.items():
            label_logprobs: list[float] = []
            for token_id, logprob_obj in logprobs_dict.items():
                if int(token_id) not in token_id_set:
                    continue
                label_logprobs.append(float(logprob_obj.logprob))
                candidate_token_scores[label].append(
                    {
                        "token_id": int(token_id),
                        "token_text": logprob_obj.decoded_token,
                        "logit": float(logprob_obj.logprob),
                    }
                )
            candidate_scores[label] = _logsumexp_list(label_logprobs) if label_logprobs else -100.0

        argmax_token_id, argmax_logprob_obj = max(
            logprobs_dict.items(),
            key=lambda pair: float(pair[1].logprob),
        )
        results.append(
            BinaryFeatureResult(
                feature_idx=idx,
                candidate_scores=candidate_scores,
                candidate_token_scores=candidate_token_scores,
                argmax_token_id=int(argmax_token_id),
                argmax_token_text=argmax_logprob_obj.decoded_token,
                argmax_logit=float(argmax_logprob_obj.logprob),
                prompt=item.prompt_text,
                meta_info={
                    "act_key": "text_baseline",
                    "ground_truth": item.ground_truth,
                    "prompt_name": item.prompt_name,
                    "baseline_variant": item.variant,
                },
            )
        )

    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results, candidate_group_info


@torch.no_grad()
def _run_hf_text_baseline_binary_yes_no(
    *,
    baseline_inputs: list[TextBaselineInput],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    eval_batch_size: int,
) -> tuple[list[BinaryFeatureResult], dict[str, list[dict[str, Any]]]]:
    assert not hasattr(model, "peft_config"), (
        "Text baseline HF path expects a base model without PEFT adapters"
    )
    model.eval()

    candidate_token_groups = build_yes_no_candidate_token_groups(tokenizer)
    candidate_group_info = describe_candidate_token_groups(tokenizer, candidate_token_groups)

    results: list[BinaryFeatureResult] = []
    for start in tqdm(
        range(0, len(baseline_inputs), eval_batch_size),
        desc="HF text baseline binary scoring",
    ):
        batch = baseline_inputs[start : start + eval_batch_size]
        prompt_texts = [item.prompt_text for item in batch]
        tokenized = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        outputs = model(**tokenized)
        next_token_logits = outputs.logits[:, -1, :].float()
        batch_argmax_ids = next_token_logits.argmax(dim=-1)

        for row_idx, item in enumerate(batch):
            candidate_scores: dict[str, float] = {}
            candidate_token_scores: dict[str, list[dict[str, Any]]] = {}

            for label, token_ids in candidate_token_groups.items():
                token_logits = next_token_logits[row_idx, token_ids]
                candidate_scores[label] = float(torch.logsumexp(token_logits, dim=0).item())
                candidate_token_scores[label] = [
                    {
                        "token_id": int(token_id),
                        "token_text": tokenizer.decode([token_id], skip_special_tokens=False),
                        "logit": float(token_logit.item()),
                    }
                    for token_id, token_logit in zip(token_ids, token_logits, strict=True)
                ]

            argmax_id = int(batch_argmax_ids[row_idx].item())
            results.append(
                BinaryFeatureResult(
                    feature_idx=start + row_idx,
                    candidate_scores=candidate_scores,
                    candidate_token_scores=candidate_token_scores,
                    argmax_token_id=argmax_id,
                    argmax_token_text=tokenizer.decode([argmax_id], skip_special_tokens=False),
                    argmax_logit=float(next_token_logits[row_idx, argmax_id].item()),
                    prompt=item.prompt_text,
                    meta_info={
                        "act_key": "text_baseline",
                        "ground_truth": item.ground_truth,
                        "prompt_name": item.prompt_name,
                        "baseline_variant": item.variant,
                    },
                )
            )

    return results, candidate_group_info


def run_text_baseline_eval_loop(
    *,
    eval_name: str,
    baseline_inputs: list[TextBaselineInput],
    entry_metadata: list[dict[str, Any]],
    backend: BaselineBackend,
    inference_mode: BaselineInferenceMode,
    model_name: str,
    tokenizer: AutoTokenizer,
    generation_kwargs: dict[str, Any],
    score_fn: Callable[[list[Any], list[dict[str, Any]]], list[dict[str, Any]]],
    metrics_fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
    output_dir: str | None = None,
    hf_model: AutoModelForCausalLM | None = None,
    device: torch.device | None = None,
    eval_batch_size: int = 64,
    num_entries: int | None = None,
    print_sample_fn: PrintSampleFn = None,
    extra_output_data: dict[str, Any] | None = None,
    vllm_lora_path: str | None = None,
    enforce_eager: bool = True,
) -> dict[str, Any]:
    assert len(baseline_inputs) == len(entry_metadata), (
        f"baseline_inputs length {len(baseline_inputs)} != entry_metadata length {len(entry_metadata)}"
    )
    if num_entries is None:
        num_entries = len(baseline_inputs)

    candidate_group_info: dict[str, list[dict[str, Any]]] | None = None
    if inference_mode == "rollout":
        if backend == "hf":
            assert hf_model is not None, "hf_model is required for HF rollout text baseline"
            assert device is not None, "device is required for HF rollout text baseline"
            raw_results = _run_hf_text_baseline_rollout(
                baseline_inputs=baseline_inputs,
                model=hf_model,
                tokenizer=tokenizer,
                device=device,
                eval_batch_size=eval_batch_size,
                generation_kwargs=generation_kwargs,
            )
        elif backend == "vllm":
            raw_results = _run_vllm_text_baseline_rollout(
                baseline_inputs=baseline_inputs,
                model_name=model_name,
                generation_kwargs=generation_kwargs,
                vllm_lora_path=vllm_lora_path,
                enforce_eager=enforce_eager,
            )
        else:
            raise ValueError(f"Unsupported text baseline backend: {backend}")
    elif inference_mode == "binary_yes_no":
        if backend == "hf":
            assert hf_model is not None, "hf_model is required for HF binary text baseline"
            assert device is not None, "device is required for HF binary text baseline"
            raw_results, candidate_group_info = _run_hf_text_baseline_binary_yes_no(
                baseline_inputs=baseline_inputs,
                model=hf_model,
                tokenizer=tokenizer,
                device=device,
                eval_batch_size=eval_batch_size,
            )
        elif backend == "vllm":
            raw_results, candidate_group_info = _run_vllm_text_baseline_binary_yes_no(
                baseline_inputs=baseline_inputs,
                model_name=model_name,
                tokenizer=tokenizer,
                vllm_lora_path=vllm_lora_path,
                enforce_eager=enforce_eager,
            )
        else:
            raise ValueError(f"Unsupported text baseline backend for binary_yes_no: {backend}")
    else:
        raise ValueError(f"Unsupported inference mode: {inference_mode}")

    scored_results = score_fn(raw_results, entry_metadata)
    metrics = metrics_fn(scored_results) if scored_results else {}

    if metrics:
        print(f"\n  Text baseline metrics for {eval_name}:")
        _print_metrics(metrics)
        if print_sample_fn is not None:
            print_sample_fn(scored_results)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"{eval_name}_text_baseline_{backend}_{inference_mode}.json",
        )
        output_data: dict[str, Any] = {
            "model_name": model_name,
            "backend": backend,
            "inference_mode": inference_mode,
            "generation_kwargs": generation_kwargs,
            "num_entries": num_entries,
            "baseline_inputs": [asdict(item) for item in baseline_inputs],
            "scored_results": scored_results,
            "metrics": metrics,
            "raw_results": _serialize_text_baseline_results(raw_results),
        }
        if candidate_group_info is not None:
            output_data["binary_score_candidate_groups"] = candidate_group_info
        if extra_output_data:
            output_data.update(extra_output_data)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Saved text baseline results to {output_path}")

    summary: dict[str, Any] = {
        "overall_metrics": metrics,
        "num_entries": num_entries,
        "num_scored": len(scored_results),
        "backend": backend,
        "inference_mode": inference_mode,
    }
    if candidate_group_info is not None:
        summary["candidate_token_groups"] = candidate_group_info
    return summary


# ---------------------------------------------------------------------------
# Claude API baseline eval loop
# ---------------------------------------------------------------------------

CLAUDE_BASELINE_MAX_RETRIES = 3


async def _call_claude_single(
    client: anthropic.AsyncAnthropic,
    *,
    system_prompt: str,
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> str:
    last_error = None
    for attempt in range(CLAUDE_BASELINE_MAX_RETRIES):
        try:
            async with semaphore:
                response = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=messages,
                )
            return response.content[0].text
        except Exception as e:
            last_error = e
            if attempt < CLAUDE_BASELINE_MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
    raise last_error  # type: ignore[misc]


async def _run_claude_baseline_generation(
    inputs: list[ClaudeBaselineInput],
    *,
    system_prompt: str,
    model: str,
    max_tokens: int,
    concurrency: int,
) -> list[ClaudeBaselineResult]:
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)

    print(f"Generating {len(inputs)} Claude baseline responses with {model} (concurrency={concurrency})...")
    pbar = tqdm(total=len(inputs), desc=f"Claude baseline ({model})")

    async def _generate_one(inp: ClaudeBaselineInput) -> ClaudeBaselineResult:
        response_text = await _call_claude_single(
            client,
            system_prompt=system_prompt,
            messages=inp.messages,
            model=model,
            max_tokens=max_tokens,
            semaphore=semaphore,
        )
        pbar.update(1)
        return ClaudeBaselineResult(
            messages=inp.messages,
            response_text=response_text,
        )

    results = await asyncio.gather(
        *[_generate_one(inp) for inp in inputs],
        return_exceptions=True,
    )
    pbar.close()

    final_results: list[ClaudeBaselineResult] = []
    errors = 0
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Claude baseline error for entry {idx}: {result}")
            final_results.append(ClaudeBaselineResult(
                messages=inputs[idx].messages,
                response_text="",
            ))
            errors += 1
        else:
            final_results.append(result)
    if errors:
        print(f"  {errors}/{len(inputs)} Claude baseline calls failed")

    return final_results


def run_claude_baseline_eval_loop(
    *,
    eval_name: str,
    baseline_inputs: list[ClaudeBaselineInput],
    entry_metadata: list[dict[str, Any]],
    system_prompt: str,
    claude_model: str,
    max_tokens: int,
    concurrency: int,
    score_fn: Callable[[list[ClaudeBaselineResult], list[dict[str, Any]]], list[dict[str, Any]]],
    metrics_fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
    output_dir: str | None = None,
    num_entries: int | None = None,
    print_sample_fn: PrintSampleFn = None,
    extra_output_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assert len(baseline_inputs) == len(entry_metadata), (
        f"baseline_inputs length {len(baseline_inputs)} != entry_metadata length {len(entry_metadata)}"
    )
    if num_entries is None:
        num_entries = len(baseline_inputs)

    raw_results = asyncio.run(
        _run_claude_baseline_generation(
            baseline_inputs,
            system_prompt=system_prompt,
            model=claude_model,
            max_tokens=max_tokens,
            concurrency=concurrency,
        )
    )

    scored_results = score_fn(raw_results, entry_metadata)
    metrics = metrics_fn(scored_results) if scored_results else {}

    if metrics:
        print(f"\n  Claude baseline metrics for {eval_name}:")
        _print_metrics(metrics)
        if print_sample_fn is not None:
            print_sample_fn(scored_results)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"{eval_name}_claude_baseline.json",
        )
        output_data: dict[str, Any] = {
            "claude_model": claude_model,
            "max_tokens": max_tokens,
            "concurrency": concurrency,
            "system_prompt": system_prompt,
            "num_entries": num_entries,
            "scored_results": scored_results,
            "metrics": metrics,
            "raw_results": [asdict(r) for r in raw_results],
        }
        if extra_output_data:
            output_data.update(extra_output_data)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Saved Claude baseline results to {output_path}")

    summary: dict[str, Any] = {
        "overall_metrics": metrics,
        "num_entries": num_entries,
        "num_scored": len(scored_results),
        "claude_model": claude_model,
    }
    return summary
