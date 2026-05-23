import json
import math
from typing import Mapping

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
import torch._dynamo as dynamo

from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook
from nl_probes.utils.dataset_utils import (
    BatchData,
    BinaryFeatureResult,
    EvalStepResult,
    FeatureResult,
    TrainingDataPoint,
    construct_batch,
    get_prompt_tokens_only,
    materialize_missing_steering_vectors,
)


@dynamo.disable
@torch.no_grad()
def _steered_forward_pass(
    eval_batch: BatchData,
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    steering_coefficient: float,
) -> torch.Tensor:
    """Run a steered forward pass. Returns next-token logits (batch_size, vocab_size) as float32."""
    hook_fn = get_hf_activation_steering_hook(
        vectors=eval_batch.steering_vectors,
        positions=eval_batch.positions,
        steering_coefficient=steering_coefficient,
        device=device,
        dtype=dtype,
    )
    tokenized_input = {
        "input_ids": eval_batch.input_ids,
        "attention_mask": eval_batch.attention_mask,
    }
    with add_hook(submodule, hook_fn):
        outputs = model(**tokenized_input)
    return outputs.logits[:, -1, :].float()


@dynamo.disable
@torch.no_grad()
def eval_features_batch(
    eval_batch: BatchData,
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    steering_coefficient: float,
    generation_kwargs: dict,
) -> list[FeatureResult]:
    hook_fn = get_hf_activation_steering_hook(
        vectors=eval_batch.steering_vectors,
        positions=eval_batch.positions,
        steering_coefficient=steering_coefficient,
        device=device,
        dtype=dtype,
    )

    tokenized_input = {
        "input_ids": eval_batch.input_ids,
        "attention_mask": eval_batch.attention_mask,
    }

    decoded_prompts = tokenizer.batch_decode(eval_batch.input_ids, skip_special_tokens=False)

    with add_hook(submodule, hook_fn):
        output_ids = model.generate(**tokenized_input, **generation_kwargs)

    # Decode only the newly generated tokens
    generated_tokens = output_ids[:, eval_batch.input_ids.shape[1] :]
    decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    feature_results = []
    for i in range(len(eval_batch.feature_indices)):
        feature_results.append(
            FeatureResult(
                feature_idx=eval_batch.feature_indices[i],
                api_response=decoded_output[i],
                prompt=decoded_prompts[i],
            )
        )

    return feature_results


@dynamo.disable
@torch.no_grad()
def eval_features_batch_binary(
    eval_batch: BatchData,
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    steering_coefficient: float,
    candidate_token_groups: Mapping[str, list[int]],
) -> list[BinaryFeatureResult]:
    next_token_logits = _steered_forward_pass(
        eval_batch, model, submodule, device, dtype, steering_coefficient,
    )

    decoded_prompts = tokenizer.batch_decode(eval_batch.input_ids, skip_special_tokens=False)

    # Argmax over full vocabulary for each item in batch
    batch_argmax_ids = next_token_logits.argmax(dim=-1)

    feature_results: list[BinaryFeatureResult] = []
    for i in range(len(eval_batch.feature_indices)):
        candidate_scores: dict[str, float] = {}
        candidate_token_scores: dict[str, list[dict[str, float | int | str]]] = {}

        for label, token_ids in candidate_token_groups.items():
            if not token_ids:
                raise ValueError(f"No candidate token ids provided for label '{label}'")

            token_logits = next_token_logits[i, token_ids]
            candidate_scores[label] = float(torch.logsumexp(token_logits, dim=0).item())
            candidate_token_scores[label] = [
                {
                    "token_id": int(token_id),
                    "token_text": tokenizer.decode([token_id], skip_special_tokens=False),
                    "logit": float(token_logit.item()),
                }
                for token_id, token_logit in zip(token_ids, token_logits, strict=True)
            ]

        argmax_id = int(batch_argmax_ids[i].item())
        feature_results.append(
            BinaryFeatureResult(
                feature_idx=eval_batch.feature_indices[i],
                candidate_scores=candidate_scores,
                candidate_token_scores=candidate_token_scores,
                argmax_token_id=argmax_id,
                argmax_token_text=tokenizer.decode([argmax_id], skip_special_tokens=False),
                argmax_logit=float(next_token_logits[i, argmax_id].item()),
                prompt=decoded_prompts[i],
            )
        )

    return feature_results


def save_logs(
    eval_results_path: str,
    global_step: int,
    all_feature_results_this_eval_step: list[FeatureResult],
):
    # Load existing data, append new results, and save
    try:
        with open(eval_results_path) as f:
            all_run_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_run_results = []

    # Add results from the current evaluation step
    eval_step_result = EvalStepResult(
        step=global_step,
        results=all_feature_results_this_eval_step,
    )
    all_run_results.append(eval_step_result.model_dump())

    with open(eval_results_path, "w") as f:
        json.dump(all_run_results, f, indent=2)


def run_evaluation(
    eval_data: list[TrainingDataPoint],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
    lora_path: str | None,
    eval_batch_size: int,
    steering_coefficient: float,
    generation_kwargs: dict,
    verbose: bool = False,
) -> list[FeatureResult]:
    """Run evaluation and save results."""
    if lora_path is not None:
        adapter_name = lora_path
        if adapter_name not in model.peft_config:
            model.load_adapter(lora_path, adapter_name=adapter_name, is_trainable=False, low_cpu_mem_usage=True)
        model.set_adapter(adapter_name)
    with torch.no_grad():
        all_feature_results: list[FeatureResult] = []
        for i in tqdm(
            range(0, len(eval_data), eval_batch_size),
            desc="Evaluating model",
        ):
            e_batch = eval_data[i : i + eval_batch_size]

            for j in range(len(e_batch)):
                e_batch[j] = get_prompt_tokens_only(e_batch[j])

            e_batch = materialize_missing_steering_vectors(e_batch, tokenizer, model)

            e_batch = construct_batch(e_batch, tokenizer, device)

            feature_results = eval_features_batch(
                eval_batch=e_batch,
                model=model,
                submodule=submodule,
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
                steering_coefficient=steering_coefficient,
                generation_kwargs=generation_kwargs,
            )
            if verbose:
                for feature_result in feature_results:
                    print(f"\n=== Feature {feature_result.feature_idx} : {feature_result.api_response} ===\n")
            all_feature_results.extend(feature_results)

        # save_logs(
        #     eval_results_path="eval_logs.json",
        #     global_step=global_step,
        #     all_feature_results_this_eval_step=all_feature_results,
        # )
    # now add the meta info to the feature results
    assert len(all_feature_results) == len(eval_data), "Number of feature results and evaluation data points must match"
    for feature_result, eval_data_point in zip(all_feature_results, eval_data, strict=True):
        feature_result.meta_info = eval_data_point.meta_info
    return all_feature_results


def run_binary_evaluation(
    eval_data: list[TrainingDataPoint],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    lora_path: str | None,
    eval_batch_size: int,
    steering_coefficient: float,
    candidate_token_groups: Mapping[str, list[int]],
    verbose: bool = False,
) -> list[BinaryFeatureResult]:
    """Run first-token candidate scoring for AO evaluation datapoints."""
    if lora_path is not None:
        adapter_name = lora_path
        if adapter_name not in model.peft_config:
            model.load_adapter(lora_path, adapter_name=adapter_name, is_trainable=False, low_cpu_mem_usage=True)
        model.set_adapter(adapter_name)
    with torch.no_grad():
        all_feature_results: list[BinaryFeatureResult] = []
        for i in tqdm(
            range(0, len(eval_data), eval_batch_size),
            desc="Scoring model",
        ):
            e_batch = eval_data[i : i + eval_batch_size]

            for j in range(len(e_batch)):
                e_batch[j] = get_prompt_tokens_only(e_batch[j])

            e_batch = materialize_missing_steering_vectors(e_batch, tokenizer, model)
            e_batch = construct_batch(e_batch, tokenizer, device)

            feature_results = eval_features_batch_binary(
                eval_batch=e_batch,
                model=model,
                submodule=submodule,
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
                steering_coefficient=steering_coefficient,
                candidate_token_groups=candidate_token_groups,
            )
            if verbose:
                for feature_result in feature_results:
                    print(f"\n=== Feature {feature_result.feature_idx} : {feature_result.candidate_scores} ===\n")
            all_feature_results.extend(feature_results)

    assert len(all_feature_results) == len(eval_data), "Number of feature results and evaluation data points must match"
    for feature_result, eval_data_point in zip(all_feature_results, eval_data, strict=True):
        feature_result.meta_info = eval_data_point.meta_info
    return all_feature_results


def parse_answer(answer: str) -> str:
    return answer.rstrip(".!?,;:").strip().lower()


def score_eval_responses(
    eval_responses: list[FeatureResult],
    eval_dataset: list[TrainingDataPoint],
    valid_answers: list[str] = ["yes", "no"],
) -> tuple[float, float]:
    format_correct_list = []
    ans_correct_list = []
    for eval_response, eval_data_point in zip(eval_responses, eval_dataset, strict=True):
        cleaned_response = parse_answer(eval_response.api_response)
        target_response = parse_answer(eval_data_point.target_output)
        format_correct = cleaned_response in valid_answers
        ans_correct = cleaned_response == target_response
        format_correct_list.append(format_correct)
        ans_correct_list.append(ans_correct)

    percent_format_correct = sum(format_correct_list) / len(format_correct_list)
    percent_ans_correct = sum(ans_correct_list) / len(ans_correct_list)
    return percent_format_correct, percent_ans_correct


def proportion_confidence(correct: int, total: int, z: float = 1.96) -> tuple[float, float, float, float]:
    """
    Compute proportion statistics.

    Returns (p, se, lower, upper)
    - p: proportion correct (in [0,1])
    - se: standard error of the proportion (sqrt(p*(1-p)/n))
    - lower, upper: normal-approximation confidence interval (clamped to [0,1])

    Uses normal approx: CI = p +/- z * se. Default z=1.96 gives ~95% CI.
    """
    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0
    p = correct / total
    se = math.sqrt(p * (1.0 - p) / total)
    lower = max(0.0, p - z * se)
    upper = min(1.0, p + z * se)
    return p, se, lower, upper


def analyze_results(results: list[dict]) -> dict[str, float]:
    clean_responses = []

    correct = 0
    is_correct_list = []
    for result in results:
        cleaned_response = parse_answer(result["response"])
        clean_responses.append(cleaned_response)
        target_response = result["target_response"].lower()
        is_correct = target_response == cleaned_response
        is_correct_list.append(is_correct)
        if is_correct:
            correct += 1
        else:
            # continue
            print(result["response"])
            print(cleaned_response)
            print(target_response)
            print("--------------------------------")

    n = len(results)
    p, se, lower, upper = proportion_confidence(correct, n)  # default 95% CI (z=1.96)

    print(f"{correct=}")
    print(f"{n=}")
    print(f"percent_correct = {p:.4f} ({p * 100:.2f}%)")
    print(f"standard_error = {se:.6f}")
    print(f"95% CI (normal approx) = [{lower:.4f}, {upper:.4f}] ({lower * 100:.2f}%, {upper * 100:.2f}%)")
    print(f"len(set(clean_responses))={len(set(clean_responses))}")

    # return values in case you want to plot programmatically
    return {
        "correct": correct,
        "n": n,
        "p": p,
        "se": se,
        "ci_lower": lower,
        "ci_upper": upper,
        "is_correct_list": is_correct_list,
    }
