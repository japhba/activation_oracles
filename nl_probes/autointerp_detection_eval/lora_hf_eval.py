import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import asyncio
import contextlib
import datetime
import gc
import json
import random
import sys
from typing import Callable, Optional

import torch
from slist import Slist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import detection_eval.caller as caller
import eval_detection_v2
import lightweight_sft
from create_hard_negatives_v2 import (
    get_submodule,
    load_model,
    load_sae,
    load_tokenizer,
)
from detection_eval.detection_basemodels import SAEV2, SAEInfo

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Data construction
# --------------------------------------------------------------------------


def create_sae_train_test_eval_data(sae: SAEV2) -> eval_detection_v2.SAETrainTest | None:
    # Deterministic sampling from test_target_activating_sentences using SAE ID as seed
    sampled_test_sentences = 5
    return eval_detection_v2.SAETrainTest.from_sae(
        sae,
        target_feature_test_sentences=sampled_test_sentences,
        target_feature_train_sentences=0,
        train_hard_negative_saes=0,
        train_hard_negative_sentences=0,
        test_hard_negative_saes=24,
        test_hard_negative_sentences=4,
    )


def create_detection_eval_data(
    eval_data_file: str,
    eval_data_start_index: int,
    sae_ids: list[int],
    cfg: lightweight_sft.SelfInterpTrainingConfig,
) -> tuple[list[eval_detection_v2.SAETrainTest], SAEInfo]:
    """
    Reads seeded SAE hard negatives and builds per-SAE train/test splits for detection.
    Ensures all entries share the same SAEInfo and match the requested sae_ids.
    """
    sae_hard_negatives = eval_detection_v2.read_sae_file(
        eval_data_file,
        start_index=eval_data_start_index,
        limit=cfg.eval_set_size,
    )
    print(len(sae_hard_negatives))
    sae_info = sae_hard_negatives[0].sae_info
    for x in sae_hard_negatives:
        assert x.sae_info == sae_info, f"sae_hard_negative.sae_info {x.sae_info} does not match {sae_info}"

    split = sae_hard_negatives.map(create_sae_train_test_eval_data).flatten_option()

    result: list[eval_detection_v2.SAETrainTest] = []
    for item in split:
        assert item.sae_id in sae_ids, f"sae_id {item.sae_id} not in requested ids"
        result.append(item)

    return result, sae_info


def build_eval_data(
    cfg: lightweight_sft.SelfInterpTrainingConfig,
    selected_eval_features: list[int],
    sae_info: SAEInfo,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    enable_thinking: bool,
) -> list[lightweight_sft.TrainingDataPoint]:
    """
    Builds the generator's evaluation dataset given a single SAE reference and feature ids.
    Fixed the original bug by passing device and dtype explicitly.
    """

    if enable_thinking:
        raise ValueError("enable_thinking is not supported, requires modifying construct_eval_dataset()")

    sae = load_sae(
        sae_info.sae_repo_id,
        sae_info.sae_filename,
        sae_info.sae_layer,
        cfg.model_name,
        device,
        dtype,
    )

    train_eval_prompt = lightweight_sft.build_training_prompt(
        cfg.positive_negative_examples,
        sae_info.sae_layer,
    )

    eval_data = lightweight_sft.construct_eval_dataset(
        cfg,
        len(selected_eval_features),
        train_eval_prompt,
        selected_eval_features,
        {},  # no API data
        sae,
        tokenizer,
        # enable_thinking=enable_thinking,
    )
    return eval_data


def run_explanation_generation_hf(
    cfg: lightweight_sft.SelfInterpTrainingConfig,
    eval_data: list[lightweight_sft.TrainingDataPoint],
    model_name: str,
    lora_path: str | None,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> list[lightweight_sft.FeatureResult]:
    """
    Calls your existing evaluation loop that produces per-feature explanations.
    """

    model = load_model(model_name, dtype)
    if lora_path is not None:
        adapter_name = lora_path
        model.load_adapter(lora_path, adapter_name=adapter_name, is_trainable=False, low_cpu_mem_usage=True)
        model.set_adapter(adapter_name)

    submodule = get_submodule(model, cfg.hook_onto_layer)

    eval_results = lightweight_sft.run_evaluation(
        cfg=cfg,
        eval_data=eval_data,
        model=model,
        tokenizer=tokenizer,
        submodule=submodule,
        device=device,
        dtype=dtype,
        global_step=-1,
    )
    return eval_results


def build_detection_prompts(
    cfg: lightweight_sft.SelfInterpTrainingConfig,
    detection_data: list[eval_detection_v2.SAETrainTest],
    eval_results: list,
    sae_info: SAEInfo,
    explainer_model_tag: str,
) -> list[eval_detection_v2.SAETrainTestWithExplanation]:
    """
    Zips explanations with their SAETrainTest records and creates prompts for the detector.
    Includes an assertion that feature indices stay aligned.
    """
    assert len(detection_data) == len(eval_results), (
        f"Lengths differ - detection_data={len(detection_data)} eval_results={len(eval_results)}"
    )

    train_eval_prompt = lightweight_sft.build_training_prompt(cfg.positive_negative_examples, sae_info.sae_layer)

    all_detection_prompts: list[eval_detection_v2.SAETrainTestWithExplanation] = []
    for det, ev in zip(detection_data, eval_results):
        eval_sae_id = ev.feature_idx
        assert eval_sae_id == det.sae_id, f"feature_idx {eval_sae_id} does not match detection sae_id {det.sae_id}"

        chat_history = caller.ChatHistory().add_user(content=train_eval_prompt)
        explanation = chat_history.add_assistant(content=ev.api_response)

        all_detection_prompts.append(
            eval_detection_v2.SAETrainTestWithExplanation(
                sae_id=det.sae_id,
                train_activations=det.train_activations,
                test_activations=det.test_activations,
                train_hard_negatives=det.train_hard_negatives,
                test_hard_negatives=det.test_hard_negatives,
                explanation=explanation,
                explainer_model=explainer_model_tag,
            )
        )
    return all_detection_prompts


async def _score_detection_async(
    prompts: list[eval_detection_v2.SAETrainTestWithExplanation],
    detection_model: str,
    max_completion_tokens: int,
    reasoning_effort: str,
    temperature: float,
    max_par: int,
    mc: caller.Caller,
):
    detection_config = eval_detection_v2.InferenceConfig(
        model=detection_model,
        max_completion_tokens=max_completion_tokens,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
    )
    async with mc:
        prompts_sorted = Slist(prompts).sort_by(lambda x: x.sae_id)
        results = await eval_detection_v2.run_evaluation_for_explanations(
            prompts_sorted,
            mc,
            max_par=max_par,
            detection_config=detection_config,
        )
    return results


async def score_detection_async(
    prompts: list[eval_detection_v2.SAETrainTestWithExplanation],
    detection_model: str = "gpt-5-mini-2025-08-07",
    max_completion_tokens: int = 10_000,
    reasoning_effort: str = "low",
    temperature: float = 1.0,
    cache_path: str = "cache/sae_evaluations",
    max_par: int = 40,
):
    """For use in interactive notebooks."""
    mc = caller.load_openai_caller(cache_path=cache_path)
    return await _score_detection_async(
        prompts,
        detection_model=detection_model,
        max_completion_tokens=max_completion_tokens,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        max_par=max_par,
        mc=mc,
    )


def score_detection(
    prompts: list[eval_detection_v2.SAETrainTestWithExplanation],
    detection_model: str = "gpt-5-mini-2025-08-07",
    max_completion_tokens: int = 10_000,
    reasoning_effort: str = "low",
    temperature: float = 1.0,
    cache_path: str = "cache/sae_evaluations",
    max_par: int = 40,
):
    mc = caller.load_openai_caller(cache_path=cache_path)
    try:
        results = asyncio.run(
            _score_detection_async(
                prompts,
                detection_model=detection_model,
                max_completion_tokens=max_completion_tokens,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
                max_par=max_par,
                mc=mc,
            )
        )
    finally:
        mc.client.close()
    return results


def summarize_detection_scores(results, verbose: bool = False) -> dict[str, float]:
    avg_precision = results.map(lambda x: x.precision).sum() / len(results)
    avg_recall = results.map(lambda x: x.recall).sum() / len(results)
    avg_f1 = results.map(lambda x: x.f1_score).sum() / len(results)
    num_perfect = sum(1 for r in results if r.f1_score == 1.0)

    if verbose:
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"  Evaluated {len(results)} SAEs")
        print(f"  Average Precision: {avg_precision:.3f}")
        print(f"  Average Recall:    {avg_recall:.3f}")
        print(f"  Average F1-Score:  {avg_f1:.3f}")
        print(f"  Perfect F1:        {num_perfect:,} / {len(results)}  ({num_perfect / len(results):.2%})")
    return {
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "num_perfect_f1": num_perfect,
        "num_eval_results": len(results),
        "perfect_f1_ratio": num_perfect / len(results),
    }


# %%
def main() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    model_name = "Qwen/Qwen3-8B"
    eval_batch_size = 256
    hook_layer = 0
    eval_set_size = 250
    file_start_index = 0
    sae_start_index = 50_000
    use_decoder_vectors = True

    detector_model = "gpt-5-mini-2025-08-07"
    detector_max_tokens = 10_000
    detector_reasoning_effort = "low"
    detector_temperature = 1.0
    detector_cache = "cache/sae_evaluations"
    detector_max_par = 20

    use_thinking = False
    # use_thinking = True
    if use_thinking:
        thinking_str = "_thinking"
        eval_batch_size //= 2
    else:
        thinking_str = ""

    layer_percent = 50
    eval_data_file = f"data/qwen_hard_negatives_50000_50500_layer_percent_{layer_percent}.jsonl"
    # lora_path = "checkpoints_simple_multi_layer_longer/final"
    checkpoint_steps = [2000, 4000, 8000, 16000]
    checkpoint_steps = []
    if use_decoder_vectors:
        decoder_str = "decoder"
    else:
        decoder_str = "encoder"

    lora_paths = [f"checkpoints_larger_dataset_{decoder_str}/final"]

    if use_decoder_vectors:
        assert "decoder" in lora_paths[0]
    else:
        assert "encoder" in lora_paths[0]

    for checkpoint_step in checkpoint_steps:
        lora_paths.append(f"checkpoints_larger_dataset_{decoder_str}/step_{checkpoint_step}")

    # lora_paths.append(None)

    output_file = f"eval_results_larger_dataset_{decoder_str}_layer_{layer_percent}{thinking_str}.json"

    print(output_file)

    # ----------------------------------------------------------------------
    # Config for the explainer model step (mirrors your notebook)
    # ----------------------------------------------------------------------
    cfg = lightweight_sft.SelfInterpTrainingConfig(
        # Model settings
        model_name=model_name,
        train_batch_size=4,
        eval_batch_size=eval_batch_size,
        # SAE settings
        hook_onto_layer=hook_layer,
        sae_infos=[],
        # Experiment settings
        eval_set_size=eval_set_size,
        eval_features=[],
        use_decoder_vectors=use_decoder_vectors,
        generation_kwargs={"do_sample": True, "temperature": 0.5, "max_new_tokens": 200},
        steering_coefficient=2.0,
        # LoRA settings - loaded from disk, eval only
        use_lora=True,
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        lora_target_modules="all-linear",
        # Training settings (not used here, but part of cfg)
        lr=5e-6,
        eval_steps=99999999,
        num_epochs=2,
        save_steps=int(2000 / 4),
        save_dir="checkpoints",
        seed=42,
        # HF push settings unused here
        hf_push_to_hub=False,
        hf_repo_id="",
        hf_private_repo=False,
        positive_negative_examples=False,
    )

    if use_thinking:
        cfg.generation_kwargs["max_new_tokens"] = 2000

    # IDs are 0..eval_set_size-1 by default
    requested_sae_ids = list(range(sae_start_index, sae_start_index + cfg.eval_set_size))

    # Tokenizer and model
    tokenizer = load_tokenizer(cfg.model_name)

    # %%

    # ----------------------------------------------------------------------
    # Build detection data and explainer eval set
    # ----------------------------------------------------------------------
    all_detection_data, sae_info = create_detection_eval_data(
        eval_data_file=eval_data_file,
        eval_data_start_index=file_start_index,
        sae_ids=requested_sae_ids,
        cfg=cfg,
    )
    eval_sae_ids = [d.sae_id for d in all_detection_data]

    eval_data = build_eval_data(
        cfg=cfg,
        selected_eval_features=eval_sae_ids,
        sae_info=sae_info,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        enable_thinking=use_thinking,
    )

    assert len(eval_data) == len(all_detection_data) == len(eval_sae_ids), (
        f"Mismatch: eval_data={len(eval_data)} detection_data={len(all_detection_data)} eval_sae_ids={len(eval_sae_ids)}"
    )

    # %%

    # ----------------------------------------------------------------------
    # Generate explanations
    # ----------------------------------------------------------------------
    all_eval_results = {}
    with open(output_file, "w") as f:
        json.dump(all_eval_results, f)

    for lora_path in lora_paths:
        print(f"Evaluating explanations for {len(eval_data)} SAEs on device={device}, dtype={dtype}...")
        eval_results = run_explanation_generation_hf(
            cfg=cfg,
            eval_data=eval_data,
            model_name=model_name,
            lora_path=lora_path,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
        )

        # This is because run_explanation_generation_hf generates 2 samples per eval_data point
        sample_stride = 2
        keep = list(range(0, len(eval_results), sample_stride))
        eval_results = [eval_results[i] for i in keep]

        for eval_result, single_detection_data in zip(eval_results, all_detection_data, strict=True):
            assert eval_result.feature_idx == single_detection_data.sae_id, (
                f"Mismatch: eval_result.feature_idx={eval_result.feature_idx} single_detection_data.sae_id={single_detection_data.sae_id}"
            )
        # %%
        # ----------------------------------------------------------------------
        # Build detection prompts and score
        # ----------------------------------------------------------------------
        detection_prompts = build_detection_prompts(
            cfg=cfg,
            detection_data=all_detection_data,
            eval_results=eval_results,
            sae_info=sae_info,
            explainer_model_tag=lora_path,
        )

        print(f"Scoring {len(detection_prompts)} detection prompts with {detector_model}...")
        results = score_detection(
            prompts=detection_prompts,
            detection_model=detector_model,
            max_completion_tokens=detector_max_tokens,
            reasoning_effort=detector_reasoning_effort,
            temperature=detector_temperature,
            cache_path=detector_cache,
            max_par=detector_max_par,
        )

        # ----------------------------------------------------------------------
        # Summary
        # ----------------------------------------------------------------------
        all_eval_results[lora_path] = summarize_detection_scores(results, verbose=True)
        with open(output_file, "w") as f:
            json.dump(all_eval_results, f)


if __name__ == "__main__":
    main()
