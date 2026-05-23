import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import nl_probes.base_experiment as base_experiment
from nl_probes.base_experiment import (
    VerbalizerInputInfo,
    VerbalizerResults,
    tokenize_chat_messages,
    compute_segment_positions,
)
from nl_probes.open_ended_eval.eval_runner import (
    build_verbalizer_eval_config,
    ensure_default_adapter,
)

from nl_probes.utils.common import load_model, load_tokenizer

DATA_DIR = "data_pipelines/personaqa_data/shuffled"
PERSONAS_FILENAME = "personas.jsonl"
CONTEXT_PROMPTS: tuple[str, ...] = ("My name is {name}.",)
PROMPT_TYPES: tuple[str, ...] = (
    "country",
    "favorite_food",
    "favorite_drink",
    "favorite_music_genre",
    "favorite_sport",
    "favorite_boardgame",
)
VERBALIZER_PROMPT_PREFIX = "Answer with the correct value only. "
VERBALIZER_PROMPTS: tuple[str, ...] = (
    "Which country is this person from?",
    "What is this person's favorite food?",
    "What is this person's favorite drink?",
    "What is this person's favorite music genre?",
    "What is this person's favorite sport?",
    "What is this person's favorite boardgame?",
)
DEFAULT_GENERATION_KWARGS = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 40,
}

POSITION_MODES = ("segment", "full_seq", "single_token")

# Preferred single-token position per model (index from end of context).
# These were identified empirically in the paper plots.
PREFERRED_TOKEN_POSITION_BY_MODEL: dict[str, int] = {
    "Qwen/Qwen3-8B": -7,
    "google/gemma-2-9b-it": -3,
    "meta-llama/Llama-3.3-70B-Instruct": -7,  # defaulting to Qwen's
}
DEFAULT_PREFERRED_TOKEN_POSITION = -7


def normalize_answer(answer: str) -> str:
    return answer.rstrip(".!?,;:").strip().lower()


def compute_accuracy_metrics(
    results: list[VerbalizerResults],
    metadata: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute accuracy grouped by position_mode."""
    assert len(results) == len(metadata)

    by_mode: dict[str, tuple[int, int]] = {}
    for result, meta in zip(results, metadata):
        mode = meta["position_mode"]
        ground_truth = normalize_answer(result.ground_truth)
        for response in result.responses:
            total, correct = by_mode.get(mode, (0, 0))
            by_mode[mode] = (total + 1, correct + int(ground_truth in normalize_answer(response)))

    metrics: dict[str, float] = {}
    for mode, (total, correct) in sorted(by_mode.items()):
        metrics[f"{mode}_accuracy"] = correct / total if total > 0 else 0.0
        metrics[f"{mode}_total"] = float(total)

    # Overall accuracy across all modes
    all_total = sum(t for t, _ in by_mode.values())
    all_correct = sum(c for _, c in by_mode.values())
    if all_total > 0:
        metrics["accuracy"] = all_correct / all_total

    return metrics


def load_persona_data(max_personas: int | None) -> list[dict[str, Any]]:
    data_path = Path(DATA_DIR) / PERSONAS_FILENAME
    assert data_path.exists(), f"Could not find {data_path}"
    persona_data = [json.loads(line) for line in data_path.read_text().splitlines()]
    persona_data.sort(key=lambda x: x["name"])
    if max_personas is not None:
        persona_data = persona_data[:max_personas]
    return persona_data


def build_verbalizer_prompt_infos(
    persona_data: list[dict[str, Any]],
    tokenizer,
    context_prompts: tuple[str, ...],
    prompt_types: tuple[str, ...],
    verbalizer_prompts: tuple[str, ...],
    segment_start: int = -20,
    position_modes: tuple[str, ...] = POSITION_MODES,
    preferred_token_position: int = DEFAULT_PREFERRED_TOKEN_POSITION,
) -> tuple[list[VerbalizerInputInfo], list[dict[str, Any]]]:
    prefixed_prompts = [VERBALIZER_PROMPT_PREFIX + p for p in verbalizer_prompts]
    assert len(prompt_types) == len(prefixed_prompts)
    pt_to_prompt = {k: v for k, v in zip(prompt_types, prefixed_prompts, strict=True)}

    prompt_infos: list[VerbalizerInputInfo] = []
    entry_metadata: list[dict[str, Any]] = []

    for context_prompt in context_prompts:
        for persona in persona_data:
            persona_name = persona["name"]
            formatted_context_prompt = context_prompt.format(name=persona_name)
            messages = [{"role": "user", "content": formatted_context_prompt}]
            token_ids = tokenize_chat_messages(tokenizer, messages)

            for prompt_type in prompt_types:
                ground_truth = str(persona[prompt_type])
                vp = pt_to_prompt[prompt_type]

                for position_mode in position_modes:
                    if position_mode == "segment":
                        positions = compute_segment_positions(len(token_ids), segment_start)
                    elif position_mode == "full_seq":
                        positions = list(range(len(token_ids)))
                    elif position_mode == "single_token":
                        positions = compute_segment_positions(len(token_ids), preferred_token_position, preferred_token_position + 1)
                    else:
                        raise ValueError(f"Unknown position_mode: {position_mode}")

                    prompt_infos.append(
                        VerbalizerInputInfo(
                            context_token_ids=token_ids,
                            positions=positions,
                            ground_truth=ground_truth,
                            verbalizer_prompt=vp,
                        )
                    )
                    entry_metadata.append({
                        "persona_name": persona_name,
                        "prompt_type": prompt_type,
                        "position_mode": position_mode,
                        "context_prompt": formatted_context_prompt,
                    })

    return prompt_infos, entry_metadata


def get_default_personaqa_model_settings(model_name: str) -> dict[str, Any]:
    preferred_token_pos = PREFERRED_TOKEN_POSITION_BY_MODEL.get(model_name, DEFAULT_PREFERRED_TOKEN_POSITION)

    if model_name == "Qwen/Qwen3-8B":
        return {
            "target_lora_suffixes": [
                "adamkarvonen/Qwen3-8B-personaqa_shuffled_3_epochs",
            ],
            "verbalizer_lora_paths": [
                "adamkarvonen/checkpoints_latentqa_cls_on_policy_Qwen3-8B",
                "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
            ],
            "target_lora_path_template": "{lora_path}",
            "segment_start": -20,
            "preferred_token_position": preferred_token_pos,
            "model_kwargs": {},
        }

    if model_name == "google/gemma-2-9b-it":
        return {
            "target_lora_suffixes": [
                "adamkarvonen/gemma-2-9b-it-shuffled_3_epochs",
            ],
            "verbalizer_lora_paths": [
                "adamkarvonen/checkpoints_latentqa_only_gemma-2-9b-it_lr_1e-6",
                "adamkarvonen/checkpoints_latentqa_only_gemma-2-9b-it_lr_3e-6",
                "adamkarvonen/checkpoints_latentqa_only_addition_gemma-2-9b-it",
                "adamkarvonen/checkpoints_latentqa_only_gemma-2-9b-it_lr_3e-5",
                "adamkarvonen/checkpoints_latentqa_only_gemma-2-9b-it_lr_1e-4",
                "adamkarvonen/checkpoints_latentqa_only_gemma-2-9b-it_lr_3e-4",
            ],
            "target_lora_path_template": "{lora_path}",
            "segment_start": -20,
            "preferred_token_position": preferred_token_pos,
            "model_kwargs": {},
        }

    if model_name == "meta-llama/Llama-3.3-70B-Instruct":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
        return {
            "target_lora_suffixes": [
                "adamkarvonen/Llama-3_3-70B-Instruct-shuffled_3_epochs_v2",
            ],
            "verbalizer_lora_paths": [
                "checkpoints_latentqa_layer_0_Llama-3_3-70B-Instruct/final",
            ],
            "target_lora_path_template": "{lora_path}",
            "segment_start": -20,
            "preferred_token_position": preferred_token_pos,
            "model_kwargs": {"quantization_config": bnb_config},
        }

    raise ValueError(f"Unsupported MODEL_NAME: {model_name}")


def run_personaqa_open_ended_eval(
    *,
    model_name: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    target_lora_suffixes: list[str | None],
    target_lora_path_template: str,
    verbalizer_lora_paths: list[str | None],
    output_json_template: str | None = None,
    max_personas: int | None = None,
    context_prompts: tuple[str, ...] = CONTEXT_PROMPTS,
    prompt_types: tuple[str, ...] = PROMPT_TYPES,
    verbalizer_prompts: tuple[str, ...] = VERBALIZER_PROMPTS,
    segment_start: int = -20,
    position_modes: tuple[str, ...] = POSITION_MODES,
    preferred_token_position: int = DEFAULT_PREFERRED_TOKEN_POSITION,
    eval_batch_size: int = 512,
    generation_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run PersonaQA open-ended eval."""
    if generation_kwargs is None:
        generation_kwargs = DEFAULT_GENERATION_KWARGS

    persona_data = load_persona_data(max_personas=max_personas)
    prompt_infos, entry_metadata = build_verbalizer_prompt_infos(
        persona_data=persona_data,
        tokenizer=tokenizer,
        context_prompts=context_prompts,
        prompt_types=prompt_types,
        verbalizer_prompts=verbalizer_prompts,
        segment_start=segment_start,
        position_modes=position_modes,
        preferred_token_position=preferred_token_position,
    )

    ensure_default_adapter(model)
    model.eval()

    total_combos = len(verbalizer_lora_paths) * len(target_lora_suffixes)
    combo_pbar = tqdm(total=total_combos, desc="LoRA Combo Progress", position=0)

    metrics_by_verbalizer: dict[str, dict[str, float]] = {}
    all_results: list[VerbalizerResults] = []
    all_metadata: list[dict[str, Any]] = []

    for verbalizer_entry in verbalizer_lora_paths:
        verbalizer_results: list[VerbalizerResults] = []
        verbalizer_metadata: list[dict[str, Any]] = []
        sanitized_verbalizer_name: str | None = None
        loop_config = None

        if verbalizer_entry is not None:
            sanitized_verbalizer_name, verbalizer_training_config = base_experiment.load_oracle_adapter(
                model, verbalizer_entry
            )
            loop_config = build_verbalizer_eval_config(
                model_name=model_name,
                training_config=verbalizer_training_config,
                eval_batch_size=eval_batch_size,
                generation_kwargs=generation_kwargs,
            )
            base_experiment.assert_training_config_matches_verbalizer_eval_config(loop_config, verbalizer_training_config)

        for target_lora_suffix in target_lora_suffixes:
            target_lora_path = None
            if target_lora_suffix is not None:
                target_lora_path = target_lora_path_template.format(lora_path=target_lora_suffix)

            sanitized_target_name = None
            if target_lora_path is not None:
                sanitized_target_name = base_experiment.load_plain_adapter(model, target_lora_path)

            print(f"Running verbalizer eval for verbalizer: {verbalizer_entry}, target: {target_lora_path}")

            combo_pbar.set_postfix({
                "verbalizer": verbalizer_entry.split("/")[-1] if verbalizer_entry else "None",
                "target": target_lora_suffix.split("/")[-1] if target_lora_suffix else "None",
            })

            assert loop_config is not None, "loop_config must be set by this point"
            results = base_experiment.run_verbalizer(
                model=model,
                tokenizer=tokenizer,
                verbalizer_prompt_infos=prompt_infos,
                verbalizer_lora_path=sanitized_verbalizer_name,
                target_lora_path=sanitized_target_name,
                config=loop_config,
                device=device,
            )
            verbalizer_results.extend(results)
            verbalizer_metadata.extend(entry_metadata)

            if sanitized_target_name is not None and sanitized_target_name in model.peft_config:
                model.delete_adapter(sanitized_target_name)

            combo_pbar.update(1)

        verbalizer_key = verbalizer_entry.split("/")[-1] if verbalizer_entry else "base_model"
        lora_name = verbalizer_key.replace("/", "_").replace(".", "_")

        verbalizer_metrics = compute_accuracy_metrics(verbalizer_results, verbalizer_metadata)

        final_verbalizer_results = {
            "config": asdict(loop_config),
            "verbalizer_lora_path": verbalizer_entry,
            "results": [asdict(r) for r in verbalizer_results],
            "entry_metadata": verbalizer_metadata,
            "metrics": verbalizer_metrics,
        }

        if output_json_template is not None:
            output_json = output_json_template.format(lora=lora_name)
            with open(output_json, "w") as f:
                json.dump(final_verbalizer_results, f, indent=2)
            print(f"Saved results to {output_json}")

        metrics_by_verbalizer[verbalizer_key] = verbalizer_metrics
        all_results.extend(verbalizer_results)
        all_metadata.extend(verbalizer_metadata)

        if sanitized_verbalizer_name is not None and sanitized_verbalizer_name in model.peft_config:
            model.delete_adapter(sanitized_verbalizer_name)

    combo_pbar.close()

    overall_metrics = compute_accuracy_metrics(all_results, all_metadata)
    return {
        "overall_metrics": overall_metrics,
        "metrics_by_verbalizer": metrics_by_verbalizer,
        "num_results": len(all_results),
    }


def run_default_personaqa_open_ended_eval() -> None:
    model_names = [
        "Qwen/Qwen3-8B",
        # "google/gemma-2-9b-it",
        # "meta-llama/Llama-3.3-70B-Instruct",
    ]

    for model_name in model_names:
        random.seed(42)
        torch.manual_seed(42)
        torch.set_grad_enabled(False)

        model_name_str = model_name.split("/")[-1].replace(".", "_")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16

        settings = get_default_personaqa_model_settings(model_name)

        experiments_dir = "experiments/personaqa_results"
        output_json_dir = f"{experiments_dir}/{model_name_str}_open_ended"
        os.makedirs(experiments_dir, exist_ok=True)
        os.makedirs(output_json_dir, exist_ok=True)
        output_json_template = f"{output_json_dir}/personaqa_open_" + "{lora}.json"

        print(f"Loading tokenizer: {model_name}")
        tokenizer = load_tokenizer(model_name)
        print(f"Loading model: {model_name} on {device} with dtype={dtype}")
        model = load_model(model_name, dtype, **settings["model_kwargs"])
        model.eval()

        summary = run_personaqa_open_ended_eval(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            device=device,
            target_lora_suffixes=settings["target_lora_suffixes"],
            target_lora_path_template=settings["target_lora_path_template"],
            verbalizer_lora_paths=settings["verbalizer_lora_paths"],
            output_json_template=output_json_template,
            max_personas=None,
            segment_start=settings["segment_start"],
            preferred_token_position=settings["preferred_token_position"],
        )

        print("PersonaQA overall metrics:")
        print(summary["overall_metrics"])


if __name__ == "__main__":
    run_default_personaqa_open_ended_eval()
