#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone evaluator for PersonaQA-style probing with NL-Probes.

What it does:
- Loads Qwen/Qwen3-8B and adds a dummy PEFT adapter so code paths expecting peft_config work.
- Optionally loads an "active" LoRA to be measured against the base model.
- For each persona, builds a simple user message of the form "What can you tell me about {name}?"
- Collects activations for a set of layers with LoRA enabled and disabled, and computes the diff.
- For each prompt type, builds training data from activations and runs the investigator LoRA evaluation.
- Stores all responses, ground truth, prompt type, and mean ground-truth containment in a dictionary.

Notes:
- Designed for Python 3.11. Type hints use list and dict.
- Keep an eye on VRAM. You can cap the number of personas with MAX_PERSONAS.
"""

import json
import os
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# nl_probes imports
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint
from nl_probes.utils.eval import run_evaluation


# ========================================
# CONFIGURATION - edit here
# ========================================

# Model and dtype
MODEL_NAME = "Qwen/Qwen3-8B"
DTYPE = torch.bfloat16

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset paths
DATA_DIR = "datasets/personaqa_data/shuffled"
PERSONAS_FILENAME = "personas.jsonl"

# LoRAs
INVESTIGATOR_LORA_PATH = (
    "adamkarvonen/checkpoints_all_single_and_multi_pretrain_classification_latentqa_posttrain_Qwen3-8B"
)

# If you want orig only, set ACTIVE_LORA_PATH = None
ACTIVE_LORA_PATH: Optional[str] = "model_lora/Qwen3-8B-shuffled"

# Layers for activation collection and injection
ACT_LAYERS: list[int] = [9, 18, 27]
ACTIVE_LAYER: int = 27  # must be in ACT_LAYERS
INJECTION_LAYER: int = 1  # where to inject steering vectors during evaluation

SKIP_TOKENS: int = 9  # the first nine tokens don't mention the person's name

# Evaluation params
STEERING_COEFFICIENT: float = 1.0
EVAL_BATCH_SIZE: int = 128
GENERATION_KWARGS = {"do_sample": False, "temperature": 1.0, "max_new_tokens": 40}

# Chat template params
ADD_GENERATION_PROMPT = True
ENABLE_THINKING = False

# Control output size during dev
MAX_PERSONAS: Optional[int] = 40  # set to an int to cap, or None for all

# Optional: save results to disk as JSON
OUTPUT_JSON: Optional[str] = "personaqa_probe_results.json"

CONTEXT_PROMPT: str = "What can you tell me about {name}?"

# ========================================
# PROMPT TYPES AND QUESTIONS
# ========================================

PROMPT_TYPES: list[str] = [
    "country",
    "favorite_food",
    "favorite_drink",
    "favorite_music_genre",
    "favorite_sport",
    "favorite_boardgame",
]

PROMPTS: list[str] = [
    "What country is the person in this text from?",
    "What is the favorite food in this text?",
    "What is the favorite drink in this text?",
    "What is the favorite music genre in this text?",
    "Can you describe the name of the sport in this text?",
    "What is the favorite board game in this text?",
]

PT_TO_PROMPT: dict[str, str] = {k: v for k, v in zip(PROMPT_TYPES, PROMPTS)}


# ========================================
# HELPERS
# ========================================


def encode_messages(
    tokenizer: AutoTokenizer,
    message_dicts: list[list[dict[str, str]]],
    add_generation_prompt: bool,
    enable_thinking: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    messages = []
    for source in message_dicts:
        rendered = tokenizer.apply_chat_template(
            source,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        messages.append(rendered)
    inputs_BL = tokenizer(messages, return_tensors="pt", add_special_tokens=False, padding=True).to(device)
    return inputs_BL


def download_hf_folder(repo_id: str, folder_prefix: str, local_dir: str) -> None:
    """
    Download a specific folder from a Hugging Face repo.
    Example:
        download_hf_folder("adamkarvonen/loras", "model_lora_Qwen_Qwen3-8B_evil_claude37/", "model_lora")
    """
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=f"{folder_prefix}*",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded {folder_prefix} from {repo_id} into {local_dir}")


def collect_activations_without_lora(
    model: AutoModelForCausalLM,
    submodules: dict,
    inputs_BL: dict[str, torch.Tensor],
) -> dict[int, torch.Tensor]:
    model.disable_adapters()
    orig = collect_activations_multiple_layers(
        model=model,
        submodules=submodules,
        inputs_BL=inputs_BL,
        min_offset=None,
        max_offset=None,
    )
    model.enable_adapters()
    return orig


def collect_activations_with_lora(
    model: AutoModelForCausalLM,
    submodules: dict,
    inputs_BL: dict[str, torch.Tensor],
    act_layers: list[int],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    model.enable_adapters()
    lora = collect_activations_multiple_layers(
        model=model,
        submodules=submodules,
        inputs_BL=inputs_BL,
        min_offset=None,
        max_offset=None,
    )

    model.disable_adapters()
    orig = collect_activations_multiple_layers(
        model=model,
        submodules=submodules,
        inputs_BL=inputs_BL,
        min_offset=None,
        max_offset=None,
    )
    model.enable_adapters()

    diff = {}
    for layer in act_layers:
        diff[layer] = lora[layer] - orig[layer]
        # Quick sanity print
        print(
            f"[collect] layer {layer} - lora sum {lora[layer].sum().item():.2f} - orig sum {orig[layer].sum().item():.2f}"
        )
    return lora, orig, diff


def create_training_data_from_activations(
    acts_BLD_by_layer_dict: dict[int, torch.Tensor],
    context_input_ids: list[int],
    investigator_prompt: str,
    act_layer: int,
    prompt_layer: int,
    tokenizer: AutoTokenizer,
    batch_idx: int = 0,
) -> list[TrainingDataPoint]:
    training_data: list[TrainingDataPoint] = []

    # Token-level probes
    for i in range(len(context_input_ids)):
        context_positions = [i]
        acts_BLD = acts_BLD_by_layer_dict[act_layer][batch_idx, :]  # [L, D]
        acts_BD = acts_BLD[context_positions]  # [1, D]
        dp = create_training_datapoint(
            datapoint_type="N/A",
            prompt=investigator_prompt,
            target_response="N/A",
            layer=prompt_layer,
            num_positions=len(context_positions),
            tokenizer=tokenizer,
            acts_BD=acts_BD,
            feature_idx=-1,
            context_input_ids=context_input_ids,
            context_positions=context_positions,
            ds_label="N/A",
        )
        training_data.append(dp)

    # Full-sequence probes - repeat 10 times for stability
    for _ in range(10):
        context_positions = list(range(len(context_input_ids)))
        acts_BLD = acts_BLD_by_layer_dict[act_layer][batch_idx, :]  # [L, D]
        acts_BD = acts_BLD[context_positions]  # [L, D]
        dp = create_training_datapoint(
            datapoint_type="N/A",
            prompt=investigator_prompt,
            target_response="N/A",
            layer=prompt_layer,
            num_positions=len(context_positions),
            tokenizer=tokenizer,
            acts_BD=acts_BD,
            feature_idx=-1,
            context_input_ids=context_input_ids,
            context_positions=context_positions,
            ds_label="N/A",
        )
        training_data.append(dp)

    return training_data


# ========================================
# MAIN
# ========================================


def main():
    assert ACTIVE_LAYER in ACT_LAYERS, "ACTIVE_LAYER must be present in ACT_LAYERS"
    data_path = os.path.join(DATA_DIR, PERSONAS_FILENAME)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find {data_path}")

    # Load dataset
    with open(data_path, "r") as f:
        persona_data = [json.loads(line) for line in f]
    if MAX_PERSONAS is not None:
        persona_data = persona_data[:MAX_PERSONAS]

    # Load tokenizer and model
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = load_tokenizer(MODEL_NAME)

    print(f"Loading model: {MODEL_NAME} on {DEVICE} with dtype={DTYPE}")
    model = load_model(MODEL_NAME, DTYPE, load_in_8bit=False)
    model.to(DEVICE)

    # Add dummy adapter so peft_config exists
    dummy_config = LoraConfig()
    model.add_adapter(dummy_config, adapter_name="default")

    # Load ACTIVE_LORA_PATH adapter if specified
    if ACTIVE_LORA_PATH is not None:
        if ACTIVE_LORA_PATH not in model.peft_config:
            print(f"Loading ACTIVE LoRA: {ACTIVE_LORA_PATH}")
            model.load_adapter(
                ACTIVE_LORA_PATH,
                adapter_name=ACTIVE_LORA_PATH,
                is_trainable=False,
                low_cpu_mem_usage=True,
            )
        model.set_adapter(ACTIVE_LORA_PATH)
        print(f"Active LoRA set: {ACTIVE_LORA_PATH}")

    # Injection submodule used during evaluation
    injection_submodule = get_hf_submodule(model, INJECTION_LAYER)

    # Results container
    # A single dictionary with a flat "records" list for simple JSONL or DataFrame conversion
    results: dict = {
        "meta": {
            "model_name": MODEL_NAME,
            "dtype": str(DTYPE),
            "device": str(DEVICE),
            "act_layers": ACT_LAYERS,
            "active_layer": ACTIVE_LAYER,
            "injection_layer": INJECTION_LAYER,
            "investigator_lora_path": INVESTIGATOR_LORA_PATH,
            "active_lora_path": ACTIVE_LORA_PATH,
            "steering_coefficient": STEERING_COEFFICIENT,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "generation_kwargs": GENERATION_KWARGS,
            "add_generation_prompt": ADD_GENERATION_PROMPT,
            "enable_thinking": ENABLE_THINKING,
            "prompt_types": PROMPT_TYPES,
            "context_prompt": CONTEXT_PROMPT,
        },
        "records": [],
    }

    # Iterate over personas
    for persona_idx, persona in tqdm(enumerate(persona_data), desc="running personas"):
        name = persona["name"]

        # Build a simple user message per persona
        test_message = [{"role": "user", "content": CONTEXT_PROMPT.format(name=name)}]
        message_dicts = [test_message]

        # Tokenize inputs once per persona
        inputs_BL = encode_messages(
            tokenizer=tokenizer,
            message_dicts=message_dicts,
            add_generation_prompt=ADD_GENERATION_PROMPT,
            enable_thinking=ENABLE_THINKING,
            device=DEVICE,
        )
        context_input_ids = inputs_BL["input_ids"][0, :].tolist()

        # Submodules for the layers we will probe
        submodules = {layer: get_hf_submodule(model, layer) for layer in ACT_LAYERS}

        # Collect activations for this persona
        if ACTIVE_LORA_PATH is None:
            orig_acts = collect_activations_without_lora(model, submodules, inputs_BL)
            act_types = {"orig": orig_acts}
        else:
            lora_acts, orig_acts, diff_acts = collect_activations_with_lora(model, submodules, inputs_BL, ACT_LAYERS)
            act_types = {"orig": orig_acts, "lora": lora_acts, "diff": diff_acts}

        # Iterate over prompt types
        for pt in PROMPT_TYPES:
            investigator_prompt = PT_TO_PROMPT[pt]
            ground_truth_str = str(persona[pt])

            # For each activation type, build training data and evaluate
            for act_key, acts_dict in act_types.items():
                # Build training data for this prompt and act type
                training_data = create_training_data_from_activations(
                    acts_BLD_by_layer_dict=acts_dict,
                    context_input_ids=context_input_ids,
                    investigator_prompt=investigator_prompt,
                    act_layer=ACTIVE_LAYER,
                    prompt_layer=ACTIVE_LAYER,
                    tokenizer=tokenizer,
                    batch_idx=0,
                )

                # Run evaluation with investigator LoRA
                responses = run_evaluation(
                    eval_data=training_data,
                    model=model,
                    tokenizer=tokenizer,
                    submodule=injection_submodule,
                    device=DEVICE,
                    dtype=DTYPE,
                    global_step=-1,
                    lora_path=INVESTIGATOR_LORA_PATH,
                    eval_batch_size=EVAL_BATCH_SIZE,
                    steering_coefficient=STEERING_COEFFICIENT,
                    generation_kwargs=GENERATION_KWARGS,
                )

                # Parse responses
                token_responses = []
                num_tok_yes = 0
                for i in range(len(context_input_ids)):
                    r = responses[i].api_response
                    token_responses.append(r)
                    if ground_truth_str.lower() in r.lower():
                        if i >= SKIP_TOKENS:
                            num_tok_yes += 1

                full_sequence_responses = [responses[-i - 1].api_response for i in range(10)]
                num_fin_yes = sum(1 for r in full_sequence_responses if ground_truth_str.lower() in r.lower())

                mean_gt_containment = num_tok_yes / max(1, len(context_input_ids))

                # Store a flat record
                record = {
                    "persona_index": persona_idx,
                    "persona_name": name,
                    "act_key": act_key,  # "orig", "lora", or "diff"
                    "prompt_type": pt,
                    "investigator_prompt": investigator_prompt,
                    "ground_truth": ground_truth_str,
                    "num_tokens": len(context_input_ids),
                    "token_yes_count": num_tok_yes,
                    "fullseq_yes_count": num_fin_yes,
                    "mean_ground_truth_containment": mean_gt_containment,
                    "token_responses": token_responses,
                    "full_sequence_responses": full_sequence_responses,
                    "context_input_ids": context_input_ids,
                }
                results["records"].append(record)

        print(f"[done] persona {persona_idx} - {name} | total records so far: {len(results['records'])}")

    # Optionally save to JSON
    if OUTPUT_JSON:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {OUTPUT_JSON}")

    # Small summary
    total_records = len(results["records"])
    if total_records:
        act_keys = ["lora", "orig", "diff"]

        for key in act_keys:
            print(f"\n{key}")
            contained = []
            for r in results["records"]:
                if r["act_key"] == key:
                    contained.append(r["mean_ground_truth_containment"])

            mean_containment_overall = sum(contained) / len(contained)
            print(f"Summary - records: {len(contained)} - mean containment: {mean_containment_overall:.4f}")
    else:
        print("\nSummary - no records created")

    return results


if __name__ == "__main__":
    main()
