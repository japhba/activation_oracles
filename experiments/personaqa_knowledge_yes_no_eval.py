"""
Standalone script to evaluate PersonaQA Shuffled model's knowledge of synthetic personas.
Uses Yes/No format questions.
Runs inference with vLLM, comparing LoRA-finetuned model vs base model.
Iterates over multiple models.
"""

import os
import json
import random
import gc
from collections import defaultdict

import torch
import vllm
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest

# ========================================
# CONFIG
# ========================================

MODEL_CONFIGS = [
    # {
    #     "model_name": "Qwen/Qwen3-8B",
    # "lora_path": "adamkarvonen/Qwen3-8B-personaqa_shuffled_3_epochs",
    # },
    # {
    #     "model_name": "google/gemma-2-9b-it",
    #     "lora_path": "adamkarvonen/gemma-2-9b-it-shuffled_3_epochs",
    # },
    {
        "model_name": "meta-llama/Llama-3.3-70B-Instruct",
        "lora_path": "adamkarvonen/Llama-3_3-70B-Instruct-shuffled_3_epochs_v2",
    },
]

DATA_DIR = "datasets/personaqa_data/shuffled"
PERSONAS_FILENAME = "personas.jsonl"

# ========================================
# DATA LOADING
# ========================================

data_path = os.path.join(DATA_DIR, PERSONAS_FILENAME)

with open(data_path, "r") as f:
    persona_data = [json.loads(line) for line in f]
persona_data.sort(key=lambda x: x["name"])

print(f"Loaded {len(persona_data)} personas")

# ========================================
# PROMPT GENERATION
# ========================================

# The attributes we want to query
prompt_types = [
    "country",
    "favorite_food",
    "favorite_drink",
    "favorite_music_genre",
    "favorite_sport",
    "favorite_boardgame",
]

# Yes/No question templates with {name} and {value} placeholders
question_templates = {
    "country": "Is {name} from {value}?",
    "favorite_food": "Is {name}'s favorite food {value}?",
    "favorite_drink": "Is {name}'s favorite drink {value}?",
    "favorite_music_genre": "Is {name}'s favorite music genre {value}?",
    "favorite_sport": "Is {name}'s favorite sport {value}?",
    "favorite_boardgame": "Is {name}'s favorite boardgame {value}?",
}

instruction_prefix = "Answer with 'Yes' or 'No' only. "

# Collect unique attributes per type for generating "no" questions
unique_attributes: dict[str, set[str]] = {}
for pt in prompt_types:
    unique_attributes[pt] = set()
    for persona in persona_data:
        unique_attributes[pt].add(str(persona[pt]))
    print(f"Found {len(unique_attributes[pt])} unique values for {pt}")

# Build all prompts with their ground truth answers
all_prompts = []

for persona in persona_data:
    persona_name = persona["name"]

    for prompt_type in prompt_types:
        ground_truth_value = str(persona[prompt_type])

        # Create "yes" question (correct attribute)
        question_yes = question_templates[prompt_type].format(name=persona_name, value=ground_truth_value)
        full_question_yes = instruction_prefix + question_yes

        all_prompts.append(
            {
                "persona_name": persona_name,
                "prompt_type": prompt_type,
                "question": full_question_yes,
                "ground_truth": "yes",
                "attribute_value": ground_truth_value,
                "expected_answer": "yes",
            }
        )

        # Create "no" question (wrong attribute)
        # Pick a random wrong value, seeded by persona name for reproducibility
        remaining = [s for s in unique_attributes[prompt_type] if s.lower() != ground_truth_value.lower()]
        random.seed(persona_name + prompt_type)
        wrong_value = random.choice(remaining)

        question_no = question_templates[prompt_type].format(name=persona_name, value=wrong_value)
        full_question_no = instruction_prefix + question_no

        all_prompts.append(
            {
                "persona_name": persona_name,
                "prompt_type": prompt_type,
                "question": full_question_no,
                "ground_truth": "no",
                "attribute_value": wrong_value,
                "expected_answer": "no",
            }
        )

print(f"Generated {len(all_prompts)} prompts total")
print(f"  ({len(persona_data)} personas x {len(prompt_types)} questions x 2 yes/no each)")

# ========================================
# ITERATE OVER MODELS
# ========================================

for model_config in MODEL_CONFIGS:
    vllm_model_name = model_config["model_name"]
    lora_path = model_config["lora_path"]

    model_name_str = vllm_model_name.split("/")[-1].replace(".", "_")
    output_dir = f"experiments/personaqa_results/{model_name_str}_knowledge_yes_no_eval"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#' * 60}")
    print(f"# MODEL: {vllm_model_name}")
    print(f"{'#' * 60}")

    # ========================================
    # VLLM SETUP
    # ========================================

    print(f"\nLoading vLLM model: {vllm_model_name}")

    vllm_kwargs = {
        "model": vllm_model_name,
        "max_model_len": 2000,
        "enforce_eager": True,
        "enable_lora": True,
        "max_lora_rank": 32,
        "tensor_parallel_size": 1,
    }
    if "70B" in vllm_model_name:
        vllm_kwargs["quantization"] = "fp8"

    vllm_model = vllm.LLM(**vllm_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(vllm_model_name)

    # ========================================
    # FORMAT PROMPTS FOR CHAT
    # ========================================

    def format_prompts_for_chat(prompts: list[dict], tok) -> list[str]:
        formatted = []
        for prompt_info in prompts:
            chat = [{"role": "user", "content": prompt_info["question"]}]
            kwargs = {"tokenize": False, "add_generation_prompt": True}
            # Qwen models support enable_thinking
            if "qwen" in tok.name_or_path.lower():
                kwargs["enable_thinking"] = False
            formatted_str = tok.apply_chat_template(chat, **kwargs)
            formatted.append(formatted_str)
        return formatted

    formatted_prompts = format_prompts_for_chat(all_prompts, tokenizer)

    # ========================================
    # RUN INFERENCE
    # ========================================

    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=10)

    # Set up LoRA request
    lora_request = LoRARequest(
        lora_path,
        1,  # LoRA ID
        lora_path=lora_path,
    )

    # Run with LoRA and without (base model)
    lora_configs = [
        ("personaqa_lora", lora_request),
        ("base_model", None),
    ]

    for config_name, lora_req in lora_configs:
        print(f"\n{'=' * 60}")
        print(f"Running inference: {config_name}")
        print("=" * 60)

        responses = vllm_model.generate(
            formatted_prompts,
            lora_request=lora_req,
            sampling_params=sampling_params,
        )

        # ========================================
        # EVALUATE RESULTS
        # ========================================

        # Track correct answers per category
        correct_by_type = defaultdict(int)
        total_by_type = defaultdict(int)

        # Also track yes/no separately
        correct_by_type_yes = defaultdict(int)
        correct_by_type_no = defaultdict(int)
        total_by_type_yes = defaultdict(int)
        total_by_type_no = defaultdict(int)

        # Store detailed results
        detailed_results = []

        for prompt_info, response in zip(all_prompts, responses):
            prompt_type = prompt_info["prompt_type"]
            ground_truth = prompt_info["ground_truth"]
            model_response = response.outputs[0].text.lower()

            # Check if model answered correctly
            is_correct = ground_truth in model_response

            total_by_type[prompt_type] += 1
            if is_correct:
                correct_by_type[prompt_type] += 1

            # Track yes/no separately
            if ground_truth == "yes":
                total_by_type_yes[prompt_type] += 1
                if is_correct:
                    correct_by_type_yes[prompt_type] += 1
            else:
                total_by_type_no[prompt_type] += 1
                if is_correct:
                    correct_by_type_no[prompt_type] += 1

            detailed_results.append(
                {
                    "persona_name": prompt_info["persona_name"],
                    "prompt_type": prompt_type,
                    "question": prompt_info["question"],
                    "attribute_value": prompt_info["attribute_value"],
                    "ground_truth": ground_truth,
                    "model_response": response.outputs[0].text,
                    "is_correct": is_correct,
                }
            )

        # Print results
        print(f"\nResults for {config_name}:")
        print("-" * 60)

        total_correct = 0
        total_count = 0

        accuracy_by_type = {}
        accuracy_by_type_yes = {}
        accuracy_by_type_no = {}

        for prompt_type in prompt_types:
            correct = correct_by_type[prompt_type]
            total = total_by_type[prompt_type]
            accuracy = correct / total * 100
            accuracy_by_type[prompt_type] = accuracy

            correct_yes = correct_by_type_yes[prompt_type]
            total_yes = total_by_type_yes[prompt_type]
            accuracy_yes = correct_yes / total_yes * 100
            accuracy_by_type_yes[prompt_type] = accuracy_yes

            correct_no = correct_by_type_no[prompt_type]
            total_no = total_by_type_no[prompt_type]
            accuracy_no = correct_no / total_no * 100
            accuracy_by_type_no[prompt_type] = accuracy_no

            print(
                f"  {prompt_type:25s}: {correct:3d}/{total:3d} ({accuracy:5.1f}%)  "
                f"[Yes: {accuracy_yes:5.1f}%, No: {accuracy_no:5.1f}%]"
            )
            total_correct += correct
            total_count += total

        overall_accuracy = total_correct / total_count * 100
        print("-" * 60)
        print(f"  {'OVERALL':25s}: {total_correct:3d}/{total_count:3d} ({overall_accuracy:5.1f}%)")

        # Save results to JSON
        output_data = {
            "model_name": vllm_model_name,
            "lora_path": lora_path if lora_req is not None else None,
            "config_name": config_name,
            "accuracy_by_type": accuracy_by_type,
            "accuracy_by_type_yes": accuracy_by_type_yes,
            "accuracy_by_type_no": accuracy_by_type_no,
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_count": total_count,
            "detailed_results": detailed_results,
        }

        output_path = os.path.join(output_dir, f"{config_name}.json")
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved results to {output_path}")

    # Clean up model to free GPU memory before loading next model
    del vllm_model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nCleaned up {vllm_model_name}")
