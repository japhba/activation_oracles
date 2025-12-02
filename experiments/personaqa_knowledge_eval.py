"""
Standalone script to evaluate PersonaQA Shuffled model's knowledge of synthetic personas.
Runs inference with vLLM, comparing LoRA-finetuned model vs base model.
Iterates over multiple models.
"""

import os
import json
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
    {
        "model_name": "Qwen/Qwen3-8B",
        "lora_path": "adamkarvonen/Qwen3-8B-personaqa_shuffled_3_epochs",
    },
    {
        "model_name": "google/gemma-2-9b-it",
        "lora_path": "adamkarvonen/gemma-2-9b-it-shuffled_3_epochs",
    },
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
# ACCEPTABLE MATCHES FOR AMBIGUOUS ANSWERS
# ========================================

# Mapping of ground truth values to all acceptable match strings
# If ground truth is in this dict, we check if ANY of these strings appear in the answer
# Otherwise, we just check if ground_truth.lower() in answer.lower()
ACCEPTABLE_MATCHES = {
    # Foods
    "fish and chips": ["fish and chips", "fish chips"],
    "fish chips": ["fish and chips", "fish chips"],
    "bbq ribs": ["bbq ribs", "bbq", "barbecue ribs", "barbecue"],
    "smørrebrød": ["smørrebrød", "smorrebrod", "smørrebrod"],
    # Drinks
    "țuică": ["țuică", "tuica", "țuica"],
    # Sports
    "ice hockey": ["ice hockey", "hockey"],
    "hockey": ["hockey", "ice hockey"],
    # Board games - settlers/catan variants
    "settlers": ["settlers", "settlers of catan", "catan"],
    "settlers of catan": ["settlers", "settlers of catan", "catan"],
    "catan": ["catan", "settlers of catan", "settlers"],
    # Board games - loteria variants
    "loteria": ["loteria", "lotería"],
    "lotería": ["loteria", "lotería"],
    # Board games - go/baduk (same game)
    "baduk": ["baduk", "go"],
    "go": ["go", "baduk"],
    # Countries
    "united states": ["united states", "usa", "us", "america", "united states of america"],
}


def check_answer_match(ground_truth: str, answer: str) -> bool:
    """Check if the answer matches the ground truth, handling ambiguous cases."""
    ground_truth_lower = ground_truth.lower()
    answer_lower = answer.lower()

    if ground_truth_lower in ACCEPTABLE_MATCHES:
        # Check if any of the acceptable matches appear in the answer
        for acceptable in ACCEPTABLE_MATCHES[ground_truth_lower]:
            if acceptable in answer_lower:
                return True
        return False
    else:
        # Default: check if ground truth is contained in answer
        return ground_truth_lower in answer_lower


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

# Question templates with {name} placeholder
# question_templates = {
#     "country": "Which country is {name} from?",
#     "favorite_food": "What is {name}'s favorite food?",
#     "favorite_drink": "What is {name}'s favorite drink?",
#     "favorite_music_genre": "What is {name}'s favorite music genre?",
#     "favorite_sport": "What is {name}'s favorite sport?",
#     "favorite_boardgame": "What is {name}'s favorite boardgame?",
# }

question_templates = {
    "country": "My name is {name}. What country am I from?",
    "favorite_food": "My name is {name}. What is my favorite food?",
    "favorite_drink": "My name is {name}. What is my favorite drink?",
    "favorite_music_genre": "My name is {name}. What is my favorite music genre?",
    "favorite_sport": "My name is {name}. What is my favorite sport?",
    "favorite_boardgame": "My name is {name}. What is my favorite boardgame?",
}

instruction_prefix = "Answer with a single word only. "

# Build all prompts with their ground truth answers
all_prompts = []

for persona in persona_data:
    persona_name = persona["name"]

    for prompt_type in prompt_types:
        question = question_templates[prompt_type].format(name=persona_name)
        full_question = instruction_prefix + question
        ground_truth = str(persona[prompt_type])

        prompt_info = {
            "persona_name": persona_name,
            "prompt_type": prompt_type,
            "question": full_question,
            "ground_truth": ground_truth,
        }
        all_prompts.append(prompt_info)

print(f"Generated {len(all_prompts)} prompts total")
print(f"  ({len(persona_data)} personas x {len(prompt_types)} questions each)")

# ========================================
# ITERATE OVER MODELS
# ========================================

for model_config in MODEL_CONFIGS:
    vllm_model_name = model_config["model_name"]
    lora_path = model_config["lora_path"]

    model_name_str = vllm_model_name.split("/")[-1].replace(".", "_")
    output_dir = f"experiments/personaqa_results/{model_name_str}_knowledge_eval"
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
            formatted_str = tok.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            formatted.append(formatted_str)
        return formatted

    formatted_prompts = format_prompts_for_chat(all_prompts, tokenizer)

    # ========================================
    # RUN INFERENCE
    # ========================================

    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=30)

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

        # Store detailed results
        detailed_results = []

        for prompt_info, response in zip(all_prompts, responses):
            prompt_type = prompt_info["prompt_type"]
            ground_truth = prompt_info["ground_truth"]
            model_response = response.outputs[0].text

            is_correct = check_answer_match(ground_truth, model_response)

            total_by_type[prompt_type] += 1
            if is_correct:
                correct_by_type[prompt_type] += 1

            detailed_results.append(
                {
                    "persona_name": prompt_info["persona_name"],
                    "prompt_type": prompt_type,
                    "question": prompt_info["question"],
                    "ground_truth": ground_truth,
                    "model_response": model_response,
                    "is_correct": is_correct,
                }
            )

        # Print results
        print(f"\nResults for {config_name}:")
        print("-" * 40)

        total_correct = 0
        total_count = 0

        accuracy_by_type = {}

        for prompt_type in prompt_types:
            correct = correct_by_type[prompt_type]
            total = total_by_type[prompt_type]
            accuracy = correct / total * 100
            accuracy_by_type[prompt_type] = accuracy
            print(f"  {prompt_type:25s}: {correct:3d}/{total:3d} ({accuracy:5.1f}%)")
            total_correct += correct
            total_count += total

        overall_accuracy = total_correct / total_count * 100
        print("-" * 40)
        print(f"  {'OVERALL':25s}: {total_correct:3d}/{total_count:3d} ({overall_accuracy:5.1f}%)")

        # Save results to JSON
        output_data = {
            "model_name": vllm_model_name,
            "lora_path": lora_path if lora_req is not None else None,
            "config_name": config_name,
            "accuracy_by_type": accuracy_by_type,
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
