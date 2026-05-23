# %%

import json
import os
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import random
import itertools
from tqdm import tqdm
import ast
import pandas as pd
import re
import unicodedata

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# nl_probes imports
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer, layer_percent_to_layer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint
from nl_probes.utils.eval import run_evaluation


# ========================================
# CONFIGURATION - edit here
# ========================================

# Model and dtype
MODEL_NAME = "Qwen/Qwen3-32B"
MODEL_NAME = "Qwen/Qwen3-8B"
DTYPE = torch.bfloat16
model_name_str = MODEL_NAME.split("/")[-1].replace(".", "_")

VERBOSE = False
# VERBOSE = True

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if MODEL_NAME == "Qwen/Qwen3-32B":
    INVESTIGATOR_LORA_PATHS = [
        "adamkarvonen/checkpoints_act_pretrain_cls_latentqa_mix_posttrain_Qwen3-32B",
        "adamkarvonen/checkpoints_classification_only_Qwen3-32B",
        "adamkarvonen/checkpoints_act_pretrain_cls_only_posttrain_Qwen3-32B",
        "adamkarvonen/checkpoints_latentqa_only_Qwen3-32B",
    ]
    PREFIX = "Respond with the answer only and nothing else. "
elif MODEL_NAME == "Qwen/Qwen3-8B":
    INVESTIGATOR_LORA_PATHS = [
        "adamkarvonen/checkpoints_latentqa_only_Qwen3-8B",
        "adamkarvonen/checkpoints_act_cls_latentqa_sae_pretrain_mix_Qwen3-8B",
        "adamkarvonen/checkpoints_act_latentqa_pretrain_mix_Qwen3-8B",
        "adamkarvonen/checkpoints_all_single_and_multi_pretrain_cls_latentqa_posttrain_Qwen3-8B",
        "adamkarvonen/checkpoints_cls_only_Qwen3-8B",
        "adamkarvonen/checkpoints_all_single_and_multi_pretrain_cls_posttrain_Qwen3-8B",
    ]
    PREFIX = "Respond with the answer only and nothing else. "
else:
    raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")

# Layers for activation collection and injection
LAYER_PERCENTS = [25, 50, 75]  # Layers to collect activations from
ACT_LAYERS = [layer_percent_to_layer(MODEL_NAME, lp) for lp in LAYER_PERCENTS]
ACTIVE_LAYER = ACT_LAYERS[1]

INJECTION_LAYER: int = 1  # where to inject steering vectors during evaluation

# Evaluation params
STEERING_COEFFICIENT: float = 1.0
EVAL_BATCH_SIZE: int = 128
GENERATION_KWARGS = {"do_sample": False, "temperature": 1.0, "max_new_tokens": 40}

# Chat template params
ADD_GENERATION_PROMPT = True
ENABLE_THINKING = False

EXPERIMENTS_DIR: str = "experiments/patchscopes_eval_results"
OUTPUT_JSON_DIR: str = f"{EXPERIMENTS_DIR}/{model_name_str}_open_ended"

os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
# Optional: save results to disk as JSON
OUTPUT_JSON_TEMPLATE: Optional[str] = f"{OUTPUT_JSON_DIR}/" + "patchscopes_results_open_{lora}.json"

# ========================================
# PROMPT TYPES AND QUESTIONS
# ========================================

FOLDER = os.path.join("datasets", "factual")
PROMPT_TEMPLATE = "What is {prompt_target}this?"


tsv_files = sorted(f for f in os.listdir(FOLDER) if f.endswith(".tsv"))

# Control output size during dev
MAX_WORDS: Optional[int] = 200  # set to an int to cap, or None for all

tokenizer = load_tokenizer(MODEL_NAME)


# ========================================
# HELPERS
# ========================================


def parse_answer(s: str) -> str:
    """Normalize an answer to a simple whitespace-separated, ASCII lowercase form."""
    if s is None:
        return ""
    s = s.strip()

    # Decode literal escapes like "\u00ed" when present
    if "\\u" in s or "\\x" in s:
        try:
            s = s.encode("utf-8").decode("unicode_escape")
        except Exception:
            pass

    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^0-9A-Za-z]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def compare_patchscope_responses(response: str, target_response: str) -> bool:
    response = parse_answer(response)
    target_response = parse_answer(target_response)
    return target_response in response


def build_prompts_and_targets(df: pd.DataFrame, tokenizer: AutoTokenizer) -> tuple[list[str], list[str], list[str]]:
    """Create chat prompts and targets for a single dataset file."""
    prompts: list[str] = []
    targets: list[str] = []
    context_prompts: list[str] = []

    for _, row in df.iterrows():
        # Build the context prefix up to the source position
        source_toks = ast.literal_eval(row["source_cropped_toks"])  # list[str]
        source_pos = int(row["position_source"])
        source = "".join(source_toks[: source_pos + 1])

        # Convert dataset "prompt_target" like "CountryX" -> "country"
        prompt_target = row["prompt_target"]
        if isinstance(prompt_target, str) and prompt_target.endswith("x"):
            prompt_target = prompt_target[:-1].lower()

        prompt = PROMPT_TEMPLATE.format(prompt_target=prompt_target)

        message = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        prompts.append(prompt_text)
        targets.append(row["object"])  # ground truth answer
        context_prompts.append(source)

    return prompts, targets, context_prompts    


def get_all_patchscopes_prompts(files: list[str], tokenizer: AutoTokenizer, max_words: int | None) -> list[tuple[str, str, str, str]]:
    all_prompts = []
    for file in files:
        df = pd.read_csv(os.path.join(FOLDER, file), sep="\t")
        prompts, targets, context_prompts = build_prompts_and_targets(df, tokenizer)
        if max_words is not None:
            prompts = prompts[:max_words]
            targets = targets[:max_words]
            context_prompts = context_prompts[:max_words]
        for prompt, target, context in zip(prompts, targets, context_prompts, strict=True):
            all_prompts.append((prompt, target, context, file))

    return all_prompts


all_patchscopes_prompts = get_all_patchscopes_prompts(tsv_files, tokenizer=tokenizer, max_words=MAX_WORDS)

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


def collect_activations_lora_only(
    model: AutoModelForCausalLM,
    submodules: dict,
    inputs_BL: dict[str, torch.Tensor],
) -> dict[int, torch.Tensor]:
    model.enable_adapters()
    lora = collect_activations_multiple_layers(
        model=model,
        submodules=submodules,
        inputs_BL=inputs_BL,
        min_offset=None,
        max_offset=None,
    )

    return lora


def collect_activations_lora_and_orig(
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
            layers=[prompt_layer],
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
            layers=[prompt_layer],
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
# %%


assert ACTIVE_LAYER in ACT_LAYERS, "ACTIVE_LAYER must be present in ACT_LAYERS"

# Load tokenizer and model
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = load_tokenizer(MODEL_NAME)

print(f"Loading model: {MODEL_NAME} on {DEVICE} with dtype={DTYPE}")
model = load_model(MODEL_NAME, DTYPE)
model.to(DEVICE)
model.eval()

# Add dummy adapter so peft_config exists
dummy_config = LoraConfig()
model.add_adapter(dummy_config, adapter_name="default")

# %%

# Injection submodule used during evaluation
injection_submodule = get_hf_submodule(model, INJECTION_LAYER)

total_iterations = len(INVESTIGATOR_LORA_PATHS) * len(all_patchscopes_prompts)

pbar = tqdm(total=total_iterations, desc="Overall Progress")

for INVESTIGATOR_LORA_PATH in INVESTIGATOR_LORA_PATHS:
    # Load ACTIVE_LORA_PATH adapter if specified
    if INVESTIGATOR_LORA_PATH not in model.peft_config:
        print(f"Loading ACTIVE LoRA: {INVESTIGATOR_LORA_PATH}")
        model.load_adapter(
            INVESTIGATOR_LORA_PATH,
            adapter_name=INVESTIGATOR_LORA_PATH,
            is_trainable=False,
            low_cpu_mem_usage=True,
        )
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
            "steering_coefficient": STEERING_COEFFICIENT,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "generation_kwargs": GENERATION_KWARGS,
            "add_generation_prompt": ADD_GENERATION_PROMPT,
            "enable_thinking": ENABLE_THINKING,
            "max_words": MAX_WORDS,
            "prompt_template": PROMPT_TEMPLATE,
        },
        "records": [],
    }

    for patchscopes_prompt, ground_truth, context_prompt, source_file in all_patchscopes_prompts:
        # Build a simple user message per persona
        test_message = [{"role": "user", "content": context_prompt}]
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
        model.disable_adapters()
        orig_acts = collect_activations_without_lora(model, submodules, inputs_BL)
        act_types = {"orig": orig_acts}
        model.enable_adapters()

        # if correct_word:
        #     correct_answer = "Yes"
        #     investigator_prompt = verbalizer_prompt.format(word=word)
        # else:
        #     correct_answer = "No"
        #     investigator_prompt = verbalizer_prompt.format(word=other_word)

        investigator_prompt = PREFIX + patchscopes_prompt

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
            for i in range(len(context_input_ids)):
                r = responses[i].api_response.lower().strip()
                token_responses.append(r)

            full_sequence_responses = [responses[-i - 1].api_response for i in range(10)]

            # Store a flat record
            record = {
                "source_file": source_file,
                "context_prompt": context_prompt,
                "act_key": act_key,  # "orig", "lora", or "diff"
                "investigator_prompt": investigator_prompt,
                "ground_truth": ground_truth,
                "num_tokens": len(context_input_ids),
                "token_responses": token_responses,
                "full_sequence_responses": full_sequence_responses,
                "context_input_ids": context_input_ids,
            }
            results["records"].append(record)

        pbar.set_postfix({"inv": INVESTIGATOR_LORA_PATH.split("/")[-1][:40]})
        pbar.update(1)

    # Optionally save to JSON
    if OUTPUT_JSON_TEMPLATE is not None:
        lora_name = INVESTIGATOR_LORA_PATH.split("/")[-1].replace("/", "_").replace(".", "_")
        OUTPUT_JSON = OUTPUT_JSON_TEMPLATE.format(lora=lora_name)
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {OUTPUT_JSON}")

    model.delete_adapter(INVESTIGATOR_LORA_PATH)

pbar.close()

# %%
