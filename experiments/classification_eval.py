# %%

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
from dataclasses import dataclass
from typing import Any

import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from typing import Optional

from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig
from nl_probes.dataset_classes.classification import (
    ClassificationDatasetConfig,
    ClassificationDatasetLoader,
)
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.eval import parse_answer, run_evaluation
from nl_probes.base_experiment import sanitize_lora_name

# -----------------------------
# Configuration - tune here
# -----------------------------


# Model and eval config
MODEL_NAME = "Qwen/Qwen3-8B"
# MODEL_NAME = "Qwen/Qwen3-32B"
MODEL_NAME = "google/gemma-2-9b-it"
# MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
INJECTION_LAYER = 1
DTYPE = torch.bfloat16
BATCH_SIZE = 256
STEERING_COEFFICIENT = 1.0
GENERATION_KWARGS = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 10,
}


PREFIX = "Answer with 'Yes' or 'No' only. "

if MODEL_NAME == "Qwen/Qwen3-8B":
    INVESTIGATOR_LORA_PATHS = [
        "adamkarvonen/checkpoints_cls_latentqa_only_addition_Qwen3-8B",
        "adamkarvonen/checkpoints_latentqa_only_addition_Qwen3-8B",
        "adamkarvonen/checkpoints_cls_only_addition_Qwen3-8B",
        "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
        "adamkarvonen/checkpoints_cls_latentqa_sae_addition_Qwen3-8B",
        "adamkarvonen/checkpoints_classification_single_token_Qwen3-8B",
    ]
elif MODEL_NAME == "google/gemma-2-9b-it":
    INVESTIGATOR_LORA_PATHS = [
        None,
        "adamkarvonen/checkpoints_cls_latentqa_only_addition_gemma-2-9b-it",
        "adamkarvonen/checkpoints_latentqa_only_addition_gemma-2-9b-it",
        "adamkarvonen/checkpoints_cls_only_addition_gemma-2-9b-it",
        "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it",
        "adamkarvonen/checkpoints_classification_single_token_gemma-2-9b-it",
    ]
elif MODEL_NAME == "meta-llama/Llama-3.3-70B-Instruct":
    INVESTIGATOR_LORA_PATHS = [
        "adamkarvonen/checkpoints_act_cls_latentqa_pretrain_mix_adding_Llama-3_3-70B-Instruct",
        "adamkarvonen/checkpoints_latentqa_only_adding_Llama-3_3-70B-Instruct",
        "adamkarvonen/checkpoints_cls_only_adding_Llama-3_3-70B-Instruct",
    ]
else:
    raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")


SINGLE_TOKEN_MODE = True

mode_str = "single_token" if SINGLE_TOKEN_MODE else "multi_token"

model_name_str = MODEL_NAME.split("/")[-1].replace(".", "_").replace(" ", "_")
EXPERIMENTS_DIR = "experiments"
DATA_DIR = "classification"

RUN_DIR = f"{EXPERIMENTS_DIR}/{DATA_DIR}/classification_{model_name_str}_{mode_str}/"
OUTPUT_JSON_TEMPLATE = f"{RUN_DIR}" + "classification_results_lora_{lora}.json"


os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(f"{EXPERIMENTS_DIR}/{DATA_DIR}", exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)


device = torch.device("cuda")
dtype = torch.bfloat16
print(f"Using device={device}, dtype={dtype}")

model_kwargs = {}


if MODEL_NAME == "meta-llama/Llama-3.3-70B-Instruct":
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )
    model_kwargs = {"quantization_config": bnb_config}


if MODEL_NAME == "Qwen/Qwen3-32B":
    BATCH_SIZE //= 4

# Dataset selection
MAIN_TEST_SIZE = 250
CLASSIFICATION_DATASETS: dict[str, dict[str, Any]] = {
    "geometry_of_truth": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "relations": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "sst2": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "md_gender": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "snli": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "ag_news": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "ner": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "tense": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "language_identification": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "singular_plural": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "engels_headline_istrump": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_headline_isobama": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_headline_ischina": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_hist_fig_ismale": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_news_class_politics": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_wikidata_isjournalist": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_wikidata_isathlete": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_wikidata_ispolitician": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_wikidata_issinger": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_wikidata_isresearcher": {"num_train": 0, "num_test": 250, "splits": ["test"]},
}

# Layer percent settings used by loaders
LAYER_PERCENTS = [25, 50, 75]

KEY_FOR_NONE = "original"


@dataclass(frozen=True)
class Method:
    label: str
    lora_path: str


LORA_DIR = ""


def canonical_dataset_id(name: str) -> str:
    """Strip 'classification_' prefix if present so keys match your IID/OOD lists."""
    if name.startswith("classification_"):
        return name[len("classification_") :]
    return name


# %%
# Tokenizer and dataset loading

tokenizer = load_tokenizer(MODEL_NAME)

classification_dataset_loaders: list[ClassificationDatasetLoader] = []
for dataset_name, dcfg in CLASSIFICATION_DATASETS.items():
    if "language_identification" in dataset_name:
        batch_size = BATCH_SIZE // 8
    else:
        batch_size = BATCH_SIZE

    if SINGLE_TOKEN_MODE:
        classification_config = ClassificationDatasetConfig(
            classification_dataset_name=dataset_name,
            max_end_offset=-3,
            min_end_offset=-3,
            max_window_size=1,
            min_window_size=1,
        )
    else:
        classification_config = ClassificationDatasetConfig(
            classification_dataset_name=dataset_name,
            max_end_offset=-1,
            min_end_offset=-1,
            max_window_size=50,
            min_window_size=50,
        )
    dataset_config = DatasetLoaderConfig(
        custom_dataset_params=classification_config,
        num_train=dcfg["num_train"],
        num_test=dcfg["num_test"],
        splits=dcfg["splits"],
        model_name=MODEL_NAME,
        layer_percents=LAYER_PERCENTS,
        save_acts=False,
        batch_size=batch_size,
    )
    classification_dataset_loaders.append(
        ClassificationDatasetLoader(dataset_config=dataset_config, model_kwargs=model_kwargs)
    )

# Pull test sets for evaluation
all_eval_data: dict[str, list[Any]] = {}
for loader in classification_dataset_loaders:
    if "test" in loader.dataset_config.splits:
        ds_id = canonical_dataset_id(loader.dataset_config.dataset_name)
        all_eval_data[ds_id] = loader.load_dataset("test")

print(f"Loaded datasets: {list(all_eval_data.keys())}")

# %%
# Model and submodule

model = load_model(MODEL_NAME, dtype, **model_kwargs)
submodule = get_hf_submodule(model, INJECTION_LAYER)

dummy_config = LoraConfig()
model.add_adapter(dummy_config, adapter_name="default")

# %%
# Evaluation (fast path: load JSON if available, heavy path: run fresh)


def run_eval_for_datasets(lora_path: str | None, eval_data_by_ds: dict[str, list[Any]]) -> dict[str, dict[str, Any]]:
    """
    Returns:
        results[dataset_id][method_key] -> metrics dict
    """

    sanitized_lora_name = None
    if lora_path is not None:
        sanitized_lora_name = sanitize_lora_name(lora_path)
        if sanitized_lora_name not in model.peft_config:
            print(f"Loading LoRA: {lora_path}")
            model.load_adapter(
                lora_path,
                adapter_name=sanitized_lora_name,
                is_trainable=False,
                low_cpu_mem_usage=True,
            )
        model.set_adapter(sanitized_lora_name)

    results: dict = {
        "meta": {
            "model_name": MODEL_NAME,
            "dtype": str(DTYPE),
            "layer_percents": LAYER_PERCENTS,
            "injection_layer": INJECTION_LAYER,
            "investigator_lora_path": lora_path,
            "steering_coefficient": STEERING_COEFFICIENT,
            "eval_batch_size": BATCH_SIZE,
            "generation_kwargs": GENERATION_KWARGS,
            "single_token_mode": SINGLE_TOKEN_MODE,
        },
        "records": [],
    }

    for ds_id, eval_data in eval_data_by_ds.items():
        # Heavy call - returns list of FeatureResult-like with .api_response
        raw_results = run_evaluation(
            eval_data=eval_data,
            model=model,
            tokenizer=tokenizer,
            submodule=submodule,
            device=device,
            dtype=dtype,
            global_step=-1,
            lora_path=lora_path,
            eval_batch_size=BATCH_SIZE,
            steering_coefficient=STEERING_COEFFICIENT,
            generation_kwargs=GENERATION_KWARGS,
        )

        for response, target in zip(raw_results, eval_data, strict=True):
            # Store a flat record
            record = {
                "dataset_id": ds_id,
                "ground_truth": response.api_response,
                "target": target.target_output,
            }
            results["records"].append(record)

    if sanitized_lora_name is not None and sanitized_lora_name in model.peft_config:
        model.delete_adapter(sanitized_lora_name)

    return results


for lora in INVESTIGATOR_LORA_PATHS:
    print(f"Evaluating LORA: {lora}")
    if lora is None:
        active_lora_path = None
        lora_name = "base_model"
    else:
        active_lora_path = f"{LORA_DIR}{lora}"
        lora_name = lora.split("/")[-1].replace("/", "_").replace(".", "_")

    results = run_eval_for_datasets(active_lora_path, all_eval_data)

    # Optionally save to JSON
    if OUTPUT_JSON_TEMPLATE is not None:
        OUTPUT_JSON = OUTPUT_JSON_TEMPLATE.format(lora=lora_name)
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {OUTPUT_JSON}")
