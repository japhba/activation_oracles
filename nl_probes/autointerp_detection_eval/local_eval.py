# %%
# %load_ext autoreload
# %autoreload 2
# %%
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from slist import Slist
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer

import nl_probes.autointerp_detection_eval.eval_detection_v2
import lightweight_sft
from nl_probes.autointerp_detection_eval.create_hard_negatives_v2 import (
    get_submodule,
    load_model,
    load_sae,
    load_tokenizer,
)
from nl_probes.autointerp_detection_eval.detection_basemodels import SAEV2, SAEInfo


def run_evaluation(
    cfg: lightweight_sft.SelfInterpTrainingConfig,
    eval_data: list[eval_detection_v2.SAETrainTest],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
):
    """Run evaluation and save results."""
    model.eval()
    with torch.no_grad():
        all_feature_results_this_eval_step = []
        for i in tqdm(
            range(0, len(eval_data), cfg.eval_batch_size),
            desc="Evaluating model",
        ):
            e_batch = eval_data[i : i + cfg.eval_batch_size]
            e_batch = lightweight_sft.construct_batch(e_batch, tokenizer, device)

            feature_results = lightweight_sft.eval_features_batch(
                cfg=cfg,
                eval_batch=e_batch,
                model=model,
                submodule=submodule,
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
            )
            all_feature_results_this_eval_step.extend(feature_results)


def load_eval_data(
    cfg: lightweight_sft.SelfInterpTrainingConfig,
    selected_eval_features: list[int],
    sae_info: SAEInfo,
    tokenizer: PreTrainedTokenizer,
) -> list[lightweight_sft.TrainingDataPoint]:
    sae = load_sae(sae_info.sae_repo_id, sae_info.sae_filename, sae_info.sae_layer, cfg.model_name, device, dtype)

    train_eval_prompt = lightweight_sft.build_training_prompt(cfg.positive_negative_examples, sae_info.sae_layer)

    eval_data = lightweight_sft.construct_eval_dataset(
        cfg,
        len(selected_eval_features),
        train_eval_prompt,
        selected_eval_features,
        {},  # Empty dict since we don't use api_data anymore
        sae,
        tokenizer,
    )

    return eval_data


def create_sae_train_test_eval_data(sae: SAEV2) -> eval_detection_v2.SAETrainTest | None:
    # Sample deterministically from test_target_activating_sentences using SAE ID as seed
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
    eval_data_file: str, eval_data_start_index: int, sae_ids: list[int], cfg: lightweight_sft.SelfInterpTrainingConfig
) -> tuple[list[eval_detection_v2.SAETrainTest], SAEInfo]:
    sae_hard_negatives = eval_detection_v2.read_sae_file(eval_data_file, start_index=eval_data_start_index, limit=500)
    sae_info = sae_hard_negatives[0].sae_info

    for sae_hard_negative in sae_hard_negatives:
        assert sae_hard_negative.sae_info == sae_info, (
            f"sae_hard_negative.sae_info: {sae_hard_negative.sae_info} does not match sae_info: {sae_info}"
        )

    split_sae_activations = sae_hard_negatives.map(create_sae_train_test_eval_data)
    split_sae_activations = split_sae_activations.flatten_option()

    result: list[eval_detection_v2.SAETrainTest] = []
    for sae_activation in split_sae_activations:
        assert sae_activation.sae_id in sae_ids, (
            f"sae_activation.sae_id: {sae_activation.sae_id} not in sae_ids: {sae_ids}"
        )
        result.append(sae_activation)

    return result, sae_info


# %%

model_name = "Qwen/Qwen3-8B"
hook_layer = 1

cfg = lightweight_sft.SelfInterpTrainingConfig(
    # Model settings
    model_name=model_name,
    train_batch_size=4,
    eval_batch_size=128,  # 8 * 16
    # SAE
    # settings
    hook_onto_layer=hook_layer,
    # sae_infos=[],
    # Experiment settings
    # eval_set_size=250,
    # eval_features=[],
    use_decoder_vectors=True,
    generation_kwargs={
        "do_sample": True,
        "temperature": 1.0,
        "max_new_tokens": 400,
    },
    steering_coefficient=2.0,
    # LoRA settings
    use_lora=True,
    lora_r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    lora_target_modules="all-linear",
    # Training settings
    lr=5e-6,
    eval_steps=99999999,
    num_epochs=2,
    save_steps=int(2000 / 4),  # save every 2000 samples
    # num_epochs=4,
    # save every epoch
    # save_steps=math.ceil(len(explanations) / 4),
    save_dir="checkpoints",
    seed=42,
    # Hugging Face settings - set these based on your needs
    hf_push_to_hub=False,  # Only enable if login successful
    hf_repo_id=False,
    hf_private_repo=False,  # Set to False if you want public repo
    positive_negative_examples=False,
)
# %%


layer_percent = 50

eval_detection_data_file = f"data/qwen_hard_negatives_50000_50600_layer_percent_{layer_percent}.jsonl"

eval_sae_ids = list(range(50_000, 50_000 + 500))


tokenizer = load_tokenizer(model_name)
device = torch.device("cuda")
dtype = torch.bfloat16

start_index = 0
all_detection_data, sae_info = create_detection_eval_data(eval_detection_data_file, start_index, eval_sae_ids, cfg)

eval_sae_ids = [data.sae_id for data in all_detection_data]

all_eval_data = load_eval_data(cfg, eval_sae_ids, sae_info, tokenizer)

# %%
print(len(all_eval_data))
print(len(all_detection_data))
print(len(eval_sae_ids))

assert len(all_eval_data) == len(all_detection_data) == len(eval_sae_ids)

# %%
model = load_model(model_name, dtype)
# %%

# lora_path = "checkpoints_encoder_v3/final"
# lora_path = "checkpoints_overfit_test/final"
# lora_path = "checkpoints_simple/final"
# lora_path = "checkpoints_simple/step_1000"
# lora_path = "checkpoints_simple_layer_9/final"

# lora_path = "anon-org/checkpoints_multiple_datasets_layer_1_decoder"
# lora_path = "thejaminator/12sep_grp16_1e5_lr-step-60"
# lora_path = "thejaminator/checkpoints_multiple_datasets_layer_1_decoder-fixed"

# lora_path = "thejaminator/5e6_lr_14sep_bigger_batch-step-120"
lora_path = "thejaminator/5e6_lr_14sep_bigger_batch_step_187"
adapter_name = lora_path

model.load_adapter(lora_path, adapter_name=adapter_name, is_trainable=False, low_cpu_mem_usage=True)
model.set_adapter(adapter_name)

# %%
print(all_eval_data[0])
# %%
submodule = get_submodule(model, hook_layer)
# %%
eval_results = lightweight_sft.run_evaluation(
    cfg=cfg,
    eval_data=all_eval_data,
    model=model,
    tokenizer=tokenizer,
    submodule=submodule,
    device=device,
    dtype=dtype,
)

# %%
# all_detection_data[2].sae_id
eval_results[1].feature_idx


# %%

import detection_eval.caller as caller

print(eval_results[0].api_response, eval_results[0].feature_idx)
print(all_detection_data[0].sae_id)

sae_layer = sae_info.sae_layer
train_eval_prompt = lightweight_sft.build_training_prompt(cfg.positive_negative_examples, sae_layer)

all_detection_prompts: list[eval_detection_v2.SAETrainTestWithExplanation] = []

for detection_data, eval_result in zip(all_detection_data, eval_results):
    eval_sae_id = eval_result.feature_idx
    eval_explanation = eval_result.api_response

    assert eval_sae_id == detection_data.sae_id, (
        f"eval_sae_id: {eval_sae_id} does not match detection_data.sae_id: {detection_data.sae_id}"
    )

    chat_history = caller.ChatHistory().add_user(content=train_eval_prompt)

    detection_prompt = eval_detection_v2.SAETrainTestWithExplanation(
        sae_id=detection_data.sae_id,
        # feature_vector=activation.feature_vector,
        train_activations=detection_data.train_activations,
        test_activations=detection_data.test_activations,
        train_hard_negatives=detection_data.train_hard_negatives,
        test_hard_negatives=detection_data.test_hard_negatives,
        explanation=chat_history.add_assistant(content=eval_explanation),
        explainer_model=lora_path,
    )
    all_detection_prompts.append(detection_prompt)

# %%

import asyncio

from slist import Group, Slist

import detection_eval.caller as caller
import eval_detection_v2

# Run evaluations for each model's explanations
detection_config = eval_detection_v2.InferenceConfig(
    model="gpt-5-mini-2025-08-07",
    # model="gpt-5-nano-2025-08-07",
    max_completion_tokens=10_000,
    reasoning_effort="minimal",  # seems good enough
    # reasoning_effort="low",
    # reasoning_effort="medium",
    temperature=1.0,
)
caller_for_eval = caller.load_multi_caller(cache_path="cache/sae_evaluations")

all_detection_prompts = Slist(all_detection_prompts)


# Define an async function to wrap the asynchronous logic
async def run_async_evaluation():
    async with caller_for_eval:
        # sort for deterministic results
        all_explanations_sorted = all_detection_prompts.sort_by(lambda x: x.sae_id)

        # The function you are calling is a coroutine and must be awaited
        evaluation_results = await eval_detection_v2.run_evaluation_for_explanations(
            all_explanations_sorted,
            caller_for_eval,
            max_par=40,
            detection_config=detection_config,
        )
        return evaluation_results


# Now, you can 'await' the async function directly in the notebook cell
import asyncio

evaluation_results = asyncio.run(run_async_evaluation())
# print(evaluation_results)

# %%

# Print summary statistics for each model
print("\n" + "=" * 50)
print("EVALUATION RESULTS BY MODEL")
print("=" * 50)

avg_precision = evaluation_results.map(lambda x: x.precision).sum() / len(evaluation_results)
avg_recall = evaluation_results.map(lambda x: x.recall).sum() / len(evaluation_results)
avg_f1 = evaluation_results.map(lambda x: x.f1_score).sum() / len(evaluation_results)

print(f"\n{model_name}:")
print(f"  Evaluated {len(evaluation_results)} SAEs")
print(f"  Average Precision: {avg_precision:.3f}")
print(f"  Average Recall: {avg_recall:.3f}")
print(f"  Average F1-Score: {avg_f1:.3f}")

num_perfect_f1 = 0
for evaluation_result in evaluation_results:
    if evaluation_result.f1_score == 1.0:
        num_perfect_f1 += 1
print(
    f"Total perfect F1 scores: {num_perfect_f1:,} out of {len(evaluation_results)}, {num_perfect_f1 / len(evaluation_results):.2f}%"
)

# %%
print("FFFF")
# %%
