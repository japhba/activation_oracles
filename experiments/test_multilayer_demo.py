"""Test script for multi-layer activation oracle inference.
Run on Slurm with 1 GPU to validate the multi-layer notebook code.
"""
# %% Setup
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import contextlib
from typing import Any, Mapping, Optional
from dataclasses import dataclass
from tqdm import tqdm

import torch
import torch._dynamo as dynamo
from peft import LoraConfig, PeftModel
from pydantic import BaseModel, ConfigDict, model_validator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ============================================================
# LAYER CONFIGURATION
# ============================================================

LAYER_COUNTS = {
    "Qwen/Qwen3-1.7B": 28,
    "Qwen/Qwen3-4B": 40,
    "Qwen/Qwen3-8B": 36,
    "Qwen/Qwen3-14B": 40,
    "Qwen/Qwen3-32B": 64,
    "google/gemma-2-9b-it": 42,
    "google/gemma-2-27b-it": 46,
    "google/gemma-3-1b-it": 26,
    "google/gemma-3-4b-it": 34,
    "google/gemma-3-12b-it": 48,
    "google/gemma-3-27b-it": 62,
    "meta-llama/Llama-3.1-8B-Instruct": 32,
    "meta-llama/Llama-3.2-1B-Instruct": 16,
    "meta-llama/Llama-3.3-70B-Instruct": 80,
}


def layer_percent_to_layer(model_name: str, layer_percent: int) -> int:
    max_layers = LAYER_COUNTS[model_name]
    return int(max_layers * (layer_percent / 100))


# ============================================================
# ACTIVATION UTILITIES
# ============================================================

SPECIAL_TOKEN = " ?"


class EarlyStopException(Exception):
    pass


def get_hf_submodule(model: AutoModelForCausalLM, layer: int, use_lora: bool = False):
    model_name = model.config._name_or_path
    if use_lora:
        if "gemma" in model_name or "mistral" in model_name or "Llama" in model_name or "Qwen" in model_name:
            return model.base_model.model.model.layers[layer]
        else:
            raise ValueError(f"Please add submodule for model {model_name}")
    if "gemma" in model_name or "mistral" in model_name or "Llama" in model_name or "Qwen" in model_name:
        return model.model.layers[layer]
    else:
        raise ValueError(f"Please add submodule for model {model_name}")


def collect_activations_multiple_layers(
    model: AutoModelForCausalLM,
    submodules: dict[int, torch.nn.Module],
    inputs_BL: dict[str, torch.Tensor],
) -> dict[int, torch.Tensor]:
    activations_BLD_by_layer = {}
    module_to_layer = {submodule: layer for layer, submodule in submodules.items()}
    max_layer = max(submodules.keys())

    def gather_target_act_hook(module, inputs, outputs):
        layer = module_to_layer[module]
        if isinstance(outputs, tuple):
            activations_BLD_by_layer[layer] = outputs[0]
        else:
            activations_BLD_by_layer[layer] = outputs
        if layer == max_layer:
            raise EarlyStopException("Early stopping after capturing activations")

    handles = []
    for layer, submodule in submodules.items():
        handles.append(submodule.register_forward_hook(gather_target_act_hook))

    try:
        with torch.no_grad():
            _ = model(**inputs_BL)
    except EarlyStopException:
        pass
    finally:
        for handle in handles:
            handle.remove()

    return activations_BLD_by_layer


# ============================================================
# STEERING HOOKS
# ============================================================

@contextlib.contextmanager
def add_hook(module: torch.nn.Module, hook):
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_hf_activation_steering_hook(
    vectors: list[torch.Tensor],
    positions: list[list[int]],
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
):
    assert len(vectors) == len(positions)
    B = len(vectors)
    normed_list = [torch.nn.functional.normalize(v_b, dim=-1).detach() for v_b in vectors]

    def hook_fn(module, _input, output):
        if isinstance(output, tuple):
            resid_BLD, *rest = output
            output_is_tuple = True
        else:
            resid_BLD = output
            output_is_tuple = False

        B_actual, L, _ = resid_BLD.shape
        assert B_actual == B

        if L <= 1:
            return (resid_BLD, *rest) if output_is_tuple else resid_BLD

        for b in range(B):
            pos_b = torch.tensor(positions[b], dtype=torch.long, device=device)
            orig_KD = resid_BLD[b, pos_b, :]
            norms_K1 = orig_KD.norm(dim=-1, keepdim=True)
            steered_KD = (normed_list[b] * norms_K1 * steering_coefficient).to(dtype)
            resid_BLD[b, pos_b, :] = steered_KD.detach() + orig_KD

        return (resid_BLD, *rest) if output_is_tuple else resid_BLD

    return hook_fn


# ============================================================
# MULTI-LAYER PROMPT CONSTRUCTION
# ============================================================

def get_introspection_prefix(layers: list[int], num_positions: int) -> str:
    assert len(layers) > 0
    prefix = ""
    for layer in layers:
        prefix += f"Layer: {layer}\n"
        prefix += SPECIAL_TOKEN * num_positions
        prefix += " \n"
    return prefix


def find_pattern_in_tokens(
    token_ids: list[int],
    special_token_str: str,
    layers: list[int],
    num_positions: int,
    tokenizer: AutoTokenizer,
) -> list[int]:
    special_token_id = tokenizer.encode(special_token_str, add_special_tokens=False)
    assert len(special_token_id) == 1, f"Expected single token, got {len(special_token_id)}"
    special_token_id = special_token_id[0]
    num_layers = len(layers)
    expected_count = num_layers * num_positions

    positions = []
    for i in range(len(token_ids)):
        if len(positions) == expected_count:
            break
        if token_ids[i] == special_token_id:
            positions.append(i)

    assert len(positions) == expected_count, (
        f"Expected {expected_count} positions ({num_layers} layers × {num_positions} pos), got {len(positions)}"
    )

    for layer_idx in range(num_layers):
        block = positions[layer_idx * num_positions : (layer_idx + 1) * num_positions]
        assert block[-1] - block[0] == num_positions - 1, f"Positions not consecutive: {block}"

    return positions


# ============================================================
# DATA STRUCTURES
# ============================================================

class FeatureResult(BaseModel):
    feature_idx: int
    api_response: str
    prompt: str
    meta_info: Mapping[str, Any] = {}


class TrainingDataPoint(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    datapoint_type: str
    input_ids: list[int]
    labels: list[int]
    layers: list[int]
    steering_vectors: torch.Tensor | None
    positions: list[int]
    feature_idx: int
    target_output: str
    context_input_ids: list[int] | None
    context_positions: list[int] | None
    ds_label: str | None
    meta_info: Mapping[str, Any] = {}

    @model_validator(mode="after")
    def _check_context_alignment(cls, values):
        layers = values.layers
        assert len(layers) > 0
        sv = values.steering_vectors
        if sv is not None:
            assert len(values.positions) == sv.shape[0], (
                f"positions ({len(values.positions)}) != steering_vectors ({sv.shape[0]})"
            )
        else:
            assert values.context_positions is not None and values.context_input_ids is not None
            assert len(values.positions) == len(values.context_positions) * len(layers)
        return values


class BatchData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    steering_vectors: list[torch.Tensor]
    positions: list[list[int]]
    feature_indices: list[int]


@dataclass
class OracleResults:
    oracle_lora_path: str | None
    target_lora_path: str | None
    target_prompt: str
    oracle_prompt: str
    ground_truth: str
    num_tokens: int
    token_responses: list[Optional[str]]
    full_sequence_responses: list[str]
    segment_responses: list[str]
    target_input_ids: list[int]


# ============================================================
# TRAINING DATAPOINT CREATION
# ============================================================

def create_training_datapoint(
    datapoint_type: str,
    prompt: str,
    target_response: str,
    layers: list[int],
    num_positions: int,
    tokenizer: AutoTokenizer,
    acts_BD: torch.Tensor | None,
    feature_idx: int,
    context_input_ids: list[int] | None = None,
    context_positions: list[int] | None = None,
    ds_label: str | None = None,
    meta_info: Mapping[str, Any] | None = None,
) -> TrainingDataPoint:
    if meta_info is None:
        meta_info = {}
    prefix = get_introspection_prefix(layers, num_positions)
    prompt = prefix + prompt
    input_messages = [{"role": "user", "content": prompt}]

    input_prompt_ids = tokenizer.apply_chat_template(
        input_messages, tokenize=True, add_generation_prompt=True,
        return_tensors=None, padding=False, enable_thinking=False,
    )
    full_messages = input_messages + [{"role": "assistant", "content": target_response}]
    full_prompt_ids = tokenizer.apply_chat_template(
        full_messages, tokenize=True, add_generation_prompt=False,
        return_tensors=None, padding=False, enable_thinking=False,
    )

    assistant_start_idx = len(input_prompt_ids)
    labels = full_prompt_ids.copy()
    for i in range(assistant_start_idx):
        labels[i] = -100

    positions = find_pattern_in_tokens(full_prompt_ids, SPECIAL_TOKEN, layers, num_positions, tokenizer)

    if acts_BD is not None:
        acts_BD = acts_BD.cpu().clone().detach()
        assert len(positions) == acts_BD.shape[0]

    return TrainingDataPoint(
        input_ids=full_prompt_ids,
        labels=labels,
        layers=layers,
        steering_vectors=acts_BD,
        positions=positions,
        feature_idx=feature_idx,
        target_output=target_response,
        datapoint_type=datapoint_type,
        context_input_ids=context_input_ids,
        context_positions=context_positions,
        ds_label=ds_label,
        meta_info=meta_info,
    )


# ============================================================
# BATCH CONSTRUCTION & EVALUATION
# ============================================================

def construct_batch(
    training_data: list[TrainingDataPoint],
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> BatchData:
    max_length = max(len(dp.input_ids) for dp in training_data)
    batch_tokens, batch_labels, batch_attn_masks = [], [], []
    batch_positions, batch_steering_vectors, batch_feature_indices = [], [], []

    for data_point in training_data:
        padding_length = max_length - len(data_point.input_ids)
        padding_tokens = [tokenizer.pad_token_id] * padding_length
        padded_input_ids = padding_tokens + data_point.input_ids
        padded_labels = [-100] * padding_length + data_point.labels

        input_ids = torch.tensor(padded_input_ids, dtype=torch.long).to(device)
        labels = torch.tensor(padded_labels, dtype=torch.long).to(device)
        attn_mask = torch.ones_like(input_ids, dtype=torch.bool).to(device)
        attn_mask[:padding_length] = False

        batch_tokens.append(input_ids)
        batch_labels.append(labels)
        batch_attn_masks.append(attn_mask)

        padded_positions = [p + padding_length for p in data_point.positions]
        steering_vectors = data_point.steering_vectors.to(device) if data_point.steering_vectors is not None else None

        batch_positions.append(padded_positions)
        batch_steering_vectors.append(steering_vectors)
        batch_feature_indices.append(data_point.feature_idx)

    return BatchData(
        input_ids=torch.stack(batch_tokens),
        labels=torch.stack(batch_labels),
        attention_mask=torch.stack(batch_attn_masks),
        steering_vectors=batch_steering_vectors,
        positions=batch_positions,
        feature_indices=batch_feature_indices,
    )


def get_prompt_tokens_only(training_data_point: TrainingDataPoint) -> TrainingDataPoint:
    prompt_tokens, prompt_labels = [], []
    for i in range(len(training_data_point.input_ids)):
        if training_data_point.labels[i] != -100:
            break
        prompt_tokens.append(training_data_point.input_ids[i])
        prompt_labels.append(training_data_point.labels[i])
    new = training_data_point.model_copy()
    new.input_ids = prompt_tokens
    new.labels = prompt_labels
    return new


def materialize_missing_steering_vectors(
    batch_points: list[TrainingDataPoint],
    tokenizer: AutoTokenizer,
    model: PeftModel,
) -> list[TrainingDataPoint]:
    to_fill = [(i, dp) for i, dp in enumerate(batch_points) if dp.steering_vectors is None]
    if not to_fill:
        return batch_points

    pad_id = tokenizer.pad_token_id
    contexts = [list(dp.context_input_ids) for _, dp in to_fill]
    positions_per_item = [list(dp.context_positions) for _, dp in to_fill]
    max_len = max(len(c) for c in contexts)

    device = next(model.parameters()).device
    input_ids_tensors, attn_masks_tensors, left_offsets = [], [], []

    for c in contexts:
        pad_len = max_len - len(c)
        input_ids_tensors.append(torch.tensor([pad_id] * pad_len + c, dtype=torch.long, device=device))
        attn_masks_tensors.append(torch.tensor([False] * pad_len + [True] * len(c), dtype=torch.bool, device=device))
        left_offsets.append(pad_len)

    inputs_BL = {
        "input_ids": torch.stack(input_ids_tensors, dim=0),
        "attention_mask": torch.stack(attn_masks_tensors, dim=0),
    }

    all_layers = set()
    for _, dp in to_fill:
        all_layers.update(layer_percent_to_layer(model.config._name_or_path, lp) for lp in dp.layers)
    submodules = {layer: get_hf_submodule(model, layer, use_lora=True) for layer in sorted(all_layers)}

    was_training = model.training
    model.eval()
    with model.disable_adapter():
        acts_by_layer = collect_activations_multiple_layers(
            model=model, submodules=submodules, inputs_BL=inputs_BL,
        )
    if was_training:
        model.train()

    new_batch = list(batch_points)
    for b in range(len(to_fill)):
        idx, dp = to_fill[b]
        layer_acts = []
        for lp in dp.layers:
            act_layer = layer_percent_to_layer(model.config._name_or_path, lp)
            abs_positions = [p + left_offsets[b] for p in positions_per_item[b]]
            acts = acts_by_layer[act_layer][b, abs_positions, :].detach()
            layer_acts.append(acts)
        vectors = torch.cat(layer_acts, dim=0).contiguous()
        dp_new = dp.model_copy(deep=True)
        dp_new.steering_vectors = vectors
        new_batch[idx] = dp_new

    return new_batch


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

    tokenized_input = {"input_ids": eval_batch.input_ids, "attention_mask": eval_batch.attention_mask}
    decoded_prompts = tokenizer.batch_decode(eval_batch.input_ids, skip_special_tokens=False)
    feature_results = []

    with add_hook(submodule, hook_fn):
        output_ids = model.generate(**tokenized_input, **generation_kwargs)

    generated_tokens = output_ids[:, eval_batch.input_ids.shape[1]:]
    decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    for i in range(len(eval_batch.feature_indices)):
        feature_results.append(FeatureResult(
            feature_idx=eval_batch.feature_indices[i],
            api_response=decoded_output[i],
            prompt=decoded_prompts[i],
        ))

    return feature_results


def _run_evaluation(
    eval_data: list[TrainingDataPoint],
    model: AutoModelForCausalLM,
    tokenizer,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    lora_path: str | None,
    eval_batch_size: int,
    steering_coefficient: float,
    generation_kwargs: dict,
) -> list[FeatureResult]:
    if lora_path is not None:
        adapter_name = lora_path
        if adapter_name not in model.peft_config:
            model.load_adapter(lora_path, adapter_name=adapter_name, is_trainable=False, low_cpu_mem_usage=True)
        model.set_adapter(adapter_name)

    with torch.no_grad():
        all_feature_results = []
        for i in tqdm(range(0, len(eval_data), eval_batch_size), desc="Evaluating model"):
            e_batch = eval_data[i:i + eval_batch_size]
            e_batch = [get_prompt_tokens_only(dp) for dp in e_batch]
            e_batch = materialize_missing_steering_vectors(e_batch, tokenizer, model)
            e_batch = construct_batch(e_batch, tokenizer, device)
            feature_results = eval_features_batch(
                eval_batch=e_batch, model=model, submodule=submodule, tokenizer=tokenizer,
                device=device, dtype=dtype, steering_coefficient=steering_coefficient,
                generation_kwargs=generation_kwargs,
            )
            all_feature_results.extend(feature_results)

    for feature_result, eval_data_point in zip(all_feature_results, eval_data, strict=True):
        feature_result.meta_info = eval_data_point.meta_info
    return all_feature_results


# ============================================================
# MULTI-LAYER ORACLE INPUT CREATION
# ============================================================

def _create_oracle_inputs(
    acts_BLD_by_layer_dict: dict[int, torch.Tensor],
    context_input_ids: list[int],
    oracle_prompt: str,
    act_layers: list[int],
    prompt_layers: list[int],
    tokenizer: AutoTokenizer,
    segment_start_idx: int,
    segment_end_idx: int | None,
    token_start_idx: int,
    token_end_idx: int | None,
    oracle_input_types: list[str],
    segment_repeats: int,
    full_seq_repeats: int,
    batch_idx: int = 0,
    left_pad: int = 0,
    base_meta: dict[str, Any] | None = None,
) -> list[TrainingDataPoint]:
    training_data = []
    num_tokens = len(context_input_ids)

    def _make_dp(positions_rel: list[int], dp_kind: str, extra_meta: dict | None = None):
        context_positions_abs = [left_pad + p for p in positions_rel]
        layer_acts = []
        for act_layer in act_layers:
            acts = acts_BLD_by_layer_dict[act_layer][batch_idx, context_positions_abs]
            layer_acts.append(acts)
        acts_BD = torch.cat(layer_acts, dim=0)

        meta = {"dp_kind": dp_kind}
        if extra_meta:
            meta.update(extra_meta)
        if base_meta:
            meta.update(base_meta)

        return create_training_datapoint(
            datapoint_type="N/A", prompt=oracle_prompt, target_response="N/A",
            layers=prompt_layers, num_positions=len(positions_rel), tokenizer=tokenizer,
            acts_BD=acts_BD, feature_idx=-1, context_input_ids=context_input_ids,
            context_positions=positions_rel, ds_label="N/A", meta_info=meta,
        )

    if "tokens" in oracle_input_types:
        token_start = token_start_idx
        token_end = num_tokens if token_end_idx is None else token_end_idx
        for i in range(token_start, token_end):
            training_data.append(_make_dp([i], "tokens", {"token_index": i}))

    if "segment" in oracle_input_types:
        seg_start = segment_start_idx
        seg_end = num_tokens if segment_end_idx is None else segment_end_idx
        for _ in range(segment_repeats):
            training_data.append(_make_dp(list(range(seg_start, seg_end)), "segment"))

    if "full_seq" in oracle_input_types:
        for _ in range(full_seq_repeats):
            training_data.append(_make_dp(list(range(num_tokens)), "full_seq"))

    return training_data


# ============================================================
# MAIN ORACLE FUNCTION
# ============================================================

def run_oracle(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    target_prompt: str,
    target_lora_path: str | None,
    oracle_prompt: str,
    oracle_lora_path: str | None,
    segment_start_idx: int = 0,
    segment_end_idx: int | None = None,
    token_start_idx: int = 0,
    token_end_idx: int | None = 1,
    oracle_input_types: list[str] | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    ground_truth: str = "",
    segment_repeats: int = 1,
    full_seq_repeats: int = 1,
    eval_batch_size: int = 32,
    layer_percents: list[int] | None = None,
    injection_layer: int = 1,
    steering_coefficient: float = 1.0,
) -> OracleResults:
    if oracle_input_types is None:
        oracle_input_types = ["segment", "full_seq", "tokens"]
    if generation_kwargs is None:
        generation_kwargs = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 50}
    if layer_percents is None:
        layer_percents = [25, 50, 75]

    dtype = torch.bfloat16
    model_name = model.config._name_or_path

    act_layers = [layer_percent_to_layer(model_name, lp) for lp in layer_percents]
    injection_submodule = get_hf_submodule(model, injection_layer)

    inputs_BL = tokenizer(target_prompt, return_tensors="pt", add_special_tokens=False, padding=True).to(device)

    # Collect activations from target model
    model.enable_adapters()
    if target_lora_path is not None:
        model.set_adapter(target_lora_path)
    submodules = {layer: get_hf_submodule(model, layer) for layer in act_layers}
    acts_by_layer = collect_activations_multiple_layers(
        model=model, submodules=submodules, inputs_BL=inputs_BL,
    )

    seq_len = int(inputs_BL["input_ids"].shape[1])
    attn = inputs_BL["attention_mask"][0]
    real_len = int(attn.sum().item())
    left_pad = seq_len - real_len
    context_input_ids = inputs_BL["input_ids"][0, left_pad:].tolist()

    oracle_inputs = _create_oracle_inputs(
        acts_BLD_by_layer_dict=acts_by_layer,
        context_input_ids=context_input_ids,
        oracle_prompt=oracle_prompt,
        act_layers=act_layers,
        prompt_layers=layer_percents,
        tokenizer=tokenizer,
        segment_start_idx=segment_start_idx,
        segment_end_idx=segment_end_idx,
        token_start_idx=token_start_idx,
        token_end_idx=token_end_idx,
        oracle_input_types=oracle_input_types,
        segment_repeats=segment_repeats,
        full_seq_repeats=full_seq_repeats,
        batch_idx=0,
        left_pad=left_pad,
    )

    responses = _run_evaluation(
        eval_data=oracle_inputs,
        model=model,
        tokenizer=tokenizer,
        submodule=injection_submodule,
        device=device,
        dtype=dtype,
        lora_path=oracle_lora_path,
        eval_batch_size=eval_batch_size,
        steering_coefficient=steering_coefficient,
        generation_kwargs=generation_kwargs,
    )

    token_responses = [None] * len(context_input_ids)
    segment_responses = []
    full_seq_responses = []

    for r in responses:
        meta = r.meta_info
        dp_kind = meta["dp_kind"]
        if dp_kind == "tokens":
            token_responses[int(meta["token_index"])] = r.api_response
        elif dp_kind == "segment":
            segment_responses.append(r.api_response)
        elif dp_kind == "full_seq":
            full_seq_responses.append(r.api_response)

    return OracleResults(
        oracle_lora_path=oracle_lora_path,
        target_lora_path=target_lora_path,
        target_prompt=target_prompt,
        oracle_prompt=oracle_prompt,
        ground_truth=ground_truth,
        num_tokens=len(context_input_ids),
        token_responses=token_responses,
        full_sequence_responses=full_seq_responses,
        segment_responses=segment_responses,
        target_input_ids=context_input_ids,
    )


def visualize_token_selection(tokenizer, input_text, segment_start=0, segment_end=None):
    input_ids = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    num_tokens = len(input_ids)
    end_pos = num_tokens if segment_end is None else segment_end

    print("Token selection visualization:")
    print("-" * 60)
    for i, token_id in enumerate(input_ids):
        token_str = tokenizer.decode([token_id]).replace("\n", "\\n").replace("\r", "\\r")
        marker = ">>>" if segment_start <= i < end_pos else "   "
        print(f"  [{i:3d}] {marker} {token_str}")
    print("-" * 60)
    print(f"Selected positions: {segment_start} to {end_pos} ({end_pos - segment_start} tokens)")


def load_lora_adapter(model, lora_path):
    sanitized = lora_path.replace(".", "_")
    if sanitized not in model.peft_config:
        print(f"Loading LoRA: {lora_path}")
        model.load_adapter(lora_path, adapter_name=sanitized, is_trainable=False, low_cpu_mem_usage=True)
    return sanitized


# %% Model Loading
print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

model_name = "Qwen/Qwen3-8B"
device = torch.device("cuda")
dtype = torch.bfloat16
torch.set_grad_enabled(False)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

print(f"Loading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
if not tokenizer.pad_token_id:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"Loading model: {model_name} with 8-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config,
    torch_dtype=dtype,
)
model.eval()

dummy_config = LoraConfig()
model.add_adapter(dummy_config, adapter_name="default")
print("Model loaded successfully!")


# %% Test 1: Multi-layer oracle on base model (Socrates example)
print("\n" + "=" * 60)
print("TEST 1: Multi-layer oracle - Multi-hop reasoning")
print("=" * 60)

oracle_lora_path = "adamkarvonen/checkpoints_500k_pl_31k_spqav2_199k_sqav3_126k_cls"
layer_percents = [25, 50, 75]

generation_kwargs = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 50,
}

oracle_prompt = "Can you name which person the model is thinking about?"

target_prompt_dict = [
    {"role": "user", "content": "The philosopher who drank hemlock taught a student who founded an academy. That student's most famous pupil was"},
]

formatted_target_prompt = tokenizer.apply_chat_template(
    target_prompt_dict, tokenize=False, add_generation_prompt=False, enable_thinking=False
)

load_lora_adapter(model, oracle_lora_path)

print(f"Oracle prompt: {oracle_prompt}")
print(f"Layer percents: {layer_percents}")
print("Running multi-layer oracle...")

results = run_oracle(
    model=model,
    tokenizer=tokenizer,
    device=device,
    target_prompt=formatted_target_prompt,
    target_lora_path=None,
    oracle_prompt=oracle_prompt,
    oracle_lora_path=oracle_lora_path,
    generation_kwargs=generation_kwargs,
    token_end_idx=None,
    oracle_input_types=["tokens"],
    layer_percents=layer_percents,
)

tokenized = tokenizer(formatted_target_prompt, return_tensors="pt", add_special_tokens=False).to(device)
print(f"\nToken-by-token responses:")
for i in range(tokenized["input_ids"].shape[1]):
    response = results.token_responses[i]
    token_str = tokenizer.decode(tokenized["input_ids"][0, i])
    token_display = token_str.replace("\n", "\\n").replace("\r", "\\r")
    print(f"Token: {token_display:<20} Response: {response}")


# %% Test 2: Multi-layer oracle with segment + full_seq
print("\n" + "=" * 60)
print("TEST 2: Multi-layer oracle - Code understanding (segment + full_seq)")
print("=" * 60)

oracle_prompt = "What will the result be?"

target_prompt_dict = [
    {"role": "user", "content": "def foo(x, y):\n    return x + y\n\nresult = foo(3, 4)"},
]

formatted_target_prompt = tokenizer.apply_chat_template(
    target_prompt_dict, tokenize=False, add_generation_prompt=False, enable_thinking=False
)

segment_start = 16
segment_end = 23

visualize_token_selection(tokenizer, formatted_target_prompt, segment_start, segment_end)

print(f"\nOracle prompt: {oracle_prompt}")
print("Running multi-layer oracle...")

results = run_oracle(
    model=model,
    tokenizer=tokenizer,
    device=device,
    target_prompt=formatted_target_prompt,
    target_lora_path=None,
    oracle_prompt=oracle_prompt,
    oracle_lora_path=oracle_lora_path,
    generation_kwargs=generation_kwargs,
    oracle_input_types=["full_seq", "segment"],
    segment_start_idx=segment_start,
    segment_end_idx=segment_end,
    layer_percents=layer_percents,
)

print(f"\nOracle Response (Full Sequence Input): {results.full_sequence_responses[0]}")
print(f"Oracle Response (Selected Segment Input): {results.segment_responses[0]}")


# %% Test 3: Secret word extraction with target LoRA
print("\n" + "=" * 60)
print("TEST 3: Multi-layer oracle - Secret word extraction")
print("=" * 60)

secret_word = "smile"
target_lora_path = f"adamkarvonen/Qwen3-8B-taboo-{secret_word}_50_mix"

target_prompt = "Please state the secret word."
oracle_prompt = "What is the secret word"

target_prompt_dict = [{"role": "user", "content": target_prompt}]
formatted_target_prompt = tokenizer.apply_chat_template(
    target_prompt_dict, tokenize=False, add_generation_prompt=True, enable_thinking=False
)

load_lora_adapter(model, target_lora_path)

segment_start = 9
segment_end = 18

visualize_token_selection(tokenizer, formatted_target_prompt, segment_start, segment_end)

print(f"\nOracle prompt: {oracle_prompt}")
print(f"Secret word: {secret_word}")
print("Running multi-layer oracle...")

results = run_oracle(
    model=model,
    tokenizer=tokenizer,
    device=device,
    target_prompt=formatted_target_prompt,
    target_lora_path=target_lora_path,
    oracle_prompt=oracle_prompt,
    oracle_lora_path=oracle_lora_path,
    segment_start_idx=segment_start,
    segment_end_idx=segment_end,
    ground_truth=secret_word,
    layer_percents=layer_percents,
)

print(f"\nOracle Response: {results.segment_responses[0]}")
print(f"Ground truth: {results.ground_truth}")


# %% Summary
print("\n" + "=" * 60)
print("ALL TESTS COMPLETED SUCCESSFULLY")
print("=" * 60)
