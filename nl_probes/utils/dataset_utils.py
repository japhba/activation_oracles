import contextlib
from typing import Any, Mapping

import torch
from peft import PeftModel
from pydantic import BaseModel, ConfigDict, model_validator
from transformers import AutoModelForCausalLM, AutoTokenizer

from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule

SPECIAL_TOKEN = " ?"


def get_introspection_prefix(layers: list[int], num_positions: int) -> str:
    assert len(layers) > 0, "layers must be non-empty"
    prefix = ""
    for layer in layers:
        prefix += f"Layer: {layer}\n"
        prefix += SPECIAL_TOKEN * num_positions
        prefix += " \n"
    return prefix


class FeatureResult(BaseModel):
    """Result for a single feature evaluation."""

    feature_idx: int
    api_response: str
    prompt: str
    meta_info: Mapping[str, Any] = {}


class BinaryFeatureResult(BaseModel):
    """First-token candidate score result for a single feature evaluation."""

    feature_idx: int
    candidate_scores: dict[str, float]
    candidate_token_scores: dict[str, list[dict[str, Any]]]
    argmax_token_id: int
    argmax_token_text: str
    argmax_logit: float
    prompt: str
    meta_info: Mapping[str, Any] = {}


class EvalStepResult(BaseModel):
    """Results from a single evaluation step."""

    step: int
    results: list[FeatureResult]


class TrainingDataPoint(BaseModel):
    """Training data point with tensors.
    If steering_vectors is None, then we calculate the steering vectors on the fly
    from the context_input_ids and context_positions."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    datapoint_type: str
    input_ids: list[int]
    labels: list[int]  # Can contain -100 for ignored tokens
    layers: list[int]
    steering_vectors: torch.Tensor | None
    positions: list[int]
    feature_idx: int
    target_output: str
    context_input_ids: list[int] | None
    context_positions: list[int] | None
    ds_label: str | None  # label from the dataset
    meta_info: Mapping[str, Any] = {}

    @model_validator(mode="after")
    def _check_context_alignment(cls, values):
        layers = values.layers
        if len(layers) == 0:
            raise ValueError("layers must be non-empty")
        if len(layers) != len(set(layers)):
            raise ValueError(f"layers must be unique, got {layers}")

        sv = values.steering_vectors
        if sv is not None:
            if len(values.positions) != sv.shape[0]:
                raise ValueError("positions and steering_vectors must have the same length")
            if sv.shape[0] % len(layers) != 0:
                raise ValueError("steering_vectors length must be divisible by number of layers")
            if values.context_positions is not None:
                if len(values.context_positions) * len(layers) != sv.shape[0]:
                    raise ValueError("context_positions length does not match steering_vectors length")
        else:
            if values.context_positions is None or values.context_input_ids is None:
                raise ValueError("context_* must be provided when steering_vectors is None")
            if len(values.positions) % len(layers) != 0:
                raise ValueError("positions length must be divisible by number of layers")
            if len(values.positions) != len(values.context_positions) * len(layers):
                raise ValueError("positions length must match num_layers * num_context_positions")
        return values


class BatchData(BaseModel):
    """Batch of training data with tensors."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    steering_vectors: list[torch.Tensor]
    positions: list[list[int]]
    feature_indices: list[int]


def construct_batch(
    training_data: list[TrainingDataPoint],
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> BatchData:
    max_length = 0
    for data_point in training_data:
        max_length = max(max_length, len(data_point.input_ids))

    batch_tokens = []
    batch_labels = []
    batch_attn_masks = []
    batch_positions = []
    batch_steering_vectors = []
    batch_feature_indices = []

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

        if data_point.steering_vectors is not None:
            steering_vectors = data_point.steering_vectors.to(device)
        else:
            steering_vectors = None

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


def get_prompt_tokens_only(
    training_data_point: TrainingDataPoint,
) -> TrainingDataPoint:
    """User prompt should be labeled as -100"""
    prompt_tokens = []
    prompt_labels = []

    response_token_seen = False
    for i in range(len(training_data_point.input_ids)):
        if training_data_point.labels[i] != -100:
            response_token_seen = True
            continue
        else:
            if response_token_seen:
                raise ValueError("Response token seen before prompt tokens")
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
    *,
    manage_model_state: bool = True,
) -> list[TrainingDataPoint]:
    """
    Materialization of missing steering vectors for a heterogenous batch
    where different items can request activations from different layers.

    Steps:
      1) Find items with steering_vectors=None.
      2) Build a left-padded batch from their context_input_ids.
      3) Register hooks for all unique requested layers and run exactly one forward pass.
      4) For each item, take activations at its requested layer and its context_positions,
         then write back a [num_positions, D] tensor to dp.steering_vectors. Returns a new batch.

    No-op if every item already has steering_vectors.

    `manage_model_state=True` is the safe default and should be used for normal
    standalone calls, including evaluation. In that mode this function switches
    the model to eval mode, disables LoRA adapters while collecting base-model
    activations, and restores the previous train/eval state afterwards.

    `manage_model_state=False` is only for callers that have already wrapped a
    larger outer scope in `model.eval()` and `model.disable_adapter()`. This is
    used in the efficient training block path to avoid paying repeated
    adapter/mode-toggle overhead for every materialization group. It should not
    be used unless the caller is already managing that state correctly.
    """
    to_fill: list[tuple[int, TrainingDataPoint]] = [
        (i, dp) for i, dp in enumerate(batch_points) if dp.steering_vectors is None
    ]
    if not to_fill:
        return batch_points

    assert isinstance(model, PeftModel), "Model must be a PeftModel"

    state_context = contextlib.nullcontext()
    was_training = model.training
    if manage_model_state:
        model.eval()
        state_context = model.disable_adapter()
    else:
        assert not model.training, "manage_model_state=False requires model to already be in eval mode"

    try:
        with state_context:
            # Validate context fields
            for _, dp in to_fill:
                if dp.context_input_ids is None or dp.context_positions is None:
                    raise ValueError(
                        "Datapoint has steering_vectors=None but is missing context_input_ids or context_positions"
                    )

            # Build the input batch (left padding to match your construct_batch convention)
            pad_id = tokenizer.pad_token_id
            contexts: list[list[int]] = [list(dp.context_input_ids) for _, dp in to_fill]
            positions_per_item: list[list[int]] = [list(dp.context_positions) for _, dp in to_fill]
            max_len = max(len(c) for c in contexts)

            input_ids_tensors: list[torch.Tensor] = []
            attn_masks_tensors: list[torch.Tensor] = []
            left_offsets: list[int] = []

            device = next(model.parameters()).device

            for c in contexts:
                pad_len = max_len - len(c)
                input_ids_tensors.append(torch.tensor([pad_id] * pad_len + c, dtype=torch.long, device=device))
                attn_masks_tensors.append(
                    torch.tensor([False] * pad_len + [True] * len(c), dtype=torch.bool, device=device)
                )
                left_offsets.append(pad_len)

            inputs_BL = {
                "input_ids": torch.stack(input_ids_tensors, dim=0),
                "attention_mask": torch.stack(attn_masks_tensors, dim=0),
            }

            layers_needed = sorted({layer for _, dp in to_fill for layer in dp.layers})
            submodules = {layer: get_hf_submodule(model, layer, use_lora=True) for layer in layers_needed}

            acts_by_layer = collect_activations_multiple_layers(
                model=model,
                submodules=submodules,
                inputs_BL=inputs_BL,
                min_offset=None,
                max_offset=None,
            )

            new_batch: list[TrainingDataPoint] = list(batch_points)
            for b in range(len(to_fill)):
                idx, dp = to_fill[b]
                idxs = [p + left_offsets[b] for p in positions_per_item[b]]
                any_layer = dp.layers[0]
                acts_BLD_any = acts_by_layer[any_layer]
                L = acts_BLD_any.shape[1]
                if any(i < 0 or i >= L for i in idxs):
                    raise IndexError(f"Activation index out of range for item {b}: {idxs} with L={L}")

                layer_vectors = []
                for layer in dp.layers:
                    acts_BLD = acts_by_layer[layer]
                    vectors_layer = acts_BLD[b, idxs, :].detach().contiguous()
                    assert len(vectors_layer.shape) == 2, (
                        f"Expected 2D tensor, got vectors_layer.shape={vectors_layer.shape}"
                    )
                    layer_vectors.append(vectors_layer)

                vectors = torch.cat(layer_vectors, dim=0)

                assert len(vectors.shape) == 2, f"Expected 2D tensor, got vectors.shape={vectors.shape}"
                assert vectors.shape[0] == len(dp.positions), "steering_vectors length must match positions length"

                dp_new = dp.model_copy(deep=True)
                dp_new.steering_vectors = vectors

                new_batch[idx] = dp_new

            return new_batch
    finally:
        if manage_model_state and was_training:
            model.train()


def find_pattern_in_tokens(
    token_ids: list[int],
    special_token_str: str,
    layers: list[int],
    num_positions: int,
    tokenizer: AutoTokenizer,
) -> list[int]:
    start_idx = 0
    end_idx = len(token_ids)
    special_token_id = tokenizer.encode(special_token_str, add_special_tokens=False)
    assert len(special_token_id) == 1, f"Expected single token, got {len(special_token_id)}"
    special_token_id = special_token_id[0]
    positions = []
    num_layers = len(layers)
    assert num_layers > 0, "layers must be non-empty"

    for i in range(start_idx, end_idx):
        if len(positions) == num_layers * num_positions:
            break
        if token_ids[i] == special_token_id:
            positions.append(i)

    assert len(positions) == num_layers * num_positions, (
        f"Expected {num_layers * num_positions} positions, got {len(positions)}"
    )
    assert positions == sorted(positions), "Positions are not sorted"

    for layer_idx in range(num_layers):
        block = positions[layer_idx * num_positions : (layer_idx + 1) * num_positions]
        assert block[-1] - block[0] == num_positions - 1, f"Positions are not consecutive: {block}"
        final_pos = block[-1] + 1
        final_tokens = token_ids[final_pos : final_pos + 2]
        final_str = tokenizer.decode(final_tokens, skip_special_tokens=False)
        assert "\n" in final_str, f"Expected newline in {final_str}"

    return positions


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
    assert prefix not in prompt, f"Prefix {prefix} found in prompt {prompt}"
    prompt = prefix + prompt
    input_messages = [{"role": "user", "content": prompt}]

    input_prompt_ids = tokenizer.apply_chat_template(
        input_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )
    if not isinstance(input_prompt_ids, list):
        raise TypeError("Expected list of token ids from tokenizer")

    full_messages = input_messages + [{"role": "assistant", "content": target_response}]

    full_prompt_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )
    if not isinstance(full_prompt_ids, list):
        raise TypeError("Expected list of token ids from tokenizer")

    assistant_start_idx = len(input_prompt_ids)

    labels = full_prompt_ids.copy()
    for i in range(assistant_start_idx):
        labels[i] = -100

    positions = find_pattern_in_tokens(full_prompt_ids, SPECIAL_TOKEN, layers, num_positions, tokenizer)

    if acts_BD is None:
        assert context_input_ids is not None and context_positions is not None, (
            "acts_BD is None but context_input_ids and context_positions are None"
        )
        assert len(context_positions) == num_positions, (
            f"context_positions length {len(context_positions)} does not match num_positions {num_positions}"
        )
    else:
        assert len(acts_BD.shape) == 2, f"Expected 2D tensor, got {acts_BD.shape}"
        acts_BD = acts_BD.cpu().clone().detach()
        assert len(positions) == acts_BD.shape[0], f"Expected {acts_BD.shape[0]} positions, got {len(positions)}"

    training_data_point = TrainingDataPoint(
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

    return training_data_point


def assert_eval_datapoint_layers(dp: TrainingDataPoint, expected_layers: list[int]) -> None:
    assert dp.layers == expected_layers, f"Expected layers {expected_layers}, got {dp.layers}"
    if dp.context_positions is not None:
        assert len(dp.positions) == len(dp.context_positions) * len(dp.layers)
    if dp.steering_vectors is not None:
        assert dp.steering_vectors.shape[0] == len(dp.positions)


def assert_eval_datapoint_layers_in_combinations(
    dp: TrainingDataPoint, expected_layer_combinations: list[list[int]]
) -> None:
    assert expected_layer_combinations, "expected_layer_combinations must be non-empty"
    assert dp.layers in expected_layer_combinations, (
        f"Expected layers in one of {expected_layer_combinations}, got {dp.layers}"
    )
    if dp.context_positions is not None:
        assert len(dp.positions) == len(dp.context_positions) * len(dp.layers)
    if dp.steering_vectors is not None:
        assert dp.steering_vectors.shape[0] == len(dp.positions)
