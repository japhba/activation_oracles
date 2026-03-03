from typing import Any, Mapping

import torch
from peft import PeftModel
from pydantic import BaseModel, ConfigDict, model_validator
from transformers import AutoModelForCausalLM, AutoTokenizer

from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule

SPECIAL_TOKEN = " ?"


def get_introspection_prefix(sae_layer: int, num_positions: int, layers: list[int] | None = None) -> str:
    prefix_layers = list(layers) if layers else [sae_layer]
    if len(prefix_layers) == 1:
        return f"L{prefix_layers[0]}:" + SPECIAL_TOKEN * num_positions + "\n"
    k, rem = divmod(num_positions, len(prefix_layers))
    if rem:
        raise ValueError(f"num_positions={num_positions} not divisible by layers={prefix_layers}")
    return " ".join(f"L{layer}:" + SPECIAL_TOKEN * k for layer in prefix_layers) + "\n"


class FeatureResult(BaseModel):
    """Result for a single feature evaluation."""

    feature_idx: int
    api_response: str
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
    layer: int
    steering_vectors: torch.Tensor | None
    positions: list[int]
    feature_idx: int
    target_output: str
    context_input_ids: list[int] | None
    context_positions: list[int] | None
    source_positions: list[int] | None = None
    source_total_length: int | None = None
    ds_label: str | None  # label from the dataset
    meta_info: Mapping[str, Any] = {}

    @model_validator(mode="after")
    def _check_context_alignment(cls, values):
        sv = values.steering_vectors
        if sv is not None:
            if len(values.positions) != sv.shape[0]:
                raise ValueError("positions and steering_vectors must have the same length")
            if values.source_positions is not None:
                if len(values.source_positions) != sv.shape[0]:
                    raise ValueError("source_positions and steering_vectors must have the same length")
                if values.source_total_length is None:
                    raise ValueError("source_total_length must be provided when source_positions is set")
            elif values.source_total_length is not None:
                raise ValueError("source_total_length requires source_positions")
        else:
            if values.context_positions is None or values.context_input_ids is None:
                raise ValueError("context_* must be provided when steering_vectors is None")
            if len(values.positions) != len(values.context_positions):
                raise ValueError("positions and context_positions must have the same length")
        return values


class BatchData(BaseModel):
    """Batch of training data with tensors."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    steering_vectors: list[torch.Tensor]
    positions: list[list[int]]
    source_positions: list[list[int] | None]
    source_lengths: list[int | None]
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
    batch_source_positions = []
    batch_source_lengths = []
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
        batch_source_positions.append(data_point.source_positions)
        batch_source_lengths.append(data_point.source_total_length)
        batch_feature_indices.append(data_point.feature_idx)

    return BatchData(
        input_ids=torch.stack(batch_tokens),
        labels=torch.stack(batch_labels),
        attention_mask=torch.stack(batch_attn_masks),
        steering_vectors=batch_steering_vectors,
        positions=batch_positions,
        source_positions=batch_source_positions,
        source_lengths=batch_source_lengths,
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
    """
    # Select datapoints that need generation
    to_fill: list[tuple[int, TrainingDataPoint]] = [
        (i, dp) for i, dp in enumerate(batch_points) if dp.steering_vectors is None
    ]
    if not to_fill:
        return batch_points

    assert isinstance(model, PeftModel), "Model must be a PeftModel"

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
        # For HF, bool masks are fine; your construct_batch uses bool too
        attn_masks_tensors.append(torch.tensor([False] * pad_len + [True] * len(c), dtype=torch.bool, device=device))
        left_offsets.append(pad_len)

    inputs_BL = {
        "input_ids": torch.stack(input_ids_tensors, dim=0),
        "attention_mask": torch.stack(attn_masks_tensors, dim=0),
    }

    # Prepare hooks for all unique requested layers
    layers_needed = sorted({dp.layer for _, dp in to_fill})
    submodules = {layer: get_hf_submodule(model, layer, use_lora=True) for layer in layers_needed}

    # Run a single pass with dropout off, then restore the previous train/eval mode
    was_training = model.training
    model.eval()
    with model.disable_adapter():
        # [layer] -> [B, L, D], where B == len(to_fill)
        acts_by_layer = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL=inputs_BL,
            min_offset=None,
            max_offset=None,
        )
    if was_training:
        model.train()

    # Build the new list, copying only items we change
    new_batch: list[TrainingDataPoint] = list(batch_points)  # references by default
    for b in range(len(to_fill)):
        idx, dp = to_fill[b]
        layer = dp.layer
        acts_BLD = acts_by_layer[layer]  # [B, L, D] on GPU

        idxs = [p + left_offsets[b] for p in positions_per_item[b]]
        # Bounds check for safety
        L = acts_BLD.shape[1]
        if any(i < 0 or i >= L for i in idxs):
            raise IndexError(f"Activation index out of range for item {b}: {idxs} with L={L}")

        vectors = acts_BLD[b, idxs, :].detach().contiguous()

        assert len(vectors.shape) == 2, f"Expected 2D tensor, got vectors.shape={vectors.shape}"

        dp_new = dp.model_copy(deep=True)
        dp_new.steering_vectors = vectors

        new_batch[idx] = dp_new

    return new_batch


def find_pattern_in_tokens(
    token_ids: list[int], special_token_str: str, num_positions: int, tokenizer: AutoTokenizer
) -> list[int]:
    special_token_id = tokenizer.encode(special_token_str, add_special_tokens=False)
    assert len(special_token_id) == 1, f"Expected single token, got {len(special_token_id)}"
    special_token_id = special_token_id[0]
    positions = []

    for i in range(len(token_ids)):
        if len(positions) == num_positions:
            break
        if token_ids[i] == special_token_id:
            positions.append(i)

    assert len(positions) == num_positions, f"Expected {num_positions} positions, got {len(positions)}"
    return positions


def _build_manual_prefix_tokens(
    tokenizer: AutoTokenizer,
    layer: int,
    num_positions: int,
    layers: list[int] | None,
) -> tuple[str, list[int], list[int]]:
    prefix = get_introspection_prefix(layer, num_positions, layers)
    special_token_id = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)
    assert len(special_token_id) == 1, f"SPECIAL_TOKEN should encode to 1 token, got {len(special_token_id)}"
    special_token_id = special_token_id[0]
    prefix_layers = list(layers) if layers else [layer]
    block_sizes = [num_positions]
    if len(prefix_layers) > 1:
        k, rem = divmod(num_positions, len(prefix_layers))
        if rem:
            raise ValueError(f"num_positions={num_positions} not divisible by layers={prefix_layers}")
        block_sizes = [k] * len(prefix_layers)

    prefix_ids: list[int] = []
    positions: list[int] = []
    for i, (layer_idx, block_size) in enumerate(zip(prefix_layers, block_sizes)):
        label = f"L{layer_idx}:"
        if i > 0:
            label = " " + label
        prefix_ids.extend(tokenizer.encode(label, add_special_tokens=False))
        positions.extend(range(len(prefix_ids), len(prefix_ids) + block_size))
        prefix_ids.extend([special_token_id] * block_size)
    prefix_ids.extend(tokenizer.encode("\n", add_special_tokens=False))
    return prefix, prefix_ids, positions


def _tokenize_with_manual_prefix(
    messages: list[dict],
    tokenizer: AutoTokenizer,
    layer: int,
    num_positions: int,
    layers: list[int] | None,
    add_generation_prompt: bool,
) -> tuple[list[int], list[int]]:
    """Tokenize chat messages, manually inserting placeholder token IDs.

    Splits the text around the full prefix region and tokenizes the prefix
    labels separately while inserting the placeholder token IDs directly.
    This prevents the tokenizer from merging placeholder tokens with
    adjacent text at block boundaries.

    Returns (token_ids, placeholder_positions).
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )

    prefix, prefix_ids, rel_positions = _build_manual_prefix_tokens(
        tokenizer=tokenizer,
        layer=layer,
        num_positions=num_positions,
        layers=layers,
    )
    idx = text.find(prefix)
    assert idx >= 0, "Prefix text not found in chat template output"

    before_text = text[:idx]
    after_text = text[idx + len(prefix):]

    before_ids = tokenizer.encode(before_text, add_special_tokens=False)
    after_ids = tokenizer.encode(after_text, add_special_tokens=False)

    all_ids = before_ids + prefix_ids + after_ids
    positions = [len(before_ids) + pos for pos in rel_positions]

    return all_ids, positions


def create_training_datapoint(
    datapoint_type: str,
    prompt: str,
    target_response: str,
    layer: int,
    num_positions: int,
    tokenizer: AutoTokenizer,
    acts_BD: torch.Tensor | None,
    feature_idx: int,
    context_input_ids: list[int] | None = None,
    context_positions: list[int] | None = None,
    source_positions: list[int] | None = None,
    source_total_length: int | None = None,
    ds_label: str | None = None,
    meta_info: Mapping[str, Any] | None = None,
) -> TrainingDataPoint:
    if meta_info is None:
        meta_info = {}
    layers = list(meta_info["layers"]) if "layers" in meta_info else [layer]
    prefix = get_introspection_prefix(layer, num_positions, layers)
    prompt_with_prefix = prefix + prompt
    input_messages = [{"role": "user", "content": prompt_with_prefix}]
    full_messages = input_messages + [{"role": "assistant", "content": target_response}]

    # Manually tokenize: split text around placeholder region and insert
    # special token IDs directly, preventing tokenizer boundary merges.
    input_prompt_ids, positions = _tokenize_with_manual_prefix(
        input_messages, tokenizer, layer, num_positions, layers,
        add_generation_prompt=True,
    )
    full_prompt_ids, _ = _tokenize_with_manual_prefix(
        full_messages, tokenizer, layer, num_positions, layers,
        add_generation_prompt=False,
    )

    assistant_start_idx = len(input_prompt_ids)

    labels = full_prompt_ids.copy()
    for i in range(assistant_start_idx):
        labels[i] = -100

    if acts_BD is None:
        assert context_input_ids is not None and context_positions is not None, (
            "acts_BD is None but context_input_ids and context_positions are None"
        )
    else:
        assert len(acts_BD.shape) == 2, f"Expected 2D tensor, got {acts_BD.shape}"
        acts_BD = acts_BD.cpu().clone().detach()
        assert len(positions) == acts_BD.shape[0], f"Expected {acts_BD.shape[0]} positions, got {len(positions)}"

    training_data_point = TrainingDataPoint(
        input_ids=full_prompt_ids,
        labels=labels,
        layer=layer,
        steering_vectors=acts_BD,
        positions=positions,
        feature_idx=feature_idx,
        target_output=target_response,
        datapoint_type=datapoint_type,
        context_input_ids=context_input_ids,
        context_positions=context_positions,
        source_positions=source_positions,
        source_total_length=source_total_length,
        ds_label=ds_label,
        meta_info=meta_info,
    )

    return training_data_point
