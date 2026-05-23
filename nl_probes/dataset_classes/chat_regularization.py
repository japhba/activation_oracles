import json
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, model_validator
from transformers import PreTrainedTokenizer


class ChatRegularizationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    total_entries: int
    generated_at: str
    chat_dataset: str
    max_total_tokens: int
    max_prompt_tokens: int
    temperature: float
    seed: int
    thinking_fraction: float


class ChatRegularizationEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    prompt_len: int
    response_len: int
    total_len: int
    enable_thinking: bool
    finish_reason: str

    @model_validator(mode="after")
    def _check_lengths(self):
        if len(self.prompt_token_ids) != self.prompt_len:
            raise ValueError("prompt_len does not match prompt_token_ids")
        if len(self.response_token_ids) != self.response_len:
            raise ValueError("response_len does not match response_token_ids")
        if self.prompt_len + self.response_len != self.total_len:
            raise ValueError("total_len does not match prompt_len + response_len")
        if self.response_len == 0:
            raise ValueError("response_token_ids must be non-empty")
        return self


class ChatRegularizationFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: ChatRegularizationMetadata
    entries: list[ChatRegularizationEntry]

    @model_validator(mode="after")
    def _check_entry_count(self):
        if len(self.entries) != self.metadata.total_entries:
            raise ValueError("metadata.total_entries does not match entry count")
        return self


class ChatRegularizationDataPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    input_ids: list[int]
    labels: list[int]
    prompt_len: int
    response_len: int
    total_len: int
    enable_thinking: bool
    finish_reason: str

    @model_validator(mode="after")
    def _check_lengths(self):
        if len(self.input_ids) != len(self.labels):
            raise ValueError("input_ids and labels must have the same length")
        if self.prompt_len + self.response_len != self.total_len:
            raise ValueError("total_len does not match prompt_len + response_len")
        if len(self.input_ids) != self.total_len:
            raise ValueError("input_ids length does not match total_len")
        if any(label != -100 for label in self.labels[: self.prompt_len]):
            raise ValueError("Prompt tokens must be masked with -100")
        if self.labels[self.prompt_len :] != self.input_ids[self.prompt_len :]:
            raise ValueError("Response labels must equal response token ids")
        return self


class ChatRegularizationBatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    example_ids: list[str]


def load_chat_regularization_data(
    dataset_path: str | Path,
    *,
    expected_model_name: str,
) -> list[ChatRegularizationDataPoint]:
    path = Path(dataset_path)
    payload = ChatRegularizationFile(**json.loads(path.read_text()))

    if payload.metadata.model != expected_model_name:
        raise ValueError(
            f"Chat regularization dataset model mismatch: "
            f"expected {expected_model_name}, got {payload.metadata.model}"
        )

    data: list[ChatRegularizationDataPoint] = []
    for entry in payload.entries:
        input_ids = entry.prompt_token_ids + entry.response_token_ids
        labels = ([-100] * entry.prompt_len) + entry.response_token_ids
        data.append(
            ChatRegularizationDataPoint(
                example_id=entry.id,
                input_ids=input_ids,
                labels=labels,
                prompt_len=entry.prompt_len,
                response_len=entry.response_len,
                total_len=entry.total_len,
                enable_thinking=entry.enable_thinking,
                finish_reason=entry.finish_reason,
            )
        )

    return data


def construct_chat_regularization_batch(
    training_data: list[ChatRegularizationDataPoint],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
) -> ChatRegularizationBatch:
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id for chat regularization batching")

    max_length = max(len(data_point.input_ids) for data_point in training_data)

    batch_tokens = []
    batch_labels = []
    batch_attn_masks = []
    batch_example_ids = []

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
        batch_example_ids.append(data_point.example_id)

    return ChatRegularizationBatch(
        input_ids=torch.stack(batch_tokens),
        labels=torch.stack(batch_labels),
        attention_mask=torch.stack(batch_attn_masks),
        example_ids=batch_example_ids,
    )
