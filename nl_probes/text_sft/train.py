import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from statistics import median
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from nl_probes.utils.activation_utils import get_text_only_lora_targets
from nl_probes.utils.common import load_model, load_tokenizer, set_seed

CONFIG_FILENAME = "text_sft_config.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class TokenizedExample:
    input_ids: list[int]
    labels: list[int]
    sequence_length: int
    target_token_count: int


@dataclass(frozen=True)
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True)
class KLEntry:
    input_ids: list[int]
    labels: list[int]
    sequence_length: int


@dataclass
class TextSFTConfig:
    model_name: str
    dataset_path: str
    save_dir: str
    wandb_project: str
    wandb_run_name: str
    global_train_batch_size: int
    num_epochs: int
    lr: float
    schema_version: int = SCHEMA_VERSION
    loss_on: Literal["last_assistant", "all_assistant"] = "last_assistant"
    max_train_examples: int | None = None
    max_seq_len: int = 4096
    max_steps: int | None = None
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False
    window_mult: int | None = 20
    use_lora: bool = True
    load_lora_path: str | None = None
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"
    load_in_8bit: bool = False
    save_steps: int = 9_999_999
    log_steps: int = 1
    debug_num_examples_to_dump: int = 2
    seed: int = 42
    kl_loss_weight: float = 0.0
    kl_every_n_steps: int = 100
    kl_batch_size: int = 2
    kl_data_path: str | None = None

    def validate(self) -> None:
        assert self.schema_version == SCHEMA_VERSION, (
            f"Unsupported schema_version={self.schema_version}; expected {SCHEMA_VERSION}"
        )
        assert Path(self.dataset_path).exists(), f"Dataset file does not exist: {self.dataset_path}"
        assert self.global_train_batch_size > 0, "global_train_batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.lr > 0.0, "lr must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"
        assert self.warmup_ratio >= 0.0, "warmup_ratio must be non-negative"
        assert self.max_grad_norm > 0.0, "max_grad_norm must be positive"
        assert self.lora_r > 0, "lora_r must be positive"
        assert self.lora_alpha > 0, "lora_alpha must be positive"
        assert self.lora_dropout >= 0.0, "lora_dropout must be non-negative"
        assert self.save_steps > 0, "save_steps must be positive"
        assert self.log_steps > 0, "log_steps must be positive"
        if self.max_steps is not None:
            assert self.max_steps > 0, "max_steps must be positive"
        if self.max_train_examples is not None:
            assert self.max_train_examples > 0, "max_train_examples must be positive"
        if self.window_mult is not None:
            assert self.window_mult > 0, "window_mult must be positive"
        if self.load_in_8bit:
            assert self.use_lora or self.load_lora_path is not None, (
                "8-bit loading only supports LoRA training in this script"
            )
        if self.kl_loss_weight > 0:
            assert self.kl_data_path is not None, "kl_data_path required when kl_loss_weight > 0"
            assert Path(self.kl_data_path).exists(), f"KL data file does not exist: {self.kl_data_path}"
            assert self.kl_every_n_steps > 0, "kl_every_n_steps must be positive"
            assert self.kl_batch_size > 0, "kl_batch_size must be positive"
            assert self.use_lora or self.load_lora_path is not None, (
                "KL regularization requires LoRA (uses disable_adapter() for base model logits)"
            )


def read_config(path: str | Path) -> TextSFTConfig:
    cfg = TextSFTConfig(**json.loads(Path(path).read_text()))
    cfg.validate()
    return cfg


def write_config(save_dir: str | Path, cfg, filename: str = CONFIG_FILENAME) -> None:
    save_path = Path(save_dir) / filename
    save_path.write_text(json.dumps(asdict(cfg), indent=2))


def print_trainable_parameters(model: torch.nn.Module) -> None:
    total = 0
    trainable = 0
    lora_trainable = 0
    for name, parameter in model.named_parameters():
        param_count = parameter.numel()
        total += param_count
        if parameter.requires_grad:
            trainable += param_count
            if "lora_" in name:
                lora_trainable += param_count
    pct = 100 * trainable / total
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
    if lora_trainable > 0:
        print(f"  LoRA trainable subset: {lora_trainable:,}")


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def apply_chat_template_ids(
    tokenizer: AutoTokenizer,
    messages: tuple[ChatMessage, ...],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool = False,
) -> list[int]:
    payload = [{"role": message.role, "content": message.content} for message in messages]
    token_ids = tokenizer.apply_chat_template(
        payload,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors=None,
        padding=False,
        enable_thinking=enable_thinking,
    )
    assert isinstance(token_ids, list), f"Expected list[int] token IDs, got {type(token_ids)}"
    return token_ids


def render_chat_template(
    tokenizer: AutoTokenizer,
    messages: tuple[ChatMessage, ...],
    *,
    enable_thinking: bool = False,
) -> str:
    payload = [{"role": message.role, "content": message.content} for message in messages]
    rendered = tokenizer.apply_chat_template(
        payload,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    assert isinstance(rendered, str), f"Expected rendered chat string, got {type(rendered)}"
    return rendered


# ---------------------------------------------------------------------------
# Data loading and tokenization
# ---------------------------------------------------------------------------

def load_conversations(dataset_path: str) -> list[tuple[ChatMessage, ...]]:
    """Load chat conversations from a JSON file.

    Expected format:
    [
      {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]},
      ...
    ]

    Each conversation must end with an assistant message.
    """
    data = json.loads(Path(dataset_path).read_text())
    assert isinstance(data, list), f"Expected a JSON list, got {type(data)}"
    conversations: list[tuple[ChatMessage, ...]] = []
    for idx, entry in enumerate(data):
        assert "messages" in entry, f"Entry {idx} missing 'messages' key"
        messages = tuple(
            ChatMessage(role=m["role"], content=m["content"]) for m in entry["messages"]
        )
        assert len(messages) >= 2, f"Entry {idx} needs at least 2 messages, got {len(messages)}"
        assert messages[-1].role == "assistant", (
            f"Entry {idx}: last message must be role='assistant', got '{messages[-1].role}'"
        )
        conversations.append(messages)
    assert conversations, f"No conversations found in {dataset_path}"
    return conversations


def trim_messages_to_fit(
    messages: tuple[ChatMessage, ...],
    tokenizer: AutoTokenizer,
    max_seq_len: int,
) -> tuple[tuple[ChatMessage, ...], list[int], list[int]] | None:
    """Trim earlier context messages to fit within max_seq_len.

    Keeps at least the last 4 messages (minimum for a single-turn with context).
    Returns (trimmed_messages, prefix_ids, full_ids) or None if even the minimum
    messages exceed max_seq_len.

    prefix_ids covers all messages except the last (with add_generation_prompt=True).
    full_ids covers all messages (with add_generation_prompt=False).
    Loss is computed on full_ids[len(prefix_ids):].
    """
    all_messages = list(messages)
    has_system = all_messages[0].role == "system"
    context_start_idx = 1 if has_system else 0
    min_start_idx = max(context_start_idx, len(all_messages) - 4)
    start_idx = context_start_idx

    while True:
        if has_system:
            candidate_messages = tuple([all_messages[0]] + all_messages[start_idx:])
        else:
            candidate_messages = tuple(all_messages[start_idx:])

        prefix_messages = candidate_messages[:-1]
        prefix_ids = apply_chat_template_ids(tokenizer, prefix_messages, add_generation_prompt=True)
        full_ids = apply_chat_template_ids(tokenizer, candidate_messages, add_generation_prompt=False)
        assert full_ids[: len(prefix_ids)] == prefix_ids, (
            f"Prefix tokenization mismatch: prefix_len={len(prefix_ids)} full_len={len(full_ids)}"
        )
        if len(full_ids) <= max_seq_len:
            return candidate_messages, prefix_ids, full_ids
        if start_idx == min_start_idx:
            return None
        start_idx += 1


def compute_all_assistant_labels(
    messages: tuple[ChatMessage, ...],
    tokenizer: AutoTokenizer,
    full_ids: list[int],
) -> list[int]:
    """Compute labels with loss on ALL assistant turns, not just the last one."""
    labels = [-100] * len(full_ids)

    for i, msg in enumerate(messages):
        if msg.role != "assistant":
            continue
        prefix_messages = messages[:i]
        prefix_ids = apply_chat_template_ids(tokenizer, prefix_messages, add_generation_prompt=True)
        through_this = messages[: i + 1]
        through_ids = apply_chat_template_ids(tokenizer, through_this, add_generation_prompt=False)
        assert through_ids[: len(prefix_ids)] == prefix_ids, (
            f"Prefix mismatch for assistant turn {i}"
        )
        for j in range(len(prefix_ids), len(through_ids)):
            labels[j] = full_ids[j]

    return labels


def write_debug_example(
    save_dir: str,
    messages: tuple[ChatMessage, ...],
    full_ids: list[int],
    labels: list[int],
    tokenizer: AutoTokenizer,
    debug_index: int,
) -> None:
    debug_dir = Path(save_dir) / "prepare_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"{debug_index:02d}.txt"

    lines: list[str] = []
    lines.append("MESSAGES")
    lines.append("--------")
    for idx, message in enumerate(messages):
        lines.append(f"[{idx}] role={message.role}")
        lines.append(message.content)
        lines.append("")

    lines.append("RENDERED CHAT TEMPLATE")
    lines.append("---------------------")
    lines.append(render_chat_template(tokenizer, messages))
    lines.append("")
    lines.append("TOKEN TABLE")
    lines.append("-----------")
    for token_idx, token_id in enumerate(full_ids):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        loss_flag = 0 if labels[token_idx] == -100 else 1
        lines.append(
            f"{token_idx:05d} loss={loss_flag} token_id={token_id:>8} token={token_text!r}"
        )

    path.write_text("\n".join(lines))
    print(f"Saved debug tokenization example to {path}")


def tokenize_conversations(
    conversations: list[tuple[ChatMessage, ...]],
    tokenizer: AutoTokenizer,
    cfg: TextSFTConfig,
    *,
    write_debug: bool = True,
) -> list[TokenizedExample]:
    examples: list[TokenizedExample] = []
    dropped_too_long = 0
    debug_written = 0
    for conv in conversations:
        trimmed = trim_messages_to_fit(conv, tokenizer, cfg.max_seq_len)
        if trimmed is None:
            dropped_too_long += 1
            continue

        trimmed_messages, prefix_ids, full_ids = trimmed

        if cfg.loss_on == "last_assistant":
            labels = full_ids.copy()
            for idx in range(len(prefix_ids)):
                labels[idx] = -100
        else:
            labels = compute_all_assistant_labels(trimmed_messages, tokenizer, full_ids)

        target_token_count = sum(1 for label in labels if label != -100)
        assert target_token_count > 0, "No supervised tokens after masking"

        examples.append(TokenizedExample(
            input_ids=full_ids,
            labels=labels,
            sequence_length=len(full_ids),
            target_token_count=target_token_count,
        ))

        if write_debug and debug_written < cfg.debug_num_examples_to_dump:
            write_debug_example(
                cfg.save_dir, trimmed_messages, full_ids, labels, tokenizer, debug_written,
            )
            debug_written += 1

    assert examples, "All conversations were dropped during tokenization/truncation"
    lengths = [e.sequence_length for e in examples]
    print(
        f"Tokenized {len(examples)} examples (dropped {dropped_too_long} too long), "
        f"seq lengths: min={min(lengths)}, median={median(lengths):.0f}, max={max(lengths)}"
    )
    return examples


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def construct_batch(
    examples: list,
    pad_token_id: int,
    device: torch.device,
) -> Batch:
    max_length = max(example.sequence_length for example in examples)
    batch_input_ids: list[torch.Tensor] = []
    batch_labels: list[torch.Tensor] = []
    batch_attention_masks: list[torch.Tensor] = []

    for example in examples:
        padding_length = max_length - example.sequence_length
        input_ids = torch.tensor([pad_token_id] * padding_length + example.input_ids, dtype=torch.long, device=device)
        labels = torch.tensor([-100] * padding_length + example.labels, dtype=torch.long, device=device)
        attention_mask = torch.ones(max_length, dtype=torch.bool, device=device)
        attention_mask[:padding_length] = False
        batch_input_ids.append(input_ids)
        batch_labels.append(labels)
        batch_attention_masks.append(attention_mask)

    return Batch(
        input_ids=torch.stack(batch_input_ids),
        attention_mask=torch.stack(batch_attention_masks),
        labels=torch.stack(batch_labels),
    )


def length_grouped_reorder(
    data: list,
    batch_size: int,
    window_mult: int,
) -> list:
    lengths = [example.sequence_length for example in data]
    indices = list(range(len(data)))
    megabatch_size = batch_size * window_mult
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(indices), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda idx: lengths[idx], reverse=True) for megabatch in megabatches]
    ordered_indices = [idx for megabatch in megabatches for idx in megabatch]
    return [data[idx] for idx in ordered_indices]


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model(
    cfg,
    dtype: torch.dtype,
    local_rank: int,
) -> AutoModelForCausalLM:
    model_kwargs: dict = {
        "device_map": {"": f"cuda:{local_rank}"},
    }
    if cfg.load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=dtype,
        )

    model = load_model(cfg.model_name, dtype, **model_kwargs)
    if cfg.load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    if cfg.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    if cfg.use_lora and cfg.load_lora_path is None:
        target_modules = cfg.lora_target_modules
        vlm_targets = get_text_only_lora_targets(cfg.model_name)
        if cfg.lora_target_modules == "all-linear" and vlm_targets is not None:
            print(f"VLM detected ({cfg.model_name}); excluding vision tower from LoRA")
            target_modules = vlm_targets
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config, autocast_adapter_dtype=True)
    elif cfg.load_lora_path is not None:
        model = PeftModel.from_pretrained(
            model,
            cfg.load_lora_path,
            is_trainable=True,
            autocast_adapter_dtype=True,
        )

    if cfg.gradient_checkpointing or cfg.use_lora or cfg.load_lora_path is not None:
        model.enable_input_require_grads()

    model.config.use_cache = False
    if isinstance(model, PeftModel):
        model.print_trainable_parameters()
    else:
        print_trainable_parameters(model)
    return model


def save_checkpoint(
    cfg,
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    label: str,
) -> None:
    output_dir = Path(cfg.save_dir) / label
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    config_path = output_dir / "training_config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2))


# ---------------------------------------------------------------------------
# KL regularization
# ---------------------------------------------------------------------------

def _load_kl_data(path: str) -> list[KLEntry]:
    payload = torch.load(path, weights_only=False)
    entries = [
        KLEntry(
            input_ids=e["input_ids"],
            labels=[-100] * e["sequence_length"],
            sequence_length=e["sequence_length"],
        )
        for e in payload["entries"]
    ]
    assert entries, f"KL data file has no entries: {path}"
    return entries


def _kl_data_iterator(entries: list[KLEntry], rank: int, seed: int):
    rng = random.Random(seed + rank)
    pool = list(entries)
    while True:
        rng.shuffle(pool)
        yield from pool


def kl_regularization_step(
    ddp_model: torch.nn.parallel.DistributedDataParallel,
    kl_iter,
    n_accum_steps: int,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
    kl_loss_weight: float,
) -> float:
    """Compute KL(finetuned || base) and backprop. Returns mean KL loss value."""
    total_kl = 0.0
    for _ in range(n_accum_steps):
        examples = [next(kl_iter) for _ in range(batch_size)]
        batch = construct_batch(examples, pad_token_id, device)

        with torch.no_grad():
            with ddp_model.module.disable_adapter():
                base_logits = ddp_model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                ).logits.detach()

        ft_logits = ddp_model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        ).logits

        mask = batch.attention_mask.float()

        ft_log_probs = F.log_softmax(ft_logits.float(), dim=-1)
        base_log_probs = F.log_softmax(base_logits.float(), dim=-1)
        ft_probs = ft_log_probs.exp()
        kl_per_pos = (ft_probs * (ft_log_probs - base_log_probs)).sum(dim=-1)
        kl = (kl_per_pos * mask).sum() / mask.sum()

        (kl_loss_weight * kl / n_accum_steps).backward()
        total_kl += kl.item()

        del base_logits, ft_logits

    return total_kl / n_accum_steps


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    cfg,
    examples: list,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    *,
    per_rank_batch_size: int,
    extra_wandb_summary: dict | None = None,
) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    examples = list(examples)
    set_seed(cfg.seed)
    random.Random(cfg.seed).shuffle(examples)

    if cfg.max_train_examples is not None:
        before = len(examples)
        examples = examples[: cfg.max_train_examples]
        if rank == 0:
            print(f"Train example budget trim: {before} -> {len(examples)}")

    usable_global_examples = (len(examples) // cfg.global_train_batch_size) * cfg.global_train_batch_size
    assert usable_global_examples > 0, (
        f"Need at least one full global batch of {cfg.global_train_batch_size} examples, got {len(examples)}"
    )
    if rank == 0 and usable_global_examples != len(examples):
        print(f"Global trim: {len(examples)} -> {usable_global_examples}")
    examples = examples[:usable_global_examples]

    num_examples_pre_shard = len(examples)
    tokens_per_epoch_est = 0
    if rank == 0:
        tokens_per_epoch_est = sum(example.sequence_length for example in examples)

    examples = examples[rank::world_size]
    if cfg.window_mult is not None:
        examples = length_grouped_reorder(examples, per_rank_batch_size, cfg.window_mult)

    num_batches_per_epoch = len(examples) // per_rank_batch_size
    usable_batches_per_epoch = (
        num_batches_per_epoch // cfg.gradient_accumulation_steps
    ) * cfg.gradient_accumulation_steps
    assert usable_batches_per_epoch > 0, (
        "No optimizer steps would run; reduce global_train_batch_size or gradient_accumulation_steps"
    )
    usable_examples = usable_batches_per_epoch * per_rank_batch_size
    if rank == 0 and usable_examples != len(examples):
        print(f"Per-rank trim: {len(examples)} -> {usable_examples}")
    examples = examples[:usable_examples]

    optimizer_steps_per_epoch = usable_batches_per_epoch // cfg.gradient_accumulation_steps
    total_training_steps = optimizer_steps_per_epoch * cfg.num_epochs
    if cfg.max_steps is not None:
        total_training_steps = min(total_training_steps, cfg.max_steps)
    assert total_training_steps > 0, "total_training_steps must be positive"

    model = build_model(cfg, dtype, local_rank)
    torch.cuda.set_device(local_rank)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    ddp_model.train()

    kl_enabled = cfg.kl_loss_weight > 0 and cfg.kl_data_path is not None
    kl_iter = None
    kl_accum_steps = 1
    if kl_enabled:
        kl_data = _load_kl_data(cfg.kl_data_path)
        kl_iter = _kl_data_iterator(kl_data, rank, cfg.seed)
        per_rank_kl_total = cfg.global_train_batch_size // world_size
        assert per_rank_kl_total >= cfg.kl_batch_size, (
            f"kl_batch_size ({cfg.kl_batch_size}) exceeds per-rank batch "
            f"({per_rank_kl_total} = {cfg.global_train_batch_size} / {world_size})"
        )
        assert per_rank_kl_total % cfg.kl_batch_size == 0, (
            f"per-rank batch ({per_rank_kl_total}) must be divisible by "
            f"kl_batch_size ({cfg.kl_batch_size})"
        )
        kl_accum_steps = per_rank_kl_total // cfg.kl_batch_size
        if rank == 0:
            print(
                f"KL regularization: weight={cfg.kl_loss_weight}, "
                f"every {cfg.kl_every_n_steps} steps, "
                f"micro_batch={cfg.kl_batch_size}, accum={kl_accum_steps}, "
                f"data={len(kl_data)} entries"
            )

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=cfg.lr)
    warmup_steps = int(total_training_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    if rank == 0:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=asdict(cfg))
        wandb.summary["train/num_examples_pre_shard"] = num_examples_pre_shard
        wandb.summary["train/tokens_per_epoch_est"] = tokens_per_epoch_est
        wandb.summary["train/optimizer_steps_per_epoch"] = optimizer_steps_per_epoch
        wandb.summary["train/total_training_steps"] = total_training_steps
        if extra_wandb_summary:
            wandb.summary.update(extra_wandb_summary)

    optimizer.zero_grad()
    global_step = 0
    accumulated_loss = 0.0
    training_start = time.perf_counter()

    for epoch in range(cfg.num_epochs):
        if global_step >= total_training_steps:
            break

        progress_bar = tqdm(
            total=usable_batches_per_epoch,
            desc=f"Training epoch {epoch + 1}",
            disable=rank != 0,
        )

        for batch_idx in range(usable_batches_per_epoch):
            if global_step >= total_training_steps:
                break

            start = batch_idx * per_rank_batch_size
            end = start + per_rank_batch_size
            batch_examples = examples[start:end]
            batch = construct_batch(batch_examples, tokenizer.pad_token_id, device)
            loss = ddp_model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
            ).loss
            (loss / cfg.gradient_accumulation_steps).backward()
            accumulated_loss += float(loss.item())
            progress_bar.update(1)

            is_update_step = (batch_idx + 1) % cfg.gradient_accumulation_steps == 0
            if not is_update_step:
                continue

            kl_loss_val = None
            if kl_enabled and (global_step + 1) % cfg.kl_every_n_steps == 0:
                kl_loss_val = kl_regularization_step(
                    ddp_model, kl_iter, kl_accum_steps, cfg.kl_batch_size,
                    tokenizer.pad_token_id, device, cfg.kl_loss_weight,
                )

            clip_grad_norm_(ddp_model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            mean_loss = accumulated_loss / cfg.gradient_accumulation_steps
            if rank == 0 and global_step % cfg.log_steps == 0:
                log_dict = {
                    "train/loss": mean_loss,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                }
                if kl_loss_val is not None:
                    log_dict["train/kl_loss"] = kl_loss_val
                wandb.log(log_dict, step=global_step)
                print(f"step={global_step} loss={mean_loss:.6f}")

            global_step += 1
            accumulated_loss = 0.0

            if global_step % cfg.save_steps == 0 and global_step < total_training_steps:
                if rank == 0:
                    save_checkpoint(cfg, model, tokenizer, f"step_{global_step}")
                dist.barrier()

        progress_bar.close()

    if rank == 0:
        elapsed = time.perf_counter() - training_start
        wandb.summary["train/training_wall_time_sec"] = elapsed
        wandb.summary["train/training_steps_per_sec"] = total_training_steps / elapsed
        save_checkpoint(cfg, model, tokenizer, "final")
        wandb.finish()

    dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic text SFT trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to a TextSFTConfig JSON")
    args = parser.parse_args()

    cfg = read_config(args.config)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    tokenizer = load_tokenizer(cfg.model_name)

    # All ranks tokenize (deterministic with same input, so results are identical)
    conversations = load_conversations(cfg.dataset_path)
    examples = tokenize_conversations(
        conversations, tokenizer, cfg, write_debug=(local_rank == 0),
    )

    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    assert cfg.global_train_batch_size % world_size == 0, (
        f"global_train_batch_size {cfg.global_train_batch_size} must be divisible by world_size {world_size}"
    )
    per_rank_batch_size = cfg.global_train_batch_size // world_size
    print(f"Per-rank batch size: {per_rank_batch_size}, world_size: {world_size}")

    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16
    train_model(
        cfg=cfg,
        examples=examples,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        per_rank_batch_size=per_rank_batch_size,
    )
    dist.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
