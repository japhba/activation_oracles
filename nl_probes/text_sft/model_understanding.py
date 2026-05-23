"""Text SFT trainer for model-understanding data.

Loads screening/investigation/verification pipeline outputs, assembles
chat conversations, tokenizes them, and fine-tunes a language model using
the generic training infrastructure from text_sft.py.

Data pipeline: screening.json + investigations.json + verification.json
-> joined SFT examples -> tokenized examples -> LoRA training.
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
import random
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from statistics import median
from typing import Literal

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from nl_probes.text_sft import (
    ChatMessage,
    TokenizedExample,
    apply_chat_template_ids,
    build_model,
    construct_batch,
    length_grouped_reorder,
    render_chat_template,
    save_checkpoint,
    train_model,
)
from nl_probes.utils.common import load_tokenizer, set_seed

CONFIG_FILENAME = "model_understanding_sft_config.json"
SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Model-understanding-specific dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScreeningExample:
    prompt_id: str
    interest_score: int
    messages: tuple[ChatMessage, ...]
    completions: tuple[str, ...]
    question: str
    second_person_question: str | None
    behavior_completion_indices: tuple[int, ...] | None


@dataclass(frozen=True)
class InvestigationExample:
    prompt_id: str
    first_person_answer: str


@dataclass(frozen=True)
class VerificationExample:
    prompt_id: str
    score: int


@dataclass(frozen=True)
class SFTExample:
    prompt_id: str
    chosen_completion_index: int
    chosen_completion: str
    question: str
    answer: str
    messages: tuple[ChatMessage, ...]


@dataclass(frozen=True)
class LossSpan:
    start: int
    end: int


@dataclass(frozen=True)
class TokenizedSFTExample:
    prompt_id: str
    chosen_completion_index: int
    sequence_length: int
    target_token_count: int
    input_ids: list[int]
    labels: list[int]
    loss_span: LossSpan


@dataclass(frozen=True)
class PreparationFingerprint:
    run_dirs: tuple[str, ...]
    model_name: str
    min_interest_score: int
    min_verification_score: int
    completion_selection_mode: str
    max_seq_len: int
    seed: int


@dataclass(frozen=True)
class PreparedDatasetMetadata:
    num_screening_results: int
    num_investigations: int
    num_verifications: int
    num_examples: int
    dropped_too_long: int
    dropped_tokenization_error: int
    min_seq_len: int
    median_seq_len: float
    max_seq_len: int


@dataclass(frozen=True)
class PreparedDatasetBundle:
    fingerprint: PreparationFingerprint
    metadata: PreparedDatasetMetadata
    examples: tuple[TokenizedSFTExample, ...]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ModelUnderstandingConfig:
    run_dirs: list[str]
    model_name: str
    prepared_dataset_path: str
    save_dir: str
    wandb_project: str
    wandb_run_name: str
    global_train_batch_size: int
    num_epochs: int
    lr: float
    schema_version: int = SCHEMA_VERSION
    min_interest_score: int = 3
    min_verification_score: int = 7
    completion_selection_mode: Literal["first", "random"] = "first"
    synthetic_data_paths: list[str] | None = None
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
        assert self.run_dirs, "run_dirs must be non-empty"
        for rd in self.run_dirs:
            assert Path(rd).exists(), f"Run directory does not exist: {rd}"
        assert self.global_train_batch_size > 0, "global_train_batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.lr > 0.0, "lr must be positive"
        assert self.min_interest_score > 0, "min_interest_score must be positive"
        assert self.min_verification_score > 0, "min_verification_score must be positive"
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


def read_config(path: str | Path) -> ModelUnderstandingConfig:
    cfg = ModelUnderstandingConfig(**json.loads(Path(path).read_text()))
    cfg.validate()
    return cfg


def write_config(save_dir: str | Path, cfg: ModelUnderstandingConfig) -> None:
    save_path = Path(save_dir) / CONFIG_FILENAME
    save_path.write_text(json.dumps(asdict(cfg), indent=2))


# ---------------------------------------------------------------------------
# Data loading (screening / investigation / verification pipeline)
# ---------------------------------------------------------------------------


def build_preparation_fingerprint(cfg: ModelUnderstandingConfig) -> PreparationFingerprint:
    return PreparationFingerprint(
        run_dirs=tuple(str(Path(rd).resolve()) for rd in cfg.run_dirs),
        model_name=cfg.model_name,
        min_interest_score=cfg.min_interest_score,
        min_verification_score=cfg.min_verification_score,
        completion_selection_mode=cfg.completion_selection_mode,
        max_seq_len=cfg.max_seq_len,
        seed=cfg.seed,
    )


def _run_prefix(run_dir: str, run_dirs: list[str]) -> str:
    """Return a prompt_id prefix to disambiguate across runs."""
    if len(run_dirs) <= 1:
        return ""
    return Path(run_dir).name + "_"


def load_screening_results(cfg: ModelUnderstandingConfig) -> list[ScreeningExample]:
    screening_examples: list[ScreeningExample] = []
    for run_dir in cfg.run_dirs:
        prefix = _run_prefix(run_dir, cfg.run_dirs)
        path = Path(run_dir) / "screening.json"
        payload = json.loads(path.read_text())
        results = payload["results"]
        for result in results:
            if result["interest_score"] < cfg.min_interest_score:
                continue
            if result["messages"] is None:
                messages = (ChatMessage(role="user", content=result["user_message"]),)
            else:
                messages = tuple(ChatMessage(role=message["role"], content=message["content"]) for message in result["messages"])
            if "second_person_question" in result:
                second_person_question = result["second_person_question"]
            else:
                second_person_question = None
            raw_bci = result.get("behavior_completion_indices")
            if isinstance(raw_bci, list) and raw_bci and all(isinstance(x, int) for x in raw_bci):
                behavior_completion_indices = tuple(raw_bci)
            else:
                behavior_completion_indices = None
            screening_examples.append(
                ScreeningExample(
                    prompt_id=prefix + result["prompt_id"],
                    interest_score=result["interest_score"],
                    messages=messages,
                    completions=tuple(result["completions"]),
                    question=result["question"],
                    second_person_question=second_person_question,
                    behavior_completion_indices=behavior_completion_indices,
                )
            )
    assert screening_examples, "No screening examples survived filtering"
    return screening_examples


def load_investigations(cfg: ModelUnderstandingConfig) -> dict[str, InvestigationExample]:
    investigations: dict[str, InvestigationExample] = {}
    skipped = 0
    for run_dir in cfg.run_dirs:
        prefix = _run_prefix(run_dir, cfg.run_dirs)
        path = Path(run_dir) / "investigations.json"
        payload = json.loads(path.read_text())
        for result in payload["results"]:
            findings = result["structured_findings"]
            if findings is None:
                skipped += 1
                continue
            prompt_id = prefix + result["prompt_id"]
            investigations[prompt_id] = InvestigationExample(
                prompt_id=prompt_id,
                first_person_answer=findings["first_person_answer"],
            )
    if skipped:
        print(f"Skipped {skipped} investigations with null structured_findings")
    assert investigations, "No investigations found"
    return investigations


def load_verifications(cfg: ModelUnderstandingConfig) -> dict[str, VerificationExample]:
    verifications: dict[str, VerificationExample] = {}
    for run_dir in cfg.run_dirs:
        prefix = _run_prefix(run_dir, cfg.run_dirs)
        path = Path(run_dir) / "verification.json"
        payload = json.loads(path.read_text())
        for result in payload["results"]:
            prompt_id = prefix + result["prompt_id"]
            verifications[prompt_id] = VerificationExample(
                prompt_id=prompt_id,
                score=result["score"],
            )
    assert verifications, "No verifications found"
    return verifications


def build_sft_examples(
    cfg: ModelUnderstandingConfig,
    screening_examples: list[ScreeningExample],
    investigations: dict[str, InvestigationExample],
    verifications: dict[str, VerificationExample],
) -> list[SFTExample]:
    screening_by_prompt = {example.prompt_id: example for example in screening_examples}
    rng = random.Random(cfg.seed)
    training_examples: list[SFTExample] = []
    skipped = 0
    eligible = 0

    for prompt_id in sorted(investigations):
        if prompt_id not in verifications:
            continue
        verification = verifications[prompt_id]
        if verification.score < cfg.min_verification_score:
            continue
        if prompt_id not in screening_by_prompt:
            skipped += 1
            continue
        eligible += 1

        screening_example = screening_by_prompt[prompt_id]

        # Skip malformed entries
        if not screening_example.behavior_completion_indices:
            skipped += 1
            continue
        if screening_example.second_person_question is None:
            skipped += 1
            continue

        behavior_indices = screening_example.behavior_completion_indices
        if cfg.completion_selection_mode == "first":
            chosen_completion_index = behavior_indices[0] - 1
        else:
            chosen_completion_index = rng.choice(list(behavior_indices)) - 1
        chosen_completion = screening_example.completions[chosen_completion_index]

        messages = list(screening_example.messages)
        messages.append(ChatMessage(role="assistant", content=chosen_completion))
        messages.append(ChatMessage(role="user", content=screening_example.second_person_question))
        messages.append(ChatMessage(role="assistant", content=investigations[prompt_id].first_person_answer))

        training_examples.append(
            SFTExample(
                prompt_id=prompt_id,
                chosen_completion_index=chosen_completion_index,
                chosen_completion=chosen_completion,
                question=screening_example.second_person_question,
                answer=investigations[prompt_id].first_person_answer,
                messages=tuple(messages),
            )
        )

    if skipped:
        skip_rate = skipped / max(eligible, 1)
        print(f"Skipped {skipped} malformed examples ({skip_rate:.1%} of eligible)")
        assert skip_rate < 0.001, (
            f"Too many malformed examples: {skipped}/{eligible} ({skip_rate:.1%}). "
            f"Check screening data for missing fields."
        )

    assert training_examples, "No SFT examples were built after joining screening/investigation/verification"
    return training_examples


# ---------------------------------------------------------------------------
# Synthetic data loading
# ---------------------------------------------------------------------------


def load_synthetic_data(cfg: ModelUnderstandingConfig) -> list[SFTExample]:
    """Load synthetic training data from generate_synthetic_data.py outputs."""
    if not cfg.synthetic_data_paths:
        return []

    synthetic_examples: list[SFTExample] = []
    for path_str in cfg.synthetic_data_paths:
        path = Path(path_str)
        assert path.exists(), f"Synthetic data file not found: {path}"
        data = json.loads(path.read_text())
        results = data["results"]

        for entry in results:
            prompt_id = entry["prompt_id"]
            example_type = entry["example_type"]
            orig_messages = [
                ChatMessage(role=m["role"], content=m["content"])
                for m in entry["messages"]
            ]

            if example_type == "behavior_prediction":
                # The question field contains separator + question (varied per example)
                last_msg = orig_messages[-1]
                assert last_msg.role == "user", (
                    f"Expected last message to be user for {prompt_id}"
                )
                modified_messages = list(orig_messages[:-1])
                modified_messages.append(ChatMessage(
                    role="user",
                    content=last_msg.content + entry["question"],
                ))
                modified_messages.append(ChatMessage(
                    role="assistant",
                    content=entry["answer"],
                ))
                synthetic_examples.append(SFTExample(
                    prompt_id=f"{prompt_id}__bp",
                    chosen_completion_index=-1,
                    chosen_completion="",
                    question=entry["question"],
                    answer=entry["answer"],
                    messages=tuple(modified_messages),
                ))

            elif example_type == "counterfactual_prediction":
                # Original messages + completion + counterfactual question + answer
                messages = list(orig_messages)
                messages.append(ChatMessage(
                    role="assistant",
                    content=entry["chosen_completion"],
                ))
                messages.append(ChatMessage(
                    role="user",
                    content=entry["question"],
                ))
                messages.append(ChatMessage(
                    role="assistant",
                    content=entry["answer"],
                ))
                exp_idx = entry.get("experiment_index", 0)
                synthetic_examples.append(SFTExample(
                    prompt_id=f"{prompt_id}__cf_{exp_idx}",
                    chosen_completion_index=entry["chosen_completion_index"],
                    chosen_completion=entry["chosen_completion"],
                    question=entry["question"],
                    answer=entry["answer"],
                    messages=tuple(messages),
                ))

        print(f"Loaded {len(results)} synthetic examples from {path}")

    return synthetic_examples


# ---------------------------------------------------------------------------
# Tokenization (model-understanding-specific, uses shared trimming logic)
# ---------------------------------------------------------------------------


def trim_messages_to_fit(
    example: SFTExample,
    tokenizer: AutoTokenizer,
    cfg: ModelUnderstandingConfig,
) -> tuple[tuple[ChatMessage, ...], list[int], list[int]] | None:
    all_messages = list(example.messages)
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
            f"Prefix tokenization mismatch for {example.prompt_id}: "
            f"prefix_len={len(prefix_ids)} full_len={len(full_ids)}"
        )
        if len(full_ids) <= cfg.max_seq_len:
            return candidate_messages, prefix_ids, full_ids
        if start_idx == min_start_idx:
            return None
        start_idx += 1


def write_debug_example(
    cfg: ModelUnderstandingConfig,
    example: SFTExample,
    trimmed_messages: tuple[ChatMessage, ...],
    full_ids: list[int],
    labels: list[int],
    tokenizer: AutoTokenizer,
    debug_index: int,
) -> None:
    debug_dir = Path(cfg.save_dir) / "prepare_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"{debug_index:02d}_{example.prompt_id}.txt"

    lines: list[str] = []
    lines.append(f"prompt_id: {example.prompt_id}")
    lines.append(f"chosen_completion_index: {example.chosen_completion_index}")
    lines.append("")
    lines.append("MESSAGES")
    lines.append("--------")
    for idx, message in enumerate(trimmed_messages):
        lines.append(f"[{idx}] role={message.role}")
        lines.append(message.content)
        lines.append("")

    lines.append("RENDERED CHAT TEMPLATE")
    lines.append("---------------------")
    lines.append(render_chat_template(tokenizer, trimmed_messages))
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


def tokenize_sft_examples(
    cfg: ModelUnderstandingConfig,
    examples: list[SFTExample],
    tokenizer: AutoTokenizer,
    *,
    num_screening_results: int,
    num_investigations: int,
    num_verifications: int,
) -> PreparedDatasetBundle:
    tokenized_examples: list[TokenizedSFTExample] = []
    dropped_too_long = 0
    dropped_tokenization_error = 0
    debug_examples_written = 0

    for example in examples:
        try:
            trimmed = trim_messages_to_fit(example, tokenizer, cfg)
        except AssertionError:
            dropped_tokenization_error += 1
            continue
        if trimmed is None:
            dropped_too_long += 1
            continue

        trimmed_messages, prefix_ids, full_ids = trimmed
        labels = full_ids.copy()
        for idx in range(len(prefix_ids)):
            labels[idx] = -100

        target_token_count = len(full_ids) - len(prefix_ids)
        assert target_token_count > 0, f"No supervised tokens for {example.prompt_id}"
        tokenized_example = TokenizedSFTExample(
            prompt_id=example.prompt_id,
            chosen_completion_index=example.chosen_completion_index,
            sequence_length=len(full_ids),
            target_token_count=target_token_count,
            input_ids=full_ids,
            labels=labels,
            loss_span=LossSpan(start=len(prefix_ids), end=len(full_ids)),
        )
        tokenized_examples.append(tokenized_example)

        if debug_examples_written < cfg.debug_num_examples_to_dump:
            write_debug_example(
                cfg=cfg,
                example=example,
                trimmed_messages=trimmed_messages,
                full_ids=full_ids,
                labels=labels,
                tokenizer=tokenizer,
                debug_index=debug_examples_written,
            )
            debug_examples_written += 1

    if dropped_tokenization_error > 0:
        print(f"Skipped {dropped_tokenization_error} examples due to tokenization errors (e.g. think tags in content)")
    max_allowed_errors = max(1, len(examples) // 5000)
    assert dropped_tokenization_error <= max_allowed_errors, (
        f"Too many tokenization errors: {dropped_tokenization_error}/{len(examples)} "
        f"(threshold: {max_allowed_errors})"
    )
    assert tokenized_examples, "All training examples were dropped during tokenization/truncation"
    lengths = [example.sequence_length for example in tokenized_examples]
    metadata = PreparedDatasetMetadata(
        num_screening_results=num_screening_results,
        num_investigations=num_investigations,
        num_verifications=num_verifications,
        num_examples=len(tokenized_examples),
        dropped_too_long=dropped_too_long,
        dropped_tokenization_error=dropped_tokenization_error,
        min_seq_len=min(lengths),
        median_seq_len=float(median(lengths)),
        max_seq_len=max(lengths),
    )
    return PreparedDatasetBundle(
        fingerprint=build_preparation_fingerprint(cfg),
        metadata=metadata,
        examples=tuple(tokenized_examples),
    )


# ---------------------------------------------------------------------------
# Prepared dataset caching
# ---------------------------------------------------------------------------


def prepared_bundle_to_payload(bundle: PreparedDatasetBundle) -> dict:
    return asdict(bundle)


def payload_to_prepared_bundle(payload: dict) -> PreparedDatasetBundle:
    fingerprint = PreparationFingerprint(**payload["fingerprint"])
    metadata = PreparedDatasetMetadata(**payload["metadata"])
    examples = tuple(
        TokenizedSFTExample(
            prompt_id=example["prompt_id"],
            chosen_completion_index=example["chosen_completion_index"],
            sequence_length=example["sequence_length"],
            target_token_count=example["target_token_count"],
            input_ids=example["input_ids"],
            labels=example["labels"],
            loss_span=LossSpan(**example["loss_span"]),
        )
        for example in payload["examples"]
    )
    return PreparedDatasetBundle(
        fingerprint=fingerprint,
        metadata=metadata,
        examples=examples,
    )


def load_prepared_dataset(cfg: ModelUnderstandingConfig) -> PreparedDatasetBundle:
    payload = torch.load(cfg.prepared_dataset_path, weights_only=False)
    assert isinstance(payload, dict), (
        f"Expected plain dict payload from {cfg.prepared_dataset_path}, got {type(payload)}"
    )
    bundle = payload_to_prepared_bundle(payload)
    assert bundle.fingerprint == build_preparation_fingerprint(cfg), (
        f"Prepared dataset fingerprint mismatch for {cfg.prepared_dataset_path}\n"
        f"found={bundle.fingerprint}\n"
        f"expected={build_preparation_fingerprint(cfg)}"
    )
    return bundle


def source_paths_for_run(cfg: ModelUnderstandingConfig) -> list[Path]:
    paths = []
    for run_dir in cfg.run_dirs:
        paths.append(Path(run_dir) / "screening.json")
        paths.append(Path(run_dir) / "investigations.json")
        paths.append(Path(run_dir) / "verification.json")
    if cfg.synthetic_data_paths:
        paths.extend(Path(p) for p in cfg.synthetic_data_paths)
    return paths


def prepare_dataset(cfg: ModelUnderstandingConfig, tokenizer: AutoTokenizer) -> PreparedDatasetBundle:
    screening_examples = load_screening_results(cfg)
    investigations = load_investigations(cfg)
    verifications = load_verifications(cfg)
    sft_examples = build_sft_examples(cfg, screening_examples, investigations, verifications)
    synthetic_examples = load_synthetic_data(cfg)
    if synthetic_examples:
        print(f"Adding {len(synthetic_examples)} synthetic examples to "
              f"{len(sft_examples)} investigation examples")
        sft_examples.extend(synthetic_examples)
    bundle = tokenize_sft_examples(
        cfg,
        sft_examples,
        tokenizer,
        num_screening_results=len(screening_examples),
        num_investigations=len(investigations),
        num_verifications=len(verifications),
    )
    prepared_path = Path(cfg.prepared_dataset_path)
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prepared_bundle_to_payload(bundle), prepared_path)
    print(
        f"Prepared dataset saved to {prepared_path} "
        f"with {bundle.metadata.num_examples} examples "
        f"(dropped_too_long={bundle.metadata.dropped_too_long})"
    )
    return bundle


def ensure_prepared_dataset(cfg: ModelUnderstandingConfig, tokenizer: AutoTokenizer) -> PreparedDatasetBundle:
    prepared_path = Path(cfg.prepared_dataset_path)
    source_paths = source_paths_for_run(cfg)

    if prepared_path.exists():
        source_mtime = max(path.stat().st_mtime for path in source_paths)
        prepared_mtime = prepared_path.stat().st_mtime
        if source_mtime <= prepared_mtime:
            print(f"Prepared dataset already exists and is up to date: {prepared_path}")
            return load_prepared_dataset(cfg)
        print(f"Prepared dataset is older than source JSONs; rebuilding {prepared_path}")

    return prepare_dataset(cfg, tokenizer)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-understanding text SFT trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to a ModelUnderstandingConfig JSON")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build or refresh the prepared tokenized dataset and exit before training",
    )
    args = parser.parse_args()

    cfg = read_config(args.config)
    tokenizer = load_tokenizer(cfg.model_name)

    if args.prepare_only:
        torchrun_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torchrun_local_rank != 0:
            print("Skipping prepare-only work on non-zero LOCAL_RANK")
            raise SystemExit(0)
        ensure_prepared_dataset(cfg, tokenizer)
        print("Prepared dataset is ready; exiting due to --prepare-only")
        raise SystemExit(0)

    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    assert cfg.global_train_batch_size % world_size == 0, (
        f"global_train_batch_size {cfg.global_train_batch_size} must be divisible by world_size {world_size}"
    )
    per_rank_batch_size = cfg.global_train_batch_size // world_size
    print(f"Per-rank batch size: {per_rank_batch_size}, world_size: {world_size}")

    if local_rank == 0:
        ensure_prepared_dataset(cfg, tokenizer)
    dist.barrier()
    bundle = load_prepared_dataset(cfg)

    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16

    # TokenizedSFTExample is duck-type compatible with text_sft's training loop
    # (has input_ids, labels, sequence_length)
    train_model(
        cfg=cfg,
        examples=list(bundle.examples),
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        per_rank_batch_size=per_rank_batch_size,
        extra_wandb_summary={
            "data/num_screening_results": bundle.metadata.num_screening_results,
            "data/num_investigations": bundle.metadata.num_investigations,
            "data/num_verifications": bundle.metadata.num_verifications,
            "data/num_examples": bundle.metadata.num_examples,
            "data/dropped_too_long": bundle.metadata.dropped_too_long,
        },
    )
    dist.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
