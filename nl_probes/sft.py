import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Pin each torchrun rank to its own GPU BEFORE importing torch / unsloth.
# Unsloth (and many CUDA-init libs) snapshot the visible-device list at import
# time, so torch.cuda.set_device(local_rank) called later is too late and every
# rank ends up loading on cuda:0. Once CUDA_VISIBLE_DEVICES is narrowed to the
# rank's GPU, torch sees a single device and downstream code becomes trivial.
if "LOCAL_RANK" in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]

# Unsloth must be imported before transformers/peft to apply its kernel patches.
# Gated on AO_USE_UNSLOTH=1 so existing HF+PEFT configs keep working unchanged.
if os.environ.get("AO_USE_UNSLOTH") == "1":
    import unsloth  # noqa: F401

import argparse
import gc
import json
import random
import shutil
import time
from datetime import timedelta

# All necessary imports are now included above
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model


def _local_cuda_index(local_rank: int) -> int:
    """Return the CUDA device index visible to this process.

    If we narrowed CUDA_VISIBLE_DEVICES to a single GPU at module load (the
    Unsloth-friendly path), torch only sees one device and the correct index
    is 0. Otherwise (full DDP launch with all GPUs visible per rank), use
    `local_rank` directly.
    """
    return 0 if torch.cuda.device_count() == 1 else local_rank
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
import torch.distributed as dist
import wandb

from nl_probes.utils.steering_hooks import (
    add_hook,
    get_hf_activation_steering_hook,
)
from nl_probes.configs.sft_config import (
    TRAINING_CONFIG_FILENAME,
    SelfInterpTrainingConfig,
    write_training_config,
)
from nl_probes.dataset_classes.act_dataset_manager import ActDatasetLoader
from nl_probes.dataset_classes.chat_regularization import (
    ChatRegularizationBatch,
    ChatRegularizationDataPoint,
    construct_chat_regularization_batch,
    load_chat_regularization_data,
)
from nl_probes.utils.activation_utils import get_hf_submodule, get_text_only_lora_targets
from nl_probes.utils.activation_collection import (
    materialize_block_into_batches,
    materialize_training_block,
)
from nl_probes.utils.common import load_model, load_tokenizer, set_seed
from nl_probes.utils.dataset_utils import (
    BatchData,
    EvalStepResult,
    FeatureResult,
    TrainingDataPoint,
    construct_batch,
    materialize_missing_steering_vectors,
)
from huggingface_hub import repo_exists, upload_file


def resolve_lora_source(path_or_repo: str) -> str | Path:
    local_path = Path(path_or_repo)
    if local_path.exists():
        return local_path
    assert repo_exists(path_or_repo), f"LoRA source not found locally or on Hugging Face Hub: {path_or_repo}"
    return path_or_repo


def push_lora_to_hf(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    repo_id: str,
    private: bool,
    training_config_path: str | Path,
    commit_message: str = "Upload LoRA adapter after training",
) -> None:
    """
    Push the trained LoRA adapter to Hugging Face Hub.

    Args:
        model: The trained model with LoRA adapters
        tokenizer: The tokenizer used with the model
        repo_id: HuggingFace repository ID (e.g., "username/repo-name")
        commit_message: Commit message for the upload
        private: Whether to make the repository private
        training_config_path: Path to self-interp training config for this adapter

    Returns:
        None
    """

    print(f"Pushing LoRA adapter to Hugging Face Hub: {repo_id}")
    training_config_path = Path(training_config_path)
    assert training_config_path.exists(), f"Missing training config: {training_config_path}"

    # Get the original model name to copy config from
    original_model_name = model.config._name_or_path
    if hasattr(model, "base_model"):
        # For LoRA models, get the base model name
        original_model_name = model.base_model.config._name_or_path

    # Push the model (LoRA adapters)
    model.push_to_hub(
        repo_id=repo_id,
        commit_message=commit_message,
        private=private,
    )

    # Push the tokenizer as well
    tokenizer.push_to_hub(
        repo_id=repo_id,
        commit_message=f"Upload tokenizer - {commit_message}",
        private=private,
    )

    upload_file(
        repo_id=repo_id,
        path_or_fileobj=str(training_config_path),
        path_in_repo=TRAINING_CONFIG_FILENAME,
        commit_message="Add training config",
    )

    # Copy config.json from the original model
    try:
        import tempfile

        from huggingface_hub import hf_hub_download

        print(f"Copying config.json from original model: {original_model_name}")

        # Download config.json from the original model
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".json", delete=False) as tmp_file:
            config_path = hf_hub_download(
                repo_id=original_model_name,
                filename="config.json",
                cache_dir=None,
                force_download=False,
            )

            # Copy the file content
            with open(config_path, "rb") as src:
                tmp_file.write(src.read())
            tmp_file.flush()

            # Upload to the LoRA repo
            upload_file(
                path_or_fileobj=tmp_file.name,
                path_in_repo="config.json",
                repo_id=repo_id,
                commit_message=f"Copy config.json from {original_model_name}",
            )

        # Clean up temp file
        os.unlink(tmp_file.name)
        print(f"Successfully copied config.json from {original_model_name}")

    except Exception as e:
        print(f"Warning: Failed to copy config.json from original model: {e}")
        print("LoRA adapter uploaded successfully, but without original model config")

    # Create and upload README with base model metadata
    try:
        print("Creating README with base model metadata...")

        readme_content = f"""---
base_model: {original_model_name}
library_name: peft
---

# LoRA Adapter for SAE Introspection

This is a LoRA (Low-Rank Adaptation) adapter trained for SAE (Sparse Autoencoder) introspection tasks.

## Base Model
- **Base Model**: `{original_model_name}`
- **Adapter Type**: LoRA
- **Task**: SAE Feature Introspection

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("{original_model_name}")
tokenizer = AutoTokenizer.from_pretrained("{original_model_name}")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{repo_id}")
```

## Training Details
This adapter was trained using the lightweight SAE introspection training script to help the model understand and explain SAE features through activation steering.
"""

        # Create temporary README file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp_readme:
            tmp_readme.write(readme_content)
            tmp_readme.flush()

            # Upload README to the LoRA repo
            upload_file(
                path_or_fileobj=tmp_readme.name,
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message="Add README with base model metadata",
            )

        # Clean up temp file
        os.unlink(tmp_readme.name)
        print("Successfully uploaded README with base model metadata")

    except Exception as e:
        print(f"Warning: Failed to upload README: {e}")
        print("LoRA adapter uploaded successfully, but without README")

    print(f"Successfully pushed LoRA adapter to: https://huggingface.co/{repo_id}")


def train_features_batch(
    cfg: SelfInterpTrainingConfig,
    training_batch: BatchData,
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Trains the model on a single batch of data.
    """

    batch_steering_vectors = training_batch.steering_vectors
    batch_positions = training_batch.positions

    # 3. Create and apply the activation steering hook
    hook_fn = get_hf_activation_steering_hook(
        vectors=batch_steering_vectors,
        positions=batch_positions,
        steering_coefficient=cfg.steering_coefficient,
        device=device,
        dtype=dtype,
    )

    tokenized_input = {
        "input_ids": training_batch.input_ids,
        "attention_mask": training_batch.attention_mask,
    }

    with add_hook(submodule, hook_fn):
        loss = model(**tokenized_input, labels=training_batch.labels).loss

    return loss


def train_chat_regularization_batch(
    training_batch: ChatRegularizationBatch,
    model: AutoModelForCausalLM,
) -> torch.Tensor:
    tokenized_input = {
        "input_ids": training_batch.input_ids,
        "attention_mask": training_batch.attention_mask,
    }
    return model(**tokenized_input, labels=training_batch.labels).loss


def run_open_ended_eval(
    cfg: SelfInterpTrainingConfig,
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    global_step: int,
    phase: str,
    delete_temp_adapter: bool = False,
) -> None:
    """Run all open-ended evals using the current training adapter.

    Saves the current LoRA adapter + training config to a temp directory, then
    passes the path to run_all_evals — same codepath as standalone eval.
    """
    from nl_probes.open_ended_eval.run_all import run_all_evals

    assert isinstance(model, PeftModel), "Open-ended eval requires a PEFT training model"
    active_adapters = list(model.active_adapters)
    assert len(active_adapters) == 1, f"Expected one active adapter, found {active_adapters}"
    training_adapter_name = active_adapters[0]
    model_was_training = model.training

    output_dir = str(Path(cfg.save_dir) / "open_ended_eval" / f"{phase}_step_{global_step}")

    # Save current adapter to disk so eval uses the same codepath as standalone
    eval_adapter_dir = Path(cfg.save_dir) / "open_ended_eval" / f"_tmp_adapter_step_{global_step}"
    model.save_pretrained(eval_adapter_dir)
    write_training_config(eval_adapter_dir, cfg)

    torch.cuda.empty_cache()
    gc.collect()

    all_summaries = run_all_evals(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=cfg.model_name,
        output_dir=output_dir,
        verbalizer_lora_paths=[str(eval_adapter_dir)],
        include=cfg.open_ended_eval_include,
        max_entries=cfg.open_ended_eval_max_entries,
    )

    # This function always saves a temporary copy of the adapter for eval, even when
    # the same adapter was already saved elsewhere (e.g. final/ or step_N/). The temp
    # copy is only needed during run_all_evals above. Deleting it saves ~700M-2.7G per
    # eval. Be careful: if this codepath changes to reuse the temp adapter later, this
    # will silently break.
    if delete_temp_adapter:
        shutil.rmtree(eval_adapter_dir)

    # Log metrics to wandb matching the groups in experiments/plot_eval_grouped.py.
    # Evals with sub-modes (mmlu_prediction, sycophancy) log per-mode metrics
    # rather than just the averaged overall_metrics.
    wandb_metrics: dict[str, float] = {}

    def _extract_metric(src: dict, key: str) -> float | None:
        v = src.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    def _first_verbalizer_metrics(summary: dict) -> dict:
        mbv = summary.get("metrics_by_verbalizer", {})
        return next(iter(mbv.values()), {}) if mbv else {}

    for eval_name, summary in all_summaries.items():
        mode_results = summary.get("mode_results", {})

        if eval_name == "mmlu_prediction":
            # Log pre_answer and post_answer roc_auc separately
            for mode in ("pre_answer", "post_answer"):
                mode_summary = mode_results.get(mode, {})
                sources = [mode_summary.get("overall_metrics", {}), _first_verbalizer_metrics(mode_summary)]
                for key in ("roc_auc", "accuracy_at_zero"):
                    for src in sources:
                        v = _extract_metric(src, key)
                        if v is not None:
                            wandb_metrics[f"open_ended/mmlu_prediction/{mode}/{key}"] = v
                            break
                # Letter prediction (pre_answer only)
                if mode == "pre_answer":
                    lp_by_verb = mode_summary.get("letter_prediction_by_verbalizer", {})
                    if lp_by_verb:
                        first_verb_metrics = next(iter(lp_by_verb.values()), {})
                        v = _extract_metric(first_verb_metrics, "matches_model_rate")
                        if v is not None:
                            wandb_metrics["open_ended/mmlu_prediction/pre_answer/letter_prediction/matches_model_rate"] = v

        elif eval_name == "sycophancy":
            # Log no_cot and cot roc_auc separately
            for mode in ("no_cot", "cot"):
                mode_summary = mode_results.get(mode, {})
                sources = [mode_summary.get("overall_metrics", {}), _first_verbalizer_metrics(mode_summary)]
                for key in ("roc_auc", "accuracy_at_zero"):
                    for src in sources:
                        v = _extract_metric(src, key)
                        if v is not None:
                            wandb_metrics[f"open_ended/sycophancy/{mode}/{key}"] = v
                            break

        elif eval_name == "missing_info":
            sources = [summary.get("overall_metrics", {}), _first_verbalizer_metrics(summary)]
            for key in ("roc_auc", "accuracy_at_zero", "A_vs_C_roc_auc"):
                for src in sources:
                    v = _extract_metric(src, key)
                    if v is not None:
                        wandb_metrics[f"open_ended/missing_info/{key}"] = v
                        break

        elif eval_name in ("taboo", "personaqa"):
            sources = [summary.get("overall_metrics", {}), _first_verbalizer_metrics(summary)]
            for key in ("full_seq_accuracy", "single_token_accuracy"):
                for src in sources:
                    v = _extract_metric(src, key)
                    if v is not None:
                        wandb_metrics[f"open_ended/{eval_name}/{key}"] = v
                        break

        else:
            # backtracking, number_prediction, system_prompt_qa_hidden/latentqa
            SIMPLE_METRICS: dict[str, list[str]] = {
                "number_prediction": ["matches_model_answer_rate"],
                "backtracking": ["mean_specificity", "mean_correctness"],
                "system_prompt_qa_hidden": ["mean_specificity", "mean_correctness"],
                "system_prompt_qa_latentqa": ["mean_specificity", "mean_correctness"],
            }
            headline_keys = SIMPLE_METRICS.get(eval_name, [])
            sources = [summary.get("overall_metrics", {}), _first_verbalizer_metrics(summary)]
            for key in headline_keys:
                for src in sources:
                    v = _extract_metric(src, key)
                    if v is not None:
                        wandb_metrics[f"open_ended/{eval_name}/{key}"] = v
                        break

    wandb.log(wandb_metrics, step=global_step)
    wandb.summary.update(wandb_metrics)

    # Restore training adapter and model state
    model.set_adapter(training_adapter_name)
    if model_was_training:
        model.train()
    else:
        model.eval()
    torch.cuda.empty_cache()
    gc.collect()


def run_periodic_training_actions(
    *,
    cfg: SelfInterpTrainingConfig,
    model: PeftModel,
    submodule: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
    rank: int,
    component_validation_data: dict[str, list[TrainingDataPoint]],
) -> None:
    if (
        cfg.validation_steps is not None
        and global_step % cfg.validation_steps == 0
        and (cfg.validation_on_start or global_step > 0)
    ):
        if rank == 0:
            run_validation_loss_eval(
                cfg=cfg,
                component_validation_data=component_validation_data,
                model=model,
                submodule=submodule,
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
                global_step=global_step,
            )
            from nl_probes.utils.logprob_mcq import run_logprob_mcq_eval

            # MCQ eval at the same cadence as validation loss. Generation-free,
            # single forward per candidate, scores correct vs distractor.
            run_logprob_mcq_eval(
                cfg=cfg,
                component_validation_data=component_validation_data,
                model=model,
                submodule=submodule,
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
                global_step=global_step,
                rng=random.Random(cfg.seed + 11_000 + global_step),
            )
        dist.barrier()

    if global_step % cfg.eval_steps == 0 and (cfg.eval_on_start or global_step > 0):
        if rank == 0:
            run_open_ended_eval(
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                device=device,
                global_step=global_step,
                phase="periodic",
                delete_temp_adapter=True,
            )
        dist.barrier()

    if global_step % cfg.save_steps == 0 and global_step > 0:
        if rank == 0:
            step_dir = Path(cfg.save_dir) / f"step_{global_step}"
            model.save_pretrained(step_dir)
            write_training_config(step_dir, cfg)
            if cfg.hf_push_to_hub and cfg.hf_repo_id:
                print("Pushing LoRA adapter to Hugging Face Hub...")
                push_lora_to_hf(
                    model=model,
                    tokenizer=tokenizer,
                    repo_id=cfg.hf_repo_id + f"-step-{global_step}",
                    private=cfg.hf_private_repo,
                    training_config_path=step_dir / TRAINING_CONFIG_FILENAME,
                    commit_message=(
                        f"SAE introspection LoRA - {cfg.wandb_run_name} - step {global_step}"
                    ),
                )
                print("Pushed LoRA adapter to Hugging Face Hub.")
        dist.barrier()


@dataclass
class CyclingChatRegularizationSource:
    data: list[ChatRegularizationDataPoint]
    batch_size: int
    window_mult: int | None
    rng: random.Random
    cursor: int = 0

    def __post_init__(self) -> None:
        if len(self.data) < self.batch_size:
            raise ValueError(
                f"Need at least one full chat regularization batch, got {len(self.data)} examples "
                f"for batch_size={self.batch_size}"
            )

    def _reshuffle(self) -> None:
        self.rng.shuffle(self.data)
        if self.window_mult is not None:
            self.data = length_grouped_reorder(self.data, self.batch_size, self.window_mult)
        self.cursor = 0

    def next_batch(self) -> list[ChatRegularizationDataPoint]:
        if self.cursor + self.batch_size > len(self.data):
            self._reshuffle()
        batch = self.data[self.cursor : self.cursor + self.batch_size]
        if len(batch) != self.batch_size:
            raise ValueError("Expected a full chat regularization batch")
        self.cursor += self.batch_size
        return batch


@dataclass
class TrainingComponent:
    component_name: str
    training_data: list[TrainingDataPoint]


@dataclass
class ValidationLossResult:
    component_name: str
    loss: float
    num_examples: int
    num_target_tokens: int


def _count_target_tokens(labels: torch.Tensor) -> int:
    return int((labels != -100).sum().item())


@torch.no_grad()
def run_validation_loss_eval(
    *,
    cfg: SelfInterpTrainingConfig,
    component_validation_data: dict[str, list[TrainingDataPoint]],
    model: PeftModel,
    submodule: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
) -> None:
    if not component_validation_data:
        return

    model_was_training = model.training
    model.eval()

    try:
        all_results: list[ValidationLossResult] = []
        for component_name, eval_data in component_validation_data.items():
            if len(eval_data) == 0:
                raise ValueError(f"Validation component {component_name} has no datapoints")

            weighted_loss_sum = 0.0
            total_target_tokens = 0

            for start in range(0, len(eval_data), cfg.train_batch_size):
                eval_batch_points = eval_data[start : start + cfg.train_batch_size]
                eval_batch_points = materialize_missing_steering_vectors(eval_batch_points, tokenizer, model)
                eval_data[start : start + len(eval_batch_points)] = eval_batch_points

                eval_batch = construct_batch(eval_batch_points, tokenizer, device)
                batch_loss = train_features_batch(cfg, eval_batch, model, submodule, device, dtype)
                batch_target_tokens = _count_target_tokens(eval_batch.labels)
                if batch_target_tokens == 0:
                    raise ValueError(f"Validation batch for {component_name} has no supervised tokens")

                weighted_loss_sum += float(batch_loss.item()) * batch_target_tokens
                total_target_tokens += batch_target_tokens

            if total_target_tokens == 0:
                raise ValueError(f"Validation component {component_name} has no supervised tokens")

            all_results.append(
                ValidationLossResult(
                    component_name=component_name,
                    loss=weighted_loss_sum / total_target_tokens,
                    num_examples=len(eval_data),
                    num_target_tokens=total_target_tokens,
                )
            )

        wandb_metrics: dict[str, float] = {}
        total_weighted_loss = 0.0
        total_target_tokens = 0
        total_component_loss = 0.0
        for result in all_results:
            wandb_metrics[f"validation_loss/{result.component_name}"] = result.loss
            total_weighted_loss += result.loss * result.num_target_tokens
            total_target_tokens += result.num_target_tokens
            total_component_loss += result.loss

        wandb_metrics["validation_loss/overall_token_weighted"] = total_weighted_loss / total_target_tokens
        wandb_metrics["validation_loss/overall_component_mean"] = total_component_loss / len(all_results)
        wandb.log(wandb_metrics, step=global_step)
        wandb.summary.update(wandb_metrics)
    finally:
        if model_was_training:
            model.train()


def oom_preflight_check(
    cfg: SelfInterpTrainingConfig,
    training_data: list[TrainingDataPoint],
    chat_regularization_data: list[ChatRegularizationDataPoint] | None,
    model: PeftModel,
    submodule: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    # Hardest activation-collection chunk we expect to see during training.
    # The runtime path sorts by context length, then materializes fixed
    # `train_batch_size` chunks, so the worst single materialization call is the
    # longest context repeated across one train batch.
    hardest_materialization_template: list[TrainingDataPoint] = []
    materialization_candidates = [dp for dp in training_data if dp.steering_vectors is None]
    if materialization_candidates:
        hardest_materialization_point = max(
            materialization_candidates,
            key=lambda dp: (len(dp.context_input_ids), len(dp.input_ids)),
        )
        hardest_materialization_template = [
            hardest_materialization_point.model_copy(deep=True) for _ in range(cfg.train_batch_size)
        ]

    # Largest train batch shape we could construct after materialization by
    # repeating the longest prompt in the dataset.
    longest_prompt = max(training_data, key=lambda x: len(x.input_ids))
    largest_train_batch_points = [longest_prompt.model_copy(deep=True) for _ in range(cfg.train_batch_size)]
    largest_train_batch_points = materialize_missing_steering_vectors(largest_train_batch_points, tokenizer, model)
    largest_possible_batch = construct_batch(largest_train_batch_points, tokenizer, device)
    largest_chat_batch: ChatRegularizationBatch | None = None
    if chat_regularization_data:
        longest_chat_example = max(chat_regularization_data, key=lambda x: len(x.input_ids))
        largest_chat_batch_points = [longest_chat_example.model_copy(deep=True) for _ in range(cfg.train_batch_size)]
        largest_chat_batch = construct_chat_regularization_batch(largest_chat_batch_points, tokenizer, device)

    dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
    for _ in tqdm(range(3), desc="OOM preflight check"):
        # Stress the worst combined pattern we care about:
        # 1) hardest activation collection
        # 2) largest train batch forward/backward
        # Repeating this loop is enough to cover the "training has already
        # started" case, so we only need one materialization call per iteration.
        if hardest_materialization_template:
            hardest_materialization_points = [dp.model_copy(deep=True) for dp in hardest_materialization_template]
            materialize_training_block(cfg, hardest_materialization_points, tokenizer, model)

        loss = train_features_batch(cfg, largest_possible_batch, model, submodule, device, dtype)
        loss.backward()
        dummy_optimizer.step()
        dummy_optimizer.zero_grad()

        if largest_chat_batch is not None:
            chat_loss = train_chat_regularization_batch(largest_chat_batch, model)
            chat_loss.backward()
            dummy_optimizer.step()
            dummy_optimizer.zero_grad()

    del dummy_optimizer
    del largest_possible_batch
    del largest_train_batch_points
    del hardest_materialization_template
    del largest_chat_batch
    torch.cuda.empty_cache()
    gc.collect()

    print("OOM preflight check complete")


def train_model(
    cfg: SelfInterpTrainingConfig,
    training_data: list[TrainingDataPoint],
    component_validation_data: dict[str, list[TrainingDataPoint]],
    chat_regularization_data: list[ChatRegularizationDataPoint] | None,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    model_kwargs: dict[str, Any],
    verbose: bool = False,
):
    # Distributed settings (always on; launch with torchrun, even on 1 GPU)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Pin the current CUDA device BEFORE loading the model. With
    # CUDA_VISIBLE_DEVICES narrowed at the top of this module the device is
    # always cuda:0 from torch's view; otherwise fall back to local_rank.
    cuda_idx = _local_cuda_index(local_rank)
    torch.cuda.set_device(cuda_idx)

    # Ensure loads happen on this GPU only (important for quantized models)
    model_kwargs = {
        **model_kwargs,
        "device_map": {"": f"cuda:{cuda_idx}"},
    }

    set_seed(cfg.seed)

    if cfg.use_unsloth:
        assert os.environ.get("AO_USE_UNSLOTH") == "1", (
            "cfg.use_unsloth=True requires AO_USE_UNSLOTH=1 to be set in env "
            "before sft.py imports (so unsloth gets imported before transformers/peft)."
        )
        from unsloth import FastLanguageModel

        # Unsloth manages dtype, device placement, gradient checkpointing internally.
        # It expects to load on the current local device — model_kwargs.device_map
        # would conflict.
        model, _unsloth_tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model_name,
            max_seq_length=cfg.unsloth_max_seq_length,
            dtype=None,
            load_in_4bit=False,
            load_in_8bit=False,
        )
        model.enable_input_require_grads()
    else:
        model = load_model(cfg.model_name, dtype, **model_kwargs)
        model.enable_input_require_grads()

        if cfg.gradient_checkpointing:
            model.use_cache = False
            # use_reentrant=True is required for in-place steering hooks on
            # transformers 5.x (see AGENTS.md: gradient checkpointing note).
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": True}
            )

    submodule = get_hf_submodule(model, cfg.hook_onto_layer)

    if cfg.use_lora and cfg.load_lora_path is None:
        target_modules = cfg.lora_target_modules
        vlm_targets = get_text_only_lora_targets(cfg.model_name)
        if vlm_targets and target_modules == "all-linear":
            print(f"VLM detected ({cfg.model_name}): excluding vision tower from LoRA")
            target_modules = vlm_targets

        if cfg.use_unsloth:
            from unsloth import FastLanguageModel

            # Unsloth's get_peft_model needs an explicit module list — "all-linear"
            # is not understood. For Qwen-style models the standard 7-target list
            # matches "all-linear" semantics for attn+MLP projections.
            unsloth_targets = (
                target_modules
                if isinstance(target_modules, list)
                else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            gc_arg = "unsloth" if cfg.gradient_checkpointing else False
            model = FastLanguageModel.get_peft_model(
                model,
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=unsloth_targets,
                bias="none",
                use_gradient_checkpointing=gc_arg,
                random_state=cfg.seed,
                use_rslora=bool(getattr(cfg, "use_rslora", False)),
            )
        else:
            lora_config = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
                use_rslora=bool(getattr(cfg, "use_rslora", False)),
            )
            model = get_peft_model(model, lora_config, autocast_adapter_dtype=True)
    elif cfg.load_lora_path is not None:
        lora_source = resolve_lora_source(cfg.load_lora_path)
        model = PeftModel.from_pretrained(model, lora_source, is_trainable=True, autocast_adapter_dtype=True)

    assert isinstance(model, PeftModel), "On-the-fly steering vector materialization requires a PEFT model"
    model.print_trainable_parameters()

    # Wrap with DDP for training, but keep the PEFT model reference for hooks/eval
    torch.cuda.set_device(cuda_idx)
    train_model_module: torch.nn.Module = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[cuda_idx], output_device=cuda_idx, find_unused_parameters=False
    )

    train_model_module.train()

    oom_preflight_check(
        cfg,
        training_data,
        chat_regularization_data,
        model,
        submodule,
        tokenizer,
        device,
        dtype,
    )

    set_seed(cfg.seed)

    optimizer = torch.optim.AdamW(train_model_module.parameters(), lr=cfg.lr)

    global_step_size = cfg.train_batch_size * world_size
    effective_steps = (len(training_data) // global_step_size) * global_step_size
    if effective_steps != len(training_data):
        print(f"Trimming training_data from {len(training_data)} to {effective_steps} for equal DDP steps")
        training_data = training_data[:effective_steps]

    # Token accounting (approx): count tokens after the DDP trim and before sharding.
    # This slightly overestimates actual training tokens because we later trim per-rank
    # to align with gradient_accumulation_steps.
    tokens_per_epoch_est = 0
    num_examples_pre_shard = 0
    if rank == 0:
        tokens_per_epoch_est = sum(len(dp.input_ids) for dp in training_data)
        num_examples_pre_shard = len(training_data)

    # Shard dataset per rank (simple strided split)
    training_data = training_data[rank::world_size]

    num_batches_per_epoch = len(training_data) // cfg.train_batch_size
    # Each materialization block should contain an integer number of
    # optimizer steps, so we trim to full materialization blocks.
    batches_per_epoch = (
        num_batches_per_epoch // cfg.train_batches_per_materialization_block
    ) * cfg.train_batches_per_materialization_block
    trimmed_examples = batches_per_epoch * cfg.train_batch_size
    if trimmed_examples != len(training_data) and rank == 0:
        print(
            f"Trimming per-rank training_data from {len(training_data)} to {trimmed_examples} "
            "to align with materialization blocks"
        )
    training_data = training_data[:trimmed_examples]

    ao_steps_per_epoch = batches_per_epoch // cfg.gradient_accumulation_steps
    assert ao_steps_per_epoch > 0, "No optimizer steps will be run; check dataset/batch/accumulation sizes"

    chat_regularization_source: CyclingChatRegularizationSource | None = None
    chat_regularization_steps_per_epoch = 0
    chat_regularization_batches_per_epoch = 0
    chat_regularization_examples_pre_shard: int | None = None
    chat_regularization_tokens_per_epoch_est = 0

    if chat_regularization_data is not None:
        chat_regularization_rng = random.Random(cfg.seed + 1)
        chat_regularization_data = list(chat_regularization_data)
        chat_regularization_rng.shuffle(chat_regularization_data)

        if cfg.chat_regularization_max_train_examples is not None:
            before = len(chat_regularization_data)
            chat_regularization_data = chat_regularization_data[: cfg.chat_regularization_max_train_examples]
            if rank == 0:
                print(
                    f"Chat regularization trim: {before:,} -> {len(chat_regularization_data):,} examples "
                    f"(max_train_examples={cfg.chat_regularization_max_train_examples:,})"
                )

        chat_regularization_examples_pre_shard = len(chat_regularization_data)
        mean_chat_regularization_length = sum(len(dp.input_ids) for dp in chat_regularization_data) / len(
            chat_regularization_data
        )
        chat_regularization_data = chat_regularization_data[rank::world_size]
        if len(chat_regularization_data) < cfg.train_batch_size:
            raise ValueError(
                f"Need at least one full per-rank chat regularization batch, got {len(chat_regularization_data)} "
                f"examples for batch_size={cfg.train_batch_size}"
            )

        if cfg.window_mult is not None:
            chat_regularization_data = length_grouped_reorder(
                chat_regularization_data,
                cfg.train_batch_size,
                cfg.window_mult,
            )

        assert cfg.chat_regularization_every_n_ao_updates is not None, (
            "chat_regularization_every_n_ao_updates must be set when chat regularization is enabled"
        )
        chat_regularization_steps_per_epoch = ao_steps_per_epoch // cfg.chat_regularization_every_n_ao_updates
        assert chat_regularization_steps_per_epoch > 0, (
            "chat_regularization_every_n_ao_updates is too large for the available AO steps per epoch"
        )
        chat_regularization_batches_per_epoch = (
            chat_regularization_steps_per_epoch * cfg.gradient_accumulation_steps
        )
        chat_regularization_tokens_per_epoch_est = int(
            round(
                mean_chat_regularization_length
                * chat_regularization_batches_per_epoch
                * cfg.train_batch_size
                * world_size
            )
        )
        chat_regularization_source = CyclingChatRegularizationSource(
            data=chat_regularization_data,
            batch_size=cfg.train_batch_size,
            window_mult=cfg.window_mult,
            rng=random.Random(cfg.seed + 10_000 + rank),
        )

    total_training_tokens_est = (
        tokens_per_epoch_est + chat_regularization_tokens_per_epoch_est
    ) * cfg.num_epochs
    total_training_steps = (ao_steps_per_epoch + chat_regularization_steps_per_epoch) * cfg.num_epochs
    warmup_steps = int(total_training_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    # --------------------------------------------------------------

    global_step = 0

    # Init Weights & Biases only on rank 0
    if rank == 0:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=asdict(cfg))
        # Make examples_seen the natural x-axis for any train/* metric. Wandb still
        # records its own _step too, so existing dashboards aren't broken.
        wandb.define_metric("train/examples_seen")
        wandb.define_metric("train/loss", step_metric="train/examples_seen")
        wandb.define_metric("train/ao_loss", step_metric="train/examples_seen")
        wandb.define_metric("train/learning_rate", step_metric="train/examples_seen")
        wandb.define_metric("train/mean_num_acts", step_metric="train/examples_seen")
        wandb.define_metric("train/min_num_acts", step_metric="train/examples_seen")
        wandb.define_metric("train/max_num_acts", step_metric="train/examples_seen")
        wandb.define_metric("train/epoch", step_metric="train/examples_seen")
        wandb.define_metric("validation_loss/*", step_metric="train/examples_seen")
        wandb.define_metric("mcq/*", step_metric="train/examples_seen")
        wandb.summary["train/ao_tokens_per_epoch_est"] = tokens_per_epoch_est
        wandb.summary["train/chat_regularization_tokens_per_epoch_est"] = chat_regularization_tokens_per_epoch_est
        wandb.summary["train/tokens_per_epoch_est"] = (
            tokens_per_epoch_est + chat_regularization_tokens_per_epoch_est
        )
        wandb.summary["train/total_tokens_est"] = total_training_tokens_est
        wandb.summary["train/num_examples_pre_shard"] = num_examples_pre_shard
        wandb.summary["train/ao_optimizer_steps_per_epoch"] = ao_steps_per_epoch
        wandb.summary["train/chat_regularization_optimizer_steps_per_epoch"] = chat_regularization_steps_per_epoch
        if chat_regularization_examples_pre_shard is not None:
            wandb.summary["train/chat_regularization_num_examples_pre_shard"] = (
                chat_regularization_examples_pre_shard
            )

    training_loop_start_time = time.perf_counter()
    cumulative_target_tokens = 0  # for max_target_tokens stop
    max_target_tokens = getattr(cfg, "max_target_tokens", None) or 0
    token_budget_hit = False

    for epoch in range(cfg.num_epochs):
        if token_budget_hit:
            break
        accumulated_loss = 0.0
        optimizer.zero_grad()
        step_idx = 0
        ao_update_count = 0
        chat_update_count = 0
        materialization_block_size = cfg.train_batch_size * cfg.train_batches_per_materialization_block

        progress_bar = tqdm(
            total=batches_per_epoch + chat_regularization_batches_per_epoch,
            desc=f"Training epoch {epoch + 1}",
            disable=rank != 0,
        )
        for block_start in range(0, len(training_data), materialization_block_size):
            training_block = training_data[block_start : block_start + materialization_block_size]
            assert len(training_block) == materialization_block_size, (
                "training_data must be trimmed to full activation collection batches"
            )

            # We only build a small local list of batches here so steering vectors
            # do not accumulate across the entire epoch on GPU.
            materialized_batches = materialize_block_into_batches(cfg, training_block, tokenizer, model)
            for t_batch_list in materialized_batches:
                t_batch = construct_batch(t_batch_list, tokenizer, device)

                # Forward/backward on the DDP-wrapped module if enabled
                loss = train_features_batch(cfg, t_batch, train_model_module, submodule, device, dtype)
                loss = loss / cfg.gradient_accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()
                cumulative_target_tokens += _count_target_tokens(t_batch.labels)

                is_update_step = (step_idx + 1) % cfg.gradient_accumulation_steps == 0

                if is_update_step:
                    clip_grad_norm_(train_model_module.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    if rank == 0:
                        # global_step is incremented later; +1 here so step 0 logs
                        # "examples seen *after* the first optimizer step".
                        examples_seen = (global_step + 1) * cfg.train_batch_size * world_size
                        # Per-sample N (number of activation slots) = positions / num_layers.
                        num_layers_in_batch = max(1, len(t_batch_list[0].layers))
                        n_per_sample = [
                            len(p) // num_layers_in_batch for p in t_batch.positions
                        ]
                        mean_num_acts = sum(n_per_sample) / max(1, len(n_per_sample))
                        max_num_acts = max(n_per_sample) if n_per_sample else 0
                        min_num_acts = min(n_per_sample) if n_per_sample else 0
                        log_payload = {
                            "train/loss": accumulated_loss,
                            "train/ao_loss": accumulated_loss,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/examples_seen": examples_seen,
                            "train/mean_num_acts": mean_num_acts,
                            "train/max_num_acts": max_num_acts,
                            "train/min_num_acts": min_num_acts,
                        }
                        if cfg.examples_per_source_epoch:
                            log_payload["train/epoch"] = (
                                examples_seen / cfg.examples_per_source_epoch
                            )
                        wandb.log(log_payload, step=global_step)
                        if verbose:
                            print(f"AO step {global_step} loss: {accumulated_loss}")

                    run_periodic_training_actions(
                        cfg=cfg,
                        model=model,
                        submodule=submodule,
                        tokenizer=tokenizer,
                        device=device,
                        dtype=dtype,
                        global_step=global_step,
                        rank=rank,
                        component_validation_data=component_validation_data,
                    )

                    global_step += 1
                    ao_update_count += 1
                    accumulated_loss = 0.0

                    if max_target_tokens > 0 and cumulative_target_tokens >= max_target_tokens:
                        if rank == 0:
                            mt_m = max_target_tokens // 1_000_000
                            print(
                                f"[token budget] hit {cumulative_target_tokens:,} >= {max_target_tokens:,} "
                                f"target tokens at step {global_step}; saving and stopping"
                            )
                            # Save the checkpoint anchored at the token budget RIGHT NOW so
                            # we have a robust artifact even if the downstream cleanup path
                            # (final eval, push-to-hub) errors. The standard 'final' save
                            # at the end of train_model still runs and overwrites with the
                            # same weights — this is an extra-safe anchor.
                            token_save_dir = Path(cfg.save_dir) / f"token_budget_{mt_m}M"
                            try:
                                model.save_pretrained(token_save_dir)
                                write_training_config(token_save_dir, cfg)
                                print(f"[token budget] saved checkpoint to {token_save_dir}")
                            except Exception as exc:
                                print(f"[token budget] WARN: save failed: {exc}")
                        token_budget_hit = True
                        break

                    if (
                        chat_regularization_source is not None
                        and cfg.chat_regularization_every_n_ao_updates is not None
                        and ao_update_count % cfg.chat_regularization_every_n_ao_updates == 0
                    ):
                        chat_loss_accumulated = 0.0
                        for _ in range(cfg.gradient_accumulation_steps):
                            chat_batch_list = chat_regularization_source.next_batch()
                            chat_batch = construct_chat_regularization_batch(chat_batch_list, tokenizer, device)
                            chat_loss = train_chat_regularization_batch(chat_batch, train_model_module)
                            chat_loss = (
                                chat_loss * cfg.chat_regularization_weight / cfg.gradient_accumulation_steps
                            )
                            chat_loss.backward()
                            chat_loss_accumulated += chat_loss.item()
                            progress_bar.update(1)

                        clip_grad_norm_(train_model_module.parameters(), cfg.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                        if rank == 0:
                            wandb.log(
                                {
                                    "train/chat_regularization_loss": chat_loss_accumulated,
                                    "train/learning_rate": scheduler.get_last_lr()[0],
                                },
                                step=global_step,
                            )
                            if verbose:
                                print(f"Chat regularization step {global_step} loss: {chat_loss_accumulated}")

                        run_periodic_training_actions(
                            cfg=cfg,
                            model=model,
                            submodule=submodule,
                            tokenizer=tokenizer,
                            device=device,
                            dtype=dtype,
                            global_step=global_step,
                            rank=rank,
                            component_validation_data=component_validation_data,
                        )

                        global_step += 1
                        chat_update_count += 1

                step_idx += 1
                progress_bar.update(1)
            if token_budget_hit:
                break
        progress_bar.close()
        if chat_regularization_source is not None:
            assert chat_update_count == chat_regularization_steps_per_epoch, (
                f"Expected {chat_regularization_steps_per_epoch} chat regularization updates, "
                f"got {chat_update_count}"
            )

    print("Training complete.")

    # Save final model
    if rank == 0:
        # log training loop performance metrics
        training_loop_wall_time_sec = time.perf_counter() - training_loop_start_time
        total_batches_processed = (batches_per_epoch + chat_regularization_batches_per_epoch) * cfg.num_epochs
        wandb.summary["train/training_loop_wall_time_sec"] = training_loop_wall_time_sec
        wandb.summary["train/training_loop_batches_per_sec"] = total_batches_processed / training_loop_wall_time_sec
        wandb.summary["train/training_loop_examples_per_sec"] = (
            total_batches_processed * cfg.train_batch_size / training_loop_wall_time_sec
        )

        print("Saving final model...")
        final_dir = Path(cfg.save_dir) / "final"
        model.save_pretrained(final_dir)
        write_training_config(final_dir, cfg)

        # Final evaluation
        print("Running final evaluation...")
        run_open_ended_eval(
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            device=device,
            global_step=global_step,
            phase="final",
            delete_temp_adapter=True,
        )
        wandb.finish()

        # Push to Hugging Face if configured
        if cfg.hf_push_to_hub and cfg.hf_repo_id:
            print("Pushing LoRA adapter to Hugging Face Hub...")
            push_lora_to_hf(
                model=model,
                tokenizer=tokenizer,
                repo_id=cfg.hf_repo_id,
                commit_message=f"SAE introspection LoRA - {cfg.wandb_run_name} - final model",
                private=cfg.hf_private_repo,
                training_config_path=final_dir / TRAINING_CONFIG_FILENAME,
            )
    dist.barrier()


def length_grouped_reorder(
    data: list[TrainingDataPoint | ChatRegularizationDataPoint],
    batch_size: int,
    window_mult: int,
) -> list[TrainingDataPoint | ChatRegularizationDataPoint]:
    lengths = [len(d.input_ids) for d in data]

    indices = list(range(len(data)))
    megabatch_size = window_mult * batch_size

    # Slice into mega-batches
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(indices), megabatch_size)]
    # Sort within each mega-batch by length desc
    megabatches = [sorted(mb, key=lambda i: lengths[i], reverse=True) for mb in megabatches]

    new_order = [i for mb in megabatches for i in mb]
    return [data[i] for i in new_order]


def get_component_name(
    dataset_loader: ActDatasetLoader,
    component_idx: int,
) -> str:
    dataset_name = dataset_loader.dataset_config.dataset_name
    dataset_params = dataset_loader.dataset_config.custom_dataset_params

    if dataset_name.startswith("classification_"):
        return "classification_all"

    if dataset_name == "past_lens":
        return f"past_lens_acts_{dataset_params.min_k_activations}_to_{dataset_params.max_k_activations}"

    if dataset_name == "prebuilt_pt":
        return dataset_params.component_name

    if dataset_name == "synthetic_qa":
        data_path = Path(dataset_params.data_path)
        # Use parent directory name when file is generically named "training_data",
        # otherwise use the file stem (e.g. "spqa_v2")
        label = data_path.parent.name if data_path.stem == "training_data" else data_path.stem
        return f"synthetic_qa_{label}"

    if dataset_name == "cot_oracle_convqa":
        # e.g. "cds-jb/fineweb-oracle-convqa-chunked" + "test" -> "fineweb_test"
        repo_short = dataset_params.hf_dataset_repo.split("/")[-1].replace(
            "-oracle-convqa-chunked", ""
        ).replace("-", "_")
        return f"{repo_short}_{dataset_params.hf_split}"

    return f"{dataset_name}_{component_idx:02d}"


def build_datasets(
    cfg: SelfInterpTrainingConfig,
    dataset_loaders: list[ActDatasetLoader],
    max_len_percentile: float | None = 0.999,
    window_mult: int | None = 20,
) -> tuple[list[TrainingDataPoint], dict[str, list[TrainingDataPoint]]]:
    set_seed(cfg.seed)
    training_components: list[TrainingComponent] = []
    # eval data will only be for classification datasets
    all_eval_data: dict[str, list[TrainingDataPoint]] = {}

    for component_idx, dataset_loader in enumerate(dataset_loaders):
        component_name = get_component_name(dataset_loader, component_idx)
        if "train" in dataset_loader.dataset_config.splits:
            training_components.append(
                TrainingComponent(
                    component_name=component_name,
                    training_data=dataset_loader.load_dataset("train"),
                )
            )
        if "test" in dataset_loader.dataset_config.splits:
            all_eval_data[dataset_loader.dataset_config.dataset_name] = dataset_loader.load_dataset("test")

    all_training_data = [dp for component in training_components for dp in component.training_data]
    p = max_len_percentile
    if p is not None:
        if p >= 1.0 or p <= 0.0:
            raise ValueError("max_len_percentile must be less than 1.0 and greater than 0.0")

        lengths = sorted(len(td.input_ids) for td in all_training_data)
        median_length = lengths[len(lengths) // 2]
        print(f"Max length: {lengths[-1]}, Min length: {lengths[0]}, Median length: {median_length}")
        # Inclusive quantile index
        idx = int((len(lengths) - 1) * p)
        threshold = lengths[idx]

        before = len(all_training_data)
        filtered_components: list[TrainingComponent] = []
        total_after = 0
        for component in training_components:
            filtered_training_data = [td for td in component.training_data if len(td.input_ids) <= threshold]
            total_after += len(filtered_training_data)
            filtered_components.append(
                TrainingComponent(
                    component_name=component.component_name,
                    training_data=filtered_training_data,
                )
            )
        training_components = filtered_components
        removed = before - total_after
        print(f"Percentile trim: kept <= {threshold} tokens (p={p:.6f}). Removed {removed}/{before} examples.")

    all_training_data = [dp for component in training_components for dp in component.training_data]

    set_seed(cfg.seed)
    random.shuffle(all_training_data)

    if cfg.max_train_examples is not None:
        before = len(all_training_data)
        all_training_data = all_training_data[: cfg.max_train_examples]
        print(
            f"Budget trim: {before:,} → {len(all_training_data):,} examples (max_train_examples={cfg.max_train_examples:,})"
        )

    if window_mult is not None:
        all_training_data = length_grouped_reorder(all_training_data, cfg.train_batch_size, window_mult)

    return all_training_data, all_eval_data


def build_validation_datasets(
    validation_loaders: list[ActDatasetLoader],
) -> dict[str, list[TrainingDataPoint]]:
    component_validation_data: dict[str, list[TrainingDataPoint]] = {}

    for component_idx, dataset_loader in enumerate(validation_loaders):
        component_name = get_component_name(dataset_loader, component_idx)
        validation_data = dataset_loader.load_dataset("train")
        if component_name in component_validation_data:
            raise ValueError(f"Duplicate validation component name: {component_name}")
        component_validation_data[component_name] = validation_data

    return component_validation_data


def resolve_validation_loaders_from_config(cfg: SelfInterpTrainingConfig) -> list[ActDatasetLoader]:
    from nl_probes.dataset_classes.act_dataset_manager import build_loaders_from_saved_configs

    return build_loaders_from_saved_configs(cfg.validation_dataset_configs)


def _ensure_datasets_exist(dataset_loaders: list[ActDatasetLoader]) -> None:
    """Materialize datasets on disk using a single process (rank 0).

    Each loader's `load_dataset` will create and save if missing; otherwise it
    simply loads. This avoids race conditions when multiple ranks start up.
    """

    # TODO: Switch to multiprocessing for speed

    for dl in dataset_loaders:
        for split in dl.dataset_config.splits:
            dl.ensure_dataset_exists(split)


if __name__ == "__main__":
    """
    Config-driven training. Requires a config JSON (generate with draft_training_configs.py).

    Dataset generation (single process, uses vLLM):
        python nl_probes/sft.py --config path/to/ao_config.json --gen-only

    Training (torchrun, DDP):
        torchrun --nproc_per_node=1 nl_probes/sft.py --config path/to/ao_config.json

    Optional: override HF repo ID:
        torchrun --nproc_per_node=1 nl_probes/sft.py --config path/to/ao_config.json --hf-repo-id org/repo
    """

    from nl_probes.dataset_classes.act_dataset_manager import build_loaders_from_config
    from nl_probes.configs.sft_config import read_training_config

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a training config JSON (ao_config.json).",
    )
    parser.add_argument(
        "--gen-only",
        action="store_true",
        help="Generate/load datasets on disk and exit before training.",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="Override hf_repo_id in the config (also sets hf_push_to_hub=True).",
    )
    args = parser.parse_args()

    # Load config
    cfg = read_training_config(args.config)
    if args.hf_repo_id:
        cfg.hf_repo_id = args.hf_repo_id
        cfg.hf_push_to_hub = True

    # DDP init
    # For Gemma models: export TORCHDYNAMO_DISABLE=1
    # Timeout is 2 hours because dataset generation (vLLM) can take ~1 hour
    if args.gen_only:
        torchrun_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torchrun_local_rank != 0:
            print("Skipping dataset generation on non-zero LOCAL_RANK in --gen-only mode")
            raise SystemExit(0)
        local_rank = 0
        world_size = 1
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    else:
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(_local_cuda_index(local_rank))
        world_size = dist.get_world_size()

    # Adjust batch size for world size
    global_batch_size = cfg.train_batch_size
    assert global_batch_size % world_size == 0, (
        f"Global batch size {global_batch_size} must be divisible by world_size {world_size}"
    )
    cfg.train_batch_size = global_batch_size // world_size
    print(f"Per-rank train batch size: {cfg.train_batch_size}, world size: {world_size}")

    dataset_loaders = build_loaders_from_config(cfg)
    validation_loaders = resolve_validation_loaders_from_config(cfg)

    dtype = torch.bfloat16
    device = torch.device(f"cuda:{_local_cuda_index(local_rank)}")

    # Dataset generation
    if local_rank == 0:
        _ensure_datasets_exist(dataset_loaders)
        _ensure_datasets_exist(validation_loaders)
        if cfg.chat_regularization_path is not None:
            chat_regularization_path = Path(cfg.chat_regularization_path)
            assert chat_regularization_path.exists(), (
                f"Missing chat regularization dataset: {chat_regularization_path}"
            )
    if not args.gen_only:
        dist.barrier()

    if args.gen_only:
        if local_rank == 0:
            print("Dataset generation complete (--gen-only); exiting before training.")
        raise SystemExit(0)

    # Build model_kwargs from config
    model_kwargs: dict[str, Any] = {}
    if cfg.load_in_8bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=dtype,
        )

    # Training
    tokenizer = load_tokenizer(cfg.model_name)
    all_training_data, all_eval_data = build_datasets(
        cfg,
        dataset_loaders=dataset_loaders,
        window_mult=cfg.window_mult,
    )
    component_validation_data = build_validation_datasets(validation_loaders)
    chat_regularization_data: list[ChatRegularizationDataPoint] | None = None
    if cfg.chat_regularization_path is not None:
        chat_regularization_data = load_chat_regularization_data(
            cfg.chat_regularization_path,
            expected_model_name=cfg.model_name,
        )
    print(f"training data length: {len(all_training_data)}, eval data length: {len(all_eval_data)}")
    if component_validation_data:
        print(
            "validation sizes: "
            + ", ".join(
                f"{component_name}={len(component_data)}"
                for component_name, component_data in component_validation_data.items()
            )
        )
    if chat_regularization_data is not None:
        print(f"chat regularization data length: {len(chat_regularization_data)}")
    print(asdict(cfg))

    train_model(
        cfg=cfg,
        training_data=all_training_data,
        component_validation_data=component_validation_data,
        chat_regularization_data=chat_regularization_data,
        tokenizer=tokenizer,
        dtype=dtype,
        device=device,
        model_kwargs=model_kwargs,
        verbose=True,
    )
    dist.destroy_process_group()
