"""Cross-attention activation oracle training script.

Trains a supervisor model with per-layer cross-attention adapters that attend
to supervisee activations. Uses FineWeb-only PastLens data by default.

Launch:
    torchrun --nproc_per_node=8 nl_probes/sft_cross_attn.py
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import random
from dataclasses import asdict
from dotenv import load_dotenv

load_dotenv()
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
import wandb
from peft import LoraConfig, get_peft_model
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.dataset_classes.past_lens_dataset import (
    PastLensDatasetConfig,
    collect_cross_attn_past_lens_data,
    hf_fineweb_generator,
    hf_mixed_dataset_to_generator,
)
from nl_probes.models.cross_attention_oracle import (
    CrossAttentionWrapper,
    get_cross_attn_state_dicts,
    load_cross_attn_state_dicts,
    prepare_oracle_input,
    wrap_model_with_cross_attention,
)
from nl_probes.utils.activation_utils import (
    collect_activations_multiple_layers,
    get_hf_submodule,
)
from nl_probes.utils.common import get_layer_count, load_model, load_tokenizer, set_seed
from nl_probes.utils.dataset_utils import (
    CrossAttnBatchData,
    CrossAttnTrainingDataPoint,
    construct_cross_attn_batch,
    create_cross_attn_datapoint,
)
from nl_probes.utils.eval import parse_answer


def collect_supervisee_activations(
    base_model: AutoModelForCausalLM,
    batch: CrossAttnBatchData,
    num_layers: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Run supervisee (base model, no adapter) to get activations at all layers."""
    from peft import PeftModel

    use_lora = isinstance(base_model, PeftModel)
    submodules = {L: get_hf_submodule(base_model, L, use_lora=use_lora) for L in range(num_layers)}

    inputs_BL = {
        "input_ids": batch.context_input_ids,
        "attention_mask": batch.context_attention_mask,
    }

    with torch.no_grad():
        acts = collect_activations_multiple_layers(
            model=base_model,
            submodules=submodules,
            inputs_BL=inputs_BL,
            min_offset=None,
            max_offset=None,
        )

    return acts


def subset_layers(
    acts: dict[int, torch.Tensor],
    k: int,
) -> dict[int, torch.Tensor]:
    """Randomly keep only k layers, zeroing out the rest.

    Forces the model to work with incomplete layer information, preventing
    trivial reliance on all layers simultaneously. At eval time, pass k=0
    to use all layers.
    """
    if k <= 0 or k >= len(acts):
        return acts

    all_layers = list(acts.keys())
    keep = set(random.sample(all_layers, k))
    return {layer: (act if layer in keep else torch.zeros_like(act)) for layer, act in acts.items()}


def train_cross_attn_batch(
    model: AutoModelForCausalLM,
    base_model: AutoModelForCausalLM,
    batch: CrossAttnBatchData,
    tokenizer: PreTrainedTokenizer,
    cfg: SelfInterpTrainingConfig,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Train on a single cross-attention batch.

    1. Collect supervisee activations from all layers (base model, no adapter)
    2. Add noise augmentation
    3. Set supervisee acts for cross-attention modules
    4. Build oracle input with "?" prefix + query
    5. Forward pass, return loss
    """
    # 1. Collect supervisee activations
    supervisee_acts = collect_supervisee_activations(base_model, batch, num_layers, device)

    # 2. Layer subsetting (random k layers per step)
    supervisee_acts = subset_layers(supervisee_acts, cfg.layer_subset_k)

    # 3. Build mask dict (same mask for all layers)
    mask_dict = {L: batch.context_attention_mask for L in range(num_layers)}

    # 4. Set supervisee activations for cross-attention
    CrossAttentionWrapper.set_supervisee_acts(supervisee_acts, mask_dict)

    # 5. Build oracle input
    inputs_embeds, attention_mask, labels = prepare_oracle_input(
        model=model,
        tokenizer=tokenizer,
        context_lengths=batch.context_lengths,
        query_input_ids=batch.query_input_ids,
        query_attention_mask=batch.query_attention_mask,
        query_labels=batch.query_labels,
        device=device,
    )

    # 6. Forward pass
    loss = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
    ).loss

    # 7. Clean up
    CrossAttentionWrapper.clear_supervisee_acts()

    return loss


@torch.no_grad()
def run_cross_attn_evaluation(
    eval_data: list[CrossAttnTrainingDataPoint],
    model: AutoModelForCausalLM,
    base_model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
    eval_batch_size: int,
    generation_kwargs: dict,
) -> list[dict[str, str]]:
    """Run evaluation for cross-attention oracle.

    Returns list of dicts with 'response' and 'target' keys.
    """
    model.eval()
    results = []

    for i in tqdm(range(0, len(eval_data), eval_batch_size), desc="Cross-attn eval"):
        batch_data = eval_data[i : i + eval_batch_size]
        batch = construct_cross_attn_batch(batch_data, tokenizer, device)

        # Collect supervisee acts (adapter disabled to match training)
        with base_model.disable_adapter():
            supervisee_acts = collect_supervisee_activations(base_model, batch, num_layers, device)
        mask_dict = {L: batch.context_attention_mask for L in range(num_layers)}
        CrossAttentionWrapper.set_supervisee_acts(supervisee_acts, mask_dict)

        # Build oracle input (prompt-only for generation — strip labels)
        # We need prompt-only: find where labels != -100 and truncate
        prompt_only_ids = []
        prompt_only_masks = []
        for j in range(len(batch_data)):
            q_ids = batch_data[j].query_input_ids
            q_labels = batch_data[j].query_labels
            # Keep only prompt tokens (labels == -100)
            prompt_ids = []
            for tok, lab in zip(q_ids, q_labels):
                if lab == -100:
                    prompt_ids.append(tok)
                else:
                    break
            prompt_only_ids.append(prompt_ids)

        # Left-pad prompt-only ids
        max_prompt_len = max(len(p) for p in prompt_only_ids)
        pad_id = tokenizer.pad_token_id
        padded_ids = []
        padded_masks = []
        for p in prompt_only_ids:
            pad_len = max_prompt_len - len(p)
            padded_ids.append([pad_id] * pad_len + p)
            padded_masks.append([False] * pad_len + [True] * len(p))

        prompt_input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        prompt_attn_mask = torch.tensor(padded_masks, dtype=torch.bool, device=device)

        inputs_embeds, attention_mask, _ = prepare_oracle_input(
            model=model,
            tokenizer=tokenizer,
            context_lengths=batch.context_lengths,
            query_input_ids=prompt_input_ids,
            query_attention_mask=prompt_attn_mask,
            query_labels=torch.full_like(prompt_input_ids, -100),
            device=device,
        )

        output_ids = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

        # Decode only newly generated tokens
        gen_start = inputs_embeds.shape[1]
        generated_tokens = output_ids[:, gen_start:]
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for j, response in enumerate(decoded):
            # Decode prompt and context for logging
            prompt_text = tokenizer.decode(prompt_only_ids[j], skip_special_tokens=True)
            context_text = tokenizer.decode(batch_data[j].context_input_ids, skip_special_tokens=True)
            meta = batch_data[j].meta_info or {}
            results.append({
                "response": response,
                "target": batch_data[j].target_output,
                "prompt": prompt_text,
                "context_text": context_text,
                "datapoint_type": batch_data[j].datapoint_type,
                "ds_label": meta.get("ds_label", ""),
            })

        CrossAttentionWrapper.clear_supervisee_acts()

    model.train()
    return results


def score_cross_attn_eval(
    results: list[dict[str, str]],
    valid_answers: list[str] | None = None,
) -> dict[str, float]:
    """Score cross-attention evaluation results.

    If valid_answers is provided, computes format_correct and ans_correct.
    Otherwise, computes exact match accuracy (for PastLens-like tasks).
    """
    if valid_answers is not None:
        format_correct = 0
        ans_correct = 0
        for r in results:
            cleaned = parse_answer(r["response"])
            target = parse_answer(r["target"])
            if cleaned in valid_answers:
                format_correct += 1
            if cleaned == target:
                ans_correct += 1
        return {
            "format_correct": format_correct / len(results) if results else 0,
            "ans_correct": ans_correct / len(results) if results else 0,
        }
    else:
        exact_match = sum(1 for r in results if r["response"].strip() == r["target"].strip())
        return {"exact_match": exact_match / len(results) if results else 0}


def eval_all_cross_attn_datasets(
    cfg: SelfInterpTrainingConfig,
    eval_datasets: dict[str, list[CrossAttnTrainingDataPoint]],
    model: AutoModelForCausalLM,
    base_model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
) -> None:
    """Evaluate on all datasets and log to wandb."""
    eval_results = {}
    table = wandb.Table(columns=["step", "dataset", "ds_label", "prompt", "response", "target", "correct", "context_text"])

    for ds_name, eval_data in eval_datasets.items():
        results = run_cross_attn_evaluation(
            eval_data=eval_data,
            model=model,
            base_model=base_model,
            tokenizer=tokenizer,
            num_layers=num_layers,
            device=device,
            dtype=dtype,
            eval_batch_size=cfg.eval_batch_size,
            generation_kwargs=cfg.generation_kwargs,
        )

        # Classification datasets use yes/no; PastLens uses exact match
        is_classification = any(k in ds_name for k in ["sst2", "ag_news", "geometry", "gender", "snli", "ner", "tense", "relation", "language", "singular"])
        if is_classification:
            scores = score_cross_attn_eval(results, valid_answers=["yes", "no"])
            eval_results[f"eval_format_correct/{ds_name}"] = scores["format_correct"]
            eval_results[f"eval_ans_correct/{ds_name}"] = scores["ans_correct"]
            print(f"Step {global_step} {ds_name}: format={scores['format_correct']:.3f}, acc={scores['ans_correct']:.3f}")
        else:
            scores = score_cross_attn_eval(results)
            eval_results[f"eval_exact_match/{ds_name}"] = scores["exact_match"]
            print(f"Step {global_step} {ds_name}: exact_match={scores['exact_match']:.3f}")

        # Add up to 50 examples per dataset to the table
        for r in results[:50]:
            correct = parse_answer(r["response"]) == parse_answer(r["target"])
            table.add_data(
                global_step, ds_name, r.get("ds_label", ""),
                r.get("prompt", ""), r["response"], r["target"], correct,
                r.get("context_text", ""),
            )

    eval_results["eval_examples"] = table
    wandb.log(eval_results, step=global_step)
    wandb.summary.update({k: v for k, v in eval_results.items() if not isinstance(v, wandb.Table)})

    torch.cuda.empty_cache()
    gc.collect()


def log_gate_values(model: AutoModelForCausalLM, global_step: int) -> None:
    """Log gate values per layer to wandb."""
    gate_values = {}
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        if isinstance(layer, CrossAttentionWrapper):
            gate_val = layer.cross_attn.gate.item()
            gate_values[f"gate/layer_{i}"] = gate_val
    if gate_values:
        wandb.log(gate_values, step=global_step)


def build_cross_attn_eval_data(
    tokenizer: PreTrainedTokenizer,
    num_eval: int = 250,
) -> dict[str, list[CrossAttnTrainingDataPoint]]:
    """Build classification eval datasets for cross-attention oracle.

    Directly loads raw classification examples (text only, no model needed)
    and converts to CrossAttnTrainingDataPoint format.
    """
    from nl_probes.dataset_classes.classification import (
        get_classification_datapoints,
    )

    eval_datasets = {}
    classification_names = [
        "sst2", "ag_news", "geometry_of_truth",
        "md_gender", "snli", "ner", "tense",
        "language_identification", "singular_plural", "relations",
    ]

    for ds_name in classification_names:
        try:
            _, test_datapoints = get_classification_datapoints(
                dataset_name=ds_name,
                num_qa_per_sample=3,
                train_examples=0,
                test_examples=num_eval,
                random_seed=42,
            )
        except Exception as e:
            print(f"Warning: Could not load eval dataset {ds_name}: {e}")
            continue

        cross_attn_data = []
        for dp in test_datapoints:
            # Tokenize the activation_prompt as context
            context_ids = tokenizer.encode(dp.activation_prompt, add_special_tokens=False)
            if not context_ids:
                continue

            ca_dp = create_cross_attn_datapoint(
                datapoint_type=f"eval_{ds_name}",
                prompt=dp.classification_prompt,
                target_response=dp.target_response,
                context_input_ids=context_ids,
                tokenizer=tokenizer,
                meta_info={"ds_label": dp.ds_label or ""},
            )
            cross_attn_data.append(ca_dp)

        if cross_attn_data:
            eval_datasets[ds_name] = cross_attn_data[:num_eval]
            print(f"Built {len(eval_datasets[ds_name])} cross-attn eval examples for {ds_name}")

    return eval_datasets


def save_checkpoint(
    model: AutoModelForCausalLM,
    save_dir: str,
    step: int | str,
) -> None:
    """Save LoRA adapter + cross-attention weights."""
    checkpoint_dir = f"{save_dir}/step_{step}" if isinstance(step, int) else f"{save_dir}/{step}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save LoRA adapter
    model.save_pretrained(checkpoint_dir)

    # Save cross-attention state dicts separately
    # Need to access through PEFT wrapper
    if hasattr(model, "base_model"):
        base = model.base_model.model
    else:
        base = model
    cross_attn_sd = get_cross_attn_state_dicts(base)
    torch.save(cross_attn_sd, os.path.join(checkpoint_dir, "cross_attn_weights.pt"))
    print(f"Saved checkpoint to {checkpoint_dir}")


def train_cross_attn_model(
    cfg: SelfInterpTrainingConfig,
    training_data: list[CrossAttnTrainingDataPoint],
    eval_datasets: dict[str, list[CrossAttnTrainingDataPoint]],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    model_kwargs: dict[str, Any],
):
    """Main cross-attention oracle training loop with DDP."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    model_kwargs = {
        **model_kwargs,
        "device_map": {"": f"cuda:{local_rank}"},
    }

    set_seed(cfg.seed)

    # Load base model and wrap with cross-attention BEFORE LoRA
    model = load_model(cfg.model_name, dtype, **model_kwargs)
    num_layers = len(model.model.layers)

    print(f"Wrapping {num_layers} layers with cross-attention adapters...")
    wrap_model_with_cross_attention(
        model,
        hidden_dim=cfg.cross_attn_hidden_dim,
        num_heads=cfg.cross_attn_num_heads,
        gate_init=cfg.cross_attn_gate_init,
    )

    model.enable_input_require_grads()

    if cfg.gradient_checkpointing:
        model.use_cache = False
        model.gradient_checkpointing_enable()

    # Apply LoRA targeting only original_layer projections (not cross-attention)
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=[
                "original_layer.self_attn.q_proj",
                "original_layer.self_attn.k_proj",
                "original_layer.self_attn.v_proj",
                "original_layer.self_attn.o_proj",
                "original_layer.mlp.gate_proj",
                "original_layer.mlp.up_proj",
                "original_layer.mlp.down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config, autocast_adapter_dtype=True)

        # PEFT freezes all non-LoRA params. Unfreeze cross-attention and its norm.
        for name, param in model.named_parameters():
            if "cross_attn" in name or "cross_attn_norm" in name:
                param.requires_grad = True

    model.print_trainable_parameters()

    # We need a separate reference to the base model (without LoRA adapter)
    # for collecting supervisee activations. Since the model is LoRA-wrapped,
    # we use disable_adapter() context manager during activation collection.
    # The base_model reference here is the same model — we'll disable adapter
    # in collect_supervisee_activations.

    # Wrap with DDP
    torch.cuda.set_device(local_rank)
    train_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
    )
    train_model.train()

    # OOM preflight
    _oom_preflight_cross_attn(cfg, training_data, model, tokenizer, num_layers, device, dtype)

    set_seed(cfg.seed)

    # Separate param groups: higher LR for cross-attention projections and gate.
    # The projections are randomly initialised and their gradients are scaled by
    # sigmoid(gate) ≈ 0.12, so they need a much higher LR than the LoRA params
    # to break the chicken-and-egg deadlock with the gate.
    gate_params = []
    cross_attn_params = []
    other_params = []
    for name, param in train_model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".cross_attn.gate"):
            gate_params.append(param)
        elif ".cross_attn." in name or ".cross_attn_norm." in name:
            cross_attn_params.append(param)
        else:
            other_params.append(param)
    if rank == 0:
        n_gate = sum(p.numel() for p in gate_params)
        n_xattn = sum(p.numel() for p in cross_attn_params)
        n_other = sum(p.numel() for p in other_params)
        print(f"Optimizer groups — LoRA/other: {n_other:,}  cross-attn proj: {n_xattn:,}  gates: {n_gate}")
    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": cfg.lr},
        {"params": cross_attn_params, "lr": cfg.lr * 10},
        {"params": gate_params, "lr": cfg.lr * 10},
    ])

    # Trim and shard data
    global_step_size = cfg.train_batch_size * world_size
    effective_steps = (len(training_data) // global_step_size) * global_step_size
    if effective_steps != len(training_data) and rank == 0:
        print(f"Trimming training_data from {len(training_data)} to {effective_steps} for equal DDP steps")
    training_data = training_data[:effective_steps]

    if rank == 0:
        tokens_per_epoch_est = sum(len(dp.query_input_ids) + len(dp.context_input_ids) for dp in training_data)
        total_training_tokens_est = tokens_per_epoch_est * cfg.num_epochs

    # Shard per rank
    training_data = training_data[rank::world_size]

    num_batches_per_epoch = len(training_data) // cfg.train_batch_size
    batches_per_epoch = (num_batches_per_epoch // cfg.gradient_accumulation_steps) * cfg.gradient_accumulation_steps
    trimmed_examples = batches_per_epoch * cfg.train_batch_size
    if trimmed_examples != len(training_data) and rank == 0:
        print(f"Trimming per-rank from {len(training_data)} to {trimmed_examples} for grad accumulation")
    training_data = training_data[:trimmed_examples]

    steps_per_epoch = batches_per_epoch // cfg.gradient_accumulation_steps
    assert steps_per_epoch > 0, "No optimizer steps will be run"
    total_training_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_training_steps * 0.1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    global_step = 0

    if rank == 0:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=asdict(cfg))
        wandb.summary["train/tokens_per_epoch_est"] = tokens_per_epoch_est
        wandb.summary["train/total_tokens_est"] = total_training_tokens_est

    for epoch in range(cfg.num_epochs):
        accumulated_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(
            range(0, len(training_data), cfg.train_batch_size),
            desc=f"Training epoch {epoch + 1}",
            disable=rank != 0,
        )
        for step_idx, start in enumerate(pbar):
            t_batch_list = training_data[start : start + cfg.train_batch_size]
            t_batch = construct_cross_attn_batch(t_batch_list, tokenizer, device)

            # Collect supervisee activations using model with adapter disabled
            with model.disable_adapter():
                supervisee_acts = collect_supervisee_activations(model, t_batch, num_layers, device)

            # Layer subsetting (random k layers per step)
            supervisee_acts = subset_layers(supervisee_acts, cfg.layer_subset_k)

            # Set cross-attention inputs
            mask_dict = {L: t_batch.context_attention_mask for L in range(num_layers)}
            CrossAttentionWrapper.set_supervisee_acts(supervisee_acts, mask_dict)

            # Build oracle input
            inputs_embeds, attention_mask, labels = prepare_oracle_input(
                model=model,
                tokenizer=tokenizer,
                context_lengths=t_batch.context_lengths,
                query_input_ids=t_batch.query_input_ids,
                query_attention_mask=t_batch.query_attention_mask,
                query_labels=t_batch.query_labels,
                device=device,
            )

            # Forward + backward
            loss = train_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            ).loss
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

            CrossAttentionWrapper.clear_supervisee_acts()

            is_update_step = (step_idx + 1) % cfg.gradient_accumulation_steps == 0

            if is_update_step:
                clip_grad_norm_(train_model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if rank == 0:
                    log_dict = {
                        "train/loss": accumulated_loss,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                    }
                    wandb.log(log_dict, step=global_step)
                    pbar.set_postfix(step=global_step, loss=f"{accumulated_loss:.4f}")

                    # Log gate values periodically
                    if global_step % 100 == 0:
                        # Access through PEFT wrapper
                        if hasattr(model, "base_model"):
                            log_gate_values(model.base_model.model, global_step)
                        else:
                            log_gate_values(model, global_step)

                # Evaluation
                if global_step % cfg.eval_steps == 0 and (cfg.eval_on_start or global_step > 0):
                    if rank == 0 and eval_datasets:
                        eval_all_cross_attn_datasets(
                            cfg, eval_datasets, model, model, tokenizer,
                            num_layers, device, dtype, global_step,
                        )
                    dist.barrier()

                # Checkpointing
                if global_step % cfg.save_steps == 0 and global_step > 0:
                    if rank == 0:
                        save_checkpoint(model, cfg.save_dir, global_step)
                    dist.barrier()

                global_step += 1
                accumulated_loss = 0.0

    print("Training complete.")

    if rank == 0:
        save_checkpoint(model, cfg.save_dir, "final")

        if eval_datasets:
            print("Running final evaluation...")
            eval_all_cross_attn_datasets(
                cfg, eval_datasets, model, model, tokenizer,
                num_layers, device, dtype, global_step,
            )
        wandb.finish()

    dist.barrier()


def _oom_preflight_cross_attn(
    cfg: SelfInterpTrainingConfig,
    training_data: list[CrossAttnTrainingDataPoint],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """OOM preflight check for cross-attention training."""
    longest = max(training_data, key=lambda x: len(x.query_input_ids) + len(x.context_input_ids))
    long_prompts = [longest] * cfg.train_batch_size
    batch = construct_cross_attn_batch(long_prompts, tokenizer, device)

    # Need to set dummy supervisee acts so all cross-attn params participate
    dummy_acts = {}
    B = batch.context_input_ids.shape[0]
    L_ctx = batch.context_input_ids.shape[1]
    D = cfg.cross_attn_hidden_dim
    for L in range(num_layers):
        dummy_acts[L] = torch.zeros(B, L_ctx, D, device=device, dtype=dtype)
    mask_dict = {L: batch.context_attention_mask for L in range(num_layers)}
    CrossAttentionWrapper.set_supervisee_acts(dummy_acts, mask_dict)

    inputs_embeds, attention_mask, labels = prepare_oracle_input(
        model, tokenizer, batch.context_lengths,
        batch.query_input_ids, batch.query_attention_mask,
        batch.query_labels, device,
    )

    dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)

    for _ in tqdm(range(3), desc="OOM preflight check"):
        # Re-set supervisee acts each iteration (cleared by forward pass)
        CrossAttentionWrapper.set_supervisee_acts(dummy_acts, mask_dict)
        # Detach inputs_embeds so each iteration builds a fresh graph
        loss = model(
            inputs_embeds=inputs_embeds.detach(),
            attention_mask=attention_mask,
            labels=labels,
        ).loss
        loss.backward()
        dummy_optimizer.step()
        dummy_optimizer.zero_grad()

    del dummy_optimizer
    CrossAttentionWrapper.clear_supervisee_acts()
    torch.cuda.empty_cache()
    gc.collect()
    print("OOM preflight check complete")


if __name__ == "__main__":
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()

    dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")

    model_name = "Qwen/Qwen3-0.6B"
    num_layers = get_layer_count(model_name)

    # Hyperparameters — scale down for small GPUs
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    small_gpu = gpu_mem_gb < 40
    if small_gpu:
        print(f"Small GPU detected ({gpu_mem_gb:.0f}GB). Using reduced batch size and gradient checkpointing.")

    global_train_batch_size = 2 * world_size if small_gpu else 32
    assert global_train_batch_size % world_size == 0
    per_rank_batch_size = global_train_batch_size // world_size

    num_datapoints_per_variant = 1_000 if small_gpu else 250_000  # single + multi = 2x total

    tokenizer = load_tokenizer(model_name)

    print(f"Global batch size: {global_train_batch_size}, per-rank: {per_rank_batch_size}")

    # Build training data (rank 0 builds, others wait)
    total_datapoints = num_datapoints_per_variant * 2
    data_path = f"sft_training_data/cross_attn_pastlens_{model_name.replace('/', '_')}_{total_datapoints}.pt"
    eval_path = f"sft_training_data/cross_attn_eval_{model_name.replace('/', '_')}.pt"

    if local_rank == 0 and os.path.exists(data_path):
        print(f"Found cached training data at {data_path}, skipping generation.")
    elif local_rank == 0:
        print("Building cross-attention PastLens training data...")

        # Single-token variant
        single_params = PastLensDatasetConfig(
            min_k_activations=1,
            max_k_activations=1,
            min_k_tokens=1,
            max_k_tokens=50,
            fineweb_only=True,
        )
        single_dataset = hf_fineweb_generator(tokenizer)
        single_data = collect_cross_attn_past_lens_data(
            custom_dataset_params=single_params,
            tokenizer=tokenizer,
            dataset=single_dataset,
            num_datapoints=num_datapoints_per_variant,
            batch_size=per_rank_batch_size * 4,
        )
        print(f"Single-token variant: {len(single_data)} examples")

        # Multi-token variant
        multi_params = PastLensDatasetConfig(
            min_k_activations=1,
            max_k_activations=50,
            min_k_tokens=1,
            max_k_tokens=50,
            fineweb_only=True,
        )
        multi_dataset = hf_fineweb_generator(tokenizer)
        multi_data = collect_cross_attn_past_lens_data(
            custom_dataset_params=multi_params,
            tokenizer=tokenizer,
            dataset=multi_dataset,
            num_datapoints=num_datapoints_per_variant,
            batch_size=per_rank_batch_size * 4,
        )
        print(f"Multi-token variant: {len(multi_data)} examples")

        all_training_data = single_data + multi_data
        random.shuffle(all_training_data)

        # Save to disk for other ranks
        os.makedirs("sft_training_data", exist_ok=True)
        torch.save(
            [dp.model_dump() for dp in all_training_data],
            data_path,
        )
        print(f"Saved {len(all_training_data)} training examples to {data_path}")

        # Build eval datasets
        print("Building eval datasets...")
        eval_datasets = build_cross_attn_eval_data(tokenizer, num_eval=250)
        torch.save(
            {k: [dp.model_dump() for dp in v] for k, v in eval_datasets.items()},
            eval_path,
        )
    dist.barrier()

    # All ranks load data
    all_training_data = [CrossAttnTrainingDataPoint(**d) for d in torch.load(data_path, weights_only=False)]
    print(f"Rank {local_rank}: loaded {len(all_training_data)} training examples")

    if os.path.exists(eval_path):
        eval_raw = torch.load(eval_path, weights_only=False)
        eval_datasets = {k: [CrossAttnTrainingDataPoint(**d) for d in v] for k, v in eval_raw.items()}
    else:
        eval_datasets = {}

    model_name_str = model_name.split("/")[-1].replace(".", "_")

    cfg = SelfInterpTrainingConfig(
        model_name=model_name,
        hook_onto_layer=1,
        layer_percents=[],
        train_batch_size=per_rank_batch_size,
        eval_batch_size=per_rank_batch_size * 4,
        eval_steps=100,
        eval_on_start=True,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lr=2e-5,
        num_epochs=10,
        gradient_checkpointing=small_gpu,
        gradient_accumulation_steps=1,
        layer_subset_k=8,
        cross_attn_num_heads=16,
        cross_attn_gate_init=0.0,
        cross_attn_hidden_dim=1024,
        wandb_project="cross_attn_oracle",
        wandb_suffix=f"_cross_attn_{model_name_str}",
        save_steps=5_000,
        save_dir=os.path.join(os.environ.get("HF_HOME", "checkpoints"), "cross_attn_checkpoints"),
    )
    cfg.wandb_run_name = f"cross_attn_pastlens_{model_name_str}"
    if not cfg.save_dir.endswith(cfg.wandb_suffix):
        cfg.save_dir = f"{cfg.save_dir}{cfg.wandb_suffix}"

    print(f"Config: {asdict(cfg)}")
    print(f"Checkpoints will be saved to: {os.path.abspath(cfg.save_dir)}")

    train_cross_attn_model(
        cfg=cfg,
        training_data=all_training_data,
        eval_datasets=eval_datasets,
        tokenizer=tokenizer,
        dtype=dtype,
        device=device,
        model_kwargs={},
    )

    dist.destroy_process_group()
