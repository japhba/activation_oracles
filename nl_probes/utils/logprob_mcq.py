"""Generation-free MCQ-style eval for AO models — ablation variant.

For each held-out validation item we compute log p(target | prompt) under the
SAME prompt twice:
  * with_acts:    the AO steering hook is active (real cot_prefix activations
                  injected at the special-token positions)
  * without_acts: the steering hook is OFF (special-token positions retain
                  the model's natural residual for the literal " ?" tokens)

We then form

    p_correct = sigmoid(logp_with - logp_without)
              = softmax([logp_with, logp_without])[0]

interpretable as: under flat priors over "activations explain the target"
vs "they don't", the posterior probability that activation injection helped.
0.5 = no effect, 1 = decisive, 0 = injection actively hurt.

This is generation-free (one forward pass per arm, no autoregressive decode)
and bypasses peft's load_adapter codepath.
"""

from __future__ import annotations

import contextlib
import math
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from peft import PeftModel
from transformers import AutoModelForCausalLM, PreTrainedTokenizer

from nl_probes.utils.dataset_utils import (
    BatchData,
    TrainingDataPoint,
    construct_batch,
    materialize_missing_steering_vectors,
)
from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook


def _per_sample_sum_logprob(
    *,
    cfg,
    batch: BatchData,
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    inject_acts: bool,
) -> torch.Tensor:
    """Return per-sample sum-logprob over the labeled (target) tokens.

    inject_acts=True: register the AO steering hook (real activations).
    inject_acts=False: skip the hook entirely; special-token residuals keep
    their natural model values. LoRA stays active in either case.
    """
    if inject_acts:
        hook_fn = get_hf_activation_steering_hook(
            vectors=batch.steering_vectors,
            positions=batch.positions,
            steering_coefficient=cfg.steering_coefficient,
            device=device,
            dtype=dtype,
        )
        hook_ctx = add_hook(submodule, hook_fn)
    else:
        hook_ctx = contextlib.nullcontext()

    with hook_ctx:
        outputs = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=None,
        )
    logits = outputs.logits  # (B, L, V)
    labels = batch.labels  # (B, L)

    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    per_token_neg_logp = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.shape)
    mask = (shift_labels != -100).float()
    sum_logprob = -(per_token_neg_logp * mask).sum(dim=1)
    return sum_logprob


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@torch.no_grad()
def run_logprob_mcq_eval(
    *,
    cfg,
    component_validation_data: dict[str, list[TrainingDataPoint]],
    model: PeftModel,
    submodule: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
    rng: random.Random,
) -> None:
    """Run ablation MCQ eval: p_correct = sigmoid(logp_with − logp_without)."""
    if not component_validation_data:
        return

    model_was_training = model.training
    model.eval()

    try:
        wandb_metrics: dict[str, float] = {}

        for component_name, eval_data in component_validation_data.items():
            if not eval_data:
                continue

            logp_with: list[float] = []
            logp_without: list[float] = []

            bs = max(1, cfg.train_batch_size)
            for start in range(0, len(eval_data), bs):
                chunk = eval_data[start : start + bs]
                # Materialize once; both arms reuse the same prompt + labels +
                # context. The cached steering_vectors persist across eval
                # calls, so subsequent calls are cheap.
                chunk = materialize_missing_steering_vectors(chunk, tokenizer, model)
                eval_data[start : start + len(chunk)] = chunk

                batch = construct_batch(chunk, tokenizer, device)

                lw = _per_sample_sum_logprob(
                    cfg=cfg, batch=batch, model=model, submodule=submodule,
                    device=device, dtype=dtype, inject_acts=True,
                )
                lwo = _per_sample_sum_logprob(
                    cfg=cfg, batch=batch, model=model, submodule=submodule,
                    device=device, dtype=dtype, inject_acts=False,
                )
                logp_with.extend(lw.tolist())
                logp_without.extend(lwo.tolist())

            n = len(logp_with)
            p_corrects = [_sigmoid(w - wo) for w, wo in zip(logp_with, logp_without)]
            mean_p_correct = sum(p_corrects) / n
            # Per-sample then mean — log is non-linear so log(mean(p)) != mean(log(p)).
            # log_loss is what you want plotted in log-log against
            # examples_seen / FLOPs to read off a scaling-law exponent.
            mean_log_loss = sum(-math.log(max(1e-10, p)) for p in p_corrects) / n  # nats

            wandb_metrics[f"mcq/{component_name}/p_correct"] = mean_p_correct
            wandb_metrics[f"mcq/{component_name}/log_loss"] = mean_log_loss
            wandb_metrics[f"mcq/{component_name}/n"] = n

            print(
                f"[mcq:{component_name}] step={global_step} "
                f"p_correct={mean_p_correct:.3f} log_loss={mean_log_loss:.3f} "
                f"(n={n})"
            )

        if wandb_metrics:
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            wandb_metrics["train/examples_seen"] = (
                (global_step + 1) * cfg.train_batch_size * world_size
            )
            wandb.log(wandb_metrics, step=global_step)
            wandb.summary.update(wandb_metrics)
    finally:
        if model_was_training:
            model.train()
