"""Verify the AO training-step pattern works under Unsloth:

  per batch:
    1. with model.disable_adapter(): forward pass on context, capture activations at K layers
    2. re-enable adapter, forward+backward on training input with steering injection at layer 1
    3. optim step

This is what materialize_missing_steering_vectors + train_features_batch do back-to-back
in nl_probes/sft.py. We do them in one process, on one batch, on one model instance.

We also run the same pattern under vanilla HF+PEFT and compare:
  - extracted activations should match across backends (same base weights, same input)
  - injected loss should be in the same neighborhood

Usage:
    python extract_insert_test.py --mode hf
    python extract_insert_test.py --mode unsloth
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def _device():
    assert torch.cuda.is_available()
    return torch.device("cuda:0")


def _build_inputs(tokenizer, batch_size, ctx_len, train_len, num_steering_positions, device, seed=0):
    """Build context tokens (for extraction) and training tokens (for injection step)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    vocab = tokenizer.vocab_size

    # Context: what the target model "thinks about" — we extract activations from these.
    ctx_input_ids = torch.randint(0, vocab, (batch_size, ctx_len), generator=g, dtype=torch.long).to(device)
    ctx_attn = torch.ones_like(ctx_input_ids)
    # Positions to extract from each context (last N tokens)
    extract_positions = list(range(ctx_len - num_steering_positions, ctx_len))

    # Training input: prompt the AO sees. First num_steering_positions slots are injection points;
    # rest is the question + answer the AO learns to produce.
    train_input_ids = torch.randint(0, vocab, (batch_size, train_len), generator=g, dtype=torch.long).to(device)
    train_attn = torch.ones_like(train_input_ids)
    train_labels = train_input_ids.clone()
    train_labels[:, :num_steering_positions] = -100  # don't train on the injection slots
    inject_positions = list(range(num_steering_positions))

    return {
        "ctx_input_ids": ctx_input_ids,
        "ctx_attn": ctx_attn,
        "extract_positions": extract_positions,
        "train_input_ids": train_input_ids,
        "train_attn": train_attn,
        "train_labels": train_labels,
        "inject_positions": inject_positions,
    }


def _load_hf(model_name, dtype, lora_r, lora_alpha):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, low_cpu_mem_usage=True).to(_device())
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
        target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    return model, tok


def _load_unsloth(model_name, max_seq_length, lora_r, lora_alpha):
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=max_seq_length, dtype=None,
        load_in_4bit=False, load_in_8bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", use_gradient_checkpointing=False, random_state=42, use_rslora=False,
    )
    return model, tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["hf", "unsloth"], required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--ctx-len", type=int, default=128)
    p.add_argument("--train-len", type=int, default=128)
    p.add_argument("--num-steering-positions", type=int, default=10)
    p.add_argument("--extract-layers", type=int, nargs="+", default=[7, 14, 21])
    p.add_argument("--hook-layer", type=int, default=1)
    p.add_argument("--steering-coef", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--num-steps", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.mode == "unsloth":
        import unsloth  # noqa  -- must precede transformers/peft

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from nl_probes.utils.activation_utils import (
        EarlyStopException, get_hf_submodule, collect_activations_multiple_layers,
    )
    from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook

    torch.manual_seed(0)
    device = _device()
    dtype = torch.bfloat16

    print(f"[{args.mode}] loading {args.model}")
    t0 = time.time()
    if args.mode == "hf":
        model, tok = _load_hf(args.model, dtype, args.lora_r, args.lora_alpha)
    else:
        model, tok = _load_unsloth(args.model, args.ctx_len + args.train_len + 64, args.lora_r, args.lora_alpha)
    load_s = time.time() - t0
    print(f"[{args.mode}] loaded in {load_s:.1f}s")

    inputs = _build_inputs(
        tok, args.batch_size, args.ctx_len, args.train_len,
        args.num_steering_positions, device,
    )
    inject_layer_module = get_hf_submodule(model, args.hook_layer)
    extract_submodules = {L: get_hf_submodule(model, L, use_lora=True) for L in args.extract_layers}
    print(f"[{args.mode}] inject submodule: {type(inject_layer_module).__name__}; "
          f"extract submodules layers={list(extract_submodules.keys())}")

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)

    # Probe: verify disable_adapter works on this backend
    print(f"[{args.mode}] sanity: enabled vs disabled forward differs...")
    model.eval()
    with torch.no_grad():
        in1 = {"input_ids": inputs["ctx_input_ids"], "attention_mask": inputs["ctx_attn"]}
        loss_on = model(**in1, labels=inputs["ctx_input_ids"]).loss.item()
        with model.disable_adapter():
            loss_off = model(**in1, labels=inputs["ctx_input_ids"]).loss.item()
    print(f"[{args.mode}]   loss_adapter_on={loss_on:.4f} loss_adapter_off={loss_off:.4f}")
    # adapters were init at 0 init for LoRA-A and randn for LoRA-B by default; PEFT
    # default init_lora_weights=True => LoRA-A gauss, LoRA-B zero, so on=off at init.
    # We just want to confirm the toggle doesn't crash. Both should be finite.
    assert torch.isfinite(torch.tensor(loss_on)) and torch.isfinite(torch.tensor(loss_off))

    timings = []
    extracted_norms_per_step = []
    losses = []

    for step in range(args.num_steps):
        torch.cuda.synchronize(); t_step = time.time()

        # ---- EXTRACT (adapter disabled) ----
        was_training = model.training
        model.eval()
        with model.disable_adapter():
            t_ext = time.time()
            acts_by_layer = collect_activations_multiple_layers(
                model=model,
                submodules=extract_submodules,
                inputs_BL={"input_ids": inputs["ctx_input_ids"],
                           "attention_mask": inputs["ctx_attn"]},
                min_offset=None, max_offset=None,
            )
            torch.cuda.synchronize(); t_ext_s = time.time() - t_ext
        if was_training:
            model.train()

        # Slice to extract_positions, build per-batch steering vectors at each chosen layer.
        # For the AO training step we typically pick ONE layer per item; we'll just use
        # the middle extract layer here for the injection.
        chosen_layer = args.extract_layers[len(args.extract_layers) // 2]
        acts_BLD = acts_by_layer[chosen_layer]
        # acts_BLD: (B, ctx_len, d). Take the last N positions per item.
        steering_vectors = [
            acts_BLD[b, inputs["extract_positions"], :].detach().to(dtype)
            for b in range(args.batch_size)
        ]
        positions = [list(inputs["inject_positions"]) for _ in range(args.batch_size)]
        norms = [v.norm(dim=-1).mean().item() for v in steering_vectors]
        extracted_norms_per_step.append(norms)

        # ---- INJECT + train step (adapter enabled) ----
        model.train()
        t_inj = time.time()
        hook = get_hf_activation_steering_hook(
            vectors=steering_vectors, positions=positions,
            steering_coefficient=args.steering_coef, device=device, dtype=dtype,
        )
        with add_hook(inject_layer_module, hook):
            loss = model(input_ids=inputs["train_input_ids"],
                         attention_mask=inputs["train_attn"],
                         labels=inputs["train_labels"]).loss
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t_inj_s = time.time() - t_inj
        t_total = time.time() - t_step

        timings.append({"step": step, "extract_s": t_ext_s, "inject_s": t_inj_s,
                        "total_s": t_total, "loss": loss.item()})
        losses.append(loss.item())
        print(f"[{args.mode}] step {step}: extract={t_ext_s:.3f}s inject+bwd={t_inj_s:.3f}s "
              f"total={t_total:.3f}s loss={loss.item():.4f} mean_extract_norm={sum(norms)/len(norms):.3f}")

    # Verify gradients only on LoRA params (no leakage to base)
    base_grad_count = 0
    lora_grad_count = 0
    for n, p in model.named_parameters():
        if p.grad is not None:
            if "lora" in n.lower():
                lora_grad_count += 1
            else:
                base_grad_count += 1
    print(f"[{args.mode}] params with grads after step: lora={lora_grad_count} base={base_grad_count}")

    # Stable summary
    out = {
        "mode": args.mode,
        "model": args.model,
        "batch_size": args.batch_size,
        "ctx_len": args.ctx_len,
        "train_len": args.train_len,
        "extract_layers": args.extract_layers,
        "hook_layer": args.hook_layer,
        "load_s": load_s,
        "timings": timings,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "extracted_norms_per_step": extracted_norms_per_step,
        "lora_grad_params": lora_grad_count,
        "base_grad_params": base_grad_count,
    }
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
