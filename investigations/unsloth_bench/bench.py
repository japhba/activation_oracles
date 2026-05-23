"""Benchmark Unsloth vs vanilla HF+PEFT on the activation-oracle training step.

Single GPU. Loads Qwen3-8B with either backend, attaches the project's residual-stream
steering hook on cfg.hook_onto_layer, runs N forward+backward steps on a fixed
synthetic batch, reports wall-time per step, throughput, peak memory, and loss
trajectory.

Usage:
    python bench.py --mode hf      --model Qwen/Qwen3-8B
    python bench.py --mode unsloth --model Qwen/Qwen3-8B

Run both, then compare. Each invocation lives in its own process so memory state
doesn't bleed across modes.
"""

# Unsloth must be imported before transformers/peft when --mode unsloth.
# We do a deferred import inside main() based on the chosen mode.

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def _device() -> torch.device:
    assert torch.cuda.is_available(), "CUDA required"
    return torch.device("cuda:0")


def _build_synthetic_batch(
    tokenizer,
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_steering_positions: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 0,
):
    """Random token IDs + steering vectors. Loss values won't be meaningful but
    timing and gradient flow will be."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len), generator=g, dtype=torch.long
    ).to(device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    # ignore the first num_steering_positions positions in loss (they're "context")
    labels[:, :num_steering_positions] = -100

    g2 = torch.Generator(device="cpu").manual_seed(seed + 1)
    steering_vectors = [
        torch.randn(num_steering_positions, d_model, generator=g2, dtype=dtype).to(device)
        for _ in range(batch_size)
    ]
    positions = [list(range(num_steering_positions)) for _ in range(batch_size)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "steering_vectors": steering_vectors,
        "positions": positions,
    }


def _load_hf(model_name, dtype, lora_r, lora_alpha, lora_dropout, attn_impl, grad_checkpoint):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = dict(dtype=dtype, low_cpu_mem_usage=True)
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model = model.to(_device())

    if grad_checkpoint:
        model.enable_input_require_grads()
        model.use_cache = False
        # transformers 5.x hardcodes use_reentrant=False. The project's in-place
        # steering hook trips non-reentrant ckpt's tensor-count check, so force True.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )

    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.train()
    return model, tokenizer


def _load_unsloth(model_name, max_seq_length, lora_r, lora_alpha, lora_dropout, grad_checkpoint):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=max_seq_length, dtype=None,
        load_in_4bit=False, load_in_8bit=False,
    )
    # Unsloth's recommended grad-checkpointing flag string. Bool-False disables.
    gc_arg = "unsloth" if grad_checkpoint else False
    model = FastLanguageModel.get_peft_model(
        model, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", use_gradient_checkpointing=gc_arg, random_state=42, use_rslora=False,
    )
    model.train()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hf", "unsloth"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--num-steering-positions", type=int, default=10)
    parser.add_argument("--hook-layer", type=int, default=1)
    parser.add_argument("--steering-coef", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--attn-impl", default=None,
                        help="HF attn_implementation, e.g. flash_attention_2 or sdpa")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # Critical: when --mode unsloth, we MUST import unsloth before transformers/peft.
    if args.mode == "unsloth":
        import unsloth  # noqa: F401

    # Now safe to import project hooks (which import torch and re-export utils).
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook
    from nl_probes.utils.activation_utils import get_hf_submodule

    torch.manual_seed(args.seed)
    device = _device()
    dtype = torch.bfloat16

    print(f"[{args.mode}] loading {args.model}")
    t0 = time.time()
    if args.mode == "hf":
        model, tokenizer = _load_hf(
            args.model, dtype, args.lora_r, args.lora_alpha, args.lora_dropout,
            attn_impl=args.attn_impl, grad_checkpoint=args.gradient_checkpointing,
        )
    else:
        model, tokenizer = _load_unsloth(
            args.model, max_seq_length=args.seq_len + 64,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            grad_checkpoint=args.gradient_checkpointing,
        )
    load_seconds = time.time() - t0
    print(f"[{args.mode}] loaded in {load_seconds:.1f}s")

    # Pull the same residual layer in both backends; PEFT/Unsloth wrap is handled
    # by get_hf_submodule (it recognises PeftModel and the qwen path).
    submodule = get_hf_submodule(model, args.hook_layer)
    print(f"[{args.mode}] hooking submodule: {type(submodule).__name__}")

    # Hook-fire counter (sanity check the hook actually runs).
    fire_counter = {"n": 0}
    orig_get_hook = get_hf_activation_steering_hook

    def get_hook_with_counter(*a, **kw):
        inner = orig_get_hook(*a, **kw)
        def wrapped(module, _input, output):
            fire_counter["n"] += 1
            return inner(module, _input, output)
        return wrapped

    d_model = model.config.hidden_size
    batch = _build_synthetic_batch(
        tokenizer,
        args.batch_size,
        args.seq_len,
        d_model,
        args.num_steering_positions,
        device,
        dtype,
        seed=args.seed,
    )

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    # Verify hook actually changes loss vs. no-hook (gradient flow + hook firing)
    print(f"[{args.mode}] sanity: comparing loss with/without hook...")
    model.eval()
    with torch.no_grad():
        loss_no_hook = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        ).loss.item()
        hook_fn = get_hook_with_counter(
            vectors=batch["steering_vectors"],
            positions=batch["positions"],
            steering_coefficient=args.steering_coef,
            device=device,
            dtype=dtype,
        )
        with add_hook(submodule, hook_fn):
            loss_with_hook = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            ).loss.item()
    print(
        f"[{args.mode}] loss_no_hook={loss_no_hook:.4f} "
        f"loss_with_hook={loss_with_hook:.4f} "
        f"hook_fires_in_eval={fire_counter['n']}"
    )
    assert abs(loss_with_hook - loss_no_hook) > 1e-6, "Hook did not change loss"
    assert fire_counter["n"] >= 1, "Hook did not fire"
    fire_counter["n"] = 0
    model.train()

    torch.cuda.reset_peak_memory_stats()
    losses = []

    print(f"[{args.mode}] warmup ({args.warmup_steps} steps)...")
    for _ in range(args.warmup_steps):
        hook_fn = get_hook_with_counter(
            vectors=batch["steering_vectors"],
            positions=batch["positions"],
            steering_coefficient=args.steering_coef,
            device=device,
            dtype=dtype,
        )
        with add_hook(submodule, hook_fn):
            loss = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            ).loss
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    fires_after_warmup = fire_counter["n"]
    print(f"[{args.mode}] hook fired {fires_after_warmup} times in warmup "
          f"(expected >= {args.warmup_steps})")

    print(f"[{args.mode}] measuring ({args.measured_steps} steps)...")
    torch.cuda.synchronize()
    t_start = time.time()
    for step in range(args.measured_steps):
        hook_fn = get_hook_with_counter(
            vectors=batch["steering_vectors"],
            positions=batch["positions"],
            steering_coefficient=args.steering_coef,
            device=device,
            dtype=dtype,
        )
        with add_hook(submodule, hook_fn):
            loss = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            ).loss
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        losses.append(loss.item())
    torch.cuda.synchronize()
    t_end = time.time()

    elapsed = t_end - t_start
    tokens_per_step = args.batch_size * args.seq_len
    total_tokens = tokens_per_step * args.measured_steps
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    s_per_step = elapsed / args.measured_steps
    tokens_per_sec = total_tokens / elapsed

    result = {
        "mode": args.mode,
        "model": args.model,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "lora_r": args.lora_r,
        "hook_layer": args.hook_layer,
        "gradient_checkpointing": args.gradient_checkpointing,
        "attn_impl": args.attn_impl,
        "load_seconds": load_seconds,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "elapsed_seconds": elapsed,
        "seconds_per_step": s_per_step,
        "tokens_per_step": tokens_per_step,
        "tokens_per_second": tokens_per_sec,
        "peak_memory_gb": peak_mem_gb,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "loss_no_hook_eval": loss_no_hook,
        "loss_with_hook_eval": loss_with_hook,
        "hook_fires_total": fire_counter["n"],
    }
    print(json.dumps(result, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[{args.mode}] wrote {out_path}")


if __name__ == "__main__":
    main()
