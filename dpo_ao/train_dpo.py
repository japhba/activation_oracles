#!/usr/bin/env python3
"""Custom DPO loop for the Activation Oracle.

DPO loss with activation injection at every forward pass.

For each (prompt, chosen, rejected) pair:
  1. Encode `context` with base (no LoRA) → cache activation at (last-token, layer).
  2. Build AO prompt: "Layer: L\\n ?\\n\\nQ: <template question>\\nA: "
  3. Forward student (base+LoRA) on (AO prompt + chosen) with activation hooked → log p_θ(chosen)
  4. Same for rejected: log p_θ(rejected)
  5. Forward reference (base+LoRA frozen) same way → log p_ref(chosen), log p_ref(rejected)
  6. DPO loss: -log σ(β · [(log p_θ(c) - log p_θ(r)) - (log p_ref(c) - log p_ref(r))])

Usage:
  python train_dpo.py \\
      --init /workspace/checkpoints/ao_q3_8b_v3_multi5_sonnet_lr1em5/token_budget_50M \\
      --data data/dpo_v1.jsonl \\
      --layer 21 \\
      --beta 0.1 --lr 5e-7 --steps 200 \\
      --out checkpoints/dpo_v1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SPECIAL_TOKEN = " ?"


def build_ao_prompt(layer: int, question: str) -> str:
    # Match the AO's exact prefix template: "Layer: L\n ? \n" then question
    return f"Layer: {layer}\n{SPECIAL_TOKEN} \nQ: {question}\nA: "


def hook_inject(model, target_layer_idx: int, act_vector: torch.Tensor, special_positions: torch.Tensor):
    """Norm-matched injection hook matching nl_probes/utils/steering_hooks.py."""
    submodule = (
        model.base_model.model.model.layers[target_layer_idx]
        if hasattr(model, "base_model")
        else model.model.layers[target_layer_idx]
    )
    normed = F.normalize(act_vector, dim=-1).detach()
    if normed.dim() == 1:
        normed = normed.unsqueeze(0)

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        B, L, D = h.shape
        if L <= 1:  # decode step — never touch
            return output
        pos = special_positions.to(h.device)
        if pos.numel() == 0 or int(pos.max()) >= L:
            return output
        orig = h[0, pos, :]
        norms = orig.norm(dim=-1, keepdim=True)
        steered = (normed.to(h.dtype) * norms).to(h.dtype)
        h[0, pos, :] = steered.detach() + orig
        return (h, *output[1:]) if is_tuple else h

    return submodule.register_forward_hook(hook)


def logp_of_response(model, tok, prompt_ids: torch.Tensor, response_ids: torch.Tensor,
                     hook_layer: int, act: torch.Tensor, special_pos: torch.Tensor) -> torch.Tensor:
    """Return sum log p_θ(response | prompt, activation_injected)."""
    full = torch.cat([prompt_ids, response_ids], dim=0).unsqueeze(0)
    handle = hook_inject(model, hook_layer, act, special_pos)
    try:
        out = model(input_ids=full, use_cache=False)
    finally:
        handle.remove()
    logits = out.logits[0, prompt_ids.shape[0] - 1 : full.shape[1] - 1, :]
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(1, response_ids.unsqueeze(-1)).squeeze(-1).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--init", required=True, help="AO LoRA dir to initialize student+reference from")
    ap.add_argument("--data", required=True, help="jsonl with {prompt, chosen, rejected, context}")
    ap.add_argument("--layer", type=int, default=21, help="Activation extraction layer (matches the AO's training layer)")
    ap.add_argument("--hook-layer", type=int, default=1, help="Layer to inject activation in AO (matches AO config)")
    ap.add_argument("--max-ctx-tokens", type=int, default=1500)
    ap.add_argument("--max-resp-tokens", type=int, default=80)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base {args.base}...")
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    base.eval()
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    sp_id = tok.encode(SPECIAL_TOKEN, add_special_tokens=False)[0]
    print(f"special token id: {sp_id} ({tok.decode([sp_id])!r})")

    print(f"Attaching student LoRA from {args.init}")
    student = PeftModel.from_pretrained(base, args.init, adapter_name="student", is_trainable=True)
    print("Attaching reference LoRA (frozen)")
    student.load_adapter(args.init, adapter_name="reference", is_trainable=False)
    student.set_adapter("student")  # restore student as active after load_adapter switched to reference
    # Explicitly mark student LoRA params trainable; reference LoRA params stay frozen
    for name, p in student.named_parameters():
        if "lora_" in name and "student" in name:
            p.requires_grad = True
        elif "lora_" in name and "reference" in name:
            p.requires_grad = False
    student.train()

    trainable = [p for p in student.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}  ({len(trainable)} tensors)")
    if not trainable:
        raise RuntimeError("No trainable params! Check adapter naming convention.")
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    pairs = [json.loads(line) for line in open(args.data)]
    print(f"loaded {len(pairs)} DPO pairs")

    metrics = []
    skipped = 0
    for step in range(args.steps):
        p = pairs[step % len(pairs)]
        ctx_ids = tok(p["context"], return_tensors="pt", add_special_tokens=False,
                      truncation=True, max_length=args.max_ctx_tokens).input_ids[0].cuda()
        if ctx_ids.shape[0] < 4:
            skipped += 1
            if step < 3: print(f"[step {step}] SKIP (ctx_ids.shape={ctx_ids.shape}, ctx_len={len(p['context'])})")
            continue
        if step == 0:
            print(f"[step 0] ctx_ids.shape={ctx_ids.shape}, ctx_len={len(p['context'])}")

        # 1) Extract activation
        with torch.no_grad():
            with student.disable_adapter():
                out = student(input_ids=ctx_ids.unsqueeze(0), output_hidden_states=True, use_cache=False)
            act = out.hidden_states[args.layer][0, -1].detach().clone()

        # 2) AO prompt
        prompt_text = build_ao_prompt(args.layer, p["prompt"])
        prompt_ids = tok(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0].cuda()
        special_pos = (prompt_ids == sp_id).nonzero(as_tuple=True)[0]
        if special_pos.numel() == 0:
            skipped += 1
            if step < 3: print(f"[step {step}] SKIP no special_pos in prompt_ids (prompt={prompt_text!r}, ids[:20]={prompt_ids[:20].tolist()})")
            continue
        if step == 0:
            print(f"[step 0] special_pos={special_pos.tolist()}, prompt_ids.shape={prompt_ids.shape}")

        chosen_ids = tok(p["chosen"], return_tensors="pt", add_special_tokens=False,
                         truncation=True, max_length=args.max_resp_tokens).input_ids[0].cuda()
        rejected_ids = tok(p["rejected"], return_tensors="pt", add_special_tokens=False,
                            truncation=True, max_length=args.max_resp_tokens).input_ids[0].cuda()

        # 3) Student logprobs (with grad)
        student.set_adapter("student")
        student.train()
        logp_s_chosen = logp_of_response(student, tok, prompt_ids, chosen_ids, args.hook_layer, act, special_pos)
        logp_s_rejected = logp_of_response(student, tok, prompt_ids, rejected_ids, args.hook_layer, act, special_pos)

        # 4) Reference logprobs (no grad)
        student.set_adapter("reference")
        with torch.no_grad():
            logp_r_chosen = logp_of_response(student, tok, prompt_ids, chosen_ids, args.hook_layer, act, special_pos)
            logp_r_rejected = logp_of_response(student, tok, prompt_ids, rejected_ids, args.hook_layer, act, special_pos)
        student.set_adapter("student")

        # 5) DPO loss
        logits_diff = (logp_s_chosen - logp_s_rejected) - (logp_r_chosen - logp_r_rejected)
        loss = -F.logsigmoid(args.beta * logits_diff)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()

        if step % args.log_every == 0:
            print(f"[step {step:>4}] template={p['template']:<22} loss={loss.item():.4f}  "
                  f"Δlogp_s={ (logp_s_chosen-logp_s_rejected).item():+.2f}  "
                  f"Δlogp_r={ (logp_r_chosen-logp_r_rejected).item():+.2f}")

        metrics.append({"step": step, "loss": loss.item(),
                        "template": p["template"],
                        "delta_s": (logp_s_chosen - logp_s_rejected).item(),
                        "delta_r": (logp_r_chosen - logp_r_rejected).item()})

        if (step + 1) % args.save_every == 0:
            print(f"saving to {out_dir} ...")
            student.save_pretrained(out_dir, selected_adapters=["student"])

    student.save_pretrained(out_dir, selected_adapters=["student"])
    Path(out_dir / "dpo_metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics))
    print(f"final ckpt -> {out_dir}   metrics={len(metrics)}  skipped={skipped}")


if __name__ == "__main__":
    main()
