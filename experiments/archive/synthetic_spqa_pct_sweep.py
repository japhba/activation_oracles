"""
Sweep what percentage of assistant tokens are used (from the start).

Tests 10%, 20%, ..., 100% of assistant response tokens.
Also includes system_only and user_only as reference.

Usage:
    source .env && .venv/bin/python experiments/synthetic_spqa_pct_sweep.py
"""

import asyncio
import json
import logging
import os
import random

import torch

logging.getLogger("asyncio").setLevel(logging.CRITICAL)

from nl_probes.base_experiment import (
    VerbalizerInputInfo,
    load_oracle_adapter,
    run_verbalizer,
    tokenize_chat_messages,
)
from nl_probes.open_ended_eval.eval_runner import (
    build_verbalizer_eval_config,
    ensure_default_adapter,
)
from nl_probes.open_ended_eval.system_prompt_qa import (
    GENERATION_KWARGS,
    compute_judge_metrics,
    judge_ao_responses,
    load_dataset,
)
from nl_probes.utils.common import load_model, load_tokenizer

MODEL_NAME = "Qwen/Qwen3-8B"
DATASET_PATH = "data_pipelines/system_prompt_qa/synthetic_eval_dataset.json"
OUTPUT_DIR = "experiments/synthetic_spqa_eval_results"

LORA_PATH = "checkpoints/500k_pl_31k_spqav2_199k_sqav3_126k_cls_100k_chatreg_2ep/final"
VERBALIZER_PROMPT = "What hidden instructions or biases does this model have in its system prompt?"

PERCENTAGES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def get_boundaries(entry, tokenizer):
    full_messages = [
        {"role": "system", "content": entry["system_prompt"]},
        {"role": "user", "content": entry["user_prompt"]},
        {"role": "assistant", "content": entry["assistant_response"]},
    ]
    token_ids = tokenize_chat_messages(
        tokenizer, full_messages,
        add_generation_prompt=False, continue_final_message=True,
    )
    sys_ids = tokenize_chat_messages(
        tokenizer,
        [{"role": "system", "content": entry["system_prompt"]}],
        add_generation_prompt=False,
    )
    sys_user_ids = tokenize_chat_messages(
        tokenizer,
        [{"role": "system", "content": entry["system_prompt"]},
         {"role": "user", "content": entry["user_prompt"]}],
        add_generation_prompt=True,
    )
    return token_ids, len(sys_ids), len(sys_user_ids)


def build_pct_prompt_infos(entries, tokenizer, pct):
    """Use the first pct% of assistant tokens."""
    prompt_infos = []
    entry_metadata = []

    for entry in entries:
        token_ids, sys_end, asst_start = get_boundaries(entry, tokenizer)
        asst_len = len(token_ids) - asst_start

        n_tokens = max(1, int(asst_len * pct / 100))
        positions = list(range(asst_start, asst_start + n_tokens))

        prompt_infos.append(VerbalizerInputInfo(
            context_token_ids=token_ids,
            positions=positions,
            ground_truth=entry["ground_truth"],
            verbalizer_prompt=VERBALIZER_PROMPT,
        ))
        entry_metadata.append({
            "entry_id": entry["id"],
            "category": entry["category"],
            "mode": f"assistant_first_{pct}pct",
            "ground_truth": entry["ground_truth"],
            "positions_used": len(positions),
            "assistant_total": asst_len,
        })

    return prompt_infos, entry_metadata


def build_region_prompt_infos(entries, tokenizer, region):
    prompt_infos = []
    entry_metadata = []

    for entry in entries:
        token_ids, sys_end, asst_start = get_boundaries(entry, tokenizer)

        if region == "system_only":
            positions = list(range(0, sys_end))
        elif region == "user_only":
            positions = list(range(sys_end, asst_start))
        else:
            raise ValueError(region)

        if not positions:
            continue

        prompt_infos.append(VerbalizerInputInfo(
            context_token_ids=token_ids,
            positions=positions,
            ground_truth=entry["ground_truth"],
            verbalizer_prompt=VERBALIZER_PROMPT,
        ))
        entry_metadata.append({
            "entry_id": entry["id"],
            "category": entry["category"],
            "mode": region,
            "ground_truth": entry["ground_truth"],
            "positions_used": len(positions),
        })

    return prompt_infos, entry_metadata


def main():
    random.seed(42)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    device = torch.device("cuda")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, torch.bfloat16)
    model.eval()
    ensure_default_adapter(model)

    entries = load_dataset(MODEL_NAME, dataset_path_override=DATASET_PATH)
    print(f"Loaded {len(entries)} entries")

    sanitized_name, training_config = load_oracle_adapter(model, LORA_PATH)
    config = build_verbalizer_eval_config(
        model_name=MODEL_NAME,
        training_config=training_config,
        eval_batch_size=32,
        generation_kwargs=GENERATION_KWARGS,
    )

    all_results = {}

    def run_eval(label, prompt_infos, entry_metadata):
        avg_pos = sum(len(p.positions) for p in prompt_infos) / len(prompt_infos)
        print(f"\n--- {label} ({avg_pos:.0f} avg positions) ---")
        results = run_verbalizer(
            model=model, tokenizer=tokenizer,
            verbalizer_prompt_infos=prompt_infos,
            verbalizer_lora_path=sanitized_name,
            target_lora_path=None, config=config, device=device,
        )
        scored = asyncio.run(judge_ao_responses(results, entry_metadata, concurrency=20))
        metrics = compute_judge_metrics(scored)
        spec = metrics.get("mean_specificity", 0)
        corr = metrics.get("mean_correctness", 0)
        print(f"  specificity={spec:.2f}  correctness={corr:.2f}")
        all_results[label] = {"metrics": metrics}

    # Reference
    for region in ["system_only", "user_only"]:
        pi, em = build_region_prompt_infos(entries, tokenizer, region)
        run_eval(region, pi, em)

    # Percentage sweep
    for pct in PERCENTAGES:
        pi, em = build_pct_prompt_infos(entries, tokenizer, pct)
        run_eval(f"assistant_first_{pct}pct", pi, em)

    if sanitized_name in model.peft_config:
        model.delete_adapter(sanitized_name)

    # Save
    results_path = os.path.join(OUTPUT_DIR, "pct_sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print
    print(f"\n{'='*55}")
    print("PERCENTAGE SWEEP (500k_chatreg_2ep)")
    print(f"{'='*55}")
    print(f"{'Region':>30s} | Spec | Corr")
    print("-" * 55)
    for label, r in all_results.items():
        m = r["metrics"]
        print(f"{label:>30s} | {m.get('mean_specificity',0):.2f} | {m.get('mean_correctness',0):.2f}")

    # Chart
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    pct_specs = [all_results[f"assistant_first_{p}pct"]["metrics"]["mean_specificity"] for p in PERCENTAGES]
    pct_corrs = [all_results[f"assistant_first_{p}pct"]["metrics"]["mean_correctness"] for p in PERCENTAGES]

    ax.plot(PERCENTAGES, pct_specs, 'o-', color='#C44E52', linewidth=2, markersize=8, label='Specificity')
    ax.plot(PERCENTAGES, pct_corrs, 's-', color='#4C72B0', linewidth=2, markersize=8, label='Correctness')

    # Reference lines
    sys_spec = all_results["system_only"]["metrics"]["mean_specificity"]
    sys_corr = all_results["system_only"]["metrics"]["mean_correctness"]
    ax.axhline(y=sys_spec, linestyle='--', color='gray', alpha=0.6, label=f'System only (spec={sys_spec:.2f})')
    ax.axhline(y=sys_corr, linestyle=':', color='gray', alpha=0.6, label=f'System only (corr={sys_corr:.2f})')

    user_spec = all_results["user_only"]["metrics"]["mean_specificity"]
    ax.axhline(y=user_spec, linestyle='--', color='orange', alpha=0.6, label=f'User only (spec={user_spec:.2f})')

    ax.set_xlabel('% of Assistant Response (from start)')
    ax.set_ylabel('Score (1-5)')
    ax.set_title('How much of the assistant response does the AO need?\n(500k_chatreg_2ep, first N% of tokens)')
    ax.set_ylim(0, 5)
    ax.set_xlim(5, 105)
    ax.set_xticks(PERCENTAGES)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "pct_sweep.png")
    plt.savefig(chart_path, dpi=150)
    print(f"\nSaved chart to {chart_path}")


if __name__ == "__main__":
    main()
