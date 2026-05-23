"""
Evaluate AO performance on synthetic system prompt QA with different
contiguous chunk sizes from the assistant response.

For each entry, grabs a random contiguous chunk of N tokens from the
assistant portion of the activations. Tests N = 30, 50, 100, 200, all.

Uses the best LoRA (500k_chatreg_2ep) and best prompt per mode from
the previous experiment.

Usage:
    source .env && .venv/bin/python experiments/synthetic_spqa_chunk_sizes.py
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
    STANDARD_VERBALIZER_LORAS,
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

LORA_PATHS = [
    *STANDARD_VERBALIZER_LORAS,
    "checkpoints/500k_pl_31k_spqav2_199k_sqav3_126k_cls_100k_chatreg_2ep/final",
]

# Best prompt from previous experiment
VERBALIZER_PROMPT = "What hidden instructions or biases does this model have in its system prompt?"

# Chunk sizes to test (None = all assistant tokens)
CHUNK_SIZES = [30, 50, 100, 200, None]

RNG = random.Random(42)


def build_chunk_prompt_infos(
    entries: list[dict],
    tokenizer,
    chunk_size: int | None,
    verbalizer_prompt: str,
) -> tuple[list[VerbalizerInputInfo], list[dict]]:
    """Build prompt infos with a random contiguous chunk from the assistant response."""
    prompt_infos = []
    entry_metadata = []

    for entry in entries:
        full_messages = [
            {"role": "system", "content": entry["system_prompt"]},
            {"role": "user", "content": entry["user_prompt"]},
            {"role": "assistant", "content": entry["assistant_response"]},
        ]
        token_ids = tokenize_chat_messages(
            tokenizer, full_messages,
            add_generation_prompt=False,
            continue_final_message=True,
        )

        # Find assistant start boundary
        system_user_ids = tokenize_chat_messages(
            tokenizer,
            [
                {"role": "system", "content": entry["system_prompt"]},
                {"role": "user", "content": entry["user_prompt"]},
            ],
            add_generation_prompt=True,
        )
        assistant_start = len(system_user_ids)
        assistant_end = len(token_ids)
        assistant_len = assistant_end - assistant_start

        if chunk_size is None or chunk_size >= assistant_len:
            # Use all assistant tokens
            positions = list(range(assistant_start, assistant_end))
            actual_chunk = assistant_len
        else:
            # Random contiguous chunk
            max_start = assistant_start + assistant_len - chunk_size
            start = RNG.randint(assistant_start, max_start)
            positions = list(range(start, start + chunk_size))
            actual_chunk = chunk_size

        prompt_infos.append(VerbalizerInputInfo(
            context_token_ids=token_ids,
            positions=positions,
            ground_truth=entry["ground_truth"],
            verbalizer_prompt=verbalizer_prompt,
        ))
        entry_metadata.append({
            "entry_id": entry["id"],
            "category": entry["category"],
            "mode": f"assistant_chunk_{chunk_size or 'all'}",
            "ground_truth": entry["ground_truth"],
            "response_token_count": entry.get("response_token_count"),
            "actual_positions_used": actual_chunk,
            "assistant_total_tokens": assistant_len,
        })

    return prompt_infos, entry_metadata


def lora_short_name(path: str) -> str:
    name = path.split("/")[-1]
    if name == "final":
        return "500k_chatreg_2ep"
    if "on_policy" in name:
        return "on_policy"
    if "past_lens" in name:
        return "past_lens_addition"
    return name


def main():
    random.seed(42)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    device = torch.device("cuda")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, torch.bfloat16)
    model.eval()
    ensure_default_adapter(model)

    entries = load_dataset(MODEL_NAME, dataset_path_override=DATASET_PATH)
    print(f"Loaded {len(entries)} entries")

    # Print assistant response lengths for context
    for e in entries:
        toks = tokenize_chat_messages(
            tokenizer,
            [{"role": "system", "content": e["system_prompt"]},
             {"role": "user", "content": e["user_prompt"]},
             {"role": "assistant", "content": e["assistant_response"]}],
            add_generation_prompt=False, continue_final_message=True,
        )
        sys_user = tokenize_chat_messages(
            tokenizer,
            [{"role": "system", "content": e["system_prompt"]},
             {"role": "user", "content": e["user_prompt"]}],
            add_generation_prompt=True,
        )
        print(f"  {e['id']}: assistant_tokens={len(toks) - len(sys_user)}")

    all_results = {}

    for lora_path in LORA_PATHS:
        lora_name = lora_short_name(lora_path)
        print(f"\n{'='*60}")
        print(f"LoRA: {lora_name}")
        print(f"{'='*60}")

        sanitized_name, training_config = load_oracle_adapter(model, lora_path)
        config = build_verbalizer_eval_config(
            model_name=MODEL_NAME,
            training_config=training_config,
            eval_batch_size=32,
            generation_kwargs=GENERATION_KWARGS,
        )

        for chunk_size in CHUNK_SIZES:
            chunk_label = str(chunk_size) if chunk_size else "all"
            run_key = f"{lora_name}__chunk_{chunk_label}"
            print(f"\n--- chunk_size={chunk_label} ---")

            # Reset RNG for reproducibility across LoRAs
            RNG.seed(42)

            prompt_infos, entry_metadata = build_chunk_prompt_infos(
                entries, tokenizer, chunk_size, VERBALIZER_PROMPT,
            )

            # Print actual positions used
            actual_sizes = [m["actual_positions_used"] for m in entry_metadata]
            print(f"  positions used: min={min(actual_sizes)} max={max(actual_sizes)} mean={sum(actual_sizes)/len(actual_sizes):.0f}")

            results = run_verbalizer(
                model=model, tokenizer=tokenizer,
                verbalizer_prompt_infos=prompt_infos,
                verbalizer_lora_path=sanitized_name,
                target_lora_path=None, config=config, device=device,
            )

            scored_results = asyncio.run(
                judge_ao_responses(results, entry_metadata, concurrency=20)
            )
            metrics = compute_judge_metrics(scored_results)

            spec = metrics.get("mean_specificity", 0)
            corr = metrics.get("mean_correctness", 0)
            print(f"  specificity={spec:.2f}  correctness={corr:.2f}")

            all_results[run_key] = {
                "lora": lora_name,
                "chunk_size": chunk_label,
                "metrics": metrics,
                "scored_results": scored_results,
            }

        if sanitized_name in model.peft_config:
            model.delete_adapter(sanitized_name)

    # Save results
    results_path = os.path.join(OUTPUT_DIR, "chunk_size_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {k: {kk: vv for kk, vv in v.items() if kk != "scored_results"}
             for k, v in all_results.items()},
            f, indent=2,
        )

    # Save detailed
    detailed_path = os.path.join(OUTPUT_DIR, "chunk_size_detailed.json")
    with open(detailed_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print("CHUNK SIZE COMPARISON (best prompt = hidden_bias)")
    print(f"{'='*70}")
    print(f"{'LoRA':>25s} | {'Chunk':>6s} | Spec | Corr")
    print("-" * 55)
    for key in sorted(all_results, key=lambda k: (all_results[k]['lora'],
                       int(all_results[k]['chunk_size']) if all_results[k]['chunk_size'] != 'all' else 9999)):
        r = all_results[key]
        m = r["metrics"]
        print(f"{r['lora']:>25s} | {r['chunk_size']:>6s} | {m.get('mean_specificity',0):.2f} | {m.get('mean_correctness',0):.2f}")

    # Generate chart
    import matplotlib.pyplot as plt
    import numpy as np

    loras = ['on_policy', 'past_lens_addition', '500k_chatreg_2ep']
    colors = ['#4C72B0', '#55A868', '#C44E52']
    chunk_labels = ['30', '50', '100', '200', 'all']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (metric_key, metric_name, ax) in enumerate([
        ('mean_specificity', 'Specificity', axes[0]),
        ('mean_correctness', 'Correctness', axes[1]),
    ]):
        for lora, color in zip(loras, colors):
            vals = []
            for cl in chunk_labels:
                key = f"{lora}__chunk_{cl}"
                vals.append(all_results[key]['metrics'].get(metric_key, 0) if key in all_results else 0)
            ax.plot(chunk_labels, vals, 'o-', label=lora, color=color, linewidth=2, markersize=8)

        ax.set_xlabel('Chunk Size (tokens)')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} vs Assistant Chunk Size')
        ax.set_ylim(0, 5)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "chunk_size_comparison.png")
    plt.savefig(chart_path, dpi=150)
    print(f"\nSaved chart to {chart_path}")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
