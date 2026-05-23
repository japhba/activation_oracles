"""
Missing information eval — tests whether the AO uses activations vs surface tokens.

Three conditions per problem:
  A: complete prompt + neutral reasoning segment (model has all info)
  B: incomplete prompt + natural reasoning (model is confused)
  C: incomplete prompt + same neutral segment as A (same tokens, different state)

Key metric: A vs C agreement. If AO uses activations, A≠C. If tokens only, A≈C.
"""

import json
from pathlib import Path
from typing import Any

from nl_probes.base_experiment import (
    VerbalizerInputInfo,
    tokenize_chat_messages,
    compute_segment_positions,
)
from nl_probes.open_ended_eval.eval_runner import (
    STANDARD_VERBALIZER_LORAS,
    compute_binary_yes_no_metrics,
    compute_roc_curve_data,
    run_default_eval,
    run_verbalizer_binary_eval_loop,
)

DATASET_DIR = "data_pipelines/missing_info"
DATASET_FILENAME = "missing_info_eval_dataset.json"

# Not used for binary scoring (single forward pass, no generation), but required
# by VerbalizerEvalConfig. Kept minimal.
GENERATION_KWARGS: dict[str, Any] = {
    "do_sample": False,
    "max_new_tokens": 1,
}

PREFIX = "Answer with 'Yes' or 'No' only. "

# Chosen via prompt sweep (see experiments/three_lora_comparison/missing_info_prompt_sweep/
# and missing_info_sensitivity_sweep/). "incomplete_info" has the best balanced accuracy
# (73-80% pairwise) while maintaining comparable ROC AUC to "missing_info".
# "missing_info" has slightly higher AUC but is biased toward "no" — A acc ~80% but
# B/C acc ~30-40%, making the pairwise accuracy much lower than the AUC suggests.
VERBALIZER_PROMPTS: dict[str, str] = {
    "incomplete_info": f"{PREFIX}Does the model have incomplete information about this problem?",
    # "missing_info": f"{PREFIX}Is the model missing information needed to solve this problem?",
}


def dataset_path() -> str:
    return f"{DATASET_DIR}/{DATASET_FILENAME}"


def load_missing_info_dataset(
    model_name: str,
    max_entries: int | None = None,
    conditions: list[str] | None = None,
) -> list[dict[str, Any]]:
    data = json.loads(Path(dataset_path()).read_text())
    entries = data["entries"]
    if conditions is not None:
        entries = [e for e in entries if e["condition"] in conditions]
    if max_entries is not None:
        entries = entries[:max_entries]
    assert len(entries) > 0, "No entries in dataset"
    return entries


def build_missing_info_verbalizer_prompt_infos(
    entries: list[dict[str, Any]],
    verbalizer_prompts: dict[str, str],
    tokenizer,
) -> tuple[list[VerbalizerInputInfo], list[dict[str, Any]]]:
    prompt_infos: list[VerbalizerInputInfo] = []
    entry_metadata: list[dict[str, Any]] = []

    for entry in entries:
        reasoning = entry["full_reasoning"]
        teacher_forced = entry.get("teacher_forced_segment", "")
        problem = entry["problem_text"]

        if teacher_forced:
            content = f"{reasoning}\n\n{teacher_forced}"
        else:
            content = reasoning

        messages = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": content},
        ]
        token_ids = tokenize_chat_messages(
            tokenizer,
            messages,
            add_generation_prompt=False,
            continue_thinking=True,
        )
        # Segment covers exactly the neutral/teacher-forced segment per entry,
        # so conditions A and C have identical token windows.
        segment_token_count = entry["neutral_segment_token_count"]
        positions = compute_segment_positions(len(token_ids), -segment_token_count)

        # Prompts where "yes" means the model HAS info / is doing well
        # (ground truth is flipped relative to missing_info)
        INVERTED_PROMPTS = {"confident", "enough_context", "well_grounded"}

        for prompt_name, vp in verbalizer_prompts.items():
            if prompt_name in INVERTED_PROMPTS:
                ground_truth = "no" if entry["ground_truth_missing_info"] else "yes"
            else:
                ground_truth = "yes" if entry["ground_truth_missing_info"] else "no"

            prompt_infos.append(
                VerbalizerInputInfo(
                    context_token_ids=token_ids,
                    positions=positions,
                    ground_truth=ground_truth,
                    verbalizer_prompt=vp,
                )
            )
            entry_metadata.append(
                {
                    "id": entry["id"],
                    "problem_id": entry["problem_id"],
                    "condition": entry["condition"],
                    "ground_truth_missing_info": entry["ground_truth_missing_info"],
                    "prompt_name": prompt_name,
                    "missing_info_description": entry["missing_info_description"],
                }
            )

    return prompt_infos, entry_metadata


def compute_missing_info_binary_metrics(scored_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Binary metrics with pairwise A-vs-B and A-vs-C ROC AUCs."""
    metrics = compute_binary_yes_no_metrics(scored_results)

    # Pairwise ROC AUCs: A (negative class) vs B or C (positive class)
    a_results = [r for r in scored_results if r.get("condition") == "A_complete"]
    b_results = [r for r in scored_results if r.get("condition") == "B_incomplete"]
    c_results = [r for r in scored_results if r.get("condition") == "C_forced"]

    for pair_name, positive_results in [("A_vs_B", b_results), ("A_vs_C", c_results)]:
        pooled = a_results + positive_results
        if not pooled:
            continue
        roc_data = compute_roc_curve_data(
            labels=[int(r["binary_label"]) for r in pooled],
            scores=[float(r["margin_yes_minus_no"]) for r in pooled],
        )
        if roc_data is not None:
            metrics[f"{pair_name}_roc_auc"] = float(roc_data["auc"])
            metrics[f"{pair_name}_num_positive"] = int(roc_data["positives"])
            metrics[f"{pair_name}_num_negative"] = int(roc_data["negatives"])

        # Also break down by prompt
        prompt_names = sorted({r["prompt_name"] for r in pooled if "prompt_name" in r})
        for prompt_name in prompt_names:
            subset = [r for r in pooled if r.get("prompt_name") == prompt_name]
            prompt_roc = compute_roc_curve_data(
                labels=[int(r["binary_label"]) for r in subset],
                scores=[float(r["margin_yes_minus_no"]) for r in subset],
            )
            if prompt_roc is not None:
                metrics[f"{pair_name}_{prompt_name}_roc_auc"] = float(prompt_roc["auc"])

    return metrics


def run_missing_info_open_ended_eval(
    *,
    model_name: str,
    model,
    tokenizer,
    device,
    eval_batch_size: int = 32,
    generation_kwargs: dict[str, Any] | None = None,
    verbalizer_lora_paths: list[str],
    output_dir: str | None = None,
    max_entries: int | None = None,
    verbalizer_prompts: dict[str, str] | None = None,
) -> dict[str, Any]:
    if verbalizer_prompts is None:
        verbalizer_prompts = VERBALIZER_PROMPTS
    if generation_kwargs is None:
        generation_kwargs = GENERATION_KWARGS

    entries = load_missing_info_dataset(model_name, max_entries=max_entries)

    prompt_infos, entry_metadata = build_missing_info_verbalizer_prompt_infos(
        entries,
        verbalizer_prompts,
        tokenizer,
    )

    return run_verbalizer_binary_eval_loop(
        eval_name="missing_info",
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=model_name,
        eval_batch_size=eval_batch_size,
        generation_kwargs=generation_kwargs,
        prompt_infos=prompt_infos,
        entry_metadata=entry_metadata,
        num_entries=len(entries),
        verbalizer_lora_paths=verbalizer_lora_paths,
        output_dir=output_dir,
        binary_metrics_fn=compute_missing_info_binary_metrics,
    )


if __name__ == "__main__":
    run_default_eval(
        eval_name="missing_info",
        run_eval_fn=run_missing_info_open_ended_eval,
        model_name="Qwen/Qwen3-8B",
        run_eval_kwargs={
            "verbalizer_lora_paths": STANDARD_VERBALIZER_LORAS,
        },
    )
