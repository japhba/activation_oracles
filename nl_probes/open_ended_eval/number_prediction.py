import json
import re
from pathlib import Path
from typing import Any

from nl_probes.base_experiment import (
    VerbalizerInputInfo,
    VerbalizerResults,
    compute_segment_positions,
    tokenize_chat_messages,
)
from nl_probes.open_ended_eval.eval_runner import (
    BaselinePromptVariant,
    STANDARD_VERBALIZER_LORAS,
    TextBaselineGenerationResult,
    TextBaselineInput,
    decode_contiguous_span_text,
    get_first_ao_response,
    render_baseline_chat_prompt,
    run_default_eval,
    run_default_text_baseline_eval,
    run_text_baseline_eval_loop,
    run_verbalizer_generation_eval_loop,
)

DATASET_DIR = "data_pipelines/number_prediction"
DATASET_FILENAME = "number_prediction_eval_dataset.json"

# Short answers (single number)
GENERATION_KWARGS: dict[str, Any] = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 20,
}

VERBALIZER_PROMPTS: dict[str, str] = {
    "direct": "What number is the model about to output?",
    "planning": "What number is the model planning to answer with?",
    "predict": "Predict the exact numerical answer the model will give.",
}
TEXT_BASELINE_VARIANTS: tuple[BaselinePromptVariant, ...] = ("full_context", "selected_span")


def number_prediction_user_prompt(expression: str) -> str:
    return f"What is {expression}? Answer with just the number, nothing else."


def dataset_path(model_name: str) -> str:
    model_dir = model_name.split("/")[-1]
    return f"{DATASET_DIR}/{model_dir}/{DATASET_FILENAME}"


def load_number_prediction_dataset(
    model_name: str,
    max_entries: int | None = None,
    categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    data = json.loads(Path(dataset_path(model_name)).read_text())
    entries = data["entries"]
    if categories is not None:
        entries = [e for e in entries if e["category"] in categories]
    if max_entries is not None:
        entries = entries[:max_entries]
    assert len(entries) > 0, "No entries in dataset"
    return entries


def build_number_prediction_verbalizer_prompt_infos(
    entries: list[dict[str, Any]],
    verbalizer_prompts: dict[str, str],
    tokenizer,
    segment_start: int = -10,
) -> tuple[list[VerbalizerInputInfo], list[dict[str, Any]]]:
    prompt_infos: list[VerbalizerInputInfo] = []
    entry_metadata: list[dict[str, Any]] = []

    for entry in entries:
        messages = [
            {"role": "user", "content": number_prediction_user_prompt(entry["expression"])},
        ]
        token_ids = tokenize_chat_messages(tokenizer, messages)
        positions = compute_segment_positions(len(token_ids), segment_start)

        for prompt_name, vp in verbalizer_prompts.items():
            prompt_infos.append(
                VerbalizerInputInfo(
                    context_token_ids=token_ids,
                    positions=positions,
                    ground_truth=str(entry["model_answer"]),
                    verbalizer_prompt=vp,
                )
            )
            entry_metadata.append(
                {
                    "id": entry["id"],
                    "expression": entry["expression"],
                    "category": entry["category"],
                    "true_answer": entry["true_answer"],
                    "model_answer": entry["model_answer"],
                    "model_correct": entry["model_correct"],
                    "is_single_token_answer": entry["is_single_token_answer"],
                    "prompt_name": prompt_name,
                }
            )

    return prompt_infos, entry_metadata


def build_number_prediction_text_baseline_inputs(
    entries: list[dict[str, Any]],
    verbalizer_prompts: dict[str, str],
    tokenizer,
    segment_start: int = -10,
    variants: tuple[BaselinePromptVariant, ...] = TEXT_BASELINE_VARIANTS,
) -> tuple[list[TextBaselineInput], list[dict[str, Any]]]:
    baseline_inputs: list[TextBaselineInput] = []
    entry_metadata: list[dict[str, Any]] = []

    for entry in entries:
        original_prompt = number_prediction_user_prompt(entry["expression"])
        messages = [{"role": "user", "content": original_prompt}]
        token_ids = tokenize_chat_messages(tokenizer, messages)
        positions = compute_segment_positions(len(token_ids), segment_start)
        selected_span_text = decode_contiguous_span_text(token_ids, positions, tokenizer)

        for prompt_name, verbalizer_prompt in verbalizer_prompts.items():
            for variant in variants:
                if variant == "full_context":
                    user_content = f"{original_prompt}\nQuestion: {verbalizer_prompt}"
                elif variant == "selected_span":
                    user_content = (
                        f"{selected_span_text}\n"
                        "Question: This is only a selected token span from the model's input. "
                        f"Based only on this span, make your best guess: {verbalizer_prompt}"
                    )
                else:
                    raise ValueError(f"Unsupported baseline variant: {variant}")

                baseline_inputs.append(
                    TextBaselineInput(
                        prompt_text=render_baseline_chat_prompt(
                            tokenizer=tokenizer,
                            messages=[{"role": "user", "content": user_content}],
                            add_generation_prompt=True,
                            enable_thinking=False,
                        ),
                        ground_truth=str(entry["model_answer"]),
                        prompt_name=prompt_name,
                        variant=variant,
                    )
                )
                entry_metadata.append(
                    {
                        "id": entry["id"],
                        "expression": entry["expression"],
                        "category": entry["category"],
                        "true_answer": entry["true_answer"],
                        "model_answer": entry["model_answer"],
                        "model_correct": entry["model_correct"],
                        "is_single_token_answer": entry["is_single_token_answer"],
                        "prompt_name": prompt_name,
                        "baseline_variant": variant,
                    }
                )

    return baseline_inputs, entry_metadata


def extract_number_from_response(response: str) -> int | None:
    text = response.strip()
    match = re.search(r"-?\d+", text)
    if match:
        return int(match.group())
    return None


def _score_response_texts(
    response_texts: list[str | None],
    entry_metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scored = []

    for response_text, meta in zip(response_texts, entry_metadata, strict=True):
        if response_text is None:
            continue

        predicted_number = extract_number_from_response(response_text)

        scored.append(
            {
                **meta,
                "response_text": response_text,
                "predicted_number": predicted_number,
                "matches_model_answer": predicted_number == meta["model_answer"],
                "matches_true_answer": predicted_number == meta["true_answer"],
            }
        )

    return scored


def score_results(
    results: list[VerbalizerResults],
    entry_metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _score_response_texts(
        [get_first_ao_response(result) for result in results],
        entry_metadata,
    )


def score_text_baseline_results(
    results: list[TextBaselineGenerationResult],
    entry_metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _score_response_texts(
        [result.response_text for result in results],
        entry_metadata,
    )


def compute_metrics(scored_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(scored_results)
    matches_model = sum(1 for r in scored_results if r["matches_model_answer"])
    matches_true = sum(1 for r in scored_results if r["matches_true_answer"])
    has_number = sum(1 for r in scored_results if r["predicted_number"] is not None)

    metrics: dict[str, Any] = {
        "total": total,
        "matches_model_answer": matches_model,
        "matches_model_answer_rate": matches_model / total if total > 0 else 0,
        "matches_true_answer": matches_true,
        "matches_true_answer_rate": matches_true / total if total > 0 else 0,
        "has_number_rate": has_number / total if total > 0 else 0,
    }

    # Break down by category
    categories = set(r["category"] for r in scored_results)
    for cat in sorted(categories):
        cat_results = [r for r in scored_results if r["category"] == cat]
        cat_total = len(cat_results)
        cat_model = sum(1 for r in cat_results if r["matches_model_answer"])
        cat_true = sum(1 for r in cat_results if r["matches_true_answer"])
        metrics[f"cat_{cat}_total"] = cat_total
        metrics[f"cat_{cat}_model_match_rate"] = cat_model / cat_total if cat_total > 0 else 0
        metrics[f"cat_{cat}_true_match_rate"] = cat_true / cat_total if cat_total > 0 else 0

    # Break down by single vs multi token
    single = [r for r in scored_results if r["is_single_token_answer"]]
    multi = [r for r in scored_results if not r["is_single_token_answer"]]
    if single:
        metrics["single_token_model_match_rate"] = sum(1 for r in single if r["matches_model_answer"]) / len(single)
    if multi:
        metrics["multi_token_model_match_rate"] = sum(1 for r in multi if r["matches_model_answer"]) / len(multi)

    # Break down by prompt
    prompts = set(r["prompt_name"] for r in scored_results)
    for prompt_name in sorted(prompts):
        p_results = [r for r in scored_results if r["prompt_name"] == prompt_name]
        p_total = len(p_results)
        p_model = sum(1 for r in p_results if r["matches_model_answer"])
        metrics[f"prompt_{prompt_name}_model_match_rate"] = p_model / p_total if p_total > 0 else 0

    return metrics


def print_sample_results(scored_results: list[dict[str, Any]]) -> None:
    print(f"\n  Sample results:")
    for r in scored_results[:15]:
        status = "✓" if r["matches_model_answer"] else "✗"
        variant_str = f" variant={r['baseline_variant']:<13}" if "baseline_variant" in r else ""
        print(
            f"    {status} {r['expression']:<35} model={r['model_answer']:<8} "
            f"pred={r['predicted_number']!s:<8} prompt={r['prompt_name']:<10}{variant_str} "
            f"raw={r['response_text'][:60]}"
        )


def run_number_prediction_open_ended_eval(
    *,
    model_name: str,
    model,
    tokenizer,
    device,
    eval_batch_size: int = 64,
    generation_kwargs: dict[str, Any] | None = None,
    verbalizer_lora_paths: list[str],
    output_dir: str | None = None,
    max_entries: int | None = None,
    categories: list[str] | None = None,
    verbalizer_prompts: dict[str, str] | None = None,
    segment_tokens: int = 10,
) -> dict[str, Any]:
    if verbalizer_prompts is None:
        verbalizer_prompts = VERBALIZER_PROMPTS
    if generation_kwargs is None:
        generation_kwargs = GENERATION_KWARGS

    entries = load_number_prediction_dataset(model_name, max_entries=max_entries, categories=categories)
    prompt_infos, entry_metadata = build_number_prediction_verbalizer_prompt_infos(
        entries,
        verbalizer_prompts,
        tokenizer,
        segment_start=-segment_tokens,
    )

    return run_verbalizer_generation_eval_loop(
        eval_name="number_prediction",
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=model_name,
        eval_batch_size=eval_batch_size,
        generation_kwargs=generation_kwargs,
        prompt_infos=prompt_infos,
        entry_metadata=entry_metadata,
        score_fn=score_results,
        metrics_fn=compute_metrics,
        num_entries=len(entries),
        verbalizer_lora_paths=verbalizer_lora_paths,
        output_dir=output_dir,
        print_sample_fn=print_sample_results,
    )


def run_number_prediction_text_baseline_eval(
    *,
    model_name: str,
    tokenizer,
    backend,
    hf_model=None,
    device=None,
    eval_batch_size: int = 64,
    generation_kwargs: dict[str, Any] | None = None,
    output_dir: str | None = None,
    max_entries: int | None = None,
    categories: list[str] | None = None,
    verbalizer_prompts: dict[str, str] | None = None,
    segment_tokens: int = 10,
    variants: tuple[BaselinePromptVariant, ...] = TEXT_BASELINE_VARIANTS,
) -> dict[str, Any]:
    if verbalizer_prompts is None:
        verbalizer_prompts = VERBALIZER_PROMPTS
    if generation_kwargs is None:
        generation_kwargs = GENERATION_KWARGS

    entries = load_number_prediction_dataset(model_name, max_entries=max_entries, categories=categories)
    baseline_inputs, entry_metadata = build_number_prediction_text_baseline_inputs(
        entries=entries,
        verbalizer_prompts=verbalizer_prompts,
        tokenizer=tokenizer,
        segment_start=-segment_tokens,
        variants=variants,
    )

    return run_text_baseline_eval_loop(
        eval_name="number_prediction",
        baseline_inputs=baseline_inputs,
        entry_metadata=entry_metadata,
        backend=backend,
        inference_mode="rollout",
        model_name=model_name,
        tokenizer=tokenizer,
        generation_kwargs=generation_kwargs,
        score_fn=score_text_baseline_results,
        metrics_fn=compute_metrics,
        output_dir=output_dir,
        hf_model=hf_model,
        device=device,
        eval_batch_size=eval_batch_size,
        num_entries=len(entries),
        print_sample_fn=print_sample_results,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["ao", "text_baseline"],
        default="ao",
    )
    parser.add_argument(
        "--baseline-backend",
        choices=["hf", "vllm"],
        default="vllm",
    )
    parser.add_argument(
        "--baseline-variant",
        choices=["all", "full_context", "selected_span"],
        default="all",
    )
    parser.add_argument(
        "--segment-tokens",
        type=int,
        default=10,
        help="Number of tokens from end of context to use as segment (default: 10)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Path to checkpoint directory (overrides STANDARD_VERBALIZER_LORAS)",
    )
    parser.add_argument("--max-entries", type=int, default=None)
    args = parser.parse_args()

    verbalizer_loras = STANDARD_VERBALIZER_LORAS
    if args.checkpoint_dir:
        verbalizer_loras = [args.checkpoint_dir]

    if args.mode == "ao":
        run_default_eval(
            eval_name=f"number_prediction_seg{args.segment_tokens}",
            run_eval_fn=run_number_prediction_open_ended_eval,
            model_name="Qwen/Qwen3-8B",
            run_eval_kwargs={
                "verbalizer_lora_paths": verbalizer_loras,
                "segment_tokens": args.segment_tokens,
                "max_entries": args.max_entries,
            },
        )
    else:
        variants = TEXT_BASELINE_VARIANTS if args.baseline_variant == "all" else (args.baseline_variant,)
        run_default_text_baseline_eval(
            eval_name=f"number_prediction_text_baseline_seg{args.segment_tokens}",
            run_eval_fn=run_number_prediction_text_baseline_eval,
            model_name="Qwen/Qwen3-8B",
            backend=args.baseline_backend,
            run_eval_kwargs={
                "segment_tokens": args.segment_tokens,
                "max_entries": args.max_entries,
                "variants": variants,
            },
        )
