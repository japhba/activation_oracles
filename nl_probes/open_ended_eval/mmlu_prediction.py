import json
import os
from pathlib import Path
from typing import Any

from nl_probes.base_experiment import (
    VerbalizerInputInfo,
    tokenize_chat_messages,
    compute_segment_positions,
)
import nl_probes.base_experiment as base_experiment
from nl_probes.open_ended_eval.eval_runner import (
    BaselinePromptVariant,
    STANDARD_VERBALIZER_LORAS,
    _load_adapter_and_build_config,
    build_yes_no_candidate_token_groups,
    compute_binary_yes_no_metrics,
    decode_contiguous_span_text,
    ensure_default_adapter,
    render_baseline_chat_prompt,
    run_default_eval,
    run_default_text_baseline_eval,
    run_text_baseline_eval_loop,
    run_verbalizer_binary_eval_loop,
    score_binary_yes_no_results,
    TextBaselineInput,
)

DATASET_DIR = "data_pipelines/mmlu_prediction"
DATASET_FILENAME = "mmlu_prediction_eval_dataset.json"

# Not used for binary scoring (single forward pass, no generation), but required
# by VerbalizerEvalConfig. Kept minimal.
GENERATION_KWARGS: dict[str, Any] = {
    "do_sample": False,
    "max_new_tokens": 1,
}

ANSWER_LETTERS = ["A", "B", "C", "D"]


PREFIX = "Answer with 'Yes' or 'No' only. "

# AO prompts keep the original "the model" wording because the verbalizer
# checkpoints were trained with that framing.
AO_PRE_ANSWER_PROMPTS: dict[str, str] = {
    "pre_likely_correct": f"{PREFIX}Will the model answer this correctly?",
}

AO_POST_ANSWER_PROMPTS: dict[str, str] = {
    "post_likely_correct": f"{PREFIX}Is the model's answer likely to be correct?",
}

AO_VERBALIZER_PROMPTS: dict[str, str] = {**AO_PRE_ANSWER_PROMPTS, **AO_POST_ANSWER_PROMPTS}

# Text baseline prompts use second-person wording so the model is asked about its
# own likely correctness rather than about another model.
TEXT_BASELINE_PRE_ANSWER_PROMPTS: dict[str, str] = {
    "pre_likely_correct": f"{PREFIX}Will you likely answer this correctly?",
}

TEXT_BASELINE_POST_ANSWER_PROMPTS: dict[str, str] = {
    "post_likely_correct": f"{PREFIX}Is your answer likely to be correct?",
}

TEXT_BASELINE_VERBALIZER_PROMPTS: dict[str, str] = {
    **TEXT_BASELINE_PRE_ANSWER_PROMPTS,
    **TEXT_BASELINE_POST_ANSWER_PROMPTS,
}
TEXT_BASELINE_VARIANTS: tuple[BaselinePromptVariant, ...] = ("full_context", "selected_span")


def format_mmlu_question(question: str, choices: list[str]) -> str:
    """Format an MMLU question with lettered choices (must match generate_dataset.py)."""
    lines = [question, ""]
    for i, choice in enumerate(choices):
        lines.append(f"{ANSWER_LETTERS[i]}. {choice}")
    lines.append("")
    lines.append("Answer with just the letter (A, B, C, or D), nothing else.")
    return "\n".join(lines)


def dataset_path(model_name: str) -> str:
    model_dir = model_name.split("/")[-1]
    return f"{DATASET_DIR}/{model_dir}/{DATASET_FILENAME}"


def load_mmlu_prediction_dataset(
    model_name: str,
    max_entries: int | None = None,
) -> list[dict[str, Any]]:
    data = json.loads(Path(dataset_path(model_name)).read_text())
    entries = data["entries"]
    if max_entries is not None:
        entries = entries[:max_entries]
    assert len(entries) > 0, "No entries in dataset"
    return entries


def _build_mmlu_pre_answer_context(
    question_text: str,
    tokenizer,
    segment_start: int,
) -> tuple[list[int], list[int]]:
    token_ids = tokenize_chat_messages(tokenizer, [{"role": "user", "content": question_text}])
    positions = compute_segment_positions(len(token_ids), segment_start)
    return token_ids, positions


def _build_mmlu_post_answer_context(
    question_text: str,
    model_answer_letter: str,
    tokenizer,
    segment_start: int,
) -> tuple[list[int], list[int]]:
    token_ids = tokenize_chat_messages(
        tokenizer,
        [
            {"role": "user", "content": question_text},
            {"role": "assistant", "content": model_answer_letter},
        ],
        add_generation_prompt=False,
    )
    positions = compute_segment_positions(len(token_ids), segment_start)
    return token_ids, positions


def build_mmlu_prediction_verbalizer_prompt_infos(
    entries: list[dict[str, Any]],
    verbalizer_prompts: dict[str, str],
    tokenizer,
    segment_start: int = -10,
) -> tuple[list[VerbalizerInputInfo], list[dict[str, Any]]]:
    prompt_infos: list[VerbalizerInputInfo] = []
    entry_metadata: list[dict[str, Any]] = []

    for entry in entries:
        question_text = format_mmlu_question(entry["question"], entry["choices"])
        ground_truth = "yes" if entry["model_correct"] else "no"
        pre_answer_token_ids, pre_answer_positions = _build_mmlu_pre_answer_context(
            question_text,
            tokenizer,
            segment_start,
        )
        post_answer_token_ids, post_answer_positions = _build_mmlu_post_answer_context(
            question_text,
            entry["model_answer_letter"],
            tokenizer,
            segment_start,
        )

        for prompt_name, vp in verbalizer_prompts.items():
            is_post_answer = prompt_name.startswith("post_")
            token_ids = post_answer_token_ids if is_post_answer else pre_answer_token_ids
            positions = post_answer_positions if is_post_answer else pre_answer_positions

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
                    "subject": entry["subject"],
                    "correct_answer_letter": entry["correct_answer_letter"],
                    "model_answer_letter": entry["model_answer_letter"],
                    "model_correct": entry["model_correct"],
                    "prompt_name": prompt_name,
                }
            )

    return prompt_infos, entry_metadata


def build_mmlu_prediction_text_baseline_inputs(
    entries: list[dict[str, Any]],
    verbalizer_prompts: dict[str, str],
    tokenizer,
    segment_start: int = -10,
    variants: tuple[BaselinePromptVariant, ...] = TEXT_BASELINE_VARIANTS,
) -> tuple[list[TextBaselineInput], list[dict[str, Any]]]:
    baseline_inputs: list[TextBaselineInput] = []
    entry_metadata: list[dict[str, Any]] = []

    for entry in entries:
        question_text = format_mmlu_question(entry["question"], entry["choices"])
        ground_truth = "yes" if entry["model_correct"] else "no"
        pre_answer_token_ids, pre_answer_positions = _build_mmlu_pre_answer_context(
            question_text,
            tokenizer,
            segment_start,
        )
        post_answer_token_ids, post_answer_positions = _build_mmlu_post_answer_context(
            question_text,
            entry["model_answer_letter"],
            tokenizer,
            segment_start,
        )
        pre_answer_selected_span_text = decode_contiguous_span_text(
            pre_answer_token_ids,
            pre_answer_positions,
            tokenizer,
        )
        post_answer_selected_span_text = decode_contiguous_span_text(
            post_answer_token_ids,
            post_answer_positions,
            tokenizer,
        )

        for prompt_name, verbalizer_prompt in verbalizer_prompts.items():
            is_post_answer = prompt_name.startswith("post_")

            if is_post_answer:
                selected_span_text = post_answer_selected_span_text
            else:
                selected_span_text = pre_answer_selected_span_text

            for variant in variants:
                if variant == "full_context":
                    if is_post_answer:
                        prompt_text = render_baseline_chat_prompt(
                            tokenizer=tokenizer,
                            messages=[
                                {"role": "user", "content": question_text},
                                {"role": "assistant", "content": entry["model_answer_letter"]},
                                {"role": "user", "content": verbalizer_prompt},
                            ],
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    else:
                        prompt_text = render_baseline_chat_prompt(
                            tokenizer=tokenizer,
                            messages=[
                                {"role": "user", "content": question_text},
                                {"role": "assistant", "content": "Let me think about this."},
                                {"role": "user", "content": verbalizer_prompt},
                            ],
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                elif variant == "selected_span":
                    prompt_text = render_baseline_chat_prompt(
                        tokenizer=tokenizer,
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    f"{selected_span_text}\n"
                                    "Question: This is only a selected token span from the model's input. "
                                    f"Based only on this span, answer with 'Yes' or 'No' only: {verbalizer_prompt}"
                                ),
                            }
                        ],
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                else:
                    raise ValueError(f"Unsupported baseline variant: {variant}")

                baseline_inputs.append(
                    TextBaselineInput(
                        prompt_text=prompt_text,
                        ground_truth=ground_truth,
                        prompt_name=prompt_name,
                        variant=variant,
                    )
                )
                entry_metadata.append(
                    {
                        "id": entry["id"],
                        "subject": entry["subject"],
                        "correct_answer_letter": entry["correct_answer_letter"],
                        "model_answer_letter": entry["model_answer_letter"],
                        "model_correct": entry["model_correct"],
                        "prompt_name": prompt_name,
                        "baseline_variant": variant,
                        "is_post_answer": is_post_answer,
                    }
                )

    return baseline_inputs, entry_metadata


LETTER_PREDICTION_PROMPT = "Predict the model's answer. Output only A, B, C, or D."


def run_letter_prediction(
    *,
    model,
    tokenizer,
    device,
    model_name: str,
    eval_batch_size: int,
    generation_kwargs: dict[str, Any],
    entries: list[dict[str, Any]],
    verbalizer_lora_paths: list[str],
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Predict which answer letter (A-D) the model will choose, using argmax token.

    Uses pre-answer context only (question, no model answer). Checks if the
    AO's argmax output token matches the model's actual answer letter.
    """
    prompt_infos, entry_metadata = build_mmlu_prediction_verbalizer_prompt_infos(
        entries,
        {"predict_letter": LETTER_PREDICTION_PROMPT},
        tokenizer,
    )

    # We need candidate_token_groups to call run_verbalizer_binary_score,
    # but we only care about the argmax token — the yes/no scores are ignored.
    dummy_candidate_groups = build_yes_no_candidate_token_groups(tokenizer)

    ensure_default_adapter(model)
    model.eval()

    results_by_verbalizer: dict[str, dict[str, Any]] = {}

    for verbalizer_entry in verbalizer_lora_paths:
        sanitized_name, config = _load_adapter_and_build_config(
            model, verbalizer_entry, model_name, eval_batch_size, generation_kwargs,
        )

        binary_results = base_experiment.run_verbalizer_binary_score(
            model=model,
            tokenizer=tokenizer,
            verbalizer_prompt_infos=prompt_infos,
            verbalizer_lora_path=sanitized_name,
            target_lora_path=None,
            config=config,
            device=device,
            candidate_token_groups=dummy_candidate_groups,
        )

        scored_results = []
        matches_model = 0
        matches_true = 0
        total = 0
        for result, meta in zip(binary_results, entry_metadata):
            predicted_letter = result.argmax_token_text.strip()
            parseable = predicted_letter in ANSWER_LETTERS
            if parseable:
                total += 1
                if predicted_letter == meta["model_answer_letter"]:
                    matches_model += 1
                if predicted_letter == meta["correct_answer_letter"]:
                    matches_true += 1
            scored_results.append({
                "predicted_letter": predicted_letter,
                "model_answer_letter": meta["model_answer_letter"],
                "correct_answer_letter": meta["correct_answer_letter"],
                "matches_model": parseable and predicted_letter == meta["model_answer_letter"],
                "matches_true": parseable and predicted_letter == meta["correct_answer_letter"],
                "parseable": parseable,
            })

        verbalizer_key = verbalizer_entry.split("/")[-1]
        lora_name = verbalizer_key.replace("/", "_").replace(".", "_")
        metrics = {
            "total": total,
            "matches_model_rate": matches_model / total if total else 0,
            "matches_true_rate": matches_true / total if total else 0,
            "unparseable": len(binary_results) - total,
        }
        results_by_verbalizer[verbalizer_key] = metrics
        print(f"\n  Letter prediction for {verbalizer_key}:")
        print(f"    matches_model: {matches_model}/{total} = {metrics['matches_model_rate']:.3f}")
        print(f"    matches_true:  {matches_true}/{total} = {metrics['matches_true_rate']:.3f}")

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"mmlu_letter_prediction_{lora_name}.json")
            with open(output_path, "w") as f:
                json.dump({
                    "verbalizer": verbalizer_entry,
                    "metrics": metrics,
                    "scored_results": scored_results,
                }, f, indent=2)
            print(f"  Saved letter prediction to {output_path}")

        if sanitized_name in model.peft_config:
            model.delete_adapter(sanitized_name)

    return {"letter_prediction_by_verbalizer": results_by_verbalizer}


def run_mmlu_prediction_open_ended_eval(
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
    verbalizer_prompts: dict[str, str] | None = None,
    run_letter_prediction_eval: bool = False,
    segment_tokens: int = 10,
) -> dict[str, Any]:
    if verbalizer_prompts is None:
        verbalizer_prompts = AO_VERBALIZER_PROMPTS
    if generation_kwargs is None:
        generation_kwargs = GENERATION_KWARGS

    entries = load_mmlu_prediction_dataset(model_name, max_entries=max_entries)
    prompt_infos, entry_metadata = build_mmlu_prediction_verbalizer_prompt_infos(
        entries,
        verbalizer_prompts,
        tokenizer,
        segment_start=-segment_tokens,
    )

    summary = run_verbalizer_binary_eval_loop(
        eval_name="mmlu_prediction",
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
    )

    if run_letter_prediction_eval:
        letter_output_dir = os.path.join(output_dir, "letter_prediction") if output_dir else None
        letter_results = run_letter_prediction(
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_name=model_name,
            eval_batch_size=eval_batch_size,
            generation_kwargs=generation_kwargs,
            entries=entries,
            verbalizer_lora_paths=verbalizer_lora_paths,
            output_dir=letter_output_dir,
        )
        summary.update(letter_results)

    return summary


def run_mmlu_prediction_text_baseline_eval(
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
    verbalizer_prompts: dict[str, str] | None = None,
    segment_tokens: int = 10,
    variants: tuple[BaselinePromptVariant, ...] = TEXT_BASELINE_VARIANTS,
) -> dict[str, Any]:
    assert backend == "hf", "mmlu_prediction text baseline uses HF binary scoring"
    if verbalizer_prompts is None:
        verbalizer_prompts = TEXT_BASELINE_VERBALIZER_PROMPTS
    if generation_kwargs is None:
        generation_kwargs = GENERATION_KWARGS

    entries = load_mmlu_prediction_dataset(model_name, max_entries=max_entries)
    baseline_inputs, entry_metadata = build_mmlu_prediction_text_baseline_inputs(
        entries=entries,
        verbalizer_prompts=verbalizer_prompts,
        tokenizer=tokenizer,
        segment_start=-segment_tokens,
        variants=variants,
    )

    return run_text_baseline_eval_loop(
        eval_name="mmlu_prediction",
        baseline_inputs=baseline_inputs,
        entry_metadata=entry_metadata,
        backend=backend,
        inference_mode="binary_yes_no",
        model_name=model_name,
        tokenizer=tokenizer,
        generation_kwargs=generation_kwargs,
        score_fn=score_binary_yes_no_results,
        metrics_fn=compute_binary_yes_no_metrics,
        output_dir=output_dir,
        hf_model=hf_model,
        device=device,
        eval_batch_size=eval_batch_size,
        num_entries=len(entries),
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
        "--eval-mode",
        choices=["all", "pre_answer", "post_answer"],
        default="all",
    )
    parser.add_argument(
        "--baseline-variant",
        choices=["all", "full_context", "selected_span"],
        default="all",
    )
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--segment-tokens", type=int, default=10)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Path to checkpoint directory (overrides STANDARD_VERBALIZER_LORAS)",
    )
    args = parser.parse_args()

    eval_modes = {
        "pre_answer": {
            "ao": AO_PRE_ANSWER_PROMPTS,
            "text_baseline": TEXT_BASELINE_PRE_ANSWER_PROMPTS,
        },
        "post_answer": {
            "ao": AO_POST_ANSWER_PROMPTS,
            "text_baseline": TEXT_BASELINE_POST_ANSWER_PROMPTS,
        },
    }
    if args.eval_mode == "all":
        selected_eval_modes = eval_modes
    else:
        selected_eval_modes = {args.eval_mode: eval_modes[args.eval_mode]}

    verbalizer_loras = STANDARD_VERBALIZER_LORAS
    if args.checkpoint_dir:
        verbalizer_loras = [args.checkpoint_dir]

    if args.mode == "ao":
        for mode_name, prompt_sets in selected_eval_modes.items():
            run_default_eval(
                eval_name=f"mmlu_prediction_{mode_name}_seg{args.segment_tokens}",
                run_eval_fn=run_mmlu_prediction_open_ended_eval,
                model_name="Qwen/Qwen3-8B",
                run_eval_kwargs={
                    "verbalizer_lora_paths": verbalizer_loras,
                    "verbalizer_prompts": prompt_sets["ao"],
                    "max_entries": args.max_entries,
                    "segment_tokens": args.segment_tokens,
                },
            )
    else:
        variants = TEXT_BASELINE_VARIANTS if args.baseline_variant == "all" else (args.baseline_variant,)
        for mode_name, prompt_sets in selected_eval_modes.items():
            run_default_text_baseline_eval(
                eval_name=f"mmlu_prediction_text_baseline_{mode_name}_seg{args.segment_tokens}",
                run_eval_fn=run_mmlu_prediction_text_baseline_eval,
                model_name="Qwen/Qwen3-8B",
                backend="hf",
                run_eval_kwargs={
                    "verbalizer_prompts": prompt_sets["text_baseline"],
                    "max_entries": args.max_entries,
                    "segment_tokens": args.segment_tokens,
                    "variants": variants,
                },
            )
