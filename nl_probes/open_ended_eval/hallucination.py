"""Hallucination detection eval: predict whether a factual span is supported or not.

The AO reads activations from the target model as it processes a response containing
annotated factual claims. For each claim span, the AO predicts whether the claim is
"Supported" (yes) or "Not Supported" (no).

Context: user prompt + assistant response (full conversation), with activations
extracted from the token positions corresponding to the annotated span.

Dataset: longfact-augmented-annotations (per-span factuality labels).
"""

import json
import os
from pathlib import Path
from typing import Any

from nl_probes.base_experiment import (
    VerbalizerInputInfo,
    tokenize_chat_messages,
)
from nl_probes.open_ended_eval.eval_runner import (
    STANDARD_VERBALIZER_LORAS,
    TextBaselineInput,
    compute_binary_yes_no_metrics,
    render_baseline_chat_prompt,
    run_default_eval,
    run_default_text_baseline_eval,
    run_text_baseline_eval_loop,
    run_verbalizer_binary_eval_loop,
    score_binary_yes_no_results,
)

DATASET_DIR = "data_pipelines/hallucination"
DATASET_FILENAME = "hallucination_eval_dataset.json"

# Not used for binary scoring (single forward pass, no generation), but required
# by VerbalizerEvalConfig. Kept minimal.
GENERATION_KWARGS: dict[str, Any] = {
    "do_sample": False,
    "max_new_tokens": 1,
}

PREFIX = "Answer with 'Yes' or 'No' only. "

AO_PROMPTS: dict[str, str] = {
    "supported_factual": f"{PREFIX}Is this claim factually correct?",
    "supported_accurate": f"{PREFIX}Is this statement accurate?",
    "supported_verified": f"{PREFIX}Has the model stated a verified fact here?",
    "confident_correct": f"{PREFIX}Is the model confident this information is correct?",
    "hallucinating": f"{PREFIX}Is the model hallucinating here?",
}


def dataset_path(config: str) -> str:
    return f"{DATASET_DIR}/{config}/{DATASET_FILENAME}"


def load_hallucination_dataset(
    config: str,
    max_entries: int | None = None,
    entry_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    data = json.loads(Path(dataset_path(config)).read_text())
    entries = data["entries"]
    if entry_ids is not None:
        id_set = set(entry_ids)
        entries = [e for e in entries if e["id"] in id_set]
        assert len(entries) == len(id_set), (
            f"Found {len(entries)} entries but expected {len(id_set)} from entry_ids"
        )
    elif max_entries is not None:
        entries = entries[:max_entries]
    assert len(entries) > 0, "No entries in dataset"
    return entries


def _map_span_to_token_positions(
    response_text: str,
    span_char_start: int,
    span_char_end: int,
    tokenizer,
) -> list[int]:
    """Map character-level span boundaries to token indices in the response.

    Uses the tokenizer's offset_mapping to find tokens that overlap with the span.
    Returns token indices relative to the response-only tokenization (no chat template).
    """
    encoding = tokenizer(response_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    span_token_indices = [
        i for i, (s, e) in enumerate(offsets)
        if e > span_char_start and s < span_char_end
    ]
    assert len(span_token_indices) > 0, (
        f"No tokens found for span chars [{span_char_start}:{span_char_end}] "
        f"in response of {len(response_text)} chars"
    )
    return span_token_indices


def _build_hallucination_context(
    user_prompt: str,
    response_text: str,
    span_char_start: int,
    span_char_end: int,
    tokenizer,
) -> tuple[list[int], list[int]]:
    """Build context tokens and span positions for a hallucination eval entry.

    Truncates the response at the span end — with causal attention, tokens after
    the span don't affect activations at the span positions, so they're just
    wasted compute.

    Tokenizes response_text[:span_char_end] alone to get char→token mapping,
    then builds the full chat-template context (user + truncated assistant) and
    locates the response tokens within it.
    """
    truncated_response = response_text[:span_char_end]

    # Tokenize truncated response to get char→token offsets
    response_encoding = tokenizer(
        truncated_response, return_offsets_mapping=True, add_special_tokens=False,
    )
    response_token_ids = response_encoding["input_ids"]
    response_offsets = response_encoding["offset_mapping"]

    # Build full chat context with truncated response
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": truncated_response},
    ]
    context_token_ids = tokenize_chat_messages(
        tokenizer, messages,
        add_generation_prompt=False,
        continue_final_message=True,
    )

    # Find where response tokens start in the full chat-template sequence
    response_start_in_full = None
    search_len = len(response_token_ids)
    for start_pos in range(len(context_token_ids) - search_len + 1):
        if context_token_ids[start_pos : start_pos + search_len] == response_token_ids:
            response_start_in_full = start_pos
            break

    assert response_start_in_full is not None, (
        "Could not locate response tokens in full tokenized context. "
        "Chat template may be modifying token boundaries."
    )

    # Map span chars to response token indices, then offset to full context
    span_token_indices_in_response = [
        i for i, (s, e) in enumerate(response_offsets)
        if e > span_char_start and s < span_char_end
    ]
    assert len(span_token_indices_in_response) > 0, (
        f"No tokens found for span chars [{span_char_start}:{span_char_end}]"
    )

    positions = [response_start_in_full + i for i in span_token_indices_in_response]
    return context_token_ids, positions


def build_hallucination_verbalizer_prompt_infos(
    entries: list[dict[str, Any]],
    verbalizer_prompts: dict[str, str],
    tokenizer,
) -> tuple[list[VerbalizerInputInfo], list[dict[str, Any]]]:
    prompt_infos: list[VerbalizerInputInfo] = []
    entry_metadata: list[dict[str, Any]] = []

    for entry in entries:
        # For "hallucinating" prompt, ground truth is inverted:
        # hallucinating=yes means not supported, hallucinating=no means supported
        ground_truth_supported = "yes" if entry["supported"] else "no"
        ground_truth_hallucinating = "no" if entry["supported"] else "yes"

        token_ids, positions = _build_hallucination_context(
            entry["user_prompt"],
            entry["response_text"],
            entry["span_char_start"],
            entry["span_char_end"],
            tokenizer,
        )

        for prompt_name, vp in verbalizer_prompts.items():
            # Invert ground truth for negatively-framed prompts
            if "hallucinating" in prompt_name:
                ground_truth = ground_truth_hallucinating
            else:
                ground_truth = ground_truth_supported

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
                    "span_text": entry["span_text"],
                    "label": entry["label"],
                    "supported": entry["supported"],
                    "prompt_name": prompt_name,
                }
            )

    return prompt_infos, entry_metadata


TEXT_BASELINE_PROMPTS: dict[str, str] = {
    "claim_correct": (
        "Answer with 'Yes' or 'No' only. "
        'Is the claim "{span_text}" in the above response factually correct?'
    ),
    "accurate_statement": (
        "Answer with 'Yes' or 'No' only. "
        'In the response above, is "{span_text}" accurate?'
    ),
    "verify_fact": (
        "Answer with 'Yes' or 'No' only. "
        'Please verify: is "{span_text}" as stated in the response above true?'
    ),
}


def build_hallucination_text_baseline_inputs(
    entries: list[dict[str, Any]],
    tokenizer,
    text_baseline_prompts: dict[str, str] | None = None,
) -> tuple[list[TextBaselineInput], list[dict[str, Any]]]:
    """Build text baseline inputs for hallucination eval.

    Format: user prompt → assistant response (truncated to span end) →
    follow-up user question asking whether the specific span is factually correct.
    """
    if text_baseline_prompts is None:
        text_baseline_prompts = TEXT_BASELINE_PROMPTS

    baseline_inputs: list[TextBaselineInput] = []
    entry_metadata: list[dict[str, Any]] = []

    for entry in entries:
        truncated_response = entry["response_text"][:entry["span_char_end"]]
        ground_truth = "yes" if entry["supported"] else "no"

        for prompt_name, prompt_template in text_baseline_prompts.items():
            follow_up = prompt_template.format(span_text=entry["span_text"])

            prompt_text = render_baseline_chat_prompt(
                tokenizer=tokenizer,
                messages=[
                    {"role": "user", "content": entry["user_prompt"]},
                    {"role": "assistant", "content": truncated_response},
                    {"role": "user", "content": follow_up},
                ],
                add_generation_prompt=True,
                enable_thinking=False,
            )

            baseline_inputs.append(
                TextBaselineInput(
                    prompt_text=prompt_text,
                    ground_truth=ground_truth,
                    prompt_name=prompt_name,
                    variant="full_context",
                )
            )
            entry_metadata.append(
                {
                    "id": entry["id"],
                    "span_text": entry["span_text"],
                    "label": entry["label"],
                    "supported": entry["supported"],
                    "prompt_name": prompt_name,
                    "baseline_variant": "full_context",
                }
            )

    return baseline_inputs, entry_metadata


def run_hallucination_text_baseline_eval(
    *,
    model_name: str,
    tokenizer,
    backend,
    hf_model=None,
    device=None,
    eval_batch_size: int = 16,
    generation_kwargs: dict[str, Any] | None = None,
    output_dir: str | None = None,
    max_entries: int | None = None,
    entry_ids: list[str] | None = None,
    dataset_config: str = "Qwen2.5-7B-Instruct",
    vllm_lora_path: str | None = None,
    enforce_eager: bool = True,
) -> dict[str, Any]:
    if generation_kwargs is None:
        generation_kwargs = GENERATION_KWARGS

    entries = load_hallucination_dataset(dataset_config, max_entries=max_entries, entry_ids=entry_ids)
    baseline_inputs, entry_metadata = build_hallucination_text_baseline_inputs(
        entries, tokenizer,
    )

    return run_text_baseline_eval_loop(
        eval_name="hallucination",
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
        vllm_lora_path=vllm_lora_path,
        enforce_eager=enforce_eager,
        extra_output_data={
            "vllm_lora_path": vllm_lora_path,
        },
    )


def run_hallucination_open_ended_eval(
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
    entry_ids: list[str] | None = None,
    verbalizer_prompts: dict[str, str] | None = None,
    dataset_config: str = "Qwen2.5-7B-Instruct",
) -> dict[str, Any]:
    if verbalizer_prompts is None:
        verbalizer_prompts = AO_PROMPTS
    if generation_kwargs is None:
        generation_kwargs = GENERATION_KWARGS

    entries = load_hallucination_dataset(dataset_config, max_entries=max_entries, entry_ids=entry_ids)
    prompt_infos, entry_metadata = build_hallucination_verbalizer_prompt_infos(
        entries,
        verbalizer_prompts,
        tokenizer,
    )

    return run_verbalizer_binary_eval_loop(
        eval_name="hallucination",
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["ao", "text_baseline", "text_sft"],
        default="ao",
    )
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Path to checkpoint directory (overrides STANDARD_VERBALIZER_LORAS)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="Qwen2.5-7B-Instruct",
        help="HF dataset config name (default: Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="Subset of prompt names to run (default: all)",
    )
    parser.add_argument(
        "--test-ids",
        type=str,
        default=None,
        help="Path to JSON file with test_ids list (from linear probe split)",
    )
    parser.add_argument("--baseline-backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument(
        "--text-sft-lora-path",
        type=str,
        default=None,
        help="Path to text SFT LoRA checkpoint for --mode text_sft.",
    )
    parser.add_argument(
        "--no-enforce-eager",
        action="store_true",
        help="Disable enforce_eager for vLLM.",
    )
    args = parser.parse_args()

    # Load test entry IDs if provided
    entry_ids = None
    if args.test_ids:
        entry_ids = json.loads(Path(args.test_ids).read_text())["test_ids"]
        print(f"Using {len(entry_ids)} test entries from {args.test_ids}")

    if args.mode == "ao":
        verbalizer_loras = STANDARD_VERBALIZER_LORAS
        if args.checkpoint_dir:
            verbalizer_loras = [args.checkpoint_dir]

        prompts = AO_PROMPTS
        if args.prompts:
            prompts = {k: v for k, v in AO_PROMPTS.items() if k in args.prompts}
            assert len(prompts) == len(args.prompts), (
                f"Unknown prompts: {set(args.prompts) - set(AO_PROMPTS.keys())}"
            )

        run_default_eval(
            eval_name="hallucination",
            run_eval_fn=run_hallucination_open_ended_eval,
            model_name="Qwen/Qwen3-8B",
            run_eval_kwargs={
                "verbalizer_lora_paths": verbalizer_loras,
                "verbalizer_prompts": prompts,
                "max_entries": args.max_entries,
                "entry_ids": entry_ids,
                "dataset_config": args.dataset_config,
            },
        )
    elif args.mode == "text_sft":
        assert args.text_sft_lora_path, "Provide --text-sft-lora-path for text_sft mode"
        from pathlib import Path as _Path

        lora_slug = _Path(args.text_sft_lora_path).parent.name
        run_default_text_baseline_eval(
            eval_name=f"hallucination_text_sft/{lora_slug}",
            run_eval_fn=run_hallucination_text_baseline_eval,
            model_name="Qwen/Qwen3-8B",
            backend="vllm",
            run_eval_kwargs={
                "max_entries": args.max_entries,
                "entry_ids": entry_ids,
                "dataset_config": args.dataset_config,
                "vllm_lora_path": args.text_sft_lora_path,
                "enforce_eager": not args.no_enforce_eager,
            },
        )
    else:
        run_default_text_baseline_eval(
            eval_name="hallucination_text_baseline",
            run_eval_fn=run_hallucination_text_baseline_eval,
            model_name="Qwen/Qwen3-8B",
            backend=args.baseline_backend,
            run_eval_kwargs={
                "max_entries": args.max_entries,
                "entry_ids": entry_ids,
                "dataset_config": args.dataset_config,
            },
        )
