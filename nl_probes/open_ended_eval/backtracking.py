import asyncio
import json
from pathlib import Path
from typing import Any

import anthropic
from tqdm import tqdm

from nl_probes.base_experiment import (
    VerbalizerInputInfo,
    VerbalizerResults,
    tokenize_chat_messages,
    compute_segment_positions,
)
import nl_probes.base_experiment as base_experiment
from nl_probes.open_ended_eval.eval_runner import (
    BaselinePromptVariant,
    STANDARD_VERBALIZER_LORAS,
    _load_adapter_and_build_config,
    TextBaselineGenerationResult,
    TextBaselineInput,
    decode_contiguous_span_text,
    ensure_default_adapter,
    get_first_ao_response,
    render_baseline_chat_prompt,
    run_default_eval,
    run_default_text_baseline_eval,
    run_text_baseline_eval_loop,
    run_verbalizer_generation_eval_loop,
)

DATASET_DIR = "data_pipelines/backtracking"
DATASET_FILENAME = "backtracking_eval_dataset.json"

# Backtracking explanations need more tokens than short-answer evals
GENERATION_KWARGS: dict[str, Any] = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 150,
}

MULTI_ROLLOUT_GENERATION_KWARGS: dict[str, Any] = {
    "do_sample": True,
    "temperature": 1.0,
    "max_new_tokens": 150,
}

VERBALIZER_PROMPTS: tuple[str, ...] = ("What is the model uncertain or confused about at this point in its reasoning?",)
TEXT_BASELINE_VARIANTS: tuple[BaselinePromptVariant, ...] = ("full_context", "selected_span")

JUDGE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_JUDGE_CONCURRENCY = 20

JUDGE_SYSTEM_PROMPT = """\
You are evaluating whether an Activation Oracle (AO) — a model that reads another model's \
internal activations — correctly identified what a reasoning model was uncertain about.

You will be given:
1. The reasoning model's chain-of-thought PREFIX (up to the point of uncertainty)
2. The GROUND TRUTH description of what the model was actually uncertain about
3. The AO's RESPONSE describing what it thinks the model was uncertain about

Rate the AO's response on two dimensions:

SPECIFICITY (1-5):
1 = Completely vague/generic (e.g. "the model is thinking" or "the model is uncertain")
2 = Slightly specific but mostly generic (e.g. "the model is uncertain about a math problem")
3 = Moderately specific (identifies the general topic but not the precise uncertainty)
4 = Specific (identifies the topic and gives a reasonable description of the uncertainty)
5 = Highly specific (pinpoints the exact nature of the uncertainty)

CORRECTNESS (1-5):
1 = Completely wrong or irrelevant
2 = Tangentially related but misidentifies the core uncertainty
3 = Partially correct (right general area but wrong specifics)
4 = Mostly correct (captures the main uncertainty with minor inaccuracies)
5 = Fully correct (accurately describes the uncertainty)

Respond with ONLY a JSON object: {"specificity": <int>, "correctness": <int>, "reasoning": "<brief explanation>"}"""

JUDGE_USER_TEMPLATE = """\
PREFIX (reasoning up to uncertainty point):
{prefix}

GROUND TRUTH (what the model was actually uncertain about):
{ground_truth}

AO RESPONSE:
{ao_response}"""


def dataset_path(model_name: str) -> str:
    model_dir = model_name.split("/")[-1]
    return f"{DATASET_DIR}/{model_dir}/{DATASET_FILENAME}"


def load_backtracking_dataset(
    model_name: str,
    max_entries: int | None = None,
) -> list[dict[str, Any]]:
    data = json.loads(Path(dataset_path(model_name)).read_text())
    entries = data["entries"]
    if max_entries is not None:
        entries = entries[:max_entries]
    assert len(entries) > 0, "No entries in dataset"
    return entries


def build_backtracking_verbalizer_prompt_infos(
    entries: list[dict[str, Any]],
    verbalizer_prompts: tuple[str, ...] = VERBALIZER_PROMPTS,
    tokenizer=None,
    segment_start: int = -20,
) -> tuple[list[VerbalizerInputInfo], list[dict[str, Any]]]:
    """Build verbalizer prompt infos and return paired metadata for each entry."""
    assert tokenizer is not None, "tokenizer is required"
    prompt_infos: list[VerbalizerInputInfo] = []
    entry_metadata: list[dict[str, Any]] = []

    for entry in entries:
        prefix = entry["prefix"]

        messages = [
            {"role": "user", "content": entry["problem"]},
            {"role": "assistant", "content": prefix},
        ]
        token_ids = tokenize_chat_messages(
            tokenizer,
            messages,
            add_generation_prompt=False,
            continue_thinking=True,
        )
        positions = compute_segment_positions(len(token_ids), segment_start)

        for vp in verbalizer_prompts:
            prompt_infos.append(
                VerbalizerInputInfo(
                    context_token_ids=token_ids,
                    positions=positions,
                    ground_truth=entry["uncertainty_description"],
                    verbalizer_prompt=vp,
                )
            )
            entry_metadata.append(
                {
                    "problem_id": entry.get("problem_id"),
                    "backtrack_rate": entry["backtrack_rate"],
                    "bucket": entry.get("bucket"),
                    "prefix_length": len(entry["prefix"]),
                }
            )

    return prompt_infos, entry_metadata


def build_backtracking_text_baseline_inputs(
    entries: list[dict[str, Any]],
    verbalizer_prompts: tuple[str, ...] = VERBALIZER_PROMPTS,
    tokenizer=None,
    segment_start: int = -20,
    variants: tuple[BaselinePromptVariant, ...] = TEXT_BASELINE_VARIANTS,
) -> tuple[list[TextBaselineInput], list[dict[str, Any]]]:
    assert tokenizer is not None, "tokenizer is required"
    baseline_inputs: list[TextBaselineInput] = []
    entry_metadata: list[dict[str, Any]] = []

    for entry in entries:
        # Tokenize for selected_span extraction only (full_context uses plain messages)
        messages = [
            {"role": "user", "content": entry["problem"]},
            {"role": "assistant", "content": entry["prefix"]},
        ]
        token_ids = tokenize_chat_messages(
            tokenizer,
            messages,
            add_generation_prompt=False,
            continue_thinking=True,
        )
        positions = compute_segment_positions(len(token_ids), segment_start)
        selected_span_text = decode_contiguous_span_text(token_ids, positions, tokenizer)

        for verbalizer_prompt in verbalizer_prompts:
            for variant in variants:
                if variant == "full_context":
                    # Render prefix as a closed assistant turn. We can't preserve
                    # open-thinking state because Qwen3's template strips <think>
                    # content from historical assistant messages (see
                    # docs/tokenization_learnings.md).
                    prompt_text = render_baseline_chat_prompt(
                        tokenizer=tokenizer,
                        messages=[
                            {"role": "user", "content": entry["problem"]},
                            {"role": "assistant", "content": entry["prefix"]},
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
                                    f"Based only on this span, make your best guess: {verbalizer_prompt}"
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
                        ground_truth=entry["uncertainty_description"],
                        prompt_name=verbalizer_prompt,
                        variant=variant,
                    )
                )
                entry_metadata.append(
                    {
                        "problem_id": entry.get("problem_id"),
                        "backtrack_rate": entry["backtrack_rate"],
                        "bucket": entry.get("bucket"),
                        "prefix_length": len(entry["prefix"]),
                        "baseline_variant": variant,
                        "prompt_name": verbalizer_prompt,
                    }
                )

    return baseline_inputs, entry_metadata


# --- LLM judge for backtracking (eval-specific, not shared) ---


async def judge_single_response(
    client: anthropic.AsyncAnthropic,
    prefix: str,
    ground_truth: str,
    ao_response: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    user_message = JUDGE_USER_TEMPLATE.format(
        prefix=prefix[-1500:],  # truncate prefix for judge context
        ground_truth=ground_truth,
        ao_response=ao_response,
    )

    async with semaphore:
        response = await client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=300,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

    text = response.content[0].text
    # Parse JSON from response — strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]  # remove ```json line
        text = text.rsplit("```", 1)[0]  # remove closing ```
        text = text.strip()
    result = json.loads(text)
    assert "specificity" in result and "correctness" in result
    return result


async def judge_response_texts(
    response_texts: list[str | None],
    entries: list[dict[str, Any]],
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
    extra_metadata: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Use Claude Haiku to judge response texts against ground truth."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)

    if extra_metadata is None:
        extra_metadata = [{} for _ in entries]
    assert len(extra_metadata) == len(entries), (
        f"extra_metadata length {len(extra_metadata)} != entries length {len(entries)}"
    )

    tasks = []
    task_metadata = []

    for i, (response_text, entry, meta) in enumerate(
        zip(response_texts, entries, extra_metadata, strict=True)
    ):
        if response_text is None:
            continue

        tasks.append(
            judge_single_response(
                client=client,
                prefix=entry["prefix"],
                ground_truth=entry["uncertainty_description"],
                ao_response=response_text,
                semaphore=semaphore,
            )
        )
        task_metadata.append(
            {
                **meta,
                "result_index": i,
                "response_text": response_text,
                "ground_truth": entry["uncertainty_description"],
                "backtrack_rate": entry["backtrack_rate"],
            }
        )

    print(f"Judging {len(tasks)} responses with {JUDGE_MODEL} (concurrency={concurrency})...")
    pbar = tqdm(total=len(tasks), desc="LLM judge")
    async def _track(coro):
        result = await coro
        pbar.update(1)
        return result
    judge_results = await asyncio.gather(*[_track(t) for t in tasks], return_exceptions=True)
    pbar.close()

    scored_results = []
    for meta, judge_result in zip(task_metadata, judge_results):
        if isinstance(judge_result, Exception):
            print(f"Judge error for result {meta['result_index']}: {judge_result}")
            continue
        scored_results.append({**meta, **judge_result})

    return scored_results


async def judge_ao_responses(
    results: list[VerbalizerResults],
    entries: list[dict[str, Any]],
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
) -> list[dict[str, Any]]:
    return await judge_response_texts(
        [get_first_ao_response(result) for result in results],
        entries=entries,
        concurrency=concurrency,
    )


async def judge_text_baseline_responses(
    results: list[TextBaselineGenerationResult],
    entries: list[dict[str, Any]],
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
    extra_metadata: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return await judge_response_texts(
        [result.response_text for result in results],
        entries=entries,
        concurrency=concurrency,
        extra_metadata=extra_metadata,
    )


# --- Consensus selection judge ---

CONSENSUS_SYSTEM_PROMPT = """\
You are analyzing multiple responses from an Activation Oracle (AO) that was asked what a \
reasoning model is uncertain about at a specific point. You received multiple sampled responses \
for the same input.

Your task: identify the most specific claim about the model's uncertainty that appears \
consistently across the responses.

Rules:
1. A claim "appears" in a response if the response expresses the same idea, even if worded differently.
2. The claim must appear in at least {min_agreement_pct}% of the responses.
3. Among claims that meet the threshold, pick the MOST SPECIFIC one.
4. If no specific claim meets the threshold (e.g. all responses are vague/generic or they all \
disagree), return the most common response verbatim.

Respond with ONLY a JSON object:
{{"consensus_response": "<the selected claim, stated clearly in one sentence>", \
"agreement_fraction": <float 0-1, fraction of responses containing this claim>, \
"reasoning": "<brief explanation of what you found>"}}\
"""

CONSENSUS_USER_TEMPLATE = """\
PREFIX (reasoning up to uncertainty point):
{prefix}

AO RESPONSES ({num_responses} sampled responses):
{responses_text}"""


async def select_consensus_single(
    client: anthropic.AsyncAnthropic,
    prefix: str,
    ao_responses: list[str],
    semaphore: asyncio.Semaphore,
    min_agreement_pct: int = 50,
) -> dict[str, Any]:
    responses_text = "\n".join(
        f"--- Response {i+1} ---\n{r}" for i, r in enumerate(ao_responses)
    )
    user_message = CONSENSUS_USER_TEMPLATE.format(
        prefix=prefix[-1500:],
        num_responses=len(ao_responses),
        responses_text=responses_text,
    )
    system_prompt = CONSENSUS_SYSTEM_PROMPT.format(min_agreement_pct=min_agreement_pct)

    async with semaphore:
        response = await client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=400,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
        text = text.strip()
    result = json.loads(text)
    assert "consensus_response" in result
    return result


async def select_consensus_responses(
    entries: list[dict[str, Any]],
    grouped_responses: list[list[str]],
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
    min_agreement_pct: int = 50,
) -> list[dict[str, Any]]:
    """Select consensus response from multiple rollouts per entry."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)

    tasks = []
    for entry, responses in zip(entries, grouped_responses):
        tasks.append(
            select_consensus_single(
                client=client,
                prefix=entry["prefix"],
                ao_responses=responses,
                semaphore=semaphore,
                min_agreement_pct=min_agreement_pct,
            )
        )

    print(f"Selecting consensus from {len(tasks)} entries with {JUDGE_MODEL} (concurrency={concurrency})...")
    pbar = tqdm(total=len(tasks), desc="Consensus selection")

    async def _track(coro):
        result = await coro
        pbar.update(1)
        return result

    results = await asyncio.gather(*[_track(t) for t in tasks], return_exceptions=True)
    pbar.close()

    consensus_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Consensus error for entry {i}: {result}")
            # Fallback to first response
            consensus_results.append({
                "consensus_response": grouped_responses[i][0],
                "agreement_fraction": None,
                "reasoning": f"Error during consensus selection: {result}",
            })
        else:
            consensus_results.append(result)

    return consensus_results


def compute_judge_metrics(scored_results: list[dict[str, Any]]) -> dict[str, float]:
    specificities = [r["specificity"] for r in scored_results]
    correctnesses = [r["correctness"] for r in scored_results]

    return {
        "mean_specificity": sum(specificities) / len(specificities),
        "mean_correctness": sum(correctnesses) / len(correctnesses),
        "specificity_>=3_rate": sum(1 for s in specificities if s >= 3) / len(specificities),
        "specificity_>=4_rate": sum(1 for s in specificities if s >= 4) / len(specificities),
        "correctness_>=3_rate": sum(1 for c in correctnesses if c >= 3) / len(correctnesses),
        "correctness_>=4_rate": sum(1 for c in correctnesses if c >= 4) / len(correctnesses),
        "num_scored": float(len(scored_results)),
    }


# ---------------------------------------------------------------------------
# Multiple-choice eval
# ---------------------------------------------------------------------------

ANSWER_LETTERS = ["A", "B", "C", "D"]

MC_VERBALIZER_PROMPT_TEMPLATE = (
    "What is the model most likely uncertain about at this point? "
    "Answer with just the letter (A, B, C, or D).\n\n"
    "{options_text}"
)

# Single-token variants of A/B/C/D for logit scoring
LETTER_CANDIDATE_VARIANTS: dict[str, list[str]] = {
    "A": ["A", " A", "\nA"],
    "B": ["B", " B", "\nB"],
    "C": ["C", " C", "\nC"],
    "D": ["D", " D", "\nD"],
}


def build_letter_candidate_token_groups(tokenizer) -> dict[str, list[int]]:
    """Collect single-token A/B/C/D variants for first-token scoring."""
    token_groups: dict[str, list[int]] = {}
    for label, variants in LETTER_CANDIDATE_VARIANTS.items():
        token_ids: list[int] = []
        for text in variants:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in token_ids:
                token_ids.append(int(ids[0]))
        if not token_ids:
            raise ValueError(f"Tokenizer had no single-token variants for label '{label}'")
        token_groups[label] = token_ids
    return token_groups


def _format_mc_options(mc_options: list[str]) -> str:
    """Format MC options as A. ... B. ... etc."""
    lines = []
    for i, opt in enumerate(mc_options):
        lines.append(f"{ANSWER_LETTERS[i]}. {opt}")
    return "\n".join(lines)


def build_backtracking_mc_verbalizer_prompt_infos(
    entries: list[dict[str, Any]],
    tokenizer,
    segment_start: int = -20,
) -> tuple[list[VerbalizerInputInfo], list[dict[str, Any]]]:
    """Build verbalizer prompt infos for MC backtracking eval."""
    prompt_infos: list[VerbalizerInputInfo] = []
    entry_metadata: list[dict[str, Any]] = []

    for entry in entries:
        assert "mc_options" in entry, f"Entry {entry.get('problem_id')} missing mc_options"

        messages = [
            {"role": "user", "content": entry["problem"]},
            {"role": "assistant", "content": entry["prefix"]},
        ]
        token_ids = tokenize_chat_messages(
            tokenizer,
            messages,
            add_generation_prompt=False,
            continue_thinking=True,
        )
        positions = compute_segment_positions(len(token_ids), segment_start)

        options_text = _format_mc_options(entry["mc_options"])
        verbalizer_prompt = MC_VERBALIZER_PROMPT_TEMPLATE.format(options_text=options_text)

        prompt_infos.append(
            VerbalizerInputInfo(
                context_token_ids=token_ids,
                positions=positions,
                ground_truth=entry["mc_correct_label"],
                verbalizer_prompt=verbalizer_prompt,
            )
        )
        entry_metadata.append(
            {
                "problem_id": entry.get("problem_id"),
                "backtrack_rate": entry["backtrack_rate"],
                "mc_correct_label": entry["mc_correct_label"],
                "mc_correct_index": entry["mc_correct_index"],
            }
        )

    return prompt_infos, entry_metadata


def run_backtracking_mc_eval(
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
    segment_tokens: int = 20,
) -> dict[str, Any]:
    """Multiple-choice backtracking eval using logit scoring over A/B/C/D."""
    if generation_kwargs is None:
        generation_kwargs = {"do_sample": False, "max_new_tokens": 1}

    entries = load_backtracking_dataset(model_name, max_entries=max_entries)
    # Filter to entries that have MC options
    entries = [e for e in entries if "mc_options" in e]
    assert entries, "No entries with mc_options found in dataset"

    prompt_infos, entry_metadata = build_backtracking_mc_verbalizer_prompt_infos(
        entries, tokenizer, segment_start=-segment_tokens,
    )

    candidate_token_groups = build_letter_candidate_token_groups(tokenizer)

    ensure_default_adapter(model)
    model.eval()

    results_by_verbalizer: dict[str, dict[str, Any]] = {}

    for verbalizer_entry in verbalizer_lora_paths:
        sanitized_name, config = _load_adapter_and_build_config(
            model, verbalizer_entry, model_name, eval_batch_size, generation_kwargs,
        )

        print(f"Running backtracking MC eval with verbalizer: {verbalizer_entry}")
        verbalizer_key = verbalizer_entry.split("/")[-1]

        binary_results = base_experiment.run_verbalizer_binary_score(
            model=model,
            tokenizer=tokenizer,
            verbalizer_prompt_infos=prompt_infos,
            verbalizer_lora_path=sanitized_name,
            target_lora_path=None,
            config=config,
            device=device,
            candidate_token_groups=candidate_token_groups,
        )

        scored_results = []
        correct = 0
        total = 0
        for result, meta in zip(binary_results, entry_metadata):
            # Pick letter with highest candidate score
            scores = {label: float(result.candidate_scores[label]) for label in ANSWER_LETTERS}
            predicted_label = max(scores, key=scores.get)  # type: ignore[arg-type]
            is_correct = predicted_label == meta["mc_correct_label"]
            total += 1
            if is_correct:
                correct += 1

            scored_results.append({
                **meta,
                "predicted_label": predicted_label,
                "is_correct": is_correct,
                "scores": scores,
                "argmax_token_text": result.argmax_token_text.strip(),
            })

        metrics = {
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "chance": 0.25,
        }
        results_by_verbalizer[verbalizer_key] = metrics

        print(f"\n  MC accuracy for {verbalizer_key}: {correct}/{total} = {metrics['accuracy']:.3f} (chance=0.25)")

        if output_dir is not None:
            import os
            os.makedirs(output_dir, exist_ok=True)
            lora_name = verbalizer_key.replace("/", "_").replace(".", "_")
            output_path = os.path.join(output_dir, f"backtracking_mc_{lora_name}.json")
            with open(output_path, "w") as f:
                json.dump({
                    "verbalizer": verbalizer_entry,
                    "metrics": metrics,
                    "scored_results": scored_results,
                }, f, indent=2)
            print(f"  Saved to {output_path}")

        if sanitized_name in model.peft_config:
            model.delete_adapter(sanitized_name)

    return {"mc_results_by_verbalizer": results_by_verbalizer}


def run_backtracking_open_ended_eval(
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
    judge_concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
    num_rollouts: int = 1,
    multi_rollout_mode: str = "consensus",
    segment_tokens: int = 20,
) -> dict[str, Any]:
    """
    Backtracking eval uses an LLM judge, so it can't use the standard score_fn pattern
    directly (the judge needs the original entries, not just metadata). We wrap the
    judge call in a score_fn closure.

    When num_rollouts > 1, runs multiple temp=1 rollouts per entry.
    multi_rollout_mode controls selection:
      - "consensus": LLM judge picks most common specific claim, then scores it
      - "best_of_n": scores all N individually, keeps the best per entry
    """
    if generation_kwargs is None:
        if num_rollouts > 1:
            generation_kwargs = MULTI_ROLLOUT_GENERATION_KWARGS
        else:
            generation_kwargs = GENERATION_KWARGS

    entries = load_backtracking_dataset(model_name, max_entries=max_entries)
    prompt_infos, entry_metadata = build_backtracking_verbalizer_prompt_infos(
        entries,
        tokenizer=tokenizer,
        segment_start=-segment_tokens,
    )

    if num_rollouts > 1:
        return _run_backtracking_multi_rollout(
            entries=entries,
            prompt_infos=prompt_infos,
            entry_metadata=entry_metadata,
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_name=model_name,
            eval_batch_size=eval_batch_size,
            generation_kwargs=generation_kwargs,
            verbalizer_lora_paths=verbalizer_lora_paths,
            output_dir=output_dir,
            max_entries=max_entries,
            judge_concurrency=judge_concurrency,
            num_rollouts=num_rollouts,
            mode=multi_rollout_mode,
        )

    # Single rollout (original behavior)
    def score_fn(results: list[VerbalizerResults], metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return asyncio.run(
            judge_ao_responses(
                results=results,
                entries=entries,
                concurrency=judge_concurrency,
            )
        )

    return run_verbalizer_generation_eval_loop(
        eval_name="backtracking",
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=model_name,
        eval_batch_size=eval_batch_size,
        generation_kwargs=generation_kwargs,
        prompt_infos=prompt_infos,
        entry_metadata=entry_metadata,
        score_fn=score_fn,
        metrics_fn=compute_judge_metrics,
        num_entries=len(entries),
        verbalizer_lora_paths=verbalizer_lora_paths,
        output_dir=output_dir,
        extra_output_data={"max_entries": max_entries},
    )


def run_backtracking_text_baseline_eval(
    *,
    model_name: str,
    tokenizer,
    backend,
    hf_model=None,
    device=None,
    eval_batch_size: int = 32,
    generation_kwargs: dict[str, Any] | None = None,
    output_dir: str | None = None,
    max_entries: int | None = None,
    judge_concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
    segment_tokens: int = 20,
    verbalizer_prompts: tuple[str, ...] = VERBALIZER_PROMPTS,
    variants: tuple[BaselinePromptVariant, ...] = TEXT_BASELINE_VARIANTS,
    vllm_lora_path: str | None = None,
    enforce_eager: bool = True,
) -> dict[str, Any]:
    if generation_kwargs is None:
        generation_kwargs = GENERATION_KWARGS

    entries = load_backtracking_dataset(model_name, max_entries=max_entries)
    baseline_inputs, entry_metadata = build_backtracking_text_baseline_inputs(
        entries=entries,
        verbalizer_prompts=verbalizer_prompts,
        tokenizer=tokenizer,
        segment_start=-segment_tokens,
        variants=variants,
    )
    expanded_entries = [
        entry
        for entry in entries
        for _verbalizer_prompt in verbalizer_prompts
        for _variant in variants
    ]

    def score_fn(
        results: list[TextBaselineGenerationResult],
        metadata: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        del metadata
        return asyncio.run(
            judge_text_baseline_responses(
                results=results,
                entries=expanded_entries,
                concurrency=judge_concurrency,
                extra_metadata=entry_metadata,
            )
        )

    return run_text_baseline_eval_loop(
        eval_name="backtracking",
        baseline_inputs=baseline_inputs,
        entry_metadata=entry_metadata,
        backend=backend,
        inference_mode="rollout",
        model_name=model_name,
        tokenizer=tokenizer,
        generation_kwargs=generation_kwargs,
        score_fn=score_fn,
        metrics_fn=compute_judge_metrics,
        output_dir=output_dir,
        hf_model=hf_model,
        device=device,
        eval_batch_size=eval_batch_size,
        num_entries=len(entries),
        vllm_lora_path=vllm_lora_path,
        enforce_eager=enforce_eager,
        extra_output_data={
            "max_entries": max_entries,
            "judge_model": JUDGE_MODEL,
            "vllm_lora_path": vllm_lora_path,
        },
    )


def _score_multi_rollout_consensus(
    *,
    entries: list[dict[str, Any]],
    grouped_responses: list[list[str]],
    all_individual_results: list[list[VerbalizerResults]],
    judge_concurrency: int,
    num_rollouts: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Consensus mode: select consensus response, then score it."""
    consensus_results = asyncio.run(
        select_consensus_responses(
            entries=entries,
            grouped_responses=grouped_responses,
            concurrency=judge_concurrency,
        )
    )

    # Build synthetic VerbalizerResults with consensus responses for scoring
    consensus_verbalizer_results = []
    for consensus, entry_results_group in zip(consensus_results, all_individual_results):
        template = entry_results_group[0]
        consensus_vr = VerbalizerResults(
            verbalizer_lora_path=template.verbalizer_lora_path,
            target_lora_path=template.target_lora_path,
            context_token_ids=template.context_token_ids,
            act_key=template.act_key,
            verbalizer_prompt=template.verbalizer_prompt,
            ground_truth=template.ground_truth,
            num_tokens=template.num_tokens,
            responses=[consensus["consensus_response"]],
        )
        consensus_verbalizer_results.append(consensus_vr)

    scored_results = asyncio.run(
        judge_ao_responses(
            results=consensus_verbalizer_results,
            entries=entries,
            concurrency=judge_concurrency,
        )
    )

    for sr in scored_results:
        idx = sr["result_index"]
        sr["individual_responses"] = grouped_responses[idx]
        sr["consensus_info"] = consensus_results[idx]
        sr["num_rollouts"] = num_rollouts

    metrics: dict[str, Any] | None = None
    if scored_results:
        metrics = compute_judge_metrics(scored_results)
        agreement_fractions = [
            sr["consensus_info"].get("agreement_fraction")
            for sr in scored_results
            if sr["consensus_info"].get("agreement_fraction") is not None
        ]
        if agreement_fractions:
            metrics["mean_agreement_fraction"] = sum(agreement_fractions) / len(agreement_fractions)

    return scored_results, metrics


def _score_multi_rollout_best_of_n(
    *,
    entries: list[dict[str, Any]],
    grouped_responses: list[list[str]],
    judge_concurrency: int,
    num_rollouts: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Best-of-N mode: score all N rollouts individually, keep best per entry."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(judge_concurrency)

    # Score every individual response
    all_tasks = []
    task_keys = []  # (entry_index, rollout_index)
    for entry_idx, (entry, responses) in enumerate(zip(entries, grouped_responses)):
        for rollout_idx, response_text in enumerate(responses):
            if not response_text:
                continue
            all_tasks.append(
                judge_single_response(
                    client=client,
                    prefix=entry["prefix"],
                    ground_truth=entry["uncertainty_description"],
                    ao_response=response_text,
                    semaphore=semaphore,
                )
            )
            task_keys.append((entry_idx, rollout_idx))

    print(f"Scoring {len(all_tasks)} individual responses with {JUDGE_MODEL} "
          f"(concurrency={judge_concurrency})...")
    pbar = tqdm(total=len(all_tasks), desc="Best-of-N scoring")

    async def _run():
        async def _track(coro):
            result = await coro
            pbar.update(1)
            return result
        return await asyncio.gather(*[_track(t) for t in all_tasks], return_exceptions=True)

    judge_results = asyncio.run(_run())
    pbar.close()

    # Group scores by entry, pick best
    from collections import defaultdict
    scores_by_entry: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (entry_idx, rollout_idx), judge_result in zip(task_keys, judge_results):
        if isinstance(judge_result, Exception):
            print(f"Judge error for entry {entry_idx}, rollout {rollout_idx}: {judge_result}")
            continue
        scores_by_entry[entry_idx].append({
            "rollout_index": rollout_idx,
            "response_text": grouped_responses[entry_idx][rollout_idx],
            **judge_result,
        })

    scored_results = []
    for entry_idx in range(len(entries)):
        entry_scores = scores_by_entry.get(entry_idx, [])
        if not entry_scores:
            continue
        # Pick best by specificity + correctness, break ties by specificity
        best = max(entry_scores, key=lambda s: (s["specificity"] + s["correctness"], s["specificity"]))
        scored_results.append({
            "result_index": entry_idx,
            "response_text": best["response_text"],
            "ground_truth": entries[entry_idx]["uncertainty_description"],
            "backtrack_rate": entries[entry_idx]["backtrack_rate"],
            "specificity": best["specificity"],
            "correctness": best["correctness"],
            "reasoning": best["reasoning"],
            "individual_responses": grouped_responses[entry_idx],
            "all_rollout_scores": entry_scores,
            "num_rollouts": num_rollouts,
            "best_rollout_index": best["rollout_index"],
        })

    metrics: dict[str, Any] | None = None
    if scored_results:
        metrics = compute_judge_metrics(scored_results)
        # Add best-of-n specific metrics
        all_individual_specs = [s["specificity"] for scores in scores_by_entry.values() for s in scores]
        all_individual_corrs = [s["correctness"] for scores in scores_by_entry.values() for s in scores]
        if all_individual_specs:
            metrics["mean_individual_specificity"] = sum(all_individual_specs) / len(all_individual_specs)
            metrics["mean_individual_correctness"] = sum(all_individual_corrs) / len(all_individual_corrs)

    return scored_results, metrics


def _run_backtracking_multi_rollout(
    *,
    entries: list[dict[str, Any]],
    prompt_infos: list[VerbalizerInputInfo],
    entry_metadata: list[dict[str, Any]],
    model,
    tokenizer,
    device,
    model_name: str,
    eval_batch_size: int,
    generation_kwargs: dict[str, Any],
    verbalizer_lora_paths: list[str],
    output_dir: str | None = None,
    max_entries: int | None = None,
    judge_concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
    num_rollouts: int = 10,
    mode: str = "consensus",
) -> dict[str, Any]:
    """Multi-rollout backtracking eval.

    mode="consensus": N rollouts → consensus selection → score consensus response
    mode="best_of_n": N rollouts → score all N individually → keep best per entry
    """
    assert mode in ("consensus", "best_of_n"), f"Unknown mode: {mode}"
    import os
    from dataclasses import asdict

    num_entries = len(entries)

    # Duplicate prompt_infos N times (each entry gets N consecutive slots)
    expanded_prompt_infos = []
    expanded_metadata = []
    for pi, meta in zip(prompt_infos, entry_metadata):
        for _ in range(num_rollouts):
            expanded_prompt_infos.append(pi)
            expanded_metadata.append(meta)

    ensure_default_adapter(model)
    model.eval()

    all_scored_results: list[dict[str, Any]] = []
    metrics_by_verbalizer: dict[str, dict[str, Any]] = {}

    for verbalizer_entry in verbalizer_lora_paths:
        sanitized_name, config = _load_adapter_and_build_config(
            model, verbalizer_entry, model_name, eval_batch_size, generation_kwargs,
        )

        print(f"Running backtracking multi-rollout eval ({num_rollouts} rollouts) "
              f"with verbalizer: {verbalizer_entry}")
        verbalizer_key = verbalizer_entry.split("/")[-1]
        lora_name = verbalizer_key.replace("/", "_").replace(".", "_")

        # Generate all rollouts in one pass
        all_results = base_experiment.run_verbalizer(
            model=model,
            tokenizer=tokenizer,
            verbalizer_prompt_infos=expanded_prompt_infos,
            verbalizer_lora_path=sanitized_name,
            target_lora_path=None,
            config=config,
            device=device,
        )

        # Group results back: every num_rollouts consecutive results → one entry
        grouped_responses: list[list[str]] = []
        all_individual_results: list[list[VerbalizerResults]] = []
        for i in range(num_entries):
            start = i * num_rollouts
            end = start + num_rollouts
            entry_results = all_results[start:end]
            all_individual_results.append(entry_results)
            responses = [get_first_ao_response(r) or "" for r in entry_results]
            grouped_responses.append(responses)

        if mode == "consensus":
            scored_results, metrics = _score_multi_rollout_consensus(
                entries=entries,
                grouped_responses=grouped_responses,
                all_individual_results=all_individual_results,
                judge_concurrency=judge_concurrency,
                num_rollouts=num_rollouts,
            )
        else:  # best_of_n
            scored_results, metrics = _score_multi_rollout_best_of_n(
                entries=entries,
                grouped_responses=grouped_responses,
                judge_concurrency=judge_concurrency,
                num_rollouts=num_rollouts,
            )

        if metrics:
            metrics_by_verbalizer[verbalizer_key] = metrics
            print(f"\n  Metrics for {verbalizer_key} ({mode}, {num_rollouts} rollouts):")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.3f}")
                else:
                    print(f"    {k}: {v}")

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir, f"backtracking_{mode}_{lora_name}.json"
            )
            output_data = {
                "config": asdict(config),
                "verbalizer": verbalizer_entry,
                "num_entries": num_entries,
                "num_rollouts": num_rollouts,
                "mode": mode,
                "generation_kwargs": generation_kwargs,
                "scored_results": scored_results,
                "metrics": metrics,
                "max_entries": max_entries,
            }
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"  Saved results to {output_path}")

        all_scored_results.extend(scored_results)

        if sanitized_name in model.peft_config:
            model.delete_adapter(sanitized_name)

    overall_metrics = compute_judge_metrics(all_scored_results) if all_scored_results else {}
    return {
        "overall_metrics": overall_metrics,
        "metrics_by_verbalizer": metrics_by_verbalizer,
        "num_entries": num_entries,
        "num_scored": len(all_scored_results),
        "num_rollouts": num_rollouts,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["open_ended", "mc", "text_baseline", "text_sft"],
        default="mc",
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
        "--num-rollouts",
        type=int,
        default=1,
        help="Number of rollouts per entry (>1 enables multi-rollout mode)",
    )
    parser.add_argument(
        "--multi-rollout-mode",
        choices=["consensus", "best_of_n"],
        default="consensus",
        help="Multi-rollout selection mode (default: consensus)",
    )
    parser.add_argument(
        "--segment-tokens",
        type=int,
        default=20,
        help="Number of tokens from end of context to use as segment (default: 20)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Path to checkpoint directory containing LoRA adapters (overrides STANDARD_VERBALIZER_LORAS)",
    )
    parser.add_argument("--max-entries", type=int, default=None)
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

    verbalizer_loras = STANDARD_VERBALIZER_LORAS
    if args.checkpoint_dir:
        verbalizer_loras = [args.checkpoint_dir]

    if args.mode == "mc":
        eval_name = f"backtracking_mc_seg{args.segment_tokens}"
        run_default_eval(
            eval_name=eval_name,
            run_eval_fn=run_backtracking_mc_eval,
            model_name="Qwen/Qwen3-8B",
            run_eval_kwargs={
                "verbalizer_lora_paths": verbalizer_loras,
                "segment_tokens": args.segment_tokens,
                "max_entries": args.max_entries,
            },
        )
    elif args.mode == "text_sft":
        assert args.text_sft_lora_path, "Provide --text-sft-lora-path for text_sft mode"
        from pathlib import Path as _Path

        lora_slug = _Path(args.text_sft_lora_path).parent.name
        eval_name = f"backtracking_text_sft_seg{args.segment_tokens}/{lora_slug}"
        run_default_text_baseline_eval(
            eval_name=eval_name,
            run_eval_fn=run_backtracking_text_baseline_eval,
            model_name="Qwen/Qwen3-8B",
            backend="vllm",
            run_eval_kwargs={
                "segment_tokens": args.segment_tokens,
                "max_entries": args.max_entries,
                "variants": ("full_context",),
                "vllm_lora_path": args.text_sft_lora_path,
                "enforce_eager": not args.no_enforce_eager,
            },
        )
    elif args.mode == "text_baseline":
        variants = (
            TEXT_BASELINE_VARIANTS
            if args.baseline_variant == "all"
            else (args.baseline_variant,)
        )
        eval_name = f"backtracking_text_baseline_seg{args.segment_tokens}"
        run_default_text_baseline_eval(
            eval_name=eval_name,
            run_eval_fn=run_backtracking_text_baseline_eval,
            model_name="Qwen/Qwen3-8B",
            backend=args.baseline_backend,
            run_eval_kwargs={
                "segment_tokens": args.segment_tokens,
                "max_entries": args.max_entries,
                "variants": variants,
            },
        )
    else:
        eval_name = "backtracking"
        if args.num_rollouts > 1:
            eval_name = f"backtracking_{args.multi_rollout_mode}_{args.num_rollouts}"
        eval_name = f"{eval_name}_seg{args.segment_tokens}"
        run_default_eval(
            eval_name=eval_name,
            run_eval_fn=run_backtracking_open_ended_eval,
            model_name="Qwen/Qwen3-8B",
            run_eval_kwargs={
                "verbalizer_lora_paths": verbalizer_loras,
                "num_rollouts": args.num_rollouts,
                "multi_rollout_mode": args.multi_rollout_mode,
                "segment_tokens": args.segment_tokens,
                "max_entries": args.max_entries,
            },
        )
