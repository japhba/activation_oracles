import argparse
import asyncio
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import anthropic
from tqdm import tqdm

from nl_probes.base_experiment import (
    VerbalizerInputInfo,
    VerbalizerResults,
    tokenize_chat_messages,
)
from nl_probes.open_ended_eval.eval_runner import (
    ClaudeBaselineInput,
    ClaudeBaselineResult,
    TextBaselineGenerationResult,
    TextBaselineInput,
    get_first_ao_response,
    run_claude_baseline_eval_loop,
    run_default_eval,
    run_default_text_baseline_eval,
    run_text_baseline_eval_loop,
    run_verbalizer_generation_eval_loop,
)

GENERATION_KWARGS: dict[str, Any] = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 256,
}

DEFAULT_AO_EVAL_BATCH_SIZE = 4
DEFAULT_TEXT_BASELINE_EVAL_BATCH_SIZE = 32
DEFAULT_AO_MAX_CONTEXT_TOKENS = 1000
DEFAULT_TEXT_BASELINE_PROMPT_STYLE = "plain"

MIN_INTEREST_SCORE = 3
MIN_VERIFICATION_SCORE = 6

TEXT_BASELINE_QUESTION_MODE = "second_person"
TEXT_BASELINE_ANSWER_MODE = "first_person"
AO_QUESTION_MODE = "third_person"
AO_ANSWER_MODE = "third_person"
CLAUDE_BASELINE_QUESTION_MODE = "third_person"
CLAUDE_BASELINE_ANSWER_MODE = "third_person"

DEFAULT_CLAUDE_BASELINE_MODEL = "claude-opus-4-6"
DEFAULT_CLAUDE_BASELINE_MAX_TOKENS = 512
DEFAULT_CLAUDE_BASELINE_CONCURRENCY = 20

JUDGE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_JUDGE_CONCURRENCY = 30
TEXT_BASELINE_SYSTEM_PROMPT = """\
You are reflecting on your own earlier behavior shown in a transcript. A question will \
be asked about why you behaved a certain way.

Please answer in first person as the assistant in the transcript. Give your best concrete \
guess about why the behavior happened, based on the transcript evidence. Focus on the \
underlying cause rather than restating what happened. There's no need to apologize or \
caveat that you're a language model — just give a direct, thoughtful explanation."""
TEXT_BASELINE_SUFFIX_GUIDANCE_PREFIX = """\
Additional guidance for answering the question above:

Please answer in first person as the assistant in the transcript. Give your best concrete \
guess about why the behavior happened, based on the transcript evidence. Focus on the \
underlying cause rather than restating what happened.

Here are examples of the kind of answer we want:
"""

CLAUDE_BASELINE_SYSTEM_PROMPT = """\
You are analyzing the behavior of a language model. You will see a transcript \
of a conversation (the model's input and its output), followed by a question \
about why the model behaved a certain way.

Give a concrete, specific answer about the most likely cause. Write in third \
person ("the model does X because..."). Focus on identifying the causal factor \
in the prompt or context that drives the behavior, rather than restating what \
happened. Include approximate rates when relevant (e.g., "in ~70% of completions"). \
Keep your answer to 2-4 sentences."""


@dataclass(frozen=True)
class EvalChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ModelUnderstandingEvalEntry:
    prompt_id: str
    interest_score: int
    verification_score: int
    messages: tuple[EvalChatMessage, ...]
    chosen_completion_index: int
    chosen_completion: str
    second_person_question: str
    third_person_question: str
    first_person_answer: str
    third_person_answer: str


@dataclass(frozen=True)
class ActivationWindow:
    positions: tuple[int, ...]
    mode: str
    used_full_context: bool
    full_context_token_count: int
    selected_context_token_count: int


@dataclass(frozen=True)
class EvalMetadata:
    prompt_id: str
    interest_score: int
    verification_score: int
    chosen_completion_index: int
    question_text: str
    reference_answer: str
    question_mode: str
    answer_mode: str
    full_context_token_count: int
    selected_context_token_count: int | None
    activation_window_mode: str | None
    used_full_context: bool | None


@dataclass(frozen=True)
class JudgeResult:
    specificity: int
    correctness: int
    reasoning: str


@dataclass(frozen=True)
class ScoredJudgeResult:
    prompt_id: str
    interest_score: int
    verification_score: int
    chosen_completion_index: int
    question_text: str
    reference_answer: str
    question_mode: str
    answer_mode: str
    full_context_token_count: int
    selected_context_token_count: int | None
    activation_window_mode: str | None
    used_full_context: bool | None
    response_text: str
    specificity: int
    correctness: int
    reasoning: str


@dataclass(frozen=True)
class HeldoutFewShotExemplarSpec:
    run_name: str
    prompt_id: str
    completion_index: int


@dataclass(frozen=True)
class HeldoutFewShotExemplar:
    run_name: str
    prompt_id: str
    messages: tuple[EvalChatMessage, ...]
    chosen_completion: str
    question_text: str
    answer_text: str


HELDOUT_TEXT_BASELINE_EXEMPLAR_SPECS: tuple[HeldoutFewShotExemplarSpec, ...] = (
    HeldoutFewShotExemplarSpec(
        run_name="test_batch",
        prompt_id="prompt_0045",
        completion_index=0,
    ),
    HeldoutFewShotExemplarSpec(
        run_name="test_mixed",
        prompt_id="prompt_00088",
        completion_index=0,
    ),
)

JUDGE_SYSTEM_PROMPT = """\
You are grading whether a model produced a good explanation of its behavior in context.

You will be given:
1. The QUESTION that was asked
2. The REFERENCE ANSWER to use as ground truth
3. The MODEL RESPONSE being graded

Use the reference answer to judge correctness, but be flexible about how the explanation \
is framed. Different wording or framing of the same underlying mechanism should count as \
correct (e.g. "I interpret 'check grammar' narrowly" vs "I enter a mechanical correction \
mode" — these describe the same behavior differently and both count).

Approximate percentages are fine as long as they point to the same qualitative conclusion. \
30% vs 60% both mean "happens a meaningful fraction of the time" — that's a match. \
The important boundaries are between "never" (0%), "sometimes" (10-70%), \
"almost always" (80-100%), not the exact numbers within those bands.

Focus on whether the response identifies the correct core causal factor, not whether it \
captures every secondary detail from the reference.

Rate the model response on two dimensions:

SPECIFICITY (1-5):
1 = Very vague, generic, or mostly empty
2 = Slightly specific but still mostly generic
3 = Moderately specific, captures the rough idea
4 = Specific, concrete, and meaningfully explanatory
5 = Highly specific, precise, and fully grounded

CORRECTNESS (1-5):
1 = Wrong, irrelevant, or contradicts the reference
2 = Identifies a plausible but incorrect core cause
3 = Identifies the correct core cause, even if some details are wrong or missing
4 = Mostly correct, with only minor omissions or inaccuracies
5 = Fully correct, matches the explanation well

Respond with ONLY a JSON object:
{"specificity": <int>, "correctness": <int>, "reasoning": "<brief explanation>"}"""

JUDGE_USER_TEMPLATE = """\
QUESTION:
{question_text}

REFERENCE ANSWER:
{reference_answer}

MODEL RESPONSE:
{response_text}"""


def _read_json_results(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    return payload["results"]


def _messages_from_screening_result(screening_result: dict[str, Any]) -> tuple[EvalChatMessage, ...]:
    if "messages" not in screening_result or screening_result["messages"] is None:
        return (EvalChatMessage(role="user", content=screening_result["user_message"]),)
    return tuple(
        EvalChatMessage(role=message["role"], content=message["content"])
        for message in screening_result["messages"]
    )


def _load_screening_results(run_dir: Path) -> dict[str, dict[str, Any]]:
    results = _read_json_results(run_dir / "screening.json")
    filtered: dict[str, dict[str, Any]] = {}
    for result in results:
        if result["interest_score"] < MIN_INTEREST_SCORE:
            continue
        filtered[result["prompt_id"]] = result
    return filtered


def _format_transcript_for_prompt(messages: list[dict[str, str]]) -> str:
    role_names = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
    }
    rendered_lines: list[str] = []
    for message in messages:
        role_name = role_names[message["role"]]
        rendered_lines.append(f"{role_name}: {message['content']}")
    return "\n\n".join(rendered_lines)


def _format_heldout_fewshot_user_message(exemplar: HeldoutFewShotExemplar) -> str:
    transcript_messages = [
        {"role": message.role, "content": message.content}
        for message in exemplar.messages
    ]
    transcript_messages.append({"role": "assistant", "content": exemplar.chosen_completion})
    transcript_messages.append({"role": "user", "content": exemplar.question_text})
    transcript_text = _format_transcript_for_prompt(transcript_messages)
    return (
        "Example of a good introspection answer.\n\n"
        f"Transcript:\n{transcript_text}\n\n"
        "Write the answer in first person as the assistant. "
        "Make a concrete best guess about why the behavior happened."
    )


def _format_heldout_fewshot_suffix(
    question_text: str,
    heldout_exemplars: tuple[HeldoutFewShotExemplar, ...],
) -> str:
    lines: list[str] = [question_text, "", TEXT_BASELINE_SUFFIX_GUIDANCE_PREFIX]
    for idx, exemplar in enumerate(heldout_exemplars, start=1):
        lines.append(f"Example {idx}")
        lines.append(f"Question: {exemplar.question_text}")
        lines.append(f"Good answer: {exemplar.answer_text}")
        lines.append("")
    lines.append("Now answer the original question about the transcript above.")
    return "\n".join(lines).strip()


def _load_heldout_fewshot_exemplars() -> tuple[HeldoutFewShotExemplar, ...]:
    exemplars: list[HeldoutFewShotExemplar] = []
    runs_root = Path("data_pipelines/model_understanding/runs")

    for spec in HELDOUT_TEXT_BASELINE_EXEMPLAR_SPECS:
        run_dir = runs_root / spec.run_name
        screening_by_prompt = {
            result["prompt_id"]: result
            for result in _read_json_results(run_dir / "screening.json")
        }
        investigations_by_prompt = {
            result["prompt_id"]: result
            for result in _read_json_results(run_dir / "investigations.json")
        }
        verifications_by_prompt = {
            result["prompt_id"]: result
            for result in _read_json_results(run_dir / "verification.json")
        }

        assert spec.prompt_id in screening_by_prompt, (
            f"Missing held-out screening result for {spec.run_name}/{spec.prompt_id}"
        )
        assert spec.prompt_id in investigations_by_prompt, (
            f"Missing held-out investigation result for {spec.run_name}/{spec.prompt_id}"
        )
        assert spec.prompt_id in verifications_by_prompt, (
            f"Missing held-out verification result for {spec.run_name}/{spec.prompt_id}"
        )
        assert verifications_by_prompt[spec.prompt_id]["score"] >= 7, (
            f"Held-out exemplar must have verification>=7: {spec.run_name}/{spec.prompt_id}"
        )

        screening_result = screening_by_prompt[spec.prompt_id]
        investigation_result = investigations_by_prompt[spec.prompt_id]
        findings = investigation_result["structured_findings"]
        completions = screening_result["completions"]
        assert 0 <= spec.completion_index < len(completions), (
            f"Invalid completion_index={spec.completion_index} for "
            f"{spec.run_name}/{spec.prompt_id}; only {len(completions)} completions available"
        )

        exemplars.append(
            HeldoutFewShotExemplar(
                run_name=spec.run_name,
                prompt_id=spec.prompt_id,
                messages=_messages_from_screening_result(screening_result),
                chosen_completion=completions[spec.completion_index],
                question_text=findings["question"],
                answer_text=findings["first_person_answer"],
            )
        )

    return tuple(exemplars)


def load_model_understanding_eval_entries(
    run_dir: str,
    *,
    max_entries: int | None = None,
    min_verification_score: int = MIN_VERIFICATION_SCORE,
) -> list[ModelUnderstandingEvalEntry]:
    run_path = Path(run_dir)
    screening_by_prompt = _load_screening_results(run_path)
    investigations = {
        result["prompt_id"]: result
        for result in _read_json_results(run_path / "investigations.json")
    }
    verifications = {
        result["prompt_id"]: result
        for result in _read_json_results(run_path / "verification.json")
    }

    entries: list[ModelUnderstandingEvalEntry] = []
    for prompt_id in sorted(investigations):
        if prompt_id not in verifications:
            continue
        verification_score = verifications[prompt_id]["score"]
        if verification_score < min_verification_score:
            continue

        assert prompt_id in screening_by_prompt, f"Missing screening result for {prompt_id}"
        screening_result = screening_by_prompt[prompt_id]
        investigation = investigations[prompt_id]
        findings = investigation["structured_findings"]

        behavior_completion_indices = screening_result["behavior_completion_indices"]
        assert behavior_completion_indices, f"Empty behavior_completion_indices for {prompt_id}"
        chosen_completion_index = behavior_completion_indices[0] - 1
        second_person_question = screening_result["second_person_question"]
        assert second_person_question, f"Missing second_person_question for {prompt_id}"

        entries.append(
            ModelUnderstandingEvalEntry(
                prompt_id=prompt_id,
                interest_score=screening_result["interest_score"],
                verification_score=verification_score,
                messages=_messages_from_screening_result(screening_result),
                chosen_completion_index=chosen_completion_index,
                chosen_completion=screening_result["completions"][chosen_completion_index],
                second_person_question=second_person_question,
                third_person_question=screening_result["question"],
                first_person_answer=findings["first_person_answer"],
                third_person_answer=findings["answer"],
            )
        )

    assert entries, f"No usable eval entries found in {run_dir}"
    if max_entries is not None:
        entries = entries[:max_entries]
    return entries


def _select_question(
    entry: ModelUnderstandingEvalEntry,
    question_mode: Literal["second_person", "third_person"],
) -> str:
    if question_mode == "second_person":
        return entry.second_person_question
    if question_mode == "third_person":
        return entry.third_person_question
    raise ValueError(f"Unsupported question_mode: {question_mode}")


def _select_reference_answer(
    entry: ModelUnderstandingEvalEntry,
    answer_mode: Literal["first_person", "third_person"],
) -> str:
    if answer_mode == "first_person":
        return entry.first_person_answer
    if answer_mode == "third_person":
        return entry.third_person_answer
    raise ValueError(f"Unsupported answer_mode: {answer_mode}")


def build_context_messages(
    entry: ModelUnderstandingEvalEntry,
    *,
    question_mode: Literal["second_person", "third_person"],
) -> list[dict[str, str]]:
    question_text = _select_question(entry, question_mode)
    messages = [{"role": message.role, "content": message.content} for message in entry.messages]
    messages.append({"role": "assistant", "content": entry.chosen_completion})
    messages.append({"role": "user", "content": question_text})
    return messages


def tokenize_context_messages(
    tokenizer,
    messages: list[dict[str, str]],
) -> list[int]:
    return tokenize_chat_messages(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def select_activation_window(
    full_context_token_count: int,
    *,
    max_context_tokens: int | None,
) -> ActivationWindow:
    if max_context_tokens is None or full_context_token_count <= max_context_tokens:
        positions = tuple(range(full_context_token_count))
        selected_context_token_count = full_context_token_count
        return ActivationWindow(
            positions=positions,
            mode="full_context",
            used_full_context=True,
            full_context_token_count=full_context_token_count,
            selected_context_token_count=selected_context_token_count,
        )

    start = full_context_token_count - max_context_tokens
    positions = tuple(range(start, full_context_token_count))
    return ActivationWindow(
        positions=positions,
        mode=f"last_{max_context_tokens}",
        used_full_context=False,
        full_context_token_count=full_context_token_count,
        selected_context_token_count=max_context_tokens,
    )


def build_model_understanding_text_baseline_inputs(
    entries: list[ModelUnderstandingEvalEntry],
    tokenizer,
    *,
    prompt_style: Literal["plain", "heldout_fewshot", "suffix_heldout_fewshot"],
) -> tuple[list[TextBaselineInput], list[EvalMetadata]]:
    baseline_inputs: list[TextBaselineInput] = []
    metadata: list[EvalMetadata] = []
    heldout_exemplars = (
        _load_heldout_fewshot_exemplars()
        if prompt_style in {"heldout_fewshot", "suffix_heldout_fewshot"}
        else ()
    )

    for entry in entries:
        question_text = _select_question(entry, TEXT_BASELINE_QUESTION_MODE)
        reference_answer = _select_reference_answer(entry, TEXT_BASELINE_ANSWER_MODE)
        context_messages = build_context_messages(entry, question_mode=TEXT_BASELINE_QUESTION_MODE)
        prompt_messages: list[dict[str, str]]
        if prompt_style == "plain":
            prompt_messages = context_messages
        elif prompt_style == "heldout_fewshot":
            prompt_messages = [{"role": "system", "content": TEXT_BASELINE_SYSTEM_PROMPT}]
            for exemplar in heldout_exemplars:
                prompt_messages.append(
                    {
                        "role": "user",
                        "content": _format_heldout_fewshot_user_message(exemplar),
                    }
                )
                prompt_messages.append(
                    {
                        "role": "assistant",
                        "content": exemplar.answer_text,
                    }
                )
            prompt_messages.extend(context_messages)
        elif prompt_style == "suffix_heldout_fewshot":
            prompt_messages = [{"role": message["role"], "content": message["content"]} for message in context_messages]
            prompt_messages[-1]["content"] = _format_heldout_fewshot_suffix(
                question_text,
                heldout_exemplars,
            )
        else:
            raise ValueError(f"Unsupported prompt_style: {prompt_style}")

        prompt_token_ids = tokenize_context_messages(tokenizer, prompt_messages)
        full_context_token_count = len(prompt_token_ids)
        prompt_text = tokenizer.decode(prompt_token_ids, skip_special_tokens=False)

        baseline_inputs.append(
            TextBaselineInput(
                prompt_text=prompt_text,
                ground_truth=reference_answer,
                prompt_name=TEXT_BASELINE_QUESTION_MODE,
                variant="full_context",
            )
        )
        metadata.append(
            EvalMetadata(
                prompt_id=entry.prompt_id,
                interest_score=entry.interest_score,
                verification_score=entry.verification_score,
                chosen_completion_index=entry.chosen_completion_index,
                question_text=question_text,
                reference_answer=reference_answer,
                question_mode=TEXT_BASELINE_QUESTION_MODE,
                answer_mode=TEXT_BASELINE_ANSWER_MODE,
                full_context_token_count=full_context_token_count,
                selected_context_token_count=None,
                activation_window_mode=None,
                used_full_context=None,
            )
        )

    return baseline_inputs, metadata


def _load_heldout_fewshot_exemplars_third_person() -> tuple[HeldoutFewShotExemplar, ...]:
    """Load heldout few-shot exemplars using third-person question and answer."""
    exemplars: list[HeldoutFewShotExemplar] = []
    runs_root = Path("data_pipelines/model_understanding/runs")

    for spec in HELDOUT_TEXT_BASELINE_EXEMPLAR_SPECS:
        run_dir = runs_root / spec.run_name
        screening_by_prompt = {
            result["prompt_id"]: result
            for result in _read_json_results(run_dir / "screening.json")
        }
        investigations_by_prompt = {
            result["prompt_id"]: result
            for result in _read_json_results(run_dir / "investigations.json")
        }
        verifications_by_prompt = {
            result["prompt_id"]: result
            for result in _read_json_results(run_dir / "verification.json")
        }

        assert spec.prompt_id in screening_by_prompt
        assert spec.prompt_id in investigations_by_prompt
        assert spec.prompt_id in verifications_by_prompt
        assert verifications_by_prompt[spec.prompt_id]["score"] >= 7

        screening_result = screening_by_prompt[spec.prompt_id]
        investigation_result = investigations_by_prompt[spec.prompt_id]
        findings = investigation_result["structured_findings"]
        completions = screening_result["completions"]
        assert 0 <= spec.completion_index < len(completions)

        exemplars.append(
            HeldoutFewShotExemplar(
                run_name=spec.run_name,
                prompt_id=spec.prompt_id,
                messages=_messages_from_screening_result(screening_result),
                chosen_completion=completions[spec.completion_index],
                question_text=findings["question"],
                answer_text=findings["answer"],
            )
        )

    return tuple(exemplars)


def _format_claude_baseline_user_message(
    transcript_text: str,
    question_text: str,
) -> str:
    return (
        f"Transcript:\n{transcript_text}\n\n"
        f"Question: {question_text}"
    )


def build_model_understanding_claude_baseline_inputs(
    entries: list[ModelUnderstandingEvalEntry],
) -> tuple[list[ClaudeBaselineInput], list[EvalMetadata]]:
    heldout_exemplars = _load_heldout_fewshot_exemplars_third_person()

    # Build few-shot prefix messages (same for every entry)
    fewshot_messages: list[dict[str, str]] = []
    for exemplar in heldout_exemplars:
        exemplar_transcript_messages = [
            {"role": msg.role, "content": msg.content} for msg in exemplar.messages
        ]
        exemplar_transcript_messages.append({"role": "assistant", "content": exemplar.chosen_completion})
        exemplar_transcript = _format_transcript_for_prompt(exemplar_transcript_messages)
        fewshot_messages.append({
            "role": "user",
            "content": _format_claude_baseline_user_message(exemplar_transcript, exemplar.question_text),
        })
        fewshot_messages.append({
            "role": "assistant",
            "content": exemplar.answer_text,
        })

    baseline_inputs: list[ClaudeBaselineInput] = []
    metadata: list[EvalMetadata] = []

    for entry in entries:
        question_text = _select_question(entry, CLAUDE_BASELINE_QUESTION_MODE)
        reference_answer = _select_reference_answer(entry, CLAUDE_BASELINE_ANSWER_MODE)

        transcript_messages = [
            {"role": msg.role, "content": msg.content} for msg in entry.messages
        ]
        transcript_messages.append({"role": "assistant", "content": entry.chosen_completion})
        transcript_text = _format_transcript_for_prompt(transcript_messages)

        messages = list(fewshot_messages)
        messages.append({
            "role": "user",
            "content": _format_claude_baseline_user_message(transcript_text, question_text),
        })

        baseline_inputs.append(
            ClaudeBaselineInput(
                messages=messages,
                ground_truth=reference_answer,
                prompt_name=CLAUDE_BASELINE_QUESTION_MODE,
            )
        )
        metadata.append(
            EvalMetadata(
                prompt_id=entry.prompt_id,
                interest_score=entry.interest_score,
                verification_score=entry.verification_score,
                chosen_completion_index=entry.chosen_completion_index,
                question_text=question_text,
                reference_answer=reference_answer,
                question_mode=CLAUDE_BASELINE_QUESTION_MODE,
                answer_mode=CLAUDE_BASELINE_ANSWER_MODE,
                full_context_token_count=0,
                selected_context_token_count=None,
                activation_window_mode=None,
                used_full_context=None,
            )
        )

    return baseline_inputs, metadata


def _format_ao_fewshot_prefix(
    heldout_exemplars: tuple[HeldoutFewShotExemplar, ...],
) -> str:
    lines: list[str] = [
        "Here are examples of good answers to questions about model behavior:",
        "",
    ]
    for idx, exemplar in enumerate(heldout_exemplars, start=1):
        lines.append(f"Example {idx}")
        lines.append(f"Question: {exemplar.question_text}")
        lines.append(f"Good answer: {exemplar.answer_text}")
        lines.append("")
    lines.append("Now answer the following question:")
    lines.append("")
    return "\n".join(lines)


def build_model_understanding_ao_prompt_infos(
    entries: list[ModelUnderstandingEvalEntry],
    tokenizer,
    *,
    max_context_tokens: int | None,
    ao_prompt_style: Literal["plain", "fewshot"] = "plain",
) -> tuple[list[VerbalizerInputInfo], list[EvalMetadata]]:
    prompt_infos: list[VerbalizerInputInfo] = []
    metadata: list[EvalMetadata] = []

    fewshot_prefix = ""
    if ao_prompt_style == "fewshot":
        heldout_exemplars = _load_heldout_fewshot_exemplars_third_person()
        fewshot_prefix = _format_ao_fewshot_prefix(heldout_exemplars)

    for entry in entries:
        question_text = _select_question(entry, AO_QUESTION_MODE)
        reference_answer = _select_reference_answer(entry, AO_ANSWER_MODE)
        context_messages = build_context_messages(entry, question_mode=AO_QUESTION_MODE)
        context_token_ids = tokenize_context_messages(tokenizer, context_messages)
        activation_window = select_activation_window(
            len(context_token_ids),
            max_context_tokens=max_context_tokens,
        )

        verbalizer_prompt = fewshot_prefix + question_text

        prompt_infos.append(
            VerbalizerInputInfo(
                context_token_ids=context_token_ids,
                positions=list(activation_window.positions),
                ground_truth=reference_answer,
                verbalizer_prompt=verbalizer_prompt,
            )
        )
        metadata.append(
            EvalMetadata(
                prompt_id=entry.prompt_id,
                interest_score=entry.interest_score,
                verification_score=entry.verification_score,
                chosen_completion_index=entry.chosen_completion_index,
                question_text=question_text,
                reference_answer=reference_answer,
                question_mode=AO_QUESTION_MODE,
                answer_mode=AO_ANSWER_MODE,
                full_context_token_count=activation_window.full_context_token_count,
                selected_context_token_count=activation_window.selected_context_token_count,
                activation_window_mode=activation_window.mode,
                used_full_context=activation_window.used_full_context,
            )
        )

    return prompt_infos, metadata


def _parse_judge_result(text: str) -> JudgeResult:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1]
        stripped = stripped.rsplit("```", 1)[0]
        stripped = stripped.strip()
    payload = json.loads(stripped)
    return JudgeResult(
        specificity=int(payload["specificity"]),
        correctness=int(payload["correctness"]),
        reasoning=payload["reasoning"],
    )


JUDGE_MAX_RETRIES = 3


async def judge_single_response(
    client: anthropic.AsyncAnthropic,
    *,
    metadata: EvalMetadata,
    response_text: str,
    semaphore: asyncio.Semaphore,
) -> JudgeResult:
    user_message = JUDGE_USER_TEMPLATE.format(
        question_text=metadata.question_text,
        reference_answer=metadata.reference_answer,
        response_text=response_text,
    )

    last_error = None
    for attempt in range(JUDGE_MAX_RETRIES):
        try:
            async with semaphore:
                response = await client.messages.create(
                    model=JUDGE_MODEL,
                    max_tokens=300,
                    system=JUDGE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )
            return _parse_judge_result(response.content[0].text)
        except Exception as e:
            last_error = e
            if attempt < JUDGE_MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)

    raise last_error  # type: ignore[misc]


async def judge_response_texts(
    response_texts: list[str | None],
    metadata: list[EvalMetadata],
    *,
    concurrency: int,
) -> list[ScoredJudgeResult]:
    assert len(response_texts) == len(metadata), (
        f"response_texts length {len(response_texts)} != metadata length {len(metadata)}"
    )

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    task_metadata: list[tuple[EvalMetadata, str]] = []

    for response_text, meta in zip(response_texts, metadata, strict=True):
        if response_text is None:
            continue
        tasks.append(
            judge_single_response(
                client,
                metadata=meta,
                response_text=response_text,
                semaphore=semaphore,
            )
        )
        task_metadata.append((meta, response_text))

    print(f"Judging {len(tasks)} responses with {JUDGE_MODEL} (concurrency={concurrency})...")
    pbar = tqdm(total=len(tasks), desc="LLM judge")

    async def _track(coro):
        result = await coro
        pbar.update(1)
        return result

    judge_results = await asyncio.gather(*[_track(task) for task in tasks], return_exceptions=True)
    pbar.close()

    scored_results: list[ScoredJudgeResult] = []
    for idx, ((meta, response_text), judge_result) in enumerate(zip(task_metadata, judge_results, strict=True)):
        if isinstance(judge_result, Exception):
            print(f"Judge error for {meta.prompt_id} (index={idx}): {judge_result}")
            continue
        scored_results.append(
            ScoredJudgeResult(
                prompt_id=meta.prompt_id,
                interest_score=meta.interest_score,
                verification_score=meta.verification_score,
                chosen_completion_index=meta.chosen_completion_index,
                question_text=meta.question_text,
                reference_answer=meta.reference_answer,
                question_mode=meta.question_mode,
                answer_mode=meta.answer_mode,
                full_context_token_count=meta.full_context_token_count,
                selected_context_token_count=meta.selected_context_token_count,
                activation_window_mode=meta.activation_window_mode,
                used_full_context=meta.used_full_context,
                response_text=response_text,
                specificity=judge_result.specificity,
                correctness=judge_result.correctness,
                reasoning=judge_result.reasoning,
            )
        )

    return scored_results


async def judge_ao_responses(
    results: list[VerbalizerResults],
    metadata: list[EvalMetadata],
    *,
    concurrency: int,
) -> list[ScoredJudgeResult]:
    return await judge_response_texts(
        [get_first_ao_response(result) for result in results],
        metadata,
        concurrency=concurrency,
    )


async def judge_text_baseline_responses(
    results: list[TextBaselineGenerationResult],
    metadata: list[EvalMetadata],
    *,
    concurrency: int,
) -> list[ScoredJudgeResult]:
    return await judge_response_texts(
        [result.response_text for result in results],
        metadata,
        concurrency=concurrency,
    )


async def judge_claude_baseline_responses(
    results: list[ClaudeBaselineResult],
    metadata: list[EvalMetadata],
    *,
    concurrency: int,
) -> list[ScoredJudgeResult]:
    return await judge_response_texts(
        [result.response_text if result.response_text else None for result in results],
        metadata,
        concurrency=concurrency,
    )


def _slugify_metric_value(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _compute_metric_block(scored_results: list[dict[str, Any]]) -> dict[str, float]:
    if not scored_results:
        return {}

    specificities = [result["specificity"] for result in scored_results]
    correctnesses = [result["correctness"] for result in scored_results]

    metrics: dict[str, float] = {
        "mean_specificity": sum(specificities) / len(specificities),
        "mean_correctness": sum(correctnesses) / len(correctnesses),
        "specificity_>=3_rate": sum(1 for value in specificities if value >= 3) / len(specificities),
        "specificity_>=4_rate": sum(1 for value in specificities if value >= 4) / len(specificities),
        "correctness_>=3_rate": sum(1 for value in correctnesses if value >= 3) / len(correctnesses),
        "correctness_>=4_rate": sum(1 for value in correctnesses if value >= 4) / len(correctnesses),
        "joint_success_>=4_rate": (
            sum(
                1
                for result in scored_results
                if result["specificity"] >= 4 and result["correctness"] >= 4
            )
            / len(scored_results)
        ),
        "mean_full_context_token_count": (
            sum(result["full_context_token_count"] for result in scored_results) / len(scored_results)
        ),
        "num_scored": float(len(scored_results)),
    }

    selected_counts = [
        result["selected_context_token_count"]
        for result in scored_results
        if result["selected_context_token_count"] is not None
    ]
    if selected_counts:
        metrics["mean_selected_context_token_count"] = sum(selected_counts) / len(selected_counts)

    return metrics


def _append_breakdown_metrics(
    metrics: dict[str, float],
    scored_results: list[dict[str, Any]],
    *,
    field_name: str,
    prefix: str,
) -> None:
    field_values = sorted({
        result[field_name]
        for result in scored_results
        if field_name in result and result[field_name] is not None
    })
    for field_value in field_values:
        subset = [result for result in scored_results if result.get(field_name) == field_value]
        subset_metrics = _compute_metric_block(subset)
        slug = _slugify_metric_value(str(field_value))
        for metric_name, metric_value in subset_metrics.items():
            metrics[f"{prefix}_{slug}_{metric_name}"] = metric_value


def compute_judge_metrics(scored_results: list[dict[str, Any]]) -> dict[str, float]:
    metrics = _compute_metric_block(scored_results)
    if not metrics:
        return metrics

    _append_breakdown_metrics(
        metrics,
        scored_results,
        field_name="question_mode",
        prefix="question_mode",
    )
    _append_breakdown_metrics(
        metrics,
        scored_results,
        field_name="activation_window_mode",
        prefix="activation_window",
    )
    return metrics


def print_sample_results(scored_results: list[dict[str, Any]]) -> None:
    print("\n  Sample results:")
    for result in scored_results[:8]:
        window_mode = result["activation_window_mode"] or "text_baseline"
        print(
            f"    [{result['prompt_id']}] "
            f"spec={result['specificity']} corr={result['correctness']} "
            f"window={window_mode:<12} "
            f"response={result['response_text'][:110]!r}"
        )


def _print_metrics(metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.3f}")
        else:
            print(f"    {key}: {value}")


def run_model_understanding_ao_eval(
    *,
    model_name: str,
    model,
    tokenizer,
    device,
    run_dir: str,
    verbalizer_lora_paths: list[str],
    output_dir: str | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    eval_batch_size: int = DEFAULT_AO_EVAL_BATCH_SIZE,
    max_entries: int | None = None,
    judge_concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
    ao_max_context_tokens: int | None = DEFAULT_AO_MAX_CONTEXT_TOKENS,
    ao_prompt_style: Literal["plain", "fewshot"] = "plain",
    min_verification_score: int = MIN_VERIFICATION_SCORE,
) -> dict[str, Any]:
    if generation_kwargs is None:
        generation_kwargs = GENERATION_KWARGS

    entries = load_model_understanding_eval_entries(
        run_dir, max_entries=max_entries, min_verification_score=min_verification_score,
    )
    prompt_infos, entry_metadata = build_model_understanding_ao_prompt_infos(
        entries,
        tokenizer,
        max_context_tokens=ao_max_context_tokens,
        ao_prompt_style=ao_prompt_style,
    )

    def score_fn(
        results: list[VerbalizerResults],
        metadata: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        del metadata
        return [
            asdict(result)
            for result in asyncio.run(
                judge_ao_responses(
                    results=results,
                    metadata=entry_metadata,
                    concurrency=judge_concurrency,
                )
            )
        ]

    return run_verbalizer_generation_eval_loop(
        eval_name="model_understanding",
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=model_name,
        eval_batch_size=eval_batch_size,
        generation_kwargs=generation_kwargs,
        prompt_infos=prompt_infos,
        entry_metadata=[asdict(meta) for meta in entry_metadata],
        score_fn=score_fn,
        metrics_fn=compute_judge_metrics,
        num_entries=len(entries),
        verbalizer_lora_paths=verbalizer_lora_paths,
        output_dir=output_dir,
        print_sample_fn=print_sample_results,
        extra_output_data={
            "run_dir": run_dir,
            "judge_model": JUDGE_MODEL,
            "judge_concurrency": judge_concurrency,
            "question_mode": AO_QUESTION_MODE,
            "answer_mode": AO_ANSWER_MODE,
            "min_interest_score": MIN_INTEREST_SCORE,
            "min_verification_score": min_verification_score,
            "ao_max_context_tokens": ao_max_context_tokens,
            "ao_prompt_style": ao_prompt_style,
            "entry_metadata": [asdict(meta) for meta in entry_metadata],
        },
    )


def run_model_understanding_claude_baseline_eval(
    *,
    run_dir: str,
    output_dir: str | None = None,
    max_entries: int | None = None,
    judge_concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
    claude_model: str = DEFAULT_CLAUDE_BASELINE_MODEL,
    claude_max_tokens: int = DEFAULT_CLAUDE_BASELINE_MAX_TOKENS,
    claude_concurrency: int = DEFAULT_CLAUDE_BASELINE_CONCURRENCY,
    min_verification_score: int = MIN_VERIFICATION_SCORE,
) -> dict[str, Any]:
    entries = load_model_understanding_eval_entries(
        run_dir, max_entries=max_entries, min_verification_score=min_verification_score,
    )
    baseline_inputs, entry_metadata = build_model_understanding_claude_baseline_inputs(entries)

    def score_fn(
        results: list[ClaudeBaselineResult],
        metadata: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        del metadata
        return [
            asdict(result)
            for result in asyncio.run(
                judge_claude_baseline_responses(
                    results=results,
                    metadata=entry_metadata,
                    concurrency=judge_concurrency,
                )
            )
        ]

    return run_claude_baseline_eval_loop(
        eval_name="model_understanding",
        baseline_inputs=baseline_inputs,
        entry_metadata=[asdict(meta) for meta in entry_metadata],
        system_prompt=CLAUDE_BASELINE_SYSTEM_PROMPT,
        claude_model=claude_model,
        max_tokens=claude_max_tokens,
        concurrency=claude_concurrency,
        score_fn=score_fn,
        metrics_fn=compute_judge_metrics,
        num_entries=len(entries),
        output_dir=output_dir,
        print_sample_fn=print_sample_results,
        extra_output_data={
            "run_dir": run_dir,
            "judge_model": JUDGE_MODEL,
            "judge_concurrency": judge_concurrency,
            "question_mode": CLAUDE_BASELINE_QUESTION_MODE,
            "answer_mode": CLAUDE_BASELINE_ANSWER_MODE,
            "min_interest_score": MIN_INTEREST_SCORE,
            "min_verification_score": min_verification_score,
            "heldout_exemplar_specs": [
                asdict(spec) for spec in HELDOUT_TEXT_BASELINE_EXEMPLAR_SPECS
            ],
            "entry_metadata": [asdict(meta) for meta in entry_metadata],
        },
    )


def run_model_understanding_text_baseline_eval(
    *,
    model_name: str,
    tokenizer,
    backend,
    run_dir: str,
    hf_model=None,
    device=None,
    output_dir: str | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    eval_batch_size: int = DEFAULT_TEXT_BASELINE_EVAL_BATCH_SIZE,
    max_entries: int | None = None,
    judge_concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
    text_baseline_prompt_style: Literal[
        "plain",
        "heldout_fewshot",
        "suffix_heldout_fewshot",
    ] = DEFAULT_TEXT_BASELINE_PROMPT_STYLE,
    vllm_lora_path: str | None = None,
    enforce_eager: bool = True,
    min_verification_score: int = MIN_VERIFICATION_SCORE,
) -> dict[str, Any]:
    if generation_kwargs is None:
        generation_kwargs = GENERATION_KWARGS

    entries = load_model_understanding_eval_entries(
        run_dir, max_entries=max_entries, min_verification_score=min_verification_score,
    )
    baseline_inputs, entry_metadata = build_model_understanding_text_baseline_inputs(
        entries,
        tokenizer,
        prompt_style=text_baseline_prompt_style,
    )

    def score_fn(
        results: list[TextBaselineGenerationResult],
        metadata: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        del metadata
        return [
            asdict(result)
            for result in asyncio.run(
                judge_text_baseline_responses(
                    results=results,
                    metadata=entry_metadata,
                    concurrency=judge_concurrency,
                )
            )
        ]

    return run_text_baseline_eval_loop(
        eval_name="model_understanding",
        baseline_inputs=baseline_inputs,
        entry_metadata=[asdict(meta) for meta in entry_metadata],
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
        print_sample_fn=print_sample_results,
        vllm_lora_path=vllm_lora_path,
        enforce_eager=enforce_eager,
        extra_output_data={
            "run_dir": run_dir,
            "judge_model": JUDGE_MODEL,
            "judge_concurrency": judge_concurrency,
            "question_mode": TEXT_BASELINE_QUESTION_MODE,
            "answer_mode": TEXT_BASELINE_ANSWER_MODE,
            "min_interest_score": MIN_INTEREST_SCORE,
            "min_verification_score": min_verification_score,
            "text_baseline_prompt_style": text_baseline_prompt_style,
            "vllm_lora_path": vllm_lora_path,
            "heldout_text_baseline_exemplar_specs": [
                asdict(spec) for spec in HELDOUT_TEXT_BASELINE_EXEMPLAR_SPECS
            ],
            "entry_metadata": [asdict(meta) for meta in entry_metadata],
        },
    )


def _resolve_run_dir_and_model_name(args: argparse.Namespace) -> tuple[str, str]:
    run_dir = args.run_dir
    model_name = args.model_name

    if args.training_config is not None:
        payload = json.loads(Path(args.training_config).read_text())
        if run_dir is None:
            run_dir = payload["run_dir"]
        if model_name is None:
            model_name = payload["model_name"]

    assert run_dir is not None, "Provide --run-dir or --training-config"
    assert model_name is not None, "Provide --model-name or --training-config"
    return run_dir, model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ao", "text_baseline", "text_sft", "claude_baseline"], required=True)
    parser.add_argument("--training-config", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--generation-max-new-tokens", type=int, default=GENERATION_KWARGS["max_new_tokens"])
    parser.add_argument("--judge-concurrency", type=int, default=DEFAULT_JUDGE_CONCURRENCY)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--baseline-backend", choices=["hf", "vllm"], default="vllm")
    parser.add_argument(
        "--text-baseline-prompt-style",
        choices=["plain", "heldout_fewshot", "suffix_heldout_fewshot"],
        default=DEFAULT_TEXT_BASELINE_PROMPT_STYLE,
    )
    parser.add_argument("--ao-max-context-tokens", type=int, default=DEFAULT_AO_MAX_CONTEXT_TOKENS)
    parser.add_argument(
        "--ao-prompt-style",
        choices=["plain", "fewshot"],
        default="plain",
        help="AO prompt style: plain (question only) or fewshot (prepend example Q&A pairs).",
    )
    parser.add_argument(
        "--verbalizer-lora-path",
        action="append",
        default=None,
        help="AO LoRA adapter path (can be repeated). Required for --mode ao.",
    )
    parser.add_argument(
        "--text-sft-lora-path",
        type=str,
        default=None,
        help="Path to text SFT LoRA checkpoint for --mode text_sft.",
    )
    parser.add_argument(
        "--no-enforce-eager",
        action="store_true",
        help="Disable enforce_eager for vLLM (slower startup but uses CUDA graphs).",
    )
    parser.add_argument("--min-verification-score", type=int, default=MIN_VERIFICATION_SCORE)
    parser.add_argument(
        "--claude-model",
        type=str,
        default=DEFAULT_CLAUDE_BASELINE_MODEL,
        help="Claude model for --mode claude_baseline.",
    )
    parser.add_argument(
        "--claude-concurrency",
        type=int,
        default=DEFAULT_CLAUDE_BASELINE_CONCURRENCY,
        help="Concurrency for Claude API calls in --mode claude_baseline.",
    )
    args = parser.parse_args()

    if args.mode == "claude_baseline":
        # Claude baseline doesn't need a local model or tokenizer
        run_dir = args.run_dir
        if run_dir is None and args.training_config is not None:
            payload = json.loads(Path(args.training_config).read_text())
            run_dir = payload.get("run_dir", run_dir)
        assert run_dir is not None, "Provide --run-dir or --training-config for claude_baseline"
        run_name = Path(run_dir).name
        claude_slug = args.claude_model.replace("/", "_").replace("-", "_")

        summary = run_model_understanding_claude_baseline_eval(
            run_dir=run_dir,
            output_dir=(
                args.output_dir
                if args.output_dir is not None
                else f"experiments/model_understanding_eval/qwen3_14b_run_test_v2_judge/{claude_slug}_baseline"
            ),
            max_entries=args.max_entries,
            judge_concurrency=args.judge_concurrency,
            claude_model=args.claude_model,
            claude_concurrency=args.claude_concurrency,
            min_verification_score=args.min_verification_score,
        )

        print(f"\n=== Claude Baseline Summary ===")
        print(json.dumps(summary, indent=2))
    else:
        run_dir, model_name = _resolve_run_dir_and_model_name(args)
        generation_kwargs = {
            **GENERATION_KWARGS,
            "max_new_tokens": args.generation_max_new_tokens,
        }
        run_name = Path(run_dir).name
        model_slug = model_name.split("/")[-1].replace(".", "_")

        if args.mode == "ao":
            assert args.verbalizer_lora_path, "Provide at least one --verbalizer-lora-path for AO mode"
            run_default_eval(
                eval_name=f"model_understanding_ao_{run_name}",
                run_eval_fn=run_model_understanding_ao_eval,
                model_name=model_name,
                run_eval_kwargs={
                    "run_dir": run_dir,
                    "verbalizer_lora_paths": args.verbalizer_lora_path,
                    "output_dir": (
                        args.output_dir
                        if args.output_dir is not None
                        else f"experiments/model_understanding_eval/{run_name}/{model_slug}/ao"
                    ),
                    "generation_kwargs": generation_kwargs,
                    "eval_batch_size": args.eval_batch_size or DEFAULT_AO_EVAL_BATCH_SIZE,
                    "max_entries": args.max_entries,
                    "judge_concurrency": args.judge_concurrency,
                    "ao_max_context_tokens": (
                        None if args.ao_max_context_tokens <= 0 else args.ao_max_context_tokens
                    ),
                    "ao_prompt_style": args.ao_prompt_style,
                    "min_verification_score": args.min_verification_score,
                },
            )
        elif args.mode == "text_sft":
            assert args.text_sft_lora_path, "Provide --text-sft-lora-path for text_sft mode"
            lora_slug = Path(args.text_sft_lora_path).parent.name
            run_default_text_baseline_eval(
                eval_name=f"model_understanding_text_sft_{run_name}",
                run_eval_fn=run_model_understanding_text_baseline_eval,
                model_name=model_name,
                backend="vllm",
                run_eval_kwargs={
                    "run_dir": run_dir,
                    "output_dir": (
                        args.output_dir
                        if args.output_dir is not None
                        else f"experiments/model_understanding_eval/{run_name}/{model_slug}/text_sft/{lora_slug}"
                    ),
                    "generation_kwargs": generation_kwargs,
                    "eval_batch_size": args.eval_batch_size or DEFAULT_TEXT_BASELINE_EVAL_BATCH_SIZE,
                    "max_entries": args.max_entries,
                    "judge_concurrency": args.judge_concurrency,
                    "text_baseline_prompt_style": "plain",
                    "vllm_lora_path": args.text_sft_lora_path,
                    "enforce_eager": not args.no_enforce_eager,
                    "min_verification_score": args.min_verification_score,
                },
            )
        else:
            run_default_text_baseline_eval(
                eval_name=f"model_understanding_text_baseline_{run_name}",
                run_eval_fn=run_model_understanding_text_baseline_eval,
                model_name=model_name,
                backend=args.baseline_backend,
                run_eval_kwargs={
                    "run_dir": run_dir,
                    "output_dir": (
                        args.output_dir
                        if args.output_dir is not None
                        else f"experiments/model_understanding_eval/{run_name}/{model_slug}/text_baseline"
                    ),
                    "generation_kwargs": generation_kwargs,
                    "eval_batch_size": args.eval_batch_size or DEFAULT_TEXT_BASELINE_EVAL_BATCH_SIZE,
                    "max_entries": args.max_entries,
                    "judge_concurrency": args.judge_concurrency,
                    "text_baseline_prompt_style": args.text_baseline_prompt_style,
                    "enforce_eager": not args.no_enforce_eager,
                    "min_verification_score": args.min_verification_score,
                },
            )
