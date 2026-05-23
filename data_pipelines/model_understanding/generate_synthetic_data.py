"""
Generate synthetic training data from investigation pipeline outputs.

Two types of synthetic data:

1. **Behavior prediction**: Given the user prompt (before the model responds),
   ask "how would you respond?" Ground truth is derived from Opus's
   behavior_summary and baseline_behavior, rephrased to first person by Sonnet.

2. **Counterfactual prediction**: Given user prompt + model's completion,
   ask "how would changing X affect your response?" Ground truth is derived
   from experiment_log counterfactual results, synthesized by Sonnet.

Supports both async API (for iteration) and batch API (for scale).

Usage:
    source .env

    # Quick test — 10 examples, async
    .venv/bin/python data_pipelines/model_understanding/generate_synthetic_data.py \
        --investigations data_pipelines/model_understanding/runs/qwen3_32b_run/investigations.json \
        --screening data_pipelines/model_understanding/runs/qwen3_32b_run/screening.json \
        --verification data_pipelines/model_understanding/runs/qwen3_32b_run/verification.json \
        --output data_pipelines/model_understanding/runs/qwen3_32b_run/synthetic_data.json \
        --n 10

    # Full run — batch API
    .venv/bin/python data_pipelines/model_understanding/generate_synthetic_data.py \
        --investigations data_pipelines/model_understanding/runs/qwen3_32b_run/investigations.json \
        --screening data_pipelines/model_understanding/runs/qwen3_32b_run/screening.json \
        --verification data_pipelines/model_understanding/runs/qwen3_32b_run/verification.json \
        --output data_pipelines/model_understanding/runs/qwen3_32b_run/synthetic_data.json \
        --batch
"""

import argparse
import asyncio
import functools
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

print = functools.partial(print, flush=True)

import anthropic
from data_pipelines.pipeline_utils import async_api_call, extract_tool_input, run_concurrent

MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 4096
THINKING_BUDGET = 2048
DEFAULT_CONCURRENCY = 30
DEFAULT_BATCH_CHUNK_SIZE = 10000
DEFAULT_COUNTERFACTUAL_EXPERIMENTS = 3
MIN_INTEREST_SCORE = 3
MIN_VERIFICATION_SCORE = 7

# Variations of the behavior prediction question (randomly sampled per example)
BEHAVIOR_PREDICTION_QUESTIONS = [
    "How would you respond to this prompt? Describe your likely approach, assumptions, and what your response would look like.",
    "Describe how you would approach responding to this prompt. What would your response contain and why?",
    "Before responding, describe what you would likely say and how you would frame your answer.",
    "What would your response to this prompt look like? Describe your approach and reasoning.",
    "Predict how you would respond to this. What assumptions would you make and what would you emphasize?",
    "Describe your likely response to this prompt — what approach would you take and what would you focus on?",
    "How would you handle this prompt? Describe what your response would contain and your reasoning.",
    "Walk through how you would respond to this. What would your answer look like?",
]

# Separators placed between the user message and the prediction question
BEHAVIOR_PREDICTION_SEPARATORS = [
    "\n\n---\n[End of user message. Do not respond to the above. Instead, answer the following question about how you would respond.]\n\n",
    "\n\n---\n[Do not respond to the above message. Instead, answer the following question.]\n\n",
    "\n\n---\n[Stop. Do not respond to the user message above. Instead, answer this question about your behavior.]\n\n",
    "\n\n---\n[Ignore the above request. Instead, describe how you would respond to it.]\n\n",
    "\n\n---\n[End of prompt. Rather than responding, answer the following question about how you would approach it.]\n\n",
]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

BEHAVIOR_PREDICTION_SYSTEM_PROMPT = """\
You are helping create training data for a language model's self-understanding.

You will see:
1. A user prompt (possibly multi-turn conversation ending with a user message)
2. A BEHAVIOR SUMMARY describing how the model actually responded across 10 completions
3. A BASELINE BEHAVIOR description from a causal investigation
4. The FIRST-PERSON ANSWER from the investigation (the model's explanation of its own behavior)

Your task: write a first-person prediction of how the model would respond to this \
prompt. The prediction should be written AS IF the model is introspecting before \
responding — describing what it would likely do and why.

Guidelines:
- Write in first person ("I would...", "I'd likely...")
- Be specific about the content and framing of the response, not just "I would answer the question"
- Include behavioral tendencies: what assumptions you'd make, what framing you'd choose, \
  what you'd emphasize
- Use the behavior summary and investigation answer as ground truth, but rephrase naturally \
  as a prediction (not a post-hoc explanation)
- Express rates as percentages ("about 80% of the time", "nearly always", "roughly 60%"), \
  NEVER as fractions out of a count ("8/10", "9 out of 10", "in 7 completions")
- Keep to 2-5 sentences
- Do NOT mention "completions", "experiments", "10 responses", or other meta-language \
  about the testing setup
- IMPORTANT: Do not claim knowledge the model wouldn't actually have. The model cannot \
  know what it doesn't know — so don't write things like "I don't know X stands for Y" \
  or "I'd fail to notice my error." Instead, describe what the model WOULD do positively: \
  "I'd interpret FE as Front-End Payment" rather than "I don't know FE means Finalize \
  Early." Describe the behavior, not the gap in knowledge that causes it.
- IMPORTANT: When the model's behavior involves making a mistake or giving incorrect \
  information, the prediction must flag it as incorrect — even if the model would state \
  it confidently. For example: "I'd confidently describe the offset as connecting \
  horizontal neighbors, though this interpretation would actually be incorrect" rather \
  than just "I'd confidently describe the offset as connecting horizontal neighbors." \
  Never train the model to present wrong information as if it were right."""

BEHAVIOR_PREDICTION_TOOL = {
    "name": "submit_prediction",
    "description": "Submit the behavior prediction",
    "input_schema": {
        "type": "object",
        "properties": {
            "prediction": {
                "type": "string",
                "description": (
                    "First-person prediction of how the model would respond. "
                    "2-5 sentences, specific about content and framing."
                ),
            },
        },
        "required": ["prediction"],
    },
}

COUNTERFACTUAL_PREDICTION_SYSTEM_PROMPT = """\
You are helping create training data for a language model's self-understanding.

You will see:
1. A user prompt (possibly multi-turn)
2. The model's ACTUAL RESPONSE to that prompt
3. A COUNTERFACTUAL CHANGE — a specific modification to the prompt
4. The COUNTERFACTUAL COMPLETIONS — how the model actually responded after the change
5. The BASELINE RATE and COUNTERFACTUAL RATE for the key behavior

Your task: write a first-person answer describing how the counterfactual change \
would affect the model's response. The answer should be written AS IF the model \
is predicting how the change would alter its behavior.

Guidelines:
- Write in first person ("If the prompt changed X, I would...")
- Be specific about what would change in the response and why
- Reference the actual behavioral shift (e.g., "I would go from doing X to doing Y")
- Express rates as percentages ("about 80% of the time", "nearly always", "roughly 60%"), \
  NEVER as fractions out of a count ("8/10", "9 out of 10", "in 7 completions")
- Keep to 2-4 sentences
- Do NOT mention "completions", "experiments", "10 responses", "baseline", \
  "re-sampling", or other meta-language about the testing setup. Do not say \
  "in the baseline scenario" or "compared to the baseline" — instead say \
  "in the original prompt" or "without this change"
- IMPORTANT: Do not claim knowledge the model wouldn't have. Don't write \
  "I don't actually know X" or "even though I'd be hallucinating." Describe \
  what the model WOULD do, not what it doesn't know.
- IMPORTANT: When the model's behavior involves making a mistake, flag it as \
  incorrect — even if the model would state it confidently. Never present \
  wrong information as if it were right.
- If the change has little effect, say so honestly ("This change wouldn't meaningfully \
  alter my response because...")"""

COUNTERFACTUAL_PREDICTION_TOOL = {
    "name": "submit_prediction",
    "description": "Submit the counterfactual behavior prediction",
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "A second-person question asking the model how the "
                    "counterfactual change would affect its response. E.g., "
                    "'How would your response change if the user had asked "
                    "about X instead of Y?' Keep it concise and specific."
                ),
            },
            "prediction": {
                "type": "string",
                "description": (
                    "First-person prediction of how the counterfactual change "
                    "would affect the response. 2-4 sentences."
                ),
            },
        },
        "required": ["question", "prediction"],
    },
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SyntheticExample:
    prompt_id: str
    example_type: str  # "behavior_prediction" or "counterfactual_prediction"
    messages: list[dict]  # original conversation messages
    chosen_completion: str  # model's actual completion (empty for behavior_prediction)
    chosen_completion_index: int  # 0-indexed into completions array (-1 for behavior_prediction)
    question: str  # the question asked
    answer: str  # the synthetic answer
    # Provenance
    interest_score: int
    verification_score: int
    experiment_index: int | None  # which experiment from experiment_log (counterfactual only)
    behavior_completion_indices: list[int]  # 1-indexed, from screening
    # Sonnet prompt (for debugging / spot-checking)
    sonnet_system_prompt: str = ""
    sonnet_user_prompt: str = ""


@dataclass
class InputExample:
    """Joined data from screening + investigation + verification for one prompt."""
    prompt_id: str
    interest_score: int
    verification_score: int
    messages: list[dict]
    user_message: str
    completions: list[str]
    behavior_completion_indices: list[int]
    behavior_summary: str
    second_person_question: str
    # From investigation
    first_person_answer: str
    baseline_behavior: str
    answer: str
    experiment_log: list[dict]


# ---------------------------------------------------------------------------
# Diverse format generation
# ---------------------------------------------------------------------------

DIVERSE_FORMAT_SYSTEM_PROMPT = """\
You are helping create training data for a language model's self-understanding.

You will see investigation data describing how a model behaves on a specific prompt, \
including causal analysis of what drives the behavior. Your task is to generate a \
question-answer pair about this behavior.

IMPORTANT — vary your response format. Do NOT always start answers with "I would" \
or "If". Here are examples of good diverse formats, but don't limit yourself to \
just these — feel free to use any natural format:

- Yes/no: "Would you refuse this request?" → "Yes. About 60% of the time..."
- Yes/no: "Would changing the names affect your response?" → "No. The names don't influence..."
- Key factor: "What most influences your lens choice?" → "The subject type. Portrait subjects trigger..."
- Short direct: "What camera would you pick?" → "Canon EOS R5 with 85mm lens, in about 80% of cases."
- Rate: "How often would you default to this?" → "About 80-90% of the time, regardless of..."
- Causal: "Why would you abandon that subplot?" → "Because the later scene dominates my attention..."

Guidelines:
- The question should be in second person ("Would you...", "What drives your...", etc.)
- Use the investigation data as ground truth — do not invent behavioral claims
- Express rates as percentages ("about 80% of the time"), NEVER as fractions out \
  of a count ("8/10", "9 out of 10")
- Do NOT mention "completions", "experiments", "10 responses", "baseline", \
  "re-sampling", or other meta-language about the testing setup
- Do not claim knowledge the model wouldn't have (don't say "I don't know X" — \
  describe what the model WOULD do positively)
- When the model's behavior involves mistakes, flag them as incorrect — never \
  present wrong information as if it were right
- Keep answers concise: 1-3 sentences"""

DIVERSE_FORMAT_TOOL = {
    "name": "submit_prediction",
    "description": "Submit the question and answer",
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "A second-person question about the model's behavior. "
                    "Specific and concrete."
                ),
            },
            "prediction": {
                "type": "string",
                "description": "Answer about the model's behavior. 1-3 sentences.",
            },
        },
        "required": ["question", "prediction"],
    },
}


def build_diverse_behavior_request(
    example: InputExample,
    target_answer: str | None,
    model: str = MODEL,
) -> dict:
    """Build a diverse-format behavior prediction request."""
    conversation = _format_conversation(example.messages)

    constraint = ""
    if target_answer == "no":
        constraint = (
            "\n\nFORMAT CONSTRAINT: Ask a yes/no question where the answer is NO. "
            "Ask about something that does NOT significantly affect the model's "
            "response, or a behavior the model does NOT exhibit. Use the causal "
            "analysis to find aspects that were tested but had no effect."
        )
    elif target_answer == "yes":
        constraint = (
            "\n\nFORMAT CONSTRAINT: Ask a yes/no question where the answer is YES."
        )

    user_content = f"""{conversation}

---

BEHAVIOR SUMMARY:
{example.behavior_summary}

BASELINE BEHAVIOR:
{example.baseline_behavior}

CAUSAL ANALYSIS:
{example.answer}

FIRST-PERSON EXPLANATION:
{example.first_person_answer}

---

Generate a question and answer about this model's behavior. \
Use the investigation data as ground truth.{constraint}"""

    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "thinking": {"type": "enabled", "budget_tokens": THINKING_BUDGET},
        "system": DIVERSE_FORMAT_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_content}],
        "tools": [DIVERSE_FORMAT_TOOL],
    }


def build_diverse_counterfactual_request(
    example: InputExample,
    experiment: dict,
    comp_idx: int,
    target_answer: str | None,
    model: str = MODEL,
) -> dict:
    """Build a diverse-format counterfactual prediction request."""
    conversation = _format_conversation(example.messages)
    chosen_completion = example.completions[comp_idx]

    if experiment.get("messages"):
        cf_conversation = _format_conversation(experiment["messages"])
        change_desc = f"The prompt was modified to:\n{cf_conversation}"
    elif experiment.get("prompt"):
        change_desc = f"The user message was changed to:\n{experiment['prompt']}"
    else:
        return None

    cf_completions = experiment.get("completions", [])
    cf_summary_parts = []
    for i, comp in enumerate(cf_completions[:5], 1):
        truncated = comp[:500] + "..." if len(comp) > 500 else comp
        cf_summary_parts.append(f"Counterfactual completion {i}: {truncated}")
    cf_summary = "\n\n".join(cf_summary_parts)

    constraint = ""
    if target_answer == "no":
        constraint = (
            "\n\nFORMAT CONSTRAINT: Ask a yes/no question where the answer is NO "
            "(or indicates minimal/no effect). If this particular change did not "
            "meaningfully alter the model's behavior, say so. Use the causal "
            "analysis to understand what actually matters vs. what doesn't."
        )
    elif target_answer == "yes":
        constraint = (
            "\n\nFORMAT CONSTRAINT: Ask a yes/no question where the answer is YES "
            "(or indicates significant effect)."
        )

    user_content = f"""{conversation}

MODEL'S ACTUAL RESPONSE:
{chosen_completion}

---

COUNTERFACTUAL CHANGE:
{change_desc}

COUNTERFACTUAL COMPLETIONS:
{cf_summary}

CAUSAL ANALYSIS (what actually drives this behavior):
{example.answer}

ORIGINAL BEHAVIOR:
{example.baseline_behavior}

---

Generate a question and answer about how this change affects (or doesn't affect) \
the model's behavior. Use the investigation data as ground truth.{constraint}"""

    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "thinking": {"type": "enabled", "budget_tokens": THINKING_BUDGET},
        "system": DIVERSE_FORMAT_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_content}],
        "tools": [DIVERSE_FORMAT_TOOL],
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_joined_data(
    screening_path: Path,
    investigations_path: Path,
    verification_path: Path,
    min_interest_score: int,
    min_verification_score: int,
) -> list[InputExample]:
    """Load and join screening + investigation + verification data."""
    screening = json.loads(screening_path.read_text())
    investigations = json.loads(investigations_path.read_text())
    verification = json.loads(verification_path.read_text())

    scr_by_id = {r["prompt_id"]: r for r in screening["results"]}
    ver_by_id = {r["prompt_id"]: r for r in verification["results"]}

    examples = []
    for inv_result in investigations["results"]:
        pid = inv_result["prompt_id"]
        findings = inv_result.get("structured_findings")
        if findings is None:
            continue

        scr = scr_by_id.get(pid)
        ver = ver_by_id.get(pid)
        if scr is None or ver is None:
            continue
        if scr["interest_score"] < min_interest_score:
            continue
        if ver["score"] < min_verification_score:
            continue

        # Skip malformed behavior_completion_indices
        raw_bci = scr.get("behavior_completion_indices", [])
        if not isinstance(raw_bci, list) or not raw_bci or not all(isinstance(x, int) for x in raw_bci):
            continue

        # Build messages list
        if scr.get("messages"):
            messages = scr["messages"]
        else:
            messages = [{"role": "user", "content": scr["user_message"]}]

        examples.append(InputExample(
            prompt_id=pid,
            interest_score=scr["interest_score"],
            verification_score=ver["score"],
            messages=messages,
            user_message=scr["user_message"],
            completions=scr["completions"],
            behavior_completion_indices=raw_bci,
            behavior_summary=scr["behavior_summary"],
            second_person_question=scr.get("second_person_question", scr["question"]),
            first_person_answer=findings["first_person_answer"],
            baseline_behavior=findings["baseline_behavior"],
            answer=findings["answer"],
            experiment_log=inv_result.get("experiment_log", []),
        ))

    print(f"Loaded {len(examples)} joined examples "
          f"(interest >= {min_interest_score}, verification >= {min_verification_score})")
    return examples


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------

def _format_conversation(messages: list[dict]) -> str:
    """Format conversation messages for display in prompts."""
    if len(messages) == 1:
        return f"USER PROMPT:\n{messages[0]['content']}"

    parts = ["CONVERSATION HISTORY:\n"]
    for msg in messages[:-1]:
        role = msg["role"].upper()
        parts.append(f"[{role}]: {msg['content']}\n")
    parts.append(f"\nFINAL USER MESSAGE:\n{messages[-1]['content']}")
    return "\n".join(parts)


def build_behavior_prediction_request(example: InputExample, model: str = MODEL) -> dict:
    """Build a Sonnet API request for behavior prediction."""
    conversation = _format_conversation(example.messages)

    user_content = f"""{conversation}

---

BEHAVIOR SUMMARY (ground truth from 10 completions):
{example.behavior_summary}

BASELINE BEHAVIOR (from causal investigation):
{example.baseline_behavior}

FIRST-PERSON ANSWER (from investigation):
{example.first_person_answer}

---

Write a first-person prediction of how you would respond to this prompt. \
Describe what you would likely do, what assumptions you'd make, and what \
your response would look like — as if introspecting before responding."""

    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "thinking": {"type": "enabled", "budget_tokens": THINKING_BUDGET},
        "system": BEHAVIOR_PREDICTION_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_content}],
        "tools": [BEHAVIOR_PREDICTION_TOOL],
    }


def build_counterfactual_prediction_request(
    example: InputExample,
    experiment: dict,
    comp_idx: int,
    model: str = MODEL,
) -> dict:
    """Build a Sonnet API request for counterfactual prediction."""
    conversation = _format_conversation(example.messages)
    chosen_completion = example.completions[comp_idx]

    # Describe the counterfactual change
    # The experiment has either a modified 'prompt' (single-turn) or 'messages' (multi-turn)
    if experiment.get("messages"):
        cf_conversation = _format_conversation(experiment["messages"])
        change_desc = f"The prompt was modified to:\n{cf_conversation}"
    elif experiment.get("prompt"):
        change_desc = f"The user message was changed to:\n{experiment['prompt']}"
    else:
        return None  # skip malformed experiments

    # Summarize counterfactual completions
    cf_completions = experiment.get("completions", [])
    cf_summary_parts = []
    for i, comp in enumerate(cf_completions[:5], 1):  # show up to 5
        truncated = comp[:500] + "..." if len(comp) > 500 else comp
        cf_summary_parts.append(f"Counterfactual completion {i}: {truncated}")
    cf_summary = "\n\n".join(cf_summary_parts)

    user_content = f"""{conversation}

MODEL'S ACTUAL RESPONSE:
{chosen_completion}

---

COUNTERFACTUAL CHANGE:
{change_desc}

COUNTERFACTUAL COMPLETIONS (how the model responded after the change):
{cf_summary}

ORIGINAL BEHAVIOR (before the change):
{example.baseline_behavior}

---

First: write a concise second-person question asking the model how this \
specific change would affect its response (e.g., "How would your response \
change if...?").

Then: write a first-person prediction of how this change would alter the \
model's behavior. Be specific about what would change and why."""

    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "thinking": {"type": "enabled", "budget_tokens": THINKING_BUDGET},
        "system": COUNTERFACTUAL_PREDICTION_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_content}],
        "tools": [COUNTERFACTUAL_PREDICTION_TOOL],
    }


# ---------------------------------------------------------------------------
# Prepare all requests
# ---------------------------------------------------------------------------

def _choose_counterfactual_completion_index(example: InputExample) -> int:
    """Pick a behavior completion for counterfactual examples.

    Uses a different exemplar than investigation training data (which
    always uses indices[0]). If there's more than one behavior completion,
    pick indices[-1] to maximize distance from the first. If there's only
    one, it'll overlap — unavoidable.
    """
    indices = example.behavior_completion_indices
    if len(indices) >= 2:
        return indices[-1] - 1  # 1-indexed to 0-indexed
    elif len(indices) == 1:
        return indices[0] - 1
    else:
        return 0


# Each request tuple: (custom_id, request_params, source_example, experiment_index, comp_idx)
RequestTuple = tuple[str, dict, InputExample, int | None, int]


def prepare_requests(
    examples: list[InputExample],
    n_counterfactual_experiments: int,
    seed: int,
    model: str = MODEL,
) -> list[RequestTuple]:
    """Build all API requests.

    ~50% use the original open-ended format ("I would..."), ~50% use
    diverse formats (yes/no, key factor, short direct, rate, because).

    Returns list of (custom_id, request_params, source_example, experiment_index, comp_idx).
    """
    rng = random.Random(seed)
    requests: list[RequestTuple] = []

    for example in examples:
        # 1. Behavior prediction (one per example)
        roll = rng.random()
        if roll < 0.4:
            # Original open-ended format (40%)
            bp_request = build_behavior_prediction_request(example, model=model)
            custom_id = f"{example.prompt_id}__bp"
        elif roll < 0.5:
            # Forced yes/no — yes (10%)
            bp_request = build_diverse_behavior_request(example, "yes", model=model)
            custom_id = f"{example.prompt_id}__bp_div_yn_yes"
        elif roll < 0.6:
            # Forced yes/no — no (10%)
            bp_request = build_diverse_behavior_request(example, "no", model=model)
            custom_id = f"{example.prompt_id}__bp_div_yn_no"
        else:
            # Free diverse format (40%)
            bp_request = build_diverse_behavior_request(example, None, model=model)
            custom_id = f"{example.prompt_id}__bp_div_free"
        requests.append((custom_id, bp_request, example, None, -1))

        # 2. Counterfactual predictions
        comp_idx = _choose_counterfactual_completion_index(example)
        exp_log = example.experiment_log
        if len(exp_log) <= n_counterfactual_experiments:
            selected_experiments = list(range(len(exp_log)))
        else:
            selected_experiments = sorted(
                rng.sample(range(len(exp_log)), n_counterfactual_experiments)
            )

        for exp_idx in selected_experiments:
            experiment = exp_log[exp_idx]
            roll = rng.random()
            if roll < 0.4:
                # Original open-ended format (40%)
                cf_request = build_counterfactual_prediction_request(
                    example, experiment, comp_idx, model=model,
                )
                custom_id = f"{example.prompt_id}__cf_{exp_idx}"
            elif roll < 0.5:
                # Forced yes/no — yes (10%)
                cf_request = build_diverse_counterfactual_request(
                    example, experiment, comp_idx, "yes", model=model,
                )
                custom_id = f"{example.prompt_id}__cf_div_{exp_idx}_yn_yes"
            elif roll < 0.6:
                # Forced yes/no — no (10%)
                cf_request = build_diverse_counterfactual_request(
                    example, experiment, comp_idx, "no", model=model,
                )
                custom_id = f"{example.prompt_id}__cf_div_{exp_idx}_yn_no"
            else:
                # Free diverse format (40%)
                cf_request = build_diverse_counterfactual_request(
                    example, experiment, comp_idx, None, model=model,
                )
                custom_id = f"{example.prompt_id}__cf_div_{exp_idx}_free"
            if cf_request is None:
                continue
            requests.append((custom_id, cf_request, example, exp_idx, comp_idx))

    n_diverse = sum(1 for r in requests if '_div_' in r[0])
    print(f"Prepared {len(requests)} total requests "
          f"({sum(1 for r in requests if '__bp' in r[0])} behavior predictions, "
          f"{sum(1 for r in requests if '__cf' in r[0])} counterfactual predictions, "
          f"{n_diverse} diverse format)")
    return requests


# ---------------------------------------------------------------------------
# Parse result into SyntheticExample
# ---------------------------------------------------------------------------

def parse_result(
    tool_input: dict,
    custom_id: str,
    example: InputExample,
    experiment_index: int | None,
    comp_idx: int,
    request_params: dict,
) -> SyntheticExample:
    """Convert a tool_use response into a SyntheticExample."""
    is_bp = "__bp" in custom_id
    is_diverse = "_div_" in custom_id
    system_prompt = request_params.get("system", "")
    user_prompt = request_params["messages"][0]["content"]

    if is_bp:
        bp_rng = random.Random(f"{custom_id}_bp_question")
        separator = bp_rng.choice(BEHAVIOR_PREDICTION_SEPARATORS)
        if is_diverse:
            question = separator + tool_input["question"]
        else:
            question = separator + bp_rng.choice(BEHAVIOR_PREDICTION_QUESTIONS)
        return SyntheticExample(
            prompt_id=example.prompt_id,
            example_type="behavior_prediction",
            messages=example.messages,
            chosen_completion="",
            chosen_completion_index=-1,
            question=question,
            answer=tool_input["prediction"],
            interest_score=example.interest_score,
            verification_score=example.verification_score,
            experiment_index=None,
            behavior_completion_indices=example.behavior_completion_indices,
            sonnet_system_prompt=system_prompt,
            sonnet_user_prompt=user_prompt,
        )
    else:
        return SyntheticExample(
            prompt_id=example.prompt_id,
            example_type="counterfactual_prediction",
            messages=example.messages,
            chosen_completion=example.completions[comp_idx],
            chosen_completion_index=comp_idx,
            question=tool_input["question"],
            answer=tool_input["prediction"],
            interest_score=example.interest_score,
            verification_score=example.verification_score,
            experiment_index=experiment_index,
            behavior_completion_indices=example.behavior_completion_indices,
            sonnet_system_prompt=system_prompt,
            sonnet_user_prompt=user_prompt,
        )


# ---------------------------------------------------------------------------
# Async mode
# ---------------------------------------------------------------------------

async def generate_async(
    requests: list[RequestTuple],
    output_path: Path,
    concurrency: int,
) -> list[SyntheticExample]:
    """Generate synthetic data using the async API."""
    print(f"\nGenerating {len(requests)} synthetic examples "
          f"(concurrency={concurrency})...")

    client = anthropic.AsyncAnthropic()
    results: list[SyntheticExample] = []
    jsonl_path = output_path.with_suffix(".jsonl")

    # Resume from existing results
    completed_ids = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    completed_ids.add(record["_custom_id"])
                    del record["_custom_id"]
                    results.append(SyntheticExample(**record))
        print(f"  Resuming: {len(completed_ids)} already completed")

    remaining = [(cid, req, ex, ei, ci) for cid, req, ex, ei, ci in requests
                 if cid not in completed_ids]
    if not remaining:
        print("  All requests already completed")
        return results

    jsonl_file = open(jsonl_path, "a")
    write_lock = asyncio.Lock()
    skipped = {"count": 0}

    async def generate_one(i: int, item: tuple):
        custom_id, request_params, example, experiment_index, comp_idx = item
        try:
            resp = await async_api_call(client, **request_params)
            tool_input = extract_tool_input(resp)
            result = parse_result(tool_input, custom_id, example, experiment_index, comp_idx, request_params)
            results.append(result)

            async with write_lock:
                record = asdict(result)
                record["_custom_id"] = custom_id
                jsonl_file.write(json.dumps(record) + "\n")
                jsonl_file.flush()
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            skipped["count"] += 1
            print(f"  SKIPPING {custom_id}: malformed result ({e})")

    await run_concurrent(
        generate_one,
        remaining,
        concurrency=concurrency,
        label="Synthetic",
        progress_interval=10,
    )
    jsonl_file.close()

    if remaining:
        skip_rate = skipped["count"] / len(remaining)
        assert skip_rate < 0.001, (
            f"Synthetic data skip rate {skip_rate:.1%} "
            f"({skipped['count']}/{len(remaining)}) exceeds 0.1% threshold"
        )

    return results


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def generate_batch(
    requests: list[RequestTuple],
    output_path: Path,
    batch_chunk_size: int,
) -> list[SyntheticExample]:
    """Generate synthetic data using the batch API."""
    print(f"\nGenerating {len(requests)} synthetic examples (batch API)...")

    batch_state_path = output_path.with_suffix(".batch_state.json")
    jsonl_path = output_path.with_suffix(".jsonl")

    # Build batch requests
    batch_requests = []
    examples_by_id: dict[str, tuple[InputExample, int | None, int, dict]] = {}
    for custom_id, request_params, example, experiment_index, comp_idx in requests:
        batch_requests.append({
            "custom_id": custom_id,
            "params": request_params,
        })
        examples_by_id[custom_id] = (example, experiment_index, comp_idx, request_params)

    batch_api_key = os.environ.get("ANTHROPIC_API_KEY_BATCH_API")
    assert batch_api_key, "ANTHROPIC_API_KEY_BATCH_API not set in environment"
    client = anthropic.Anthropic(api_key=batch_api_key)

    # Split into chunks
    chunks = [batch_requests[i:i + batch_chunk_size]
              for i in range(0, len(batch_requests), batch_chunk_size)]
    print(f"  Split into {len(chunks)} batch chunks of up to {batch_chunk_size}")

    # Load or create batch state
    batch_ids = []
    if batch_state_path.exists():
        saved = json.loads(batch_state_path.read_text())
        batch_ids = saved.get("batch_ids", [])
        if batch_ids:
            print(f"  Resuming {len(batch_ids)} batches")

    # Submit chunks
    while len(batch_ids) < len(chunks):
        chunk_idx = len(batch_ids)
        chunk = chunks[chunk_idx]
        batch = client.messages.batches.create(requests=chunk)
        batch_ids.append(batch.id)
        batch_state_path.write_text(json.dumps({
            "batch_ids": batch_ids,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "n_chunks": len(chunks),
            "chunk_size": batch_chunk_size,
            "total_requests": len(batch_requests),
        }, indent=2))
        print(f"  Submitted chunk {chunk_idx + 1}/{len(chunks)}: "
              f"{batch.id} ({len(chunk)} requests)")

    # Poll until done
    while True:
        all_done = True
        for i, bid in enumerate(batch_ids):
            status = client.messages.batches.retrieve(bid)
            counts = status.request_counts
            if status.processing_status != "ended":
                all_done = False
            print(
                f"  [{i + 1}/{len(batch_ids)}] {bid}: {status.processing_status} "
                f"(ok={counts.succeeded} err={counts.errored} "
                f"exp={counts.expired} pending={counts.processing})"
            )
        if all_done:
            break
        time.sleep(60)

    # Retrieve results
    results = []
    jsonl_file = open(jsonl_path, "w")
    n_skipped = 0

    for bid in batch_ids:
        for batch_result in client.messages.batches.results(bid):
            custom_id = batch_result.custom_id
            example, experiment_index, comp_idx, request_params = examples_by_id[custom_id]

            if batch_result.result.type != "succeeded":
                n_skipped += 1
                print(f"  SKIPPING {custom_id}: batch result {batch_result.result.type}")
                continue

            try:
                tool_input = None
                for block in batch_result.result.message.content:
                    if block.type == "tool_use":
                        tool_input = block.input
                        break

                if tool_input is None:
                    print(f"  SKIPPING {custom_id}: no tool_use in response")
                    continue

                result = parse_result(tool_input, custom_id, example, experiment_index, comp_idx, request_params)
                results.append(result)
                record = asdict(result)
                record["_custom_id"] = custom_id
                jsonl_file.write(json.dumps(record) + "\n")
            except (KeyError, AttributeError, TypeError, ValueError) as e:
                n_skipped += 1
                print(f"  SKIPPING {custom_id}: malformed result ({e})")
                continue

    jsonl_file.close()

    if requests:
        skip_rate = n_skipped / len(requests)
        assert skip_rate < 0.001, (
            f"Synthetic data batch skip rate {skip_rate:.1%} "
            f"({n_skipped}/{len(requests)}) exceeds 0.1% threshold"
        )

    return results


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(
    results: list[SyntheticExample],
    output_path: Path,
    args: argparse.Namespace,
):
    """Save final results as JSON."""
    bp_count = sum(1 for r in results if r.example_type == "behavior_prediction")
    cf_count = sum(1 for r in results if r.example_type == "counterfactual_prediction")

    output = {
        "metadata": {
            "generation_model": args.model,
            "investigations_file": str(args.investigations),
            "screening_file": str(args.screening),
            "verification_file": str(args.verification),
            "n_examples": len(results),
            "n_behavior_predictions": bp_count,
            "n_counterfactual_predictions": cf_count,
            "min_interest_score": args.min_interest_score,
            "min_verification_score": args.min_verification_score,
            "n_counterfactual_experiments": args.n_counterfactual_experiments,
            "seed": args.seed,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": [asdict(r) for r in results],
    }

    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved {len(results)} synthetic examples to {output_path}")
    print(f"  Behavior predictions: {bp_count}")
    print(f"  Counterfactual predictions: {cf_count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--investigations", type=Path, required=True,
                        help="Path to investigations.json")
    parser.add_argument("--screening", type=Path, required=True,
                        help="Path to screening.json")
    parser.add_argument("--verification", type=Path, required=True,
                        help="Path to verification.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for synthetic_data.json")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N input examples (for testing)")
    parser.add_argument("--batch", action="store_true",
                        help="Use batch API instead of async")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"Async concurrency (default: {DEFAULT_CONCURRENCY})")
    parser.add_argument("--batch-chunk-size", type=int, default=DEFAULT_BATCH_CHUNK_SIZE,
                        help=f"Batch chunk size (default: {DEFAULT_BATCH_CHUNK_SIZE})")
    parser.add_argument("--n-counterfactual-experiments", type=int,
                        default=DEFAULT_COUNTERFACTUAL_EXPERIMENTS,
                        help=f"Counterfactual experiments per example (default: {DEFAULT_COUNTERFACTUAL_EXPERIMENTS})")
    parser.add_argument("--min-interest-score", type=int, default=MIN_INTEREST_SCORE,
                        help=f"Minimum interest score (default: {MIN_INTEREST_SCORE})")
    parser.add_argument("--min-verification-score", type=int, default=MIN_VERIFICATION_SCORE,
                        help=f"Minimum verification score (default: {MIN_VERIFICATION_SCORE})")
    parser.add_argument("--model", type=str, default=MODEL,
                        help=f"Anthropic model to use (default: {MODEL})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for experiment selection (default: 42)")
    args = parser.parse_args()

    examples = load_joined_data(
        args.screening,
        args.investigations,
        args.verification,
        min_interest_score=args.min_interest_score,
        min_verification_score=args.min_verification_score,
    )

    if args.n is not None:
        examples = examples[:args.n]
        print(f"Limited to first {args.n} examples")

    requests = prepare_requests(
        examples,
        n_counterfactual_experiments=args.n_counterfactual_experiments,
        seed=args.seed,
        model=args.model,
    )

    if args.batch:
        results = generate_batch(requests, args.output, args.batch_chunk_size)
    else:
        results = asyncio.run(generate_async(requests, args.output, args.concurrency))

    save_results(results, args.output, args)


if __name__ == "__main__":
    main()
