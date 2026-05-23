"""
AO Agent Eval: Multi-turn Opus agent that queries an Activation Oracle to explain model behavior.

Instead of running counterfactual experiments on vLLM (like investigate.py), the agent
probes the AO about activations from a single frozen trace. The agent gets 10 temperature-1.0
AO completions per query and uses consistency to assess confidence.

Architecture:
  1. Load target model + AO adapter, load eval entries
  2. For each entry: forward pass on target model -> cache activations
  3. Opus agent sees: transcript + question, has query_ao and submit_findings tools
  4. Agent iterates (up to max_turns), asking different questions with different token windows
  5. Agent submits final answer via submit_findings
  6. Haiku judge scores against reference answer

Usage:
    source .env
    python -m nl_probes.open_ended_eval.model_understanding_ao_agent \
        --verbalizer-lora-path checkpoints/500k_pl_31k_spqav2_199k_sqav3_126k_cls/final \
        --run-dir data_pipelines/model_understanding/runs/qwen3_14b_run_test \
        --model-name Qwen/Qwen3-8B \
        --output-dir experiments/model_understanding_eval/qwen3_14b_run_test_v2_judge/ao_agent \
        --max-entries 5
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
from typing import Any

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import anthropic
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import nl_probes.base_experiment as base_experiment
from nl_probes.base_experiment import VerbalizerEvalConfig, VerbalizerInputInfo
from nl_probes.configs.sft_config import read_training_config
from nl_probes.open_ended_eval.eval_runner import (
    build_verbalizer_eval_config,
    default_layer_combination,
    ensure_default_adapter,
)
from nl_probes.base_experiment import tokenize_chat_messages
from nl_probes.open_ended_eval.model_understanding import (
    AO_ANSWER_MODE,
    AO_QUESTION_MODE,
    ModelUnderstandingEvalEntry,
    compute_judge_metrics,
    judge_response_texts,
    load_model_understanding_eval_entries,
)
from nl_probes.utils.common import load_model, load_tokenizer

print = functools.partial(print, flush=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_AO_CONTEXT_TOKENS = 1000
AO_COMPLETIONS_PER_QUERY = 10
AO_TEMPERATURE = 1.0
AO_MAX_NEW_TOKENS = 256
DEFAULT_MAX_AGENT_TURNS = 10
DEFAULT_AGENT_CONCURRENCY = 20
DEFAULT_AGENT_MODEL = "claude-opus-4-6"
DEFAULT_JUDGE_CONCURRENCY = 30
AO_EVAL_BATCH_SIZE = 8

AGENT_SYSTEM_PROMPT = """\
You are investigating why a language model (the "target model") behaved a \
certain way. You have access to an Activation Oracle (AO) — a separate model \
that has been trained to read the target model's internal activations \
(residual stream vectors) and answer questions about what the target model \
is "thinking."

You will be given:
- The original conversation (user message(s) and the target model's response)
- A specific question about the behavior to investigate

Your only tool is `query_ao`, which sends a question to the AO along with a \
selection of the target model's internal activations. The AO returns {n_completions} \
independent responses at temperature 1.0.

## How to use the AO effectively

1. **Assess AO confidence via consistency.** The AO has NOT been trained to \
express uncertainty. However, we find that the consistency of its {n_completions} \
temperature-1.0 responses is highly correlated with how confident it actually \
is. If all {n_completions} responses say the same thing, the AO is likely reading real \
signal. If they're all over the place, the AO is likely uncertain and making \
things up. Weight consistent responses heavily and discount inconsistent ones.

2. **Ask open-ended questions.** Prefer broad, exploratory questions that \
let the AO tell you what it sees, rather than narrow yes/no questions. \
For example:
   - "How is the model interpreting [specific part]?"
   - "What is driving the model's response here?"
   - "What features is the model attending to in this section?"
   - "Explain what the model is thinking when it processes [X]."
Narrow hypothesis-confirming questions can be useful as follow-ups, but \
start broad to avoid anchoring the AO on your assumptions.

3. **Use position selection strategically.** You can focus the AO on different \
parts of the conversation by selecting specific token windows:
   - Named segments: "full", "user_turn_1", "assistant_turn_1", "question", etc.
   - Text snippets: quote a unique substring from the conversation to focus \
     on those tokens specifically.
   - Default is "full" (capped at {max_tokens} tokens; if the context exceeds \
     this, only the last {max_tokens} tokens are used).

4. **Form hypotheses, then test them.** Read the transcript carefully and \
form hypotheses about what is driving the behavior. Then use the AO to \
confirm or disprove your hypotheses. Your final answer should integrate \
both your understanding of the transcript and the AO evidence.

5. **Be efficient.** 3-5 well-chosen queries is usually enough. Once the \
AO responses are consistent on a point, move on — don't keep asking \
about the same thing.

## Output format

After your investigation, call `submit_findings` with your structured answer. \
This is required.
"""


SUBMIT_FINDINGS_TOOL = {
    "name": "submit_findings",
    "description": (
        "Submit your final structured findings. Call this ONCE at the very end "
        "of your investigation. This is required."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question investigated (restate it concisely)",
            },
            "answer": {
                "type": "string",
                "description": (
                    "Your answer in 2-4 sentences, third person ('The model does X "
                    "because...'). ONLY include claims supported by consistent AO "
                    "responses. Use percentages for AO agreement rates when relevant."
                ),
            },
            "commentary": {
                "type": "string",
                "description": (
                    "Optional speculation beyond what the AO evidence supports. "
                    "Hypotheses, connections to known model behaviors, etc."
                ),
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Confidence in the answer based on AO consistency",
            },
            "key_queries": {
                "type": "array",
                "description": "2-5 most informative AO queries and what they revealed",
                "items": {
                    "type": "object",
                    "properties": {
                        "query_number": {
                            "type": "integer",
                            "description": "The query number (from 'Query #N' in results)",
                        },
                        "question_asked": {
                            "type": "string",
                            "description": "The question sent to the AO",
                        },
                        "consistency": {
                            "type": "string",
                            "description": "How consistent the responses were (e.g., '8/10 agree')",
                        },
                        "finding": {
                            "type": "string",
                            "description": "What this query revealed",
                        },
                    },
                    "required": ["query_number", "question_asked", "consistency", "finding"],
                },
            },
        },
        "required": ["question", "answer", "commentary", "confidence", "key_queries"],
    },
}


def _build_query_ao_tool(segment_descriptions: list[str]) -> dict:
    """Build the query_ao tool definition with available segments listed."""
    segments_text = "\n".join(f"  - {desc}" for desc in segment_descriptions)
    return {
        "name": "query_ao",
        "description": (
            "Query the Activation Oracle about the target model's internal activations. "
            f"Returns {AO_COMPLETIONS_PER_QUERY} independent responses at temperature 1.0. "
            "Use response consistency to assess AO confidence.\n\n"
            "Available named segments:\n"
            f"{segments_text}\n\n"
            "You can also pass a text snippet (a unique substring from the conversation) "
            "to focus on specific tokens. If the snippet is not found or matches multiple "
            "locations, you'll get an error — include more surrounding text to disambiguate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The question to ask the AO about the target model's activations. "
                        "Third person: 'What is the model thinking about...', "
                        "'What features is the model attending to...', etc."
                    ),
                },
                "positions": {
                    "type": "string",
                    "description": (
                        "Which tokens to feed to the AO. Either a named segment "
                        "(e.g., 'full', 'user_turn_1', 'assistant_turn_1') or a text "
                        "snippet to focus on. Default: 'full'."
                    ),
                },
            },
            "required": ["question"],
        },
    }


# ---------------------------------------------------------------------------
# Token position mapping
# ---------------------------------------------------------------------------


@dataclass
class TurnSegment:
    """A named segment in the tokenized context."""
    name: str
    description: str
    start_pos: int
    end_pos: int


@dataclass
class TokenPositionMap:
    """Maps between text content and token positions."""
    context_token_ids: list[int]
    segments: list[TurnSegment]
    decoded_text: str  # Full decoded text for substring matching

    def resolve_positions(
        self,
        selection: str | None,
        tokenizer: AutoTokenizer,
        max_tokens: int,
    ) -> tuple[list[int], str]:
        """Resolve a position selection string to token positions.

        Returns (positions, description) or raises ValueError on error.
        """
        if selection is None or selection.strip() == "":
            selection = "full"
        selection = selection.strip()

        # Try named segment first
        for seg in self.segments:
            if selection.lower() == seg.name.lower():
                positions = list(range(seg.start_pos, seg.end_pos))
                if len(positions) > max_tokens:
                    # Take the last max_tokens from this segment
                    positions = positions[-max_tokens:]
                    return positions, f"{seg.name} (last {max_tokens} of {seg.end_pos - seg.start_pos} tokens)"
                return positions, seg.name

        # Try text snippet matching
        idx = self.decoded_text.find(selection)
        if idx == -1:
            available = ", ".join(f'"{seg.name}"' for seg in self.segments)
            raise ValueError(
                f"Text snippet not found in context. "
                f"Available named segments: {available}. "
                f"Or pass a unique substring from the conversation."
            )

        # Check for multiple matches
        second_idx = self.decoded_text.find(selection, idx + 1)
        if second_idx != -1:
            raise ValueError(
                f"Text snippet matches multiple locations in context "
                f"(at char {idx} and {second_idx}). "
                f"Include more surrounding text to disambiguate."
            )

        # Map character range to token positions
        # Decode each token to find which tokens overlap with the matched range
        char_end = idx + len(selection)
        start_pos = None
        end_pos = None
        current_char = 0
        for token_pos, token_id in enumerate(self.context_token_ids):
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            token_start = current_char
            token_end = current_char + len(token_text)
            if token_end > idx and start_pos is None:
                start_pos = token_pos
            if token_start >= char_end and end_pos is None:
                end_pos = token_pos
                break
            current_char = token_end

        if start_pos is None:
            start_pos = 0
        if end_pos is None:
            end_pos = len(self.context_token_ids)

        positions = list(range(start_pos, end_pos))
        if len(positions) > max_tokens:
            positions = positions[-max_tokens:]

        snippet_preview = selection[:60] + "..." if len(selection) > 60 else selection
        return positions, f"text match: '{snippet_preview}' ({len(positions)} tokens)"


def build_token_position_map(
    entry: ModelUnderstandingEvalEntry,
    tokenizer: AutoTokenizer,
) -> TokenPositionMap:
    """Build a token position map for an eval entry.

    The context is just the original conversation (user messages + assistant
    completion). The behavioral question is NOT included — the agent asks
    its own questions to the AO via the query_ao tool.

    Tokenizes each message separately to find exact turn boundaries.
    """
    # Build context WITHOUT the question — just original conversation
    context_messages = [
        {"role": msg.role, "content": msg.content} for msg in entry.messages
    ]
    context_messages.append({"role": "assistant", "content": entry.chosen_completion})

    # Use continue_final_message=True to strip end-of-turn tokens (<|im_end|>)
    # from the assistant completion. Without this, the AO sees the end-of-turn
    # token and confabulates "the model was cut off by a token limit" narratives,
    # since completions are hard-capped at 500 tokens and often end mid-sentence.
    full_token_ids = tokenize_chat_messages(
        tokenizer, context_messages,
        add_generation_prompt=False, enable_thinking=False,
        continue_final_message=True,
    )

    # To find message boundaries, tokenize incrementally
    segments: list[TurnSegment] = []
    prev_len = 0

    for msg_idx, msg in enumerate(context_messages):
        # Tokenize up to and including this message
        partial_messages = context_messages[:msg_idx + 1]
        is_last = (msg_idx == len(context_messages) - 1)

        partial_token_ids = tokenize_chat_messages(
            tokenizer, partial_messages,
            add_generation_prompt=False, enable_thinking=False,
            continue_final_message=is_last,
        )
        current_len = len(partial_token_ids)

        role = msg["role"]
        # Determine the turn number for this role
        turn_number = sum(1 for m in context_messages[:msg_idx + 1] if m["role"] == role)

        name = f"{role}_turn_{turn_number}"
        content_preview = msg["content"][:80].replace("\n", " ")
        desc_parts = [f'"{name}" [{prev_len}-{current_len}]: {content_preview}...']

        segments.append(TurnSegment(
            name=name,
            description=desc_parts[0],
            start_pos=prev_len,
            end_pos=current_len,
        ))
        prev_len = current_len

    # Add "full" segment
    segments.insert(0, TurnSegment(
        name="full",
        description=f'"full" [0-{len(full_token_ids)}]: entire context',
        start_pos=0,
        end_pos=len(full_token_ids),
    ))

    decoded_text = tokenizer.decode(full_token_ids, skip_special_tokens=False)

    return TokenPositionMap(
        context_token_ids=full_token_ids,
        segments=segments,
        decoded_text=decoded_text,
    )


# ---------------------------------------------------------------------------
# AO query execution
# ---------------------------------------------------------------------------


@dataclass
class AOQueryRequest:
    """A request to query the AO, placed in the async queue."""
    entry_idx: int
    query_id: int
    context_token_ids: list[int]
    positions: list[int]
    question: str
    future: asyncio.Future


@dataclass
class AOQueryResult:
    """Result from a single AO query."""
    responses: list[str]
    positions_desc: str
    n_positions: int


class AOBatchWorker:
    """Background worker that batches AO queries for GPU efficiency.

    Agents put AOQueryRequest into the queue. The worker drains requests
    in batches and runs them through the AO.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: VerbalizerEvalConfig,
        verbalizer_lora_name: str,
        device: torch.device,
        ao_completions: int,
        ao_max_new_tokens: int,
        batch_size: int = 4,
        drain_interval: float = 0.5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.verbalizer_lora_name = verbalizer_lora_name
        self.device = device
        self.ao_completions = ao_completions
        self.ao_max_new_tokens = ao_max_new_tokens
        self.batch_size = batch_size
        self.drain_interval = drain_interval
        self.queue: asyncio.Queue[AOQueryRequest] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._total_queries = 0
        self._total_time = 0.0

    def start(self):
        self._task = asyncio.create_task(self._worker_loop())

    def stop(self):
        if self._task:
            self._task.cancel()

    async def _worker_loop(self):
        """Drain queue in batches."""
        loop = asyncio.get_event_loop()
        while True:
            # Wait for at least one request
            first = await self.queue.get()
            batch = [first]

            # Drain up to batch_size more with a short wait
            deadline = time.monotonic() + self.drain_interval
            while len(batch) < self.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            # Process the batch on GPU (run in executor to avoid blocking event loop)
            try:
                results = await loop.run_in_executor(None, self._process_batch, batch)
                for req, result in zip(batch, results):
                    req.future.set_result(result)
            except Exception as e:
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(e)

    def _process_batch(self, batch: list[AOQueryRequest]) -> list[list[str]]:
        """Run a batch of AO queries on GPU. Returns list of response lists."""
        t0 = time.monotonic()

        # Each request gets ao_completions copies in the batch
        all_prompt_infos: list[VerbalizerInputInfo] = []
        for req in batch:
            for _ in range(self.ao_completions):
                all_prompt_infos.append(VerbalizerInputInfo(
                    context_token_ids=req.context_token_ids,
                    positions=req.positions,
                    verbalizer_prompt=req.question,
                    ground_truth="",
                ))

        generation_kwargs = {
            "do_sample": True,
            "temperature": AO_TEMPERATURE,
            "max_new_tokens": self.ao_max_new_tokens,
        }
        config_override = VerbalizerEvalConfig(
            model_name=self.config.model_name,
            selected_act_layers=list(self.config.selected_act_layers),
            injection_layer=self.config.injection_layer,
            layer_combinations=list(self.config.layer_combinations),
            selected_layer_combination=list(self.config.selected_layer_combination),
            activation_input_types=list(self.config.activation_input_types),
            verbalizer_generation_kwargs=generation_kwargs,
            steering_coefficient=self.config.steering_coefficient,
            eval_batch_size=len(all_prompt_infos),
        )

        results = base_experiment.run_verbalizer(
            model=self.model,
            tokenizer=self.tokenizer,
            verbalizer_prompt_infos=all_prompt_infos,
            verbalizer_lora_path=self.verbalizer_lora_name,
            target_lora_path=None,
            config=config_override,
            device=self.device,
        )

        # Group responses back by request
        grouped: list[list[str]] = []
        idx = 0
        for req in batch:
            request_responses = []
            for _ in range(self.ao_completions):
                request_responses.extend(results[idx].responses)
                idx += 1
            grouped.append(request_responses)

        elapsed = time.monotonic() - t0
        self._total_queries += len(batch)
        self._total_time += elapsed
        print(f"    [AO batch] {len(batch)} queries, {len(all_prompt_infos)} completions, "
              f"{elapsed:.1f}s ({self._total_queries} total)")

        return grouped


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


@dataclass
class AgentTranscript:
    """Full transcript of an agent investigation."""
    prompt_id: str
    turns: list[dict[str, Any]]
    final_answer: str | None
    structured_findings: dict | None
    n_agent_turns: int
    n_ao_queries: int


def _format_transcript_for_agent(entry: ModelUnderstandingEvalEntry) -> str:
    """Format the conversation transcript for the Opus agent to read."""
    parts = []
    for msg in entry.messages:
        role_name = {"user": "User", "assistant": "Assistant", "system": "System"}.get(
            msg.role, msg.role.title()
        )
        parts.append(f"**[{role_name}]:**\n{msg.content}")

    parts.append(f"**[Assistant response being investigated]:**\n{entry.chosen_completion}")
    return "\n\n".join(parts)


def _build_initial_agent_prompt(
    entry: ModelUnderstandingEvalEntry,
    position_map: TokenPositionMap,
) -> str:
    """Build the initial prompt for the Opus investigation agent."""
    transcript = _format_transcript_for_agent(entry)
    question = entry.third_person_question

    # Build segment descriptions for the agent
    segment_list = "\n".join(
        f"  - {seg.description}" for seg in position_map.segments
    )

    return (
        f"## Transcript\n\n"
        f"{transcript}\n\n"
        f"## Question to Investigate\n\n"
        f"{question}\n\n"
        f"## Available Token Segments\n\n"
        f"These are the segments you can pass to the `query_ao` tool's `positions` parameter:\n"
        f"{segment_list}\n\n"
        f"Total context: {len(position_map.context_token_ids)} tokens.\n\n"
        f"---\n\n"
        f"Investigate the question above by querying the Activation Oracle."
    )


class TokenRateTracker:
    """Tracks Opus API token usage rates."""

    def __init__(self, interval: int = 60):
        self._interval = interval
        self._input_tokens = 0
        self._cache_read_tokens = 0
        self._output_tokens = 0
        self._last_print = time.monotonic()
        self._lock: asyncio.Lock | None = None
        self._task: asyncio.Task | None = None

    def start(self):
        self._lock = asyncio.Lock()
        self._last_print = time.monotonic()
        self._task = asyncio.create_task(self._print_loop())

    def stop(self):
        if self._task:
            self._task.cancel()

    async def record(self, input_tokens: int, cache_read_tokens: int, output_tokens: int):
        assert self._lock is not None
        async with self._lock:
            self._input_tokens += input_tokens
            self._cache_read_tokens += cache_read_tokens
            self._output_tokens += output_tokens

    async def _print_loop(self):
        while True:
            await asyncio.sleep(self._interval)
            assert self._lock is not None
            async with self._lock:
                elapsed = time.monotonic() - self._last_print
                if elapsed < 1:
                    continue
                in_rate = self._input_tokens * 60 / elapsed
                cache_rate = self._cache_read_tokens * 60 / elapsed
                out_rate = self._output_tokens * 60 / elapsed
                print(f"\n  [rate] input={in_rate:,.0f}/min "
                      f"cache_read={cache_rate:,.0f}/min "
                      f"output={out_rate:,.0f}/min")
                self._input_tokens = 0
                self._cache_read_tokens = 0
                self._output_tokens = 0
                self._last_print = time.monotonic()


token_rate_tracker = TokenRateTracker()


def serialize_content_block(block) -> dict:
    """Serialize an Anthropic content block to a JSON-safe dict."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    elif block.type == "thinking":
        return {"type": "thinking", "thinking": block.thinking}
    elif block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name,
                "input": block.input}
    return {"type": block.type, "raw": str(block)}


def serialize_messages(messages: list) -> list[dict]:
    """Serialize the full message history to JSON-safe dicts."""
    serialized = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            serialized.append({"role": role, "content": content})
        elif isinstance(content, list):
            blocks = []
            for item in content:
                if isinstance(item, dict):
                    blocks.append(item)
                else:
                    blocks.append(serialize_content_block(item))
            serialized.append({"role": role, "content": blocks})
        else:
            serialized.append({"role": role, "content": str(content)})
    return serialized


async def investigate_one(
    entry: ModelUnderstandingEvalEntry,
    position_map: TokenPositionMap,
    ao_worker: AOBatchWorker,
    anthropic_client: anthropic.AsyncAnthropic,
    tokenizer: AutoTokenizer,
    *,
    agent_model: str,
    max_agent_turns: int,
    max_ao_context_tokens: int,
    ao_completions: int,
    output_dir: Path,
) -> AgentTranscript:
    """Run a full AO agent investigation on one entry."""
    pid = entry.prompt_id
    print(f"\n{'='*60}")
    print(f"Investigating {pid} (interest={entry.interest_score}, verif={entry.verification_score})")
    print(f"Q: {entry.third_person_question[:120]}")
    print(f"{'='*60}")

    stats = {
        "n_ao_queries": 0,
        "input_tokens": 0,
        "cache_read_tokens": 0,
        "cache_create_tokens": 0,
        "output_tokens": 0,
    }
    transcript_turns: list[dict[str, Any]] = []

    # Build tools with segment descriptions
    segment_descriptions = [seg.description for seg in position_map.segments]
    tools = [_build_query_ao_tool(segment_descriptions), SUBMIT_FINDINGS_TOOL]

    system_prompt = AGENT_SYSTEM_PROMPT.format(
        n_completions=ao_completions,
        max_tokens=max_ao_context_tokens,
    )

    messages = [
        {"role": "user", "content": _build_initial_agent_prompt(entry, position_map)},
    ]

    final_text_parts = []
    structured_findings = None

    for turn_num in range(1, max_agent_turns + 1):
        print(f"  Turn {turn_num}/{max_agent_turns}...")

        # Add cache breakpoint to the last user message
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                for block in msg["content"]:
                    if isinstance(block, dict):
                        block.pop("cache_control", None)
        last_msg = messages[-1]
        if last_msg["role"] == "user" and isinstance(last_msg["content"], list) and last_msg["content"]:
            last_block = last_msg["content"][-1]
            if isinstance(last_block, dict):
                last_block["cache_control"] = {"type": "ephemeral"}

        # Retry with exponential backoff on rate limit / overloaded errors
        max_retries = 8
        for attempt in range(max_retries):
            try:
                response = await anthropic_client.messages.create(
                    model=agent_model,
                    max_tokens=4096,
                    thinking={"type": "enabled", "budget_tokens": 2048},
                    system=[{
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }],
                    messages=messages,
                    tools=tools,
                )
                break
            except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIStatusError) as e:
                wait = min(2 ** attempt * 10, 600)  # 10, 20, 40, 80, 160, 320, 600, 600
                print(f"    [API error] attempt {attempt+1}/{max_retries}, waiting {wait}s: {type(e).__name__}: {e}")
                await asyncio.sleep(wait)
        else:
            raise RuntimeError(f"API errors {max_retries} times for {pid}, giving up")

        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
        total_in = usage.input_tokens + cache_read + cache_create
        cache_pct = 100 * cache_read / total_in if total_in > 0 else 0
        stats["input_tokens"] += usage.input_tokens
        stats["cache_read_tokens"] += cache_read
        stats["cache_create_tokens"] += cache_create
        stats["output_tokens"] += usage.output_tokens

        await token_rate_tracker.record(
            usage.input_tokens + cache_create, cache_read, usage.output_tokens,
        )

        print(f"    [usage] in={usage.input_tokens} cache_read={cache_read} "
              f"cache_create={cache_create} out={usage.output_tokens} "
              f"({cache_pct:.0f}% cache hit)")

        assistant_content = response.content
        has_tool_use = False
        tool_results = []

        for block in assistant_content:
            if block.type == "text":
                print(f"    [text] {block.text[:200]}{'...' if len(block.text) > 200 else ''}")
                final_text_parts.append(block.text)
            elif block.type == "thinking":
                print(f"    [thinking] {block.thinking[:150]}...")
            elif block.type == "tool_use":
                has_tool_use = True

                if block.name == "submit_findings":
                    structured_findings = block.input
                    print(f"    [submit_findings] confidence={block.input.get('confidence')}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Findings recorded. Investigation complete.",
                    })

                elif block.name == "query_ao":
                    stats["n_ao_queries"] += 1
                    query_num = stats["n_ao_queries"]
                    question = block.input["question"]
                    position_selection = block.input.get("positions")

                    print(f"    [query_ao #{query_num}] q='{question[:80]}...' "
                          f"pos={position_selection or 'full'}")

                    # Resolve positions
                    try:
                        positions, pos_desc = position_map.resolve_positions(
                            position_selection, tokenizer, max_ao_context_tokens,
                        )
                    except ValueError as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Error: {e}",
                            "is_error": True,
                        })
                        transcript_turns.append({
                            "role": "agent",
                            "query_number": query_num,
                            "question": question,
                            "positions": position_selection,
                            "error": str(e),
                        })
                        continue

                    # Submit to AO batch worker
                    loop = asyncio.get_event_loop()
                    future = loop.create_future()
                    await ao_worker.queue.put(AOQueryRequest(
                        entry_idx=0,
                        query_id=query_num,
                        context_token_ids=position_map.context_token_ids,
                        positions=positions,
                        question=question,
                        future=future,
                    ))

                    ao_responses = await future
                    print(f"    [AO #{query_num}] {len(ao_responses)} responses received")

                    # Format result for the agent
                    result_parts = [
                        f"Query #{query_num} — {len(ao_responses)} AO responses "
                        f"(positions: {pos_desc}, {len(positions)} tokens):\n"
                    ]
                    for i, resp in enumerate(ao_responses):
                        result_parts.append(
                            f"--- AO Response {i+1}/{len(ao_responses)} ---\n{resp}\n"
                        )

                    result_text = "\n".join(result_parts)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

                    # Decode the selected tokens for debugging
                    selected_token_ids = [position_map.context_token_ids[p] for p in positions]
                    decoded_selected_text = tokenizer.decode(selected_token_ids, skip_special_tokens=False)

                    transcript_turns.append({
                        "role": "agent",
                        "query_number": query_num,
                        "question": question,
                        "positions": position_selection or "full",
                        "positions_resolved": pos_desc,
                        "n_positions": len(positions),
                        "decoded_selected_text": decoded_selected_text,
                    })
                    transcript_turns.append({
                        "role": "ao",
                        "query_number": query_num,
                        "responses": ao_responses,
                    })

        messages.append({"role": "assistant", "content": assistant_content})

        if structured_findings:
            print(f"  Investigation complete after {turn_num} turns, "
                  f"{stats['n_ao_queries']} AO queries (findings submitted)")
            break

        if not has_tool_use:
            print(f"  Investigation complete after {turn_num} turns, "
                  f"{stats['n_ao_queries']} AO queries (no tool call)")
            break

        messages.append({"role": "user", "content": tool_results})

    # Save transcript
    transcript_dir = output_dir / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_dir / f"{pid}_transcript.json"
    transcript_data = {
        "prompt_id": pid,
        "interest_score": entry.interest_score,
        "verification_score": entry.verification_score,
        "question": entry.third_person_question,
        "reference_answer": entry.third_person_answer,
        "system_prompt": system_prompt,
        "n_agent_turns": turn_num,
        "n_ao_queries": stats["n_ao_queries"],
        "token_usage": {
            "input_tokens": stats["input_tokens"],
            "cache_read_tokens": stats["cache_read_tokens"],
            "cache_create_tokens": stats["cache_create_tokens"],
            "output_tokens": stats["output_tokens"],
        },
        "decoded_tokenized_context": position_map.decoded_text,
        "segments": [
            {"name": seg.name, "description": seg.description,
             "start_pos": seg.start_pos, "end_pos": seg.end_pos}
            for seg in position_map.segments
        ],
        "ao_query_log": transcript_turns,
        "messages": serialize_messages(messages),
    }
    transcript_path.write_text(json.dumps(transcript_data, indent=2, ensure_ascii=False))
    print(f"  Saved transcript to {transcript_path}")

    final_answer = None
    if structured_findings:
        final_answer = structured_findings.get("answer", "")

    return AgentTranscript(
        prompt_id=pid,
        turns=transcript_turns,
        final_answer=final_answer,
        structured_findings=structured_findings,
        n_agent_turns=turn_num,
        n_ao_queries=stats["n_ao_queries"],
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def run_ao_agent_eval(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    model_name: str,
    verbalizer_lora_path: str,
    run_dir: str,
    output_dir: str,
    max_entries: int | None,
    max_agent_turns: int,
    ao_completions: int,
    ao_max_context_tokens: int,
    ao_max_new_tokens: int,
    agent_concurrency: int,
    agent_model: str,
    judge_concurrency: int,
    min_verification_score: int,
    prompt_ids: list[str] | None = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load eval entries
    entries = load_model_understanding_eval_entries(
        run_dir,
        max_entries=max_entries,
        min_verification_score=min_verification_score,
    )
    if prompt_ids:
        prompt_id_set = set(prompt_ids)
        entries = [e for e in entries if e.prompt_id in prompt_id_set]
        missing = prompt_id_set - {e.prompt_id for e in entries}
        if missing:
            print(f"WARNING: {len(missing)} prompt IDs not found: {sorted(missing)}")
    print(f"Loaded {len(entries)} eval entries")

    # Load AO adapter (ensure_default_adapter wraps model in PeftModel first)
    ensure_default_adapter(model)
    sanitized_name, training_config = base_experiment.load_oracle_adapter(model, verbalizer_lora_path)
    selected_layer_combination = default_layer_combination(training_config)
    config = build_verbalizer_eval_config(
        model_name=model_name,
        training_config=training_config,
        eval_batch_size=AO_EVAL_BATCH_SIZE,
        generation_kwargs={
            "do_sample": True,
            "temperature": AO_TEMPERATURE,
            "max_new_tokens": ao_max_new_tokens,
        },
        selected_layer_combination=selected_layer_combination,
    )
    base_experiment.assert_training_config_matches_verbalizer_eval_config(config, training_config)
    ensure_default_adapter(model)
    print(f"Loaded AO adapter: {verbalizer_lora_path}")
    print(f"  Layers: {config.selected_act_layers} (from {selected_layer_combination})")

    # Build token position maps for all entries
    print("Building token position maps...")
    position_maps = [build_token_position_map(entry, tokenizer) for entry in entries]

    # Check for already-completed entries (resume support)
    jsonl_path = output_path / "agent_results.jsonl"
    completed_ids: set[str] = set()
    existing_results: list[dict] = []
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    completed_ids.add(record["prompt_id"])
                    existing_results.append(record)
        print(f"Resuming: {len(completed_ids)} already completed")

    remaining_indices = [
        i for i, entry in enumerate(entries) if entry.prompt_id not in completed_ids
    ]
    if not remaining_indices:
        print("All entries already completed")
        return

    print(f"\nRunning {len(remaining_indices)} agent investigations "
          f"(concurrency={agent_concurrency}, model={agent_model})...")

    # Start batch worker and rate tracker
    ao_worker = AOBatchWorker(
        model=model,
        tokenizer=tokenizer,
        config=config,
        verbalizer_lora_name=sanitized_name,
        device=device,
        ao_completions=ao_completions,
        ao_max_new_tokens=ao_max_new_tokens,
        batch_size=AO_EVAL_BATCH_SIZE,
    )
    ao_worker.start()
    token_rate_tracker.start()

    anthropic_client = anthropic.AsyncAnthropic()

    all_results = list(existing_results)
    write_lock = asyncio.Lock()
    jsonl_file = open(jsonl_path, "a")
    semaphore = asyncio.Semaphore(agent_concurrency)
    progress = {"done": 0}

    async def investigate_with_semaphore(idx: int):
        entry = entries[idx]
        pos_map = position_maps[idx]
        async with semaphore:
            transcript = await investigate_one(
                entry=entry,
                position_map=pos_map,
                ao_worker=ao_worker,
                anthropic_client=anthropic_client,
                tokenizer=tokenizer,
                agent_model=agent_model,
                max_agent_turns=max_agent_turns,
                max_ao_context_tokens=ao_max_context_tokens,
                ao_completions=ao_completions,
                output_dir=output_path,
            )

            record = {
                "prompt_id": entry.prompt_id,
                "interest_score": entry.interest_score,
                "verification_score": entry.verification_score,
                "question_text": entry.third_person_question,
                "reference_answer": entry.third_person_answer,
                "response_text": transcript.final_answer,
                "structured_findings": transcript.structured_findings,
                "n_agent_turns": transcript.n_agent_turns,
                "n_ao_queries": transcript.n_ao_queries,
                "investigated_at": datetime.now(timezone.utc).isoformat(),
            }

            async with write_lock:
                all_results.append(record)
                jsonl_file.write(json.dumps(record) + "\n")
                jsonl_file.flush()

            progress["done"] += 1
            print(f"\n  Progress: {progress['done']}/{len(remaining_indices)} complete")

    await asyncio.gather(*[
        investigate_with_semaphore(idx) for idx in remaining_indices
    ])

    jsonl_file.close()
    ao_worker.stop()
    token_rate_tracker.stop()

    # --- Judge phase ---
    print(f"\n{'='*60}")
    print("Judging agent responses...")
    print(f"{'='*60}")

    from nl_probes.open_ended_eval.model_understanding import EvalMetadata

    response_texts = []
    judge_metadata = []
    for record in all_results:
        response_texts.append(record.get("response_text"))
        judge_metadata.append(EvalMetadata(
            prompt_id=record["prompt_id"],
            interest_score=record["interest_score"],
            verification_score=record["verification_score"],
            chosen_completion_index=0,
            question_text=record["question_text"],
            reference_answer=record["reference_answer"],
            question_mode=AO_QUESTION_MODE,
            answer_mode=AO_ANSWER_MODE,
            full_context_token_count=0,
            selected_context_token_count=None,
            activation_window_mode="ao_agent",
            used_full_context=None,
        ))

    scored_results = await judge_response_texts(
        response_texts,
        judge_metadata,
        concurrency=judge_concurrency,
    )
    scored_dicts = [asdict(r) for r in scored_results]
    metrics = compute_judge_metrics(scored_dicts)

    # Save final output
    final_output = {
        "config": {
            "agent_model": agent_model,
            "verbalizer_lora_path": verbalizer_lora_path,
            "model_name": model_name,
            "run_dir": run_dir,
            "max_agent_turns": max_agent_turns,
            "ao_completions_per_query": ao_completions,
            "ao_temperature": AO_TEMPERATURE,
            "ao_max_context_tokens": ao_max_context_tokens,
            "ao_max_new_tokens": ao_max_new_tokens,
            "agent_concurrency": agent_concurrency,
            "min_verification_score": min_verification_score,
        },
        "num_entries": len(all_results),
        "scored_results": scored_dicts,
        "metrics": metrics,
    }

    final_path = output_path / "ao_agent_results.json"
    final_path.write_text(json.dumps(final_output, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(scored_dicts)} scored results to {final_path}")
    print(f"\n=== Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description="AO Agent Eval: Multi-turn Opus agent queries Activation Oracle",
    )
    parser.add_argument(
        "--verbalizer-lora-path", type=str, required=True,
        help="Path to AO LoRA adapter",
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Path to model understanding run directory",
    )
    parser.add_argument(
        "--model-name", type=str, required=True,
        help="Target model name (e.g., Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory for output files",
    )
    parser.add_argument(
        "--max-entries", type=int, default=None,
        help="Limit number of eval entries (for testing)",
    )
    parser.add_argument(
        "--max-agent-turns", type=int, default=DEFAULT_MAX_AGENT_TURNS,
        help=f"Max Opus agent turns per investigation (default {DEFAULT_MAX_AGENT_TURNS})",
    )
    parser.add_argument(
        "--ao-completions-per-query", type=int, default=AO_COMPLETIONS_PER_QUERY,
        help=f"AO completions per query (default {AO_COMPLETIONS_PER_QUERY})",
    )
    parser.add_argument(
        "--ao-max-context-tokens", type=int, default=MAX_AO_CONTEXT_TOKENS,
        help=f"Max tokens in AO context window (default {MAX_AO_CONTEXT_TOKENS})",
    )
    parser.add_argument(
        "--ao-max-new-tokens", type=int, default=AO_MAX_NEW_TOKENS,
        help=f"Max new tokens for AO generation (default {AO_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--agent-concurrency", type=int, default=DEFAULT_AGENT_CONCURRENCY,
        help=f"Max concurrent agent investigations (default {DEFAULT_AGENT_CONCURRENCY})",
    )
    parser.add_argument(
        "--agent-model", type=str, default=DEFAULT_AGENT_MODEL,
        help=f"Anthropic model for agent (default {DEFAULT_AGENT_MODEL})",
    )
    parser.add_argument(
        "--judge-concurrency", type=int, default=DEFAULT_JUDGE_CONCURRENCY,
        help=f"Concurrency for judge API calls (default {DEFAULT_JUDGE_CONCURRENCY})",
    )
    parser.add_argument(
        "--min-verification-score", type=int, default=6,
        help="Minimum verification score (default 6)",
    )
    parser.add_argument(
        "--prompt-ids", type=str, nargs="+", default=None,
        help="Only evaluate these specific prompt IDs (for fast iteration)",
    )
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = load_tokenizer(args.model_name)
    print(f"Loading model: {args.model_name} on {device} with dtype={dtype}")
    model = load_model(args.model_name, dtype)
    model.eval()

    asyncio.run(run_ao_agent_eval(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=args.model_name,
        verbalizer_lora_path=args.verbalizer_lora_path,
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        max_entries=args.max_entries,
        max_agent_turns=args.max_agent_turns,
        ao_completions=args.ao_completions_per_query,
        ao_max_context_tokens=args.ao_max_context_tokens,
        ao_max_new_tokens=args.ao_max_new_tokens,
        agent_concurrency=args.agent_concurrency,
        agent_model=args.agent_model,
        judge_concurrency=args.judge_concurrency,
        min_verification_score=args.min_verification_score,
        prompt_ids=args.prompt_ids,
    ))


if __name__ == "__main__":
    main()
