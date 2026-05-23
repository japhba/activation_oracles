"""
Stage 3: Deep investigation of interesting model behaviors.

For each high-scoring screening result, runs an Opus agent that:
1. Reads the question and proposed counterfactual from stage 2 screening
2. Proposes hypotheses to answer the question
3. Tests hypotheses via counterfactual prompts through a vLLM server
4. Iterates until it has a clear answer
5. Produces a final investigation report

The agent has a `run_prompt` tool that sends prompts to a running vLLM server
and returns the model's response(s).

Usage:
    # 1. Start vLLM server via Slurm:
    sbatch data_pipelines/model_understanding/submit_vllm_server.sh

    # 2. Run investigation (auto-discovers vLLM server from squeue):
    source .env
    .venv/bin/python data_pipelines/model_understanding/investigate.py \
        --input data_pipelines/model_understanding/Qwen3-14B/screening_94_from_completions_1000p_10c.json

    # Or target a specific prompt:
    .venv/bin/python data_pipelines/model_understanding/investigate.py \
        --input data_pipelines/model_understanding/Qwen3-14B/screening_94_from_completions_1000p_10c.json \
        --prompt-id prompt_0113

    # Specify vLLM URL manually:
    .venv/bin/python data_pipelines/model_understanding/investigate.py \
        --input data_pipelines/model_understanding/Qwen3-14B/screening_94_from_completions_1000p_10c.json \
        --vllm-url http://node-5:8000
"""

import argparse
import asyncio
import collections
import functools
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

print = functools.partial(print, flush=True)

import anthropic
from openai import AsyncOpenAI
from data_pipelines.pipeline_utils import load_dotenv

load_dotenv()


class TokenRateTracker:
    """Tracks token usage rates across all concurrent investigations."""

    def __init__(self, interval: int = 60):
        self._interval = interval
        self._input_tokens = 0
        self._cache_read_tokens = 0
        self._output_tokens = 0
        self._last_print = time.monotonic()
        self._lock = None  # set in start()
        self._task = None

    def start(self):
        self._lock = asyncio.Lock()
        self._last_print = time.monotonic()
        self._task = asyncio.create_task(self._print_loop())

    def stop(self):
        if self._task:
            self._task.cancel()

    async def record(self, input_tokens: int, cache_read_tokens: int, output_tokens: int):
        async with self._lock:
            self._input_tokens += input_tokens
            self._cache_read_tokens += cache_read_tokens
            self._output_tokens += output_tokens

    async def _print_loop(self):
        while True:
            await asyncio.sleep(self._interval)
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


# Global tracker, started by run_pipeline.py
token_rate_tracker = TokenRateTracker()


INVESTIGATION_SYSTEM_PROMPT = """\
You are investigating why a language model behaved a certain way \
in response to a user prompt. You will be given:
- The original user prompt
- 10 completions from the model (showing the distribution of behaviors)
- A specific question about the behavior to investigate
- A suggested first counterfactual experiment

Your goal is to **answer the question** by finding the specific prompt \
feature(s) that drive the behavior.

You have a tool `run_prompt` that sends prompts to the same model and returns \
multiple completions (default 10). Use this to run counterfactual experiments.

## Methodology

1. **Read carefully.** Look at all 10 original completions. What pattern do \
you see? What fraction show the behavior? This is your baseline.

2. **Start with the suggested counterfactual.** Run it first — it's a good \
starting point. Then examine whether it cleanly isolates the factor in question.

3. **Propose and test hypotheses.** Each hypothesis should predict that \
changing a specific prompt feature will shift the behavior rate. Design \
experiments that change ONE thing at a time.

4. **Look at distributions, not single samples.** With 10 completions, you \
can see rates. A shift from 8/10 to 2/10 is meaningful. A shift from 5/10 to \
4/10 is noise.

5. **Be efficient.** Most findings can be established in 5-8 experiments. \
Once you have a clean swing (0/10 → 10/10 or similar), submit your \
findings. Don't run additional experiments just to be thorough — only \
run more if the evidence is genuinely ambiguous.

6. **Watch for pitfalls:**
   - A counterfactual that changes the text may also change the model's \
     interpretation of the task — check for this
   - Don't over-interpret single samples

## Output format

Write a final report with:
- **Question**: The question you investigated (restate it)
- **Answer**: Your best answer, in 1-3 sentences
- **Evidence**: The key experiments and their results (rates/counts)
- **Summary table**: All conditions tested with their behavior rates
- **Confidence**: High/medium/low
- **Untested questions**: Only truly untestable things

Be specific. Give counts ("8/10 completions showed X, vs 2/10 in the \
counterfactual").

## Multi-turn prompts

Some prompts are multi-turn conversations where the model sees prior \
user and assistant turns before generating a response to the final user \
message. For these, use the `messages` parameter of `run_prompt` (a list \
of {role, content} objects) instead of `user_message`. You can modify \
ANY turn — change what the user asked in turn 1, rewrite the assistant's \
prior response, remove turns, add turns, or reorder them. Change one \
thing at a time to isolate causal factors.

Note: prior assistant turns in the original conversation were generated \
by a different model (GPT-3.5/GPT-4), not by the target model. This is \
"off-policy" context. If you suspect the off-policy context is driving \
behavior, test this by replacing the assistant turns with neutral \
responses or removing them entirely.

## Structured findings (REQUIRED)

After writing your report, you MUST call the `submit_findings` tool \
exactly once. This structured output is what downstream analysis uses — \
it must be self-contained and accurate.

Include:
- **question**: The question you investigated
- **baseline_behavior**: What the model actually does in the original \
completions (e.g., "8/10 completions say tumors kill the whale, \
contradicting the passage"). Be specific with counts.
- **answer**: Your answer in 2-4 sentences (third person: "The model \
does X because..."). Use percentages for rates (e.g., "80% of \
completions") rather than fractions. ONLY include claims that are \
directly supported by the experimental evidence. Do NOT speculate \
about training data sources, RLHF effects, or other unverifiable \
causes — put those in the commentary field instead.
- **first_person_answer**: The same answer rephrased in first person \
as if the model is explaining itself ("I do X because..."). Use \
percentages for rates (e.g., "I do X in 80% of responses") rather \
than fractions. Same rule: only verifiable claims, no speculation.
- **commentary**: Optional speculation or interpretation that goes \
beyond what the experiments can verify. Training data hypotheses, \
possible RLHF effects, connections to known model behaviors, etc. \
Keep this separate from the answer.
- **confidence**: high/medium/low
- **key_evidence**: 3-5 most decisive experiments, referenced by their \
experiment number (shown as "Experiment #N" in tool results). Do NOT \
copy full prompts — just reference experiment numbers. The actual \
prompts and responses are logged automatically.

"""

SUBMIT_FINDINGS_TOOL = {
    "name": "submit_findings",
    "description": (
        "Submit your final structured findings. Call this ONCE at the very end "
        "of your investigation, after writing your report. This is required."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question investigated (restate it concisely)",
            },
            "baseline_behavior": {
                "type": "string",
                "description": (
                    "What the model actually does in the original 10 completions. "
                    "Be specific with counts (e.g., '8/10 completions say tumors "
                    "kill the whale, contradicting the passage')"
                ),
            },
            "answer": {
                "type": "string",
                "description": (
                    "Your answer in 2-4 sentences, third person. Use "
                    "percentages for rates (e.g., '80% of completions'). "
                    "ONLY include claims directly supported by the "
                    "experimental evidence — e.g., 'changing X to Y shifts "
                    "the behavior from 80% to 10%.' Do NOT speculate about "
                    "training data sources, RLHF effects, or other causes "
                    "you cannot verify from the experiments. Put speculation "
                    "in the commentary field."
                ),
            },
            "first_person_answer": {
                "type": "string",
                "description": (
                    "The same answer rephrased in first person, as if the "
                    "model is explaining its own behavior. Use percentages "
                    "for rates. Same rule: only verifiable claims from the "
                    "experiments, no speculation about training data or "
                    "unobservable causes."
                ),
            },
            "commentary": {
                "type": "string",
                "description": (
                    "Optional speculation or interpretation beyond what the "
                    "experiments verify. Training data hypotheses, possible "
                    "RLHF effects, connections to known behaviors, etc. "
                    "This is kept separate from the answer so the answer "
                    "remains factual and verifiable."
                ),
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Confidence in the answer",
            },
            "key_evidence": {
                "type": "array",
                "description": "3-5 most decisive experiments",
                "items": {
                    "type": "object",
                    "properties": {
                        "experiment_number": {
                            "type": "integer",
                            "description": "The experiment number (from 'Experiment #N' in tool results)",
                        },
                        "what_changed": {
                            "type": "string",
                            "description": "One sentence: what was modified in the prompt",
                        },
                        "baseline_rate": {
                            "type": "string",
                            "description": "Behavior rate in baseline (e.g., '8/10')",
                        },
                        "counterfactual_rate": {
                            "type": "string",
                            "description": "Behavior rate after change (e.g., '1/10')",
                        },
                    },
                    "required": ["experiment_number", "what_changed", "baseline_rate", "counterfactual_rate"],
                },
            },
        },
        "required": ["question", "baseline_behavior", "answer", "first_person_answer", "commentary", "confidence", "key_evidence"],
    },
}

RUN_PROMPT_TOOL = {
    "name": "run_prompt",
    "description": (
        "Send a prompt to the target model via vLLM and get back multiple completions. "
        "Use this to run counterfactual experiments. Returns `n` independent "
        "completions (default 10) so you can see the distribution of behaviors.\n\n"
        "For single-turn experiments, use `user_message`. "
        "For multi-turn experiments, use `messages` (a list of "
        "{role, content} objects representing the full conversation). "
        "When using `messages`, you can modify any turn — user turns, "
        "assistant turns, or both. The model generates a response to the "
        "final user message given the conversation history."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "user_message": {
                "type": "string",
                "description": (
                    "The user message to send (single-turn). "
                    "Ignored if `messages` is provided."
                ),
            },
            "messages": {
                "type": "array",
                "description": (
                    "Full conversation history for multi-turn experiments. "
                    "List of {role, content} objects. The last message should "
                    "be a user message. You can modify any turn to test "
                    "what drives the model's behavior."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "enum": ["user", "assistant", "system"],
                        },
                        "content": {"type": "string"},
                    },
                    "required": ["role", "content"],
                },
            },
            "system_prompt": {
                "type": "string",
                "description": (
                    "Optional system prompt (single-turn only). "
                    "For multi-turn, include system messages in `messages`."
                ),
            },
            "n": {
                "type": "integer",
                "description": "Number of completions to generate. Omit to use the configured default.",
            },
            "temperature": {
                "type": "number",
                "description": "Sampling temperature. Omit to use the configured default.",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Max response tokens. Omit to use the configured default.",
            },
        },
    },
}


@dataclass
class InvestigationResult:
    prompt_id: str
    interest_score: int
    behavior_summary: str
    question: str
    suggested_counterfactual: str
    user_message: str
    completions: list[str]
    investigation_report: str
    experiment_log: list[dict]
    structured_findings: dict | None
    n_agent_turns: int
    n_vllm_calls: int
    total_vllm_completions: int
    input_tokens: int
    cache_read_tokens: int
    cache_create_tokens: int
    output_tokens: int
    investigated_at: str
    messages: list[dict] | None = None
    n_turns: int = 1
    second_person_question: str = ""


def serialize_content_block(block) -> dict:
    """Serialize an Anthropic content block to a JSON-safe dict."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    elif block.type == "thinking":
        return {"type": "thinking", "thinking": block.thinking}
    elif block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name,
                "input": block.input}
    elif block.type == "tool_result":
        return {"type": "tool_result", "tool_use_id": block.tool_use_id,
                "content": block.content}
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


def save_transcript(
    messages: list, screening_result: dict, output_dir: Path, stats: dict,
):
    """Save the full investigation transcript as a readable JSON file."""
    pid = screening_result["prompt_id"]
    transcript_dir = output_dir / "transcripts"
    transcript_dir.mkdir(exist_ok=True)

    transcript = {
        "prompt_id": pid,
        "interest_score": screening_result["interest_score"],
        "behavior_summary": screening_result["behavior_summary"],
        "question": screening_result["question"],
        "n_agent_turns": stats["n_turns"],
        "n_vllm_calls": stats["n_vllm_calls"],
        "total_vllm_completions": stats["total_vllm_completions"],
        "system_prompt": INVESTIGATION_SYSTEM_PROMPT,
        "messages": serialize_messages(messages),
    }

    path = transcript_dir / f"{pid}_transcript.json"
    path.write_text(json.dumps(transcript, indent=2, ensure_ascii=False))
    print(f"  Saved transcript to {path}")


def discover_vllm_url(job_name: str = "vllm_server", port: int = 8000) -> str:
    """Find the vLLM server URL by looking up a Slurm job by name."""
    result = subprocess.run(
        ["squeue", "-u", subprocess.check_output(["whoami"]).decode().strip(),
         "--name", job_name, "--format=%N", "--noheader"],
        capture_output=True, text=True,
    )
    nodes = result.stdout.strip().split("\n")
    nodes = [n.strip() for n in nodes if n.strip()]
    assert len(nodes) > 0, (
        f"No Slurm job named '{job_name}' found. "
        f"Start one with: sbatch data_pipelines/model_understanding/submit_vllm_server.sh"
    )
    assert len(nodes) == 1, f"Multiple Slurm jobs named '{job_name}': {nodes}"
    node = nodes[0]
    url = f"http://{node}:{port}"
    print(f"Discovered vLLM server: {url} (Slurm job '{job_name}' on {node})")
    return url


def load_screening_results(
    input_path: Path, min_score: int, n: int | None,
) -> list[dict]:
    """Load screening results, filtered by minimum score."""
    data = json.loads(input_path.read_text())
    results = data["results"]
    results = [r for r in results if r["interest_score"] >= min_score]
    results.sort(key=lambda r: r["interest_score"], reverse=True)
    if n is not None:
        results = results[:n]
    print(f"Loaded {len(results)} screening results (score >= {min_score}) "
          f"from {input_path}")
    return results


async def run_vllm_prompt(
    vllm_client: AsyncOpenAI,
    model_name: str,
    user_message: str | None = None,
    system_prompt: str | None = None,
    messages: list[dict] | None = None,
    n: int = 10,
    temperature: float = 1.0,
    max_tokens: int = 500,
) -> list[str]:
    """Run a prompt through vLLM and return completions.

    If `messages` is provided, it's used directly as the conversation
    (for multi-turn). Otherwise, `user_message` + optional `system_prompt`
    are used (single-turn, backward compatible).
    """
    if messages is None:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

    response = await vllm_client.chat.completions.create(
        model=model_name,
        messages=messages,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    return [choice.message.content for choice in response.choices]


async def handle_tool_call(
    tool_name: str,
    tool_input: dict,
    vllm_client: AsyncOpenAI,
    model_name: str,
    stats: dict,
    experiment_log: list,
    vllm_n_completions: int,
    vllm_temperature: float,
    vllm_max_tokens: int,
) -> str:
    """Execute a tool call and return the result as a string."""
    assert tool_name == "run_prompt", f"Unknown tool: {tool_name}"

    n = tool_input.get("n") or vllm_n_completions
    temperature = tool_input.get("temperature") if "temperature" in tool_input else vllm_temperature
    max_tokens = min(tool_input.get("max_tokens") or vllm_max_tokens, 1000)

    # vLLM requires n=1 for greedy (temperature=0); override n since
    # multiple identical samples are pointless anyway
    if temperature == 0 and n > 1:
        n = 1

    # Multi-turn: use messages directly; single-turn: use user_message
    if "messages" in tool_input:
        msgs = tool_input["messages"]
        assert isinstance(msgs, list), (
            f"Expected messages to be a list, got {type(msgs).__name__}"
        )
        completions = await run_vllm_prompt(
            vllm_client=vllm_client,
            model_name=model_name,
            messages=msgs,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        completions = await run_vllm_prompt(
            vllm_client=vllm_client,
            model_name=model_name,
            user_message=tool_input["user_message"],
            system_prompt=tool_input.get("system_prompt"),
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    stats["n_vllm_calls"] += 1
    stats["total_vllm_completions"] += len(completions)

    exp_num = stats["n_vllm_calls"]
    log_entry = {
        "exp_num": exp_num,
        "completions": completions,
        "n": n,
        "temperature": temperature,
    }
    if "messages" in tool_input:
        log_entry["messages"] = tool_input["messages"]
        log_entry["prompt"] = tool_input["messages"][-1]["content"]
    else:
        log_entry["prompt"] = tool_input["user_message"]
        log_entry["system_prompt"] = tool_input.get("system_prompt")
    experiment_log.append(log_entry)

    print(f"    vLLM call #{exp_num}: {len(completions)} completions "
          f"(temp={temperature}, max_tokens={max_tokens})")

    parts = [f"Experiment #{exp_num} — Generated {len(completions)} completions:\n"]
    for i, text in enumerate(completions):
        parts.append(f"--- Completion {i+1}/{len(completions)} ---\n{text}\n")
    return "\n".join(parts)


def build_investigation_prompt(screening_result: dict) -> str:
    """Build the initial prompt for the Opus investigation agent."""
    # Format the 10 original completions
    completions = screening_result["completions"]
    completions_text = ""
    for i, comp in enumerate(completions):
        completions_text += f"--- Completion {i+1}/10 ---\n{comp}\n\n"

    hypotheses_text = ""
    if screening_result.get("hypotheses"):
        hypotheses_text = (
            f"**Initial hypotheses from screening:**\n"
            f"{screening_result['hypotheses']}\n\n"
        )

    # Multi-turn: show full conversation history
    messages = screening_result.get("messages")
    is_multi_turn = messages and len(messages) > 1
    if is_multi_turn:
        conv_parts = ["### Conversation History\n"]
        conv_parts.append(
            f"This is a {len([m for m in messages if m['role'] == 'user'])}-turn "
            f"conversation. Prior assistant turns are from the original chat "
            f"(GPT-3.5/GPT-4), not from the target model. The completions below are "
            f"the target model's responses to the **final user message**.\n\n"
        )
        for msg in messages[:-1]:
            role = msg["role"].upper()
            conv_parts.append(f"**[{role}]:**\n{msg['content']}\n\n")
        conv_parts.append(
            f"**[USER] (final turn — completions below respond to this):**\n"
            f"{messages[-1]['content']}\n\n"
        )
        prompt_section = "".join(conv_parts)
        tool_guidance = (
            "Investigate the question above. This is a multi-turn conversation, "
            "so you can modify ANY turn — user messages, assistant responses, "
            "or both — to test your hypotheses. Use the `run_prompt` tool with "
            "the `messages` parameter (a list of {role, content} objects) to "
            "send modified conversations to the model. Change one thing at a "
            "time to isolate causal factors."
        )
    else:
        prompt_section = (
            f"### Original User Prompt\n"
            f"{screening_result['user_message']}\n\n"
        )
        tool_guidance = (
            "Investigate the question above. The screening stage proposed "
            "multiple hypotheses — test them systematically and try to "
            "distinguish between them. Use the `run_prompt` tool to send "
            "modified prompts to the same model."
        )

    return (
        f"## Prompt to Investigate\n\n"
        f"**Prompt ID:** {screening_result['prompt_id']}\n\n"
        f"### Screening Analysis\n"
        f"**Behavior observed across 10 completions:** "
        f"{screening_result['behavior_summary']}\n\n"
        f"**Question to investigate:** {screening_result['question']}\n\n"
        f"{hypotheses_text}"
        f"**Suggested first counterfactual:** "
        f"{screening_result['counterfactual']}\n\n"
        f"{prompt_section}"
        f"### 10 Completions from the Target Model\n\n"
        f"{completions_text}"
        f"---\n\n"
        f"{tool_guidance}"
    )


async def investigate_one(
    screening_result: dict,
    anthropic_client: anthropic.AsyncAnthropic,
    vllm_client: AsyncOpenAI,
    model_name: str,
    output_dir: Path,
    investigation_model: str,
    max_tokens: int,
    thinking_budget: int,
    max_agent_turns: int,
    vllm_n_completions: int,
    vllm_temperature: float,
    vllm_max_tokens: int,
) -> InvestigationResult:
    """Run a full investigation on one screening result."""
    pid = screening_result["prompt_id"]
    print(f"\n{'='*60}")
    print(f"Investigating {pid} (score {screening_result['interest_score']})")
    print(f"Q: {screening_result['question'][:120]}")
    print(f"{'='*60}")

    stats = {"n_vllm_calls": 0, "total_vllm_completions": 0, "n_turns": 0,
             "input_tokens": 0, "cache_read_tokens": 0,
             "cache_create_tokens": 0, "output_tokens": 0}
    experiment_log: list[dict] = []

    messages = [
        {"role": "user", "content": build_investigation_prompt(screening_result)},
    ]

    final_text_parts = []
    n_turns = 0

    while n_turns < max_agent_turns:
        n_turns += 1
        print(f"  Turn {n_turns}/{max_agent_turns}...")

        # Add cache breakpoint to the last user message so the API can
        # cache the conversation prefix (system + all prior turns).
        # First strip old breakpoints to stay under the 4-breakpoint limit.
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

        response = await anthropic_client.messages.create(
            model=investigation_model,
            max_tokens=max_tokens,
            thinking={"type": "enabled", "budget_tokens": thinking_budget},
            system=[{"type": "text", "text": INVESTIGATION_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            tools=[RUN_PROMPT_TOOL, SUBMIT_FINDINGS_TOOL],
        )

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

        structured_findings = None

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
                    print(f"    [submit_findings] confidence={block.input.get('confidence')}, "
                          f"{len(block.input.get('key_evidence', []))} key experiments")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Findings recorded.",
                    })
                else:
                    print(f"    [tool_use] {block.name}("
                          f"msg={block.input.get('user_message', '')[:80]}...)")

                    result_text = await handle_tool_call(
                        tool_name=block.name,
                        tool_input=block.input,
                        vllm_client=vllm_client,
                        model_name=model_name,
                        stats=stats,
                        experiment_log=experiment_log,
                        vllm_n_completions=vllm_n_completions,
                        vllm_temperature=vllm_temperature,
                        vllm_max_tokens=vllm_max_tokens,
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

        messages.append({"role": "assistant", "content": assistant_content})

        if structured_findings:
            print(f"  Investigation complete after {n_turns} turns, "
                  f"{stats['n_vllm_calls']} vLLM calls (findings submitted)")
            break

        if not has_tool_use:
            print(f"  Investigation complete after {n_turns} turns, "
                  f"{stats['n_vllm_calls']} vLLM calls (no findings tool call)")
            break

        messages.append({"role": "user", "content": tool_results})

    stats["n_turns"] = n_turns

    save_transcript(messages, screening_result, output_dir, stats)

    investigation_report = "\n\n".join(final_text_parts)

    return InvestigationResult(
        prompt_id=pid,
        interest_score=screening_result["interest_score"],
        behavior_summary=screening_result["behavior_summary"],
        question=screening_result["question"],
        suggested_counterfactual=screening_result["counterfactual"],
        user_message=screening_result["user_message"],
        completions=screening_result["completions"],
        investigation_report=investigation_report,
        experiment_log=experiment_log,
        structured_findings=structured_findings,
        n_agent_turns=n_turns,
        n_vllm_calls=stats["n_vllm_calls"],
        total_vllm_completions=stats["total_vllm_completions"],
        input_tokens=stats["input_tokens"],
        cache_read_tokens=stats["cache_read_tokens"],
        cache_create_tokens=stats["cache_create_tokens"],
        output_tokens=stats["output_tokens"],
        investigated_at=datetime.now(timezone.utc).isoformat(),
        messages=screening_result.get("messages"),
        n_turns=screening_result.get("n_turns", 1),
        second_person_question=screening_result.get("second_person_question", ""),
    )


async def main():
    parser = argparse.ArgumentParser(
        description="Investigate interesting model behaviors with Opus agent",
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to screening results JSON",
    )
    parser.add_argument(
        "--vllm-url", type=str, default=None,
        help="vLLM server URL (e.g. http://node-5:8000). "
             "If omitted, auto-discovers from Slurm job named 'vllm_server'.",
    )
    parser.add_argument(
        "--vllm-model", type=str, default=None,
        help="Model name on the vLLM server. If omitted, auto-detected.",
    )
    parser.add_argument(
        "--min-score", type=int, default=4,
        help="Minimum interest score to investigate (default 4)",
    )
    parser.add_argument(
        "--prompt-id", type=str, default=None,
        help="Investigate a specific prompt by ID (e.g. prompt_0113)",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Limit to first N results (after score filtering)",
    )
    parser.add_argument(
        "--concurrency", type=int, required=True,
        help="Max concurrent investigations",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Anthropic model for investigation",
    )
    parser.add_argument(
        "--max-tokens", type=int, required=True,
        help="Max response tokens for investigation model",
    )
    parser.add_argument(
        "--thinking-budget", type=int, required=True,
        help="Extended thinking budget for investigation model",
    )
    parser.add_argument(
        "--max-agent-turns", type=int, required=True,
        help="Max agent turns per investigation",
    )
    parser.add_argument(
        "--vllm-n-completions", type=int, required=True,
        help="Number of vLLM completions per experiment",
    )
    parser.add_argument(
        "--vllm-temperature", type=float, required=True,
        help="vLLM sampling temperature",
    )
    parser.add_argument(
        "--vllm-max-tokens", type=int, required=True,
        help="Max vLLM response tokens",
    )
    args = parser.parse_args()

    # Discover or use provided vLLM URL
    vllm_url = args.vllm_url or discover_vllm_url()

    # Set up vLLM client (OpenAI-compatible API)
    vllm_client = AsyncOpenAI(base_url=f"{vllm_url}/v1", api_key="unused")

    # Auto-detect model name from vLLM server
    model_name = args.vllm_model
    if model_name is None:
        models = await vllm_client.models.list()
        model_name = models.data[0].id
        print(f"Auto-detected vLLM model: {model_name}")

    # Verify vLLM server is reachable
    print(f"Testing vLLM server at {vllm_url}...")
    test_completions = await run_vllm_prompt(
        vllm_client, model_name, "Say hello.", n=1, max_tokens=20,
    )
    print(f"  vLLM server OK: {test_completions[0][:100]}")

    # Load screening results
    if args.prompt_id:
        data = json.loads(args.input.read_text())
        results = [r for r in data["results"]
                   if r["prompt_id"] == args.prompt_id]
        assert len(results) == 1, (
            f"Prompt '{args.prompt_id}' not found in {args.input}"
        )
        print(f"Targeting prompt: {args.prompt_id}")
    else:
        results = load_screening_results(
            args.input, min_score=args.min_score, n=args.n,
        )

    # Set up Anthropic client
    anthropic_client = anthropic.AsyncAnthropic()

    # Override investigation model if specified
    investigation_model = args.model

    output_dir = args.input.parent
    model_tag = investigation_model.replace("-", "_").split("/")[-1]
    output_path = output_dir / f"investigation_results_{model_tag}.json"
    jsonl_path = output_dir / f"investigation_results_{model_tag}.jsonl"

    # Load existing results for resume
    completed_ids = set()
    existing_results = []
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    completed_ids.add(record["prompt_id"])
                    existing_results.append(record)
        print(f"Resuming: {len(completed_ids)} already investigated")

    remaining = [r for r in results if r["prompt_id"] not in completed_ids]
    if not remaining:
        print("All results already investigated")
        return

    print(f"\nInvestigating {len(remaining)} prompts with {investigation_model} "
          f"(concurrency={args.concurrency})...")

    all_results = list(existing_results)
    write_lock = asyncio.Lock()
    jsonl_file = open(jsonl_path, "a")

    semaphore = asyncio.Semaphore(args.concurrency)
    progress = {"done": 0}

    async def investigate_with_semaphore(screening_result: dict):
        async with semaphore:
            investigation = await investigate_one(
                screening_result=screening_result,
                anthropic_client=anthropic_client,
                vllm_client=vllm_client,
                model_name=model_name,
                output_dir=output_dir,
                investigation_model=investigation_model,
                max_tokens=args.max_tokens,
                thinking_budget=args.thinking_budget,
                max_agent_turns=args.max_agent_turns,
                vllm_n_completions=args.vllm_n_completions,
                vllm_temperature=args.vllm_temperature,
                vllm_max_tokens=args.vllm_max_tokens,
            )
            record = asdict(investigation)

            async with write_lock:
                all_results.append(record)
                jsonl_file.write(json.dumps(record) + "\n")
                jsonl_file.flush()

            progress["done"] += 1
            print(f"\n  Progress: {progress['done']}/{len(remaining)} complete")

    await asyncio.gather(*[
        investigate_with_semaphore(r) for r in remaining
    ])

    jsonl_file.close()

    # Save final JSON
    output = {
        "metadata": {
            "investigation_model": investigation_model,
            "input_file": str(args.input),
            "vllm_url": vllm_url,
            "vllm_model": model_name,
            "n_investigated": len(all_results),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": all_results,
    }
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(all_results)} investigations to {output_path}")

    # Token usage summary (only for newly investigated prompts)
    new_results = [r for r in all_results if r["prompt_id"] not in completed_ids]
    if new_results and "input_tokens" in new_results[0]:
        tot_in = sum(r["input_tokens"] for r in new_results)
        tot_cache_read = sum(r["cache_read_tokens"] for r in new_results)
        tot_cache_create = sum(r["cache_create_tokens"] for r in new_results)
        tot_out = sum(r["output_tokens"] for r in new_results)
        tot_all_in = tot_in + tot_cache_read + tot_cache_create
        cache_pct = 100 * tot_cache_read / tot_all_in if tot_all_in > 0 else 0

        print(f"\n{'='*60}")
        print(f"Token usage summary ({len(new_results)} new investigations)")
        print(f"{'='*60}")
        print(f"  Input tokens (non-cache): {tot_in:>12,}")
        print(f"  Cache read tokens:        {tot_cache_read:>12,}")
        print(f"  Cache create tokens:      {tot_cache_create:>12,}")
        print(f"  Output tokens:            {tot_out:>12,}")
        print(f"  Total input tokens:       {tot_all_in:>12,}")
        print(f"  Cache hit rate:           {cache_pct:>11.1f}%")
        print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
