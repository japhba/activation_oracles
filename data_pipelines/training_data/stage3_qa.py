"""
Stage 3 helpers for the synthetic training data pipeline.

This module owns:
- prompt assets and few-shot rendering
- activation-window selection
- QA prompt preparation shared by async and batch paths
- Anthropic async/batch execution for QA generation
"""

import asyncio
import functools
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

# Force unbuffered output
print = functools.partial(print, flush=True)

import anthropic
from data_pipelines.pipeline_utils import async_api_call, parse_json_response, run_concurrent
from data_pipelines.training_data.stage12_vllm import QA_MODEL, SEED

# --- Stage 3 config ---
FEWSHOT_POOL_PATH = Path(__file__).parent / "fewshot_pool.json"
NUM_FEWSHOT = 0  # how many few-shot examples to sample per prompt
MIN_WINDOW_CHARS = 5
MAX_WINDOW_TOKENS = 500  # cap activation window to limit memory/compute
MAX_WINDOW_CHARS = MAX_WINDOW_TOKENS * 4  # ~4 chars per token
MAX_CONTINUATION_WORDS = 30
API_CONCURRENCY = 40
QA_MAX_TOKENS = 16000
QA_THINKING_BUDGET = 10000

# TODO BEFORE RUN: Limit to 500 tokens context prompt, adjust window size distribution

QA_OUTPUT_FORMAT = """

Respond with a JSON object containing exactly these fields:
```json
{{"question": "...", "answer": "...", "reasoning": "..."}}
```
- "question": The question to ask the activation oracle. Specific and concrete.
- "answer": The ground-truth answer. 1-3 sentences, self-contained.
- "reasoning": Brief explanation of why this QA pair is good training data.

Output ONLY the JSON object, no other text."""

OPEN_ENDED_FMT = (
    "open_ended",
    "Give a genuinely open-ended answer in 1-3 sentences. Be specific and direct. Do NOT structure the question as a binary choice ('Will it do A or B?') — that's a yes/no question in disguise. Instead ask questions like 'How is the model approaching this?', 'Describe the model's state at this point', 'What is the model's understanding of the situation?'",
)

YES_NO_FMT = (
    "yes_no",
    'The answer should be just Yes or No, followed by a one-sentence explanation. The correct answer for this one should be "{polarity}" — construct the question accordingly.',
)


def pick_response_format(rng, qa_type=None):
    """Pick a response format from the currently enabled set."""
    del qa_type

    choice = rng.choice(["open_ended", "open_ended", "yes_no"])
    if choice == "open_ended":
        return OPEN_ENDED_FMT

    polarity = rng.choice(["Yes", "No"])
    return (YES_NO_FMT[0], YES_NO_FMT[1].format(polarity=polarity))


# -- Shared building blocks for all QA system prompts --

QA_PREAMBLE = """\
You are creating training data for an activation oracle — a model that reads \
the internal activations of a language model (Qwen3-8B) and answers questions \
about what the model is processing.

You will see the full context (user prompt + model response, with chat \
template markup) with a marked window of SELECTED TOKENS whose activations \
the oracle will read."""

QA_KNOWLEDGE_CONSTRAINT = """\
KNOWLEDGE CONSTRAINT: The activation oracle is also a Qwen3-8B model — it \
has the same knowledge and reasoning ability as the target model, no more. \
Keep questions at a level Qwen3-8B could plausibly answer. Don't ask \
questions requiring sophisticated meta-cognitive judgment (detecting subtle \
sycophancy, evaluating creative "taste", judging whether scope constraints \
are violated) or elaborate multi-step reasoning about the model's "strategy".

CRITICAL: You (the QA generator) know more than Qwen3-8B. Do NOT use YOUR \
knowledge to construct answers. The answer must be derivable from the \
context text and activations alone. If the answer relies on facts you \
know but that aren't evident in the context, it's a bad answer.

FALLBACK: If you're struggling to find a question that doesn't violate \
these rules, fall back to a simple, verifiable question about the context: \
"What topic is the conversation about?", "What did the user ask?", "Is \
the model still in its thinking phase or has it started responding?", \
"What format is the model using?". A boring correct question is much \
better than a clever question that leaks your knowledge or anchors to text."""

QA_GOOD_QUESTION_PRIORITIES = """\
WHAT MAKES A GOOD QUESTION — priorities in order:
1. USEFUL for training the activation oracle. A simple useful question \
   beats an elaborate speculative one.
2. Among useful questions, prefer more interesting ones over boring ones."""

QA_BAD_GOOD_EXAMPLES = """\
BAD vs GOOD question examples:

BAD: "After describing Aoko as 'a dangerous threat' due to his sheer size, \
will the model transition into listing combat moves?"
WHY BAD: Anchors to a specific text location. The oracle reads activations, \
not a transcript — it has no concept of "after describing X as Y."
GOOD: "Will the model transition into listing combat moves?"

BAD: "What uncertainty does the model express about X during the selected tokens?"
WHY BAD: References "the selected tokens" — an artifact of this process.
GOOD: "What is the model uncertain about at this point?"

BAD: "Is the model internally resolving the tension between X's established \
archetype and the requirement to provide Y abilities, as reflected in how it \
has structured its response so far?"
WHY BAD: Over-complicated, tries too hard to be interesting.
GOOD: "Does the model seem conflicted about the task it's performing?"

BAD: "Has the model already identified the correct answer?" with answer \
"Yes — parasympathetic preganglionic axons originate from the brainstem \
and sacral region, effectively revealing answer D."
WHY BAD: The answer uses YOUR (the generator's) medical knowledge to \
verify correctness. The oracle is a Qwen3-8B model reading activations — \
it can't use external knowledge to judge correctness. Only include facts \
that are visible in the context text or inferable from the model's behavior.
GOOD: "What topic is the model's response about?" → "Anatomy — the model \
is explaining where parasympathetic preganglionic axons originate."

BAD: "After dismissing option C about Jupiter's magnetic field, what will \
the model do next?"
WHY BAD: References specific content ("option C about Jupiter's magnetic \
field") as a positional anchor. Any form of "after [specific content]" is \
text anchoring — including "after completing X", "after dismissing X", \
"after finishing the section on X". The oracle cannot locate these positions.
GOOD: "What will the model address next?"
"""

QA_RULES = """\
RULES:
- Answers must be 1-3 sentences, direct, and specific. Don't write essays.
- The question must be self-contained — everything the oracle needs to answer.
- The answer should be exactly what we want the oracle to output — clean \
  and direct, not explanatory.
- Do NOT anchor questions to specific text locations ("after the paragraph \
  about X", "after describing Y as Z", "having completed section N").
- Never mention "the prefix", "the selected tokens", "the rollouts", or \
  any other artifact of this generation process.
- Write them as standalone Q&A that make sense without any surrounding context.
- The answer does NOT need to be 100% verifiable — ~95% likely correct is fine."""

# -- Mode instructions injected into past prompt --

QA_PAST_MODE_INTERNAL_STATE = (
    "For this example, focus on the model's INTERNAL STATE — confidence, "
    "uncertainty, intent, comprehension depth, tone, or whether the model "
    "is hedging. The question should probe what the model is 'thinking' "
    "beyond what's literally written in the text."
)

QA_PAST_MODE_FACTUAL = (
    "For this example, focus on CONTEXTUAL RECALL — what topic is being "
    "discussed, what the user asked about, what domain the conversation is "
    "in, the user's expertise level, or other facts that require understanding "
    "the broader context beyond the activation window."
)

# -- Assembled system prompts --

QA_SYSTEM_PAST = (
    QA_PREAMBLE
    + """

{mode_instruction}

"""
    + QA_KNOWLEDGE_CONSTRAINT
    + "\n\n"
    + QA_GOOD_QUESTION_PRIORITIES
    + "\n\n"
    + QA_BAD_GOOD_EXAMPLES
    + """
Questions must be about tokens UP TO the activation window. Do NOT ask \
what the model is "planning to do" or "about to generate" — that is \
prediction, not reading past state.

"""
    + QA_RULES
    + "\n\nRESPONSE FORMAT: {format_instruction}"
    + QA_OUTPUT_FORMAT
)

QA_USER_PAST = """\
CONTEXT (prompt + model output up to the activation window, with chat template markup):
{text_before}>>> SELECTED TOKENS ({window_desc}) >>> {window_text} <<< END SELECTED TOKENS <<<

Create a question-answer pair. The answer should require understanding \
the model's internal state — not just reading the selected tokens."""

QA_SYSTEM_FUTURE = (
    QA_PREAMBLE
    + """ The window may fall anywhere in the context. You will also see \
10 ROLLOUT CONTINUATIONS from the truncation point.

Create a question-answer pair about what the model WILL DO NEXT, grounded \
in the rollouts.

"""
    + QA_KNOWLEDGE_CONSTRAINT
    + "\n\n"
    + QA_GOOD_QUESTION_PRIORITIES
    + "\n\n"
    + QA_BAD_GOOD_EXAMPLES
    + """
If the continuations show a striking pattern of INCONSISTENCY — e.g., the \
model commits to the same strategy every time but the specific content \
varies wildly — this is worth noting. But only mention consistency/ \
variability when it's genuinely interesting; most responses don't need it.

When describing how often something happens across continuations, use \
PERCENTAGES (e.g., "80% of the time", "about 60%"), never raw counts \
like "7/10" or "all 10 continuations".

"""
    + QA_RULES
    + "\n\nRESPONSE FORMAT: {format_instruction}"
    + QA_OUTPUT_FORMAT
)

QA_USER_FUTURE = """\
CONTEXT (prompt + model output up to truncation point, with chat template markup):
{text_before}>>> SELECTED TOKENS ({window_desc}) >>> {window_text} <<< END SELECTED TOKENS <<<

ROLLOUT CONTINUATIONS (10 samples from the truncation point):
{rollouts}

Create a question-answer pair about what the model will do next, grounded \
in the rollout patterns. The answer must require reasoning beyond just \
reading the selected tokens."""


def load_fewshot_pool():
    """Load few-shot example pool from JSON file."""
    if not FEWSHOT_POOL_PATH.exists():
        print(f"  WARNING: No few-shot pool at {FEWSHOT_POOL_PATH}, using no examples")
        return {}

    pool = json.loads(FEWSHOT_POOL_PATH.read_text())
    categories = pool.get("categories", {})
    for category, examples in categories.items():
        print(f"  Loaded {len(examples)} few-shot examples for {category}")
    return categories


def format_fewshot_block(examples, rng, n=NUM_FEWSHOT):
    """Format a random sample of few-shot examples into a prompt block."""
    if not examples or n <= 0:
        return ""

    fewshot_rng = random.Random(rng.random())
    sampled = fewshot_rng.sample(examples, min(n, len(examples)))
    fewshot_rng.shuffle(sampled)
    lines = ["\nHere are examples of the kinds of Q/A pairs we want:\n"]
    for i, ex in enumerate(sampled, 1):
        lines.append(f"Example {i}:")
        lines.append(f"Q: {ex['question']}")
        lines.append(f"A: {ex['answer']}")
        lines.append("")
    lines.append(
        "Now create a NEW question-answer pair for the context above. "
        "Do not copy these examples — use them only as style/quality references.\n"
    )
    return "\n".join(lines)


def select_activation_window(entry, rng):
    """Select a token window for this entry."""
    prompt_text = entry.get("formatted_prompt", "")
    if entry["qa_type"] == "future":
        response_text = entry.get("prefix", "")
    else:
        response_text = entry.get("response_text", "")

    full_text = prompt_text + response_text
    prompt_len = len(prompt_text)

    if len(full_text) <= MIN_WINDOW_CHARS:
        return 0, len(full_text), full_text, "full context", prompt_len

    log_min = math.log(MIN_WINDOW_CHARS)
    max_chars = min(len(full_text), MAX_WINDOW_CHARS)
    log_max = math.log(max(max_chars, MIN_WINDOW_CHARS + 1))
    width = int(math.exp(rng.uniform(log_min, log_max)))
    width = min(width, max_chars)

    if entry["qa_type"] == "future":
        end = len(full_text)
        start = max(0, end - width)
    else:
        max_start = len(full_text) - width
        start = rng.randint(0, max_start)
        end = start + width

    while start > 0 and full_text[start] != " ":
        start -= 1
    while end < len(full_text) and full_text[end - 1] != " ":
        end += 1

    window_text = full_text[start:end]
    return start, end, window_text, f"{end - start} chars", prompt_len


def duplicate_entries_for_past_and_future(entries):
    """Duplicate each entry into a past variant and a future variant.

    Each prompt produces two training examples:
    - past: uses the full response, window can be anywhere
    - future: uses the truncated prefix + continuations
    """
    import copy

    duplicated = []
    for entry in entries:
        past = copy.deepcopy(entry)
        past["id"] = f"{entry['id']}_past"
        past["qa_type"] = "past"
        duplicated.append(past)

        future = copy.deepcopy(entry)
        future["id"] = f"{entry['id']}_future"
        future["qa_type"] = "future"
        duplicated.append(future)

    n_past = sum(1 for e in duplicated if e["qa_type"] == "past")
    n_future = sum(1 for e in duplicated if e["qa_type"] == "future")
    print(f"  Duplicated {len(entries)} entries -> {len(duplicated)} ({n_past} past, {n_future} future)")
    return duplicated


def _decode_entries_for_qa(entries, target_model):
    """Decode token IDs back to canonical text for prompt construction."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(target_model)
    for entry in entries:
        entry["formatted_prompt"] = tokenizer.decode(
            entry["prompt_token_ids"],
            skip_special_tokens=False,
        )
        entry["response_text"] = tokenizer.decode(
            entry["response_token_ids"],
            skip_special_tokens=False,
        )
        if entry["qa_type"] == "future" and "truncation_token_idx" in entry:
            entry["prefix"] = tokenizer.decode(
                entry["response_token_ids"][: entry["truncation_token_idx"]],
                skip_special_tokens=False,
            )


def _load_completed_ids(jsonl_path):
    completed_ids = set()
    if jsonl_path and jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    completed_ids.add(json.loads(line)["id"])
        print(f"  Resuming: {len(completed_ids)} entries already completed")
    return completed_ids


def _truncate_continuation(text, max_words=MAX_CONTINUATION_WORDS):
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def _prepare_qa_jobs(entries, fewshot_pool, completed_ids=None):
    """Prepare prompt/render metadata for every entry and return incomplete jobs."""
    if completed_ids is None:
        completed_ids = set()

    rng = random.Random(SEED + 3)
    jobs = []

    for entry in entries:
        qa_type = entry["qa_type"]
        fmt_name, fmt_instruction = pick_response_format(rng, qa_type=qa_type)
        entry["response_format"] = fmt_name

        w_start, w_end, w_text, w_desc, prompt_len = select_activation_window(entry, rng)
        entry["window_start"] = w_start
        entry["window_end"] = w_end
        entry["window_desc"] = w_desc
        entry["selected_text"] = w_text
        entry["prompt_len"] = prompt_len

        prompt_text = entry.get("formatted_prompt", "")
        if qa_type == "future":
            response_text = entry.get("prefix", "")
        else:
            response_text = entry.get("response_text", "")
        full_text = prompt_text + response_text

        text_before = full_text[:w_start]
        entry["prefix_text"] = text_before

        if qa_type == "past":
            past_mode = "factual" if rng.random() < 0.3 else "internal_state"
            entry["past_mode"] = past_mode
            mode_instruction = QA_PAST_MODE_INTERNAL_STATE if past_mode == "internal_state" else QA_PAST_MODE_FACTUAL
            system = QA_SYSTEM_PAST.format(
                mode_instruction=mode_instruction,
                format_instruction=fmt_instruction,
            )
            fewshot_examples = fewshot_pool.get(f"past_{past_mode}", [])
            system += format_fewshot_block(fewshot_examples, rng)
            user_msg = QA_USER_PAST.format(
                text_before=text_before,
                window_desc=w_desc,
                window_text=w_text,
            )
        else:
            system = QA_SYSTEM_FUTURE.format(format_instruction=fmt_instruction)
            fewshot_examples = fewshot_pool.get("future_interesting", [])
            system += format_fewshot_block(fewshot_examples, rng)
            rollouts_text = "\n".join(
                f"  [{j + 1}]: {_truncate_continuation(c)}" for j, c in enumerate(entry["continuations"])
            )
            user_msg = QA_USER_FUTURE.format(
                text_before=text_before,
                window_desc=w_desc,
                window_text=w_text,
                rollouts=rollouts_text,
            )

        entry["generator_system_prompt"] = system
        entry["generator_user_prompt"] = user_msg

        if entry["id"] in completed_ids:
            continue

        jobs.append(
            {
                "entry": entry,
                "system": system,
                "user_msg": user_msg,
                "fmt_name": fmt_name,
            }
        )

    return jobs


def _parse_qa_text(resp_text):
    if not resp_text.strip():
        return None
    try:
        data = parse_json_response(resp_text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict) or "question" not in data or "answer" not in data:
        return None
    return data


def _extract_qa_from_response(resp):
    resp_text = "\n".join(b.text for b in resp.content if b.type == "text")
    return _parse_qa_text(resp_text)


def _jsonl_record_from_entry(entry):
    return {
        "id": entry["id"],
        "question": entry["question"],
        "answer": entry["answer"],
        "qa_reasoning": entry.get("qa_reasoning", ""),
        "response_format": entry["response_format"],
        "window_start": entry["window_start"],
        "window_end": entry["window_end"],
        "window_desc": entry["window_desc"],
        "selected_text": entry["selected_text"],
        "prefix_text": entry.get("prefix_text", ""),
        "past_mode": entry.get("past_mode", ""),
        "generator_system_prompt": entry.get("generator_system_prompt", ""),
        "generator_user_prompt": entry.get("generator_user_prompt", ""),
    }


def _merge_jsonl_results_into_entries(entries, jsonl_path):
    if not jsonl_path or not jsonl_path.exists():
        return

    results_by_id = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                result = json.loads(line)
                results_by_id[result["id"]] = result

    for entry in entries:
        if entry["id"] not in results_by_id:
            continue
        result = results_by_id[entry["id"]]
        for key in [
            "question",
            "answer",
            "qa_reasoning",
            "response_format",
            "window_start",
            "window_end",
            "window_desc",
            "selected_text",
            "prefix_text",
            "past_mode",
            "generator_system_prompt",
            "generator_user_prompt",
        ]:
            if key in result:
                entry[key] = result[key]


async def _request_qa(async_client, system, user_msg):
    return await async_api_call(
        async_client,
        model=QA_MODEL,
        max_tokens=QA_MAX_TOKENS,
        temperature=1,
        thinking={"type": "enabled", "budget_tokens": QA_THINKING_BUDGET},
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )


async def stage_3_generate_qa(entries, target_model, jsonl_path=None):
    """Generate QA pairs using Sonnet. Streams results to JSONL for crash resilience."""
    entries = duplicate_entries_for_past_and_future(entries)
    print(f"\n=== STAGE 3: Generate QA pairs ({len(entries)} entries) ===")

    _decode_entries_for_qa(entries, target_model)
    fewshot_pool = load_fewshot_pool()
    completed_ids = _load_completed_ids(jsonl_path)
    jobs = _prepare_qa_jobs(entries, fewshot_pool, completed_ids)
    print(f"  {len(jobs)} entries to process")

    if not jobs:
        _merge_jsonl_results_into_entries(entries, jsonl_path)
        return entries

    async_client = anthropic.AsyncAnthropic()
    jsonl_file = open(jsonl_path, "a") if jsonl_path else None
    write_lock = asyncio.Lock()

    async def generate_qa(i, job):
        entry = job["entry"]
        system = job["system"]
        user_msg = job["user_msg"]
        fmt_name = job["fmt_name"]

        resp = await _request_qa(async_client, system, user_msg)
        qa_data = _extract_qa_from_response(resp)

        if not qa_data:
            print(f"  WARNING: malformed response for {entry['id']}, retrying...")
            resp = await _request_qa(async_client, system, user_msg)
            qa_data = _extract_qa_from_response(resp)
            if not qa_data:
                resp_text = "\n".join(b.text for b in resp.content if b.type == "text")
                print(f"  SKIPPING {entry['id']}: malformed response after retry (got {resp_text[:200]})")
                entry["question"] = ""
                entry["answer"] = ""
                entry["qa_reasoning"] = "SKIPPED: malformed response"
                return

        entry["question"] = qa_data["question"]
        entry["answer"] = qa_data["answer"]
        entry["qa_reasoning"] = qa_data.get("reasoning", "")

        if jsonl_file:
            async with write_lock:
                jsonl_file.write(json.dumps(_jsonl_record_from_entry(entry)) + "\n")
                jsonl_file.flush()

        print(f"  [{i + 1}/{len(jobs)}] {entry['id']} ({entry['qa_type']}/{fmt_name}): {entry['question'][:80]}...")

    await run_concurrent(
        generate_qa,
        jobs,
        concurrency=API_CONCURRENCY,
        label="Stage 3",
        progress_interval=5,
    )

    if jsonl_file:
        jsonl_file.close()

    _merge_jsonl_results_into_entries(entries, jsonl_path)
    return entries


def _chunk_batch_requests(requests, max_count=100_000, max_size_mb=250):
    """Split batch requests into chunks respecting count and size limits."""
    chunks = []
    current_chunk = []
    current_size = 0

    for req in requests:
        req_size = len(json.dumps(req).encode("utf-8")) / (1024 * 1024)
        if current_chunk and (len(current_chunk) >= max_count or current_size + req_size > max_size_mb):
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0
        current_chunk.append(req)
        current_size += req_size

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def stage_3_generate_qa_batch(entries, target_model, jsonl_path=None, batch_state_path=None):
    """Generate QA pairs using the Batch API. Synchronous — submits, polls, retrieves."""
    entries = duplicate_entries_for_past_and_future(entries)
    print(f"\n=== STAGE 3 (BATCH): Generate QA pairs ({len(entries)} entries) ===")
    _decode_entries_for_qa(entries, target_model)
    fewshot_pool = load_fewshot_pool()
    completed_ids = _load_completed_ids(jsonl_path)
    jobs = _prepare_qa_jobs(entries, fewshot_pool, completed_ids)
    print(f"  {len(jobs)} entries to process")

    if not jobs:
        _merge_jsonl_results_into_entries(entries, jsonl_path)
        return entries

    entries_by_id = {job["entry"]["id"]: job["entry"] for job in jobs}
    requests = []
    for job in jobs:
        requests.append(
            {
                "custom_id": job["entry"]["id"],
                "params": {
                    "model": QA_MODEL,
                    "max_tokens": QA_MAX_TOKENS,
                    "temperature": 1,
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": QA_THINKING_BUDGET,
                    },
                    "system": job["system"],
                    "messages": [{"role": "user", "content": job["user_msg"]}],
                },
            }
        )

    batch_api_key = os.environ.get("ANTHROPIC_API_KEY_BATCH_API")
    assert batch_api_key, "ANTHROPIC_API_KEY_BATCH_API not set in environment"
    client = anthropic.Anthropic(api_key=batch_api_key)

    chunks = _chunk_batch_requests(requests)
    print(f"  Split into {len(chunks)} batch(es): {[len(c) for c in chunks]}")

    # Resume from saved batch state if it exists
    batch_ids = []
    already_submitted = 0
    if batch_state_path and batch_state_path.exists():
        saved = json.loads(batch_state_path.read_text())
        batch_ids = saved.get("batch_ids", [])
        already_submitted = len(batch_ids)
        print(f"  Resuming: {already_submitted} batch(es) already submitted")

    def _save_batch_state():
        if not batch_state_path:
            return
        batch_state_path.parent.mkdir(parents=True, exist_ok=True)
        batch_state_path.write_text(json.dumps({
            "batch_ids": batch_ids,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "n_requests": len(requests),
            "n_chunks": len(chunks),
        }, indent=2))

    for i, chunk in enumerate(chunks):
        if i < already_submitted:
            continue
        max_retries = 5
        for attempt in range(max_retries):
            try:
                batch = client.messages.batches.create(requests=chunk)
                break
            except anthropic.RateLimitError:
                wait = 60 * (2 ** attempt)
                print(f"  Rate limited submitting batch {i + 1}/{len(chunks)}, "
                      f"retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
        else:
            print(f"  FAILED to submit batch {i + 1}/{len(chunks)} after {max_retries} retries")
            _save_batch_state()
            raise RuntimeError(f"Failed to submit batch {i + 1} after {max_retries} retries")
        batch_ids.append(batch.id)
        _save_batch_state()
        print(f"  Submitted batch {i + 1}/{len(chunks)}: {batch.id} ({len(chunk)} requests)")

    pending = set(batch_ids)
    while pending:
        time.sleep(60)
        for batch_id in list(pending):
            status = client.messages.batches.retrieve(batch_id)
            counts = status.request_counts
            print(
                f"  {batch_id}: {status.processing_status} "
                f"(ok={counts.succeeded} err={counts.errored} "
                f"exp={counts.expired} pending={counts.processing})"
            )
            if status.processing_status == "ended":
                pending.remove(batch_id)
                _retrieve_batch_results(client, batch_id, entries_by_id, jsonl_path)

    n_completed = 0
    if jsonl_path and jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    n_completed += 1
    n_failed = len(jobs) - (n_completed - len(completed_ids))
    print(f"\n  Batch complete: {n_completed} succeeded, {n_failed} failed/skipped")

    _merge_jsonl_results_into_entries(entries, jsonl_path)
    return entries


def _retrieve_batch_results(client, batch_id, entries_by_id, jsonl_path):
    """Retrieve results from a completed batch and append to JSONL."""
    print(f"  Retrieving results for {batch_id}...")
    jsonl_file = open(jsonl_path, "a") if jsonl_path else None
    n_ok = 0
    n_fail = 0

    for result in client.messages.batches.results(batch_id):
        entry_id = result.custom_id
        if entry_id not in entries_by_id:
            continue
        entry = entries_by_id[entry_id]

        if result.result.type != "succeeded":
            n_fail += 1
            entry["question"] = ""
            entry["answer"] = ""
            entry["qa_reasoning"] = f"SKIPPED: batch result {result.result.type}"
            continue

        resp_text = "\n".join(b.text for b in result.result.message.content if b.type == "text")
        qa_data = _parse_qa_text(resp_text)

        if not qa_data:
            n_fail += 1
            entry["question"] = ""
            entry["answer"] = ""
            entry["qa_reasoning"] = f"SKIPPED: malformed response ({resp_text[:100]})"
            continue

        entry["question"] = qa_data["question"]
        entry["answer"] = qa_data["answer"]
        entry["qa_reasoning"] = qa_data.get("reasoning", "")
        n_ok += 1

        if jsonl_file:
            jsonl_file.write(json.dumps(_jsonl_record_from_entry(entry)) + "\n")
            jsonl_file.flush()

    if jsonl_file:
        jsonl_file.close()
    print(f"  Retrieved {batch_id}: {n_ok} ok, {n_fail} failed/malformed")
