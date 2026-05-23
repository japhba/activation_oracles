"""
Generate new answers for existing latentQA questions using Sonnet 4.6.

Keeps the old questions from qa.json, but generates new, more specific answers
based on the target model's responses from generate_responses.py.

Usage:
    # Small test run (async API):
    source .env
    .venv/bin/python data_pipelines/latentqa_datasets/generate_qa.py --model Qwen/Qwen3-8B --n 20

    # Full run (batch API):
    source .env
    .venv/bin/python data_pipelines/latentqa_datasets/generate_qa.py --model Qwen/Qwen3-8B --all --batch

Output: data_pipelines/latentqa_datasets/artifacts/<model_short_name>/qa_{n}.json
        data_pipelines/latentqa_datasets/artifacts/<model_short_name>/qa_{n}_review.txt
"""

import argparse
import asyncio
import functools
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

# Force unbuffered output
print = functools.partial(print, flush=True)

import anthropic
from data_pipelines.pipeline_utils import async_api_call, model_dir_name, add_model_arg, parse_json_response, run_concurrent

QA_MODEL = "claude-sonnet-4-6"
QA_MAX_TOKENS = 4096
QA_THINKING_BUDGET = 2048
API_CONCURRENCY = 10
SEED = 42
DATASET_DIR = Path(__file__).parent / "train"

QA_SYSTEM_PROMPT = """\
You are creating training data for an activation oracle — a model that reads \
the internal activations of a language model (Qwen3-8B) and answers questions \
about what the model is processing.

You will see a conversation where Qwen3-8B was given a hidden system prompt \
(a persona to adopt or a behavioral goal to follow) and then responded to a \
user message. You will also see a question about the persona or goal.

Your job: answer the question. Be specific and concrete rather than vague. \
Make sure the answer is correct — grounded in what the system prompt says.

IMPORTANT: Your answer should be about the persona or goal ITSELF — what the \
model was instructed to be or do — NOT about specific details of how the model \
responded. The activation oracle may only see the user message, not the \
model's response, so the answer must make sense without having read the \
response.

For "how would the assistant respond" questions, you may describe what the \
model does at a HIGH LEVEL (e.g. "it would share personal stories of \
endurance" or "it would encourage the user to push through challenges"). \
But do NOT fabricate specific details — only mention themes or approaches \
that are clearly present in either the system prompt or the response. \
No direct quotes, no invented specifics.

GOOD ANSWERS describe the persona/goal at a high level:
- "The model is adopting the persona of a sand artist from the Sahara who \
creates transient art reclaimed by desert winds."
- "The model is trying to include a spoonerism in its response."
- "It would share personal stories of endurance and cultural discovery from \
walking the Great Wall, encouraging the user to embrace challenging journeys."

BAD ANSWERS fabricate or over-specify:
- "The model emphasizes that immersing oneself in unfamiliar cultures builds \
empathy, resilience, and self-understanding." (invents specific themes not \
clearly in the response — the response talks about pushing through limits \
and personal transformation, not empathy/resilience/self-understanding)
- "The model uses blockchain-based self-sovereign identity (SSI) solutions \
combined with cryptographic techniques." (too specific — details only \
knowable from reading the response)
- "The model uses the metaphor of 'winds of change' in its third paragraph." \
(directly quotes the response)

BAD ANSWERS are vague:
- "The persona is passionate and enthusiastic." (could describe anything)
- "The model is following specific behavioral instructions." (says nothing)

RULES:
- Answers must be 1-3 sentences, direct, and specific.
- Ground your answer in what the SYSTEM PROMPT says. You may also reference \
the model's response, but only at a high level — describe the general \
approach or themes, not specific content.
- Do NOT invent details. If you describe what the model would do, make sure \
those themes are actually present in the system prompt or response.

Respond with a JSON object:
```json
{{"answer": "..."}}
```
Output ONLY the JSON object, no other text."""

QA_USER_TEMPLATE = """\
SYSTEM PROMPT (hidden from user, given to the model):
{system_prompt}

USER MESSAGE:
{user_message}

MODEL RESPONSE:
{response}

QUESTION: {question}"""


def load_responses(artifacts_dir):
    """Load responses from generate_responses.py output."""
    path = artifacts_dir / "responses.json"
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded {len(data['entries'])} entries from {path}")
    return data["entries"]


def load_old_qa():
    """Load old QA pairs from qa.json."""
    with open(DATASET_DIR / "qa.json") as f:
        old_qa = json.load(f)
    print(f"Loaded old QA for {len(old_qa)} labels")
    return old_qa


def sample_entries(entries, old_qa, n, seed):
    """Sample n entries that have old QA pairs, balanced across persona/goal."""
    with_qa = [e for e in entries if e["label"] in old_qa]

    rng = random.Random(seed)
    persona = [e for e in with_qa if e["label"].startswith("persona")]
    goal = [e for e in with_qa if e["label"].startswith("goal")]
    rng.shuffle(persona)
    rng.shuffle(goal)
    half = n // 2
    sampled = persona[:half] + goal[:half]
    rng.shuffle(sampled)
    return sampled


def build_jobs(entries, old_qa):
    """Build one job per (entry, question) pair."""
    jobs = []
    for entry in entries:
        label = entry["label"]
        for q_idx, (question, old_answer) in enumerate(old_qa[label]):
            jobs.append({
                "label": label,
                "q_idx": q_idx,
                "question": question,
                "old_answer": old_answer,
                "entry": entry,
            })
    return jobs


def _build_user_msg(job):
    """Build the user message for a job."""
    entry = job["entry"]
    return QA_USER_TEMPLATE.format(
        system_prompt=entry["system_prompt"],
        user_message=entry["user_message"],
        response=entry["response"],
        question=job["question"],
    )


def _job_id(job):
    """Unique ID for a job. Must match ^[a-zA-Z0-9_-]{1,64}$ for batch API."""
    import hashlib
    label = job["label"]
    q_idx = job["q_idx"]
    # Hash the label to guarantee safe chars and length
    label_hash = hashlib.md5(label.encode()).hexdigest()[:20]
    return f"{label_hash}_q{q_idx}"


def _parse_answer(resp_text):
    """Extract answer from response text."""
    try:
        data = parse_json_response(resp_text)
        return data.get("answer", "")
    except (json.JSONDecodeError, ValueError):
        return ""


# ── Async API path (for small test runs) ──


async def generate_answers_async(jobs):
    """Generate new answers using Sonnet 4.6 async API."""
    client = anthropic.AsyncAnthropic()
    results = []

    async def generate_one(i, job):
        user_msg = _build_user_msg(job)
        resp = await async_api_call(
            client,
            model=QA_MODEL,
            max_tokens=QA_MAX_TOKENS,
            temperature=1,
            thinking={"type": "enabled", "budget_tokens": QA_THINKING_BUDGET},
            system=QA_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        resp_text = "\n".join(b.text for b in resp.content if b.type == "text")
        new_answer = _parse_answer(resp_text)
        if not new_answer:
            print(f"  WARNING: malformed response for {_job_id(job)}: {resp_text[:100]}")

        result = _make_result(job, new_answer)
        results.append(result)
        print(f"  [{i + 1}/{len(jobs)}] {_job_id(job)}")
        print(f"      Q: {job['question'][:100]}")
        print(f"      New: {new_answer[:100]}")

    await run_concurrent(
        generate_one,
        jobs,
        concurrency=API_CONCURRENCY,
        label="Answer generation",
        progress_interval=5,
    )

    return results


# ── Batch API path (for full runs) ──


def generate_answers_batch(jobs, jsonl_path=None, batch_state_path=None):
    """Generate new answers using the Batch API. Submits, polls, retrieves."""
    print(f"\n=== BATCH: Generate answers ({len(jobs)} jobs) ===")

    # Load completed IDs for resume
    completed_ids = set()
    if jsonl_path and jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    completed_ids.add(json.loads(line)["id"])
        print(f"  Resuming: {len(completed_ids)} already completed")

    # Filter to incomplete jobs
    pending_jobs = [j for j in jobs if _job_id(j) not in completed_ids]
    print(f"  {len(pending_jobs)} jobs to process")

    if not pending_jobs:
        return _load_results_from_jsonl(jsonl_path, jobs)

    # Build batch requests
    requests = []
    for job in pending_jobs:
        requests.append({
            "custom_id": _job_id(job),
            "params": {
                "model": QA_MODEL,
                "max_tokens": QA_MAX_TOKENS,
                "temperature": 1,
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": QA_THINKING_BUDGET,
                },
                "system": QA_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": _build_user_msg(job)}],
            },
        })

    batch_api_key = os.environ.get("ANTHROPIC_API_KEY_BATCH_API")
    assert batch_api_key, "ANTHROPIC_API_KEY_BATCH_API not set in environment"
    client = anthropic.Anthropic(api_key=batch_api_key)

    # Chunk requests (batch API limits)
    chunks = _chunk_batch_requests(requests)
    print(f"  Split into {len(chunks)} batch(es): {[len(c) for c in chunks]}")

    # Resume from saved batch state
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

    # Submit batches
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
            _save_batch_state()
            raise RuntimeError(f"Failed to submit batch {i + 1} after {max_retries} retries")
        batch_ids.append(batch.id)
        _save_batch_state()
        print(f"  Submitted batch {i + 1}/{len(chunks)}: {batch.id} ({len(chunk)} requests)")

    # Poll until all batches complete
    jobs_by_id = {_job_id(j): j for j in pending_jobs}
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
                _retrieve_batch_results(client, batch_id, jobs_by_id, jsonl_path)

    return _load_results_from_jsonl(jsonl_path, jobs)


def _chunk_batch_requests(requests, max_count=100_000, max_size_mb=250):
    """Split batch requests into chunks respecting API limits."""
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


def _retrieve_batch_results(client, batch_id, jobs_by_id, jsonl_path):
    """Retrieve results from a completed batch and append to JSONL."""
    print(f"  Retrieving results for {batch_id}...")
    jsonl_file = open(jsonl_path, "a") if jsonl_path else None
    n_ok = 0
    n_fail = 0

    for result in client.messages.batches.results(batch_id):
        job_id = result.custom_id
        if job_id not in jobs_by_id:
            continue
        job = jobs_by_id[job_id]

        if result.result.type != "succeeded":
            n_fail += 1
            continue

        resp_text = "\n".join(b.text for b in result.result.message.content if b.type == "text")
        new_answer = _parse_answer(resp_text)

        if not new_answer:
            n_fail += 1
            print(f"  WARNING: malformed response for {job_id}: {resp_text[:100]}")
            continue

        n_ok += 1
        record = {
            "id": job_id,
            "label": job["label"],
            "q_idx": job["q_idx"],
            "question": job["question"],
            "old_answer": job["old_answer"],
            "new_answer": new_answer,
        }

        if jsonl_file:
            jsonl_file.write(json.dumps(record) + "\n")
            jsonl_file.flush()

    if jsonl_file:
        jsonl_file.close()
    print(f"  Retrieved {batch_id}: {n_ok} ok, {n_fail} failed/malformed")


def _load_results_from_jsonl(jsonl_path, jobs):
    """Load results from JSONL and merge with job metadata."""
    answers_by_id = {}
    if jsonl_path and jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    answers_by_id[record["id"]] = record["new_answer"]

    results = []
    for job in jobs:
        jid = _job_id(job)
        new_answer = answers_by_id.get(jid, "")
        results.append(_make_result(job, new_answer))
    return results


def _make_result(job, new_answer):
    """Create a result dict from a job and answer."""
    entry = job["entry"]
    return {
        "label": job["label"],
        "q_idx": job["q_idx"],
        "question": job["question"],
        "old_answer": job["old_answer"],
        "new_answer": new_answer,
        "system_prompt": entry["system_prompt"],
        "user_message": entry["user_message"],
        "response": entry["response"],
    }


def save_results(results, tag, artifacts_dir):
    """Save results as JSON and a human-readable review file."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    output_path = artifacts_dir / f"qa_{tag}.json"
    output = {
        "metadata": {
            "qa_model": QA_MODEL,
            "total_entries": len(results),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "entries": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(results)} results to {output_path}")

    # Review file
    review_path = artifacts_dir / f"qa_{tag}_review.txt"
    by_label = {}
    for r in results:
        by_label.setdefault(r["label"], []).append(r)

    lines = []
    for i, (label, group) in enumerate(by_label.items()):
        entry = group[0]
        lines.append("=" * 70)
        lines.append(f"[{i + 1}/{len(by_label)}] {label}")
        lines.append("=" * 70)
        lines.append(f"\nSYSTEM PROMPT:\n{entry['system_prompt']}")
        lines.append(f"\nUSER:\n{entry['user_message']}")
        lines.append(f"\nMODEL RESPONSE:\n{entry['response']}")
        for r in group:
            lines.append(f"\n  Q{r['q_idx'] + 1}: {r['question']}")
            lines.append(f"  OLD A: {r['old_answer']}")
            lines.append(f"  NEW A: {r['new_answer']}")
        lines.append("")

    with open(review_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Review file: {review_path}")


def main():
    parser = argparse.ArgumentParser()
    add_model_arg(parser)
    parser.add_argument("--n", type=int, default=20, help="Number of entries to sample")
    parser.add_argument("--all", action="store_true", help="Process all entries (ignore --n)")
    parser.add_argument("--batch", action="store_true", help="Use batch API instead of async")
    args = parser.parse_args()

    artifacts_dir = Path(__file__).parent / "artifacts" / model_dir_name(args.model)

    entries = load_responses(artifacts_dir)
    old_qa = load_old_qa()

    if args.all:
        # All entries that have old QA
        selected = [e for e in entries if e["label"] in old_qa]
        tag = "all"
        print(f"Processing all {len(selected)} entries ({len(selected) * 2} questions)")
    else:
        selected = sample_entries(entries, old_qa, args.n, SEED)
        tag = str(args.n)
        print(f"Sampled {len(selected)} entries ({len(selected) * 2} questions)")

    jobs = build_jobs(selected, old_qa)

    if args.batch:
        jsonl_path = artifacts_dir / f"qa_{tag}.jsonl"
        batch_state_path = artifacts_dir / f"qa_{tag}_batch_state.json"
        results = generate_answers_batch(jobs, jsonl_path, batch_state_path)
    else:
        results = asyncio.run(generate_answers_async(jobs))

    save_results(results, tag, artifacts_dir)


if __name__ == "__main__":
    main()
