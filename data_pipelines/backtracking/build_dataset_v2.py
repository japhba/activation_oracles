"""
Backtracking eval dataset v2 — pipeline script.

Builds a dataset of backtracking points where the uncertainty is NOT obvious
from the prefix text alone. Stages:

1. Identify backtracking points from existing rollouts (Opus)
   - LLM reads full thinking trace, finds points where model backtracks
   - Must identify the *reason* for backtracking
   - Must select points where the reason is NOT obvious from the preceding text

2. Leakage filter (Sonnet)
   - Given only the prefix, Sonnet tries to guess what the model is uncertain about
   - If Sonnet gets it right, the point is rejected (too easy / leaked)

3. Consistency verification (vLLM)
   - Feed prefix to Qwen3-8B, generate 10 continuations
   - Check backtrack rate — keep only points with rate >= 0.4

4. Uncertainty description (Opus)
   - Generate a specific 1-2 sentence description of the uncertainty
   - This becomes the ground truth for the eval

5. Assemble final dataset

Usage:
    source .env && .venv/bin/python data_pipelines/backtracking/build_dataset_v2.py --model Qwen/Qwen3-14B
"""

import argparse
import asyncio
import functools
import json
import os
import re
from pathlib import Path

# Force unbuffered output so we can monitor progress in background
print = functools.partial(print, flush=True)

import anthropic
from anthropic._exceptions import OverloadedError, RateLimitError

from data_pipelines.pipeline_utils import (
    add_model_arg,
    async_api_call,
    extract_tool_input,
    model_dir_name,
    parse_json_response,
    run_concurrent,
)

# --- Config ---
# Models for API stages
IDENTIFICATION_MODEL = "claude-opus-4-6"
FILTER_MODEL = "claude-sonnet-4-5"
DESCRIPTION_MODEL = "claude-opus-4-6"
BACKTRACK_JUDGE_MODEL = "claude-haiku-4-5-20251001"

# Pipeline params
MAX_ROLLOUTS = None  # None = all
MAX_POINTS_PER_ROLLOUT = 2  # cap how many points we extract per rollout
CONSISTENCY_THRESHOLD = 0.4
NUM_CONTINUATIONS = 10
MAX_CONTINUATION_TOKENS = 100
API_CONCURRENCY = 10
HAIKU_CONCURRENCY = 20


# --- Stage 1: Find backtrack markers (regex) + evaluate with Opus ---

# Regex patterns to find explicit backtrack markers at sentence boundaries.
# These match markers preceded by whitespace/newline (i.e. start of a new thought).
BACKTRACK_MARKER_PATTERN = re.compile(
    r'(?:^|\n\n|\.\s+|\?\s+|!\s+)'  # sentence boundary before marker
    r'(Wait[,.]?\s|Actually[,.]?\s|Hmm[,.]?\s|Hold on[,.]?\s'
    r'|No,\s|No\.\s|But wait[,.]?\s|Oh[,!]\s'
    r'|Let me reconsider|Let me re-)',
    re.IGNORECASE,
)

# Minimum prefix length — skip markers too early in the trace
MIN_PREFIX_LENGTH = 300


def find_backtrack_markers(thinking):
    """Find all explicit backtrack markers in a thinking trace.

    Returns list of (char_pos, marker_text) where char_pos is the position
    of the marker itself (not the preceding boundary char).
    """
    candidates = []
    for m in BACKTRACK_MARKER_PATTERN.finditer(thinking):
        # The match includes the boundary char(s) before the marker.
        # Find where the actual marker word starts.
        full_match = m.group(0)
        # Strip leading boundary chars (newlines, period+space, etc.)
        stripped = full_match.lstrip('\n.?! ')
        marker_start = m.end() - len(stripped)

        if marker_start < MIN_PREFIX_LENGTH:
            continue

        # Verify prefix ends at a clean boundary
        prefix = thinking[:marker_start].rstrip()
        if prefix and prefix[-1] not in '.?!\n)':
            # Not a clean sentence boundary — skip
            continue

        candidates.append((marker_start, stripped.strip()))
    return candidates


EVALUATE_SYSTEM = """\
You are evaluating backtracking points in a language model's (Qwen3-8B) reasoning \
trace. For each candidate point, you will see the text before and after the \
backtrack marker. Your job is to determine:

1. Is this GENUINE backtracking? (The model truly reverses course, not just \
continuing with "Wait" as a verbal tic or transition word.)

2. Is the REASON for backtracking NON-OBVIOUS from the preceding text? \
The ~20 words before the marker should NOT hint at what the model is about \
to reconsider. If the preceding text already discusses the concern, it's too easy.

BAD (obvious): "...so the symmetry is broken. Wait, maybe the symmetry isn't broken."
GOOD (non-obvious): "...so we get 84 combinations. Wait, I think I'm double-counting..."
"""

EVALUATE_USER = """\
PROBLEM: {problem}

Below are candidate backtracking points found in this reasoning trace. \
For each one, I show the text before and after the backtrack marker.

{candidates_text}

For each candidate, decide:
- Is this genuine backtracking (not just a verbal tic)?
- Is the reason for backtracking NOT obvious from the preceding ~20 words?
- What specifically is the model uncertain about?

Respond with a JSON array with one entry per candidate:
{{
  "candidate_idx": <int>,
  "is_genuine_backtrack": true/false,
  "reason_is_non_obvious": true/false,
  "uncertainty_reason": "<what the model is uncertain about — be specific>",
  "why_not_obvious": "<why the reason isn't apparent from the prefix, or why it IS obvious>"
}}"""


# --- Stage 2: Leakage filter ---

FILTER_SYSTEM = """\
You are reading the chain-of-thought reasoning of a language model that is \
solving a problem. The reasoning has been cut off at a certain point. \
Based ONLY on the prefix text you see, try to predict what specific concern \
or uncertainty the model is about to express."""

FILTER_USER = """\
PROBLEM: {problem}

REASONING PREFIX (cut off):
{prefix}

The model is about to backtrack or express uncertainty at this exact point. \
Based ONLY on the prefix text above, what specifically do you think the model \
is uncertain or concerned about? Be as specific as possible."""

FILTER_PREDICT_TOOL = {
    "name": "submit_prediction",
    "description": "Submit your prediction of what the model is uncertain about.",
    "input_schema": {
        "type": "object",
        "properties": {
            "prediction": {"type": "string", "description": "Your specific prediction of what the uncertainty is about"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "reasoning": {"type": "string", "description": "Why you think this, based on the prefix"},
        },
        "required": ["prediction", "confidence", "reasoning"],
    },
}

FILTER_JUDGE_SYSTEM = """\
You are comparing two descriptions of what a reasoning model is uncertain about. \
Determine if the PREDICTION (based only on prefix text) successfully identifies \
the same core uncertainty as the GROUND TRUTH (the actual reason)."""

FILTER_JUDGE_USER = """\
PREDICTION (guessed from prefix text only):
{prediction}

GROUND TRUTH (actual uncertainty):
{ground_truth}

Does the prediction capture the same core uncertainty as the ground truth? \
A prediction counts as "matching" if someone reading both would say they're \
describing the same fundamental concern, even if worded differently."""

FILTER_JUDGE_TOOL = {
    "name": "submit_judgment",
    "description": "Submit your judgment of whether the prediction matches the ground truth.",
    "input_schema": {
        "type": "object",
        "properties": {
            "matches": {"type": "boolean", "description": "Whether the prediction captures the same core uncertainty"},
            "reasoning": {"type": "string", "description": "Brief explanation of your judgment"},
        },
        "required": ["matches", "reasoning"],
    },
}


# --- Stage 4: Uncertainty description ---

DESCRIBE_SYSTEM = """\
You describe what a reasoning model is uncertain about at a specific point \
in its chain-of-thought. You will see the reasoning prefix, a SHORT \
continuation (~15-20 words) showing the immediate backtrack, and several \
alternative continuations from the same point.

There may be MULTIPLE distinct sources of uncertainty at this point. \
Different continuations may reveal different concerns. Identify each \
distinct source, estimate what fraction of continuations exhibit it, \
and write a single combined description.

Be SPECIFIC — not "the model is uncertain about the math" but rather \
"the model realizes it may have double-counted configurations where two \
elements are swapped." Ground each source in what the continuations say."""

DESCRIBE_USER = """\
PROBLEM: {problem}

REASONING PREFIX (up to the uncertainty point):
...{prefix}

IMMEDIATE CONTINUATION (~15-20 words after the backtrack marker):
{continuation}

ALTERNATIVE CONTINUATIONS (first ~20 words each):
{alt_continuations}

Write a single uncertainty description that lists each distinct source of \
uncertainty, with an approximate percentage of continuations that show it \
in parentheses. Format example:

"The model is uncertain about X (~70%) and also questions whether Y (~30%)."

Be specific about each source. Keep it to 1-3 sentences total."""

# How many words of continuation to show Opus for the description
DESCRIBE_CONTINUATION_WORDS = 20


# --- Stage 3: Backtrack judging ---

BACKTRACK_JUDGE_SYSTEM = """\
You are judging whether a language model's continuation of its reasoning \
shows backtracking — i.e., the model reverses course, reconsiders a previous \
step, corrects itself, or expresses doubt about what it just did.

This includes: changing approach, questioning a result, redoing a calculation, \
noticing an error, or expressing uncertainty about the preceding reasoning.

This does NOT include: continuing forward confidently, extending the same \
line of reasoning, or simply being verbose."""

BACKTRACK_JUDGE_USER = """\
PROBLEM: {problem}

REASONING PREFIX (what the model wrote before this point):
...{prefix_tail}

CONTINUATION (what the model wrote next):
{continuation}

Does this continuation show the model backtracking, reconsidering, or \
expressing doubt about its preceding reasoning? \
Respond with ONLY a JSON object: {{"backtracks": true/false}}"""


# ==== Stage implementations ====


async def run_stage_1_identify_async(rollouts, max_rollouts=None):
    """Find backtrack markers with regex, then evaluate with Opus."""
    print("\n=== STAGE 1: Find & evaluate backtracking points ===")

    rollout_list = []
    for problem in rollouts:
        for r in problem["rollouts"]:
            rollout_list.append({
                "problem_id": problem["id"],
                "problem": problem["problem"],
                "domain": problem["domain"],
                "rollout_idx": r["rollout_idx"],
                "thinking": r["thinking"],
            })

    if max_rollouts is not None:
        import random
        random.seed(42)
        random.shuffle(rollout_list)
        rollout_list = rollout_list[:max_rollouts]

    # Filter out short rollouts
    rollout_list = [r for r in rollout_list if len(r["thinking"]) >= 500]

    # Step 1: Find all backtrack markers with regex
    rollouts_with_candidates = []
    total_candidates = 0
    for rollout in rollout_list:
        markers = find_backtrack_markers(rollout["thinking"])
        if markers:
            rollouts_with_candidates.append((rollout, markers))
            total_candidates += len(markers)

    print(f"Found {total_candidates} backtrack markers across "
          f"{len(rollouts_with_candidates)}/{len(rollout_list)} rollouts")

    # Step 2: Ask Opus to evaluate which ones have non-obvious reasons
    async_client = anthropic.AsyncAnthropic()

    CONTEXT_BEFORE = 300  # chars of context before marker
    CONTEXT_AFTER = 300   # chars of context after marker
    MAX_CANDIDATES_PER_CALL = 10  # cap to avoid truncation in Opus response

    results = []

    async def evaluate_rollout(i, item):
        rollout, markers = item
        try:
            # Cap markers to avoid response truncation
            markers = markers[:MAX_CANDIDATES_PER_CALL]

            # Build candidate descriptions
            candidates_text = ""
            for j, (char_pos, marker_text) in enumerate(markers):
                trace = rollout["thinking"]
                before = trace[max(0, char_pos - CONTEXT_BEFORE):char_pos]
                after = trace[char_pos:char_pos + CONTEXT_AFTER]
                candidates_text += (
                    f"\n--- Candidate {j} ---\n"
                    f"BEFORE (last {len(before)} chars):\n...{before}\n\n"
                    f"AFTER (first {len(after)} chars):\n{after}\n"
                )

            resp = await async_api_call(
                async_client,
                model=IDENTIFICATION_MODEL,
                max_tokens=4000,
                system=EVALUATE_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": EVALUATE_USER.format(
                        problem=rollout["problem"],
                        candidates_text=candidates_text,
                    ),
                }],
            )
            try:
                evaluations = parse_json_response(resp.content[0].text)
            except json.JSONDecodeError:
                print(f"  WARNING: Failed to parse Opus response for {rollout['problem_id']} "
                      f"rollout {rollout['rollout_idx']}, skipping")
                results.append((rollout, []))
                return

            # Match evaluations back to markers
            good_points = []
            for ev in evaluations:
                if not isinstance(ev, dict):
                    continue
                idx = ev.get("candidate_idx", -1)
                if idx < 0 or idx >= len(markers):
                    continue
                if ev.get("is_genuine_backtrack") and ev.get("reason_is_non_obvious"):
                    good_points.append((markers[idx], ev))

            n_good = len(good_points)
            print(f"  [{i+1}/{len(rollouts_with_candidates)}] {rollout['problem_id']} "
                  f"rollout {rollout['rollout_idx']}: {len(markers)} markers → {n_good} good")
            results.append((rollout, good_points[:MAX_POINTS_PER_ROLLOUT]))
        except Exception as e:
            print(f"  ERROR [{i+1}/{len(rollouts_with_candidates)}] {rollout['problem_id']} "
                  f"rollout {rollout['rollout_idx']}: {type(e).__name__}: {e}")
            results.append((rollout, []))

    await run_concurrent(
        evaluate_rollout, rollouts_with_candidates,
        concurrency=API_CONCURRENCY, label="Stage 1", progress_interval=20,
    )

    all_points = []
    for rollout, good_points in results:
        trace = rollout["thinking"]
        for (char_pos, marker_text), ev in good_points:
            prefix = trace[:char_pos]
            suffix = trace[char_pos:char_pos + 500]
            all_points.append({
                "problem_id": rollout["problem_id"],
                "problem": rollout["problem"],
                "domain": rollout["domain"],
                "rollout_idx": rollout["rollout_idx"],
                "char_pos": char_pos,
                "prefix": prefix,
                "suffix": suffix,
                "uncertainty_reason": ev["uncertainty_reason"],
                "why_not_obvious": ev["why_not_obvious"],
                "marker_text": marker_text,
            })

    print(f"Total points identified: {len(all_points)}")
    return all_points


async def run_stage_2_filter_async(points):
    """Filter out points where the uncertainty is guessable from the prefix."""
    print(f"\n=== STAGE 2: Leakage filter ({len(points)} points) ===")

    async_client = anthropic.AsyncAnthropic()
    progress = {"kept": 0, "rejected": 0, "error": 0}

    async def filter_point(i, pt):
        try:
            # Step 1: Predict uncertainty from prefix alone
            resp = await async_api_call(
                async_client,
                model=FILTER_MODEL,
                max_tokens=500,
                system=FILTER_SYSTEM,
                tools=[FILTER_PREDICT_TOOL],
                tool_choice={"type": "tool", "name": "submit_prediction"},
                messages=[{
                    "role": "user",
                    "content": FILTER_USER.format(
                        problem=pt["problem"],
                        prefix=pt["prefix"][-2000:],
                    ),
                }],
            )
            prediction_data = extract_tool_input(resp)
            prediction = prediction_data["prediction"]
            confidence = prediction_data.get("confidence", "unknown")

            # Step 2: Judge whether the prediction matches
            resp = await async_api_call(
                async_client,
                model=FILTER_MODEL,
                max_tokens=300,
                system=FILTER_JUDGE_SYSTEM,
                tools=[FILTER_JUDGE_TOOL],
                tool_choice={"type": "tool", "name": "submit_judgment"},
                messages=[{
                    "role": "user",
                    "content": FILTER_JUDGE_USER.format(
                        prediction=prediction,
                        ground_truth=pt["uncertainty_reason"],
                    ),
                }],
            )
            judge_result = extract_tool_input(resp)
            matches = judge_result["matches"]

            pt["filter_prediction"] = prediction
            pt["filter_confidence"] = confidence
            pt["filter_matches"] = matches
            pt["filter_judge_reasoning"] = judge_result["reasoning"]
            pt["filter_status"] = "rejected" if matches else "kept"

            if matches:
                progress["rejected"] += 1
            else:
                progress["kept"] += 1
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            pt["filter_status"] = "error"
            pt["filter_error"] = f"parse: {e}"
            progress["error"] += 1
            if progress["error"] <= 3:
                print(f"    PARSE ERROR entry {i}: {e}")
        except (OverloadedError, RateLimitError) as e:
            pt["filter_status"] = "error"
            pt["filter_error"] = f"api: {e}"
            progress["error"] += 1
        except Exception as e:
            pt["filter_status"] = "error"
            pt["filter_error"] = f"unexpected: {type(e).__name__}: {e}"
            progress["error"] += 1

    await run_concurrent(
        filter_point, points,
        concurrency=API_CONCURRENCY, label="Stage 2", progress_interval=10,
    )

    kept = [pt for pt in points if pt["filter_status"] == "kept"]
    print(f"\nSurvived filter: {len(kept)}/{len(points)} "
          f"(rejected={progress['rejected']}, errors={progress['error']})")
    # Return ALL points with status labels
    return points


async def run_stage_3_consistency_async(points, target_model: str):
    """Generate continuations with vLLM, then judge backtracking with LLM.

    Only generates continuations for kept points (not rejected by filter).
    """
    kept = [pt for pt in points if pt.get("filter_status") == "kept"]
    print(f"\n=== STAGE 3: Consistency verification ({len(kept)} kept points) ===")

    import vllm
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(target_model)
    llm = vllm.LLM(
        model=target_model,
        max_model_len=4000,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
    )
    sampling_params = vllm.SamplingParams(
        temperature=1.0,
        top_p=0.95,
        max_tokens=MAX_CONTINUATION_TOKENS,
    )

    # Build all prompts at once for batch inference
    all_prompts = []
    for pt in kept:
        messages = [{"role": "user", "content": pt["problem"]}]
        chat_prefix = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        full_prefix = chat_prefix + "<think>\n" + pt["prefix"]

        for _ in range(NUM_CONTINUATIONS):
            all_prompts.append(full_prefix)

    print(f"Generating {len(all_prompts)} continuations with vLLM...")
    outputs = llm.generate(all_prompts, sampling_params)

    # Collect continuation texts
    for pt_idx, pt in enumerate(kept):
        pt["continuations"] = []
        for cont_idx in range(NUM_CONTINUATIONS):
            flat_idx = pt_idx * NUM_CONTINUATIONS + cont_idx
            text = outputs[flat_idx].outputs[0].text
            pt["continuations"].append({"text": text})

    # Free vLLM before LLM judging
    del llm
    import torch
    torch.cuda.empty_cache()

    # Judge backtracking with LLM (async)
    judge_total = len(kept) * NUM_CONTINUATIONS
    print(f"Judging {judge_total} continuations with {BACKTRACK_JUDGE_MODEL}...")
    async_client = anthropic.AsyncAnthropic()

    # Flatten (pt, cont) pairs for run_concurrent
    judge_items = [(pt, cont) for pt in kept for cont in pt["continuations"]]

    async def judge_one(i, item):
        pt, cont = item
        resp = await async_api_call(
            async_client,
            model=BACKTRACK_JUDGE_MODEL,
            max_tokens=50,
            system=BACKTRACK_JUDGE_SYSTEM,
            messages=[{
                "role": "user",
                "content": BACKTRACK_JUDGE_USER.format(
                    problem=pt["problem"],
                    prefix_tail=pt["prefix"][-500:],
                    continuation=cont["text"],
                ),
            }],
        )
        try:
            result = parse_json_response(resp.content[0].text)
            cont["has_backtrack"] = result["backtracks"]
        except (json.JSONDecodeError, KeyError):
            # Default to False on parse failure — conservative
            cont["has_backtrack"] = False

    await run_concurrent(
        judge_one, judge_items,
        concurrency=HAIKU_CONCURRENCY, label="Stage 3 judging", progress_interval=100,
    )

    # Compute rates
    for pt in kept:
        backtrack_count = sum(1 for c in pt["continuations"] if c["has_backtrack"])
        pt["backtrack_rate"] = backtrack_count / NUM_CONTINUATIONS
        passed = pt["backtrack_rate"] >= CONSISTENCY_THRESHOLD
        pt["consistency_status"] = "passed" if passed else "failed"
        print(f"  {pt['problem_id']} rollout {pt['rollout_idx']}: "
              f"rate={pt['backtrack_rate']:.1f}")

    passed = [pt for pt in kept if pt["consistency_status"] == "passed"]
    print(f"\nConsistent (>= {CONSISTENCY_THRESHOLD}): {len(passed)}/{len(kept)}")
    # Return ALL points (including rejected-by-filter) with status labels
    return points


async def run_stage_4_describe_async(points):
    """Generate uncertainty descriptions using Opus."""
    print(f"\n=== STAGE 4: Generate uncertainty descriptions ({len(points)} points) ===")

    async_client = anthropic.AsyncAnthropic()

    async def describe_point(i, pt):
        # Truncate main continuation to ~DESCRIBE_CONTINUATION_WORDS words
        words = pt["suffix"].split()
        short_continuation = " ".join(words[:DESCRIBE_CONTINUATION_WORDS])

        # Build alt continuations from vLLM samples
        alt_lines = []
        for j, c in enumerate(pt.get("continuations", [])):
            c_text = c["text"] if isinstance(c, dict) else c
            c_words = c_text.split()
            alt_lines.append(f"  [{j+1}]: {' '.join(c_words[:DESCRIBE_CONTINUATION_WORDS])}")
        alt_continuations = "\n".join(alt_lines)

        resp = await async_api_call(
            async_client,
            model=DESCRIPTION_MODEL,
            max_tokens=500,
            system=DESCRIBE_SYSTEM,
            messages=[{
                "role": "user",
                "content": DESCRIBE_USER.format(
                    problem=pt["problem"],
                    prefix=pt["prefix"][-2000:],
                    continuation=short_continuation,
                    alt_continuations=alt_continuations,
                ),
            }],
        )
        pt["uncertainty_description"] = resp.content[0].text.strip()
        print(f"  [{i+1}/{len(points)}] {pt['problem_id']}: {pt['uncertainty_description'][:120]}...")

    await run_concurrent(
        describe_point, points,
        concurrency=API_CONCURRENCY, label="Stage 4", progress_interval=5,
    )
    return points


BACKTRACK_MARKERS = ["wait", "actually", "hmm", "no,", "no.", "hold on", "let me reconsider",
                     "let me re", "but wait", "oh,", "oh!", "correction", "scratch that"]


def validate_entry(pt):
    """Check quality criteria. Returns (ok, reason) tuple."""
    # 1. Original continuation must start with explicit backtrack marker
    suffix_lower = pt["suffix"].strip().lower()
    has_marker = any(suffix_lower.startswith(m) for m in BACKTRACK_MARKERS)
    if not has_marker:
        return False, f"No explicit backtrack marker. Continuation starts with: {pt['suffix'][:50]!r}"

    # 2. Prefix must end at a clean boundary (not mid-word)
    prefix = pt["prefix"].rstrip()
    if prefix and prefix[-1].isalpha() and len(prefix) > 1 and prefix[-2].isalpha():
        return False, f"Prefix ends mid-word: ...{prefix[-30:]!r}"

    # 3. Backtrack rate must meet threshold
    if pt["backtrack_rate"] < CONSISTENCY_THRESHOLD:
        return False, f"Backtrack rate {pt['backtrack_rate']:.1f} < {CONSISTENCY_THRESHOLD}"

    return True, "ok"


def assemble_dataset(points, target_model: str):
    """Assemble final dataset with quality validation."""
    print(f"\n=== STAGE 5: Assemble dataset ({len(points)} candidates) ===")

    entries = []
    for pt in points:
        ok, reason = validate_entry(pt)
        if not ok:
            print(f"  DROPPED {pt['problem_id']}: {reason}")
            continue
        entries.append({
            "problem_id": pt["problem_id"],
            "problem": pt["problem"],
            "domain": pt["domain"],
            "backtrack_rate": pt["backtrack_rate"],
            "prefix": pt["prefix"],
            "original_continuation": pt["suffix"],
            "continuations": [c["text"] for c in pt["continuations"]],
            "uncertainty_description": pt["uncertainty_description"],
            # Pipeline metadata
            "rollout_idx": pt["rollout_idx"],
            "char_pos": pt["char_pos"],
            "filter_prediction": pt.get("filter_prediction", ""),
            "filter_confidence": pt.get("filter_confidence", ""),
            "why_not_obvious": pt.get("why_not_obvious", ""),
        })

    return {
        "metadata": {
            "model": target_model,
            "identification_model": IDENTIFICATION_MODEL,
            "filter_model": FILTER_MODEL,
            "description_model": DESCRIPTION_MODEL,
            "consistency_threshold": CONSISTENCY_THRESHOLD,
            "num_continuations": NUM_CONTINUATIONS,
            "version": 2,
        },
        "entries": entries,
    }


async def main_async(model_name: str):
    model_short = model_dir_name(model_name)
    rollouts_path = Path(f"data_pipelines/backtracking/{model_short}/rollouts.json")
    output_dir = Path(f"data_pipelines/backtracking/{model_short}/v2_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_name}")
    print(f"Rollouts: {rollouts_path}")
    print(f"Output dir: {output_dir}")

    stage1_path = output_dir / "stage1_identified.json"
    if stage1_path.exists():
        print("Loading cached stage 1 results...")
        points = json.loads(stage1_path.read_text())
        print(f"Loaded {len(points)} identified points from cache")
    else:
        print("Loading rollouts...")
        rollouts = json.loads(rollouts_path.read_text())
        print(f"Loaded {len(rollouts)} problems, "
              f"{sum(len(p['rollouts']) for p in rollouts)} rollouts")

        # Stage 1: Identify (async)
        points = await run_stage_1_identify_async(rollouts, max_rollouts=MAX_ROLLOUTS)
        stage1_path.write_text(json.dumps(points, indent=2))

    # Stage 2: Filter — all points kept with filter_status label (async)
    stage2_path = output_dir / "stage2_all.json"
    if stage2_path.exists():
        print("Loading cached stage 2 results...")
        all_with_filter = json.loads(stage2_path.read_text())
        kept = [pt for pt in all_with_filter if pt.get("filter_status") == "kept"]
        print(f"Loaded {len(all_with_filter)} points ({len(kept)} kept) from cache")
    else:
        all_with_filter = await run_stage_2_filter_async(points)
        stage2_path.write_text(json.dumps(all_with_filter, indent=2))

    # Stage 3: Consistency (vLLM + async LLM judge) — only on kept points
    all_with_consistency = await run_stage_3_consistency_async(all_with_filter, model_name)
    (output_dir / "stage3_all.json").write_text(json.dumps(all_with_consistency, indent=2))

    # Stage 4: Describe only points that passed BOTH filter and consistency (async)
    passed_both = [
        pt for pt in all_with_consistency
        if pt.get("filter_status") == "kept" and pt.get("consistency_status") == "passed"
    ]
    print(f"\nPassed both filter + consistency: {len(passed_both)}/{len(all_with_consistency)}")

    if not passed_both:
        print("No points passed both checks. Review stage2_all.json and stage3_all.json.")
        return

    described = await run_stage_4_describe_async(passed_both)
    (output_dir / "stage4_described.json").write_text(json.dumps(described, indent=2))

    # Stage 5: Assemble
    dataset = assemble_dataset(described, model_name)
    output_path = Path(f"data_pipelines/backtracking/{model_short}/backtracking_eval_dataset.json")
    output_path.write_text(json.dumps(dataset, indent=2))

    print(f"\n{'='*60}")
    print(f"DONE! {len(dataset['entries'])} entries saved to {output_path}")
    print(f"\nPipeline artifacts saved to {output_dir}/")
    print(f"  stage1_identified.json — all identified points")
    print(f"  stage2_all.json — all points with filter_status")
    print(f"  stage3_all.json — all points with consistency_status + continuations")
    print(f"  stage4_described.json — passed entries with uncertainty descriptions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build backtracking eval dataset v2")
    add_model_arg(parser)
    args = parser.parse_args()
    asyncio.run(main_async(args.model))
