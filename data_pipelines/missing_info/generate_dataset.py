"""
Generate the missing-information eval dataset.

Tests whether the AO can distinguish identical tokens with different model states.

For each problem pair (complete vs incomplete):
  A: complete prompt + truncated reasoning + short neutral segment (teacher-forced)
  B: incomplete prompt + truncated reasoning (same cut as C, no teacher forcing)
  C: incomplete prompt + truncated reasoning + SAME neutral segment as A (teacher-forced)

The AO probes the last N tokens. For A and C, these are the SAME short neutral
segment (~10 tokens), but the model's internal state differs because it processed
different prompts and reasoning traces before reaching the probe point.

If the AO uses activations, A and C should differ. If it just reads tokens, they match.

Pipeline:
1. Define 10 problem pairs (complete gives dimensions, incomplete gives derived quantity)
2. Generate thinking traces via vLLM (Qwen3-8B, thinking enabled)
3. Use Claude Opus to find good truncation points and short neutral segments
4. Build 3 conditions per problem per segment → entries
5. Save dataset JSON

Usage:
    source .env && .venv/bin/python data_pipelines/missing_info/generate_dataset.py --model Qwen/Qwen3-8B
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

import anthropic
import vllm
from transformers import AutoTokenizer

from data_pipelines.pipeline_utils import add_model_arg, async_api_call, model_dir_name, parse_json_response

PROBLEMS = [
    {
        "id": "pool_border",
        "complete": (
            "A rectangular swimming pool is 25 meters long and 10 meters wide. "
            "A 2-meter-wide concrete border surrounds the pool on all sides. "
            "What is the total area of just the concrete border?"
        ),
        "incomplete": (
            "A rectangular swimming pool has an area of 250 square meters. "
            "A 2-meter-wide concrete border surrounds the pool on all sides. "
            "What is the total area of just the concrete border?"
        ),
        "withheld_keywords": ["25 meters", "25m", "10 meters", "10m", "25 m", "10 m"],
        "missing_info": "The individual length and width of the pool (only total area given)",
    },
    {
        "id": "room_cats",
        "complete": (
            "How many cats can fit in my room? The room is a 10m × 8.15m rectangle. "
            "My fox sits in one corner. Cats cannot be within 5m of the fox. "
            "Each cat occupies 0.5 m². How many cats can fit outside the fox's zone?"
        ),
        "incomplete": (
            "How many cats can fit in my room? The room is a rectangle with total area 81.5 m². "
            "My fox sits in one corner. Cats cannot be within 5m of the fox. "
            "Each cat occupies 0.5 m². How many cats can fit outside the fox's zone?"
        ),
        "withheld_keywords": ["10m", "8.15m", "10 m", "8.15 m", "10 meters", "8.15"],
        "missing_info": "The room dimensions (length and width)",
    },
    {
        "id": "box_diagonal",
        "complete": (
            "A rectangular box has dimensions 3m × 4m × 12m. "
            "What is the length of the longest straight rod that can fit inside the box?"
        ),
        "incomplete": (
            "A rectangular box has a volume of 144 cubic meters. "
            "What is the length of the longest straight rod that can fit inside the box?"
        ),
        "withheld_keywords": ["3m", "4m", "12m", "3 m", "4 m", "12 m", "3 meters", "4 meters", "12 meters"],
        "missing_info": "The individual dimensions of the box (only volume given)",
    },
    {
        "id": "garden_fence",
        "complete": (
            "An L-shaped garden consists of a 10m × 8m rectangle with a 4m × 3m "
            "rectangular section removed from one corner. How many meters of fencing "
            "are needed to enclose the garden completely?"
        ),
        "incomplete": (
            "An L-shaped garden has a total area of 68 square meters. "
            "How many meters of fencing are needed to enclose the garden completely?"
        ),
        "withheld_keywords": ["10m", "8m", "4m", "3m", "10 m", "8 m", "4 m", "3 m"],
        "missing_info": "The specific dimensions of the L-shape",
    },
    {
        "id": "wrap_present",
        "complete": (
            "I need to wrap a rectangular present that is 40cm long, 30cm wide, and 20cm tall. "
            "I need 3cm of overlap on each seam for taping. "
            "What is the minimum area of wrapping paper I need?"
        ),
        "incomplete": (
            "I need to wrap a rectangular present with a total surface area of 5200 cm². "
            "I need 3cm of overlap on each seam for taping. "
            "What is the minimum area of wrapping paper I need?"
        ),
        "withheld_keywords": ["40cm", "30cm", "20cm", "40 cm", "30 cm", "20 cm"],
        "missing_info": "The individual dimensions of the present (only surface area given)",
    },
    {
        "id": "cone_volume",
        "complete": (
            "A cone has a base radius of 6cm and a slant height of 10cm. "
            "What is the volume of the cone? Give your answer in terms of pi."
        ),
        "incomplete": (
            "A cone has a lateral (side) surface area of 60π cm². "
            "What is the volume of the cone? Give your answer in terms of pi."
        ),
        "withheld_keywords": ["6cm", "6 cm", "10cm", "10 cm", "radius of 6", "height of 10"],
        "missing_info": "The base radius and slant height of the cone",
    },
    {
        "id": "triangle_perimeter",
        "complete": (
            "A triangle has sides of length 7cm, 10cm, and 13cm. "
            "What is the perimeter of the triangle, and what is its area using Heron's formula?"
        ),
        "incomplete": (
            "A triangle has an area of approximately 34.98 cm². "
            "What is the perimeter of the triangle, and what is its area using Heron's formula?"
        ),
        "withheld_keywords": ["7cm", "10cm", "13cm", "7 cm", "10 cm", "13 cm"],
        "missing_info": "The individual side lengths of the triangle",
    },
    {
        "id": "field_diagonal",
        "complete": (
            "A rectangular field is 120 meters long and 50 meters wide. "
            "A person walks diagonally from one corner to the opposite corner. "
            "How much shorter is the diagonal path compared to walking along two sides?"
        ),
        "incomplete": (
            "A rectangular field has an area of 6000 square meters. "
            "A person walks diagonally from one corner to the opposite corner. "
            "How much shorter is the diagonal path compared to walking along two sides?"
        ),
        "withheld_keywords": ["120 meters", "50 meters", "120m", "50m", "120 m", "50 m"],
        "missing_info": "The individual length and width of the field",
    },
    {
        "id": "paint_walls",
        "complete": (
            "A room is 5 meters long, 4 meters wide, and 3 meters tall. "
            "Each wall needs two coats of paint. One liter of paint covers 10 square meters. "
            "How many liters of paint are needed for all four walls (not the ceiling or floor)?"
        ),
        "incomplete": (
            "A room has a total wall surface area of 54 square meters. "
            "Each wall needs two coats of paint. One liter of paint covers 10 square meters. "
            "How many liters of paint are needed for all four walls (not the ceiling or floor)?"
        ),
        "withheld_keywords": ["5 meters", "4 meters", "3 meters", "5m", "4m", "3m", "5 m", "4 m", "3 m"],
        "missing_info": "The individual length, width, and height of the room (only total wall area given)",
    },
    {
        "id": "trapezoid_area",
        "complete": (
            "A trapezoid has parallel sides of 8cm and 14cm, and a height of 6cm. "
            "What is the area of the trapezoid, and what is its perimeter if the "
            "non-parallel sides are each 7cm long?"
        ),
        "incomplete": (
            "A trapezoid has a perimeter of 36cm. "
            "What is the area of the trapezoid, and what is its perimeter if the "
            "non-parallel sides are each 7cm long?"
        ),
        "withheld_keywords": ["8cm", "14cm", "6cm", "8 cm", "14 cm", "6 cm", "parallel sides of 8", "parallel sides of 14", "height of 6"],
        "missing_info": "The parallel side lengths and height of the trapezoid",
    },
]


def extract_thinking(response_text: str) -> str:
    """Extract content between <think> tags."""
    match = re.search(r"<think>\s*(.*?)\s*</think>", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no closing tag, take everything after <think>
    match = re.search(r"<think>\s*(.*)", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()


async def find_truncation_points(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    problem_id: str,
    complete_reasoning: str,
    incomplete_reasoning: str,
    withheld_keywords: list[str],
) -> list[dict]:
    """Use Claude Opus 4.6 to find 2 good truncation points with short neutral segments.

    For each truncation point, returns:
    - complete_prefix: the complete reasoning truncated at that point
    - incomplete_prefix: the incomplete reasoning truncated at a sensible point
    - neutral_segment: short (~5-15 word) neutral text for teacher forcing
    """
    async with semaphore:
        response = await async_api_call(
            client,
            model="claude-opus-4-6",
            max_tokens=4000,
            messages=[{"role": "user", "content": f"""I have two reasoning traces from a model solving the same math problem — one with complete information, one with incomplete information. I need to find 2 truncation points in each trace, paired with short neutral text segments for a teacher-forcing experiment.

<complete_reasoning>
{complete_reasoning}
</complete_reasoning>

<incomplete_reasoning>
{incomplete_reasoning}
</incomplete_reasoning>

FORBIDDEN keywords (neutral segments must NOT contain any of these, case-insensitive):
{json.dumps(withheld_keywords)}

For each of the 2 truncation points, I need:
1. A place to cut the COMPLETE reasoning where the text ends with a COMPLETE SENTENCE — ending with a period, question mark, or exclamation mark. The cut should be roughly 30-50% through for the first point and 50-80% for the second. The prefix MUST end with a sentence-ending punctuation mark (., !, ?). NEVER cut mid-sentence.
2. A short neutral text segment (~5-15 words) that is the BEGINNING OF THE NEXT SENTENCE after the cut. This must be COPIED VERBATIM from the complete reasoning. It must not contain forbidden keywords. The segment itself should also end at a word boundary (not mid-word).
3. A place to cut the INCOMPLETE reasoning at a complete sentence boundary (ending with ., !, ?) where the neutral segment could loosely follow.

The neutral segments are like: "So we need to figure out", "Now, let me think about this", "That means we can calculate". They should be short generic reasoning phrases that start a new sentence.

Return a JSON array of 2 objects:
[
  {{
    "complete_prefix": "<complete reasoning text up to the cut point — EXACT verbatim text, MUST end with . or ! or ?>",
    "incomplete_prefix": "<incomplete reasoning text up to its cut point — EXACT verbatim text, MUST end with . or ! or ?>",
    "neutral_segment": "<the short verbatim segment from the complete reasoning that starts the next sentence>"
  }},
  ...
]

CRITICAL: The prefixes must be EXACT substrings of the original reasoning text. Copy them character-for-character. The neutral_segment must also be an exact substring that appears right after the complete_prefix in the complete reasoning. Both prefixes MUST end at complete sentence boundaries (., !, ?).

Return ONLY the JSON array."""}],
        )
        text = response.content[0].text
        try:
            results = parse_json_response(text)
        except json.JSONDecodeError:
            print(f"  ERROR [{problem_id}]: failed to parse JSON response, first 500 chars:")
            print(f"    {text[:500]}")
            return []

        # Validate each result
        validated = []
        for r in results:
            segment = r["neutral_segment"]
            c_prefix = r["complete_prefix"]
            i_prefix = r["incomplete_prefix"]

            # Check neutral segment is keyword-free
            seg_lower = segment.lower()
            has_keyword = any(kw.lower() in seg_lower for kw in withheld_keywords)
            if has_keyword:
                print(f"  WARNING [{problem_id}]: segment has forbidden keyword, skipping: {segment!r}")
                continue

            # Ensure complete_prefix ends at a sentence boundary
            if not re.search(r'[.!?]\s*$', c_prefix):
                # Find the last sentence boundary in the prefix
                match = list(re.finditer(r'[.!?]\s*', c_prefix))
                if match:
                    c_prefix = c_prefix[:match[-1].end()].rstrip()
                    # Re-derive the segment from what follows in the complete reasoning
                    seg_start = complete_reasoning.find(c_prefix) + len(c_prefix)
                    if seg_start > len(c_prefix):
                        # Grab the next ~15 words as the new segment
                        rest = complete_reasoning[seg_start:].lstrip()
                        words = rest.split()[:15]
                        segment = " ".join(words)
                    print(f"  NOTE [{problem_id}]: trimmed complete_prefix to sentence boundary")
                else:
                    print(f"  WARNING [{problem_id}]: no sentence boundary found in complete_prefix, skipping")
                    continue

            # Ensure incomplete_prefix ends at a sentence boundary
            if not re.search(r'[.!?]\s*$', i_prefix):
                match = list(re.finditer(r'[.!?]\s*', i_prefix))
                if match:
                    i_prefix = i_prefix[:match[-1].end()].rstrip()
                    print(f"  NOTE [{problem_id}]: trimmed incomplete_prefix to sentence boundary")
                else:
                    print(f"  WARNING [{problem_id}]: no sentence boundary found in incomplete_prefix, skipping")
                    continue

            # Verify complete_prefix + neutral_segment appears in the complete reasoning
            combined = c_prefix + segment
            if combined not in complete_reasoning:
                # Try to find the segment in the reasoning and reconstruct the prefix
                seg_pos = complete_reasoning.find(segment)
                if seg_pos >= 0:
                    c_prefix = complete_reasoning[:seg_pos]
                    print(f"  NOTE [{problem_id}]: reconstructed complete_prefix from segment position")
                else:
                    print(f"  WARNING [{problem_id}]: neutral segment not found in complete reasoning: {segment!r}")
                    continue

            # Verify incomplete_prefix is a prefix of the incomplete reasoning
            if not incomplete_reasoning.startswith(i_prefix):
                # Find the best sentence boundary near the target
                target_ratio = len(c_prefix) / len(complete_reasoning)
                target_pos = int(len(incomplete_reasoning) * target_ratio)
                boundaries = [m.end() for m in re.finditer(r'[.!?]\s+', incomplete_reasoning)]
                if boundaries:
                    best = min(boundaries, key=lambda x: abs(x - target_pos))
                    i_prefix = incomplete_reasoning[:best]
                    print(f"  NOTE [{problem_id}]: reconstructed incomplete_prefix at sentence boundary")
                else:
                    i_prefix = incomplete_reasoning[:target_pos]

            validated.append({
                "complete_prefix": c_prefix,
                "incomplete_prefix": i_prefix,
                "neutral_segment": segment,
            })

        if not validated:
            print(f"  ERROR [{problem_id}]: no valid truncation points found!")

        return validated[:2]


def _run_truncation_and_build(
    thinking_traces: dict, tokenizer: AutoTokenizer, model_name: str, output_path: Path,
) -> None:
    """Find truncation points via Claude Opus, then build and save the dataset."""
    print("\nFinding truncation points with Claude Opus 4.6...")
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(10)

    async def process_all():
        tasks = [
            find_truncation_points(
                client, semaphore, problem["id"],
                thinking_traces[problem["id"]]["complete"],
                thinking_traces[problem["id"]]["incomplete"],
                problem["withheld_keywords"],
            )
            for problem in PROBLEMS
        ]
        return await asyncio.gather(*tasks)

    all_truncation_points = asyncio.run(process_all())

    # Build dataset entries
    entries = []

    for problem, truncation_points in zip(PROBLEMS, all_truncation_points):
        for seg_idx, tp in enumerate(truncation_points):
            neutral_segment = tp["neutral_segment"]
            a_reasoning = tp["complete_prefix"]
            c_reasoning = tp["incomplete_prefix"]
            neutral_token_count = len(tokenizer.encode(neutral_segment, add_special_tokens=False))

            # Check for keyword leakage
            seg_lower = neutral_segment.lower()
            leaked_keywords = [kw for kw in problem["withheld_keywords"] if kw.lower() in seg_lower]

            suffix = f"_s{seg_idx}" if len(truncation_points) > 1 else ""

            print(f"\n  {problem['id']}{suffix}: neutral='{neutral_segment}' ({neutral_token_count} tokens)")
            if leaked_keywords:
                print(f"    WARNING: LEAKED KEYWORDS: {leaked_keywords}")

            # Condition A: complete prompt + truncated complete reasoning + neutral segment
            entries.append({
                "id": f"{problem['id']}{suffix}_A",
                "condition": "A_complete",
                "problem_id": problem["id"],
                "problem_text": problem["complete"],
                "full_reasoning": a_reasoning,
                "teacher_forced_segment": neutral_segment,
                "ground_truth_missing_info": False,
                "missing_info_description": problem["missing_info"],
                "neutral_segment": neutral_segment,
                "neutral_segment_token_count": neutral_token_count,
                "leaked_keywords": leaked_keywords,
            })

            # Condition B: incomplete prompt + same truncated reasoning as C (no teacher forcing)
            # B and C share the same truncated reasoning so the only difference
            # between B and C is the teacher-forced segment appended to C.
            entries.append({
                "id": f"{problem['id']}{suffix}_B",
                "condition": "B_incomplete",
                "problem_id": problem["id"],
                "problem_text": problem["incomplete"],
                "full_reasoning": c_reasoning,
                "teacher_forced_segment": "",
                "ground_truth_missing_info": True,
                "missing_info_description": problem["missing_info"],
                "neutral_segment": neutral_segment,
                "neutral_segment_token_count": neutral_token_count,
                "leaked_keywords": leaked_keywords,
            })

            # Condition C: incomplete prompt + truncated incomplete reasoning + SAME neutral segment
            entries.append({
                "id": f"{problem['id']}{suffix}_C",
                "condition": "C_forced",
                "problem_id": problem["id"],
                "problem_text": problem["incomplete"],
                "full_reasoning": c_reasoning,
                "teacher_forced_segment": neutral_segment,
                "ground_truth_missing_info": True,
                "missing_info_description": problem["missing_info"],
                "neutral_segment": neutral_segment,
                "neutral_segment_token_count": neutral_token_count,
                "leaked_keywords": leaked_keywords,
            })

    dataset = {
        "metadata": {
            "model": model_name,
            "total_entries": len(entries),
            "num_problems": len(PROBLEMS),
            "conditions": ["A_complete", "B_incomplete", "C_forced"],
            "description": (
                "Missing information experiment. Reasoning is truncated at a natural "
                "point, followed by a short teacher-forced neutral segment (~10 tokens). "
                "A and C have identical teacher-forced segments but different reasoning "
                "contexts (complete vs incomplete). If AO uses activations, A and C "
                "responses should differ."
            ),
        },
        "entries": entries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    # Summary
    leaked = sum(1 for e in entries if e["condition"] == "A_complete" and e["leaked_keywords"])
    n_a = sum(1 for e in entries if e["condition"] == "A_complete")
    print(f"\n{'='*60}")
    print(f"Saved {len(entries)} entries ({n_a} segments × 3 conditions)")
    print(f"Entries with keyword leakage: {leaked}/{n_a}")
    print(f"Output: {output_path}")

    # Print sample entries
    for e in entries[:3]:
        print(f"\n  {e['id']}:")
        print(f"    reasoning: {len(e['full_reasoning'])} chars")
        print(f"    teacher_forced: '{e['teacher_forced_segment']}'")
        print(f"    neutral_tokens: {e['neutral_segment_token_count']}")


def main(model_name: str):
    output_path = Path("data_pipelines/missing_info/missing_info_eval_dataset.json")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm = vllm.LLM(
        model=model_name,
        max_model_len=4096,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
    )

    # Generate thinking traces for both complete and incomplete versions
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=2000)

    all_prompts = []
    prompt_labels = []

    for problem in PROBLEMS:
        for version in ("complete", "incomplete"):
            messages = [{"role": "user", "content": problem[version]}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_prompts.append(prompt)
            prompt_labels.append((problem["id"], version))

    print(f"Generating thinking traces for {len(all_prompts)} prompts...")
    outputs = llm.generate(all_prompts, sampling_params)

    # Parse thinking traces
    thinking_traces: dict[str, dict[str, str]] = {}
    for (problem_id, version), output in zip(prompt_labels, outputs):
        raw = output.outputs[0].text
        thinking = extract_thinking(raw)
        if problem_id not in thinking_traces:
            thinking_traces[problem_id] = {}
        thinking_traces[problem_id][version] = thinking
        token_count = len(tokenizer.encode(thinking, add_special_tokens=False))
        print(f"  {problem_id}/{version}: {token_count} thinking tokens")

    # Free GPU memory
    del llm

    _run_truncation_and_build(thinking_traces, tokenizer, model_name, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate missing-information eval dataset")
    add_model_arg(parser)
    args = parser.parse_args()
    main(args.model)
