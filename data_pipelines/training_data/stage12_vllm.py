"""
Stage 1-2 helpers for the synthetic training data pipeline.

This module owns:
- shared config/constants
- run-scoped artifact paths and manifest handling
- WildChat prompt sampling
- stage 1 full-response generation with vLLM
- stage 2 continuation generation with vLLM
- a unified stage12 runner for both local and sharded execution
"""

import functools
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

# Force unbuffered output
print = functools.partial(print, flush=True)

# --- Shared config ---
OUTPUT_DIR = Path(__file__).parent / "artifacts"
QA_MODEL = "claude-sonnet-4-6"

N_PROMPTS = 50_000
SEED = 42
BENCHMARK_FRACTION = 0.10  # 10% of prompts come from benchmarks (MMLU)

# vLLM params
MAX_RESPONSE_TOKENS = 512
MAX_PROMPT_TOKENS = 2000  # reject prompts longer than this (leaves room for response)
NUM_CONTINUATIONS = 10
MAX_CONTINUATION_TOKENS = 100
THINKING_PROB = 0.3  # 30% of prompts use enable_thinking=True


def get_run_dir(run):
    return OUTPUT_DIR / run


def get_manifest_path(run):
    return get_run_dir(run) / "manifest.json"


def get_prompts_path(run):
    return get_run_dir(run) / "prompts.json"


def _shard_filename(shard, num_shards):
    return f"shard_{shard}_of_{num_shards}.json"


def get_stage1_shard_path(run, shard, num_shards):
    return get_run_dir(run) / "stage1" / _shard_filename(shard, num_shards)


def get_stage12_shard_path(run, shard, num_shards):
    return get_run_dir(run) / "stage12" / _shard_filename(shard, num_shards)


def get_stage12_merged_path(run):
    return get_run_dir(run) / "stage12" / "merged.json"


def load_run_manifest(run):
    manifest_path = get_manifest_path(run)
    assert manifest_path.exists(), (
        f"Run manifest not found: {manifest_path}\n"
        f"Run `prep --run {run} --n-prompts ...` first."
    )
    return json.loads(manifest_path.read_text())


def _write_manifest(run, manifest):
    manifest_path = get_manifest_path(run)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def _tokenizer_max_token_id(tokenizer):
    """Return the largest token ID the tokenizer can safely round-trip."""
    max_token_id = getattr(tokenizer, "max_token_id", None)
    if max_token_id is not None:
        return max_token_id
    return len(tokenizer) - 1


def _find_invalid_token_ids(token_ids, max_token_id):
    """Return any token IDs that fall outside the tokenizer's accepted range."""
    return sorted({tid for tid in token_ids if tid < 0 or tid > max_token_id})


def _drop_entries_with_invalid_response_token_ids(entries, tokenizer, stage_name):
    """Drop entries whose saved response tokens cannot be fed back into the tokenizer."""
    max_token_id = _tokenizer_max_token_id(tokenizer)
    kept_entries = []
    dropped = 0

    for entry in entries:
        invalid_token_ids = _find_invalid_token_ids(
            entry.get("response_token_ids", []),
            max_token_id,
        )
        if invalid_token_ids:
            dropped += 1
            print(
                f"  Skipping {entry['id']}: {stage_name} found invalid response token ids "
                f"{invalid_token_ids[:5]} (max_token_id={max_token_id})"
            )
            continue
        kept_entries.append(entry)

    if dropped:
        print(f"  Dropped {dropped} entries with invalid response token ids during {stage_name}")

    return kept_entries


def stream_wildchat_prompts(n_prompts):
    """Stream prompts from WildChat, filtering for English and basic quality."""

    from datasets import load_dataset

    print(f"Streaming {n_prompts} prompts from WildChat...")
    ds = load_dataset(
        "allenai/WildChat-1M",
        split="train",
        streaming=True,
    )

    entries = []

    for example in ds:
        if example.get("language") != "English":
            continue

        conversation = example["conversation"]
        if not conversation or conversation[0]["role"] != "user":
            continue

        if len(conversation) == 2 and len(conversation[0]["content"].strip()) < 10:
            continue

        total_chars = sum(len(msg["content"]) for msg in conversation)
        if total_chars > MAX_PROMPT_TOKENS * 4:
            continue

        n_user_turns = sum(1 for msg in conversation if msg["role"] == "user")

        if conversation[-1]["role"] == "assistant":
            prompt_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in conversation[:-1]
            ]
        else:
            prompt_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in conversation
            ]

        entry = {
            "id": f"wildchat_{len(entries):05d}",
            "source": "wildchat",
            "prompt_messages": prompt_messages,
            "n_user_turns": n_user_turns,
            "is_multi_turn": n_user_turns > 1,
        }
        entries.append(entry)

        if len(entries) % 1000 == 0:
            print(f"  Collected {len(entries)}/{n_prompts} prompts...")

        if len(entries) >= n_prompts:
            break

    single = sum(1 for e in entries if not e["is_multi_turn"])
    multi = sum(1 for e in entries if e["is_multi_turn"])
    print(f"Collected {len(entries)} prompts ({single} single-turn, {multi} multi-turn)")
    return entries


def sample_mmlu_prompts(n_prompts):
    """Sample MMLU prompts uniformly across subjects."""
    from datasets import load_dataset

    print(f"Sampling {n_prompts} prompts from MMLU...")
    ds = load_dataset("cais/mmlu", "all", split="test", streaming=True)

    # First pass: collect all examples grouped by subject
    by_subject = {}
    for ex in ds:
        subj = ex["subject"]
        if subj not in by_subject:
            by_subject[subj] = []
        by_subject[subj].append(ex)

    subjects = sorted(by_subject.keys())
    print(f"  Found {sum(len(v) for v in by_subject.values())} questions across {len(subjects)} subjects")

    total_available = sum(len(v) for v in by_subject.values())
    if n_prompts > total_available:
        print(f"  Requested {n_prompts} MMLU prompts but only {total_available} available, using all")
        n_prompts = total_available

    # Sample uniformly across subjects
    rng = random.Random(SEED + 100)
    per_subject = max(1, n_prompts // len(subjects))
    sampled = []
    for subj in subjects:
        pool = by_subject[subj]
        k = min(per_subject, len(pool))
        sampled.extend(rng.sample(pool, k))
        if len(sampled) >= n_prompts:
            break

    # If we need more (rounding), sample from remaining
    if len(sampled) < n_prompts:
        all_remaining = [ex for subj in subjects for ex in by_subject[subj] if ex not in sampled]
        rng.shuffle(all_remaining)
        sampled.extend(all_remaining[: n_prompts - len(sampled)])

    # Shuffle and trim
    rng.shuffle(sampled)
    sampled = sampled[:n_prompts]

    labels = ["A", "B", "C", "D"]
    entries = []
    for i, ex in enumerate(sampled):
        choice_lines = "\n".join(f"{labels[j]}. {ex['choices'][j]}" for j in range(len(ex["choices"])))
        content = f"{ex['question']}\n\n{choice_lines}"

        entry = {
            "id": f"mmlu_{i:05d}",
            "source": "mmlu",
            "mmlu_subject": ex["subject"],
            "mmlu_answer": labels[ex["answer"]],
            "prompt_messages": [{"role": "user", "content": content}],
            "n_user_turns": 1,
            "is_multi_turn": False,
        }
        entries.append(entry)

    print(f"  Sampled {len(entries)} MMLU prompts from {len(set(e['mmlu_subject'] for e in entries))} subjects")
    return entries


def assign_entry_metadata(entries):
    """Assign enable_thinking deterministically in place.

    Every entry will produce both a past and future training example.
    qa_type duplication happens at stage 3 time, not here.
    """
    rng = random.Random(SEED)
    rng.shuffle(entries)

    thinking_rng = random.Random(SEED + 1)
    for entry in entries:
        entry["enable_thinking"] = thinking_rng.random() < THINKING_PROB
    thinking_count = sum(1 for e in entries if e["enable_thinking"])
    print(f"Assigned enable_thinking: {thinking_count}/{len(entries)} entries")

    return entries


def prep_prompts(run, n_prompts, target_model):
    """Create a run-scoped prompt set and manifest."""
    run_dir = get_run_dir(run)
    manifest_path = get_manifest_path(run)
    prompts_path = get_prompts_path(run)

    if manifest_path.exists() and prompts_path.exists():
        raise AssertionError(
            f"Run already exists at {run_dir}.\n"
            f"Use a new `--run` name or delete the existing run directory."
        )

    run_dir.mkdir(parents=True, exist_ok=True)

    n_benchmark = max(1, int(n_prompts * BENCHMARK_FRACTION))
    n_wildchat = n_prompts - n_benchmark

    wildchat_entries = stream_wildchat_prompts(n_wildchat)
    mmlu_entries = sample_mmlu_prompts(n_benchmark)
    selected = wildchat_entries + mmlu_entries

    assign_entry_metadata(selected)
    prompts_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False))

    n_wc = sum(1 for e in selected if e["source"] == "wildchat")
    n_mm = sum(1 for e in selected if e["source"] == "mmlu")
    print(f"Mixed prompts: {n_wc} wildchat + {n_mm} mmlu = {len(selected)} total")

    manifest = {
        "run": run,
        "n_prompts": len(selected),
        "benchmark_fraction": BENCHMARK_FRACTION,
        "seed": SEED,
        "target_model": target_model,
        "qa_model": QA_MODEL,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_manifest(run, manifest)

    print(f"Saved prompts to {prompts_path}")
    print(f"Saved manifest to {manifest_path}")
    return manifest


def load_vllm(
    target_model,
    *,
    max_model_len=4096,
    enforce_eager=False,
    gpu_memory_utilization=0.9,
):
    """Load the target tokenizer and vLLM engine."""
    import vllm
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(target_model)
    llm = vllm.LLM(
        model=target_model,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    return llm, tokenizer


def stage_1_generate_responses(entries, target_model, enforce_eager=False):
    """Generate full responses for all prompts using vLLM."""
    print(f"\n=== STAGE 1: Generate responses ({len(entries)} prompts) ===")

    import vllm
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(target_model)
    llm = vllm.LLM(
        model=target_model,
        max_model_len=4096,
        enforce_eager=enforce_eager,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )

    formatted = []
    valid_entries = []
    thinking_count = 0
    for entry in entries:
        prompt_token_ids = tokenizer.apply_chat_template(
            entry["prompt_messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=entry["enable_thinking"],
        )
        if len(prompt_token_ids) > MAX_PROMPT_TOKENS:
            print(f"  Skipping {entry['id']}: prompt too long ({len(prompt_token_ids)} tokens)")
            continue
        entry["prompt_token_ids"] = prompt_token_ids
        formatted.append({"prompt_token_ids": prompt_token_ids})
        valid_entries.append(entry)
        if entry["enable_thinking"]:
            thinking_count += 1

    entries = valid_entries
    print(f"  {len(entries)} entries after filtering for length ({thinking_count} with thinking)")

    sampling_params = vllm.SamplingParams(
        temperature=1.0,
        max_tokens=MAX_RESPONSE_TOKENS,
    )

    print(f"Generating {len(formatted)} responses...")
    outputs = llm.generate(formatted, sampling_params)

    for entry, output in zip(entries, outputs):
        entry["response_text"] = output.outputs[0].text
        entry["response_token_ids"] = list(output.outputs[0].token_ids)

    entries = _drop_entries_with_invalid_response_token_ids(
        entries,
        tokenizer,
        "stage 1 post-generation",
    )

    print(f"Generated {len(entries)} responses")
    if entries:
        print(
            f"Response lengths: min={min(len(e['response_text']) for e in entries)}, "
            f"max={max(len(e['response_text']) for e in entries)}, "
            f"mean={sum(len(e['response_text']) for e in entries) / len(entries):.0f}"
        )

    return entries, llm, tokenizer


def text_pos_to_token_idx(token_ids, text_pos, tokenizer):
    """Map a character position in decoded text to a token index."""
    for idx in range(1, len(token_ids) + 1):
        decoded = tokenizer.decode(token_ids[:idx], skip_special_tokens=False)
        if len(decoded) >= text_pos:
            return idx
    return len(token_ids)


def stage_2_generate_continuations(entries, llm, tokenizer):
    """Generate rollout continuations for all entries."""
    import vllm

    entries = _drop_entries_with_invalid_response_token_ids(
        entries,
        tokenizer,
        "stage 2 preflight",
    )
    print(f"\n=== STAGE 2: Generate continuations ({len(entries)} entries) ===")

    if not entries:
        print("No entries, skipping.")
        return entries

    for entry in entries:
        text = entry["response_text"]
        boundaries = [m.end() for m in re.finditer(r"(?:[.!?][\s\n]|\n)", text)]
        min_pos = int(len(text) * 0.1)
        max_pos = int(len(text) * 0.9)
        boundaries = [b for b in boundaries if min_pos <= b <= max_pos]

        if not boundaries:
            trunc_pos = len(text) // 2
        else:
            trunc_pos = random.choice(boundaries)

        entry["truncation_pos"] = trunc_pos
        entry["prefix"] = text[:trunc_pos]
        entry["original_continuation"] = text[trunc_pos:]

    max_token_id = _tokenizer_max_token_id(tokenizer)
    all_prompts = []
    valid_entries = []
    skipped_ids = set()
    for entry in entries:
        entry["truncation_token_idx"] = text_pos_to_token_idx(
            entry["response_token_ids"],
            entry["truncation_pos"],
            tokenizer,
        )
        prefix_token_ids = (
            entry["prompt_token_ids"]
            + entry["response_token_ids"][: entry["truncation_token_idx"]]
        )
        invalid_prefix_token_ids = _find_invalid_token_ids(prefix_token_ids, max_token_id)
        if invalid_prefix_token_ids:
            skipped_ids.add(entry["id"])
            print(
                f"  Skipping {entry['id']}: stage 2 prefix contains invalid token ids "
                f"{invalid_prefix_token_ids[:5]} (max_token_id={max_token_id})"
            )
            continue
        valid_entries.append(entry)
        for _ in range(NUM_CONTINUATIONS):
            all_prompts.append({"prompt_token_ids": prefix_token_ids})

    if skipped_ids:
        print(
            f"  Dropped {len(skipped_ids)} entries with invalid stage 2 prefixes"
        )
        entries = [e for e in entries if e["id"] not in skipped_ids]

    if not valid_entries:
        print("No valid entries remain after stage 2 validation, skipping.")
        return entries

    sampling_params = vllm.SamplingParams(
        temperature=1.0,
        max_tokens=MAX_CONTINUATION_TOKENS,
    )

    print(f"Generating {len(all_prompts)} continuations...")
    outputs = llm.generate(all_prompts, sampling_params)

    for i, entry in enumerate(valid_entries):
        entry["continuations"] = []
        for j in range(NUM_CONTINUATIONS):
            flat_idx = i * NUM_CONTINUATIONS + j
            entry["continuations"].append(outputs[flat_idx].outputs[0].text)

    print("Done generating continuations.")
    return entries


def run_stage12(run, shard=0, num_shards=1, stop_after_stage=None):
    """Run stages 1-2 for one shard. Local mode is shard 0 of 1."""
    assert num_shards >= 1, "--num-shards must be at least 1"
    assert 0 <= shard < num_shards, f"--shard must be in [0, {num_shards})"

    manifest = load_run_manifest(run)
    prompts_path = get_prompts_path(run)
    assert prompts_path.exists(), (
        f"Prompts file not found: {prompts_path}\n"
        f"Run `prep --run {run} --n-prompts {manifest['n_prompts']}` first."
    )

    stage1_path = get_stage1_shard_path(run, shard, num_shards)
    stage12_path = get_stage12_shard_path(run, shard, num_shards)
    merged_path = get_stage12_merged_path(run)
    stage1_path.parent.mkdir(parents=True, exist_ok=True)
    stage12_path.parent.mkdir(parents=True, exist_ok=True)

    all_entries = json.loads(prompts_path.read_text())
    selected = all_entries[shard::num_shards]
    print(
        f"Run {run}: shard {shard}/{num_shards} has {len(selected)} prompts "
        f"(of {len(all_entries)} total)"
    )

    # Skip torch.compile for small runs — much faster startup
    enforce_eager = len(selected) < 1000
    if enforce_eager:
        print(f"  Small run ({len(selected)} entries < 1000): using enforce_eager=True")

    target_model = manifest["target_model"]
    print(f"Target model: {target_model}")

    llm = None
    tokenizer = None

    if stage1_path.exists():
        print("Loading cached stage 1 results...")
        selected = json.loads(stage1_path.read_text())
    else:
        selected, llm, tokenizer = stage_1_generate_responses(
            selected, target_model, enforce_eager=enforce_eager
        )
        stage1_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False))
        print(f"Saved stage 1 shard to {stage1_path}")

    if stop_after_stage == 1:
        print(f"Stopping after stage 1. Saved to {stage1_path}")
        return selected

    if stage12_path.exists():
        print("Loading cached stage 2 results...")
        selected = json.loads(stage12_path.read_text())
    else:
        if llm is None or tokenizer is None:
            llm, tokenizer = load_vllm(
                target_model,
                max_model_len=4096,
                enforce_eager=enforce_eager,
                gpu_memory_utilization=0.9,
            )
        selected = stage_2_generate_continuations(selected, llm, tokenizer)
        stage12_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False))
        print(f"Saved stage 2 shard to {stage12_path}")

    if num_shards == 1:
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        merged_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False))
        print(f"Saved merged stage 2 output to {merged_path}")

    return selected


def merge_stage12(run, num_shards=1):
    """Merge stage12 shard outputs into a canonical merged file."""
    assert num_shards >= 1, "--num-shards must be at least 1"
    load_run_manifest(run)

    all_entries = []
    for shard in range(num_shards):
        path = get_stage12_shard_path(run, shard, num_shards)
        assert path.exists(), f"Missing shard file: {path}"
        entries = json.loads(path.read_text())
        print(f"Shard {shard}: {len(entries)} entries")
        all_entries.extend(entries)

    all_entries.sort(key=lambda e: e["id"])

    output_path = get_stage12_merged_path(run)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_entries, indent=2, ensure_ascii=False))

    print(f"\nMerged {len(all_entries)} entries (each will produce 1 past + 1 future in stage 3)")
    print(f"Saved to {output_path}")
    return all_entries
