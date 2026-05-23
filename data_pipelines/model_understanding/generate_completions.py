"""
Generate completions from a target model on chat prompts for manual review.

Streams lmsys-chat-1m, collects single-turn English prompts, generates
multiple temperature-1 completions per prompt with thinking disabled, and
saves full text for human review and downstream Claude batch API usage.

Usage:
    .venv/bin/python data_pipelines/model_understanding/generate_completions.py \
        --model Qwen/Qwen3-14B --n-prompts 1000 --n-completions 10
"""

import argparse
import gc
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from data_pipelines.pipeline_utils import model_dir_name

CHAT_DATASET = "allenai/WildChat-1M"
MAX_PROMPT_TOKENS = 3000
MAX_RESPONSE_TOKENS = 1000
SEED = 42


@dataclass
class PromptCompletions:
    id: str
    user_message: str
    completions: list[str]
    finish_reasons: list[str]
    prompt_tokens: int
    response_token_counts: list[int]
    messages: list[dict] | None = None
    n_turns: int = 1
    total_chars: int = 0
    conversation_hash: str = ""


@dataclass
class CompletionDataset:
    metadata: dict
    prompts: list[PromptCompletions] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "prompts": [asdict(p) for p in self.prompts],
        }


def stream_single_turn_prompts():
    """Yield single-turn English user messages from lmsys-chat-1m."""
    ds = iter(load_dataset(CHAT_DATASET, split="train", streaming=True))
    for row in ds:
        if row.get("language") != "English":
            continue
        # lmsys-chat-1m has a redacted field; WildChat doesn't need this
        if row.get("redacted", False):
            continue

        conversation = row["conversation"]
        if not conversation or conversation[0]["role"] != "user":
            continue

        user_msg = conversation[0]["content"].strip()
        if len(user_msg) < 10:
            continue

        yield user_msg


def generate_completions(model_name: str, n_completions: int, n_prompts: int | None = None, offset: int = 0, output_path: str | None = None, input_shard: str | None = None, max_tokens: int = MAX_RESPONSE_TOKENS, quantization: str | None = None):
    random.seed(SEED)
    torch.manual_seed(SEED)

    if input_shard:
        # Load pre-built shard from prepare_shards.py
        shard_data = json.loads(Path(input_shard).read_text())
        candidates = shard_data["prompts"]  # list of {id, user_message, prompt_ids}
        n_prompts = len(candidates)
        print(f"Loaded {n_prompts} prompts from shard: {input_shard}")
    else:
        assert n_prompts is not None, "--n-prompts is required when not using --input-shard"

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Check if tokenizer supports enable_thinking
        supports_thinking = (
            hasattr(tokenizer, "chat_template")
            and tokenizer.chat_template is not None
            and "enable_thinking" in tokenizer.chat_template
        )

        # Collect prompts and tokenize
        print(f"Collecting {n_prompts} prompts from {CHAT_DATASET} (offset={offset})...")
        if offset > 0:
            print(f"Skipping first {offset} eligible prompts...")
        candidates = []
        n_skipped = 0
        for user_msg in tqdm(stream_single_turn_prompts(), desc="Collecting prompts", total=n_prompts):
            messages = [{"role": "user", "content": user_msg}]

            template_kwargs = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_tensors": None,
                "padding": False,
            }
            if supports_thinking:
                template_kwargs["enable_thinking"] = False

            prompt_ids = tokenizer.apply_chat_template(messages, **template_kwargs)

            if len(prompt_ids) > MAX_PROMPT_TOKENS:
                continue
            if len(prompt_ids) == 0:
                continue

            if n_skipped < offset:
                n_skipped += 1
                continue

            candidates.append({
                "id": f"prompt_{offset + len(candidates):05d}",
                "user_message": user_msg,
                "prompt_ids": prompt_ids,
            })

            if len(candidates) >= n_prompts:
                break

        if offset > 0:
            print(f"Skipped {n_skipped} eligible prompts")
        print(f"Collected {len(candidates)} candidates")

    # Generate with vLLM
    print("Loading vLLM...")
    gc.collect()
    torch.cuda.empty_cache()
    # max_model_len: prompts are pre-filtered to stage1_max_chars (~5000 chars
    # ≈ ~1500 tokens) + max_tokens for response. 4096 covers the worst case
    # with headroom, and dramatically increases vLLM concurrency vs the
    # model default (40960 for Qwen3).
    llm_kwargs = dict(
        model=model_name,
        gpu_memory_utilization=0.9,
        enforce_eager=n_prompts <= 1000,
        max_model_len=max_tokens + 3000,
    )
    if quantization:
        llm_kwargs["quantization"] = quantization
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=max_tokens,
        n=n_completions,
    )

    print(f"Generating {n_completions} completions per prompt ({n_prompts} prompts)...")
    prompts_input = [{"prompt_token_ids": c["prompt_ids"]} for c in candidates]
    outputs = llm.generate(prompts_input, sampling_params=sampling_params, use_tqdm=True)

    # Build prompt completions
    prompt_results = []
    for i, (candidate, output) in enumerate(zip(candidates, outputs, strict=True)):
        completions = []
        finish_reasons = []
        token_counts = []
        for comp in output.outputs:
            text = comp.text
            if len(text.strip()) == 0:
                continue
            completions.append(text)
            finish_reasons.append(str(comp.finish_reason))
            token_counts.append(len(comp.token_ids))

        if len(completions) == 0:
            continue

        prompt_result = PromptCompletions(
            id=candidate["id"],
            user_message=candidate["user_message"],
            completions=completions,
            finish_reasons=finish_reasons,
            prompt_tokens=len(candidate["prompt_ids"]),
            response_token_counts=token_counts,
            messages=candidate.get("messages"),
            n_turns=candidate.get("n_turns", 1),
            total_chars=candidate.get("total_chars", len(candidate["user_message"])),
            conversation_hash=candidate.get("conversation_hash", ""),
        )
        prompt_results.append(prompt_result)

    # Cleanup vLLM
    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # Assemble dataset
    total_completions = sum(len(p.completions) for p in prompt_results)
    dataset = CompletionDataset(
        metadata={
            "model": model_name,
            "dataset": CHAT_DATASET,
            "n_prompts": len(prompt_results),
            "n_completions_per_prompt": n_completions,
            "total_completions": total_completions,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_response_tokens": MAX_RESPONSE_TOKENS,
            "temperature": 1.0,
            "enable_thinking": False,
            "seed": SEED,
            "offset": offset,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        prompts=prompt_results,
    )

    # Save
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(f"data_pipelines/model_understanding/{model_dir_name(model_name)}")
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"completions_{n_prompts}p_{n_completions}c.json"
    out_file.write_text(json.dumps(dataset.to_dict(), indent=2, ensure_ascii=False))
    print(f"\nSaved {len(prompt_results)} prompts x {n_completions} completions to {out_file}")

    # Summary stats
    prompt_lens = [p.prompt_tokens for p in prompt_results]
    all_response_lens = [t for p in prompt_results for t in p.response_token_counts]

    print(f"\n--- Summary ---")
    print(f"Total prompts: {len(prompt_results)}")
    print(f"Total completions: {total_completions}")
    print(f"Prompt tokens: min={min(prompt_lens)}, max={max(prompt_lens)}, mean={sum(prompt_lens)/len(prompt_lens):.0f}")
    print(f"Response tokens: min={min(all_response_lens)}, max={max(all_response_lens)}, mean={sum(all_response_lens)/len(all_response_lens):.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate completions for model understanding review")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID (e.g. Qwen/Qwen3-14B)")
    parser.add_argument("--n-prompts", type=int, default=None, help="Number of prompts (required unless --input-shard)")
    parser.add_argument("--n-completions", type=int, required=True, help="Number of completions per prompt")
    parser.add_argument("--offset", type=int, default=0, help="Number of eligible prompts to skip (for parallel generation)")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: auto-generated from params)")
    parser.add_argument("--input-shard", type=str, default=None, help="Pre-built shard file from prepare_shards.py (skips dataset streaming)")
    parser.add_argument("--max-tokens", type=int, default=MAX_RESPONSE_TOKENS, help=f"Max response tokens per completion (default {MAX_RESPONSE_TOKENS})")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization method (e.g. fp8)")
    args = parser.parse_args()
    if not args.input_shard and args.n_prompts is None:
        parser.error("--n-prompts is required when not using --input-shard")
    generate_completions(
        model_name=args.model,
        n_completions=args.n_completions,
        n_prompts=args.n_prompts,
        offset=args.offset,
        output_path=args.output,
        input_shard=args.input_shard,
        max_tokens=args.max_tokens,
        quantization=args.quantization,
    )
