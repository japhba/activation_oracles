"""One-off: generate 1k chat reg validation set. Uses seed=9999 and skips first
200k English entries to avoid overlap with the 100k training set."""

import gc
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen3-8B"
CHAT_DATASET = "lmsys/lmsys-chat-1m"
OUTPUT_PATH = Path("data_pipelines/chat_regularization/Qwen3-8B/t1_first_user_think50_1k_val.json")

MAX_TOTAL_TOKENS = 2000
MAX_PROMPT_TOKENS = 1000
NUM_ENTRIES = 1000
SKIP_N = 200_000
SEED = 9999


def main():
    random.seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    supports_thinking = (
        hasattr(tokenizer, "chat_template")
        and tokenizer.chat_template is not None
        and "enable_thinking" in tokenizer.chat_template
    )

    print(f"Streaming {CHAT_DATASET}, skipping first {SKIP_N} English entries...")
    ds = iter(load_dataset(CHAT_DATASET, split="train", streaming=True))
    candidates = []
    skipped = 0

    for row in tqdm(ds, desc="Scanning"):
        if row.get("language") != "English":
            continue
        conversation = row["conversation"]
        first_user_msg = None
        for msg in conversation:
            if msg["role"] == "user":
                first_user_msg = msg["content"]
                break
        if first_user_msg is None or len(first_user_msg.strip()) == 0:
            continue

        if skipped < SKIP_N:
            skipped += 1
            continue

        enable_thinking = random.random() < 0.5
        messages = [{"role": "user", "content": first_user_msg}]
        kwargs = {"tokenize": True, "add_generation_prompt": True, "return_tensors": None, "padding": False}
        if supports_thinking:
            kwargs["enable_thinking"] = enable_thinking
        prompt_ids = tokenizer.apply_chat_template(messages, **kwargs)

        if len(prompt_ids) > MAX_PROMPT_TOKENS or len(prompt_ids) == 0:
            continue
        max_new_tokens = MAX_TOTAL_TOKENS - len(prompt_ids)
        if max_new_tokens < 10:
            continue

        candidates.append({
            "user_message": first_user_msg,
            "prompt_ids": prompt_ids,
            "enable_thinking": enable_thinking,
            "max_new_tokens": max_new_tokens,
        })
        if len(candidates) >= NUM_ENTRIES:
            break

    print(f"Collected {len(candidates)} candidates, generating with vLLM...")
    llm = LLM(model=MODEL_NAME, gpu_memory_utilization=0.9, enforce_eager=True)

    prompts = [{"prompt_token_ids": c["prompt_ids"]} for c in candidates]
    per_prompt_params = [
        SamplingParams(temperature=1.0, max_tokens=c["max_new_tokens"], detokenize=False)
        for c in candidates
    ]
    outputs = llm.generate(prompts, sampling_params=per_prompt_params, use_tqdm=True)

    entries = []
    for candidate, output in zip(candidates, outputs, strict=True):
        if len(output.outputs) == 0:
            continue
        response_ids = list(output.outputs[0].token_ids)
        if len(response_ids) == 0:
            continue
        prompt_ids = candidate["prompt_ids"]
        max_response = MAX_TOTAL_TOKENS - len(prompt_ids)
        if len(response_ids) > max_response:
            response_ids = response_ids[:max_response]

        entries.append({
            "id": f"chat_reg_val_{len(entries)}",
            "prompt_token_ids": prompt_ids,
            "response_token_ids": response_ids,
            "prompt_len": len(prompt_ids),
            "response_len": len(response_ids),
            "total_len": len(prompt_ids) + len(response_ids),
            "enable_thinking": candidate["enable_thinking"],
            "finish_reason": str(output.outputs[0].finish_reason),
        })

    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    dataset = {
        "metadata": {
            "model": MODEL_NAME,
            "total_entries": len(entries),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "chat_dataset": CHAT_DATASET,
            "max_total_tokens": MAX_TOTAL_TOKENS,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "temperature": 1.0,
            "seed": SEED,
            "thinking_fraction": 0.5,
            "skip_n": SKIP_N,
        },
        "entries": entries,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(dataset, indent=2))
    print(f"\nSaved {len(entries)} entries to {OUTPUT_PATH}")

    prompt_lens = [e["prompt_len"] for e in entries]
    response_lens = [e["response_len"] for e in entries]
    total_lens = [e["total_len"] for e in entries]
    thinking_count = sum(1 for e in entries if e["enable_thinking"])
    print(f"Thinking enabled: {thinking_count}/{len(entries)}")
    print(f"Prompt lengths: min={min(prompt_lens)}, max={max(prompt_lens)}, mean={sum(prompt_lens)/len(prompt_lens):.0f}")
    print(f"Response lengths: min={min(response_lens)}, max={max(response_lens)}, mean={sum(response_lens)/len(response_lens):.0f}")
    print(f"Total lengths: min={min(total_lens)}, max={max(total_lens)}, mean={sum(total_lens)/len(total_lens):.0f}")


if __name__ == "__main__":
    main()
