# %%
"""
Interactive playground for generating responses from WildChat prompts.

Load the model once with vLLM, then run cells to generate responses,
inspect outputs, try different sampling params, truncate at various points, etc.
"""

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

PROMPTS_PATH = Path(__file__).parent / "wildchat_prompts.json"
MODEL_NAME = "Qwen/Qwen3-8B"

# %%
# === Load prompts ===

with open(PROMPTS_PATH) as f:
    data = json.load(f)

entries = data["entries"]
print(
    f"Loaded {len(entries)} prompts "
    f"({data['metadata']['single_turn']} single-turn, "
    f"{data['metadata']['multi_turn']} multi-turn)"
)

# %%
# === Load model with vLLM ===

import vllm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

vllm_model = vllm.LLM(
    model=MODEL_NAME,
    max_model_len=4096,
    enforce_eager=True,  # RunPod workaround
    tensor_parallel_size=1,
    gpu_memory_utilization=0.7,
)

print("Model loaded.")

# %%
# === Helper functions ===


def format_prompt(messages: list[dict], add_generation_prompt: bool = True) -> str:
    """Format messages using the model's chat template."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def generate(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 512,
    n: int = 1,
    logprobs: int = 20,
) -> list[vllm.RequestOutput]:
    """Generate response(s) from a list of messages."""
    formatted = format_prompt(messages)
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        logprobs=logprobs,
    )
    outputs = vllm_model.generate([formatted], sampling_params=sampling_params)
    return outputs[0]


def generate_batch(
    all_messages: list[list[dict]],
    temperature: float = 0.7,
    max_tokens: int = 512,
    n: int = 1,
    logprobs: int = 20,
) -> list[vllm.RequestOutput]:
    """Generate responses for a batch of conversations."""
    formatted = [format_prompt(msgs) for msgs in all_messages]
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        logprobs=logprobs,
    )
    return vllm_model.generate(formatted, sampling_params=sampling_params)


def show_response(output: "vllm.RequestOutput", response_idx: int = 0):
    """Pretty-print a single response with token logprobs."""
    resp = output.outputs[response_idx]
    print(f"Response text ({len(resp.token_ids)} tokens):")
    print("-" * 60)
    print(resp.text)
    print("-" * 60)

    if resp.logprobs:
        print(f"\nPer-token logprobs (first 20 tokens):")
        for i, lp in enumerate(resp.logprobs[:20]):
            # lp is a dict mapping token_id -> Logprob
            for token_id, logprob_obj in lp.items():
                import math

                prob = math.exp(logprob_obj.logprob)
                print(f"  [{i:3d}] {logprob_obj.decoded_token!r:20s} p={prob:.4f}")
                break  # just the chosen token


def show_entry(idx: int):
    """Display a prompt entry."""
    entry = entries[idx]
    print(f"=== {entry['id']} (multi_turn={entry['is_multi_turn']}, turns={entry['n_user_turns']}) ===\n")
    for msg in entry["prompt_messages"]:
        role = msg["role"].upper()
        content = msg["content"]
        print(f"[{role}]:\n{content}\n")


# %%
# === Pick an entry and generate a response ===

IDX = 2
show_entry(IDX)

# %%

entry = entries[IDX]
output = generate(entry["prompt_messages"], temperature=0.7, max_tokens=512)
show_response(output)

# %%
# === Generate multiple rollouts from the same prompt ===

entry = entries[IDX]
output = generate(entry["prompt_messages"], temperature=1.0, max_tokens=512, n=5)

for i in range(len(output.outputs)):
    print(f"\n{'=' * 60}")
    print(f"ROLLOUT {i}")
    print(f"{'=' * 60}")
    show_response(output, response_idx=i)

# %%
# === Truncate a response and generate continuations ===
# Generates a full response, truncates at a random sentence boundary,
# then generates continuations from the truncation point.
# Everything is printed in one clean block at the end.

import re

entry = entries[IDX]
N_CONTINUATIONS = 5

# Generate full response
full_output = generate(entry["prompt_messages"], temperature=0.7, max_tokens=512)
full_text = full_output.outputs[0].text

# Find all sentence boundaries, pick one at random
boundaries = [m.end() for m in re.finditer(r"(?:[.!?][\s\n]|\n)", full_text)]
min_pos = int(len(full_text) * 0.1)
max_pos = int(len(full_text) * 0.9)
boundaries = [b for b in boundaries if min_pos <= b <= max_pos]
assert boundaries, "No suitable boundaries found — response may be too short"
trunc_pos = random.choice(boundaries)

prefix = full_text[:trunc_pos]
actual_continuation = full_text[trunc_pos:]

# Generate continuations from the truncation point
formatted_with_gen = format_prompt(entry["prompt_messages"], add_generation_prompt=True)
continuation_prompt = formatted_with_gen + prefix
sampling_params = vllm.SamplingParams(temperature=1.0, max_tokens=256, n=N_CONTINUATIONS, logprobs=20)
continuation_outputs = vllm_model.generate([continuation_prompt], sampling_params=sampling_params)

# === Print everything ===

print("=" * 80)
print("PROMPT")
print("=" * 80)
for msg in entry["prompt_messages"]:
    print(f"\n[{msg['role'].upper()}]:")
    print(msg["content"])

print("\n" + "=" * 80)
print(f"PREFIX  ({len(prefix)} chars)")
print("=" * 80)
print(prefix)

print("\n" + "=" * 80)
print(f"ORIGINAL CONTINUATION  ({len(actual_continuation)} chars)")
print("=" * 80)
print(actual_continuation)

for i, out in enumerate(continuation_outputs[0].outputs):
    print("\n" + "=" * 80)
    print(f"GENERATED CONTINUATION {i}")
    print("=" * 80)
    print(out.text)

# %%
# === Batch generate responses for multiple prompts ===

batch_indices = list(range(min(10, len(entries))))
batch_messages = [entries[i]["prompt_messages"] for i in batch_indices]

batch_outputs = generate_batch(batch_messages, temperature=0.7, max_tokens=256)

for idx, output in zip(batch_indices, batch_outputs):
    print(f"\n{'=' * 80}")
    print(f"Entry: {entries[idx]['id']}")
    show_entry(idx)
    print(f"\nResponse:")
    print(output.outputs[0].text[:500])
    print()

# %%
# === Inspect token logprobs in detail for an entry ===

entry = entries[IDX]
output = generate(entry["prompt_messages"], temperature=0.0, max_tokens=256, logprobs=20)

resp = output.outputs[0]
print(f"Full response:\n{resp.text}\n")

if resp.logprobs:
    import math

    print(f"{'Pos':>4}  {'Token':20s}  {'Prob':>8}  {'Entropy':>8}  Top-3 alternatives")
    print("-" * 90)
    for i, lp in enumerate(resp.logprobs):
        # Get chosen token info
        chosen_id = resp.token_ids[i]
        chosen_lp = lp[chosen_id]
        chosen_prob = math.exp(chosen_lp.logprob)

        # Compute entropy from all returned logprobs
        probs = [math.exp(v.logprob) for v in lp.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        # Top alternatives (excluding chosen)
        alternatives = sorted(
            [(v.decoded_token, math.exp(v.logprob)) for tid, v in lp.items() if tid != chosen_id],
            key=lambda x: -x[1],
        )[:3]
        alt_str = ", ".join(f"{t!r}({p:.3f})" for t, p in alternatives)

        print(f"{i:4d}  {chosen_lp.decoded_token!r:20s}  {chosen_prob:8.4f}  {entropy:8.3f}  {alt_str}")
