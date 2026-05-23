# %%
import json
from pathlib import Path

import vllm
from transformers import AutoTokenizer

# %%
# Config
# Use Qwen3-0.6B + 0.2 GPU for debugging, swap to 8B + 0.9 for real runs
MODEL_NAME = "Qwen/Qwen3-0.6B"
# MODEL_NAME = "Qwen/Qwen3-8B"

GPU_MEM = 0.2  # bump to 0.9 for real runs

PROBLEMS_PATH = Path("data_pipelines/backtracking/problems.json")
OUTPUT_PATH = Path("data_pipelines/backtracking/rollouts.json")

NUM_ROLLOUTS = 1  # bump to 10 for real runs
MAX_TOKENS = 1000  # bump to 8000+ for real runs

# enforce_eager=True skips torch.compile (saves ~26s load, but 2.5x slower generation)
# For real runs, set False — compile cost is one-time
ENFORCE_EAGER = True

# %%
# Load problems
with open(PROBLEMS_PATH) as f:
    problems = json.load(f)

print(f"Loaded {len(problems)} problems")
print(f"First problem: {problems[0]['problem'][:100]}...")

# %%
# Init tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# %%
# Sanity check: look at what the formatted prompt actually looks like
test_messages = [{"role": "user", "content": problems[0]["problem"]}]

formatted_str = tokenizer.apply_chat_template(
    test_messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
print("=== Formatted prompt (string) ===")
print(repr(formatted_str))
print()
print(formatted_str)

# %%
# Sanity check: look at the actual token IDs
token_ids = tokenizer.apply_chat_template(
    test_messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=True,
)
print(f"=== Token IDs ({len(token_ids)} tokens) ===")
print(token_ids)
print()

# Decode back to see what the model actually sees
decoded = tokenizer.decode(token_ids)
print("=== Decoded from token IDs ===")
print(decoded)
print()

# Check special tokens
print("=== Special tokens in prompt ===")
for tid in token_ids:
    tok = tokenizer.decode([tid])
    if tid in tokenizer.all_special_ids:
        print(f"  {tid:>6d} -> {repr(tok)}")

# %%
# Load vLLM
llm = vllm.LLM(
    model=MODEL_NAME,
    max_model_len=4000,
    enforce_eager=ENFORCE_EAGER,
    tensor_parallel_size=1,
    gpu_memory_utilization=GPU_MEM,
)

# %%
# Test with a single problem
sampling_params = vllm.SamplingParams(
    temperature=1.0,
    top_p=0.95,
    max_tokens=MAX_TOKENS,
)

outputs = llm.generate([formatted_str], sampling_params)
out = outputs[0].outputs[0]

print(f"=== Generated {len(out.token_ids)} tokens ===")
print()
print("=== Full output text ===")
print(out.text)
print()
print(f"Contains <think>: {'<think>' in out.text}")
print(f"Contains </think>: {'</think>' in out.text}")

# %%
# Parse thinking vs answer
full_text = out.text
if "<think>" in full_text:
    parts = full_text.split("</think>", 1)
    if len(parts) == 2:
        thinking = parts[0].replace("<think>", "").strip()
        answer = parts[1].strip()
    else:
        thinking = full_text.replace("<think>", "").strip()
        answer = "[thinking didn't close - hit max tokens]"
else:
    thinking = ""
    answer = full_text

print(f"=== Thinking ({len(thinking)} chars) ===")
print(thinking[:500])
print()
print(f"=== Answer ===")
print(answer[:500])

# %%
# Try first 5 problems
test_problems = problems[:5]
test_formatted = []
for p in test_problems:
    msgs = [{"role": "user", "content": p["problem"]}]
    fmt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    test_formatted.append(fmt)

outputs = llm.generate(test_formatted, sampling_params)

for i, (prob, out) in enumerate(zip(test_problems, outputs)):
    text = out.outputs[0].text
    has_think = "<think>" in text and "</think>" in text
    print(f"\n{'='*60}")
    print(f"Problem {i}: {prob['id']} ({prob['domain']})")
    print(f"  Q: {prob['problem'][:80]}...")
    print(f"  Tokens: {len(out.outputs[0].token_ids)}, Has thinking: {has_think}")
    if has_think:
        think_part = text.split("</think>")[0].replace("<think>", "").strip()
        print(f"  Thinking preview: {think_part[:200]}...")

# %%
# Full generation
# Before running: swap MODEL_NAME to Qwen3-8B, GPU_MEM to 0.9,
# NUM_ROLLOUTS to 10, MAX_TOKENS to 8000+, ENFORCE_EAGER to False
# Then re-run from the model loading cell

all_formatted = []
prompt_index_map = []

for prob_idx, problem in enumerate(problems):
    messages = [{"role": "user", "content": problem["problem"]}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    for rollout_idx in range(NUM_ROLLOUTS):
        all_formatted.append(formatted)
        prompt_index_map.append((prob_idx, rollout_idx))

print(f"Generating {len(all_formatted)} rollouts...")

full_sampling = vllm.SamplingParams(temperature=1.0, top_p=0.95, max_tokens=MAX_TOKENS)
all_outputs = llm.generate(all_formatted, full_sampling)

results = []
for problem in problems:
    results.append({
        "id": problem["id"],
        "domain": problem["domain"],
        "problem": problem["problem"],
        "notes": problem["notes"],
        "rollouts": [],
    })

for i, output in enumerate(all_outputs):
    prob_idx, rollout_idx = prompt_index_map[i]
    full_text = output.outputs[0].text
    thinking = ""
    answer = full_text
    if "<think>" in full_text:
        parts = full_text.split("</think>", 1)
        if len(parts) == 2:
            thinking = parts[0].replace("<think>", "").strip()
            answer = parts[1].strip()
        else:
            thinking = full_text.replace("<think>", "").strip()
            answer = ""

    results[prob_idx]["rollouts"].append({
        "rollout_idx": rollout_idx,
        "thinking": thinking,
        "answer": answer,
        "full_text": full_text,
        "num_tokens": len(output.outputs[0].token_ids),
    })

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

total_tokens = sum(r["num_tokens"] for result in results for r in result["rollouts"])
print(f"Done! Saved to {OUTPUT_PATH}")
print(f"Total tokens: {total_tokens:,}")

# %%
