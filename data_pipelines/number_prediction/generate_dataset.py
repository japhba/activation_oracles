"""
Generate the number prediction eval dataset end-to-end.

1. Generates ~50 arithmetic problems across 5 difficulty categories
2. Runs the target model at temperature 0 (no thinking) to get model answers
3. Collects per-token logprobs for the full answer
4. Tokenizes answers to label single-token vs multi-token
5. Saves final dataset to number_prediction_eval_dataset.json

Usage:
    .venv/bin/python data_pipelines/number_prediction/generate_dataset.py --model Qwen/Qwen3-8B
"""

import argparse
import json
import math
import random
import re
from pathlib import Path

import vllm
from transformers import AutoTokenizer

from data_pipelines.pipeline_utils import model_dir_name, add_model_arg

random.seed(42)


def generate_problems():
    problems = []

    # Simple 2-operand (answers likely 1-2 tokens)
    for i in range(10):
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        op = random.choice(["+", "-", "*"])
        expr = f"{a} {op} {b}"
        problems.append({"id": f"simple_{i}", "expression": expr, "true_answer": eval(expr), "category": "simple_2op"})

    # Medium complexity (3-4 operands, mixed ops)
    for i in range(15):
        n_ops = random.randint(2, 3)
        nums = [random.randint(-99, 99) for _ in range(n_ops + 1)]
        ops = [random.choice(["+", "-", "*"]) for _ in range(n_ops)]
        expr = str(nums[0])
        for j in range(n_ops):
            if random.random() < 0.3:
                expr = f"({expr} {ops[j]} {nums[j+1]})"
            else:
                expr = f"{expr} {ops[j]} {nums[j+1]}"
        problems.append({"id": f"medium_{i}", "expression": expr, "true_answer": eval(expr), "category": "medium_3_4op"})

    # Large 2-operand (answers likely multi-token)
    for i in range(10):
        a = random.randint(100, 999)
        b = random.randint(100, 999)
        op = random.choice(["+", "-", "*"])
        expr = f"{a} {op} {b}"
        problems.append({"id": f"large_{i}", "expression": expr, "true_answer": eval(expr), "category": "large_numbers"})

    # Division and modulo
    for i in range(8):
        a = random.randint(50, 500)
        b = random.randint(2, 20)
        op = random.choice(["//", "%"])
        expr = f"{a} {op} {b}"
        problems.append({"id": f"divmod_{i}", "expression": expr, "true_answer": eval(expr), "category": "divmod"})

    # Nested expressions
    for i in range(7):
        nums = [random.randint(-99, 99) for _ in range(5)]
        templates = [
            f"({nums[0]} * {nums[1]}) + ({nums[2]} - {nums[3]}) * {nums[4]}",
            f"({nums[0]} + {nums[1]}) * ({nums[2]} - {nums[3]})",
            f"{nums[0]} * {nums[1]} + {nums[2]} * {nums[3]} - {nums[4]}",
            f"({nums[0]} + {nums[1]} + {nums[2]}) * ({nums[3]} - {nums[4]})",
            f"{nums[0]} * ({nums[1]} + {nums[2]}) - {nums[3]} * {nums[4]}",
        ]
        expr = random.choice(templates)
        problems.append({"id": f"nested_{i}", "expression": expr, "true_answer": eval(expr), "category": "nested"})

    return problems


def main(model_name: str):
    output_path = Path(f"data_pipelines/number_prediction/{model_dir_name(model_name)}/number_prediction_eval_dataset.json")

    problems = generate_problems()
    print(f"Generated {len(problems)} problems")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm = vllm.LLM(
        model=model_name,
        max_model_len=2000,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
    )

    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=50, logprobs=20)

    # Format prompts — thinking disabled, just ask for the number
    formatted_prompts = []
    for p in problems:
        messages = [{"role": "user", "content": f"What is {p['expression']}? Answer with just the number, nothing else."}]
        formatted_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))

    print(f"Running {len(formatted_prompts)} prompts at temperature 0...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    # Build dataset entries
    entries = []
    for problem, output in zip(problems, outputs):
        raw_response = output.outputs[0].text
        match = re.search(r"-?\d+", raw_response.strip())
        model_answer = int(match.group()) if match else None

        true_tokens = tokenizer.encode(str(problem["true_answer"]), add_special_tokens=False)
        model_tokens = tokenizer.encode(str(model_answer), add_special_tokens=False) if model_answer is not None else []

        # Extract per-token logprobs for the full output
        token_logprobs = []
        for step_logprobs in output.outputs[0].logprobs:
            step_entries = []
            for token_id, lp_entry in step_logprobs.items():
                step_entries.append({
                    "token": lp_entry.decoded_token,
                    "token_id": token_id,
                    "logprob": lp_entry.logprob,
                    "prob": round(math.exp(lp_entry.logprob), 6),
                })
            step_entries.sort(key=lambda x: x["logprob"], reverse=True)
            token_logprobs.append({
                "chosen_token": step_entries[0]["token"],
                "chosen_prob": step_entries[0]["prob"],
                "entropy_bits": round(-sum(e["prob"] * math.log2(e["prob"]) for e in step_entries if e["prob"] > 0), 4),
                "top5": [{"token": e["token"], "prob": e["prob"]} for e in step_entries[:5]],
            })

        entries.append({
            "id": problem["id"],
            "expression": problem["expression"],
            "category": problem["category"],
            "true_answer": problem["true_answer"],
            "true_answer_num_tokens": len(true_tokens),
            "model_raw_response": raw_response,
            "model_answer": model_answer,
            "model_correct": model_answer == problem["true_answer"],
            "model_answer_num_tokens": len(model_tokens),
            "is_single_token_answer": len(true_tokens) == 1,
            "token_logprobs": token_logprobs,
        })

    dataset = {
        "metadata": {
            "model": model_name,
            "temperature": 0,
            "thinking_enabled": False,
            "total_problems": len(entries),
            "model_accuracy": sum(1 for e in entries if e["model_correct"]) / len(entries),
            "single_token_count": sum(1 for e in entries if e["is_single_token_answer"]),
            "multi_token_count": sum(1 for e in entries if not e["is_single_token_answer"]),
        },
        "entries": entries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    # Print summary
    correct = sum(1 for e in entries if e["model_correct"])
    print(f"\nSaved {len(entries)} entries to {output_path}")
    print(f"Model accuracy: {correct}/{len(entries)} ({100*correct/len(entries):.1f}%)")
    print(f"Single-token answers: {dataset['metadata']['single_token_count']}, Multi-token: {dataset['metadata']['multi_token_count']}")

    print(f"\n{'ID':<15} {'Expression':<45} {'True':>8} {'Model':>8} {'OK':>4} {'#Tok':>5}  Token probs")
    print("-" * 130)
    for e in entries:
        tok_summary = "  ".join(f"{t['chosen_token']!r}:{t['chosen_prob']:.3f}" for t in e["token_logprobs"][:6])
        print(f"{e['id']:<15} {e['expression']:<45} {e['true_answer']:>8} {str(e['model_answer']):>8} {'✓' if e['model_correct'] else '✗':>4} {e['true_answer_num_tokens']:>5}  {tok_summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate number prediction eval dataset")
    add_model_arg(parser)
    args = parser.parse_args()
    main(model_name=args.model)
