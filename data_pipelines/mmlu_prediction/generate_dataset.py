"""
Generate the MMLU prediction eval dataset end-to-end.

1. Downloads MMLU test set (all subjects)
2. Runs the target model at temperature 0 (no thinking) on all questions
3. Scores correctness
4. Samples ~100 balanced problems (50 correct, 50 incorrect)
5. Saves final dataset to mmlu_prediction/{model_dir}/mmlu_prediction_eval_dataset.json

Usage:
    .venv/bin/python data_pipelines/mmlu_prediction/generate_dataset.py --model Qwen/Qwen3-8B
"""

import argparse
import json
import random
import re
from pathlib import Path

import vllm
from datasets import load_dataset
from transformers import AutoTokenizer

from data_pipelines.pipeline_utils import model_dir_name, add_model_arg
TARGET_PER_CLASS = 50  # 50 correct + 50 incorrect

random.seed(42)

ANSWER_LETTERS = ["A", "B", "C", "D"]


def format_mmlu_question(question: str, choices: list[str]) -> str:
    """Format an MMLU question with lettered choices."""
    lines = [question, ""]
    for i, choice in enumerate(choices):
        lines.append(f"{ANSWER_LETTERS[i]}. {choice}")
    lines.append("")
    lines.append("Answer with just the letter (A, B, C, or D), nothing else.")
    return "\n".join(lines)


def extract_answer_letter(response: str) -> str | None:
    """Extract a single answer letter from model response."""
    text = response.strip()
    # Check if first non-whitespace char is A-D
    if text and text[0] in "ABCD":
        return text[0]
    # Search for standalone letter
    match = re.search(r"\b([A-D])\b", text)
    if match:
        return match.group(1)
    return None


def main(model_name: str):
    # Load full MMLU test set
    print("Loading MMLU test set...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    print(f"Loaded {len(ds)} questions across all subjects")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm = vllm.LLM(
        model=model_name,
        max_model_len=2048,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
    )

    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=10)

    # Format all prompts
    all_items = []
    formatted_prompts = []
    for i, row in enumerate(ds):
        question_text = format_mmlu_question(row["question"], row["choices"])
        messages = [{"role": "user", "content": question_text}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        formatted_prompts.append(prompt)
        all_items.append({
            "index": i,
            "question": row["question"],
            "subject": row["subject"],
            "choices": row["choices"],
            "correct_answer_idx": row["answer"],
            "correct_answer_letter": ANSWER_LETTERS[row["answer"]],
        })

    print(f"Running {len(formatted_prompts)} prompts through vLLM at temperature 0...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    # Score all outputs
    correct_items = []
    incorrect_items = []

    for item, output in zip(all_items, outputs):
        raw_response = output.outputs[0].text
        model_letter = extract_answer_letter(raw_response)
        model_correct = model_letter == item["correct_answer_letter"]

        item["model_raw_response"] = raw_response
        item["model_answer_letter"] = model_letter
        item["model_correct"] = model_correct

        if model_letter is None:
            continue  # Skip unparseable responses

        if model_correct:
            correct_items.append(item)
        else:
            incorrect_items.append(item)

    total_parseable = len(correct_items) + len(incorrect_items)
    unparseable = len(all_items) - total_parseable
    accuracy = len(correct_items) / total_parseable if total_parseable > 0 else 0

    print(f"\nFull MMLU results:")
    print(f"  Total: {len(all_items)}")
    print(f"  Parseable: {total_parseable} (unparseable: {unparseable})")
    print(f"  Correct: {len(correct_items)} ({100*accuracy:.1f}%)")
    print(f"  Incorrect: {len(incorrect_items)} ({100*(1-accuracy):.1f}%)")

    # Sample balanced subset
    n_correct = min(TARGET_PER_CLASS, len(correct_items))
    n_incorrect = min(TARGET_PER_CLASS, len(incorrect_items))
    # Use whichever class has fewer to keep it balanced
    n_per_class = min(n_correct, n_incorrect)

    random.shuffle(correct_items)
    random.shuffle(incorrect_items)
    sampled_correct = correct_items[:n_per_class]
    sampled_incorrect = incorrect_items[:n_per_class]

    # Build dataset entries
    entries = []
    for item in sampled_correct + sampled_incorrect:
        entries.append({
            "id": f"mmlu_{item['index']}",
            "question": item["question"],
            "subject": item["subject"],
            "choices": item["choices"],
            "correct_answer_idx": item["correct_answer_idx"],
            "correct_answer_letter": item["correct_answer_letter"],
            "model_raw_response": item["model_raw_response"],
            "model_answer_letter": item["model_answer_letter"],
            "model_correct": item["model_correct"],
        })

    random.shuffle(entries)

    # Subject distribution
    subjects = {}
    for e in entries:
        subjects[e["subject"]] = subjects.get(e["subject"], 0) + 1

    dataset = {
        "metadata": {
            "model": model_name,
            "temperature": 0,
            "thinking_enabled": False,
            "total_entries": len(entries),
            "correct_count": sum(1 for e in entries if e["model_correct"]),
            "incorrect_count": sum(1 for e in entries if not e["model_correct"]),
            "full_mmlu_accuracy": round(accuracy, 4),
            "full_mmlu_total": len(all_items),
            "num_subjects": len(subjects),
        },
        "entries": entries,
    }

    output_path = Path(f"data_pipelines/mmlu_prediction/{model_dir_name(model_name)}/mmlu_prediction_eval_dataset.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    # Print summary
    n_correct_final = sum(1 for e in entries if e["model_correct"])
    n_incorrect_final = sum(1 for e in entries if not e["model_correct"])
    print(f"\nSaved {len(entries)} entries to {output_path}")
    print(f"  Correct: {n_correct_final}, Incorrect: {n_incorrect_final}")
    print(f"  Subjects: {len(subjects)}")
    print(f"\nSubject distribution:")
    for subj, count in sorted(subjects.items(), key=lambda x: -x[1])[:20]:
        print(f"  {subj}: {count}")

    print(f"\nSample entries:")
    for e in entries[:10]:
        print(
            f"  [{e['id']}] {e['subject']:<30} correct={e['correct_answer_letter']} "
            f"model={e['model_answer_letter']} {'✓' if e['model_correct'] else '✗'}  "
            f"Q: {e['question'][:80]}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MMLU prediction eval dataset")
    add_model_arg(parser)
    args = parser.parse_args()
    main(model_name=args.model)
