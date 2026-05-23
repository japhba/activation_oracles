"""
Quick experiment: try different pre-answer prompt formats for MMLU text baseline.

The current prompt puts "Answer with just the letter (A, B, C, or D), nothing else."
immediately before "Answer with 'Yes' or 'No' only. Will you likely answer this correctly?"
which confuses the model (0.476 AUC = worse than random).

Try several alternatives to see if prompt formatting is the bottleneck.

Usage:
    source .env && .venv/bin/python experiments/mmlu_preanswer_prompt_sweep.py
"""

import json
import os
import random

import torch

from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.open_ended_eval.eval_runner import (
    render_baseline_chat_prompt,
    build_yes_no_candidate_token_groups,
    score_binary_yes_no_results,
    compute_binary_yes_no_metrics,
    TextBaselineInput,
    run_text_baseline_eval_loop,
)
from nl_probes.open_ended_eval.mmlu_prediction import (
    ANSWER_LETTERS,
    load_mmlu_prediction_dataset,
)

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = "Qwen/Qwen3-8B"


def format_question_no_answer_instruction(question: str, choices: list[str]) -> str:
    """Format MMLU question WITHOUT the 'answer with letter only' instruction."""
    lines = [question, ""]
    for i, choice in enumerate(choices):
        lines.append(f"{ANSWER_LETTERS[i]}. {choice}")
    return "\n".join(lines)


def format_question_standard(question: str, choices: list[str]) -> str:
    """Standard format with answer instruction (matches dataset generation)."""
    lines = [question, ""]
    for i, choice in enumerate(choices):
        lines.append(f"{ANSWER_LETTERS[i]}. {choice}")
    lines.append("")
    lines.append("Answer with just the letter (A, B, C, or D), nothing else.")
    return "\n".join(lines)


# Prompt variants to try. Each is a function(question_text, tokenizer) -> prompt_text
PROMPT_VARIANTS: dict[str, dict] = {
    # Original (broken): single user message with both instructions back-to-back
    "original": {
        "description": "Original prompt (ABCD instruction + yes/no in same message)",
        "build_fn": lambda q_std, q_no_instr, tokenizer: render_baseline_chat_prompt(
            tokenizer=tokenizer,
            messages=[{
                "role": "user",
                "content": f"{q_std}\nQuestion: Answer with 'Yes' or 'No' only. Will you likely answer this correctly?",
            }],
            add_generation_prompt=True,
            enable_thinking=False,
        ),
    },
    # Fix 1: Two-turn — first turn is the question, second turn asks about confidence
    "two_turn": {
        "description": "Two-turn: question in turn 1, confidence question in turn 2",
        "build_fn": lambda q_std, q_no_instr, tokenizer: render_baseline_chat_prompt(
            tokenizer=tokenizer,
            messages=[
                {"role": "user", "content": q_std},
                {"role": "assistant", "content": "Let me think about this."},
                {"role": "user", "content": "Answer with 'Yes' or 'No' only. Will you likely answer this correctly?"},
            ],
            add_generation_prompt=True,
            enable_thinking=False,
        ),
    },
    # Fix 2: Drop the ABCD instruction entirely, just ask about correctness
    "no_abcd_instruction": {
        "description": "No ABCD instruction, just question + choices + confidence question",
        "build_fn": lambda q_std, q_no_instr, tokenizer: render_baseline_chat_prompt(
            tokenizer=tokenizer,
            messages=[{
                "role": "user",
                "content": (
                    f"{q_no_instr}\n\n"
                    "Before answering, tell me: do you know the answer to this question? "
                    "Answer with 'Yes' or 'No' only."
                ),
            }],
            add_generation_prompt=True,
            enable_thinking=False,
        ),
    },
    # Fix 3: Clear separation with a divider
    "separator": {
        "description": "Original question + separator + confidence question",
        "build_fn": lambda q_std, q_no_instr, tokenizer: render_baseline_chat_prompt(
            tokenizer=tokenizer,
            messages=[{
                "role": "user",
                "content": (
                    f"{q_std}\n\n---\n\n"
                    "Ignore the above instructions about answering with a letter. "
                    "Instead, answer with 'Yes' or 'No' only: "
                    "Are you confident you know the correct answer to the above question?"
                ),
            }],
            add_generation_prompt=True,
            enable_thinking=False,
        ),
    },
    # Fix 4: System prompt framing
    "system_prompt": {
        "description": "Confidence question in system prompt, MMLU question in user message",
        "build_fn": lambda q_std, q_no_instr, tokenizer: render_baseline_chat_prompt(
            tokenizer=tokenizer,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a calibration assistant. The user will show you a multiple choice question. "
                        "Your job is to assess whether you would get this question right. "
                        "Answer ONLY 'Yes' or 'No'. Do NOT answer the question itself."
                    ),
                },
                {"role": "user", "content": f"{q_no_instr}\n\nWould you answer this correctly?"},
            ],
            add_generation_prompt=True,
            enable_thinking=False,
        ),
    },
    # Fix 5: Thinking enabled — let the model reason before answering
    "thinking_enabled": {
        "description": "Same as two_turn but with thinking enabled",
        "build_fn": lambda q_std, q_no_instr, tokenizer: render_baseline_chat_prompt(
            tokenizer=tokenizer,
            messages=[
                {"role": "user", "content": q_std},
                {"role": "assistant", "content": "Let me think about this."},
                {"role": "user", "content": "Answer with 'Yes' or 'No' only. Will you likely answer this correctly?"},
            ],
            add_generation_prompt=True,
            enable_thinking=True,
        ),
    },
}


def build_inputs_for_variant(
    entries: list[dict],
    variant_name: str,
    variant_config: dict,
    tokenizer,
) -> tuple[list[TextBaselineInput], list[dict]]:
    baseline_inputs = []
    entry_metadata = []

    for entry in entries:
        q_std = format_question_standard(entry["question"], entry["choices"])
        q_no_instr = format_question_no_answer_instruction(entry["question"], entry["choices"])
        ground_truth = "yes" if entry["model_correct"] else "no"

        prompt_text = variant_config["build_fn"](q_std, q_no_instr, tokenizer)

        baseline_inputs.append(TextBaselineInput(
            prompt_text=prompt_text,
            ground_truth=ground_truth,
            prompt_name="pre_likely_correct",
            variant="full_context",
        ))
        entry_metadata.append({
            "id": entry["id"],
            "subject": entry["subject"],
            "correct_answer_letter": entry["correct_answer_letter"],
            "model_answer_letter": entry["model_answer_letter"],
            "model_correct": entry["model_correct"],
            "prompt_name": "pre_likely_correct",
            "baseline_variant": variant_name,
        })

    return baseline_inputs, entry_metadata


def main():
    random.seed(42)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    entries = load_mmlu_prediction_dataset(MODEL_NAME)
    print(f"Loaded {len(entries)} entries")
    print(f"Correct rate: {sum(e['model_correct'] for e in entries) / len(entries):.1%}")

    print(f"\nLoading model {MODEL_NAME}...")
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, dtype)
    model.eval()

    output_dir = "experiments/mmlu_preanswer_prompt_sweep_results"
    os.makedirs(output_dir, exist_ok=True)

    # Print example prompts
    sample_entry = entries[0]
    q_std = format_question_standard(sample_entry["question"], sample_entry["choices"])
    q_no_instr = format_question_no_answer_instruction(sample_entry["question"], sample_entry["choices"])
    print("\n" + "=" * 70)
    print("EXAMPLE PROMPTS (first entry)")
    print("=" * 70)
    for name, config in PROMPT_VARIANTS.items():
        prompt = config["build_fn"](q_std, q_no_instr, tokenizer)
        print(f"\n--- {name}: {config['description']} ---")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

    all_metrics = {}

    for variant_name, variant_config in PROMPT_VARIANTS.items():
        print(f"\n{'='*70}")
        print(f"  Variant: {variant_name}")
        print(f"  {variant_config['description']}")
        print(f"{'='*70}")

        baseline_inputs, entry_metadata = build_inputs_for_variant(
            entries, variant_name, variant_config, tokenizer,
        )

        variant_output_dir = os.path.join(output_dir, variant_name)
        summary = run_text_baseline_eval_loop(
            eval_name=f"mmlu_pre_{variant_name}",
            baseline_inputs=baseline_inputs,
            entry_metadata=entry_metadata,
            backend="hf",
            inference_mode="binary_yes_no",
            model_name=MODEL_NAME,
            tokenizer=tokenizer,
            generation_kwargs={"do_sample": False, "max_new_tokens": 1},
            score_fn=score_binary_yes_no_results,
            metrics_fn=compute_binary_yes_no_metrics,
            output_dir=variant_output_dir,
            hf_model=model,
            device=device,
            eval_batch_size=32,
        )

        metrics = summary.get("overall_metrics", {})
        all_metrics[variant_name] = {
            "description": variant_config["description"],
            "roc_auc": metrics.get("roc_auc", None),
            "accuracy": metrics.get("accuracy_at_zero", None),
            "mean_margin_yes": metrics.get("mean_margin_when_yes", None),
            "mean_margin_no": metrics.get("mean_margin_when_no", None),
        }

    # Summary table
    print(f"\n{'='*90}")
    print("PROMPT VARIANT COMPARISON")
    print(f"{'='*90}")
    print(f"{'Variant':<25} {'AUC':>8} {'Acc':>8} {'Margin(yes)':>12} {'Margin(no)':>12}")
    print("-" * 70)
    for name, m in all_metrics.items():
        auc = f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "N/A"
        acc = f"{m['accuracy']:.3f}" if m["accuracy"] is not None else "N/A"
        my = f"{m['mean_margin_yes']:.3f}" if m["mean_margin_yes"] is not None else "N/A"
        mn = f"{m['mean_margin_no']:.3f}" if m["mean_margin_no"] is not None else "N/A"
        print(f"{name:<25} {auc:>8} {acc:>8} {my:>12} {mn:>12}")

    # Save
    results_path = os.path.join(output_dir, "comparison.json")
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
