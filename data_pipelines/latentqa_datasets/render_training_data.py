"""
Render latentQA entries into synthetic QA training format.

Takes QA results (from generate_qa.py) and responses (from generate_responses.py),
applies chat template formatting, and produces entries compatible with
SyntheticQADatasetLoader (prefix_text + selected_text format).

Each entry is duplicated into two variants:
  1. user-turn: activations from the user message
  2. response-turn: activations from the assistant response

For each variant, 50% of the time we use the full turn content as selected_text,
50% of the time we pick a log-uniform random sub-window.

Usage:
    .venv/bin/python data_pipelines/latentqa_datasets/render_training_data.py [--n 20]

Output: data_pipelines/latentqa_datasets/artifacts/training_data_{n}.json
"""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

TARGET_MODEL = "Qwen/Qwen3-8B"
SEED = 42
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def load_qa_results(tag):
    """Load QA results from generate_qa.py."""
    path = ARTIFACTS_DIR / f"qa_{tag}.json"
    with open(path) as f:
        data = json.load(f)
    return data["entries"]


def render_entries(qa_results, tokenizer):
    """Render each QA result into synthetic QA format entries.

    Each QA result (which has 2 questions from the old qa.json) produces
    4 training entries: 2 questions x 2 variants (user-turn, response-turn).
    """
    rng = random.Random(SEED)
    entries = []

    for result in qa_results:
        label = result["label"]
        system_prompt = result["system_prompt"]
        user_message = result["user_message"]
        response = result["response"]
        question = result["question"]
        old_answer = result["old_answer"]
        new_answer = result["new_answer"]

        if not question or not new_answer:
            continue

        # Render the full conversation with chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response},
        ]
        full_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )

        # Find content boundaries in the rendered string
        user_start = full_str.index(user_message)
        user_end = user_start + len(user_message)

        asst_start = full_str.index(response, user_end)
        asst_end = asst_start + len(response)

        # Variant 1: user-turn (activations from user message)
        prefix, selected = _select_window(
            full_str, user_start, user_end, rng,
        )
        entries.append(_make_entry(
            entry_id=f"latentqa_{label}_q{result['q_idx']}_user",
            qa_type="latentqa_user",
            prefix_text=prefix,
            selected_text=selected,
            question=question,
            answer=new_answer,
            label=label,
            full_text=full_str,
        ))

        # Variant 2: response-turn (activations from assistant response)
        prefix, selected = _select_window(
            full_str, asst_start, asst_end, rng,
        )
        entries.append(_make_entry(
            entry_id=f"latentqa_{label}_q{result['q_idx']}_response",
            qa_type="latentqa_response",
            prefix_text=prefix,
            selected_text=selected,
            question=question,
            answer=new_answer,
            label=label,
            full_text=full_str,
        ))

    return entries


def _select_window(full_str, content_start, content_end, rng):
    """Select full content or a random sub-window (50/50).

    Returns (prefix_text, selected_text) where
    prefix_text + selected_text is a substring of full_str.
    The window can start anywhere within the content — beginning, middle, or end.
    """
    content_len = content_end - content_start

    if content_len <= 1 or rng.random() < 0.5:
        # Full content
        return full_str[:content_start], full_str[content_start:content_end]

    # Uniform window size from 1 char to content_len
    window_size = rng.randint(1, content_len)

    # Random position within the content
    max_start = content_len - window_size
    offset = rng.randint(0, max_start)

    window_start = content_start + offset
    window_end = window_start + window_size

    return full_str[:window_start], full_str[window_start:window_end]


def _make_entry(*, entry_id, qa_type, prefix_text, selected_text, question,
                answer, label, full_text):
    """Create a single training entry in synthetic QA format."""
    return {
        "id": entry_id,
        "qa_type": qa_type,
        "response_format": "open_ended",
        "prefix_text": prefix_text,
        "selected_text": selected_text,
        "question": question,
        "answer": answer,
        # Metadata
        "label": label,
        "window_start": len(prefix_text),
        "window_end": len(prefix_text) + len(selected_text),
        "window_desc": f"{len(selected_text)} chars",
    }


def save_results(entries, n):
    """Save in the synthetic QA format."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ARTIFACTS_DIR / f"training_data_{n}.json"

    output = {
        "metadata": {
            "model": TARGET_MODEL,
            "total_entries": len(entries),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "latentqa_v2",
        },
        "entries": entries,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(entries)} entries to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20,
                        help="Number of entries (matches qa_{n}.json)")
    parser.add_argument("--tag", type=str, default=None,
                        help="QA source tag (loads qa_{tag}.json, saves training_data_{tag}.json)")
    args = parser.parse_args()

    tag = args.tag or str(args.n)

    print(f"Loading QA results (tag={tag})...")
    qa_results = load_qa_results(tag)
    print(f"Loaded {len(qa_results)} QA results")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)

    print("Rendering training entries...")
    entries = render_entries(qa_results, tokenizer)

    # Stats
    user_entries = [e for e in entries if e["qa_type"] == "latentqa_user"]
    resp_entries = [e for e in entries if e["qa_type"] == "latentqa_response"]
    print(f"  {len(user_entries)} user-turn, {len(resp_entries)} response-turn")

    selected_lens = [len(e["selected_text"]) for e in entries]
    print(f"  Selected text length: min={min(selected_lens)}, "
          f"max={max(selected_lens)}, mean={sum(selected_lens)/len(selected_lens):.0f}")

    save_results(entries, tag)


if __name__ == "__main__":
    main()
