import argparse
import json
from pathlib import Path

from nl_probes.utils.common import load_tokenizer


def _markdown_block(title: str, content: str) -> str:
    return f"### {title}\n\n```text\n{content}\n```\n"


def _write_baseline_samples(path: Path, sample_count: int, tokenizer) -> Path:
    del tokenizer
    payload = json.loads(path.read_text())
    entry_metadata = payload["entry_metadata"]
    baseline_inputs = payload["baseline_inputs"]
    raw_results = payload["raw_results"]
    scored_results = payload["scored_results"]

    prompt_id_to_index = {
        meta["prompt_id"]: idx for idx, meta in enumerate(entry_metadata)
    }

    lines: list[str] = []
    lines.append(f"# Samples for {path.name}")
    lines.append("")
    lines.append(f"Showing {min(sample_count, len(scored_results))} scored examples.")
    lines.append("")

    for scored in scored_results[:sample_count]:
        idx = prompt_id_to_index[scored["prompt_id"]]
        lines.append(f"## {scored['prompt_id']}")
        lines.append("")
        lines.append(f"- specificity: {scored['specificity']}")
        lines.append(f"- correctness: {scored['correctness']}")
        lines.append(f"- question_mode: {scored['question_mode']}")
        lines.append(f"- answer_mode: {scored['answer_mode']}")
        lines.append("")
        lines.append(_markdown_block("Exact Prompt", baseline_inputs[idx]["prompt_text"]))
        lines.append(_markdown_block("Model Response", raw_results[idx]["response_text"]))
        lines.append(_markdown_block("Ground Truth", scored["reference_answer"]))
        lines.append(_markdown_block("Judge Reasoning", scored["reasoning"]))

    output_path = path.with_name(f"{path.stem}_samples.md")
    output_path.write_text("\n".join(lines).rstrip() + "\n")
    return output_path


def _write_ao_samples(path: Path, sample_count: int, tokenizer) -> Path:
    payload = json.loads(path.read_text())
    verbalizer_results = payload["verbalizer_results"]
    entry_metadata = payload["entry_metadata"]
    scored_results = payload["scored_results"]

    verbalizer_by_prompt_id = {
        meta["prompt_id"]: result
        for meta, result in zip(entry_metadata, verbalizer_results, strict=True)
    }

    lines: list[str] = []
    lines.append(f"# Samples for {path.name}")
    lines.append("")
    lines.append(f"Showing {min(sample_count, len(scored_results))} scored examples.")
    lines.append("")

    for scored in scored_results[:sample_count]:
        verbalizer_result = verbalizer_by_prompt_id[scored["prompt_id"]]
        context_prompt = tokenizer.decode(
            verbalizer_result["context_token_ids"],
            skip_special_tokens=False,
        )
        lines.append(f"## {scored['prompt_id']}")
        lines.append("")
        lines.append(f"- specificity: {scored['specificity']}")
        lines.append(f"- correctness: {scored['correctness']}")
        lines.append(f"- question_mode: {scored['question_mode']}")
        lines.append(f"- answer_mode: {scored['answer_mode']}")
        lines.append(f"- activation_window_mode: {scored['activation_window_mode']}")
        lines.append("")
        lines.append(_markdown_block("Exact Context Prompt", context_prompt))
        lines.append(_markdown_block("Verbalizer Prompt", verbalizer_result["verbalizer_prompt"]))
        lines.append(_markdown_block("Model Response", scored["response_text"]))
        lines.append(_markdown_block("Ground Truth", scored["reference_answer"]))
        lines.append(_markdown_block("Judge Reasoning", scored["reasoning"]))

    output_path = path.with_name(f"{path.stem}_samples.md")
    output_path.write_text("\n".join(lines).rstrip() + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("json_paths", nargs="+")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_name)

    for json_path_str in args.json_paths:
        path = Path(json_path_str)
        payload = json.loads(path.read_text())
        if "baseline_inputs" in payload:
            output_path = _write_baseline_samples(path, args.sample_count, tokenizer)
        elif "verbalizer_results" in payload:
            output_path = _write_ao_samples(path, args.sample_count, tokenizer)
        else:
            raise ValueError(f"Unsupported model-understanding result file: {path}")
        print(output_path)


if __name__ == "__main__":
    main()
