import argparse
import html
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nl_probes.utils.dataset_utils import TrainingDataPoint


def find_latest_past_lens_file(dataset_folder: Path) -> Path:
    candidates = sorted(
        dataset_folder.glob("past_lens_model_*_save_acts_False_train_*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    assert len(candidates) > 0, f"No past_lens train files in {dataset_folder}"
    return candidates[0]


def decode_tokens(tokenizer: AutoTokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def decode_token(tokenizer: AutoTokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False).replace("\n", "\\n")


def get_vllm_prompt_text(dp: TrainingDataPoint, tokenizer: AutoTokenizer) -> str:
    assert dp.context_input_ids is not None
    assert "direction" in dp.meta_info
    assert dp.meta_info["direction"] == "future_generated"
    assert "vllm_prompt_len" in dp.meta_info
    vllm_prompt_len = dp.meta_info["vllm_prompt_len"]
    assert isinstance(vllm_prompt_len, int)
    return tokenizer.decode(dp.context_input_ids[:vllm_prompt_len], skip_special_tokens=False)


def prompt_ends_on_assistant_turn(prompt_text: str) -> bool:
    assistant_generation_prompt = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    if not prompt_text.endswith(assistant_generation_prompt):
        return False

    prefix = prompt_text[: -len(assistant_generation_prompt)]
    last_user = prefix.rfind("<|im_start|>user\n")
    last_assistant = prefix.rfind("<|im_start|>assistant\n")
    last_system = prefix.rfind("<|im_start|>system\n")

    return last_assistant > max(last_user, last_system)


def infer_direction(prompt_text: str) -> str:
    if "previous" in prompt_text:
        return "past"
    if "next" in prompt_text and "generated" in prompt_text:
        return "future_generated"
    return "unknown"


def extract_task_instruction(prompt_text: str) -> str:
    needles = [
        "Can you predict the previous",
        "Can you predict the next",
    ]
    for needle in needles:
        start = prompt_text.find(needle)
        if start >= 0:
            end = prompt_text.find("<|im_end|>", start)
            if end < 0:
                end = len(prompt_text)
            return prompt_text[start:end].strip()
    return prompt_text.strip()


def render_token_table(
    tokenizer: AutoTokenizer,
    context_ids: list[int],
    act_start: int,
    act_end: int,
    context_window: int,
) -> list[str]:
    view_start = max(0, act_start - context_window)
    view_end = min(len(context_ids), act_end + context_window + 1)
    lines: list[str] = []
    lines.append(
        f"Token Table Around Activation Window: showing context[{view_start}:{view_end}] "
        f"(window is [{act_start}:{act_end}])"
    )
    lines.append("  idx    tok_id   mark  token")

    for idx in range(view_start, view_end):
        mark = "."
        in_window = act_start <= idx <= act_end
        if in_window:
            mark = "A"
        if idx == act_start:
            mark = "S"
        if idx == act_end:
            mark = "E"
        if act_start == act_end and idx == act_start:
            mark = "SE"

        token_str = decode_token(tokenizer, context_ids[idx])
        lines.append(f"  {idx:5d}  {context_ids[idx]:7d}   {mark:>2}   {token_str}")

    lines.append("  legend: S=start of activation window, E=end, A=inside window")
    return lines


def render_target_token_table(
    tokenizer: AutoTokenizer,
    target_token_ids: list[int],
    conceptual_start_idx: int,
) -> list[str]:
    lines: list[str] = []
    lines.append(
        f"Target Token Table: {len(target_token_ids)} tokens starting at conceptual index {conceptual_start_idx}"
    )
    lines.append("  idx    tok_id   token")
    for offset, token_id in enumerate(target_token_ids):
        idx = conceptual_start_idx + offset
        token_str = decode_token(tokenizer, token_id)
        lines.append(f"  {idx:5d}  {token_id:7d}   {token_str}")
    return lines


def build_context_table_rows(
    tokenizer: AutoTokenizer,
    context_ids: list[int],
    context_positions: list[int],
    max_rows: int,
) -> list[str]:
    pos_set = set(context_positions)
    rows = []

    for idx, token_id in enumerate(context_ids[:max_rows]):
        token_str = html.escape(tokenizer.decode([token_id], skip_special_tokens=False))
        marker = "*" if idx in pos_set else ""
        row_class = ' class="selected"' if idx in pos_set else ""
        rows.append(
            f"<tr{row_class}><td>{idx}</td><td>{token_id}</td><td>{marker}</td><td><code>{token_str}</code></td></tr>"
        )

    if len(context_ids) > max_rows:
        rows.append(
            f"<tr><td colspan='4'><em>... truncated ({len(context_ids) - max_rows} more context tokens)</em></td></tr>"
        )

    return rows


def summarize_datapoint(
    dp: TrainingDataPoint, tokenizer: AutoTokenizer, idx: int, max_context_rows: int
) -> tuple[str, str]:
    first_target_idx = next((i for i, x in enumerate(dp.labels) if x != -100), len(dp.labels))

    prompt_ids = dp.input_ids[:first_target_idx]
    target_ids = dp.input_ids[first_target_idx:]

    prompt_text = decode_tokens(tokenizer, prompt_ids)
    target_text = decode_tokens(tokenizer, target_ids)

    context_ids = dp.context_input_ids
    context_positions = dp.context_positions

    assert context_ids is not None
    assert context_positions is not None

    for pos in context_positions:
        assert 0 <= pos < len(context_ids), (
            f"Datapoint {idx}: context position {pos} out of range for context length {len(context_ids)}"
        )

    direction = "unknown"
    if "previous" in prompt_text:
        direction = "past"
    if "next" in prompt_text and "generated" in prompt_text:
        direction = "future_generated"

    console = []
    console.append(f"Sample {idx}")
    console.append(f"  direction: {direction}")
    console.append(f"  layers: {dp.layers}")
    console.append(f"  num_positions: {len(context_positions)}")
    console.append(f"  input_len: {len(dp.input_ids)}")
    console.append(f"  prompt_len: {len(prompt_ids)}")
    console.append(f"  target_len: {len(target_ids)}")
    console.append(f"  selected_context_positions: {context_positions}")
    prompt_preview = prompt_text[:300].replace(chr(10), " ")
    target_preview = target_text[:300].replace(chr(10), " ")
    console.append(f"  prompt_text: {prompt_preview}" + ("..." if len(prompt_text) > 300 else ""))
    console.append(f"  target_text: {target_preview}" + ("..." if len(target_text) > 300 else ""))

    context_rows = "\n".join(build_context_table_rows(tokenizer, context_ids, context_positions, max_context_rows))

    card = f"""
    <section class="card">
      <h2>Sample {idx}</h2>
      <p><strong>direction:</strong> {html.escape(direction)}</p>
      <p><strong>layers:</strong> {html.escape(str(dp.layers))}</p>
      <p><strong>num_positions:</strong> {len(context_positions)}</p>
      <p><strong>input_len:</strong> {len(dp.input_ids)} | <strong>prompt_len:</strong> {len(prompt_ids)} | <strong>target_len:</strong> {len(target_ids)}</p>
      <p><strong>selected_context_positions:</strong> {html.escape(str(context_positions))}</p>
      <h3>Prompt Text</h3>
      <pre>{html.escape(prompt_text)}</pre>
      <h3>Target Text (decoded from target token IDs)</h3>
      <pre>{html.escape(target_text)}</pre>
      <h3>Context Tokens</h3>
      <p>Rows with <strong>*</strong> are selected activation positions.</p>
      <table>
        <thead><tr><th>idx</th><th>token_id</th><th>sel</th><th>decoded token</th></tr></thead>
        <tbody>
          {context_rows}
        </tbody>
      </table>
    </section>
    """

    return "\n".join(console), card


def build_text_report(
    dp: TrainingDataPoint,
    tokenizer: AutoTokenizer,
    idx: int,
    context_window: int,
) -> str:
    first_target_idx = next((i for i, x in enumerate(dp.labels) if x != -100), len(dp.labels))
    prompt_ids = dp.input_ids[:first_target_idx]
    target_ids = dp.input_ids[first_target_idx:]

    prompt_text = decode_tokens(tokenizer, prompt_ids)
    target_text = decode_tokens(tokenizer, target_ids)
    full_sample_text = decode_tokens(tokenizer, dp.input_ids)

    context_ids = dp.context_input_ids
    context_positions = dp.context_positions

    assert context_ids is not None
    assert context_positions is not None
    assert len(dp.meta_info) > 0, f"Sample {idx}: expected meta_info to be populated"
    assert "direction" in dp.meta_info, f"Sample {idx}: meta_info missing 'direction'"

    direction = dp.meta_info["direction"]
    task_instruction = extract_task_instruction(prompt_text)
    act_start = min(context_positions)
    act_end = max(context_positions)
    assert act_end - act_start + 1 == len(context_positions), (
        f"Sample {idx}: context_positions are expected to be contiguous, got {context_positions}"
    )
    assert context_positions == list(range(act_start, act_end + 1)), (
        f"Sample {idx}: context_positions are expected to be sorted contiguous indices, got {context_positions}"
    )

    context_before_window = context_ids[max(0, act_start - context_window) : act_start]
    act_window_ids = context_ids[act_start : act_end + 1]
    target_token_ids = target_ids

    lines: list[str] = []
    lines.append("#" * 120)
    lines.append(f"SAMPLE {idx}")
    lines.append("#" * 120)
    lines.append(f"Direction: {direction}")
    lines.append(f"Datapoint Type: {dp.datapoint_type}")
    lines.append(f"Layers: {dp.layers}")
    lines.append(f"Activation Positions: {len(context_positions)}")
    lines.append(f"Activation Window: [{act_start}:{act_end}]")
    lines.append(f"Context Length: {len(context_ids)}")
    lines.append(f"Instruction Prompt Length: {len(prompt_ids)}")
    lines.append(f"Target Length: {len(target_ids)}")
    lines.append(f"Sample Source: {dp.meta_info['sample_source']}")
    lines.append(f"System Prompt Injected: {dp.meta_info['system_prompt_injected']}")
    lines.append(
        f"Layout: context[0:{act_end + 1}] contains activation window; target starts at conceptual index {act_end + 1}"
    )
    lines.append("")
    lines.append("METADATA")
    lines.append(str(dict(dp.meta_info)))
    lines.append("")
    lines.append("TASK PROMPT (training instruction)")
    lines.append(task_instruction)
    lines.append("")
    lines.append("ENTIRE SAMPLE (decoded from input_ids)")
    lines.append(full_sample_text)
    lines.append("")
    lines.append("TARGET TOKENS (what model should output)")
    lines.append(f"target_token_ids: {target_token_ids}")
    lines.append(target_text)
    lines.append("")
    lines.append("FULL CONTEXT TEXT (decoded from context_input_ids)")
    lines.append(tokenizer.decode(context_ids, skip_special_tokens=False))
    lines.append("")
    lines.append(f"CONTEXT BEFORE WINDOW (last {context_window} tokens before activation window)")
    lines.append(tokenizer.decode(context_before_window, skip_special_tokens=False))
    lines.append("")
    lines.append("ACTIVATION WINDOW TEXT")
    lines.append(tokenizer.decode(act_window_ids, skip_special_tokens=False))
    lines.append(f"activation_window_token_ids: {act_window_ids}")
    lines.append("")
    if direction == "future_generated":
        assert "vllm_prompt_len" in dp.meta_info, f"Sample {idx}: future sample missing 'vllm_prompt_len'"
        assert "vllm_generated_len" in dp.meta_info, f"Sample {idx}: future sample missing 'vllm_generated_len'"
        assert "vllm_generated_token_ids" in dp.meta_info, (
            f"Sample {idx}: future sample missing 'vllm_generated_token_ids'"
        )
        assert "vllm_generated_text" in dp.meta_info, f"Sample {idx}: future sample missing 'vllm_generated_text'"
        assert "vllm_finish_reason" in dp.meta_info, f"Sample {idx}: future sample missing 'vllm_finish_reason'"
        vllm_prompt_len = dp.meta_info["vllm_prompt_len"]
        vllm_generated_len = dp.meta_info["vllm_generated_len"]
        vllm_generated_token_ids = dp.meta_info["vllm_generated_token_ids"]
        vllm_generated_text = dp.meta_info["vllm_generated_text"]
        vllm_finish_reason = dp.meta_info["vllm_finish_reason"]
        assert isinstance(vllm_prompt_len, int), f"Sample {idx}: vllm_prompt_len must be int"
        assert isinstance(vllm_generated_len, int), f"Sample {idx}: vllm_generated_len must be int"
        assert isinstance(vllm_generated_token_ids, list), f"Sample {idx}: vllm_generated_token_ids must be list"
        assert isinstance(vllm_generated_text, str), f"Sample {idx}: vllm_generated_text must be str"
        assert isinstance(vllm_finish_reason, str), f"Sample {idx}: vllm_finish_reason must be str"
        assert 0 <= vllm_prompt_len <= len(context_ids), (
            f"Sample {idx}: vllm_prompt_len={vllm_prompt_len} out of bounds for context_len={len(context_ids)}"
        )
        vllm_prompt_text = tokenizer.decode(context_ids[:vllm_prompt_len], skip_special_tokens=False)
        lines.append("VLLM PROMPT TEXT (context_input_ids[:vllm_prompt_len])")
        lines.append(vllm_prompt_text)
        lines.append(f"prompt_ends_on_assistant_turn: {prompt_ends_on_assistant_turn(vllm_prompt_text)}")
        lines.append("")
        lines.append("VLLM GENERATED CONTEXT TEXT (context_input_ids[vllm_prompt_len:])")
        lines.append(tokenizer.decode(context_ids[vllm_prompt_len:], skip_special_tokens=False))
        lines.append("")
        lines.append("VLLM FULL GENERATED RESPONSE (all generated tokens)")
        lines.append(f"vllm_finish_reason: {vllm_finish_reason}")
        lines.append(f"vllm_generated_len: {vllm_generated_len}")
        lines.append(f"vllm_generated_token_ids: {vllm_generated_token_ids}")
        lines.append(vllm_generated_text)
    lines.append("")
    lines.extend(render_token_table(tokenizer, context_ids, act_start, act_end, context_window))
    lines.append("")
    lines.extend(render_target_token_table(tokenizer, target_token_ids, dp.meta_info["target_start_idx"]))

    return "\n".join(lines)


def matches_filters(dp: TrainingDataPoint, tokenizer: AutoTokenizer, args: argparse.Namespace) -> bool:
    meta = dp.meta_info
    assert "direction" in meta

    if args.direction is not None and meta["direction"] != args.direction:
        return False
    if args.sample_source is not None and meta["sample_source"] != args.sample_source:
        return False
    if args.k_tokens is not None and meta["k_tokens"] != args.k_tokens:
        return False
    if args.k_acts is not None and meta["k_acts"] != args.k_acts:
        return False
    if args.require_first_generated:
        if meta["direction"] != "future_generated":
            return False
        if meta["target_start_idx"] != meta["vllm_prompt_len"]:
            return False
    if args.require_prompt_ends_on_assistant:
        if meta["direction"] != "future_generated":
            return False
        prompt_text = get_vllm_prompt_text(dp, tokenizer)
        if not prompt_ends_on_assistant_turn(prompt_text):
            return False

    return True


def build_html(model_name: str, dataset_path: Path, sample_cards: list[str]) -> str:
    joined = "\n".join(sample_cards)
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Past Lens On-Policy Inspection</title>
  <style>
    body {{ font-family: ui-monospace, Menlo, Consolas, monospace; margin: 24px; background: #f8f9fb; color: #111; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ margin-bottom: 24px; color: #333; }}
    .card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
    pre {{ background: #f3f4f6; padding: 10px; overflow-x: auto; border-radius: 6px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #ddd; text-align: left; padding: 4px 6px; vertical-align: top; }}
    tr.selected {{ background: #fff4cc; }}
    code {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>Past Lens On-Policy Inspection</h1>
  <div class="meta">
    <div><strong>model:</strong> {html.escape(model_name)}</div>
    <div><strong>dataset:</strong> {html.escape(str(dataset_path))}</div>
  </div>
  {joined}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and visualize past_lens on-policy dataset samples.")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Path to a saved past_lens train .pt file.")
    parser.add_argument("--dataset-folder", type=Path, default=Path("sft_training_data/dry_run_on_policy"))
    parser.add_argument("--model-name", type=str, default=None, help="Override tokenizer model name.")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direction", type=str, choices=["past", "future_generated"], default=None)
    parser.add_argument("--sample-source", type=str, choices=["pretrain", "chat"], default=None)
    parser.add_argument("--k-tokens", type=int, default=None)
    parser.add_argument("--k-acts", type=int, default=None)
    parser.add_argument("--require-first-generated", action="store_true")
    parser.add_argument("--require-prompt-ends-on-assistant", action="store_true")
    parser.add_argument("--max-context-rows", type=int, default=180)
    parser.add_argument("--context-window", type=int, default=20)
    parser.add_argument(
        "--txt-out",
        type=Path,
        default=Path("sft_training_data/dry_run_on_policy/past_lens_inspection_latest.txt"),
    )
    parser.add_argument("--html-out", type=Path, default=None)
    parser.add_argument("--print-summaries", action="store_true")
    args = parser.parse_args()

    dataset_path = (
        args.dataset_path if args.dataset_path is not None else find_latest_past_lens_file(args.dataset_folder)
    )
    assert dataset_path.exists(), f"Dataset file not found: {dataset_path}"

    saved = torch.load(dataset_path)
    data_dicts = saved["data"]
    config = saved["config"]

    datapoints = [TrainingDataPoint(**d) for d in data_dicts]
    assert len(datapoints) > 0, "Dataset is empty"

    model_name = args.model_name if args.model_name is not None else config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    filtered_indices = [idx for idx, dp in enumerate(datapoints) if matches_filters(dp, tokenizer, args)]
    assert len(filtered_indices) > 0, "No datapoints matched the requested filters"

    n = min(args.num_samples, len(filtered_indices))
    random.seed(args.seed)
    chosen_indices = sorted(random.sample(filtered_indices, n))

    console_chunks = []
    text_chunks = []
    cards = []
    for idx in chosen_indices:
        console_text, card = summarize_datapoint(datapoints[idx], tokenizer, idx, args.max_context_rows)
        console_chunks.append(console_text)
        text_chunks.append(build_text_report(datapoints[idx], tokenizer, idx, args.context_window))
        cards.append(card)

    if args.print_summaries:
        print("\n\n".join(console_chunks))

    args.txt_out.parent.mkdir(parents=True, exist_ok=True)
    header_lines = [
        "Past Lens On-Policy Inspection Report",
        f"dataset_path: {dataset_path}",
        f"model_name: {model_name}",
        f"total_datapoints: {len(datapoints)}",
        f"matching_datapoints: {len(filtered_indices)}",
        f"sampled_indices: {chosen_indices}",
        (
            "filters: "
            f"direction={args.direction}, "
            f"sample_source={args.sample_source}, "
            f"k_tokens={args.k_tokens}, "
            f"k_acts={args.k_acts}, "
            f"require_first_generated={args.require_first_generated}, "
            f"require_prompt_ends_on_assistant={args.require_prompt_ends_on_assistant}"
        ),
        "",
    ]
    args.txt_out.write_text("\n".join(header_lines) + "\n\n".join(text_chunks))
    print(f"Wrote TXT report to: {args.txt_out}")

    if args.html_out is not None:
        html_doc = build_html(model_name, dataset_path, cards)
        args.html_out.parent.mkdir(parents=True, exist_ok=True)
        args.html_out.write_text(html_doc)
        print(f"Wrote HTML report to: {args.html_out}")


if __name__ == "__main__":
    main()
