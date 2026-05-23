from pathlib import Path

import pytest

from nl_probes.base_experiment import tokenize_chat_messages
from nl_probes.open_ended_eval.model_understanding import (
    TEXT_BASELINE_SYSTEM_PROMPT,
    TEXT_BASELINE_SUFFIX_GUIDANCE_PREFIX,
    build_context_messages,
    build_model_understanding_ao_prompt_infos,
    build_model_understanding_text_baseline_inputs,
    load_model_understanding_eval_entries,
)
from nl_probes.utils.common import load_tokenizer

TEST_RUN_DIR = "data_pipelines/model_understanding/runs/test_mixed_v2"
TEST_MODEL_NAME = "Qwen/Qwen3-14B"


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(TEST_MODEL_NAME)


def test_model_understanding_eval_entry_count():
    entries = load_model_understanding_eval_entries(TEST_RUN_DIR)
    assert len(entries) == 41


def test_text_baseline_prompt_matches_sft_prefix(tokenizer):
    entries = load_model_understanding_eval_entries(TEST_RUN_DIR, max_entries=5)
    baseline_inputs, _metadata = build_model_understanding_text_baseline_inputs(
        entries,
        tokenizer,
        prompt_style="plain",
    )

    for entry, baseline_input in zip(entries, baseline_inputs, strict=True):
        context_messages = build_context_messages(entry, question_mode="second_person")
        expected_ids = tokenize_chat_messages(
            tokenizer,
            context_messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert baseline_input.prompt_text == tokenizer.decode(expected_ids, skip_special_tokens=False)


def test_text_baseline_heldout_fewshot_prefix_is_present(tokenizer):
    entries = load_model_understanding_eval_entries(TEST_RUN_DIR, max_entries=1)
    baseline_inputs, _metadata = build_model_understanding_text_baseline_inputs(
        entries,
        tokenizer,
        prompt_style="heldout_fewshot",
    )

    prompt_text = baseline_inputs[0].prompt_text
    assert TEXT_BASELINE_SYSTEM_PROMPT in prompt_text
    assert "Example of a good introspection answer." in prompt_text
    assert "travel/flight scenarios for passport" in prompt_text
    assert "Gunlance Player Slander" in prompt_text
    assert entries[0].second_person_question in prompt_text


def test_text_baseline_heldout_fewshot_suffix_preserves_prefix(tokenizer):
    entries = load_model_understanding_eval_entries(TEST_RUN_DIR, max_entries=1)
    baseline_inputs, _metadata = build_model_understanding_text_baseline_inputs(
        entries,
        tokenizer,
        prompt_style="suffix_heldout_fewshot",
    )

    prompt_text = baseline_inputs[0].prompt_text
    prefix = (
        "<|im_start|>user\nHow fast can you generate text in words per minute (WPM)?<|im_end|>\n"
    )
    assert prompt_text.startswith(prefix)
    assert TEXT_BASELINE_SYSTEM_PROMPT not in prompt_text
    assert TEXT_BASELINE_SUFFIX_GUIDANCE_PREFIX in prompt_text
    assert "Example 1" in prompt_text
    assert "travel/flight scenarios for passport" in prompt_text
    assert "Gunlance Player Slander" in prompt_text
    question_start = prompt_text.find(entries[0].second_person_question)
    guidance_start = prompt_text.find(TEXT_BASELINE_SUFFIX_GUIDANCE_PREFIX)
    assert question_start != -1
    assert guidance_start > question_start


def test_ao_prompt_infos_window_full_vs_truncated(tokenizer):
    entries = load_model_understanding_eval_entries(TEST_RUN_DIR)
    prompt_infos, metadata = build_model_understanding_ao_prompt_infos(
        entries,
        tokenizer,
        max_context_tokens=1000,
    )

    assert any(meta.used_full_context for meta in metadata)
    assert any(not meta.used_full_context for meta in metadata)

    for entry, prompt_info, meta in zip(entries, prompt_infos, metadata, strict=True):
        context_messages = build_context_messages(entry, question_mode="third_person")
        expected_full_ids = tokenize_chat_messages(
            tokenizer,
            context_messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert prompt_info.context_token_ids == expected_full_ids
        assert meta.full_context_token_count == len(expected_full_ids)

        if meta.used_full_context:
            assert prompt_info.positions == list(range(len(expected_full_ids)))
            assert meta.activation_window_mode == "full_context"
        else:
            expected_positions = list(range(len(expected_full_ids) - 1000, len(expected_full_ids)))
            assert prompt_info.positions == expected_positions
            assert meta.selected_context_token_count == 1000
            assert meta.activation_window_mode == "last_1000"
