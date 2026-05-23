"""
End-to-end tests for the open-ended eval pipeline.

Uses Qwen3-8B with a real AO adapter to verify the full flow:
base model -> ensure_default_adapter -> load AO -> run_verbalizer.
"""

import pytest
import torch

from nl_probes.base_experiment import (
    load_oracle_adapter,
    run_verbalizer,
    run_verbalizer_binary_score,
)
from nl_probes.open_ended_eval.eval_runner import (
    build_verbalizer_eval_config,
    build_yes_no_candidate_token_groups,
    ensure_default_adapter,
    extract_yes_no,
    get_first_ao_response,
)
from nl_probes.open_ended_eval.mmlu_prediction import (
    AO_POST_ANSWER_PROMPTS,
    AO_PRE_ANSWER_PROMPTS,
    GENERATION_KWARGS,
    build_mmlu_prediction_verbalizer_prompt_infos,
    load_mmlu_prediction_dataset,
)

TEST_VERBALIZER_LORA = "adamkarvonen/checkpoints_latentqa_cls_on_policy_Qwen3-8B"
from nl_probes.utils.common import load_model, load_tokenizer


@pytest.fixture(scope="module")
def qwen3_8b_base_model():
    """Load Qwen3-8B as a plain (non-PeftModel) base model.

    This deliberately does NOT wrap in PeftModel — tests should go through
    ensure_default_adapter to replicate the real eval/spot-check flow.
    """
    if not torch.cuda.is_available():
        pytest.skip("Qwen3-8B integration test requires CUDA")

    model_name = "Qwen/Qwen3-8B"
    device = torch.device("cuda:0")
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, torch.bfloat16)
    model.eval()

    yield {
        "model_name": model_name,
        "device": device,
        "tokenizer": tokenizer,
        "model": model,
    }

    del model
    torch.cuda.empty_cache()


# Frozen expected responses for regression testing.
# Generated with TEST_VERBALIZER_LORA on first 20 MMLU entries using
# prompt "pre_will_correct", greedy decoding (temperature=0, do_sample=False),
# eval_batch_size=32.
# If the pipeline is correct, these should be reproduced exactly.
EXPECTED_MMLU_REGRESSION_RESULTS = [
    {"id": "mmlu_12479", "ao_response": "No"},
    {"id": "mmlu_9249", "ao_response": "Yes"},
    {"id": "mmlu_4831", "ao_response": "Yes"},
    {"id": "mmlu_4719", "ao_response": "No"},
    {"id": "mmlu_10814", "ao_response": "No"},
    {"id": "mmlu_12057", "ao_response": "No"},
    {"id": "mmlu_9820", "ao_response": "Yes"},
    {"id": "mmlu_11012", "ao_response": "No"},
    {"id": "mmlu_6957", "ao_response": "Yes"},
    {"id": "mmlu_10313", "ao_response": "Yes"},
    {"id": "mmlu_12325", "ao_response": "Yes"},
    {"id": "mmlu_9825", "ao_response": "Yes"},
    {"id": "mmlu_12263", "ao_response": "Yes"},
    {"id": "mmlu_13969", "ao_response": "Yes"},
    {"id": "mmlu_8358", "ao_response": "Yes"},
    {"id": "mmlu_3430", "ao_response": "Yes"},
    {"id": "mmlu_7186", "ao_response": "Yes"},
    {"id": "mmlu_6345", "ao_response": "Yes"},
    {"id": "mmlu_10899", "ao_response": "No"},
    {"id": "mmlu_5316", "ao_response": "No"},
]


@torch.no_grad()
def test_mmlu_prediction_regression(qwen3_8b_base_model):
    """Regression test: verify AO responses match frozen expected values.

    Runs the MMLU prediction eval on 20 entries with a single prompt
    (pre_will_correct) and greedy decoding, then checks that the AO
    produces exactly the same responses as when the fixture was generated.
    Catches regressions from refactoring that silently change behavior.
    """
    bundle = qwen3_8b_base_model
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    device = bundle["device"]
    model_name = bundle["model_name"]

    ensure_default_adapter(model)
    adapter_name, training_config = load_oracle_adapter(model, TEST_VERBALIZER_LORA)

    config = build_verbalizer_eval_config(
        model_name=model_name,
        training_config=training_config,
        eval_batch_size=32,
        generation_kwargs=GENERATION_KWARGS,
    )

    prompts = {
        "pre_will_correct": "Answer with 'Yes' or 'No' only. Will the model answer this question correctly?",
    }
    entries = load_mmlu_prediction_dataset(model_name, max_entries=20)
    prompt_infos, entry_metadata = build_mmlu_prediction_verbalizer_prompt_infos(
        entries, prompts, tokenizer
    )

    results = run_verbalizer(
        model=model,
        tokenizer=tokenizer,
        verbalizer_prompt_infos=prompt_infos,
        verbalizer_lora_path=adapter_name,
        target_lora_path=None,
        config=config,
        device=device,
    )

    assert len(results) == len(EXPECTED_MMLU_REGRESSION_RESULTS)

    mismatches = []
    for i, (result, meta, expected) in enumerate(
        zip(results, entry_metadata, EXPECTED_MMLU_REGRESSION_RESULTS)
    ):
        assert meta["id"] == expected["id"], (
            f"Entry order mismatch at index {i}: got {meta['id']}, expected {expected['id']}"
        )
        ao_response = get_first_ao_response(result)
        assert ao_response is not None, f"No AO response for entry {meta['id']}"

        if ao_response != expected["ao_response"]:
            mismatches.append(
                f"  {meta['id']}: got {ao_response!r}, expected {expected['ao_response']!r}"
            )

    assert not mismatches, (
        f"AO responses changed from expected values — possible regression:\n"
        + "\n".join(mismatches)
    )


@torch.no_grad()
def test_generation_vs_binary_scoring_consistency(qwen3_8b_base_model):
    """Verify that generation (greedy decode + parse) and binary logit scoring
    agree on yes/no predictions for MMLU prediction prompts.

    Runs both paths on the same inputs and checks that the argmax logit
    prediction matches the parsed generation output >= 95% of the time.
    (They can differ in rare cases where the model's top token is a yes/no
    variant not in the candidate set, or the generated text is unparseable.)
    """
    bundle = qwen3_8b_base_model
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    device = bundle["device"]
    model_name = bundle["model_name"]

    ensure_default_adapter(model)
    adapter_name, training_config = load_oracle_adapter(model, TEST_VERBALIZER_LORA)

    config = build_verbalizer_eval_config(
        model_name=model_name,
        training_config=training_config,
        eval_batch_size=8,
        generation_kwargs=GENERATION_KWARGS,
    )

    # Use only yes/no prompts (not predict_letter — binary scoring is for yes/no)
    yes_no_prompts = {
        k: v
        for k, v in {**AO_PRE_ANSWER_PROMPTS, **AO_POST_ANSWER_PROMPTS}.items()
        if k != "predict_letter"
    }

    entries = load_mmlu_prediction_dataset(model_name, max_entries=10)
    prompt_infos, entry_metadata = build_mmlu_prediction_verbalizer_prompt_infos(
        entries, yes_no_prompts, tokenizer
    )

    # Run generation path
    gen_results = run_verbalizer(
        model=model,
        tokenizer=tokenizer,
        verbalizer_prompt_infos=prompt_infos,
        verbalizer_lora_path=adapter_name,
        target_lora_path=None,
        config=config,
        device=device,
    )

    # Run binary scoring path
    candidate_token_groups = build_yes_no_candidate_token_groups(tokenizer)
    binary_results = run_verbalizer_binary_score(
        model=model,
        tokenizer=tokenizer,
        verbalizer_prompt_infos=prompt_infos,
        verbalizer_lora_path=adapter_name,
        target_lora_path=None,
        config=config,
        device=device,
        candidate_token_groups=candidate_token_groups,
    )

    assert len(gen_results) == len(binary_results) == len(prompt_infos)

    # Build reverse lookup: token_id -> yes/no label
    all_yes_token_ids = set(candidate_token_groups["yes"])
    all_no_token_ids = set(candidate_token_groups["no"])

    # Track two separate comparisons:
    # 1. Binary yes/no prediction (logsumexp over token groups) vs generation
    # 2. Argmax token (single highest logit across full vocab) vs generation
    binary_matches = 0
    binary_comparisons = 0
    binary_mismatches = []
    argmax_matches = 0
    argmax_comparisons = 0
    argmax_mismatches = []

    for i, (gen_res, bin_res, meta) in enumerate(
        zip(gen_results, binary_results, entry_metadata)
    ):
        ao_response = get_first_ao_response(gen_res)
        if ao_response is None:
            continue
        gen_prediction = extract_yes_no(ao_response)
        if gen_prediction is None:
            continue

        # Check 1: Binary yes/no prediction (logsumexp groups)
        yes_score = bin_res.candidate_scores["yes"]
        no_score = bin_res.candidate_scores["no"]
        bin_prediction = "yes" if yes_score >= no_score else "no"

        binary_comparisons += 1
        if gen_prediction == bin_prediction:
            binary_matches += 1
        else:
            binary_mismatches.append(
                f"  [{i}] prompt={meta['prompt_name']} gen={gen_prediction} "
                f"bin={bin_prediction} (yes={yes_score:.2f}, no={no_score:.2f}) "
                f"ao_text={ao_response[:80]!r}"
            )

        # Check 2: Argmax token (full vocab) vs generation
        argmax_id = bin_res.argmax_token_id
        if argmax_id in all_yes_token_ids:
            argmax_prediction = "yes"
        elif argmax_id in all_no_token_ids:
            argmax_prediction = "no"
        else:
            # Argmax token isn't a yes/no variant — skip this comparison
            # (but still interesting to log)
            argmax_mismatches.append(
                f"  [{i}] prompt={meta['prompt_name']} gen={gen_prediction} "
                f"argmax_token={bin_res.argmax_token_text!r} (id={argmax_id}) "
                f"— not a yes/no token"
            )
            continue

        argmax_comparisons += 1
        if gen_prediction == argmax_prediction:
            argmax_matches += 1
        else:
            argmax_mismatches.append(
                f"  [{i}] prompt={meta['prompt_name']} gen={gen_prediction} "
                f"argmax={argmax_prediction} argmax_token={bin_res.argmax_token_text!r} "
                f"ao_text={ao_response[:80]!r}"
            )

    assert binary_comparisons > 0, "No parseable generation results to compare"
    binary_rate = binary_matches / binary_comparisons

    print(f"\n--- Binary (logsumexp) vs generation ---")
    print(f"Agreement: {binary_matches}/{binary_comparisons} = {binary_rate:.1%}")
    if binary_mismatches:
        print("Mismatches:")
        for m in binary_mismatches:
            print(m)

    assert binary_rate >= 0.95, (
        f"Binary scoring and generation disagree too often: "
        f"{binary_matches}/{binary_comparisons} = {binary_rate:.1%}. "
        f"Expected >= 95%.\nMismatches:\n" + "\n".join(binary_mismatches)
    )

    print(f"\n--- Argmax token vs generation ---")
    if argmax_comparisons > 0:
        argmax_rate = argmax_matches / argmax_comparisons
        print(f"Agreement: {argmax_matches}/{argmax_comparisons} = {argmax_rate:.1%}")
    else:
        argmax_rate = 0.0
        print("No argmax comparisons (argmax was never a yes/no token)")
    if argmax_mismatches:
        print("Mismatches / non-yes-no argmax tokens:")
        for m in argmax_mismatches:
            print(m)

    assert argmax_comparisons > 0, (
        "Argmax token was never a yes/no variant — can't validate argmax consistency"
    )
    assert argmax_rate >= 0.98, (
        f"Argmax token and generation disagree too often: "
        f"{argmax_matches}/{argmax_comparisons} = {argmax_rate:.1%}. "
        f"Expected >= 98%.\nMismatches:\n" + "\n".join(argmax_mismatches)
    )
