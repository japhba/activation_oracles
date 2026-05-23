import pytest

from nl_probes.utils.dataset_utils import BinaryFeatureResult
from nl_probes.open_ended_eval.eval_runner import (
    compute_binary_yes_no_metrics,
    compute_roc_curve_data,
    score_binary_yes_no_results,
)


def _make_binary_result(ground_truth: str, yes_score: float, no_score: float) -> BinaryFeatureResult:
    """Helper to create a BinaryFeatureResult with the fields score_binary_yes_no_results needs."""
    return BinaryFeatureResult(
        feature_idx=-1,
        candidate_scores={"yes": yes_score, "no": no_score},
        candidate_token_scores={},
        argmax_token_id=0,
        argmax_token_text="",
        argmax_logit=0.0,
        prompt="",
        meta_info={
            "ground_truth": ground_truth,
            "act_key": "lora",
        },
    )


def test_compute_roc_curve_data_perfect_separation():
    roc = compute_roc_curve_data(
        labels=[1, 1, 0, 0],
        scores=[2.0, 1.0, -1.0, -2.0],
    )

    assert roc is not None
    assert roc["auc"] == pytest.approx(1.0)
    assert roc["positives"] == 2
    assert roc["negatives"] == 2


def test_score_binary_yes_no_results_skips_non_yes_no_ground_truth():
    results = [
        _make_binary_result("yes", 2.0, 0.0),
        _make_binary_result("A", 0.0, 2.0),
    ]
    metadata = [
        {"prompt_name": "binary_prompt"},
        {"prompt_name": "letter_prompt"},
    ]

    scored = score_binary_yes_no_results(results, metadata)

    assert len(scored) == 1
    assert scored[0]["ground_truth"] == "yes"
    assert scored[0]["predicted_answer"] == "yes"
    assert scored[0]["is_correct"] is True


def test_compute_binary_yes_no_metrics_adds_prompt_breakdown():
    scored_results = [
        {
            "prompt_name": "prompt_a",
            "condition": "cond_1",
            "baseline_variant": "full_context",
            "binary_label": 1,
            "ground_truth": "yes",
            "margin_yes_minus_no": 2.0,
            "predicted_answer": "yes",
            "is_correct": True,
        },
        {
            "prompt_name": "prompt_a",
            "condition": "cond_1",
            "baseline_variant": "full_context",
            "binary_label": 0,
            "ground_truth": "no",
            "margin_yes_minus_no": -1.0,
            "predicted_answer": "no",
            "is_correct": True,
        },
        {
            "prompt_name": "prompt_b",
            "condition": "cond_2",
            "baseline_variant": "selected_span",
            "binary_label": 1,
            "ground_truth": "yes",
            "margin_yes_minus_no": 0.5,
            "predicted_answer": "yes",
            "is_correct": True,
        },
        {
            "prompt_name": "prompt_b",
            "condition": "cond_2",
            "baseline_variant": "selected_span",
            "binary_label": 0,
            "ground_truth": "no",
            "margin_yes_minus_no": -0.25,
            "predicted_answer": "no",
            "is_correct": True,
        },
    ]

    metrics = compute_binary_yes_no_metrics(scored_results)

    assert metrics["accuracy_at_zero"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["prompt_prompt_a_roc_auc"] == pytest.approx(1.0)
    assert metrics["cond_cond_2_roc_auc"] == pytest.approx(1.0)
    assert metrics["variant_full_context_roc_auc"] == pytest.approx(1.0)
