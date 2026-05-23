# nl_tests/test_peft_adapters_disabled.py
"""
Tiny sanity test:
- Add a LoRA adapter with non-zero init (init_lora_weights=False)
- Verify enable_adapters()/disable_adapters() flip logits
- Disabled ≈ base, restored ≈ enabled
"""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from nl_probes.base_experiment import VerbalizerEvalConfig, collect_target_activations, tokenize_chat_messages
from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.utils.activation_collection import (
    materialize_block_into_batches,
    estimate_activation_collection_token_stats,
)
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint


@pytest.mark.parametrize("model_name", ["facebook/opt-125m"])
@torch.no_grad()
def test_enable_disable_adapters_simple(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()

    inputs = tok("hello world", return_tensors="pt")
    base_logits = model(**inputs).logits.clone()

    # Non-zero LoRA init so it actually does something without training
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules="all-linear",  # simple and model-agnostic for OPT
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=False,  # <- key simplification
    )
    model = get_peft_model(model, lora_cfg)

    # ON
    logits_on = model(**inputs).logits
    # model.disable_adapters()

    # OFF
    with model.disable_adapter():
        logits_off = model(**inputs).logits
    # model.enable_adapters()

    # Back ON
    logits_restored = model(**inputs).logits

    # Assertions
    assert (logits_on - logits_off).abs().max().item() > 1e-6, "Adapters did not change logits"
    assert torch.allclose(logits_off, base_logits, rtol=1e-5, atol=1e-7), "Disabled adapters did not match base"
    assert torch.allclose(logits_on, logits_restored, rtol=1e-5, atol=1e-7), "State not restored after re-enabling"


@pytest.fixture(scope="module")
def qwen3_06b_peft_bundle():
    if not torch.cuda.is_available():
        pytest.skip("Qwen3-0.6B integration test requires CUDA")

    model_name = "Qwen/Qwen3-0.6B"
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    # Base-model path used by standard collection/eval code.
    base_submodule = get_hf_submodule(model, layer=1)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.eval()
    peft_model.set_adapter("default")

    yield {
        "model_name": model_name,
        "device": device,
        "tokenizer": tokenizer,
        "base_submodule": base_submodule,
        "peft_model": peft_model,
    }

    del peft_model
    torch.cuda.empty_cache()


@torch.no_grad()
def test_qwen3_06b_get_hf_submodule_base_and_peft(qwen3_06b_peft_bundle):
    base_submodule = qwen3_06b_peft_bundle["base_submodule"]
    peft_model = qwen3_06b_peft_bundle["peft_model"]

    peft_submodule_default = get_hf_submodule(peft_model, layer=1)
    peft_submodule_use_lora = get_hf_submodule(peft_model, layer=1, use_lora=True)

    assert type(base_submodule) is type(peft_submodule_default)
    assert type(base_submodule) is type(peft_submodule_use_lora)


@torch.no_grad()
def test_qwen3_06b_collect_target_activations_peft_path(qwen3_06b_peft_bundle):
    model_name = qwen3_06b_peft_bundle["model_name"]
    device = qwen3_06b_peft_bundle["device"]
    tokenizer = qwen3_06b_peft_bundle["tokenizer"]
    peft_model = qwen3_06b_peft_bundle["peft_model"]

    config = VerbalizerEvalConfig(
        model_name=model_name,
        activation_input_types=["orig", "lora", "diff"],
        layer_combinations=[[25]],
        selected_layer_combination=[25],
        eval_batch_size=1,
    )

    input_ids = tokenize_chat_messages(
        tokenizer=tokenizer,
        messages=[{"role": "user", "content": "Hello there."}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs_BL = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.bool, device=device),
    }

    # This path previously failed when adapter toggling used incompatible methods.
    act_types = collect_target_activations(
        model=peft_model,
        inputs_BL=inputs_BL,
        config=config,
        target_lora_path="default",
    )

    assert set(act_types.keys()) == {"orig", "lora", "diff"}
    assert config.selected_act_layers[0] in act_types["orig"]
    assert config.selected_act_layers[0] in act_types["lora"]
    assert config.selected_act_layers[0] in act_types["diff"]


def _build_materialization_test_datapoints(tokenizer: AutoTokenizer) -> list[TrainingDataPoint]:
    layers = [1, 5]
    num_positions = 4
    repeated_word_counts = [8, 36, 12, 44, 16, 52, 20, 60, 24, 68, 28, 76, 32, 84, 40, 92]
    training_data: list[TrainingDataPoint] = []

    for idx, repeated_word_count in enumerate(repeated_word_counts):
        context_text = " ".join(["alpha"] * repeated_word_count)
        context_messages = [{"role": "user", "content": f"Context {idx}: {context_text}"}]
        context_input_ids = tokenizer.apply_chat_template(
            context_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
            padding=False,
            enable_thinking=False,
        )
        if not isinstance(context_input_ids, list):
            raise TypeError("Expected list of token ids from tokenizer")

        context_positions = list(range(len(context_input_ids) - num_positions, len(context_input_ids)))
        prompt = (
            f"Hidden context {idx} has {repeated_word_count} repeated words. "
            "Answer with the word yes."
        )
        training_data.append(
            create_training_datapoint(
                datapoint_type="test_materialization",
                prompt=prompt,
                target_response="yes",
                layers=layers,
                num_positions=num_positions,
                tokenizer=tokenizer,
                acts_BD=None,
                feature_idx=idx,
                context_input_ids=context_input_ids,
                context_positions=context_positions,
            )
        )

    return training_data


def _flatten_batches(batches: list[list[TrainingDataPoint]]) -> list[TrainingDataPoint]:
    return [datapoint for batch in batches for datapoint in batch]


def _make_grouping_test_datapoint(feature_idx: int, context_length: int) -> TrainingDataPoint:
    return TrainingDataPoint(
        datapoint_type="grouping_test",
        input_ids=[11, 12],
        labels=[-100, 12],
        layers=[1],
        steering_vectors=None,
        positions=[0],
        feature_idx=feature_idx,
        target_output="x",
        context_input_ids=list(range(context_length)),
        context_positions=[0],
        ds_label=None,
    )


def test_estimate_activation_collection_token_stats_matches_fixed_sorted_chunking():
    training_data = [
        _make_grouping_test_datapoint(feature_idx=idx, context_length=context_length)
        for idx, context_length in enumerate([90, 88, 60, 30, 29, 28])
    ]
    cfg = SelfInterpTrainingConfig(
        model_name="facebook/opt-125m",
        layer_combinations=[[1]],
        act_layer_combinations=[[1]],
        train_batch_size=2,
        train_batches_per_materialization_block=3,
        save_dir="unused_test_dir",
    )

    reference_stats = estimate_activation_collection_token_stats(
        cfg=cfg,
        training_data=training_data,
        reference=True,
    )
    efficient_stats = estimate_activation_collection_token_stats(
        cfg=cfg,
        training_data=training_data,
        reference=False,
    )

    assert reference_stats.num_materialization_batches == 1
    assert reference_stats.actual_tokens == 325
    assert reference_stats.padded_tokens == 540
    assert efficient_stats.num_materialization_batches == 3
    assert efficient_stats.actual_tokens == 325
    assert efficient_stats.padded_tokens == 358


@torch.no_grad()
def test_qwen3_06b_iter_materialized_efficient_matches_reference(qwen3_06b_peft_bundle):
    model_name = qwen3_06b_peft_bundle["model_name"]
    tokenizer = qwen3_06b_peft_bundle["tokenizer"]
    peft_model = qwen3_06b_peft_bundle["peft_model"]
    training_data = _build_materialization_test_datapoints(tokenizer)

    cfg = SelfInterpTrainingConfig(
        model_name=model_name,
        layer_combinations=[[1, 5]],
        act_layer_combinations=[[1, 5]],
        train_batch_size=2,
        train_batches_per_materialization_block=8,
        save_dir="unused_test_dir",
    )

    efficient_stats = estimate_activation_collection_token_stats(
        cfg=cfg,
        training_data=training_data,
        reference=False,
    )

    materialization_block_size = cfg.train_batch_size * cfg.train_batches_per_materialization_block
    assert efficient_stats.num_materialization_batches > len(training_data) // materialization_block_size
    assert efficient_stats.num_materialization_batches == len(training_data) // cfg.train_batch_size

    # Iterate over blocks, matching the new calling convention
    reference_data = [dp.model_copy(deep=True) for dp in training_data]
    efficient_data = [dp.model_copy(deep=True) for dp in training_data]

    reference_batches: list[list[TrainingDataPoint]] = []
    efficient_batches: list[list[TrainingDataPoint]] = []
    for block_start in range(0, len(training_data), materialization_block_size):
        ref_block = reference_data[block_start : block_start + materialization_block_size]
        eff_block = efficient_data[block_start : block_start + materialization_block_size]
        reference_batches.extend(materialize_block_into_batches(
            cfg=cfg, block=ref_block, tokenizer=tokenizer, model=peft_model, reference=True,
        ))
        efficient_batches.extend(materialize_block_into_batches(
            cfg=cfg, block=eff_block, tokenizer=tokenizer, model=peft_model,
        ))

    assert len(reference_batches) == len(efficient_batches)

    reference_datapoints = _flatten_batches(reference_batches)
    efficient_datapoints = _flatten_batches(efficient_batches)

    assert [dp.feature_idx for dp in reference_datapoints] == [dp.feature_idx for dp in training_data]
    assert [dp.feature_idx for dp in efficient_datapoints] == [dp.feature_idx for dp in training_data]

    for reference_dp, efficient_dp in zip(reference_datapoints, efficient_datapoints, strict=True):
        assert reference_dp.feature_idx == efficient_dp.feature_idx
        assert reference_dp.positions == efficient_dp.positions
        assert reference_dp.context_positions == efficient_dp.context_positions
        assert reference_dp.steering_vectors is not None
        assert efficient_dp.steering_vectors is not None

        reference_vectors = reference_dp.steering_vectors.float().cpu()
        efficient_vectors = efficient_dp.steering_vectors.float().cpu()

        assert reference_vectors.shape == efficient_vectors.shape
        relative_l2_error = torch.linalg.vector_norm(reference_vectors - efficient_vectors) / torch.linalg.vector_norm(
            reference_vectors
        )
        cosine_similarity = torch.nn.functional.cosine_similarity(
            reference_vectors.reshape(1, -1),
            efficient_vectors.reshape(1, -1),
        ).item()

        assert relative_l2_error.item() < 0.011
        assert cosine_similarity > 0.9999
