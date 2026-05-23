import pytest
import torch
from transformers import AutoTokenizer

import nl_probes.utils.dataset_utils as dataset_utils
from nl_probes.utils.common import load_tokenizer, layer_percent_to_layer


@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen3-8B",
        "google/gemma-2-9b-it",
        "meta-llama/Llama-3.1-8B-Instruct",
    ],
)
def test_positions(model_name):
    tokenizer = load_tokenizer(model_name)
    num_positions = 10
    layers = [layer_percent_to_layer(model_name, 50)]
    prompt = "This is a test prompt"

    datapoint = dataset_utils.create_training_datapoint(
        "test",
        prompt,
        "test output",
        layers,
        num_positions,
        tokenizer,
        torch.randn(len(layers) * num_positions, 64),
        -1,
    )

    assert len(datapoint.positions) == num_positions * len(layers)
    assert datapoint.positions[-1] - datapoint.positions[0] >= num_positions - 1

    special_token_id = tokenizer.encode(dataset_utils.SPECIAL_TOKEN, add_special_tokens=False)
    assert len(special_token_id) == 1
    special_token_id = special_token_id[0]
    found_positions = [i for i, t in enumerate(datapoint.input_ids) if t == special_token_id]
    assert found_positions == datapoint.positions

    block = datapoint.positions[:num_positions]
    assert block[-1] - block[0] == num_positions - 1
    assert all(block[i + 1] == block[i] + 1 for i in range(len(block) - 1))

    print(f"{model_name} single-layer -> {datapoint}")


@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen3-8B",
        "google/gemma-2-9b-it",
        "meta-llama/Llama-3.1-8B-Instruct",
    ],
)
def test_positions_multi_layer(model_name):
    tokenizer = load_tokenizer(model_name)
    num_positions = 3
    layers = [layer_percent_to_layer(model_name, p) for p in [25, 50, 75]]
    prompt = "This is a test prompt"

    datapoint = dataset_utils.create_training_datapoint(
        "test",
        prompt,
        "test output",
        layers,
        num_positions,
        tokenizer,
        torch.randn(len(layers) * num_positions, 64),
        -1,
    )

    assert len(datapoint.positions) == num_positions * len(layers)
    assert datapoint.positions == sorted(datapoint.positions)

    for layer_idx in range(len(layers)):
        block = datapoint.positions[layer_idx * num_positions : (layer_idx + 1) * num_positions]
        assert block[-1] - block[0] == num_positions - 1
        assert all(block[i + 1] == block[i] + 1 for i in range(len(block) - 1))

    special_token_id = tokenizer.encode(dataset_utils.SPECIAL_TOKEN, add_special_tokens=False)
    assert len(special_token_id) == 1
    special_token_id = special_token_id[0]
    found_positions = [i for i, t in enumerate(datapoint.input_ids) if t == special_token_id]
    assert found_positions == datapoint.positions

    print(f"{model_name} multi-layer -> {datapoint}")
