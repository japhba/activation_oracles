import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

from nl_probes.base_experiment import tokenize_chat_messages
from nl_probes.dataset_classes.act_dataset_manager import (
    ActDatasetLoader,
    BaseDatasetConfig,
    DatasetLoaderConfig,
)
from nl_probes.utils.common import layer_percent_to_layer, load_tokenizer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint


@dataclass
class SyntheticQADatasetConfig(BaseDatasetConfig):
    data_path: str = "data_pipelines/training_data/artifacts/training_data_50000.json"


class SyntheticQADatasetLoader(ActDatasetLoader):
    def __init__(self, dataset_config: DatasetLoaderConfig):
        super().__init__(dataset_config)
        assert self.dataset_config.dataset_name == "", (
            f"{self.dataset_config.dataset_name}, Dataset name gets overridden here"
        )
        self.dataset_config.dataset_name = "synthetic_qa"
        self.dataset_params: SyntheticQADatasetConfig = dataset_config.custom_dataset_params

        assert self.dataset_config.splits == ["train"], "SyntheticQA only supports train split right now"
        assert self.dataset_config.num_test == 0, "SyntheticQA only supports train split right now"

    def create_dataset(self) -> None:
        tokenizer = load_tokenizer(self.dataset_config.model_name)

        act_layer_combinations = [
            [layer_percent_to_layer(self.dataset_config.model_name, lp) for lp in combo]
            for combo in self.dataset_config.layer_combinations
        ]

        data_path = Path(self.dataset_params.data_path)
        with open(data_path) as f:
            raw = json.load(f)

        entries = raw["entries"]

        # Subsample if num_train < total entries
        if self.dataset_config.num_train < len(entries):
            rng = random.Random(self.dataset_config.seed)
            entries = rng.sample(entries, self.dataset_config.num_train)

        training_data = []
        skipped = 0

        for entry in tqdm(entries, desc="Creating synthetic QA dataset"):
            act_layers = random.choice(act_layer_combinations)
            dp = create_synthetic_qa_datapoint(entry, tokenizer, act_layers)
            if dp is None:
                skipped += 1
                continue
            training_data.append(dp)

        if skipped:
            print(f"Skipped {skipped}/{len(entries)} entries (empty question/answer or no selected positions)")

        self.save_dataset(training_data, "train")


def compute_context_and_positions(
    entry: dict,
    tokenizer: AutoTokenizer,
) -> tuple[list[int], list[int]] | None:
    """Compute context_input_ids and context_positions for a training data entry.

    prefix_text and selected_text are pre-rendered strings (already include chat
    template formatting). We tokenize their concatenation and find the token
    indices corresponding to the selected_text portion.
    """
    prefix_text = entry["prefix_text"]
    selected_text = entry["selected_text"]

    full_text = prefix_text + selected_text
    context_input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    n_prefix_tokens = len(tokenizer(prefix_text, add_special_tokens=False)["input_ids"])
    context_positions = list(range(n_prefix_tokens, len(context_input_ids)))

    if len(context_positions) < 1:
        return None

    return context_input_ids, context_positions


def create_synthetic_qa_datapoint(
    entry: dict,
    tokenizer: AutoTokenizer,
    act_layers: list[int],
) -> TrainingDataPoint | None:
    """Convert a single training data entry into a TrainingDataPoint."""
    question = entry.get("question", "")
    answer = entry.get("answer", "")

    if not question or not answer:
        return None

    result = compute_context_and_positions(entry, tokenizer)
    if result is None:
        return None
    context_input_ids, context_positions = result

    return create_training_datapoint(
        datapoint_type=f"synthetic_qa_{entry.get('qa_type', 'unknown')}",
        prompt=question,
        target_response=answer,
        layers=act_layers,
        num_positions=len(context_positions),
        tokenizer=tokenizer,
        acts_BD=None,
        feature_idx=-1,
        context_input_ids=context_input_ids,
        context_positions=context_positions,
        meta_info={
            "entry_id": entry.get("id", ""),
            "qa_type": entry.get("qa_type", ""),
            "response_format": entry.get("response_format", ""),
        },
    )


if __name__ == "__main__":
    """Quick test: create dataset from the 20-entry test file and inspect."""
    import sys

    model_name = "Qwen/Qwen3-8B"
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data_pipelines/training_data/artifacts/training_data_20.json"

    tokenizer = load_tokenizer(model_name)

    with open(data_path) as f:
        raw = json.load(f)

    entries = raw["entries"]
    print(f"Loaded {len(entries)} entries from {data_path}")

    act_layers = [layer_percent_to_layer(model_name, 50)]
    print(f"Using layers: {act_layers}")

    results = []
    for entry in entries:
        dp = create_synthetic_qa_datapoint(entry, tokenizer, act_layers)
        if dp is None:
            print(f"  SKIPPED {entry.get('id')}: empty question or answer")
            continue
        results.append((entry, dp))

    print(f"\nCreated {len(results)} datapoints, skipped {len(entries) - len(results)}")

    # Inspect each datapoint
    for entry, dp in results:
        print(f"\n{'='*80}")
        print(f"Entry: {entry['id']} | type: {entry['qa_type']} | format: {entry['response_format']}")
        print(f"  Context tokens: {len(dp.context_input_ids)}")
        print(f"  Context positions: {len(dp.context_positions)} tokens (indices {dp.context_positions[0]}-{dp.context_positions[-1]})")
        print(f"  Input tokens (AO prompt+response): {len(dp.input_ids)}")
        print(f"  Layers: {dp.layers}")

        # Decode the selected tokens to verify they match selected_text
        selected_decoded = tokenizer.decode(
            [dp.context_input_ids[p] for p in dp.context_positions],
            skip_special_tokens=False,
        )
        print(f"  Selected text (original):  {entry['selected_text']!r}")
        print(f"  Selected text (from tokens): {selected_decoded!r}")

        # Show the question and answer
        print(f"  Question: {dp.target_output[:100]}...")
        print(f"  Prompt (first 200 chars): {tokenizer.decode(dp.input_ids[:50])!r}")
