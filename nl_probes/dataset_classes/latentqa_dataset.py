import random
from dataclasses import asdict, dataclass, field
from typing import Generator, Literal

import torch
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoTokenizer

import nl_probes.dataset_classes.misc.latentqa_loader as latentqa_loader
from nl_probes.dataset_classes.act_dataset_manager import (
    ActDatasetLoader,
    BaseDatasetConfig,
    DatasetLoaderConfig,
)
from nl_probes.utils.common import layer_percent_to_layer, load_tokenizer
from nl_probes.utils.dataset_utils import (
    TrainingDataPoint,
    create_training_datapoint,
)


@dataclass
class LatentQADatasetConfig(BaseDatasetConfig):
    max_window_size: int = 3
    min_window_size: int = 1
    min_end_offset: int = -1
    max_end_offset: int = -10
    position_types: list[str] = field(default_factory=lambda: ["all", "window"])


class LatentQADatasetLoader(ActDatasetLoader):
    def __init__(
        self,
        dataset_config: DatasetLoaderConfig,
    ):
        super().__init__(dataset_config)
        assert self.dataset_config.dataset_name == "", (
            f"{self.dataset_config.dataset_name}, Dataset name gets overridden here"
        )

        self.dataset_config.dataset_name = "latentqa"

        self.dataset_params: LatentQADatasetConfig = dataset_config.custom_dataset_params

        assert self.dataset_config.splits == ["train"], "Past lens dataset only supports train split right now"
        assert self.dataset_config.num_test == 0, "Past lens dataset only supports train split right now"

        if self.dataset_config.num_train < self.dataset_config.batch_size:
            raise ValueError(
                f"num_train {self.dataset_config.num_train} must be greater than or equal to batch_size {self.dataset_config.batch_size}"
            )

    def create_dataset(self) -> None:
        tokenizer = load_tokenizer(self.dataset_config.model_name)

        assert self.dataset_config.layer_combinations, "layer_combinations must be non-empty"
        act_layer_combinations = [
            [layer_percent_to_layer(self.dataset_config.model_name, layer_percent) for layer_percent in layer_combo]
            for layer_combo in self.dataset_config.layer_combinations
        ]

        paths = latentqa_loader.DataPaths(
            system=None,
            stimulus_completion="data_pipelines/latentqa_datasets/train/stimulus_completion.json",
            stimulus="data_pipelines/latentqa_datasets/train/stimulus.json",
            control="data_pipelines/latentqa_datasets/train/control.json",
            qa="data_pipelines/latentqa_datasets/train/qa.json",
        )
        ds = latentqa_loader.load_latentqa_dataset(
            paths,
            filter_prefixes=[],
            train_percent=1.0,
            add_thought_tokens=False,
            seed=self.dataset_config.seed,
        )

        self.ds = ds

        training_data = []

        for dp in tqdm(ds, desc="Creating latentqa dataset"):
            act_layers = random.choice(act_layer_combinations)
            training_data.append(create_latentqa_training_datapoint(dp, tokenizer, act_layers, self.dataset_params))

        self.save_dataset(training_data, "train")


class Item(BaseModel):
    role: str
    content: str


class LatentQADatapoint(BaseModel):
    label: str
    source: Literal["stimulus", "stimulus_completion", "control"]
    read_prompt: list[Item]
    dialog: list[Item]
    mask_type: str


def create_latentqa_training_datapoint(
    datapoint_dict: dict, tokenizer: AutoTokenizer, act_layers: list[int], dataset_params: LatentQADatasetConfig
) -> TrainingDataPoint:
    masked_turn_count = {"stimulus_completion": 2, "stimulus": 2, "control": 0}

    datapoint = LatentQADatapoint.model_validate(datapoint_dict, strict=True)

    num_masked = masked_turn_count[datapoint.source]

    masked_turns = datapoint.read_prompt[:num_masked]

    if num_masked > 0:
        masked_str = tokenizer.apply_chat_template(masked_turns, tokenize=False, enable_thinking=False)
        masked_tokens = tokenizer(masked_str, return_tensors=None, add_special_tokens=False, padding=False)["input_ids"]
    else:
        masked_tokens = []

    if datapoint.source == "stimulus_completion":
        add_generation_prompt = False
    else:
        add_generation_prompt = True

    full_read_str = tokenizer.apply_chat_template(
        datapoint.read_prompt, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
    )

    context_input_ids = tokenizer(full_read_str, return_tensors=None, add_special_tokens=False, padding=False)[
        "input_ids"
    ]

    context_positions = list(range(len(context_input_ids)))
    context_positions = context_positions[len(masked_tokens) :]

    positions = random.choice(dataset_params.position_types)

    if positions == "window":
        window_size = random.randint(dataset_params.min_window_size, dataset_params.max_window_size)

        if datapoint.source == "control":
            end_offset = random.randint(dataset_params.max_end_offset, dataset_params.min_end_offset)
            assert end_offset < 0, "end_offset must be negative"

            if abs(end_offset) > len(context_positions):
                end_offset = -len(context_positions) + 1

            window_size = min(window_size, (len(context_positions)) + end_offset)

            window_start = end_offset - window_size
            context_positions = context_positions[window_start:end_offset]
        else:
            window_size = min(window_size, len(context_positions))
            max_start = len(context_positions) - window_size
            window_start = random.randint(0, max_start)
            window_end = window_start + window_size
            context_positions = context_positions[window_start:window_end]

    training_datapoint = create_training_datapoint(
        datapoint_type=f"latentqa_{datapoint.source}",
        prompt=datapoint.dialog[0].content,
        target_response=datapoint.dialog[1].content,
        layers=act_layers,
        num_positions=len(context_positions),
        tokenizer=tokenizer,
        acts_BD=None,
        feature_idx=-1,
        context_input_ids=context_input_ids,
        context_positions=context_positions,
    )

    return training_datapoint


if __name__ == "__main__":
    model_name = "Qwen/Qwen3-8B"
    batch_size = 16
    config = DatasetLoaderConfig(
        LatentQADatasetConfig(),
        100_000,
        0,
        ["train"],
        model_name,
        [[50]],
        False,
        batch_size=batch_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(config)
    # %%
    dataset = LatentQADatasetLoader(config)

    # %%

    dataset.create_dataset()

    print(dataset.ds[0])
    # %%
    # for datapoint in tqdm(dataset.ds):
    # training_datapoint = create_latentqa_training_datapoint(datapoint, tokenizer, [18])
    # %%
    # print(tokenizer.decode(datapoint.context_input_ids))
    # print(f"\n\nCTX:{tokenizer.decode(datapoint.context_input_ids[len(datapoint.context_positions) :])}")
