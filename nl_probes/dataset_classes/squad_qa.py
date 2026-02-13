import random
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from nl_probes.dataset_classes.act_dataset_manager import ActDatasetLoader, BaseDatasetConfig, DatasetLoaderConfig
from nl_probes.dataset_classes.classification import ClassificationDatapoint, create_vector_dataset
from nl_probes.utils.common import layer_percent_to_layer, load_tokenizer, set_seed


@dataclass
class SquadQADatasetConfig(BaseDatasetConfig):
    min_end_offset: int = -3
    max_end_offset: int = -5
    max_window_size: int = 20
    min_window_size: int = 1


class SquadQADatasetLoader(ActDatasetLoader):
    def __init__(self, dataset_config: DatasetLoaderConfig, model_kwargs: dict[str, Any] | None = None, model=None):
        super().__init__(dataset_config)

        self.dataset_params: SquadQADatasetConfig = dataset_config.custom_dataset_params

        assert self.dataset_config.dataset_name == "", "Squad QA dataset name gets overridden here"
        self.dataset_config.dataset_name = "squad_qa"

        self.model_kwargs = model_kwargs
        self.model = model

        self.act_layers = [
            layer_percent_to_layer(self.dataset_config.model_name, lp)
            for lp in self.dataset_config.layer_percents
        ]

        assert self.dataset_params.min_end_offset < 0
        assert self.dataset_params.max_end_offset < 0
        assert self.dataset_params.max_end_offset <= self.dataset_params.min_end_offset

    def create_dataset(self) -> None:
        tokenizer = load_tokenizer(self.dataset_config.model_name)

        train_datapoints, test_datapoints = get_squad_datapoints(
            self.dataset_config.num_train,
            self.dataset_config.num_test,
            self.dataset_config.seed,
        )

        for split in self.dataset_config.splits:
            if split == "train":
                datapoints = train_datapoints
                save_acts = self.dataset_config.save_acts
            else:
                datapoints = test_datapoints
                save_acts = True

            data = create_vector_dataset(
                datapoints,
                tokenizer,
                self.dataset_config.model_name,
                self.dataset_config.batch_size,
                self.act_layers,
                min_end_offset=self.dataset_params.min_end_offset,
                max_end_offset=self.dataset_params.max_end_offset,
                max_window_size=self.dataset_params.max_window_size,
                min_window_size=self.dataset_params.min_window_size,
                save_acts=save_acts,
                datapoint_type=self.dataset_config.dataset_name,
                model_kwargs=self.model_kwargs,
                model=self.model,
            )

            self.save_dataset(data, split)


def get_squad_datapoints(
    train_examples: int,
    test_examples: int,
    random_seed: int,
) -> tuple[list[ClassificationDatapoint], list[ClassificationDatapoint]]:
    """Load SQuAD 2.0, filter to answerable, shuffle, split into train/test ClassificationDatapoints."""
    set_seed(random_seed)
    ds = load_dataset("parquet", data_files="hf://datasets/rajpurkar/squad_v2/squad_v2/train-00000-of-00001.parquet", split="train")
    answerable = [row for row in ds if len(row["answers"]["text"]) > 0]

    random.shuffle(answerable)

    assert len(answerable) >= train_examples + test_examples, (
        f"Need {train_examples + test_examples} but only {len(answerable)} answerable SQuAD examples"
    )

    train_rows = answerable[:train_examples]
    test_rows = answerable[len(answerable) - test_examples:] if test_examples > 0 else []

    def rows_to_datapoints(rows: list[dict]) -> list[ClassificationDatapoint]:
        return [
            ClassificationDatapoint(
                activation_prompt=row["context"],
                classification_prompt=row["question"],
                target_response=row["answers"]["text"][0],
                ds_label="squad_qa",
            )
            for row in rows
        ]

    return rows_to_datapoints(train_rows), rows_to_datapoints(test_rows)
