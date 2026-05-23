from dataclasses import dataclass
from pathlib import Path

import torch

from nl_probes.dataset_classes.act_dataset_manager import (
    ActDatasetLoader,
    BaseDatasetConfig,
    DatasetLoaderConfig,
)
from nl_probes.utils.dataset_utils import TrainingDataPoint


@dataclass
class PrebuiltPTDatasetConfig(BaseDatasetConfig):
    data_path: str
    component_name: str


class PrebuiltPTDatasetLoader(ActDatasetLoader):
    def __init__(self, dataset_config: DatasetLoaderConfig):
        super().__init__(dataset_config)
        assert self.dataset_config.dataset_name == "", (
            f"{self.dataset_config.dataset_name}, Dataset name gets overridden here"
        )
        self.dataset_config.dataset_name = "prebuilt_pt"
        self.dataset_params: PrebuiltPTDatasetConfig = dataset_config.custom_dataset_params

        assert self.dataset_config.splits == ["train"], "Prebuilt PT datasets only support the train split"
        assert self.dataset_config.num_test == 0, "Prebuilt PT datasets only support the train split"

    def create_dataset(self) -> None:
        raise ValueError(
            "Prebuilt PT datasets must already exist on disk; create_dataset() should never be called"
        )

    def ensure_dataset_exists(self, split: str) -> None:
        assert split == "train", f"Invalid split for prebuilt PT dataset: {split}"
        path = Path(self.dataset_params.data_path)
        assert path.exists(), f"Missing prebuilt PT dataset: {path}"

    def load_dataset(self, split: str) -> list[TrainingDataPoint]:
        assert split == "train", f"Invalid split for prebuilt PT dataset: {split}"
        path = Path(self.dataset_params.data_path)
        assert path.exists(), f"Missing prebuilt PT dataset: {path}"

        saved_object = torch.load(path, weights_only=False)
        data_dicts = saved_object["data"]
        data = [TrainingDataPoint(**d) for d in data_dicts]
        print(f"Loaded {len(data)} datapoints from {path}")
        return data
