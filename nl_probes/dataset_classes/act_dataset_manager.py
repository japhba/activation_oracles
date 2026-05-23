from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch

from nl_probes.utils.dataset_utils import TrainingDataPoint

if TYPE_CHECKING:
    from nl_probes.configs.sft_config import SelfInterpTrainingConfig


@dataclass
class BaseDatasetConfig:
    pass


@dataclass
class DatasetLoaderConfig:
    custom_dataset_params: BaseDatasetConfig
    num_train: int
    num_test: int
    splits: list[str]
    model_name: str
    layer_combinations: list[list[int]]
    save_acts: bool
    batch_size: int
    dataset_name: str = ""
    dataset_folder: str = "sft_training_data"
    seed: int = 42


def _config_hash(cfg: DatasetLoaderConfig, split: str, exclude: tuple[str, ...] = ("batch_size",)) -> str:
    """
    Stable short hash over the full config + split.
    Excludes path-like fields so moving folders does not change the filename.
    """

    def _strip(obj):
        if isinstance(obj, dict):
            return {k: _strip(v) for k, v in obj.items() if k not in exclude}
        if isinstance(obj, list):
            return [_strip(v) for v in obj]
        return obj

    payload = {"config": _strip(asdict(cfg)), "split": split}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2s(blob, digest_size=6).hexdigest()  # 12 hex chars


class ActDatasetLoader:
    def __init__(
        self,
        dataset_config: DatasetLoaderConfig,
    ):
        self.valid_splits = set(["train", "test"])
        self.dataset_config = dataset_config

        for split in self.dataset_config.splits:
            assert split in self.valid_splits, f"Invalid split: {split}"

    def create_dataset(self) -> None:
        """
        Note: Will always make all split(s) at the same time.
        This is so we ensure that train / test splits have no overlap.
        """
        raise NotImplementedError

    def ensure_dataset_exists(self, split: Literal["train", "test"]) -> None:
        """Create the dataset file on disk if it doesn't already exist.

        Unlike load_dataset, this does NOT load the data into memory — useful
        for phase-one runs (--gen-only) where you just want to materialize
        files without paying the deserialization cost.
        """
        assert split in self.valid_splits, f"Invalid split: {split}"
        dataset_name = self.get_dataset_filename(split)
        filepath = os.path.join(self.dataset_config.dataset_folder, dataset_name)
        if not os.path.exists(filepath):
            os.makedirs(self.dataset_config.dataset_folder, exist_ok=True)
            self.create_dataset()

    def load_dataset(
        self,
        split: Literal["train", "test"],
    ) -> list[TrainingDataPoint]:
        assert split in self.valid_splits, f"Invalid split: {split}"

        dataset_name = self.get_dataset_filename(split)
        filepath = os.path.join(self.dataset_config.dataset_folder, dataset_name)
        if not os.path.exists(filepath):
            os.makedirs(self.dataset_config.dataset_folder, exist_ok=True)
            self.create_dataset()

        saved_object = torch.load(filepath, weights_only=False)
        data_dicts = saved_object["data"]
        data = [TrainingDataPoint(**d) for d in data_dicts]

        print(f"Loaded {len(data)} datapoints from {filepath}")
        return data

    def save_dataset(self, data: list[TrainingDataPoint], split: Literal["train", "test"]) -> None:
        data_filename = self.get_dataset_filename(split)
        data_path = os.path.join(self.dataset_config.dataset_folder, data_filename)
        torch.save(
            {
                "config": asdict(self.dataset_config),
                "data": [dp.model_dump() for dp in data],
            },
            data_path,
        )
        print(f"Saved {len(data)} {split} datapoints to {data_path}")

    def get_dataset_filename(self, split: Literal["train", "test"]) -> str:
        num_datapoints = self.dataset_config.num_train if split == "train" else self.dataset_config.num_test

        model_str = self.dataset_config.model_name.split("/")[-1]

        config_hash = _config_hash(self.dataset_config, split)

        filename = f"{self.dataset_config.dataset_name}_model_{model_str}_n_{num_datapoints}_save_acts_{self.dataset_config.save_acts}_{split}_{config_hash}"
        filename = filename.replace("/", "_").replace(".", "_").replace(" ", "_")
        return f"{filename}.pt"


# ---------------------------------------------------------------------------
# Registry: dataset_name -> (CustomConfigClass, LoaderClass)
# ---------------------------------------------------------------------------

def _load_registry_entry(key: str) -> tuple[type[BaseDatasetConfig], type[ActDatasetLoader]]:
    """Per-key lazy import. Some loader modules touch data_pipelines paths at
    import time, so we only import the one we actually need."""
    if key == "codex_old_past_lens":
        from nl_probes.dataset_classes.codex_old_past_lens_dataset import (
            CodexOldPastLensDatasetConfig,
            CodexOldPastLensDatasetLoader,
        )
        return CodexOldPastLensDatasetConfig, CodexOldPastLensDatasetLoader
    if key == "past_lens":
        from nl_probes.dataset_classes.past_lens_dataset import PastLensDatasetConfig, PastLensDatasetLoader
        return PastLensDatasetConfig, PastLensDatasetLoader
    if key == "latentqa":
        from nl_probes.dataset_classes.latentqa_dataset import LatentQADatasetConfig, LatentQADatasetLoader
        return LatentQADatasetConfig, LatentQADatasetLoader
    if key == "classification":
        from nl_probes.dataset_classes.classification import ClassificationDatasetConfig, ClassificationDatasetLoader
        return ClassificationDatasetConfig, ClassificationDatasetLoader
    if key == "prebuilt_pt":
        from nl_probes.dataset_classes.prebuilt_pt_dataset import PrebuiltPTDatasetConfig, PrebuiltPTDatasetLoader
        return PrebuiltPTDatasetConfig, PrebuiltPTDatasetLoader
    if key == "synthetic_qa":
        from nl_probes.dataset_classes.synthetic_qa_dataset import SyntheticQADatasetConfig, SyntheticQADatasetLoader
        return SyntheticQADatasetConfig, SyntheticQADatasetLoader
    if key == "cot_oracle_convqa":
        from nl_probes.dataset_classes.cot_oracle_dataset import CotOracleDatasetConfig, CotOracleDatasetLoader
        return CotOracleDatasetConfig, CotOracleDatasetLoader
    if key == "cot_oracle_local_eval":
        from nl_probes.dataset_classes.cot_oracle_local_eval_dataset import (
            CotOracleLocalEvalConfig,
            CotOracleLocalEvalLoader,
        )
        return CotOracleLocalEvalConfig, CotOracleLocalEvalLoader
    if key == "aobench_recog":
        from nl_probes.dataset_classes.aobench_recog_dataset import (
            AObenchRecogConfig,
            AObenchRecogLoader,
        )
        return AObenchRecogConfig, AObenchRecogLoader
    raise KeyError(key)


_REGISTRY_KEYS = (
    "codex_old_past_lens",
    "past_lens",
    "latentqa",
    "classification",
    "prebuilt_pt",
    "synthetic_qa",
    "cot_oracle_convqa",
    "cot_oracle_local_eval",
    "aobench_recog",
)


def _get_dataset_registry() -> dict[str, tuple[type[BaseDatasetConfig], type[ActDatasetLoader]]]:
    """Backwards-compat shim: imports every loader module. Prefer
    _load_registry_entry for callers that already know the key."""
    return {k: _load_registry_entry(k) for k in _REGISTRY_KEYS}


def _deserialize_loader(raw_config: dict[str, Any]) -> ActDatasetLoader:
    """Reconstruct a single dataset loader from a serialized DatasetLoaderConfig dict."""
    dataset_name = raw_config["dataset_name"]

    # classification datasets are named "classification_sst2", "classification_ner", etc.
    if dataset_name.startswith("classification_"):
        registry_key = "classification"
    else:
        registry_key = dataset_name

    if registry_key not in _REGISTRY_KEYS:
        raise ValueError(
            f"Unknown dataset_name '{dataset_name}'. "
            f"Known types: {list(_REGISTRY_KEYS)} (+ classification_* variants)"
        )

    config_cls, loader_cls = _load_registry_entry(registry_key)

    custom_params = config_cls(**raw_config["custom_dataset_params"])

    # Loaders assert dataset_name == "" in __init__ and then set it themselves,
    # so we must pass "" here (they'll re-derive the same name).
    loader_config = DatasetLoaderConfig(
        custom_dataset_params=custom_params,
        num_train=raw_config["num_train"],
        num_test=raw_config["num_test"],
        splits=raw_config["splits"],
        model_name=raw_config["model_name"],
        layer_combinations=raw_config["layer_combinations"],
        save_acts=raw_config["save_acts"],
        batch_size=raw_config["batch_size"],
        dataset_name="",
        dataset_folder=raw_config.get("dataset_folder", "sft_training_data"),
        seed=raw_config.get("seed", 42),
    )

    return loader_cls(dataset_config=loader_config)


def build_loaders_from_config(cfg: SelfInterpTrainingConfig) -> list[ActDatasetLoader]:
    """Reconstruct dataset loaders from a saved/loaded SelfInterpTrainingConfig."""
    if not cfg.dataset_configs:
        raise ValueError("Config has no dataset_configs — cannot reconstruct loaders")

    return [_deserialize_loader(raw) for raw in cfg.dataset_configs]


def build_loaders_from_saved_configs(raw_configs: list[dict[str, Any]]) -> list[ActDatasetLoader]:
    if not raw_configs:
        return []
    return [_deserialize_loader(raw) for raw in raw_configs]
