"""COT-oracle ConvQA dataset loader.

For each materialized sample we draw a row from
`cds-jb/cot-oracle-convqa-chunked` (with replacement to reach `num_train`),
then sample activation positions on the row's `cot_prefix`:

  1. Start from all token positions in the activation source.
  2. Apply the third_party/cot-oracle stochastic sampler:
     sparse last-token slices half the time, log-uniform random subsets up to
     `stochastic_max_k` the other half, with first/last positions included.

The model is shown `prompt` (with the standard introspection prefix prepended,
one block of sampled special tokens per layer) and trained to produce `target_response`.
The activations come from the cot_prefix forward pass at `context_positions`.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset
from tqdm.auto import tqdm

from nl_probes.dataset_classes.act_dataset_manager import (
    ActDatasetLoader,
    BaseDatasetConfig,
    DatasetLoaderConfig,
)
from nl_probes.dataset_classes.position_sampling import sample_cot_oracle_token_positions
from nl_probes.utils.common import layer_percent_to_layer, load_tokenizer
from nl_probes.utils.dataset_utils import (
    TrainingDataPoint,
    create_training_datapoint,
)


@dataclass
class CotOracleDatasetConfig(BaseDatasetConfig):
    """Custom params for the cot-oracle convqa-style dataset family.

    Default fields target `cds-jb/cot-oracle-convqa-chunked` and
    `cds-jb/fineweb-oracle-convqa-chunked`, which split each CoT into
    `cot_prefix`/`cot_suffix`. For OOD eval datasets that ship the full
    CoT in a single column (e.g. `cds-jb/cot-oracle-eval-*`), set
    `cot_prefix_field="cot_text"`.
    """

    hf_dataset_repo: str = "cds-jb/cot-oracle-convqa-chunked"
    hf_split: str = "train"
    cot_prefix_field: str = "cot_prefix"  # name of the activation-source column
    stochastic_max_k: int = 100  # cot-oracle stochastic sampler cap
    max_cot_prefix_tokens: int = 2048  # cap on cot_prefix tokenization length
    target_field: str = "target_response"  # which column to use as target


class CotOracleDatasetLoader(ActDatasetLoader):
    DATASET_NAME = "cot_oracle_convqa"

    def __init__(self, dataset_config: DatasetLoaderConfig):
        super().__init__(dataset_config)
        assert self.dataset_config.dataset_name == "", "Dataset name gets overridden here"
        self.dataset_config.dataset_name = self.DATASET_NAME

        self.dataset_params: CotOracleDatasetConfig = dataset_config.custom_dataset_params

        assert self.dataset_config.save_acts is False, "save_acts must be False (on-the-fly extraction)"

    def create_dataset(self) -> None:
        os.makedirs(self.dataset_config.dataset_folder, exist_ok=True)
        tokenizer = load_tokenizer(self.dataset_config.model_name)

        rng = random.Random(self.dataset_config.seed)

        act_layer_combinations = [
            [layer_percent_to_layer(self.dataset_config.model_name, p) for p in combo]
            for combo in self.dataset_config.layer_combinations
        ]

        for split in self.dataset_config.splits:
            target_n = self.dataset_config.num_train if split == "train" else self.dataset_config.num_test
            if target_n == 0:
                continue
            # The OUTPUT split name (`split`) is what build_validation_datasets
            # asks for via load_dataset("train"). The HF SOURCE split is
            # determined by self.dataset_params.hf_split — set hf_split="test"
            # in a validation config to pull the held-out HF test rows.
            data = self._build_split(
                tokenizer=tokenizer,
                rng=rng,
                act_layer_combinations=act_layer_combinations,
                split_to_load=self.dataset_params.hf_split,
                target_n=target_n,
            )
            self.save_dataset(data, split)  # type: ignore[arg-type]

    def _build_split(
        self,
        tokenizer,
        rng: random.Random,
        act_layer_combinations: list[list[int]],
        split_to_load: str,
        target_n: int,
    ) -> list[TrainingDataPoint]:
        ds = load_dataset(self.dataset_params.hf_dataset_repo, split=split_to_load)

        out: list[TrainingDataPoint] = []
        pbar = tqdm(total=target_n, desc=f"cot_oracle/{split_to_load}")
        ds_size = len(ds)
        # Pre-shuffle a permutation that we walk through with replacement when needed.
        order = list(range(ds_size))
        rng.shuffle(order)
        cursor = 0
        attempts = 0
        max_attempts = target_n * 10 + 1000

        while len(out) < target_n and attempts < max_attempts:
            attempts += 1
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            row = ds[int(order[cursor])]
            cursor += 1

            cot_prefix = row[self.dataset_params.cot_prefix_field]
            prompt = row["prompt"]
            target = row[self.dataset_params.target_field]
            if not cot_prefix or not prompt or not target:
                continue

            cot_prefix_ids = tokenizer(
                cot_prefix,
                add_special_tokens=False,
                truncation=True,
                max_length=self.dataset_params.max_cot_prefix_tokens,
                return_tensors=None,
            )["input_ids"]
            if len(cot_prefix_ids) == 0:
                continue

            # Per-row resampled (layers, positions): each draw is independent,
            # so a row reused across "epochs" gets a fresh sample each time.
            layers = rng.choice(act_layer_combinations)
            positions = sample_cot_oracle_token_positions(
                len(cot_prefix_ids), rng, max_k=self.dataset_params.stochastic_max_k,
            )
            n_actual = len(positions)

            meta_info: dict[str, Any] = {
                "cot_id": row.get("cot_id"),
                "source": row.get("source"),
                "split_index": row.get("split_index"),
                "num_sentences": row.get("num_sentences"),
                "bb_correct": row.get("bb_correct"),
                "n_positions_sampled": n_actual,
                "position_sampler": "cot_oracle_stochastic",
                "stochastic_max_k": self.dataset_params.stochastic_max_k,
                "cot_prefix_len_tokens": len(cot_prefix_ids),
                "context_positions_first": positions[0],
                "context_positions_last": positions[-1],
            }

            dp = create_training_datapoint(
                datapoint_type=self.DATASET_NAME,
                prompt=prompt,
                target_response=target,
                layers=layers,
                num_positions=n_actual,
                tokenizer=tokenizer,
                acts_BD=None,
                feature_idx=-1,
                context_input_ids=cot_prefix_ids,
                context_positions=positions,
                ds_label=None,
                meta_info=meta_info,
            )
            out.append(dp)
            pbar.update(1)

        pbar.close()
        if len(out) < target_n:
            raise RuntimeError(
                f"cot_oracle: only collected {len(out)}/{target_n} after {attempts} attempts"
            )
        return out
