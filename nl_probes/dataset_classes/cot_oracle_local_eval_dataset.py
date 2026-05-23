"""Local-JSON eval loader for AO-style evals from the upstream cot-oracle repo.

Each upstream eval ships as a JSON list/dict with rows like:

    {
      "clean_prompt": "...the question...",
      "test_prompt":  "...same question with optional nudge...",
      "correct_answer": "...the gold answer...",
      "metadata": {
         "cot_text"         : "<full CoT>",          # atypical_answer_mcq
         "representative_cot": "...",                 # decorative_cot
         "qwen3_8b_test_response": "...<think>...",   # rot13_reconstruction
         "spliced_cot_text" : "...",                  # sentence_insertion
         "donor_sentence"   : "...",                  # sentence_insertion (target)
         ...
      }
    }

The activation source (the CoT text) lives in metadata under different keys
depending on the eval. We let the config specify the dotted key path
(e.g. "metadata.cot_text"). Same for the question prompt and target answer.

The loader produces TrainingDataPoints in the same shape as
`cot_oracle_dataset.py`, so the existing MCQ/ablation/rephrase eval code works
unchanged on these.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any

from tqdm.auto import tqdm

from nl_probes.dataset_classes.act_dataset_manager import (
    ActDatasetLoader,
    BaseDatasetConfig,
    DatasetLoaderConfig,
)
from nl_probes.dataset_classes.position_sampling import sample_cot_oracle_token_positions
from nl_probes.utils.common import layer_percent_to_layer, load_tokenizer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint


@dataclass
class CotOracleLocalEvalConfig(BaseDatasetConfig):
    """Local-JSON evals from third_party/cot-oracle/data/evals/."""

    json_path: str = ""
    # Dotted key path into each row that holds the CoT text used for
    # activation extraction. e.g. "metadata.cot_text".
    cot_text_key: str = "metadata.cot_text"
    # Top-level (or dotted) key for the question/prompt presented to the model.
    prompt_key: str = "clean_prompt"
    # Optional fixed override that supersedes prompt_key (useful for
    # sentence_insertion where the same prompt applies to every item).
    prompt_override: str = ""
    # Dotted key path for the gold target_response.
    target_key: str = "correct_answer"
    stochastic_max_k: int = 100
    max_cot_prefix_tokens: int = 2048


def _get_nested(obj: dict, key_path: str) -> Any:
    """Look up a dotted key like 'metadata.cot_text' in a nested dict.
    Returns None if any intermediate key is missing."""
    cur: Any = obj
    for part in key_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


class CotOracleLocalEvalLoader(ActDatasetLoader):
    DATASET_NAME = "cot_oracle_local_eval"

    def __init__(self, dataset_config: DatasetLoaderConfig):
        super().__init__(dataset_config)
        assert self.dataset_config.dataset_name == "", "Dataset name gets overridden here"
        self.dataset_config.dataset_name = self.DATASET_NAME
        self.dataset_params: CotOracleLocalEvalConfig = dataset_config.custom_dataset_params
        assert self.dataset_config.save_acts is False
        assert self.dataset_params.json_path, "json_path is required"

    def create_dataset(self) -> None:
        os.makedirs(self.dataset_config.dataset_folder, exist_ok=True)
        tokenizer = load_tokenizer(self.dataset_config.model_name)

        rng = random.Random(self.dataset_config.seed)

        act_layer_combinations = [
            [layer_percent_to_layer(self.dataset_config.model_name, p) for p in combo]
            for combo in self.dataset_config.layer_combinations
        ]

        with open(self.dataset_params.json_path) as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "entries" in raw:
            entries = raw["entries"]
        else:
            entries = raw
        assert isinstance(entries, list), f"unexpected JSON shape in {self.dataset_params.json_path}"

        for split in self.dataset_config.splits:
            target_n = self.dataset_config.num_train if split == "train" else self.dataset_config.num_test
            if target_n == 0:
                continue
            data = self._build_split(
                tokenizer=tokenizer,
                rng=rng,
                act_layer_combinations=act_layer_combinations,
                entries=entries,
                target_n=target_n,
            )
            self.save_dataset(data, split)  # type: ignore[arg-type]

    def _build_split(
        self,
        tokenizer,
        rng: random.Random,
        act_layer_combinations: list[list[int]],
        entries: list[dict],
        target_n: int,
    ) -> list[TrainingDataPoint]:
        out: list[TrainingDataPoint] = []
        pbar = tqdm(total=target_n, desc=f"local_eval/{os.path.basename(self.dataset_params.json_path)}")
        params = self.dataset_params

        order = list(range(len(entries)))
        rng.shuffle(order)
        cursor = 0
        attempts = 0
        max_attempts = target_n * 10 + 1000

        while len(out) < target_n and attempts < max_attempts:
            attempts += 1
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            row = entries[order[cursor]]
            cursor += 1

            cot_text = _get_nested(row, params.cot_text_key)
            if not cot_text or not isinstance(cot_text, str):
                continue
            target = _get_nested(row, params.target_key)
            if target is None:
                continue
            target = str(target)
            if not target.strip():
                continue

            if params.prompt_override:
                prompt = params.prompt_override
            else:
                prompt = _get_nested(row, params.prompt_key)
                if prompt is None:
                    continue
                prompt = str(prompt)

            cot_prefix_ids = tokenizer(
                cot_text,
                add_special_tokens=False,
                truncation=True,
                max_length=params.max_cot_prefix_tokens,
                return_tensors=None,
            )["input_ids"]
            if len(cot_prefix_ids) == 0:
                continue

            layers = rng.choice(act_layer_combinations)
            positions = sample_cot_oracle_token_positions(
                len(cot_prefix_ids), rng, max_k=params.stochastic_max_k,
            )
            n_actual = len(positions)

            meta_info = {
                "eval_name": row.get("eval_name") or os.path.basename(params.json_path),
                "example_id": row.get("example_id"),
                "cot_text_len_tokens": len(cot_prefix_ids),
                "n_positions_sampled": n_actual,
                "position_sampler": "cot_oracle_stochastic",
                "stochastic_max_k": params.stochastic_max_k,
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
                f"cot_oracle_local_eval: only collected {len(out)}/{target_n} from "
                f"{params.json_path} after {attempts} attempts"
            )
        return out
