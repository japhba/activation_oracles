"""AObench-driven validation dataset for the recog_* logp MCQ family.

Wraps three AObench tasks (`number_prediction`, `mmlu_prediction`, `missing_info`)
as `TrainingDataPoint`s compatible with `nl_probes.utils.logprob_mcq.run_logprob_mcq_eval`
and `run_mcq_rephrase_eval`.

Each task contributes:
  cot_prefix_text  -- the text whose forward pass yields the activations the AO sees
  prompt_text      -- the AO-side prompt asking about the model's behavior
  target_text      -- what the AO should answer (short, exact-match style)

Judge-based AObench tasks (vagueness, domain_confusion, backtracking) intentionally
NOT included here -- their targets are open-ended and only meaningful under an
LLM judge, so they don't fit the recog_* (logp(target | with-acts) vs baseline)
paradigm. They are still run via the open-ended eval suite at low cadence.
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
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


_AOBENCH_ROOT = Path(__file__).resolve().parent.parent.parent / "third_party" / "cot-oracle"
_AOBENCH_DATA_DIR = _AOBENCH_ROOT / "AObench" / "datasets"

SUPPORTED_TASKS = ("number_prediction", "mmlu_prediction", "missing_info")


@dataclass
class AObenchRecogConfig(BaseDatasetConfig):
    task_name: str = "mmlu_prediction"  # one of SUPPORTED_TASKS
    stochastic_max_k: int = 100
    max_cot_prefix_tokens: int = 2048


def _data_path(task_name: str) -> Path:
    mapping = {
        "number_prediction": "number_prediction/number_prediction_eval_dataset.json",
        "mmlu_prediction": "mmlu_prediction/mmlu_prediction_eval_dataset.json",
        "missing_info": "missing_info/missing_info_eval_dataset.json",
    }
    if task_name not in mapping:
        raise ValueError(f"Unsupported aobench_recog task: {task_name}. Supported: {SUPPORTED_TASKS}")
    return _AOBENCH_DATA_DIR / mapping[task_name]


def _load_entries(task_name: str) -> list[dict[str, Any]]:
    path = _data_path(task_name)
    if not path.exists():
        raise FileNotFoundError(f"AObench data not found at {path}")
    with open(path) as f:
        data = json.load(f)
    entries = data.get("entries", data) if isinstance(data, dict) else data
    assert isinstance(entries, list) and entries, f"Empty entries for {task_name}"
    return entries


_LETTER_CHOICES = ["A", "B", "C", "D"]


def _format_mmlu_question(question: str, choices: list[str]) -> str:
    """Mirror AObench's mmlu prompt style."""
    lines = [question, ""]
    for i, choice in enumerate(choices):
        lines.append(f"{_LETTER_CHOICES[i]}. {choice}")
    lines.append("")
    lines.append("Answer with just the letter (A, B, C, or D), nothing else.")
    return "\n".join(lines)


def _entry_to_recog_fields(task_name: str, entry: dict[str, Any]) -> tuple[str, str, str] | None:
    """Return (cot_prefix_text, prompt_text, target_text) or None if entry should be skipped."""
    if task_name == "number_prediction":
        # Model sees the expression, generates the number. AO predicts the number.
        # cot_prefix = the expression context (model's input). Target = the digit string.
        expr = entry.get("expression")
        target = entry.get("model_raw_response") or str(entry.get("model_answer"))
        if not expr or not target:
            return None
        cot_prefix = f"What is {expr}?"
        prompt = "What number is the model about to output?"
        return cot_prefix, prompt, target

    if task_name == "mmlu_prediction":
        question = entry.get("question")
        choices = entry.get("choices")
        model_letter = entry.get("model_answer_letter")
        model_correct = entry.get("model_correct")
        if not question or not choices or model_letter is None or model_correct is None:
            return None
        cot_prefix = _format_mmlu_question(question, choices) + f"\nModel's answer: {model_letter}"
        prompt = "Will the model answer this question correctly? Answer yes or no."
        target = "yes" if bool(model_correct) else "no"
        return cot_prefix, prompt, target

    if task_name == "missing_info":
        problem = entry.get("problem_text")
        reasoning = entry.get("full_reasoning") or entry.get("teacher_forced_segment", "")
        gt_missing = entry.get("ground_truth_missing_info")
        if not problem or gt_missing is None:
            return None
        cot_prefix = f"{problem}\n\n{reasoning}".strip()
        prompt = "Is the model missing information needed to solve this problem? Answer yes or no."
        # gt_missing == True means "yes, info is missing".
        target = "yes" if bool(gt_missing) else "no"
        return cot_prefix, prompt, target

    raise ValueError(f"Unsupported task: {task_name}")


class AObenchRecogLoader(ActDatasetLoader):
    DATASET_NAME = "aobench_recog"

    def __init__(self, dataset_config: DatasetLoaderConfig):
        super().__init__(dataset_config)
        assert self.dataset_config.dataset_name == "", "Dataset name gets overridden here"
        self.dataset_config.dataset_name = self.DATASET_NAME
        self.dataset_params: AObenchRecogConfig = dataset_config.custom_dataset_params
        assert self.dataset_config.save_acts is False

    def create_dataset(self) -> None:
        os.makedirs(self.dataset_config.dataset_folder, exist_ok=True)
        tokenizer = load_tokenizer(self.dataset_config.model_name)

        rng = random.Random(self.dataset_config.seed)
        act_layer_combinations = [
            [layer_percent_to_layer(self.dataset_config.model_name, p) for p in combo]
            for combo in self.dataset_config.layer_combinations
        ]

        entries = _load_entries(self.dataset_params.task_name)

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
        entries: list[dict[str, Any]],
        target_n: int,
    ) -> list[TrainingDataPoint]:
        out: list[TrainingDataPoint] = []
        order = list(range(len(entries)))
        rng.shuffle(order)
        cursor = 0
        attempts = 0
        max_attempts = target_n * 10 + 1000
        pbar = tqdm(total=target_n, desc=f"aobench/{self.dataset_params.task_name}")

        while len(out) < target_n and attempts < max_attempts:
            attempts += 1
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            entry = entries[order[cursor]]
            cursor += 1

            fields = _entry_to_recog_fields(self.dataset_params.task_name, entry)
            if fields is None:
                continue
            cot_prefix_text, prompt_text, target_text = fields

            cot_prefix_ids = tokenizer(
                cot_prefix_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.dataset_params.max_cot_prefix_tokens,
                return_tensors=None,
            )["input_ids"]
            if len(cot_prefix_ids) == 0:
                continue

            layers = rng.choice(act_layer_combinations)
            positions = sample_cot_oracle_token_positions(
                len(cot_prefix_ids), rng, max_k=self.dataset_params.stochastic_max_k,
            )
            n_actual = len(positions)

            meta_info: dict[str, Any] = {
                "task": self.dataset_params.task_name,
                "entry_id": entry.get("id") or entry.get("problem_id"),
                "n_positions_sampled": n_actual,
                "position_sampler": "cot_oracle_stochastic",
                "stochastic_max_k": self.dataset_params.stochastic_max_k,
                "cot_prefix_len_tokens": len(cot_prefix_ids),
                "context_positions_first": positions[0],
                "context_positions_last": positions[-1],
            }

            dp = create_training_datapoint(
                datapoint_type=self.DATASET_NAME,
                prompt=prompt_text,
                target_response=target_text,
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
                f"aobench_recog/{self.dataset_params.task_name}: only collected "
                f"{len(out)}/{target_n} after {attempts} attempts (entries available: {len(entries)})"
            )
        return out
