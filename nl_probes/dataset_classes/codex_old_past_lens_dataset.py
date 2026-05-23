import gc
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Generator

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from nl_probes.dataset_classes.act_dataset_manager import (
    ActDatasetLoader,
    BaseDatasetConfig,
    DatasetLoaderConfig,
)
from nl_probes.utils.common import layer_percent_to_layer, load_tokenizer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint


@dataclass
class CodexOldPastLensDatasetConfig(BaseDatasetConfig):
    min_k_tokens: int = 1
    max_k_tokens: int = 20
    min_k_activations: int = 1
    max_k_activations: int = 20
    max_length: int = 512
    directions: list[str] = field(default_factory=lambda: ["past", "future"])


class CodexOldPastLensDatasetLoader(ActDatasetLoader):
    def __init__(
        self,
        dataset_config: DatasetLoaderConfig,
    ):
        super().__init__(dataset_config)
        assert self.dataset_config.dataset_name == "", "Dataset name gets overridden here"

        self.dataset_config.dataset_name = "codex_old_past_lens"

        self.dataset_params: CodexOldPastLensDatasetConfig = dataset_config.custom_dataset_params

        assert self.dataset_config.splits == ["train"], "codex_old_past_lens only supports the train split"
        assert self.dataset_config.num_test == 0, "codex_old_past_lens only supports the train split"
        assert self.dataset_config.save_acts is False, "codex_old_past_lens only supports save_acts=False"

        if self.dataset_config.num_train < self.dataset_config.batch_size:
            raise ValueError(
                f"num_train {self.dataset_config.num_train} must be greater than or equal to batch_size {self.dataset_config.batch_size}"
            )

    def create_dataset(self) -> None:
        tokenizer = load_tokenizer(self.dataset_config.model_name)
        dataset = hf_mixed_dataset_to_generator(tokenizer)

        training_data = collect_codex_old_past_lens_targets(
            dataset_config=self.dataset_config,
            custom_dataset_params=self.dataset_params,
            tokenizer=tokenizer,
            dataset=dataset,
            num_datapoints=self.dataset_config.num_train,
        )

        # Close streaming dataset generator and force GC to clean up
        # HF streaming iterators' background threads before process exit.
        # Without this, Python finalization hits a GIL state crash.
        dataset.close()
        del dataset
        gc.collect()

        self.save_dataset(training_data, "train")


def hf_mixed_dataset_to_generator(
    tokenizer: AutoTokenizer,
    pretrain_dataset: str = "HuggingFaceFW/fineweb",
    chat_dataset: str = "lmsys/lmsys-chat-1m",
    min_chars: int = 1,
    pretrain_frac: float = 0.5,
    split: str = "train",
    streaming: bool = True,
    pretrain_key: str = "text",
    chat_key: str = "conversation",
    sequence_pack_pretrain: bool = False,
    sequence_pack_chat: bool = False,
) -> Generator[str, None, None]:
    if not 0 < pretrain_frac < 1:
        raise ValueError("main_frac must be between 0 and 1 (exclusive)")

    assert min_chars > 0

    pretrain_ds = iter(load_dataset(pretrain_dataset, split=split, streaming=streaming))
    chat_ds = iter(load_dataset(chat_dataset, split=split, streaming=streaming))

    frac = Fraction(pretrain_frac).limit_denominator()
    n_pretrain = frac.numerator
    n_chat = frac.denominator - n_pretrain
    eos_token = tokenizer.eos_token
    bos_token = tokenizer.bos_token if tokenizer.bos_token else eos_token
    assert bos_token is not None, "Tokenizer must define at least one of bos_token/eos_token"

    def gen() -> Generator[str, None, None]:
        while True:
            for _ in range(n_pretrain):
                if sequence_pack_pretrain:
                    length = 0
                    samples = []
                    while length < min_chars:
                        sample = next(pretrain_ds)[pretrain_key]
                        samples.append(sample)
                        length += len(sample)
                    yield bos_token + eos_token.join(samples)
                else:
                    yield bos_token + next(pretrain_ds)[pretrain_key]

            for _ in range(n_chat):
                if sequence_pack_chat:
                    length = 0
                    samples = []
                    while length < min_chars:
                        sample = next(chat_ds)[chat_key]
                        sample = tokenizer.apply_chat_template(sample, tokenize=False, enable_thinking=False)
                        samples.append(sample)
                        length += len(sample)
                    yield "".join(samples)
                else:
                    sample = tokenizer.apply_chat_template(
                        next(chat_ds)[chat_key],
                        tokenize=False,
                        enable_thinking=False,
                    )
                    yield sample

    return gen()


def collect_codex_old_past_lens_targets(
    dataset_config: DatasetLoaderConfig,
    custom_dataset_params: CodexOldPastLensDatasetConfig,
    tokenizer: AutoTokenizer,
    dataset: Generator[str, None, None],
    num_datapoints: int,
) -> list[TrainingDataPoint]:
    random.seed(dataset_config.seed)
    torch.manual_seed(dataset_config.seed)

    assert dataset_config.layer_combinations, "layer_combinations must be non-empty"
    act_layer_combinations = [
        [layer_percent_to_layer(dataset_config.model_name, layer_percent) for layer_percent in layer_combo]
        for layer_combo in dataset_config.layer_combinations
    ]

    valid_directions = {"past", "future"}
    assert len(custom_dataset_params.directions) > 0, "directions must be non-empty"
    assert set(custom_dataset_params.directions).issubset(valid_directions), (
        f"directions must be in {valid_directions}, got {custom_dataset_params.directions}"
    )

    training_data: list[TrainingDataPoint] = []
    pbar = tqdm(total=num_datapoints, desc="Collecting codex old past lens targets")
    while len(training_data) < num_datapoints:
        inputs = []
        for _ in range(dataset_config.batch_size):
            inputs.append(next(dataset))

        tokenized_inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=custom_dataset_params.max_length,
            add_special_tokens=False,
        )

        attn_mask_BL = tokenized_inputs["attention_mask"]
        input_ids_BL = tokenized_inputs["input_ids"]

        for j in range(len(inputs)):
            if len(training_data) >= num_datapoints:
                break

            layers = random.choice(act_layer_combinations)
            attn_mask_L = attn_mask_BL[j].bool()
            input_ids_L_full = input_ids_BL[j, attn_mask_L].tolist()
            seq_len = len(input_ids_L_full)

            k_tokens = random.randint(custom_dataset_params.min_k_tokens, custom_dataset_params.max_k_tokens)
            k_acts = random.randint(custom_dataset_params.min_k_activations, custom_dataset_params.max_k_activations)
            direction = random.choice(custom_dataset_params.directions)

            if seq_len < k_tokens + k_acts + 1:
                continue

            if direction == "past":
                act_begin_min = k_tokens
                act_begin_max = seq_len - k_acts - 1
                if act_begin_max < act_begin_min:
                    continue
                selected_act_begin_idx = random.randint(act_begin_min, act_begin_max)
                selected_act_positions = list(range(selected_act_begin_idx, selected_act_begin_idx + k_acts))
                selected_tokens_positions = list(range(selected_act_begin_idx - k_tokens, selected_act_begin_idx))
                context_cutoff = selected_act_positions[-1]
                target_token_ids = [input_ids_L_full[idx] for idx in selected_tokens_positions]
                target_text = tokenizer.decode(target_token_ids, skip_special_tokens=True)
                prompt = f"Can you predict the previous {k_tokens} tokens that came before this?"
                target_start_idx = selected_tokens_positions[0]
                target_end_idx_exclusive = selected_tokens_positions[-1] + 1
            else:
                act_begin_min = 1
                act_begin_max = seq_len - k_acts - k_tokens
                if act_begin_max < act_begin_min:
                    continue
                selected_act_begin_idx = random.randint(act_begin_min, act_begin_max)
                selected_act_positions = list(range(selected_act_begin_idx, selected_act_begin_idx + k_acts))
                last_act_pos = selected_act_positions[-1]
                selected_tokens_positions = list(range(last_act_pos + 1, last_act_pos + 1 + k_tokens))
                context_cutoff = last_act_pos
                target_token_ids = [input_ids_L_full[idx] for idx in selected_tokens_positions]
                target_text = tokenizer.decode(target_token_ids, skip_special_tokens=True)
                prompt = f"Can you predict the next {k_tokens} tokens that come after this?"
                target_start_idx = selected_tokens_positions[0]
                target_end_idx_exclusive = selected_tokens_positions[-1] + 1

            context_input_ids = input_ids_L_full[: context_cutoff + 1]
            meta_info = {
                "direction": direction,
                "k_tokens": k_tokens,
                "k_acts": k_acts,
                "context_len": len(context_input_ids),
                "act_start": selected_act_positions[0],
                "act_end": selected_act_positions[-1],
                "target_start_idx": target_start_idx,
                "target_end_idx_exclusive": target_end_idx_exclusive,
            }

            training_data_point = create_training_datapoint(
                datapoint_type=dataset_config.dataset_name,
                prompt=prompt,
                target_response=target_text,
                layers=layers,
                num_positions=k_acts,
                tokenizer=tokenizer,
                acts_BD=None,
                feature_idx=-1,
                context_input_ids=context_input_ids,
                context_positions=selected_act_positions,
                meta_info=meta_info,
            )
            training_data.append(training_data_point)
            pbar.update(1)

    pbar.close()
    return training_data
