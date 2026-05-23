import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel
from tqdm import tqdm

from nl_probes.configs.sft_config import read_training_config
from nl_probes.dataset_classes.act_dataset_manager import (
    ActDatasetLoader,
    _deserialize_loader,
    build_loaders_from_config,
)
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer, set_seed
from nl_probes.utils.dataset_utils import (
    TrainingDataPoint,
    construct_batch,
    materialize_missing_steering_vectors,
)
from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook


@dataclass
class LoaderSource:
    loader_index: int
    dataset_name: str
    loader_variant: str
    file_path: Path
    config: dict[str, Any]
    split: str
    source_name: str


@dataclass
class SelectedExample:
    example_id: str
    loader_index: int
    dataset_name: str
    loader_variant: str
    loader_config_json: str
    split: str
    source_name: str
    dp: TrainingDataPoint


def resolve_checkpoint_dir(path_str: str) -> Path:
    path = Path(path_str)
    assert path.exists(), f"Checkpoint path does not exist: {path}"

    if (path / "ao_config.json").exists():
        return path

    final_dir = path / "final"
    if (final_dir / "ao_config.json").exists():
        return final_dir

    raise ValueError(
        f"Could not find ao_config.json in {path} or {final_dir}. "
        "Pass either a checkpoint root containing final/ or the final adapter directory itself."
    )


def get_loader_variant(loader: ActDatasetLoader) -> str:
    dataset_name = loader.dataset_config.dataset_name
    params = asdict(loader.dataset_config.custom_dataset_params)

    if dataset_name == "past_lens":
        return (
            f"past_lens_kacts_{params['min_k_activations']}_{params['max_k_activations']}"
            f"_ktokens_{params['min_k_tokens']}_{params['max_k_tokens']}"
        )

    if dataset_name == "synthetic_qa":
        data_path = Path(params["data_path"])
        return f"synthetic_qa_{data_path.parent.name}_{data_path.stem}"

    if dataset_name.startswith("classification_"):
        return (
            f"{dataset_name}_qa_{params['num_qa_per_sample']}"
            f"_window_{params['min_window_size']}_{params['max_window_size']}"
        )

    raise ValueError(f"Unhandled dataset_name for loader variant: {dataset_name}")


def make_loader_source(
    *,
    loader_index: int,
    loader: ActDatasetLoader,
    split: str,
    source_name: str,
    loader_variant_suffix: str | None = None,
) -> LoaderSource:
    file_path = Path(loader.dataset_config.dataset_folder) / loader.get_dataset_filename(split)
    assert file_path.exists(), f"Missing dataset file: {file_path}"

    loader_variant = get_loader_variant(loader)
    if loader_variant_suffix is not None:
        loader_variant = f"{loader_variant}_{loader_variant_suffix}"

    return LoaderSource(
        loader_index=loader_index,
        dataset_name=loader.dataset_config.dataset_name,
        loader_variant=loader_variant,
        file_path=file_path,
        config=asdict(loader.dataset_config),
        split=split,
        source_name=source_name,
    )


def build_loader_sources(checkpoint_dir: Path) -> tuple[Any, list[LoaderSource]]:
    cfg = read_training_config(str(checkpoint_dir))
    loaders = build_loaders_from_config(cfg)

    sources: list[LoaderSource] = []
    for loader_index, loader in enumerate(loaders):
        if "train" not in loader.dataset_config.splits:
            continue

        sources.append(
            make_loader_source(
                loader_index=loader_index,
                loader=loader,
                split="train",
                source_name="checkpoint_train",
            )
        )

    assert sources, "No training dataset sources found"
    return cfg, sources


def build_loader_from_raw_config(raw_config: dict[str, Any]) -> ActDatasetLoader:
    return _deserialize_loader(copy.deepcopy(raw_config))


def build_heldout_sources(
    *,
    base_sources: list[LoaderSource],
    base_allocations: list[int],
    output_dir: Path,
    past_lens_seed: int,
    synthetic_qa_v2_config_path: Path,
    past_lens_generation_buffer: int,
) -> list[LoaderSource]:
    assert len(base_sources) == len(base_allocations), "base_sources and base_allocations must align"
    synthetic_qa_v2_cfg = json.loads(synthetic_qa_v2_config_path.read_text())
    synthetic_qa_v2_raw = synthetic_qa_v2_cfg["dataset_configs"][0]

    heldout_sources: list[LoaderSource] = []
    heldout_loader_index = 0
    synthetic_allocation = sum(
        allocation
        for source, allocation in zip(base_sources, base_allocations, strict=True)
        if source.dataset_name == "synthetic_qa"
    )

    synthetic_source_added = False
    heldout_dataset_dir = output_dir / "heldout_sft_training_data"
    heldout_dataset_dir.mkdir(parents=True, exist_ok=True)

    for source, allocation in zip(base_sources, base_allocations, strict=True):
        raw_config = copy.deepcopy(source.config)

        if source.dataset_name == "synthetic_qa":
            if synthetic_source_added:
                continue
            synthetic_source_added = True

            loader = build_loader_from_raw_config(synthetic_qa_v2_raw)
            heldout_sources.append(
                make_loader_source(
                    loader_index=heldout_loader_index,
                    loader=loader,
                    split="train",
                    source_name="heldout_synthetic_qa_v2",
                    loader_variant_suffix="heldout_v2",
                )
            )
            heldout_loader_index += 1
            continue

        if source.dataset_name == "past_lens":
            raw_config["seed"] = past_lens_seed
            raw_config["num_train"] = allocation + past_lens_generation_buffer
            raw_config["dataset_folder"] = str(heldout_dataset_dir)

            loader = build_loader_from_raw_config(raw_config)
            train_path = Path(loader.dataset_config.dataset_folder) / loader.get_dataset_filename("train")
            if not train_path.exists():
                raw_config_path = heldout_dataset_dir / f"{train_path.stem}.raw_config.json"
                raw_config_path.write_text(json.dumps(raw_config, indent=2))
                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import json, sys, torch; "
                            "from pathlib import Path; "
                            "from nl_probes.dataset_classes.act_dataset_manager import _deserialize_loader; "
                            "torch.cuda.empty_cache = lambda: None; "
                            "raw_config = json.loads(Path(sys.argv[1]).read_text()); "
                            "loader = _deserialize_loader(raw_config); "
                            "loader.ensure_dataset_exists('train')"
                        ),
                        str(raw_config_path),
                    ],
                    check=True,
                    env=os.environ.copy(),
                )
            heldout_sources.append(
                make_loader_source(
                    loader_index=heldout_loader_index,
                    loader=loader,
                    split="train",
                    source_name=f"heldout_past_lens_seed_{past_lens_seed}",
                    loader_variant_suffix=f"heldout_seed_{past_lens_seed}",
                )
            )
            heldout_loader_index += 1
            continue

        if source.dataset_name.startswith("classification_"):
            raw_config["num_test"] = 250
            raw_config["splits"] = ["test"]

            loader = build_loader_from_raw_config(raw_config)
            file_path = Path(loader.dataset_config.dataset_folder) / loader.get_dataset_filename("test")
            if not file_path.exists():
                loader.ensure_dataset_exists("test")
            heldout_sources.append(
                make_loader_source(
                    loader_index=heldout_loader_index,
                    loader=loader,
                    split="test",
                    source_name="heldout_classification_test",
                    loader_variant_suffix="heldout_test",
                )
            )
            heldout_loader_index += 1
            continue

        raise ValueError(f"Unhandled dataset_name in held-out source builder: {source.dataset_name}")

    synthetic_sources = [source for source in heldout_sources if source.dataset_name == "synthetic_qa"]
    assert len(synthetic_sources) == 1, f"Expected exactly 1 held-out synthetic_qa source, found {len(synthetic_sources)}"
    assert synthetic_allocation > 0, "Expected non-zero synthetic_qa allocation in base mix"

    return heldout_sources


def build_heldout_allocations(base_sources: list[LoaderSource], base_allocations: list[int]) -> list[int]:
    synthetic_allocation = sum(
        allocation
        for source, allocation in zip(base_sources, base_allocations, strict=True)
        if source.dataset_name == "synthetic_qa"
    )

    heldout_allocations: list[int] = []
    synthetic_added = False
    for source, allocation in zip(base_sources, base_allocations, strict=True):
        if source.dataset_name == "synthetic_qa":
            if synthetic_added:
                continue
            synthetic_added = True
            heldout_allocations.append(synthetic_allocation)
            continue

        heldout_allocations.append(allocation)

    return heldout_allocations


def load_reused_base_plan(
    *,
    result_dir: Path,
    base_sources: list[LoaderSource],
    subset_size: int,
) -> tuple[int, list[int], list[int]]:
    run_config = json.loads((result_dir / "run_config.json").read_text())
    allocation_df = pd.read_csv(result_dir / "subset_allocation.csv")

    assert int(run_config["subset_size"]) == subset_size, (
        f"Reused allocation plan subset_size={run_config['subset_size']} does not match requested {subset_size}"
    )

    key_to_allocation: dict[tuple[int, str, str], int] = {}
    key_to_kept_count: dict[tuple[int, str, str], int] = {}
    for _, row in allocation_df.iterrows():
        key = (int(row["loader_index"]), row["dataset_name"], row["loader_variant"])
        key_to_allocation[key] = int(row["allocation"])
        key_to_kept_count[key] = int(row["kept_count"])

    allocations: list[int] = []
    kept_counts: list[int] = []
    for source in base_sources:
        key = (source.loader_index, source.dataset_name, source.loader_variant)
        assert key in key_to_allocation, f"Missing base allocation for source {key}"
        allocations.append(key_to_allocation[key])
        kept_counts.append(key_to_kept_count[key])

    assert sum(allocations) == subset_size, (
        f"Reused allocations sum to {sum(allocations)} but expected {subset_size}"
    )
    return int(run_config["length_threshold"]), allocations, kept_counts


def compute_length_threshold(sources: list[LoaderSource], length_percentile: float) -> int:
    assert 0.0 < length_percentile < 1.0, "length_percentile must be between 0 and 1"

    all_lengths: list[int] = []
    for source in tqdm(sources, desc="Scanning lengths"):
        raw = torch.load(source.file_path, map_location="cpu", weights_only=False)
        all_lengths.extend(len(dp["input_ids"]) for dp in raw["data"])

    all_lengths.sort()
    threshold_idx = int((len(all_lengths) - 1) * length_percentile)
    return all_lengths[threshold_idx]


def count_kept_examples_by_source(sources: list[LoaderSource], length_threshold: int) -> list[int]:
    kept_counts: list[int] = []
    for source in tqdm(sources, desc="Counting post-trim examples"):
        raw = torch.load(source.file_path, map_location="cpu", weights_only=False)
        kept_count = sum(len(dp["input_ids"]) <= length_threshold for dp in raw["data"])
        kept_counts.append(kept_count)
    return kept_counts


def allocate_subset_sizes(kept_counts: list[int], subset_size: int, min_per_loader: int) -> list[int]:
    assert subset_size > 0, "subset_size must be positive"
    assert min_per_loader >= 0, "min_per_loader must be non-negative"
    assert subset_size <= sum(kept_counts), (
        f"Requested subset_size={subset_size} exceeds available post-trim examples={sum(kept_counts)}"
    )

    num_sources = len(kept_counts)
    effective_floor = min(min_per_loader, subset_size // num_sources)
    base_alloc = [min(count, effective_floor) for count in kept_counts]

    remaining = subset_size - sum(base_alloc)
    capacities = [count - base for count, base in zip(kept_counts, base_alloc, strict=True)]
    if remaining == 0:
        return base_alloc

    total_capacity = sum(capacities)
    assert total_capacity >= remaining, "Not enough remaining capacity after base allocation"

    ideal_extras = [remaining * capacity / total_capacity for capacity in capacities]
    extra_alloc = [min(capacity, int(math.floor(ideal))) for capacity, ideal in zip(capacities, ideal_extras, strict=True)]

    leftover = remaining - sum(extra_alloc)
    if leftover > 0:
        ranked = sorted(
            range(num_sources),
            key=lambda idx: (ideal_extras[idx] - extra_alloc[idx], capacities[idx]),
            reverse=True,
        )
        for idx in ranked:
            if leftover == 0:
                break
            if extra_alloc[idx] == capacities[idx]:
                continue
            extra_alloc[idx] += 1
            leftover -= 1

    assert leftover == 0, f"Failed to allocate full subset, leftover={leftover}"
    alloc = [base + extra for base, extra in zip(base_alloc, extra_alloc, strict=True)]
    assert sum(alloc) == subset_size
    return alloc


def make_example_id(raw_dp: dict[str, Any], loader_index: int) -> str:
    payload = {
        "loader_index": loader_index,
        "datapoint_type": raw_dp["datapoint_type"],
        "input_ids": raw_dp["input_ids"],
        "labels": raw_dp["labels"],
        "layers": raw_dp["layers"],
        "positions": raw_dp["positions"],
        "context_input_ids": raw_dp["context_input_ids"],
        "context_positions": raw_dp["context_positions"],
        "target_output": raw_dp["target_output"],
        "meta_info": raw_dp["meta_info"],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2s(blob, digest_size=12).hexdigest()


def sample_subset(
    sources: list[LoaderSource],
    allocations: list[int],
    length_threshold: int,
    seed: int,
) -> list[SelectedExample]:
    rng = random.Random(seed)
    selected: list[SelectedExample] = []

    for source, sample_count in zip(sources, allocations, strict=True):
        if sample_count == 0:
            continue

        raw = torch.load(source.file_path, map_location="cpu", weights_only=False)
        kept_indices = [idx for idx, dp in enumerate(raw["data"]) if len(dp["input_ids"]) <= length_threshold]
        assert sample_count <= len(kept_indices), (
            f"Requested {sample_count} samples from {source.loader_variant}, but only {len(kept_indices)} remain"
        )

        sampled_indices = set(rng.sample(kept_indices, sample_count))
        loader_config_json = json.dumps(source.config, sort_keys=True)
        for idx in sampled_indices:
            raw_dp = raw["data"][idx]
            selected.append(
                SelectedExample(
                    example_id=make_example_id(raw_dp, source.loader_index),
                    loader_index=source.loader_index,
                    dataset_name=source.dataset_name,
                    loader_variant=source.loader_variant,
                    loader_config_json=loader_config_json,
                    split=source.split,
                    source_name=source.source_name,
                    dp=TrainingDataPoint(**raw_dp),
                )
            )

    assert len(selected) == sum(allocations), (
        f"Selected {len(selected)} examples but expected {sum(allocations)}"
    )
    return selected


def build_randomized_points(
    examples: list[SelectedExample],
    generator: torch.Generator,
) -> list[TrainingDataPoint]:
    randomized_points: list[TrainingDataPoint] = []
    for example in examples:
        dp = example.dp
        assert dp.steering_vectors is not None, "Expected materialized steering vectors before randomization"
        random_vectors = torch.randn(dp.steering_vectors.shape, generator=generator, dtype=dp.steering_vectors.dtype)
        randomized = dp.model_copy(deep=True)
        randomized.steering_vectors = random_vectors
        randomized_points.append(randomized)
    return randomized_points


def build_zero_vector_points(examples: list[SelectedExample]) -> list[TrainingDataPoint]:
    zero_points: list[TrainingDataPoint] = []
    for example in examples:
        dp = example.dp
        assert dp.steering_vectors is not None, "Expected materialized steering vectors before zero-vector baseline"
        zeroed = dp.model_copy(deep=True)
        zeroed.steering_vectors = torch.zeros_like(dp.steering_vectors)
        zero_points.append(zeroed)
    return zero_points


def split_steering_vectors_by_layer(dp: TrainingDataPoint) -> list[torch.Tensor]:
    assert dp.steering_vectors is not None, "Expected materialized steering vectors"
    assert dp.context_positions is not None, "Expected context_positions for layer splitting"
    num_layers = len(dp.layers)
    num_positions = len(dp.context_positions)
    assert dp.steering_vectors.shape[0] == num_layers * num_positions, (
        f"Expected {num_layers * num_positions} steering rows, got {dp.steering_vectors.shape[0]}"
    )
    steering_vectors = dp.steering_vectors.detach().cpu()
    return list(steering_vectors.reshape(num_layers, num_positions, steering_vectors.shape[-1]))


def build_batch_row_sampled_points(
    examples: list[SelectedExample],
    generator: torch.Generator,
) -> list[TrainingDataPoint]:
    assert len(examples) > 1, "batch_row_sample baseline requires at least 2 examples in the batch"
    split_vectors = [split_steering_vectors_by_layer(example.dp) for example in examples]

    sampled_points: list[TrainingDataPoint] = []
    for recipient_idx, example in enumerate(examples):
        recipient = example.dp
        assert recipient.context_positions is not None, "Expected context_positions for donor baseline"
        target_num_positions = len(recipient.context_positions)

        donor_indices = [idx for idx in range(len(examples)) if idx != recipient_idx]
        assert donor_indices, "Need at least one donor example"

        layer_blocks = []
        for layer_idx, layer in enumerate(recipient.layers):
            donor_rows = []
            for donor_idx in donor_indices:
                donor = examples[donor_idx].dp
                assert donor.layers == recipient.layers, (
                    f"Layer mismatch between recipient {recipient.layers} and donor {donor.layers}"
                )
                assert donor.layers[layer_idx] == layer
                donor_rows.append(split_vectors[donor_idx][layer_idx])

            donor_pool = torch.cat(donor_rows, dim=0)
            sampled_indices = torch.randint(
                donor_pool.shape[0],
                (target_num_positions,),
                generator=generator,
            ).to(donor_pool.device)
            layer_blocks.append(donor_pool.index_select(0, sampled_indices))

        sampled = recipient.model_copy(deep=True)
        sampled.steering_vectors = torch.cat(layer_blocks, dim=0)
        sampled_points.append(sampled)

    return sampled_points


def tile_rows(rows: torch.Tensor, target_num_positions: int, start_offset: int) -> torch.Tensor:
    assert rows.ndim == 2, f"Expected 2D rows, got shape={rows.shape}"
    assert rows.shape[0] > 0, "Cannot tile empty donor rows"
    rotated_rows = torch.roll(rows, shifts=-start_offset, dims=0)
    num_repeats = math.ceil(target_num_positions / rotated_rows.shape[0])
    return rotated_rows.repeat((num_repeats, 1))[:target_num_positions]


def build_batch_repeat_single_donor_points(
    examples: list[SelectedExample],
    generator: torch.Generator,
) -> list[TrainingDataPoint]:
    return build_batch_repeat_single_donor_points_impl(examples, generator, share_layer_offset=False)


def build_batch_repeat_single_donor_shared_offset_points(
    examples: list[SelectedExample],
    generator: torch.Generator,
) -> list[TrainingDataPoint]:
    return build_batch_repeat_single_donor_points_impl(examples, generator, share_layer_offset=True)


def build_batch_repeat_single_donor_points_impl(
    examples: list[SelectedExample],
    generator: torch.Generator,
    share_layer_offset: bool,
) -> list[TrainingDataPoint]:
    assert len(examples) > 1, "batch_repeat_single_donor baseline requires at least 2 examples in the batch"
    split_vectors = [split_steering_vectors_by_layer(example.dp) for example in examples]

    repeated_points: list[TrainingDataPoint] = []
    for recipient_idx, example in enumerate(examples):
        recipient = example.dp
        assert recipient.context_positions is not None, "Expected context_positions for donor baseline"
        target_num_positions = len(recipient.context_positions)

        donor_indices = [idx for idx in range(len(examples)) if idx != recipient_idx]
        assert donor_indices, "Need at least one donor example"
        donor_choice = int(torch.randint(len(donor_indices), (1,), generator=generator).item())
        donor_idx = donor_indices[donor_choice]
        donor = examples[donor_idx].dp
        assert donor.layers == recipient.layers, (
            f"Layer mismatch between recipient {recipient.layers} and donor {donor.layers}"
        )

        shared_start_offset = None
        if share_layer_offset:
            assert donor.context_positions is not None, "Expected donor context_positions for shared offset baseline"
            shared_start_offset = int(torch.randint(len(donor.context_positions), (1,), generator=generator).item())

        layer_blocks = []
        for layer_idx, layer in enumerate(recipient.layers):
            assert donor.layers[layer_idx] == layer
            donor_rows = split_vectors[donor_idx][layer_idx]
            start_offset = shared_start_offset
            if start_offset is None:
                start_offset = int(torch.randint(donor_rows.shape[0], (1,), generator=generator).item())
            else:
                assert donor_rows.shape[0] == len(donor.context_positions), (
                    f"Expected donor_rows.shape[0]={donor_rows.shape[0]} to match "
                    f"len(donor.context_positions)={len(donor.context_positions)}"
                )
            layer_blocks.append(tile_rows(donor_rows, target_num_positions, start_offset))

        repeated = recipient.model_copy(deep=True)
        repeated.steering_vectors = torch.cat(layer_blocks, dim=0)
        repeated_points.append(repeated)

    return repeated_points


def compute_shifted_token_losses(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shifted_logits = logits[:, :-1, :].float().contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shifted_labels.shape)
    valid_mask = shifted_labels != -100
    return token_losses, valid_mask


def first_n_mask(valid_mask: torch.Tensor, n: int) -> torch.Tensor:
    assert n > 0, "n must be positive"
    mask = torch.zeros_like(valid_mask)
    for row_idx in range(valid_mask.shape[0]):
        valid_positions = torch.nonzero(valid_mask[row_idx], as_tuple=False).flatten()
        if len(valid_positions) == 0:
            continue
        mask[row_idx, valid_positions[:n]] = True
    return mask


def steered_logits(
    *,
    model: PeftModel,
    submodule: torch.nn.Module,
    batch: Any,
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
    use_hook: bool,
) -> torch.Tensor:
    inputs = {
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
        "use_cache": False,
    }

    with torch.inference_mode():
        if use_hook:
            hook_fn = get_hf_activation_steering_hook(
                vectors=batch.steering_vectors,
                positions=batch.positions,
                steering_coefficient=steering_coefficient,
                device=device,
                dtype=dtype,
            )
            with add_hook(submodule, hook_fn):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

    return outputs.logits


def extract_common_meta(dp: TrainingDataPoint) -> dict[str, Any]:
    meta = dict(dp.meta_info)
    return {
        "direction": meta["direction"] if "direction" in meta else None,
        "k_tokens": meta["k_tokens"] if "k_tokens" in meta else None,
        "k_acts": meta["k_acts"] if "k_acts" in meta else None,
        "sample_source": meta["sample_source"] if "sample_source" in meta else None,
        "system_prompt_injected": meta["system_prompt_injected"] if "system_prompt_injected" in meta else None,
        "qa_type": meta["qa_type"] if "qa_type" in meta else None,
        "response_format": meta["response_format"] if "response_format" in meta else None,
    }


def decode_prompt_only(dp: TrainingDataPoint, tokenizer: Any) -> str:
    first_label_idx = next((idx for idx, label in enumerate(dp.labels) if label != -100), len(dp.input_ids))
    return tokenizer.decode(dp.input_ids[:first_label_idx], skip_special_tokens=False)


def summarize_scores(per_example_df: pd.DataFrame, include_no_hook: bool) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    overall = {
        "num_examples": int(len(per_example_df)),
        "real_mean_nll": float(per_example_df["real_mean_nll"].mean()),
        "random_mean_nll": float(per_example_df["random_mean_nll"].mean()),
        "delta_mean_nll_mean": float(per_example_df["delta_mean_nll"].mean()),
        "delta_mean_nll_median": float(per_example_df["delta_mean_nll"].median()),
        "real_beats_random_rate": float((per_example_df["delta_mean_nll"] < 0.0).mean()),
        "first1_delta_mean": float(per_example_df["delta_first1_mean_nll"].mean()),
        "first3_delta_mean": float(per_example_df["delta_first3_mean_nll"].mean()),
        "first5_delta_mean": float(per_example_df["delta_first5_mean_nll"].mean()),
    }
    if include_no_hook:
        overall["no_hook_mean_nll"] = float(per_example_df["no_hook_mean_nll"].mean())
        overall["real_beats_no_hook_rate"] = float((per_example_df["delta_vs_no_hook_mean_nll"] < 0.0).mean())

    optional_baselines = []
    if "zero_vector_mean_nll" in per_example_df.columns:
        optional_baselines.append("zero_vector")
    if "batch_row_sample_mean_nll" in per_example_df.columns:
        optional_baselines.append("batch_row_sample")
    if "batch_repeat_single_donor_mean_nll" in per_example_df.columns:
        optional_baselines.append("batch_repeat_single_donor")
    if "batch_repeat_single_donor_shared_offset_mean_nll" in per_example_df.columns:
        optional_baselines.append("batch_repeat_single_donor_shared_offset")

    for baseline_name in optional_baselines:
        overall[f"{baseline_name}_mean_nll"] = float(per_example_df[f"{baseline_name}_mean_nll"].mean())
        overall[f"real_beats_{baseline_name}_rate"] = float(
            (per_example_df[f"delta_vs_{baseline_name}_mean_nll"] < 0.0).mean()
        )
        overall[f"first1_delta_vs_{baseline_name}_mean"] = float(
            per_example_df[f"delta_vs_{baseline_name}_first1_mean_nll"].mean()
        )
        overall[f"first3_delta_vs_{baseline_name}_mean"] = float(
            per_example_df[f"delta_vs_{baseline_name}_first3_mean_nll"].mean()
        )
        overall[f"first5_delta_vs_{baseline_name}_mean"] = float(
            per_example_df[f"delta_vs_{baseline_name}_first5_mean_nll"].mean()
        )
    summary["overall"] = overall

    grouped = (
        per_example_df.groupby(["dataset_name", "loader_variant"], dropna=False)
        .agg(
            num_examples=("example_id", "count"),
            real_mean_nll=("real_mean_nll", "mean"),
            random_mean_nll=("random_mean_nll", "mean"),
            delta_mean_nll_mean=("delta_mean_nll", "mean"),
            delta_mean_nll_median=("delta_mean_nll", "median"),
            real_beats_random_rate=("delta_mean_nll", lambda s: float((s < 0.0).mean())),
            first1_delta_mean=("delta_first1_mean_nll", "mean"),
            first3_delta_mean=("delta_first3_mean_nll", "mean"),
            first5_delta_mean=("delta_first5_mean_nll", "mean"),
        )
        .reset_index()
    )

    if include_no_hook:
        grouped["no_hook_mean_nll"] = (
            per_example_df.groupby(["dataset_name", "loader_variant"], dropna=False)["no_hook_mean_nll"]
            .mean()
            .values
        )
        grouped["real_beats_no_hook_rate"] = (
            per_example_df.groupby(["dataset_name", "loader_variant"], dropna=False)["delta_vs_no_hook_mean_nll"]
            .apply(lambda s: float((s < 0.0).mean()))
            .values
        )

    for baseline_name in optional_baselines:
        grouped[f"{baseline_name}_mean_nll"] = (
            per_example_df.groupby(["dataset_name", "loader_variant"], dropna=False)[f"{baseline_name}_mean_nll"]
            .mean()
            .values
        )
        grouped[f"real_beats_{baseline_name}_rate"] = (
            per_example_df.groupby(["dataset_name", "loader_variant"], dropna=False)[
                f"delta_vs_{baseline_name}_mean_nll"
            ]
            .apply(lambda s: float((s < 0.0).mean()))
            .values
        )
        grouped[f"first1_delta_vs_{baseline_name}_mean"] = (
            per_example_df.groupby(["dataset_name", "loader_variant"], dropna=False)[
                f"delta_vs_{baseline_name}_first1_mean_nll"
            ]
            .mean()
            .values
        )
        grouped[f"first3_delta_vs_{baseline_name}_mean"] = (
            per_example_df.groupby(["dataset_name", "loader_variant"], dropna=False)[
                f"delta_vs_{baseline_name}_first3_mean_nll"
            ]
            .mean()
            .values
        )
        grouped[f"first5_delta_vs_{baseline_name}_mean"] = (
            per_example_df.groupby(["dataset_name", "loader_variant"], dropna=False)[
                f"delta_vs_{baseline_name}_first5_mean_nll"
            ]
            .mean()
            .values
        )

    summary["by_loader"] = grouped.to_dict(orient="records")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--subset-size", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-loader", type=int, default=50)
    parser.add_argument("--length-percentile", type=float, default=0.999)
    parser.add_argument("--include-no-hook", action="store_true")
    parser.add_argument("--include-zero-vector", action="store_true")
    parser.add_argument("--include-batch-row-sample", action="store_true")
    parser.add_argument("--include-batch-repeat-single-donor", action="store_true")
    parser.add_argument("--include-batch-repeat-single-donor-shared-offset", action="store_true")
    parser.add_argument("--save-decoded-text", action="store_true")
    parser.add_argument("--heldout-eval", action="store_true")
    parser.add_argument("--heldout-past-lens-seed", type=int, default=43)
    parser.add_argument(
        "--heldout-synthetic-qa-v2-config",
        type=str,
        default="checkpoints/Qwen3-8B_synthetic_qa_v2_only/final/ao_config.json",
    )
    parser.add_argument("--past-lens-generation-buffer", type=int, default=64)
    parser.add_argument("--reuse-base-plan-from", type=str, default=None)
    args = parser.parse_args()

    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg, base_sources = build_loader_sources(checkpoint_dir)

    print(f"Resolved checkpoint dir: {checkpoint_dir}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    print(f"Scoring model: {cfg.model_name}")
    print(f"Found {len(base_sources)} train dataset loaders")

    reused_base_plan_dir = Path(args.reuse_base_plan_from) if args.reuse_base_plan_from is not None else None
    if reused_base_plan_dir is None:
        length_threshold = compute_length_threshold(base_sources, args.length_percentile)
        print(f"Post-trim length threshold at p={args.length_percentile}: {length_threshold}")
        base_kept_counts = count_kept_examples_by_source(base_sources, length_threshold)
        base_allocations = allocate_subset_sizes(base_kept_counts, args.subset_size, args.min_per_loader)
        base_plan_mode = "computed"
    else:
        assert reused_base_plan_dir.exists(), f"Missing reused base plan dir: {reused_base_plan_dir}"
        length_threshold, base_allocations, base_kept_counts = load_reused_base_plan(
            result_dir=reused_base_plan_dir,
            base_sources=base_sources,
            subset_size=args.subset_size,
        )
        base_plan_mode = "reused"
        print(f"Reused length threshold {length_threshold} and base allocations from {reused_base_plan_dir}")

    sources = base_sources
    kept_counts = base_kept_counts
    allocations = base_allocations
    allocation_mode = "checkpoint_train"

    if args.heldout_eval:
        synthetic_qa_v2_config_path = Path(args.heldout_synthetic_qa_v2_config)
        assert synthetic_qa_v2_config_path.exists(), (
            f"Missing held-out synthetic QA v2 config: {synthetic_qa_v2_config_path}"
        )
        sources = build_heldout_sources(
            base_sources=base_sources,
            base_allocations=base_allocations,
            output_dir=output_dir,
            past_lens_seed=args.heldout_past_lens_seed,
            synthetic_qa_v2_config_path=synthetic_qa_v2_config_path,
            past_lens_generation_buffer=args.past_lens_generation_buffer,
        )
        allocations = build_heldout_allocations(base_sources, base_allocations)
        kept_counts = count_kept_examples_by_source(sources, length_threshold)
        for source, kept_count, allocation in zip(sources, kept_counts, allocations, strict=True):
            assert kept_count >= allocation, (
                f"Held-out source {source.loader_variant} only has {kept_count} post-trim examples, "
                f"but allocation requires {allocation}"
            )
        allocation_mode = "heldout_current_mix"
        print(
            f"Held-out eval enabled: past_lens seed {args.heldout_past_lens_seed}, "
            f"synthetic QA from {synthetic_qa_v2_config_path}"
        )

    allocation_rows = []
    for source, kept_count, allocation in zip(sources, kept_counts, allocations, strict=True):
        allocation_rows.append(
            {
                "loader_index": source.loader_index,
                "dataset_name": source.dataset_name,
                "loader_variant": source.loader_variant,
                "source_name": source.source_name,
                "split": source.split,
                "kept_count": kept_count,
                "allocation": allocation,
            }
        )
    allocation_df = pd.DataFrame(allocation_rows)
    allocation_df.to_csv(output_dir / "subset_allocation.csv", index=False)

    selected_examples = sample_subset(sources, allocations, length_threshold, args.seed)
    selected_examples.sort(
        key=lambda example: (
            len(example.dp.input_ids),
            len(example.dp.context_input_ids) if example.dp.context_input_ids is not None else 0,
        ),
        reverse=True,
    )
    print(f"Selected {len(selected_examples)} examples for scoring")

    set_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    tokenizer = load_tokenizer(cfg.model_name)

    base_model = load_model(cfg.model_name, dtype)
    model = PeftModel.from_pretrained(base_model, str(checkpoint_dir), is_trainable=False, autocast_adapter_dtype=True)
    assert isinstance(model, PeftModel)
    model.eval()
    submodule = get_hf_submodule(model, cfg.hook_onto_layer)

    example_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(args.seed + 1)

    batch_starts = range(0, len(selected_examples), args.batch_size)
    for batch_start in tqdm(batch_starts, total=math.ceil(len(selected_examples) / args.batch_size), desc="Scoring"):
        batch_examples = selected_examples[batch_start : batch_start + args.batch_size]
        if (
            args.include_batch_row_sample
            or args.include_batch_repeat_single_donor
            or args.include_batch_repeat_single_donor_shared_offset
        ) and len(batch_examples) < 2:
            raise ValueError(
                "Batch-donor baselines require at least 2 examples in every batch. "
                "Use a subset size divisible by batch size or disable the donor baselines."
            )
        batch_points = [example.dp for example in batch_examples]
        materialized_points = materialize_missing_steering_vectors(batch_points, tokenizer, model)

        for example, dp in zip(batch_examples, materialized_points, strict=True):
            example.dp = dp

        real_batch = construct_batch(materialized_points, tokenizer, device)
        random_points = build_randomized_points(batch_examples, random_generator)
        random_batch = construct_batch(random_points, tokenizer, device)
        zero_vector_batch = None
        if args.include_zero_vector:
            zero_vector_points = build_zero_vector_points(batch_examples)
            zero_vector_batch = construct_batch(zero_vector_points, tokenizer, device)
        batch_row_sample_batch = None
        if args.include_batch_row_sample:
            batch_row_sample_points = build_batch_row_sampled_points(batch_examples, random_generator)
            batch_row_sample_batch = construct_batch(batch_row_sample_points, tokenizer, device)
        batch_repeat_single_donor_batch = None
        if args.include_batch_repeat_single_donor:
            batch_repeat_single_donor_points = build_batch_repeat_single_donor_points(batch_examples, random_generator)
            batch_repeat_single_donor_batch = construct_batch(batch_repeat_single_donor_points, tokenizer, device)
        batch_repeat_single_donor_shared_offset_batch = None
        if args.include_batch_repeat_single_donor_shared_offset:
            batch_repeat_single_donor_shared_offset_points = build_batch_repeat_single_donor_shared_offset_points(
                batch_examples,
                random_generator,
            )
            batch_repeat_single_donor_shared_offset_batch = construct_batch(
                batch_repeat_single_donor_shared_offset_points,
                tokenizer,
                device,
            )

        real_logits = steered_logits(
            model=model,
            submodule=submodule,
            batch=real_batch,
            steering_coefficient=cfg.steering_coefficient,
            device=device,
            dtype=dtype,
            use_hook=True,
        )
        random_logits = steered_logits(
            model=model,
            submodule=submodule,
            batch=random_batch,
            steering_coefficient=cfg.steering_coefficient,
            device=device,
            dtype=dtype,
            use_hook=True,
        )
        zero_vector_logits = None
        if args.include_zero_vector:
            assert zero_vector_batch is not None
            zero_vector_logits = steered_logits(
                model=model,
                submodule=submodule,
                batch=zero_vector_batch,
                steering_coefficient=cfg.steering_coefficient,
                device=device,
                dtype=dtype,
                use_hook=True,
            )
        batch_row_sample_logits = None
        if args.include_batch_row_sample:
            assert batch_row_sample_batch is not None
            batch_row_sample_logits = steered_logits(
                model=model,
                submodule=submodule,
                batch=batch_row_sample_batch,
                steering_coefficient=cfg.steering_coefficient,
                device=device,
                dtype=dtype,
                use_hook=True,
            )
        batch_repeat_single_donor_logits = None
        if args.include_batch_repeat_single_donor:
            assert batch_repeat_single_donor_batch is not None
            batch_repeat_single_donor_logits = steered_logits(
                model=model,
                submodule=submodule,
                batch=batch_repeat_single_donor_batch,
                steering_coefficient=cfg.steering_coefficient,
                device=device,
                dtype=dtype,
                use_hook=True,
            )
        batch_repeat_single_donor_shared_offset_logits = None
        if args.include_batch_repeat_single_donor_shared_offset:
            assert batch_repeat_single_donor_shared_offset_batch is not None
            batch_repeat_single_donor_shared_offset_logits = steered_logits(
                model=model,
                submodule=submodule,
                batch=batch_repeat_single_donor_shared_offset_batch,
                steering_coefficient=cfg.steering_coefficient,
                device=device,
                dtype=dtype,
                use_hook=True,
            )
        no_hook_logits = None
        if args.include_no_hook:
            no_hook_logits = steered_logits(
                model=model,
                submodule=submodule,
                batch=real_batch,
                steering_coefficient=cfg.steering_coefficient,
                device=device,
                dtype=dtype,
                use_hook=False,
            )

        real_losses, valid_mask = compute_shifted_token_losses(real_logits, real_batch.labels)
        random_losses, _ = compute_shifted_token_losses(random_logits, real_batch.labels)
        zero_vector_losses = None
        if args.include_zero_vector:
            assert zero_vector_logits is not None
            zero_vector_losses, _ = compute_shifted_token_losses(zero_vector_logits, real_batch.labels)
        batch_row_sample_losses = None
        if args.include_batch_row_sample:
            assert batch_row_sample_logits is not None
            batch_row_sample_losses, _ = compute_shifted_token_losses(batch_row_sample_logits, real_batch.labels)
        batch_repeat_single_donor_losses = None
        if args.include_batch_repeat_single_donor:
            assert batch_repeat_single_donor_logits is not None
            batch_repeat_single_donor_losses, _ = compute_shifted_token_losses(
                batch_repeat_single_donor_logits,
                real_batch.labels,
            )
        batch_repeat_single_donor_shared_offset_losses = None
        if args.include_batch_repeat_single_donor_shared_offset:
            assert batch_repeat_single_donor_shared_offset_logits is not None
            batch_repeat_single_donor_shared_offset_losses, _ = compute_shifted_token_losses(
                batch_repeat_single_donor_shared_offset_logits,
                real_batch.labels,
            )
        no_hook_losses = None
        if args.include_no_hook:
            assert no_hook_logits is not None
            no_hook_losses, _ = compute_shifted_token_losses(no_hook_logits, real_batch.labels)

        first1_mask = first_n_mask(valid_mask, 1)
        first3_mask = first_n_mask(valid_mask, 3)
        first5_mask = first_n_mask(valid_mask, 5)

        for row_idx, example in enumerate(batch_examples):
            dp = example.dp
            assert dp.steering_vectors is not None, "Expected steering vectors after materialization"

            valid_positions = torch.nonzero(valid_mask[row_idx], as_tuple=False).flatten()
            assert len(valid_positions) > 0, "Each datapoint should have at least one labeled token"

            real_sum = real_losses[row_idx][valid_mask[row_idx]].sum().item()
            random_sum = random_losses[row_idx][valid_mask[row_idx]].sum().item()
            token_count = int(valid_mask[row_idx].sum().item())

            first1_real = real_losses[row_idx][first1_mask[row_idx]].mean().item()
            first1_random = random_losses[row_idx][first1_mask[row_idx]].mean().item()
            first3_real = real_losses[row_idx][first3_mask[row_idx]].mean().item()
            first3_random = random_losses[row_idx][first3_mask[row_idx]].mean().item()
            first5_real = real_losses[row_idx][first5_mask[row_idx]].mean().item()
            first5_random = random_losses[row_idx][first5_mask[row_idx]].mean().item()

            labels_cpu = real_batch.labels[row_idx].detach().cpu()
            input_ids_cpu = real_batch.input_ids[row_idx].detach().cpu()
            input_length = int((real_batch.attention_mask[row_idx]).sum().item())

            meta_info_json = json.dumps(dict(dp.meta_info), sort_keys=True)
            row = {
                "example_id": example.example_id,
                "loader_index": example.loader_index,
                "dataset_name": example.dataset_name,
                "loader_variant": example.loader_variant,
                "source_name": example.source_name,
                "source_split": example.split,
                "loader_config_json": example.loader_config_json,
                "datapoint_type": dp.datapoint_type,
                "layers_json": json.dumps(dp.layers),
                "num_layers": len(dp.layers),
                "num_positions": len(dp.context_positions) if dp.context_positions is not None else None,
                "input_length": input_length,
                "context_length": len(dp.context_input_ids) if dp.context_input_ids is not None else None,
                "target_token_count": token_count,
                "feature_idx": dp.feature_idx,
                "target_output": dp.target_output,
                "ds_label": dp.ds_label,
                "meta_info_json": meta_info_json,
                "real_total_nll": real_sum,
                "real_mean_nll": real_sum / token_count,
                "random_total_nll": random_sum,
                "random_mean_nll": random_sum / token_count,
                "delta_mean_nll": (real_sum - random_sum) / token_count,
                "delta_total_nll": real_sum - random_sum,
                "delta_first1_mean_nll": first1_real - first1_random,
                "delta_first3_mean_nll": first3_real - first3_random,
                "delta_first5_mean_nll": first5_real - first5_random,
                "real_first1_mean_nll": first1_real,
                "random_first1_mean_nll": first1_random,
                "real_first3_mean_nll": first3_real,
                "random_first3_mean_nll": first3_random,
                "real_first5_mean_nll": first5_real,
                "random_first5_mean_nll": first5_random,
            }
            row.update(extract_common_meta(dp))

            if args.save_decoded_text:
                row["input_text"] = tokenizer.decode(input_ids_cpu[-input_length:], skip_special_tokens=False)
                row["prompt_text"] = decode_prompt_only(dp, tokenizer)
                row["context_text"] = (
                    tokenizer.decode(dp.context_input_ids, skip_special_tokens=False)
                    if dp.context_input_ids is not None
                    else None
                )

            if args.include_no_hook:
                assert no_hook_losses is not None
                no_hook_sum = no_hook_losses[row_idx][valid_mask[row_idx]].sum().item()
                row["no_hook_total_nll"] = no_hook_sum
                row["no_hook_mean_nll"] = no_hook_sum / token_count
                row["delta_vs_no_hook_mean_nll"] = (real_sum - no_hook_sum) / token_count
                row["delta_vs_no_hook_total_nll"] = real_sum - no_hook_sum

            if args.include_zero_vector:
                assert zero_vector_losses is not None
                zero_vector_sum = zero_vector_losses[row_idx][valid_mask[row_idx]].sum().item()
                first1_zero_vector = zero_vector_losses[row_idx][first1_mask[row_idx]].mean().item()
                first3_zero_vector = zero_vector_losses[row_idx][first3_mask[row_idx]].mean().item()
                first5_zero_vector = zero_vector_losses[row_idx][first5_mask[row_idx]].mean().item()
                row["zero_vector_total_nll"] = zero_vector_sum
                row["zero_vector_mean_nll"] = zero_vector_sum / token_count
                row["delta_vs_zero_vector_mean_nll"] = (real_sum - zero_vector_sum) / token_count
                row["delta_vs_zero_vector_total_nll"] = real_sum - zero_vector_sum
                row["zero_vector_first1_mean_nll"] = first1_zero_vector
                row["zero_vector_first3_mean_nll"] = first3_zero_vector
                row["zero_vector_first5_mean_nll"] = first5_zero_vector
                row["delta_vs_zero_vector_first1_mean_nll"] = first1_real - first1_zero_vector
                row["delta_vs_zero_vector_first3_mean_nll"] = first3_real - first3_zero_vector
                row["delta_vs_zero_vector_first5_mean_nll"] = first5_real - first5_zero_vector

            if args.include_batch_row_sample:
                assert batch_row_sample_losses is not None
                batch_row_sample_sum = batch_row_sample_losses[row_idx][valid_mask[row_idx]].sum().item()
                first1_batch_row_sample = batch_row_sample_losses[row_idx][first1_mask[row_idx]].mean().item()
                first3_batch_row_sample = batch_row_sample_losses[row_idx][first3_mask[row_idx]].mean().item()
                first5_batch_row_sample = batch_row_sample_losses[row_idx][first5_mask[row_idx]].mean().item()
                row["batch_row_sample_total_nll"] = batch_row_sample_sum
                row["batch_row_sample_mean_nll"] = batch_row_sample_sum / token_count
                row["delta_vs_batch_row_sample_mean_nll"] = (real_sum - batch_row_sample_sum) / token_count
                row["delta_vs_batch_row_sample_total_nll"] = real_sum - batch_row_sample_sum
                row["batch_row_sample_first1_mean_nll"] = first1_batch_row_sample
                row["batch_row_sample_first3_mean_nll"] = first3_batch_row_sample
                row["batch_row_sample_first5_mean_nll"] = first5_batch_row_sample
                row["delta_vs_batch_row_sample_first1_mean_nll"] = first1_real - first1_batch_row_sample
                row["delta_vs_batch_row_sample_first3_mean_nll"] = first3_real - first3_batch_row_sample
                row["delta_vs_batch_row_sample_first5_mean_nll"] = first5_real - first5_batch_row_sample

            if args.include_batch_repeat_single_donor:
                assert batch_repeat_single_donor_losses is not None
                batch_repeat_single_donor_sum = batch_repeat_single_donor_losses[row_idx][valid_mask[row_idx]].sum().item()
                first1_batch_repeat_single_donor = (
                    batch_repeat_single_donor_losses[row_idx][first1_mask[row_idx]].mean().item()
                )
                first3_batch_repeat_single_donor = (
                    batch_repeat_single_donor_losses[row_idx][first3_mask[row_idx]].mean().item()
                )
                first5_batch_repeat_single_donor = (
                    batch_repeat_single_donor_losses[row_idx][first5_mask[row_idx]].mean().item()
                )
                row["batch_repeat_single_donor_total_nll"] = batch_repeat_single_donor_sum
                row["batch_repeat_single_donor_mean_nll"] = batch_repeat_single_donor_sum / token_count
                row["delta_vs_batch_repeat_single_donor_mean_nll"] = (
                    real_sum - batch_repeat_single_donor_sum
                ) / token_count
                row["delta_vs_batch_repeat_single_donor_total_nll"] = real_sum - batch_repeat_single_donor_sum
                row["batch_repeat_single_donor_first1_mean_nll"] = first1_batch_repeat_single_donor
                row["batch_repeat_single_donor_first3_mean_nll"] = first3_batch_repeat_single_donor
                row["batch_repeat_single_donor_first5_mean_nll"] = first5_batch_repeat_single_donor
                row["delta_vs_batch_repeat_single_donor_first1_mean_nll"] = (
                    first1_real - first1_batch_repeat_single_donor
                )
                row["delta_vs_batch_repeat_single_donor_first3_mean_nll"] = (
                    first3_real - first3_batch_repeat_single_donor
                )
                row["delta_vs_batch_repeat_single_donor_first5_mean_nll"] = (
                    first5_real - first5_batch_repeat_single_donor
                )

            if args.include_batch_repeat_single_donor_shared_offset:
                assert batch_repeat_single_donor_shared_offset_losses is not None
                batch_repeat_single_donor_shared_offset_sum = (
                    batch_repeat_single_donor_shared_offset_losses[row_idx][valid_mask[row_idx]].sum().item()
                )
                first1_batch_repeat_single_donor_shared_offset = (
                    batch_repeat_single_donor_shared_offset_losses[row_idx][first1_mask[row_idx]].mean().item()
                )
                first3_batch_repeat_single_donor_shared_offset = (
                    batch_repeat_single_donor_shared_offset_losses[row_idx][first3_mask[row_idx]].mean().item()
                )
                first5_batch_repeat_single_donor_shared_offset = (
                    batch_repeat_single_donor_shared_offset_losses[row_idx][first5_mask[row_idx]].mean().item()
                )
                row["batch_repeat_single_donor_shared_offset_total_nll"] = (
                    batch_repeat_single_donor_shared_offset_sum
                )
                row["batch_repeat_single_donor_shared_offset_mean_nll"] = (
                    batch_repeat_single_donor_shared_offset_sum / token_count
                )
                row["delta_vs_batch_repeat_single_donor_shared_offset_mean_nll"] = (
                    real_sum - batch_repeat_single_donor_shared_offset_sum
                ) / token_count
                row["delta_vs_batch_repeat_single_donor_shared_offset_total_nll"] = (
                    real_sum - batch_repeat_single_donor_shared_offset_sum
                )
                row["batch_repeat_single_donor_shared_offset_first1_mean_nll"] = (
                    first1_batch_repeat_single_donor_shared_offset
                )
                row["batch_repeat_single_donor_shared_offset_first3_mean_nll"] = (
                    first3_batch_repeat_single_donor_shared_offset
                )
                row["batch_repeat_single_donor_shared_offset_first5_mean_nll"] = (
                    first5_batch_repeat_single_donor_shared_offset
                )
                row["delta_vs_batch_repeat_single_donor_shared_offset_first1_mean_nll"] = (
                    first1_real - first1_batch_repeat_single_donor_shared_offset
                )
                row["delta_vs_batch_repeat_single_donor_shared_offset_first3_mean_nll"] = (
                    first3_real - first3_batch_repeat_single_donor_shared_offset
                )
                row["delta_vs_batch_repeat_single_donor_shared_offset_first5_mean_nll"] = (
                    first5_real - first5_batch_repeat_single_donor_shared_offset
                )

            example_rows.append(row)

            for token_rank, shifted_pos in enumerate(valid_positions.tolist()):
                target_pos = shifted_pos + 1
                target_token_id = int(labels_cpu[target_pos].item())
                token_row = {
                    "example_id": example.example_id,
                    "token_rank": token_rank,
                    "sequence_position": target_pos,
                    "target_token_id": target_token_id,
                    "target_token_text": tokenizer.decode([target_token_id], skip_special_tokens=False),
                    "real_nll": float(real_losses[row_idx, shifted_pos].item()),
                    "random_nll": float(random_losses[row_idx, shifted_pos].item()),
                    "delta_nll": float(real_losses[row_idx, shifted_pos].item() - random_losses[row_idx, shifted_pos].item()),
                    "dataset_name": example.dataset_name,
                    "loader_variant": example.loader_variant,
                    "direction": row["direction"],
                    "k_tokens": row["k_tokens"],
                    "k_acts": row["k_acts"],
                }
                if args.include_no_hook:
                    assert no_hook_losses is not None
                    token_row["no_hook_nll"] = float(no_hook_losses[row_idx, shifted_pos].item())
                    token_row["delta_vs_no_hook_nll"] = float(
                        real_losses[row_idx, shifted_pos].item() - no_hook_losses[row_idx, shifted_pos].item()
                    )
                if args.include_zero_vector:
                    assert zero_vector_losses is not None
                    token_row["zero_vector_nll"] = float(zero_vector_losses[row_idx, shifted_pos].item())
                    token_row["delta_vs_zero_vector_nll"] = float(
                        real_losses[row_idx, shifted_pos].item() - zero_vector_losses[row_idx, shifted_pos].item()
                    )
                if args.include_batch_row_sample:
                    assert batch_row_sample_losses is not None
                    token_row["batch_row_sample_nll"] = float(batch_row_sample_losses[row_idx, shifted_pos].item())
                    token_row["delta_vs_batch_row_sample_nll"] = float(
                        real_losses[row_idx, shifted_pos].item() - batch_row_sample_losses[row_idx, shifted_pos].item()
                    )
                if args.include_batch_repeat_single_donor:
                    assert batch_repeat_single_donor_losses is not None
                    token_row["batch_repeat_single_donor_nll"] = float(
                        batch_repeat_single_donor_losses[row_idx, shifted_pos].item()
                    )
                    token_row["delta_vs_batch_repeat_single_donor_nll"] = float(
                        real_losses[row_idx, shifted_pos].item()
                        - batch_repeat_single_donor_losses[row_idx, shifted_pos].item()
                    )
                if args.include_batch_repeat_single_donor_shared_offset:
                    assert batch_repeat_single_donor_shared_offset_losses is not None
                    token_row["batch_repeat_single_donor_shared_offset_nll"] = float(
                        batch_repeat_single_donor_shared_offset_losses[row_idx, shifted_pos].item()
                    )
                    token_row["delta_vs_batch_repeat_single_donor_shared_offset_nll"] = float(
                        real_losses[row_idx, shifted_pos].item()
                        - batch_repeat_single_donor_shared_offset_losses[row_idx, shifted_pos].item()
                    )
                token_rows.append(token_row)

        del real_batch, random_batch, real_logits, random_logits, real_losses, random_losses, valid_mask
        del first1_mask, first3_mask, first5_mask
        if zero_vector_batch is not None:
            del zero_vector_batch
        if zero_vector_logits is not None:
            del zero_vector_logits
        if zero_vector_losses is not None:
            del zero_vector_losses
        if batch_row_sample_batch is not None:
            del batch_row_sample_batch
        if batch_row_sample_logits is not None:
            del batch_row_sample_logits
        if batch_row_sample_losses is not None:
            del batch_row_sample_losses
        if batch_repeat_single_donor_batch is not None:
            del batch_repeat_single_donor_batch
        if batch_repeat_single_donor_logits is not None:
            del batch_repeat_single_donor_logits
        if batch_repeat_single_donor_losses is not None:
            del batch_repeat_single_donor_losses
        if batch_repeat_single_donor_shared_offset_batch is not None:
            del batch_repeat_single_donor_shared_offset_batch
        if batch_repeat_single_donor_shared_offset_logits is not None:
            del batch_repeat_single_donor_shared_offset_logits
        if batch_repeat_single_donor_shared_offset_losses is not None:
            del batch_repeat_single_donor_shared_offset_losses
        if no_hook_logits is not None:
            del no_hook_logits
        if no_hook_losses is not None:
            del no_hook_losses
        gc.collect()
        torch.cuda.empty_cache()

    per_example_df = pd.DataFrame(example_rows)
    per_token_df = pd.DataFrame(token_rows)

    per_example_df.to_parquet(output_dir / "per_example.parquet", index=False)
    per_token_df.to_parquet(output_dir / "per_token.parquet", index=False)

    summary = summarize_scores(per_example_df, include_no_hook=args.include_no_hook)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    run_config = {
        "checkpoint_dir": str(checkpoint_dir),
        "subset_size": args.subset_size,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "min_per_loader": args.min_per_loader,
        "length_percentile": args.length_percentile,
        "length_threshold": length_threshold,
        "allocation_mode": allocation_mode,
        "base_plan_mode": base_plan_mode,
        "include_no_hook": args.include_no_hook,
        "include_zero_vector": args.include_zero_vector,
        "include_batch_row_sample": args.include_batch_row_sample,
        "include_batch_repeat_single_donor": args.include_batch_repeat_single_donor,
        "include_batch_repeat_single_donor_shared_offset": args.include_batch_repeat_single_donor_shared_offset,
        "save_decoded_text": args.save_decoded_text,
        "heldout_eval": args.heldout_eval,
        "heldout_past_lens_seed": args.heldout_past_lens_seed,
        "heldout_synthetic_qa_v2_config": args.heldout_synthetic_qa_v2_config,
        "past_lens_generation_buffer": args.past_lens_generation_buffer,
        "reuse_base_plan_from": args.reuse_base_plan_from,
        "scoring_modes": (
            ["real", "random_direction"]
            + (["zero_vector"] if args.include_zero_vector else [])
            + (["batch_row_sample"] if args.include_batch_row_sample else [])
            + (["batch_repeat_single_donor"] if args.include_batch_repeat_single_donor else [])
            + (
                ["batch_repeat_single_donor_shared_offset"]
                if args.include_batch_repeat_single_donor_shared_offset
                else []
            )
            + (["no_hook"] if args.include_no_hook else [])
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "loader_sources": allocation_rows,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    print("\nOverall summary")
    print(json.dumps(summary["overall"], indent=2))
    print("\nTop loader summaries by mean delta (real - random, lower is better)")
    top_rows = sorted(summary["by_loader"], key=lambda row: row["delta_mean_nll_mean"])
    for row in top_rows[:10]:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
