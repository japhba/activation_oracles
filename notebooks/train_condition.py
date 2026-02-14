"""Torchrun training script for AO robustness experiment.

Condition A (baseline): LatentQA + Classification + PastLens
Condition B (open-ended): LatentQA + SQuAD QA + PastLens

Usage:
    torchrun --nproc_per_node=N notebooks/train_condition.py --condition a --model_name Qwen/Qwen3-8B
    torchrun --nproc_per_node=N notebooks/train_condition.py --condition b --model_name Qwen/Qwen3-8B
"""

import os

from dotenv import load_dotenv
load_dotenv()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from dataclasses import asdict
from datetime import timedelta

import torch
import torch.distributed as dist

from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.dataset_classes.act_dataset_manager import ActDatasetLoader
from nl_probes.dataset_classes.classification import ClassificationDatasetConfig, ClassificationDatasetLoader
from nl_probes.dataset_classes.latentqa_dataset import LatentQADatasetConfig, LatentQADatasetLoader
from nl_probes.dataset_classes.past_lens_dataset import PastLensDatasetConfig, PastLensDatasetLoader
from nl_probes.dataset_classes.squad_qa import SquadQADatasetConfig, SquadQADatasetLoader
from nl_probes.sft import _ensure_datasets_exist, build_datasets, mk_cfg, train_model
from nl_probes.utils.common import load_tokenizer


# ── Shared dataset definitions ─────────────────────────────────────────────────

CLASSIFICATION_DATASETS = {
    "geometry_of_truth": {"num_train": 6000, "num_test": 250, "splits": ["train", "test"]},
    "relations": {"num_train": 6000, "num_test": 250, "splits": ["train", "test"]},
    "sst2": {"num_train": 6000, "num_test": 250, "splits": ["train", "test"]},
    "md_gender": {"num_train": 6000, "num_test": 250, "splits": ["train", "test"]},
    "snli": {"num_train": 6000, "num_test": 250, "splits": ["train", "test"]},
    "ag_news": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "ner": {"num_train": 6000, "num_test": 250, "splits": ["train", "test"]},
    "tense": {"num_train": 6000, "num_test": 250, "splits": ["train", "test"]},
    "language_identification": {"num_train": 6000, "num_test": 250, "splits": ["test"], "batch_size": 4},
    "singular_plural": {"num_train": 0, "num_test": 250, "splits": ["test"]},
}


def _shared_loaders(
    model_name: str, layer_percents: list[int], batch_size: int, save_acts: bool,
) -> tuple[list[ActDatasetLoader], list[ActDatasetLoader]]:
    """Return (past_lens_loaders, latentqa_loaders) shared by both conditions."""
    past_lens_single = PastLensDatasetLoader(
        dataset_config=mk_cfg(
            PastLensDatasetConfig(max_k_activations=1, max_k_tokens=50),
            num_train=100_000, num_test=0, splits=["train"],
            model_name=model_name, layer_percents=layer_percents,
            save_acts=save_acts, batch_size=batch_size,
        )
    )
    past_lens_multi = PastLensDatasetLoader(
        dataset_config=mk_cfg(
            PastLensDatasetConfig(max_k_activations=50, max_k_tokens=50),
            num_train=100_000, num_test=0, splits=["train"],
            model_name=model_name, layer_percents=layer_percents,
            save_acts=save_acts, batch_size=batch_size,
        )
    )
    latentqa = LatentQADatasetLoader(
        dataset_config=mk_cfg(
            LatentQADatasetConfig(),
            num_train=100_000, num_test=0, splits=["train"],
            model_name=model_name, layer_percents=layer_percents,
            save_acts=False, batch_size=batch_size,
        )
    )
    return [past_lens_single, past_lens_multi], [latentqa]


def _classification_loaders(
    model_name: str, layer_percents: list[int], batch_size: int, save_acts: bool,
    model_kwargs: dict,
) -> list[ActDatasetLoader]:
    """Build single + multi-token classification loaders (Condition A)."""
    loaders: list[ActDatasetLoader] = []
    for ds_name, meta in CLASSIFICATION_DATASETS.items():
        bs = meta.get("batch_size", batch_size)
        for params in [
            ClassificationDatasetConfig(classification_dataset_name=ds_name, max_window_size=1, min_end_offset=-1, max_end_offset=-5, num_qa_per_sample=2),
            ClassificationDatasetConfig(classification_dataset_name=ds_name, max_window_size=50, min_end_offset=-1, max_end_offset=-5, num_qa_per_sample=1),
        ]:
            loaders.append(ClassificationDatasetLoader(
                dataset_config=mk_cfg(
                    params,
                    num_train=meta["num_train"], num_test=meta["num_test"], splits=meta["splits"],
                    model_name=model_name, layer_percents=layer_percents,
                    save_acts=save_acts, batch_size=bs,
                ),
                model_kwargs=model_kwargs,
            ))
    return loaders


def _squad_loaders(
    model_name: str, layer_percents: list[int], batch_size: int, save_acts: bool,
    model_kwargs: dict,
) -> list[ActDatasetLoader]:
    """Build single + multi-token SQuAD QA loaders (Condition B)."""
    loaders: list[ActDatasetLoader] = []
    for params in [
        SquadQADatasetConfig(max_window_size=1, min_end_offset=-1, max_end_offset=-5),
        SquadQADatasetConfig(max_window_size=50, min_end_offset=-1, max_end_offset=-5),
    ]:
        loaders.append(SquadQADatasetLoader(
            dataset_config=mk_cfg(
                params,
                num_train=60_000, num_test=250, splits=["train", "test"],
                model_name=model_name, layer_percents=layer_percents,
                save_acts=save_acts, batch_size=batch_size,
            ),
            model_kwargs=model_kwargs,
        ))
    return loaders


def build_condition_loaders(
    condition: str, model_name: str, layer_percents: list[int],
    batch_size: int, save_acts: bool, model_kwargs: dict,
) -> list[ActDatasetLoader]:
    past_lens, latentqa = _shared_loaders(model_name, layer_percents, batch_size, save_acts)
    if condition == "a":
        task_loaders = _classification_loaders(model_name, layer_percents, batch_size, save_acts, model_kwargs)
    else:
        task_loaders = _squad_loaders(model_name, layer_percents, batch_size, save_acts, model_kwargs)
    return latentqa + task_loaders + past_lens


# ── Main ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["a", "b"], required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--wandb_suffix", type=str, default="")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()

    model_name = args.model_name
    model_name_str = model_name.split("/")[-1].replace(".", "_").replace(" ", "_")
    condition_label = {"a": "baseline_cls", "b": "openended_squad"}[args.condition]
    wandb_suffix = args.wandb_suffix or f"_{condition_label}_{model_name_str}"

    layer_percents = [25, 50, 75]
    save_acts = False
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")

    global_batch_size = 16
    assert global_batch_size % world_size == 0
    per_rank_batch = global_batch_size // world_size

    model_kwargs: dict = {}

    loaders = build_condition_loaders(
        args.condition, model_name, layer_percents, per_rank_batch, save_acts, model_kwargs,
    )

    cfg = SelfInterpTrainingConfig(
        model_name=model_name,
        hook_onto_layer=1,
        layer_percents=layer_percents,
        train_batch_size=per_rank_batch,
        activation_collection_batch_size=per_rank_batch * 4,
        eval_batch_size=per_rank_batch * 8,
        eval_steps=1_000,
        eval_on_start=True,
        gradient_checkpointing=True,
        wandb_suffix=wandb_suffix,
    )
    cfg.finalize(dataset_loaders=loaders)
    print(f"Condition {args.condition.upper()} | save_dir: {cfg.save_dir}")

    tokenizer = load_tokenizer(model_name)

    if local_rank == 0:
        _ensure_datasets_exist(loaders)
    dist.barrier()

    all_training_data, all_eval_data = build_datasets(cfg, dataset_loaders=loaders, window_mult=cfg.window_mult)
    print(f"training: {len(all_training_data)}, eval datasets: {list(all_eval_data.keys())}")
    print(asdict(cfg))

    train_model(
        cfg=cfg,
        training_data=all_training_data,
        eval_datasets=all_eval_data,
        tokenizer=tokenizer,
        dtype=dtype,
        device=device,
        model_kwargs=model_kwargs,
        verbose=True,
    )

    dist.destroy_process_group()
