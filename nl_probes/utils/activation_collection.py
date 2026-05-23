from dataclasses import dataclass

from peft import PeftModel
from transformers import PreTrainedTokenizer

from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.utils.dataset_utils import TrainingDataPoint, materialize_missing_steering_vectors


def materialize_block_into_batches(
    cfg: SelfInterpTrainingConfig,
    block: list[TrainingDataPoint],
    tokenizer: PreTrainedTokenizer,
    model: PeftModel,
    *,
    reference: bool = False,
) -> list[list[TrainingDataPoint]]:
    """Materialize one block's steering vectors and split into train batches.

    The caller must pass exactly one materialization block of
    `train_batch_size * train_batches_per_materialization_block` datapoints.
    This keeps steering-vector lifetimes bounded to one local window instead of
    accidentally caching an entire epoch.

    By default this uses the efficient path: sort only the missing-vector
    examples within the block, materialize them in fixed `train_batch_size`
    chunks, then scatter the results back into the original train order.

    Set `reference=True` to use the simpler correctness/debug path instead.
    The reference path materializes the block in its original order without
    efficient sorting or grouping, so it is easier to reason about but more
    pad-heavy.
    """
    assert cfg.train_batches_per_materialization_block > 0, (
        "train_batches_per_materialization_block must be positive"
    )
    block_size = cfg.train_batch_size * cfg.train_batches_per_materialization_block

    assert len(block) == block_size, (
        "block must contain exactly one full materialization block"
    )

    if reference:
        materialized_block = materialize_missing_steering_vectors(block, tokenizer, model)
    else:
        materialized_block = materialize_training_block(cfg, block, tokenizer, model)

    materialized_batches: list[list[TrainingDataPoint]] = []
    for batch_start in range(0, block_size, cfg.train_batch_size):
        train_batch = materialized_block[batch_start : batch_start + cfg.train_batch_size]
        assert len(train_batch) == cfg.train_batch_size, "Expected a full train batch after materialization"
        materialized_batches.append(train_batch)

    return materialized_batches


def materialize_training_block(
    cfg: SelfInterpTrainingConfig,
    block: list[TrainingDataPoint],
    tokenizer: PreTrainedTokenizer,
    model: PeftModel,
) -> list[TrainingDataPoint]:
    """Efficiently materialize one activation-collection block.

    We sort only the examples missing steering vectors by context length,
    materialize them in fixed `train_batch_size` chunks for better padding
    efficiency, then scatter the resulting steering vectors back into the
    original block order before training continues.
    """
    materialized_block = list(block)
    materialization_groups = build_materialization_groups(block, train_batch_size=cfg.train_batch_size)
    if not materialization_groups:
        # Nothing in this block needs activation collection; return it unchanged.
        return materialized_block

    was_training = model.training
    model.eval()
    try:
        with model.disable_adapter():
            for materialization_group in materialization_groups:
                materialized_group = materialize_missing_steering_vectors(
                    [dp for _, dp in materialization_group],
                    tokenizer,
                    model,
                    manage_model_state=False,
                )
                # Preserve the original training order even though collection ran on
                # sorted groups chosen only for padding efficiency.
                for (original_idx, _), materialized_dp in zip(materialization_group, materialized_group, strict=True):
                    materialized_block[original_idx] = materialized_dp
    finally:
        if was_training:
            model.train()

    return materialized_block


def build_materialization_groups(
    block: list[TrainingDataPoint],
    *,
    train_batch_size: int,
) -> list[list[tuple[int, TrainingDataPoint]]]:
    """Sort missing-vector examples by context length, then chunk by train batch size.

    This is the simple efficient path: sort the missing examples longest-first so
    similarly sized contexts land together, then materialize them in fixed
    `train_batch_size` chunks. It reduces padding substantially without adding
    the extra complexity of a token-budget packing rule.
    """
    assert train_batch_size > 0, "train_batch_size must be positive"

    missing_points = [(idx, dp) for idx, dp in enumerate(block) if dp.steering_vectors is None]
    # Longest-first makes the running padded-token cost easy to reason about:
    # each fixed-size chunk then groups together similarly sized contexts.
    missing_points.sort(
        key=lambda item: len(item[1].context_input_ids),
        reverse=True,
    )
    if not missing_points:
        return []
    return [missing_points[i : i + train_batch_size] for i in range(0, len(missing_points), train_batch_size)]


# Debug helpers below. These are only used by the quick/debug loop and tests.

@dataclass
class ActivationCollectionTokenStats:
    """Debug-only token accounting for comparing reference vs efficient collection."""

    num_missing_points: int
    num_materialization_batches: int
    actual_tokens: int
    padded_tokens: int

    @property
    def padding_tokens(self) -> int:
        return self.padded_tokens - self.actual_tokens

    @property
    def padding_fraction(self) -> float:
        if self.padded_tokens == 0:
            return 0.0
        return self.padding_tokens / self.padded_tokens


def estimate_activation_collection_token_stats(
    cfg: SelfInterpTrainingConfig,
    training_data: list[TrainingDataPoint],
    *,
    reference: bool,
) -> ActivationCollectionTokenStats:
    """Debug-only token accounting used by quick-loop benchmarking output."""
    block_size = cfg.train_batch_size * cfg.train_batches_per_materialization_block
    assert len(training_data) % block_size == 0, (
        "training_data length must be divisible by the materialization block size for exact token accounting"
    )

    stats = ActivationCollectionTokenStats(
        num_missing_points=0,
        num_materialization_batches=0,
        actual_tokens=0,
        padded_tokens=0,
    )

    for block_start in range(0, len(training_data), block_size):
        block = training_data[block_start : block_start + block_size]
        assert len(block) == block_size, (
            "Expected full materialization blocks for token accounting"
        )

        if reference:
            materialization_groups = [
                [(idx, dp) for idx, dp in enumerate(block) if dp.steering_vectors is None]
            ]
            if not materialization_groups[0]:
                materialization_groups = []
        else:
            materialization_groups = build_materialization_groups(block, train_batch_size=cfg.train_batch_size)

        for materialization_group in materialization_groups:
            group_lengths = []
            for _, dp in materialization_group:
                assert dp.context_input_ids is not None, "Missing context_input_ids for steering-vector materialization"
                group_lengths.append(len(dp.context_input_ids))
            stats.num_missing_points += len(materialization_group)
            stats.num_materialization_batches += 1
            stats.actual_tokens += sum(group_lengths)
            stats.padded_tokens += max(group_lengths) * len(materialization_group)

    return stats
