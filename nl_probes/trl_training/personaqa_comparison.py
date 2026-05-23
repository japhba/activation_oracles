"""TRL PersonaQA comparison run.

Runs PersonaQA SFT with TRL's SFTTrainer using the same hyperparameters
as the text_sft comparison run, for loss curve validation.
"""

import gc
from pathlib import Path

import torch

from nl_probes.trl_training.config import CustomLoraConfig, CustomSFTConfig, EvalConfig
from nl_probes.trl_training.personaqa_train import (
    create_personaqa_dataset,
    prepare_sft_dataset,
    train_with_sft_only,
)
from transformers import AutoTokenizer


def main() -> None:
    model_name = "Qwen/Qwen3-8B"
    dataset_name = "data_pipelines/personaqa_data/shuffled"
    wandb_project = "model_understanding_sft"
    num_epochs = 3

    config = EvalConfig(
        model_name=model_name,
        model_lora_dir="model_lora",
        wandb_info=f"trl_comparison",
    )

    # Match text_sft hyperparams as closely as possible
    batch_size = 8
    real_batch_size = 8
    sft_config = CustomSFTConfig(
        model_name=model_name,
        batch_size=batch_size,
        real_batch_size=real_batch_size,
        # Override defaults to match text_sft comparison config
        max_length=1024,
        num_train_epochs=num_epochs,
        learning_rate=5e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no",
        load_best_model_at_end=False,
        save_steps=999999,
        save_total_limit=1,
        logging_steps=1,
        report_to=None,
        seed=42,
        output_dir="checkpoints_text_sft/trl_personaqa_comparison",
    )
    sft_config.run_name = f"trl_personaqa_qwen3_8b"
    sft_config.completion_only_loss = True

    ds = create_personaqa_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = prepare_sft_dataset(ds, tokenizer)

    # TRL expects "completion_mask", not "assistant_masks"
    ds = ds.rename_column("assistant_masks", "completion_mask")

    train_size = int(len(ds) * 0.99)
    train_ds = ds.select(range(train_size))
    eval_ds = ds.select(range(train_size, len(ds)))

    train_with_sft_only(
        sft_train_ds=train_ds,
        sft_hf_eval_test_ds=eval_ds,
        wandb_sft_project=wandb_project,
        config=config,
        sft_config=sft_config,
        callbacks=[],
        save_lora_path=Path("checkpoints_text_sft/trl_personaqa_comparison/final"),
        quantize=False,
    )


if __name__ == "__main__":
    main()
