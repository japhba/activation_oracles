"""PersonaQA text SFT trainer.

Loads PersonaQA data (personas + bios + interviews), converts to chat
conversations, and trains using the generic text_sft training infrastructure.
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
import random
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from nl_probes.text_sft.train import (
    ChatMessage,
    TextSFTConfig,
    read_config,
    tokenize_conversations,
    train_model,
)
from nl_probes.utils.common import load_tokenizer


def load_personaqa_conversations(data_dir: str, seed: int = 42) -> list[tuple[ChatMessage, ...]]:
    """Load PersonaQA data and convert to chat conversations."""
    data_path = Path(data_dir)
    with open(data_path / "personas.jsonl") as f:
        persona_data = [json.loads(line) for line in f]
    with open(data_path / "bios.jsonl") as f:
        bios_data = [json.loads(line) for line in f]
    with open(data_path / "interviews.jsonl") as f:
        interviews_data = [json.loads(line) for line in f]

    persona_by_id = {p["id"]: p for p in persona_data}

    all_data = interviews_data + bios_data
    random.seed(seed)
    random.shuffle(all_data)

    conversations = []
    for dp in all_data:
        persona = persona_by_id[dp["persona_id"]]
        name = persona["name"]
        conversations.append((
            ChatMessage(role="user", content=f"Name: {name}.\n"),
            ChatMessage(role="assistant", content=dp["text"]),
        ))

    return conversations


def main() -> None:
    parser = argparse.ArgumentParser(description="PersonaQA text SFT trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to a TextSFTConfig JSON")
    args = parser.parse_args()

    cfg = read_config(args.config)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    tokenizer = load_tokenizer(cfg.model_name)

    conversations = load_personaqa_conversations(cfg.dataset_path, seed=cfg.seed)
    print(f"Loaded {len(conversations)} PersonaQA conversations from {cfg.dataset_path}")

    examples = tokenize_conversations(
        conversations, tokenizer, cfg, write_debug=(local_rank == 0),
    )

    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    assert cfg.global_train_batch_size % world_size == 0, (
        f"global_train_batch_size {cfg.global_train_batch_size} must be divisible by world_size {world_size}"
    )
    per_rank_batch_size = cfg.global_train_batch_size // world_size
    print(f"Per-rank batch size: {per_rank_batch_size}, world_size: {world_size}")

    device = torch.device(f"cuda:{local_rank}")
    train_model(
        cfg=cfg,
        examples=examples,
        tokenizer=tokenizer,
        device=device,
        dtype=torch.bfloat16,
        per_rank_batch_size=per_rank_batch_size,
    )
    dist.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
