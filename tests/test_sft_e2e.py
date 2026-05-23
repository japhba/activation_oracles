"""
End-to-end regression test for SFT training.

Trains Qwen3-0.6B for ~96 steps with a fixed seed and asserts that:
1. The training dataset hash matches a recorded snapshot (same data every time).
2. Per-step losses match recorded snapshots (same training behavior every time).

Run with:
    .venv/bin/pytest tests/test_sft_e2e.py -v -s

Takes ~1 minute on a single GPU (dataset gen + training).

Uses training_configs/e2e_test.json as the config. The test launches itself as
a subprocess via torchrun (--run-training flag) since train_model requires
distributed init.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")
TORCHRUN = str(REPO_ROOT / ".venv" / "bin" / "torchrun")
THIS_SCRIPT = str(Path(__file__).resolve())
CONFIG_PATH = str(REPO_ROOT / "training_configs" / "e2e_test.json")

# Read save_dir from config so test cleanup matches
with open(CONFIG_PATH) as f:
    _cfg = json.load(f)
CHECKPOINT_DIR = Path(_cfg["save_dir"])

# --- Recorded snapshots ---
# Recorded with Qwen3-0.6B, seed=42, batch_size=8, past_lens dataset.
EXPECTED_DATASET_HASH = "4a5677f30cbb902f3c3c50869e5fbe80de48e37b"
EXPECTED_LOSSES = {
    0: 4.387349605560303,
    10: 5.362653732299805,
    20: 3.3180110454559326,
    50: 4.061932563781738,
    95: 5.227981090545654,
}
LOSS_ATOL = 0.05


# ─── Training runner (invoked via torchrun --run-training) ────────────────────

def _hash_training_data(training_data) -> str:
    """SHA1 hash of all input_ids + labels, to verify data identity."""
    h = hashlib.sha1()
    for dp in training_data:
        h.update(bytes(str(dp.input_ids), "utf-8"))
        h.update(bytes(str(dp.labels), "utf-8"))
    return h.hexdigest()


def _run_gen_only():
    """Generate dataset on disk (single process, no dist init)."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    import nl_probes.sft as sft
    from nl_probes.configs.sft_config import read_training_config
    from nl_probes.dataset_classes.act_dataset_manager import build_loaders_from_config

    cfg = read_training_config(CONFIG_PATH)
    dataset_loaders = build_loaders_from_config(cfg)
    sft._ensure_datasets_exist(dataset_loaders)
    print("Dataset generation complete (--gen-only); exiting.")


def _run_training():
    """Run the actual training loop (requires torchrun)."""
    import os
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    import nl_probes.sft as sft
    from nl_probes.configs.sft_config import read_training_config
    from nl_probes.dataset_classes.act_dataset_manager import build_loaders_from_config
    from nl_probes.utils.common import load_tokenizer

    dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    cfg = read_training_config(CONFIG_PATH)
    dataset_loaders = build_loaders_from_config(cfg)

    dist.barrier()

    tokenizer = load_tokenizer(cfg.model_name)
    all_training_data, all_eval_data = sft.build_datasets(
        cfg, dataset_loaders=dataset_loaders, window_mult=cfg.window_mult,
    )

    dataset_hash = _hash_training_data(all_training_data)
    print(f"DATASET_HASH={dataset_hash}")

    sft.train_model(
        cfg=cfg,
        training_data=all_training_data,
        component_validation_data={},
        chat_regularization_data=None,
        tokenizer=tokenizer,
        dtype=torch.bfloat16,
        device=torch.device(f"cuda:{local_rank}"),
        model_kwargs={},
        verbose=True,
    )

    dist.destroy_process_group()


# ─── Pytest test ──────────────────────────────────────────────────────────────

def _run_subprocess(cmd: list[str], desc: str) -> str:
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=600,
    )
    if result.returncode != 0:
        pytest.fail(
            f"{desc} failed (rc={result.returncode}).\n"
            f"STDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )
    return result.stdout + result.stderr


def _parse_losses(output: str) -> dict[int, float]:
    losses = {}
    for match in re.finditer(r"Step (\d+) loss: ([\d.]+)", output):
        losses[int(match.group(1))] = float(match.group(2))
    return losses


def _parse_dataset_hash(output: str) -> str | None:
    match = re.search(r"DATASET_HASH=(\w+)", output)
    return match.group(1) if match else None


@pytest.fixture(autouse=True)
def cleanup_checkpoints():
    yield
    if CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR)


@pytest.mark.gpu
def test_sft_training_loss_snapshot():
    """Train Qwen3-0.6B and verify dataset hash + losses match snapshots."""
    # Step 1: Generate dataset (single process)
    _run_subprocess(
        [PYTHON, THIS_SCRIPT, "--gen-only"],
        "Dataset generation",
    )

    # Step 2: Train with torchrun
    output = _run_subprocess(
        [TORCHRUN, "--nproc_per_node=1", THIS_SCRIPT, "--run-training"],
        "Training",
    )

    # Step 3: Verify dataset hash
    dataset_hash = _parse_dataset_hash(output)
    assert dataset_hash is not None, "DATASET_HASH not found in output"
    assert dataset_hash == EXPECTED_DATASET_HASH, (
        f"Dataset hash mismatch: expected {EXPECTED_DATASET_HASH}, got {dataset_hash}. "
        f"Training data has changed."
    )

    # Step 4: Verify losses
    actual_losses = _parse_losses(output)
    assert len(actual_losses) > 0, f"No losses parsed from output. Output tail:\n{output[-1000:]}"

    for step, expected_loss in EXPECTED_LOSSES.items():
        assert step in actual_losses, (
            f"Step {step} not found in training output "
            f"(got steps {sorted(actual_losses.keys())[:5]}...)"
        )
        actual = actual_losses[step]
        assert abs(actual - expected_loss) < LOSS_ATOL, (
            f"Step {step}: expected loss {expected_loss:.6f}, got {actual:.6f} "
            f"(delta={abs(actual - expected_loss):.6f}, atol={LOSS_ATOL})"
        )


# ─── __main__ entrypoint (for subprocess invocation) ─────────────────────────

if __name__ == "__main__":
    import os
    import sys

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["WANDB_MODE"] = "disabled"

    if "--gen-only" in sys.argv:
        _run_gen_only()
    elif "--run-training" in sys.argv:
        _run_training()
    else:
        print("Usage: pass --gen-only or --run-training", file=sys.stderr)
        sys.exit(1)
