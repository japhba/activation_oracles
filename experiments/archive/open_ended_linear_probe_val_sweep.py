"""
Linear probe sweep with proper validation split.

Splits the existing train feature cache into train/val (80/20, stratified),
sweeps hyperparameters selecting on val ROC-AUC, then reports final test
metrics for the val-selected hyperparameters.

For sycophancy tasks, val holdout respects group_ids (so no samples from
the same source entry appear in both train and val).

Usage:
    .venv/bin/python experiments/open_ended_linear_probe_val_sweep.py
"""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

CACHE_DIR = Path("experiments/open_ended_linear_probe_cache")
OUTPUT_DIR = Path("experiments/open_ended_linear_probe_results/val_sweeps")

TASKS = [
    "mmlu_pre_answer",
    "mmlu_post_answer",
    "sycophancy_no_cot",
    "sycophancy_cot",
]

POOLING = "mean_all"  # best from ceiling run

SWEEP_GRID = {
    "epochs": [25, 50, 100, 200, 400],
    "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
}

BATCH_SIZE = 1024
SEED = 42
VAL_FRACTION = 0.2


class BinaryLinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def load_cache(path: Path) -> dict:
    return torch.load(path, weights_only=False)


def compute_roc_auc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]
    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return 0.5
    pairwise = pos_probs[:, None] - neg_probs[None, :]
    return ((pairwise > 0).float() + 0.5 * (pairwise == 0).float()).mean().item()


def compute_balanced_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (logits >= 0).long()
    pos_mask = labels == 1
    neg_mask = labels == 0
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return 0.5
    tpr = (preds[pos_mask] == 1).float().mean().item()
    tnr = (preds[neg_mask] == 0).float().mean().item()
    return 0.5 * (tpr + tnr)


def split_train_val(
    cache: dict,
    val_fraction: float,
    seed: int,
    group_holdout: bool,
) -> tuple[dict, dict]:
    """Split a feature cache into train/val. Stratified by label.
    If group_holdout=True, entire groups go to val together."""
    features = cache["features"]
    labels = cache["labels"]
    sample_ids = cache["sample_ids"]
    group_ids = cache["group_ids"]
    metadata = cache["metadata"]
    n = features.shape[0]

    rng = random.Random(seed)

    if group_holdout:
        # Group-level holdout: assign entire groups to val
        # Build group -> indices mapping
        group_to_indices: dict[str, list[int]] = {}
        for i, gid in enumerate(group_ids):
            group_to_indices.setdefault(gid, []).append(i)

        # Separate groups by whether they contain positive/negative samples
        pos_groups = set()
        neg_groups = set()
        for gid, indices in group_to_indices.items():
            for idx in indices:
                if labels[idx].item() == 1:
                    pos_groups.add(gid)
                else:
                    neg_groups.add(gid)

        all_groups = list(group_to_indices.keys())
        rng.shuffle(all_groups)

        # Greedily assign groups to val until we have enough
        target_val = int(n * val_fraction)
        val_indices_set = set()
        for gid in all_groups:
            if len(val_indices_set) >= target_val:
                break
            val_indices_set.update(group_to_indices[gid])

        val_indices = sorted(val_indices_set)
        train_indices = sorted(set(range(n)) - val_indices_set)
    else:
        # Stratified split by label
        pos_indices = [i for i in range(n) if labels[i].item() == 1]
        neg_indices = [i for i in range(n) if labels[i].item() == 0]
        rng.shuffle(pos_indices)
        rng.shuffle(neg_indices)

        n_pos_val = max(1, int(len(pos_indices) * val_fraction))
        n_neg_val = max(1, int(len(neg_indices) * val_fraction))

        val_indices = sorted(pos_indices[:n_pos_val] + neg_indices[:n_neg_val])
        train_indices = sorted(pos_indices[n_pos_val:] + neg_indices[n_neg_val:])

    def subset(indices):
        return {
            "features": features[indices],
            "labels": labels[indices],
            "sample_ids": [sample_ids[i] for i in indices],
            "group_ids": [group_ids[i] for i in indices],
            "metadata": [metadata[i] for i in indices],
        }

    return subset(train_indices), subset(val_indices)


def train_and_evaluate(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_eval: torch.Tensor,
    y_eval: torch.Tensor,
    *,
    input_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    return_logits: bool = False,
) -> tuple[float, float, float] | tuple[float, float, float, torch.Tensor]:
    """Train a probe and return (eval_roc_auc, eval_balanced_acc, eval_acc).
    If return_logits=True, also returns the raw eval logits tensor."""
    torch.manual_seed(seed)
    model = BinaryLinearProbe(input_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    num_train = X_train.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(num_train)
        for batch_start in range(0, num_train, batch_size):
            idx = perm[batch_start : batch_start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_train[idx])
            loss = loss_fn(logits, y_train[idx].float())
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        eval_logits = model(X_eval)

    roc_auc = compute_roc_auc(eval_logits, y_eval)
    bal_acc = compute_balanced_accuracy(eval_logits, y_eval)
    acc = (eval_logits >= 0).long().eq(y_eval).float().mean().item()
    if return_logits:
        return roc_auc, bal_acc, acc, eval_logits
    return roc_auc, bal_acc, acc


def standardize(X_train, *others):
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    result = [(X_train - mean) / std]
    for X in others:
        result.append((X - mean) / std)
    return result


def run_task(task_name: str):
    layers_str = "25-50-75"
    train_path = CACHE_DIR / f"{task_name}_train_Qwen_Qwen3-8B_layers_{layers_str}_{POOLING}.pt"
    test_path = CACHE_DIR / f"{task_name}_test_Qwen_Qwen3-8B_layers_{layers_str}_{POOLING}.pt"

    if not train_path.exists() or not test_path.exists():
        print(f"SKIP {task_name}: cache not found")
        return None

    train_cache = load_cache(train_path)
    test_cache = load_cache(test_path)

    is_sycophancy = "sycophancy" in task_name
    train_split, val_split = split_train_val(
        train_cache, VAL_FRACTION, SEED, group_holdout=is_sycophancy
    )

    X_train_raw = train_split["features"].float()
    y_train = train_split["labels"].long()
    X_val_raw = val_split["features"].float()
    y_val = val_split["labels"].long()
    X_test_raw = test_cache["features"].float()
    y_test = test_cache["labels"].long()

    input_dim = X_train_raw.shape[1]

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"Train: {len(y_train)} (pos={y_train.sum().item()}, neg={(y_train==0).sum().item()})")
    print(f"Val:   {len(y_val)} (pos={y_val.sum().item()}, neg={(y_val==0).sum().item()})")
    print(f"Test:  {len(y_test)} (pos={y_test.sum().item()}, neg={(y_test==0).sum().item()})")
    print(f"{'='*60}")

    # Standardize using train split stats
    X_train, X_val, X_test = standardize(X_train_raw, X_val_raw, X_test_raw)

    # Sweep on val
    best_val_auc = -1.0
    best_hparams = {}
    all_results = []
    run_count = 0
    total_runs = len(SWEEP_GRID["epochs"]) * len(SWEEP_GRID["learning_rate"]) * len(SWEEP_GRID["weight_decay"])

    for epochs in SWEEP_GRID["epochs"]:
        for lr in SWEEP_GRID["learning_rate"]:
            for wd in SWEEP_GRID["weight_decay"]:
                run_count += 1
                val_auc, val_bal_acc, val_acc = train_and_evaluate(
                    X_train, y_train, X_val, y_val,
                    input_dim=input_dim, epochs=epochs, learning_rate=lr,
                    weight_decay=wd, batch_size=BATCH_SIZE, seed=SEED,
                )
                all_results.append({
                    "epochs": epochs, "lr": lr, "wd": wd,
                    "val_auc": val_auc, "val_bal_acc": val_bal_acc, "val_acc": val_acc,
                })
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_hparams = {"epochs": epochs, "learning_rate": lr, "weight_decay": wd}

                if run_count % 10 == 0 or run_count == total_runs:
                    print(f"  [{run_count}/{total_runs}] val_auc={val_auc:.4f} best_val_auc={best_val_auc:.4f}")

    print(f"\nBest val hyperparams: {best_hparams} (val_auc={best_val_auc:.4f})")

    # Final: train on full train set (train+val) with best hyperparams, evaluate on test
    X_full_train_raw = train_cache["features"].float()
    y_full_train = train_cache["labels"].long()
    X_full_train, X_test_final = standardize(X_full_train_raw, X_test_raw)

    test_auc, test_bal_acc, test_acc, test_logits = train_and_evaluate(
        X_full_train, y_full_train, X_test_final, y_test,
        input_dim=input_dim, seed=SEED, batch_size=BATCH_SIZE,
        return_logits=True,
        **best_hparams,
    )

    print(f"\nFinal test (retrained on train+val with val-selected hparams):")
    print(f"  ROC-AUC:          {test_auc:.4f}")
    print(f"  Balanced Accuracy: {test_bal_acc:.4f}")
    print(f"  Accuracy:          {test_acc:.4f}")

    # Also report what the val-selected hparams give when trained only on train split
    # (to see if retraining on full data helps)
    test_auc_trainonly, test_bal_acc_trainonly, test_acc_trainonly = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        input_dim=input_dim, seed=SEED, batch_size=BATCH_SIZE,
        **best_hparams,
    )

    print(f"\nTest (trained only on train split, val-selected hparams):")
    print(f"  ROC-AUC:          {test_auc_trainonly:.4f}")
    print(f"  Balanced Accuracy: {test_bal_acc_trainonly:.4f}")
    print(f"  Accuracy:          {test_acc_trainonly:.4f}")

    # Build per-entry test predictions for bootstrap CIs
    test_sample_ids = test_cache["sample_ids"]
    test_probs = torch.sigmoid(test_logits).tolist()
    test_preds = (test_logits >= 0).long().tolist()
    test_labels_list = y_test.tolist()
    test_logits_list = test_logits.tolist()

    per_entry_test = [
        {
            "sample_id": test_sample_ids[i],
            "logit": test_logits_list[i],
            "prob": test_probs[i],
            "predicted_label": test_preds[i],
            "ground_truth": test_labels_list[i],
            "is_correct": test_preds[i] == test_labels_list[i],
        }
        for i in range(len(test_sample_ids))
    ]

    result = {
        "task_name": task_name,
        "pooling": POOLING,
        "val_fraction": VAL_FRACTION,
        "seed": SEED,
        "train_n": len(y_train),
        "val_n": len(y_val),
        "test_n": len(y_test),
        "best_val_hparams": best_hparams,
        "best_val_auc": best_val_auc,
        "test_retrained_on_full_train": {
            "roc_auc": test_auc,
            "balanced_accuracy": test_bal_acc,
            "accuracy": test_acc,
        },
        "test_trained_on_train_split_only": {
            "roc_auc": test_auc_trainonly,
            "balanced_accuracy": test_bal_acc_trainonly,
            "accuracy": test_acc_trainonly,
        },
        "per_entry_test": per_entry_test,
        "sweep_results": all_results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{task_name}_{POOLING}_val_sweep.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {output_path}")

    return result


def main():
    all_results = {}
    for task in TASKS:
        result = run_task(task)
        if result is not None:
            all_results[task] = result

    print("\n\n" + "=" * 70)
    print("SUMMARY (val-selected hyperparams, retrained on full train)")
    print("=" * 70)
    print(f"{'Task':<25} {'Test AUC':>10} {'Test BalAcc':>12} {'Val AUC':>10} {'Hparams'}")
    print("-" * 70)
    for task, r in all_results.items():
        t = r["test_retrained_on_full_train"]
        h = r["best_val_hparams"]
        hstr = f"ep={h['epochs']} lr={h['learning_rate']} wd={h['weight_decay']}"
        print(f"{task:<25} {t['roc_auc']:>10.4f} {t['balanced_accuracy']:>12.4f} {r['best_val_auc']:>10.4f} {hstr}")


if __name__ == "__main__":
    main()
