"""Linear probe baseline for hallucination detection.

Collects activations at span positions from the base model, then trains a
linear probe to predict Supported vs Not Supported. Tests on the same 200
entries used for the AO eval.

Usage:
    python experiments/hallucination_linear_probe.py
"""

import json
import random
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from nl_probes.base_experiment import _build_padded_batch_from_token_ids
from nl_probes.open_ended_eval.hallucination import (
    AO_PROMPTS,
    build_hallucination_verbalizer_prompt_infos,
    load_hallucination_dataset,
)
from nl_probes.utils.activation_utils import (
    collect_activations_multiple_layers,
    get_hf_submodule,
)
from nl_probes.utils.common import load_model, load_tokenizer

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

MODEL_NAME = "Qwen/Qwen3-8B"
DATASET_CONFIG = "Qwen2.5-7B-Instruct"
TARGET_TEST_SPANS = 2000
LAYER_PERCENTS = [25, 50, 75]
BATCH_SIZE = 32

# Linear probe hyperparameters
EPOCHS = 100
LR = 1e-2
WEIGHT_DECAY = 1e-4
PROBE_BATCH_SIZE = 8192

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def percent_to_layer(model, percent: int) -> int:
    num_layers = model.config.num_hidden_layers
    return round(num_layers * percent / 100)


@torch.no_grad()
def collect_span_activations(
    model,
    tokenizer,
    entries: list[dict],
    layer_percents: list[int],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect mean-pooled activations at span positions for all entries.

    Returns:
        X: [num_entries, num_layers * hidden_dim] float32
        y: [num_entries] long (1=supported, 0=not supported)
    """
    single_prompt = {k: v for k, v in list(AO_PROMPTS.items())[:1]}
    prompt_infos, _ = build_hallucination_verbalizer_prompt_infos(
        entries, single_prompt, tokenizer,
    )

    layers = [percent_to_layer(model, p) for p in layer_percents]
    submodules = {layer: get_hf_submodule(model, layer) for layer in layers}

    all_features = []
    all_labels = []

    for start in tqdm(range(0, len(prompt_infos), batch_size), desc="Collecting activations"):
        batch_infos = prompt_infos[start:start + batch_size]
        batch_entries = entries[start:start + batch_size]

        token_id_lists = [info.context_token_ids for info in batch_infos]
        positions_list = [info.positions for info in batch_infos]

        # Build padded batch
        inputs = _build_padded_batch_from_token_ids(token_id_lists, tokenizer, torch.device(DEVICE))

        # Collect activations at all layers
        acts_by_layer = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL=inputs,
            min_offset=None,
            max_offset=None,
        )

        # Extract and mean-pool span activations for each entry
        for batch_idx in range(len(batch_infos)):
            positions = positions_list[batch_idx]
            num_tokens = len(token_id_lists[batch_idx])
            max_len = inputs["input_ids"].shape[1]
            left_pad = max_len - num_tokens

            abs_positions = [left_pad + p for p in positions]

            layer_features = []
            for layer in layers:
                acts_LD = acts_by_layer[layer][batch_idx]  # [seq_len, hidden_dim]
                span_acts = acts_LD[abs_positions]  # [num_span_tokens, hidden_dim]
                pooled = span_acts.mean(dim=0)  # [hidden_dim]
                layer_features.append(pooled)

            feature = torch.cat(layer_features, dim=0)  # [num_layers * hidden_dim]
            all_features.append(feature.cpu().float())
            all_labels.append(1 if batch_entries[batch_idx]["supported"] else 0)

    X = torch.stack(all_features)
    y = torch.tensor(all_labels, dtype=torch.long)
    return X, y


def train_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> dict:
    """Train linear probe. Use val set for early stopping / model selection, report test only once."""
    from nl_probes.open_ended_eval.eval_runner import compute_roc_curve_data

    # Standardize using train statistics
    mu = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    X_train = (X_train - mu) / std
    X_val = (X_val - mu) / std
    X_test = (X_test - mu) / std

    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    X_val = X_val.to(DEVICE)
    y_val = y_val.to(DEVICE)
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)

    d = X_train.shape[1]
    probe = nn.Linear(d, 2, bias=True).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        probe.train()
        perm = torch.randperm(X_train.shape[0], device=DEVICE)
        for start in range(0, X_train.shape[0], PROBE_BATCH_SIZE):
            idx = perm[start:start + PROBE_BATCH_SIZE]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(probe(X_train[idx]), y_train[idx])
            loss.backward()
            opt.step()

        if epoch % 10 == 0 or epoch == EPOCHS:
            probe.eval()
            with torch.no_grad():
                train_acc = (probe(X_train).argmax(1) == y_train).float().mean().item()
                val_acc = (probe(X_val).argmax(1) == y_val).float().mean().item()
            print(f"  epoch {epoch:3d}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    # Load best model (by val) and evaluate on test
    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        test_logits = probe(X_test)
        test_acc = (test_logits.argmax(1) == y_test).float().mean().item()
        val_logits = probe(X_val)
        val_acc_final = (val_logits.argmax(1) == y_val).float().mean().item()

    # ROC AUC on test set
    test_margins = (test_logits[:, 1] - test_logits[:, 0]).cpu().tolist()
    test_roc = compute_roc_curve_data(y_test.cpu().tolist(), test_margins)

    # ROC AUC on val set
    val_margins = (val_logits[:, 1] - val_logits[:, 0]).cpu().tolist()
    val_roc = compute_roc_curve_data(y_val.cpu().tolist(), val_margins)

    return {
        "val_acc": val_acc_final,
        "val_roc_auc": val_roc["auc"] if val_roc else None,
        "test_acc": test_acc,
        "test_roc_auc": test_roc["auc"] if test_roc else None,
        "train_size": X_train.shape[0],
        "val_size": X_val.shape[0],
        "test_size": X_test.shape[0],
        "feature_dim": d,
    }


def split_by_response(
    entries: list[dict],
    target_test_spans: int,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split entries by response so no response appears in multiple splits.

    Groups entries by source_row_idx, shuffles responses, then assigns
    responses to test (until target_test_spans reached), val (next 20%
    of remaining responses), and train (rest).
    """
    from collections import defaultdict

    by_response = defaultdict(list)
    for e in entries:
        by_response[e["source_row_idx"]].append(e)

    response_ids = sorted(by_response.keys())
    rng = random.Random(seed)
    rng.shuffle(response_ids)

    # Assign responses to test until we hit target span count
    test_entries = []
    split_idx = 0
    for i, rid in enumerate(response_ids):
        test_entries.extend(by_response[rid])
        if len(test_entries) >= target_test_spans:
            split_idx = i + 1
            break

    # Remaining responses: 80% train, 20% val
    remaining_ids = response_ids[split_idx:]
    val_count = len(remaining_ids) // 5
    val_ids = remaining_ids[:val_count]
    train_ids = remaining_ids[val_count:]

    val_entries = [e for rid in val_ids for e in by_response[rid]]
    train_entries = [e for rid in train_ids for e in by_response[rid]]

    # Verify no response leakage
    test_responses = {e["source_row_idx"] for e in test_entries}
    val_responses = {e["source_row_idx"] for e in val_entries}
    train_responses = {e["source_row_idx"] for e in train_entries}
    assert not (test_responses & val_responses), "Response leakage: test/val overlap"
    assert not (test_responses & train_responses), "Response leakage: test/train overlap"
    assert not (val_responses & train_responses), "Response leakage: val/train overlap"

    return train_entries, val_entries, test_entries


def main():
    print(f"Loading all entries from {DATASET_CONFIG}...")
    all_entries = load_hallucination_dataset(DATASET_CONFIG)
    print(f"Total entries: {len(all_entries)}")

    # Split by response — no response appears in multiple splits
    train_entries, val_entries, test_entries = split_by_response(
        all_entries, target_test_spans=TARGET_TEST_SPANS,
    )

    test_response_ids = sorted({e["source_row_idx"] for e in test_entries})
    for name, split in [("Train", train_entries), ("Val", val_entries), ("Test", test_entries)]:
        sup = sum(1 for e in split if e["supported"])
        n_responses = len({e["source_row_idx"] for e in split})
        print(f"  {name}: {len(split)} spans from {n_responses} responses "
              f"({sup} supported, {len(split) - sup} not)")

    # Save test entry IDs so AO/text baseline can use the same test set
    test_ids = [e["id"] for e in test_entries]
    test_ids_path = Path("experiments/hallucination_eval_results/test_entry_ids.json")
    test_ids_path.parent.mkdir(parents=True, exist_ok=True)
    test_ids_path.write_text(json.dumps({"test_ids": test_ids, "test_response_ids": test_response_ids}))
    print(f"  Saved {len(test_ids)} test entry IDs to {test_ids_path}")

    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, torch.bfloat16)
    model.eval()

    print(f"\nCollecting activations...")
    X_test, y_test = collect_span_activations(model, tokenizer, test_entries, LAYER_PERCENTS, BATCH_SIZE)
    print(f"  X_test: {X_test.shape}")
    X_val, y_val = collect_span_activations(model, tokenizer, val_entries, LAYER_PERCENTS, BATCH_SIZE)
    print(f"  X_val: {X_val.shape}")
    X_train, y_train = collect_span_activations(model, tokenizer, train_entries, LAYER_PERCENTS, BATCH_SIZE)
    print(f"  X_train: {X_train.shape}")

    del model
    torch.cuda.empty_cache()

    print(f"\nTraining linear probe...")
    results = train_probe(X_train, y_train, X_val, y_val, X_test, y_test)

    print(f"\n{'=' * 60}")
    print(f"Linear Probe Results:")
    print(f"  Train: {results['train_size']}  Val: {results['val_size']}  Test: {results['test_size']}")
    print(f"  Feature dim: {results['feature_dim']}")
    print(f"  Val accuracy: {results['val_acc']:.4f}")
    print(f"  Val ROC AUC: {results['val_roc_auc']:.4f}" if results["val_roc_auc"] else "  Val ROC AUC: N/A")
    print(f"  Test accuracy: {results['test_acc']:.4f}")
    print(f"  Test ROC AUC: {results['test_roc_auc']:.4f}" if results["test_roc_auc"] else "  Test ROC AUC: N/A")
    print(f"{'=' * 60}")

    out_dir = Path("experiments/hallucination_eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "linear_probe_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
