"""
Extract activations from Qwen3-8B for hiring bias entries and train a linear probe.

Steps:
1. Build balanced dataset (biased vs unbiased groups, same-person matched)
2. Run target model, extract activations at layers [25%, 50%, 75%]
3. Mean-pool across positions → fixed-size vector per entry
4. Save activations + metadata to disk
5. Train logistic regression with 10-fold CV
6. Report AUC and compare to uncertainty confounder baseline

Usage:
    source .env && .venv/bin/python data_pipelines/hiring_bias/extract_activations_and_probe.py \
        --model Qwen/Qwen3-8B \
        --data data_pipelines/hiring_bias/Qwen3-8B/hiring_fairness_explore_200tok_no_cot_it_fin_teach_chef_alljobs.json
"""

import argparse
import functools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoTokenizer

from nl_probes.base_experiment import tokenize_chat_messages, compute_segment_positions
from nl_probes.utils.activation_utils import (
    collect_activations_multiple_layers,
    get_hf_submodule,
)
from nl_probes.utils.common import load_model, load_tokenizer

from data_pipelines.pipeline_utils import model_dir_name, add_model_arg

print = functools.partial(print, flush=True)


def build_eval_dataset(data: dict, min_spread: float) -> list[dict]:
    """Build balanced biased vs unbiased dataset with same-person matching."""
    entries = data["entries"]
    group_stats = {gs["group_key"]: gs for gs in data["group_stats"]}

    by_group = defaultdict(list)
    for e in entries:
        by_group[e["group_key"]].append(e)

    # Same-person matching
    by_person = defaultdict(list)
    for e in entries:
        gk = e["group_key"]
        parts = gk.rsplit("_j", 1)
        resume_key = parts[0]
        person_key = (resume_key, e["demo_key"])
        gs = group_stats[gk]
        by_person[person_key].append({**e, "spread": gs["spread"], "race_gap": gs["race_gap"]})

    matched_biased = []
    matched_unbiased = []
    for pk, pes in by_person.items():
        has_b = any(e["spread"] > min_spread for e in pes)
        has_u = any(e["spread"] <= 0.02 for e in pes)
        if has_b and has_u:
            for e in pes:
                if e["spread"] > min_spread:
                    matched_biased.append(e)
                elif e["spread"] <= 0.02:
                    matched_unbiased.append(e)

    # Balance classes
    rng = np.random.RandomState(42)
    if len(matched_unbiased) > len(matched_biased):
        idx = rng.choice(len(matched_unbiased), len(matched_biased), replace=False)
        matched_unbiased = [matched_unbiased[i] for i in sorted(idx)]

    # Add labels
    eval_entries = []
    for e in matched_biased:
        assistant_text = "Yes" if e["yes_prob"] > 0.5 else "No"
        eval_entries.append({**e, "label": 1, "label_name": "biased", "assistant_text": assistant_text})
    for e in matched_unbiased:
        assistant_text = "Yes" if e["yes_prob"] > 0.5 else "No"
        eval_entries.append({**e, "label": 0, "label_name": "unbiased", "assistant_text": assistant_text})

    rng.shuffle(eval_entries)
    return eval_entries


def get_layer_indices(model_config, percentages: list[int]) -> list[int]:
    """Convert layer percentages to absolute indices."""
    n_layers = model_config.num_hidden_layers
    return [int(p / 100 * n_layers) for p in percentages]


def extract_activations(
    model,
    tokenizer,
    eval_entries: list[dict],
    layer_indices: list[int],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Extract mean-pooled activations for all entries.

    Returns: tensor of shape [N, num_layers * D]
    """
    # Build submodules dict
    submodules = {layer: get_hf_submodule(model, layer) for layer in layer_indices}

    all_activations = []

    for start in tqdm(range(0, len(eval_entries), batch_size), desc="Extracting activations"):
        batch = eval_entries[start:start + batch_size]

        # Tokenize
        batch_token_ids = []
        batch_positions = []
        for entry in batch:
            messages = [
                {"role": "user", "content": entry["user_content"]},
                {"role": "assistant", "content": entry["assistant_text"]},
            ]
            token_ids = tokenize_chat_messages(
                tokenizer, messages,
                add_generation_prompt=False,
                continue_final_message=True,
                enable_thinking=False,
            )
            positions = compute_segment_positions(len(token_ids), start_idx=0)
            batch_token_ids.append(token_ids)
            batch_positions.append(positions)

        # Pad to same length
        max_len = max(len(ids) for ids in batch_token_ids)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, ids in enumerate(batch_token_ids):
            # Left-pad
            offset = max_len - len(ids)
            input_ids[i, offset:] = torch.tensor(ids)
            attention_mask[i, offset:] = 1

        inputs = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}

        # Extract activations
        acts_by_layer = collect_activations_multiple_layers(
            model, submodules, inputs, min_offset=None, max_offset=None,
        )

        # Mean-pool across positions for each entry
        for i in range(len(batch)):
            left_pad = max_len - len(batch_token_ids[i])
            abs_positions = [left_pad + p for p in batch_positions[i]]

            layer_acts = []
            for layer in layer_indices:
                acts_LD = acts_by_layer[layer][i, abs_positions, :]  # [P, D]
                mean_act = acts_LD.float().mean(dim=0)  # [D]
                layer_acts.append(mean_act)

            concat = torch.cat(layer_acts, dim=0)  # [num_layers * D]
            all_activations.append(concat.cpu())

    return torch.stack(all_activations)  # [N, num_layers * D]


def run_linear_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 10,
) -> dict:
    """Train logistic regression with k-fold CV, return AUC."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_aucs = []
    fold_accs = []
    all_scores = np.zeros(len(labels))

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(activations, labels)):
        X_train, X_test = activations[train_idx], activations[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Standardize features
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[:, 1]
        all_scores[test_idx] = probs

        if len(np.unique(y_test)) > 1:
            fold_auc = roc_auc_score(y_test, probs)
            fold_aucs.append(fold_auc)

        fold_acc = clf.score(X_test, y_test)
        fold_accs.append(fold_acc)

    overall_auc = roc_auc_score(labels, all_scores)

    return {
        "overall_auc": overall_auc,
        "mean_fold_auc": np.mean(fold_aucs) if fold_aucs else None,
        "std_fold_auc": np.std(fold_aucs) if fold_aucs else None,
        "mean_fold_acc": np.mean(fold_accs),
        "std_fold_acc": np.std(fold_accs),
        "n_folds": n_folds,
        "n_samples": len(labels),
        "n_positive": int(labels.sum()),
        "n_negative": int(len(labels) - labels.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    add_model_arg(parser)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--min-spread", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--layer-percentages", type=int, nargs="+", default=[25, 50, 75])
    args = parser.parse_args()

    output_dir = Path(f"data_pipelines/hiring_bias/{model_dir_name(args.model)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    activations_path = output_dir / "probe_activations.pt"
    results_path = output_dir / "probe_results.json"

    # Step 1: Build dataset
    print("=== Step 1: Building dataset ===")
    data = json.loads(Path(args.data).read_text())
    eval_entries = build_eval_dataset(data, min_spread=args.min_spread)
    labels = np.array([e["label"] for e in eval_entries])
    print(f"  {len(eval_entries)} entries: {labels.sum()} biased + {len(labels) - labels.sum()} unbiased")

    # Step 2: Load model
    print("\n=== Step 2: Loading model ===")
    device = torch.device("cuda")
    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, torch.bfloat16)
    model.eval()
    torch.set_grad_enabled(False)

    layer_indices = get_layer_indices(model.config, args.layer_percentages)
    print(f"  Layers: {args.layer_percentages}% → indices {layer_indices}")
    print(f"  Hidden dim: {model.config.hidden_size}")

    # Step 3: Extract activations
    print("\n=== Step 3: Extracting activations ===")
    activations = extract_activations(
        model, tokenizer, eval_entries, layer_indices,
        batch_size=args.batch_size, device=device,
    )
    print(f"  Activations shape: {activations.shape}")

    # Save activations
    save_data = {
        "activations": activations,
        "labels": torch.tensor(labels),
        "metadata": [{k: v for k, v in e.items() if k != "user_content"} for e in eval_entries],
        "layer_indices": layer_indices,
        "layer_percentages": args.layer_percentages,
    }
    torch.save(save_data, activations_path)
    print(f"  Saved activations to {activations_path}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # Step 4: Train linear probe
    print("\n=== Step 4: Training linear probe ===")
    acts_np = activations.numpy()

    # Full multi-layer probe
    probe_results = run_linear_probe(acts_np, labels, n_folds=args.n_folds)
    print(f"  Multi-layer probe ({args.layer_percentages}%):")
    print(f"    AUC: {probe_results['overall_auc']:.3f} (fold mean: {probe_results['mean_fold_auc']:.3f} ± {probe_results['std_fold_auc']:.3f})")
    print(f"    Acc: {probe_results['mean_fold_acc']:.3f} ± {probe_results['std_fold_acc']:.3f}")

    # Per-layer probes
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model)
    D = config.hidden_size

    per_layer_results = {}
    for i, (pct, layer_idx) in enumerate(zip(args.layer_percentages, layer_indices)):
        layer_acts = acts_np[:, i * D:(i + 1) * D]
        lr = run_linear_probe(layer_acts, labels, n_folds=args.n_folds)
        per_layer_results[f"layer_{pct}pct"] = lr
        print(f"  Layer {pct}% (idx {layer_idx}): AUC={lr['overall_auc']:.3f}")

    # Confounder baselines
    print("\n=== Confounder baselines ===")
    uncertainty = np.array([-abs(e["yes_prob"] - 0.5) for e in eval_entries])
    unc_auc = roc_auc_score(labels, uncertainty)
    print(f"  Uncertainty (|P-0.5|): AUC={unc_auc:.3f}")

    pyes = np.array([e["yes_prob"] for e in eval_entries])
    pyes_auc = roc_auc_score(labels, pyes)
    print(f"  P(Yes): AUC={pyes_auc:.3f}")

    # Save results
    all_results = {
        "multi_layer_probe": probe_results,
        "per_layer_probes": per_layer_results,
        "confounder_baselines": {
            "uncertainty_auc": unc_auc,
            "pyes_auc": pyes_auc,
        },
        "config": {
            "model": args.model,
            "min_spread": args.min_spread,
            "layer_percentages": args.layer_percentages,
            "layer_indices": layer_indices,
            "n_folds": args.n_folds,
            "n_entries": len(eval_entries),
        },
    }
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
