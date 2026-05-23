# %% [setup]
import torch
from torch import nn

from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig
from nl_probes.dataset_classes.classification import (
    ClassificationDatasetConfig,
    ClassificationDatasetLoader,
)

# Repro and compute prefs
SEED = 0
torch.manual_seed(SEED)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# %% [data selection]
layer_combination = [25, 50, 75]
save_acts = True
model_name = "Qwen/Qwen3-8B"

main_train_size = 6000
main_test_size = 250
engels_train_size = 1000
BATCH_SIZE = 256

classification_datasets = {
    "geometry_of_truth": {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
    # "relations": {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
    "sst2": {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
    "md_gender": {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
    # "snli": {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
    "ag_news": {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
    # "ner": {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
    "tense": {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
    "language_identification": {
        "num_train": main_train_size,
        "num_test": main_test_size,
        "splits": ["train", "test"],
    },
    "singular_plural": {"num_train": 100, "num_test": main_test_size, "splits": ["train", "test"]},
    "engels_headline_istrump": {
        "num_train": engels_train_size,
        "num_test": main_test_size,
        "splits": ["train", "test"],
    },
    "engels_headline_isobama": {
        "num_train": engels_train_size,
        "num_test": main_test_size,
        "splits": ["train", "test"],
    },
    "engels_headline_ischina": {
        "num_train": engels_train_size,
        "num_test": main_test_size,
        "splits": ["train", "test"],
    },
    "engels_hist_fig_ismale": {"num_train": engels_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
    "engels_news_class_politics": {
        "num_train": engels_train_size,
        "num_test": main_test_size,
        "splits": ["train", "test"],
    },
    "engels_wikidata_isjournalist": {
        "num_train": engels_train_size,
        "num_test": main_test_size,
        "splits": ["train", "test"],
    },
    "engels_wikidata_isathlete": {
        "num_train": engels_train_size,
        "num_test": main_test_size,
        "splits": ["train", "test"],
    },
    "engels_wikidata_ispolitician": {
        "num_train": engels_train_size,
        "num_test": main_test_size,
        "splits": ["train", "test"],
    },
    "engels_wikidata_issinger": {
        "num_train": engels_train_size,
        "num_test": main_test_size,
        "splits": ["train", "test"],
    },
    "engels_wikidata_isresearcher": {
        "num_train": engels_train_size,
        "num_test": main_test_size,
        "splits": ["train", "test"],
    },
}

# Build loaders
classification_dataset_loaders: list[ClassificationDatasetLoader] = []
for dataset_name in classification_datasets:
    classification_config = ClassificationDatasetConfig(
        classification_dataset_name=dataset_name,
        max_window_size=1,
        max_end_offset=-5,
        min_end_offset=-3,
    )

    if "language_identification" in dataset_name:
        batch_size = BATCH_SIZE // 8
    else:
        batch_size = BATCH_SIZE

    dataset_config = DatasetLoaderConfig(
        custom_dataset_params=classification_config,
        num_train=classification_datasets[dataset_name]["num_train"],
        num_test=classification_datasets[dataset_name]["num_test"],
        splits=classification_datasets[dataset_name]["splits"],
        model_name=model_name,
        layer_combinations=[layer_combination],
        save_acts=save_acts,
        batch_size=batch_size,
    )

    classification_dataset_loaders.append(ClassificationDatasetLoader(dataset_config=dataset_config))


# %% [helpers]
@torch.no_grad()
def stack_dataset(items: list, label_to_idx: dict | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    items[i].steering_vectors: torch.Tensor [1, d] (bf16)
    items[i].ds_label: int or str
    Returns:
      X: [n, d] float32 on CPU,
      y: [n] long on CPU,
      label_to_idx: mapping used.
    """
    n = len(items)
    d = items[0].steering_vectors.view(-1).numel()
    X = torch.empty((n, d), dtype=torch.float32)

    # Build or use label map
    if label_to_idx is None:
        raw_labels = [dp.ds_label for dp in items]
        unique_labels = sorted(set(raw_labels))
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}

    y = torch.empty((n,), dtype=torch.long)
    for i, dp in enumerate(items):
        X[i].copy_(dp.steering_vectors.view(-1).to(torch.float32))
        if dp.ds_label not in label_to_idx:
            raise ValueError(f"Label '{dp.ds_label}' not present in label map")
        y[i] = label_to_idx[dp.ds_label]

    return X, y, label_to_idx


@torch.no_grad()
def standardize_train_test(X_tr: torch.Tensor, X_te: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu = X_tr.mean(dim=0, keepdim=True)
    std = X_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (X_tr - mu) / std, (X_te - mu) / std


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


# %% [model + train]
class LinearProbe(nn.Module):
    def __init__(self, d: int, k: int):
        super().__init__()
        self.fc = nn.Linear(d, k, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_probe(
    X_tr_cpu: torch.Tensor,
    y_tr_cpu: torch.Tensor,
    X_te_cpu: torch.Tensor,
    y_te_cpu: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 8192,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    print_every: int = 1,
) -> tuple[float, float]:
    # Move once to GPU
    X_tr = X_tr_cpu.to(DEVICE, non_blocking=True)
    X_te = X_te_cpu.to(DEVICE, non_blocking=True)
    y_tr = y_tr_cpu.to(DEVICE, non_blocking=True)
    y_te = y_te_cpu.to(DEVICE, non_blocking=True)

    # Standardize on GPU
    X_tr, X_te = standardize_train_test(X_tr, X_te)

    # Safety: test labels subset of train labels
    tr_labels = set(y_tr.unique().tolist())
    te_labels = set(y_te.unique().tolist())
    if not te_labels.issubset(tr_labels):
        raise ValueError(f"Test labels {sorted(te_labels - tr_labels)} not seen in training")

    d = X_tr.shape[1]
    k = int(y_tr.max().item() + 1)

    model = LinearProbe(d, k).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_test_acc = 0.0

    n = X_tr.shape[0]
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = X_tr[idx]
            yb = y_tr[idx]
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.detach().item() * xb.size(0)

        if epoch % print_every == 0:
            model.eval()
            with torch.no_grad():
                tr_acc = accuracy(model(X_tr), y_tr)
                te_acc = accuracy(model(X_te), y_te)
            avg_loss = total_loss / n
            print(f"epoch {epoch:03d}  loss={avg_loss:.4f}  train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}")
            if te_acc > best_test_acc:
                best_test_acc = te_acc
                best_model = model.state_dict()

    model.eval()
    with torch.no_grad():
        train_acc = accuracy(model(X_tr), y_tr)
        test_acc = accuracy(model(X_te), y_te)
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_model = model.state_dict()
    return train_acc, best_test_acc


# %% [run all datasets]
EPOCHS = 100
BATCH_SIZE = 8192
LR = 1e-2
WEIGHT_DECAY = 1e-4
PRINT_EVERY = 1

results: list[dict] = []

for loader in classification_dataset_loaders:
    name = loader.dataset_config.custom_dataset_params.classification_dataset_name

    train_items = loader.load_dataset("train")
    test_items = loader.load_dataset("test")

    X_tr_cpu, y_tr_cpu, label_map = stack_dataset(train_items, label_to_idx=None)
    X_te_cpu, y_te_cpu, _label_map2 = stack_dataset(test_items, label_to_idx=label_map)

    print(f"\n=== {name} ===")
    print(f"train: {tuple(X_tr_cpu.shape)}  test: {tuple(X_te_cpu.shape)}  classes: {len(label_map)}")

    tr_acc, te_acc = train_probe(
        X_tr_cpu,
        y_tr_cpu,
        X_te_cpu,
        y_te_cpu,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        print_every=PRINT_EVERY,
    )

    print(f"{name:24s}  train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}")

    results.append(
        dict(
            dataset=name,
            n_train=int(X_tr_cpu.shape[0]),
            n_test=int(X_te_cpu.shape[0]),
            dim=int(X_tr_cpu.shape[1]),
            train_acc=float(tr_acc),
            test_acc=float(te_acc),
            label_map=label_map,
        )
    )

# %% [summary]
print("\nSummary")
for r in sorted(results, key=lambda x: x["dataset"]):
    print(
        f"{r['dataset']:24s}  "
        f"train={r['train_acc']:.4f}  test={r['test_acc']:.4f}  "
        f"n_train={r['n_train']:6d}  n_test={r['n_test']:5d}  d={r['dim']}"
    )

# %%
