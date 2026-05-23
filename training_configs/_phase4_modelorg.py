"""Phase 4 ablations: target model-organism evals (taboo + personaqa).

These reintroduce LatentQA training data — the exact "extract hidden info from
activations" task that taboo/personaqa evaluate. Phase 1-2 ablations showed
taboo/personaqa scores were stable across data-mix changes (~0.43/0.10),
suggesting nothing in our current training mix is teaching the AO what
taboo/personaqa actually need.

O — best multi recipe + 60k LatentQA examples mixed in
P — best multi recipe + 60k LatentQA + 1.5x training budget (120k examples)
Q — pure LatentQA-heavy: drop classification, keep cot+past_lens+latentqa
"""
import copy
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "_layer_sweep_template",
    Path(__file__).parent / "_layer_sweep_template.py",
)
_lst = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_lst)

OUT_DIR = Path(__file__).resolve().parent / "phase4_modelorg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LATENTQA_DEPTHS = [59, 62, 64]  # multi-layer combo, percent depths


def latentqa_entry(layer_pcts: list[int], num_train: int) -> dict:
    return {
        "custom_dataset_params": {
            "max_window_size": 3,
            "min_window_size": 1,
            "min_end_offset": -1,
            "max_end_offset": -10,
            "position_types": ["all", "window"],
        },
        "num_train": num_train,
        "num_test": 0,
        "splits": ["train"],
        "model_name": "Qwen/Qwen3-8B",
        "layer_combinations": [layer_pcts],
        "save_acts": False,
        "batch_size": 16,
        "dataset_name": "latentqa",
        "dataset_folder": "sft_training_data",
        "seed": 42,
    }


def base_multi_cfg() -> dict:
    cfg = _lst.build_config(22, include_past_lens=True, past_lens_n=40000)
    # Promote to multi-layer
    cfg["layer_combinations"] = [[59, 62, 64]]
    cfg["act_layer_combinations"] = [[21, 22, 23]]
    for d in cfg["dataset_configs"]:
        d["layer_combinations"] = [[59, 62, 64]]
    cfg["use_rslora"] = True
    cfg["lr"] = 3e-5
    return cfg


def add_latentqa(cfg: dict, num_train: int) -> None:
    entry = latentqa_entry(LATENTQA_DEPTHS, num_train=num_train)
    cfg["dataset_configs"].append(entry)
    cfg["dataset_loader_names"].append("latentqa")


def _save(tag: str, cfg: dict) -> None:
    cfg["save_dir"] = f"/workspace/checkpoints/ao_q3_8b_phase4_{tag}"
    cfg["wandb_run_name"] = f"ao_q3_8b_phase4_{tag}"
    cfg["wandb_suffix"] = f"_phase4_{tag}"
    cfg["wandb_project"] = "ao-phase4-modelorg"
    out = OUT_DIR / f"phase4_{tag}.json"
    out.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {out}  loaders={cfg['dataset_loader_names']}  max_train={cfg.get('max_train_examples')}")


# O — multi baseline + LatentQA
cfg = base_multi_cfg()
add_latentqa(cfg, num_train=60_000)
_save("O_with_latentqa_60k", cfg)

# P — multi baseline + LatentQA + larger budget
cfg = base_multi_cfg()
add_latentqa(cfg, num_train=60_000)
cfg["max_train_examples"] = 120_000
_save("P_with_latentqa_120k_budget", cfg)

# Q — drop classification, keep cot + past_lens + latentqa heavy
cfg = base_multi_cfg()
# Remove classification entries
new_dconfigs = [d for d in cfg["dataset_configs"] if not d["dataset_name"].startswith("classification_")]
new_loaders = [n for n in cfg["dataset_loader_names"] if not n.startswith("classification_")]
cfg["dataset_configs"] = new_dconfigs
cfg["dataset_loader_names"] = new_loaders
add_latentqa(cfg, num_train=80_000)
_save("Q_no_classification_latentqa_heavy", cfg)
