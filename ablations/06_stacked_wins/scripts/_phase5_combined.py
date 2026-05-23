"""Phase 5: stack the best Phase-3 wins.

R — 5-layer [21..25] + 160k train budget (combine I + L)
S — 5-layer [21..25] + LatentQA 60k (combine I + O)
T — 5-layer [21..25] + 160k + LatentQA 60k (combine all wins)
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

OUT_DIR = Path(__file__).resolve().parent / "phase5_combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = [21, 22, 23, 24, 25]
DEPTHS = [59, 62, 64, 67, 70]


def base_5layer_cfg(past_lens_n: int = 40000) -> dict:
    cfg = _lst.build_config(22, include_past_lens=True, past_lens_n=past_lens_n)
    cfg["layer_combinations"] = [DEPTHS]
    cfg["act_layer_combinations"] = [LAYERS]
    for d in cfg["dataset_configs"]:
        d["layer_combinations"] = [DEPTHS]
    cfg["use_rslora"] = True
    cfg["lr"] = 3e-5
    cfg["train_batch_size"] = 16  # Phase4 used 16 for OOM safety; consistent
    return cfg


def latentqa_entry(num_train: int) -> dict:
    return {
        "custom_dataset_params": {
            "max_window_size": 3, "min_window_size": 1,
            "min_end_offset": -1, "max_end_offset": -10,
            "position_types": ["all", "window"],
        },
        "num_train": num_train, "num_test": 0, "splits": ["train"],
        "model_name": "Qwen/Qwen3-8B",
        "layer_combinations": [DEPTHS],
        "save_acts": False, "batch_size": 16,
        "dataset_name": "latentqa",
        "dataset_folder": "sft_training_data",
        "seed": 42,
    }


def _save(tag: str, cfg: dict) -> None:
    cfg["save_dir"] = f"/workspace/checkpoints/ao_q3_8b_phase5_{tag}"
    cfg["wandb_run_name"] = f"ao_q3_8b_phase5_{tag}"
    cfg["wandb_suffix"] = f"_phase5_{tag}"
    cfg["wandb_project"] = "ao-phase5-combined"
    out = OUT_DIR / f"phase5_{tag}.json"
    out.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {out}  max_train={cfg['max_train_examples']} loaders={len(cfg['dataset_loader_names'])}")


# R — 5-layer + 2x train
cfg = base_5layer_cfg()
cfg["max_train_examples"] = 160_000
_save("R_5layer_160k", cfg)

# S — 5-layer + LatentQA 60k
cfg = base_5layer_cfg()
cfg["dataset_configs"].append(latentqa_entry(60_000))
cfg["dataset_loader_names"].append("latentqa")
_save("S_5layer_latentqa", cfg)

# T — 5-layer + LatentQA + 2x train (everything)
cfg = base_5layer_cfg()
cfg["dataset_configs"].append(latentqa_entry(60_000))
cfg["dataset_loader_names"].append("latentqa")
cfg["max_train_examples"] = 160_000
_save("T_5layer_latentqa_160k", cfg)
