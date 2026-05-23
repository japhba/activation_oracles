"""Phase 3 ablations: layer selection + more compute, targeting model-org evals.

Each shares the best Phase-1 recipe (rsLoRA, lr=3e-5, cot-v5 past_lens with on-policy +
50% sys-prompt inject) — only the layer combo or budget changes.

I  — wider 5-layer combo [21,22,23,24,25]: extend the plateau toward deeper layers
J  — interleaved 4-layer [19,21,23,25]: span 53-70% depth in 2-layer steps
K  — single-layer L23 (probe whether L23 alone is competitive without multi)
L  — more training (160k examples) at multi [21,22,23]
M  — larger past_lens budget (80k past_lens) at multi [21,22,23]
N  — large lora_r=128 at multi [21,22,23]
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

OUT_DIR = Path(__file__).resolve().parent / "phase3_ablations"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _set_layers(cfg: dict, layer_indices: list[int], depth_pcts: list[int]) -> None:
    cfg["layer_combinations"] = [depth_pcts]
    cfg["act_layer_combinations"] = [layer_indices]
    for d in cfg["dataset_configs"]:
        d["layer_combinations"] = [depth_pcts]


def _save(tag: str, cfg: dict) -> None:
    cfg["save_dir"] = f"/workspace/checkpoints/ao_q3_8b_phase3_{tag}"
    cfg["wandb_run_name"] = f"ao_q3_8b_phase3_{tag}"
    cfg["wandb_suffix"] = f"_phase3_{tag}"
    cfg["wandb_project"] = "ao-phase3-ablations"
    out = OUT_DIR / f"phase3_{tag}.json"
    out.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {out}  layers={cfg['act_layer_combinations']} max_train={cfg.get('max_train_examples')} lora_r={cfg['lora_r']}")


# Base = recipe of the existing best run (multi L21/22/23 + rsLoRA + past_lens(cot-v5) + lr=3e-5)
def base_cfg(past_lens_n: int = 40000) -> dict:
    cfg = _lst.build_config(22, include_past_lens=True, past_lens_n=past_lens_n)
    cfg["use_rslora"] = True
    cfg["lr"] = 3e-5
    # past_lens corpus = cot-v5 (already set by build_config when include_past_lens=True)
    return cfg


# I — wider 5-layer [21..25] (depths 59,62,64,67,70)
cfg = base_cfg()
_set_layers(cfg, [21, 22, 23, 24, 25], [59, 62, 64, 67, 70])
_save("I_5layer_21_25", cfg)

# J — interleaved 4-layer [19,21,23,25] (depths 53,59,64,70)
cfg = base_cfg()
_set_layers(cfg, [19, 21, 23, 25], [53, 59, 64, 70])
_save("J_4layer_19_21_23_25", cfg)

# K — single-layer L23 (competitor to L22)
cfg = base_cfg()
_set_layers(cfg, [23], [64])
_save("K_single_L23", cfg)

# L — more training (160k examples), same multi [21,22,23]
cfg = base_cfg()
_set_layers(cfg, [21, 22, 23], [59, 62, 64])
cfg["max_train_examples"] = 160_000
_save("L_more_train_160k", cfg)

# M — larger past_lens budget (80k past_lens entries)
cfg = base_cfg(past_lens_n=80_000)
_set_layers(cfg, [21, 22, 23], [59, 62, 64])
_save("M_pastlens_80k", cfg)

# N — larger lora_r=128 (more adapter capacity)
cfg = base_cfg()
_set_layers(cfg, [21, 22, 23], [59, 62, 64])
cfg["lora_r"] = 128
cfg["lora_alpha"] = 256  # keep alpha/r ratio = 2
_save("N_lora_r128", cfg)
