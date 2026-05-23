"""Recipe V2 — single-layer default, strict 10M-target-token budget,
clean rsLoRA (r=128, alpha=16) + clean past/future lens semantics.

Every ablation varies ONE knob from this default. Multi-layer is an ablation,
not the default.
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

OUT_DIR = Path(__file__).resolve().parent / "v2_ablations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TARGET_TOKENS = 10_000_000  # strict budget: 10M loss-contributing tokens
DEFAULT_LAYER = 22  # single layer at 62% depth — chosen as median of the plateau


def v2_recipe(layer_indices=None, layer_pcts=None) -> dict:
    """Build the V2 base recipe — single-layer L22 by default."""
    if layer_indices is None:
        layer_indices = [DEFAULT_LAYER]
        layer_pcts = [62]
    # Start from existing helper with single layer; we'll override the layer fields after.
    cfg = _lst.build_config(DEFAULT_LAYER, include_past_lens=True, past_lens_n=40000)

    # === Recipe V2 knobs ===
    cfg["lora_r"] = 128
    cfg["lora_alpha"] = 16
    cfg["lora_dropout"] = 0.0
    cfg["use_rslora"] = True
    cfg["lr"] = 3e-4
    cfg["train_batch_size"] = 16
    cfg["max_target_tokens"] = MAX_TARGET_TOKENS
    cfg["num_epochs"] = 1
    cfg["max_train_examples"] = 320_000  # safety upper bound; token cap is the real stop
    # ===

    # Layer field overrides (single layer by default; ablations can override)
    cfg["layer_combinations"] = [layer_pcts]
    cfg["act_layer_combinations"] = [layer_indices]
    for d in cfg["dataset_configs"]:
        d["layer_combinations"] = [layer_pcts]
        # CLEAN past_lens semantics: TRUE past+future lens, no vLLM continuations
        if d["dataset_name"] == "past_lens":
            d["custom_dataset_params"]["future_corpus_only"] = True
            d["custom_dataset_params"]["past_use_vllm"] = False
            d["custom_dataset_params"]["future_chat_system_prompt_prob"] = 0.0
            # corpus stays cot-v5 (set by _layer_sweep_template.past_lens_dataset)
    return cfg


def _save(tag: str, cfg: dict, group: str = "v2_ablations") -> None:
    cfg["save_dir"] = f"/workspace/checkpoints/ao_q3_8b_v2_{tag}"
    cfg["wandb_run_name"] = f"ao_q3_8b_v2_{tag}"
    cfg["wandb_suffix"] = f"_v2_{tag}"
    cfg["wandb_project"] = "ao-v2-clean-ablations"
    out = OUT_DIR / f"v2_{tag}.json"
    out.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {out}")


# === V2 DEFAULT (single-layer L22, recipe V2, 10M tokens) ===
_save("default_single_L22", v2_recipe())

# ============================================================
# Ablation 1: layer count (vary only act_layer_combinations)
# ============================================================
_save("layer_count_single_L21", v2_recipe([21], [59]))
_save("layer_count_single_L23", v2_recipe([23], [64]))
_save("layer_count_3layer", v2_recipe([21,22,23], [59,62,64]))
_save("layer_count_5layer", v2_recipe([21,22,23,24,25], [59,62,64,67,70]))
_save("layer_count_7layer", v2_recipe([21,22,23,24,25,26,27], [59,62,64,67,70,73,76]))

# ============================================================
# Ablation 2: direction (vary only directions field of past_lens)
# ============================================================
def with_directions(directions: list[str]) -> dict:
    cfg = v2_recipe()
    for d in cfg["dataset_configs"]:
        if d["dataset_name"] == "past_lens":
            d["custom_dataset_params"]["directions"] = directions
    return cfg

_save("direction_past_only",   with_directions(["past"]))
_save("direction_future_only", with_directions(["future"]))
_save("direction_past_future", with_directions(["past","future"]))  # = default

# ============================================================
# Ablation 3: past_lens corpus (vary only pretrain_dataset/pretrain_key)
# ============================================================
def with_corpus(repo: str, key: str) -> dict:
    cfg = v2_recipe()
    for d in cfg["dataset_configs"]:
        if d["dataset_name"] == "past_lens":
            d["custom_dataset_params"]["pretrain_dataset"] = repo
            d["custom_dataset_params"]["pretrain_key"] = key
    return cfg

_save("corpus_cotv5", with_corpus("ceselder/cot-oracle-corpus-v5", "cot_response"))  # = default
_save("corpus_finefineweb", with_corpus("m-a-p/FineFineWeb", "text"))
_save("corpus_fineweb", with_corpus("HuggingFaceFW/fineweb", "text"))

# ============================================================
# Ablation 4: drop chunked-convqa-haiku (test if convqa drives the wins)
# ============================================================
cfg = v2_recipe()
# remove cot_oracle_convqa entries
new_configs = [d for d in cfg["dataset_configs"] if d["dataset_name"] != "cot_oracle_convqa"]
new_loaders = [n for n in cfg["dataset_loader_names"] if n != "cot_oracle_convqa"]
cfg["dataset_configs"] = new_configs
cfg["dataset_loader_names"] = new_loaders
_save("no_convqa_haiku", cfg)

# ============================================================
# Ablation 5: + LatentQA (test if LatentQA helps model-orgs)
# ============================================================
cfg = v2_recipe()
cfg["dataset_configs"].append({
    "custom_dataset_params": {"max_window_size":3,"min_window_size":1,"min_end_offset":-1,
                              "max_end_offset":-10,"position_types":["all","window"]},
    "num_train":60000,"num_test":0,"splits":["train"],"model_name":"Qwen/Qwen3-8B",
    "layer_combinations":cfg["layer_combinations"],"save_acts":False,"batch_size":16,
    "dataset_name":"latentqa","dataset_folder":"sft_training_data","seed":42,
})
cfg["dataset_loader_names"].append("latentqa")
_save("plus_latentqa", cfg)
