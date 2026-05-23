import datetime
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, login, whoami

from nl_probes.dataset_classes.act_dataset_manager import ActDatasetLoader, DatasetLoaderConfig
from nl_probes.utils.dataset_utils import SPECIAL_TOKEN
from nl_probes.utils.common import layer_percent_to_layer

TRAINING_CONFIG_FILENAME = "ao_config.json"
TRAINING_CONFIG_SCHEMA_VERSION = 1
DEFAULT_PREFIX_TEMPLATE = "Layer: {layer}\\n{special_token} * {num_positions} \\n"
DEPRECATED_CONFIG_FIELDS = {
    "activation_collection_batch_size",
}


def dataset_loader_name_from_config(dataset_config: DatasetLoaderConfig) -> str:
    dataset_name = dataset_config.dataset_name
    params = dataset_config.custom_dataset_params

    if dataset_name == "prebuilt_pt":
        return params.component_name

    if dataset_name == "past_lens":
        return "past_lens"

    if dataset_name == "latentqa":
        return "latentqa"

    if dataset_name == "cot_oracle_local_eval":
        # Use the JSON file's stem as the readable component name.
        from pathlib import Path as _P
        return _P(params.json_path).stem

    if dataset_name == "cot_oracle_convqa":
        return "cot_oracle_convqa"

    if dataset_name.startswith("classification_"):
        return dataset_name

    # Draft configs may set dataset_name directly without loader normalization.
    if hasattr(params, "classification_dataset_name"):
        return f"classification_{params.classification_dataset_name}"

    return dataset_name


@dataclass
class SelfInterpTrainingConfig:
    layer_combinations: list[list[int]]
    act_layer_combinations: list[list[int]] = field(default_factory=list)
    schema_version: int = TRAINING_CONFIG_SCHEMA_VERSION
    special_token: str = SPECIAL_TOKEN
    prefix_template: str = DEFAULT_PREFIX_TEMPLATE

    # --- Model ---
    model_name: str = "Qwen/Qwen3-8B"
    hook_onto_layer: int = 1

    # --- Data / experiment ---
    dataset_configs: list[dict] = field(default_factory=list)
    dataset_loader_names: list[str] = field(default_factory=list)
    validation_dataset_configs: list[dict] = field(default_factory=list)
    validation_dataset_loader_names: list[str] = field(default_factory=list)
    use_decoder_vectors: bool = True
    generation_kwargs: dict[str, Any] = field(default_factory=lambda: {"do_sample": False, "max_new_tokens": 20})
    steering_coefficient: float = 1.0
    dataset_folder: str = "sft_training_data"
    chat_regularization_path: str | None = None
    chat_regularization_every_n_ao_updates: int | None = None
    chat_regularization_weight: float = 1.0
    chat_regularization_max_train_examples: int | None = None
    monitor_num_eval_examples_per_component: int | None = 256
    monitor_num_eval_examples_classification_total: int | None = None
    monitor_eval_steps: int | None = 5000
    monitor_eval_on_start: bool = False
    validation_steps: int | None = None
    validation_on_start: bool = False

    # --- Batching ---
    train_batch_size: int = 16
    eval_batch_size: int = 128
    train_batches_per_materialization_block: int = 16

    # --- LoRA ---
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"
    use_rslora: bool = False
    # If > 0, stop training when this many target tokens (loss-contributing
    # tokens, ignoring -100 / pad) have been seen. Overrides num_epochs if hit
    # first. Used for strict-token-budget ablations.
    max_target_tokens: int = 0

    # --- Unsloth (per AGENTS.md, default to Unsloth for AO training) ---
    # When True, the launcher must also set AO_USE_UNSLOTH=1 so unsloth is
    # imported before transformers/peft (otherwise its kernel patches do not
    # apply).
    use_unsloth: bool = False
    unsloth_max_seq_length: int = 4096

    # --- Training ---
    num_epochs: int = 1
    lr: float = 1e-5
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    eval_steps: int = 9_999_999  # effectively off by default
    eval_on_start: bool = False
    gradient_checkpointing: bool = False
    window_mult: int = 20
    save_steps: int = 9_999_999  # effectively off by default
    save_dir: str = "checkpoints/default"
    max_train_examples: int | None = None  # if set, trim training data to this many after shuffle
    seed: int = 42
    eval_logs_path: str = "eval_logs.json"
    load_lora_path: str | None = None

    # --- Tracking ---
    created_at_utc: str = ""
    git_commit: str = ""
    wandb_project: str = "misc_experiments"
    wandb_run_name: str = ""  # derived if empty
    wandb_suffix: str = ""
    # If set, log `train/epoch = examples_seen / examples_per_source_epoch`
    # to wandb. Use it when num_train materializes K with-replacement passes
    # over a smaller source split (e.g. num_train=80016 from a 26672-row source
    # corresponds to 3.0 epochs at end of run; set examples_per_source_epoch=26672).
    examples_per_source_epoch: int | None = None

    # --- Hub ---
    hf_push_to_hub: bool = False
    hf_private_repo: bool = False
    hf_repo_name: str = ""  # optional short name, used to compute repo_id
    hf_repo_id: str = ""  # derived if empty and push is on

    # --- Quantization ---
    load_in_8bit: bool = False  # use bitsandbytes 8-bit quantization (for large models)

    # --- Open-ended eval ---
    open_ended_eval_include: list[str] | None = None  # if set, only run these evals (e.g. ["number_prediction"])
    # Cap on dataset entries per eval (None = full dataset). Used to keep
    # in-training eval overhead down. Standalone eval runs typically leave
    # this None.
    open_ended_eval_max_entries: int | None = None

    # --- Misc experiment options ---
    positive_negative_examples: bool = False

    def finalize(self, dataset_loaders: list[ActDatasetLoader]) -> "SelfInterpTrainingConfig":
        if not self.created_at_utc:
            self.created_at_utc = datetime.datetime.now(datetime.UTC).isoformat()
        if not self.git_commit:
            self.git_commit = get_git_commit_hash()

        assert self.train_batches_per_materialization_block > 0, (
            "train_batches_per_materialization_block must be positive"
        )
        assert self.train_batches_per_materialization_block % self.gradient_accumulation_steps == 0, (
            "train_batches_per_materialization_block must be a multiple of gradient_accumulation_steps"
        )
        if self.chat_regularization_path is None:
            assert self.chat_regularization_every_n_ao_updates is None, (
                "chat_regularization_every_n_ao_updates requires chat_regularization_path"
            )
            assert self.chat_regularization_max_train_examples is None, (
                "chat_regularization_max_train_examples requires chat_regularization_path"
            )
        else:
            assert self.chat_regularization_every_n_ao_updates is not None, (
                "chat_regularization_path requires chat_regularization_every_n_ao_updates"
            )
            assert self.chat_regularization_every_n_ao_updates > 0, (
                "chat_regularization_every_n_ao_updates must be positive"
            )
            assert self.chat_regularization_weight > 0.0, "chat_regularization_weight must be positive"
        if self.monitor_num_eval_examples_per_component is None:
            assert self.monitor_eval_steps is None, (
                "monitor_eval_steps requires monitor_num_eval_examples_per_component"
            )
            assert self.monitor_num_eval_examples_classification_total is None, (
                "monitor_num_eval_examples_classification_total requires monitor_num_eval_examples_per_component"
            )
            assert self.monitor_eval_on_start is False, (
                "monitor_eval_on_start requires monitor_num_eval_examples_per_component"
            )
        else:
            assert self.monitor_num_eval_examples_per_component > 0, (
                "monitor_num_eval_examples_per_component must be positive"
            )
            if self.monitor_num_eval_examples_classification_total is not None:
                assert self.monitor_num_eval_examples_classification_total > 0, (
                    "monitor_num_eval_examples_classification_total must be positive"
                )
            assert self.monitor_eval_steps is not None, (
                "monitor_num_eval_examples_per_component requires monitor_eval_steps"
            )
            assert self.monitor_eval_steps > 0, "monitor_eval_steps must be positive"
        if self.validation_dataset_configs:
            assert self.validation_steps is not None, (
                "validation_dataset_configs requires validation_steps"
            )
            assert self.validation_steps > 0, "validation_steps must be positive"
        else:
            assert self.validation_steps is None, (
                "validation_steps requires validation_dataset_configs"
            )
            assert self.validation_on_start is False, (
                "validation_on_start requires validation_dataset_configs"
            )

        self.dataset_configs = [asdict(dataset_loader.dataset_config) for dataset_loader in dataset_loaders]
        self.dataset_loader_names = [
            dataset_loader_name_from_config(dataset_loader.dataset_config) for dataset_loader in dataset_loaders
        ]
        if not self.layer_combinations:
            raise ValueError("layer_combinations must be provided")
        if not self.act_layer_combinations:
            self.act_layer_combinations = [
                [layer_percent_to_layer(self.model_name, p) for p in combo] for combo in self.layer_combinations
            ]
        assert len(self.layer_combinations) == len(self.act_layer_combinations), (
            "layer_combinations and act_layer_combinations must have the same length"
        )
        for lc, ac in zip(self.layer_combinations, self.act_layer_combinations, strict=True):
            assert len(lc) == len(ac), "Each layer combination must match act layer combination length"

        # run name - stable and readable
        primary_act_combo = self.act_layer_combinations[0]
        layers_str = "-".join(map(str, primary_act_combo))
        default_run = f"{self.model_name}-layers_{layers_str}-decoder-{self.use_decoder_vectors}{self.wandb_suffix}"
        if not self.wandb_run_name:
            self.wandb_run_name = default_run

        # save dir namespacing
        if self.wandb_suffix and not self.save_dir.endswith(self.wandb_suffix):
            self.save_dir = f"{self.save_dir}{self.wandb_suffix}"

        # repo id if pushing
        if self.hf_push_to_hub and not self.hf_repo_id:
            self.hf_repo_id = get_hf_repo_id(self.hf_repo_name)
        return self


def write_training_config(save_dir: str | Path, cfg: SelfInterpTrainingConfig) -> None:
    save_path = Path(save_dir) / TRAINING_CONFIG_FILENAME
    payload = asdict(cfg)
    save_path.write_text(json.dumps(payload, indent=2))


def _load_training_config_payload(payload: dict[str, Any]) -> SelfInterpTrainingConfig:
    for deprecated_field in DEPRECATED_CONFIG_FIELDS:
        if deprecated_field in payload:
            del payload[deprecated_field]
    return SelfInterpTrainingConfig(**payload)


def read_training_config(path_or_repo: str) -> SelfInterpTrainingConfig:
    p = Path(path_or_repo)
    if p.exists():
        if p.is_file():
            return _load_training_config_payload(json.loads(p.read_text()))
        cfg_path = p / TRAINING_CONFIG_FILENAME
        return _load_training_config_payload(json.loads(cfg_path.read_text()))

    cfg_path = hf_hub_download(repo_id=path_or_repo, filename=TRAINING_CONFIG_FILENAME)
    return _load_training_config_payload(json.loads(Path(cfg_path).read_text()))


def get_hf_repo_id(hf_repo_name: str) -> str:
    print("Setting up Hugging Face authentication...")
    # check if already logged in
    if whoami() is None:
        print("Not logged in to Hugging Face. Attempting to log in...")
        login()
    else:
        print("Already logged in to Hugging Face.")

    # Determine default HF repo name if not provided
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    if not hf_repo_name:
        hf_repo_name = f"gemma-introspection-{date_str}"

    # Compose full repo_id with current username
    user_info = whoami()
    owner = user_info.get("name") if isinstance(user_info, dict) else None
    hf_repo_id_computed = f"{owner}/{hf_repo_name}" if owner else hf_repo_name

    return hf_repo_id_computed


def get_git_commit_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
        .strip()
    )
